#!/usr/bin/env python3
from __future__ import annotations

import os, json, math, time, threading, logging, asyncio, traceback, random, secrets
from typing import Any, Dict, Optional, Tuple, List, Set, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dataclasses import dataclass

# ---------- Logging ----------
try:
    from live_lib import jlog  # shared JSON logger
except Exception:  # pragma: no cover
    def jlog(level: int, event: str, **fields):
        print(json.dumps({"level": level, "event": event, **fields}))


async def _pause(secs: float, phase: str, slug: str, salt: str) -> None:
    if secs and secs > 0:
        try:
            jlog(logging.INFO, "atomic_wait", phase=phase, secs=float(secs), slug=slug, salt=salt)
        finally:
            await asyncio.sleep(float(secs))


# ---------- Student-t CDF ----------
from utils import student_t_cdf_np  # type: ignore

# ---------- Quote signals are precomputed upstream (quote_seq.py / quote_iv.py) ----------
@dataclass
class _SubmitState:
    task: Optional[asyncio.Task] = None
    version: int = 0          # which seq / quote version this submit belongs to
    stale: bool = False       # did a newer quote arrive while this submit was running?

# ---------- trade_lib (WS + REST) ----------
from trade_lib import (
    subscribe_user_ws,
    register_position_listener,
    register_ws_status_listener,
    submit_orders_batch,
    cancel_orders,
    cancel_all_orders,
    get_ws_positions,
    TokenIndex,
    MMParams,
    DEFAULT_MM_BY_BASE,
    BASE_CFG,
    WINDOW_SEC,
)

# ---------- Constants ----------
DO_NOT_TRADE_HEAD_SEC = 8.0
DO_NOT_TRADE_TAIL_SEC = 15.0

ATOMIC_POST_SUBMIT_WAIT_S = float(os.getenv("MM_POST_SUBMIT_WAIT_S", "0.002"))

# (Optional) hard TTL-per-order safety valve. Default disabled for queue priority.
ORDER_TTL_S = float(os.getenv("MM_ORDER_TTL_S", "0.0"))
TTL_POLL_INTERVAL_S = float(os.getenv("MM_TTL_POLL_INTERVAL_S", "0.25"))

# New meaning: inactivity cancel by slug if no cb_tick_pred (on_tick) arrives in this many seconds.
ORDER_TTS_CANCEL_S = float(os.getenv("MM_ORDER_TTS_CANCEL_S", "0.5"))
INACTIVITY_POLL_S = float(os.getenv("MM_INACTIVITY_POLL_S", "0.10"))

# Max *submits* per second per slug (not cancels)
MAX_SUBMITS_PER_SEC = float(os.getenv("MM_MAX_SUBMITS_PER_SEC", "1.0"))

# Order-count management
MAX_LIVE_PER_SIDE = int(os.getenv("MM_MAX_LIVE_PER_SIDE", "4"))   # per Polymarket side BUY/SELL
MAX_SAME_PRICE = int(os.getenv("MM_MAX_SAME_PRICE", "2"))         # max stacking at same (token,side,price)

# micro-opts
_log = math.log
_sqrt = math.sqrt
_floor = math.floor
_ceil = math.ceil


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _floor_to_tick(p: float, tick: float) -> float:
    if tick <= 0.0:
        return _clip01(p)
    return _clip01(_floor(float(p) / tick) * tick)


def _ceil_to_tick(p: float, tick: float) -> float:
    if tick <= 0.0:
        return _clip01(p)
    return _clip01(_ceil(float(p) / tick) * tick)


def _salt_from(snapshot: Dict[str, Any]) -> str:
    s = snapshot.get("salt")
    if isinstance(s, str) and s:
        return s
    return secrets.token_hex(8)


class TradeEngine:
    """
    Real trading engine with **atomic per-slug updates**.

    Order management (UPDATED):
      - Allows stacking at the same (token_id, side, price) up to MAX_SAME_PRICE (default 2).
      - Enforces max live orders per slug per Polymarket side BUY/SELL: MAX_LIVE_PER_SIDE (default 4).
          *If a new order would exceed the limit, cancel the furthest existing order (same BUY/SELL side),
           then submit the new one.*
      - Inactivity cancel: if no on_tick for a slug arrives for ORDER_TTS_CANCEL_S (default 0.5s),
        cancel ALL live orders for that slug.

    Update handling policy:
      - While a slug's atomic cycle is running, new updates bump desired_seq.
      - AFTER the cycle finishes, if desired_seq advanced, the loop processes the latest snapshot.
    """

    def __init__(
        self,
        temp_dir: str = "temp",
        allowed_bases={"BTC", "ETH", "SOL", "XRP"},
        mm_by_base: Optional[Dict[str, MMParams]] = None,
    ):
        # Token index
        self.tokens = TokenIndex(temp_dir=temp_dir)
        self.mm_by_base = {k.upper(): v for k, v in (mm_by_base or DEFAULT_MM_BY_BASE).items()}

        # State
        self._pending_snap: Dict[str, Dict[str, Any]] = {}      # yes_id -> latest snapshot
        self._desired_seq: Dict[str, int] = {}                  # yes_id -> desired sequence
        self._applied_seq: Dict[str, int] = {}                  # yes_id -> last applied sequence
        self._workers: Dict[str, asyncio.Task] = {}             # yes_id -> worker task
        self._last_snapshot_by_yes: Dict[str, Dict[str, Any]] = {}

        # live orders
        self._live_order_ids_by_slug: Dict[str, List[str]] = {}
        self._live_orders_by_slug: Dict[str, List[Dict[str, Any]]] = {}

        # submit-queue state per slug
        self._submit_state_by_slug: Dict[str, "_SubmitState"] = {}
        self._submit_times_by_slug: Dict[str, List[float]] = {}

        # Latest fair price snapshot per slug for post-submit checks
        self._fair_state_by_slug: Dict[str, Dict[str, Any]] = {}

        # for inactivity cancel
        self._last_tick_ts_by_slug: Dict[str, float] = {}

        # Executor + concurrency guard for sync I/O
        self._io_pool = ThreadPoolExecutor(
            max_workers=int(os.getenv("MM_IO_EXEC_WORKERS", "8"))
        )
        self._io_sem = asyncio.Semaphore(int(os.getenv("MM_MAX_PARALLEL_IO", "8")))

        # Retry/timeout knobs
        self._io_timeout_s: float = float(os.getenv("MM_IO_TIMEOUT_S", "0.8"))
        self._io_tries: int = int(os.getenv("MM_IO_RETRIES", "2"))
        self._io_backoff_s: float = float(os.getenv("MM_IO_BACKOFF_S", "0.05"))

        # --- Dedicated event loop the engine owns ---
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever, daemon=True
            )
            self._loop_thread.start()

        # cancel-all once per session (keyed by start_ts)
        self._session_cancel_fired: Set[int] = set()

        self.allowed_bases = {b.upper() for b in allowed_bases} if allowed_bases else None

        # per-slug volume tracking (from WS)
        self._vol_used_by_slug: Dict[str, float] = {}
        self._vol_lock = threading.Lock()

        # Start WS and inventory listener — soft start (no hard exit)
        self._enabled = True
        self._disabled_reason: Optional[str] = None

        try:
            subscribe_user_ws(fatal_on_error=False, fatal_on_close=False)
            jlog(logging.INFO, "ws_user_started", salt="boot")
        except Exception as e:
            jlog(logging.ERROR, "ws_user_start_error", error=repr(e), salt="boot")

        register_position_listener(self._on_inventory_delta)

        # Soft-shutdown hook + auto-resume for user WS
        def _ws_status_cb(status: str, info: Dict[str, Any]):
            if status == "dead":
                loop = getattr(self, "_loop", None)
                if loop and loop.is_running():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.shutdown(reason="user_ws_dead"), loop
                        )
                    except Exception as e:
                        jlog(logging.ERROR, "engine_shutdown_schedule_error", error=repr(e), info=info)
                else:
                    self._disabled_reason = "user_ws_dead"
                    self._enabled = False
                    try:
                        cancel_all_orders()
                    except Exception as e:
                        jlog(logging.ERROR, "engine_shutdown_fallback_cancel_error", error=repr(e), info=info)

            elif status == "open":
                try:
                    if self._loop:
                        self._loop.call_soon_threadsafe(self._resume_after_ws_open, info)
                except Exception as e:
                    jlog(logging.ERROR, "engine_resume_after_ws_open_error", error=repr(e), info=info)

        register_ws_status_listener(_ws_status_cb)

        # Optional per-order TTL canceller (disabled by default)
        if ORDER_TTL_S and ORDER_TTL_S > 0:
            self._schedule(self._ttl_canceller_loop())

        # Inactivity canceller (required)
        self._schedule(self._inactivity_canceller_loop())

    def _resume_after_ws_open(self, info: Dict[str, Any]) -> None:
        if getattr(self, "_enabled", True):
            return
        if self._disabled_reason != "user_ws_dead":
            return

        self._enabled = True
        self._disabled_reason = None

        try:
            self._session_cancel_fired.clear()
            self._live_order_ids_by_slug.clear()
            self._live_orders_by_slug.clear()
        except Exception:
            pass

        try:
            if ORDER_TTL_S and ORDER_TTL_S > 0:
                self._schedule(self._ttl_canceller_loop())
        except Exception:
            pass

        try:
            self._schedule(self._inactivity_canceller_loop())
        except Exception:
            pass

        try:
            self._kick_current_slugs()
        except Exception:
            pass

        try:
            for yes_id, snap in list(self._last_snapshot_by_yes.items()):
                if snap:
                    self._enqueue_update(yes_id, dict(snap))
        except Exception:
            pass

        jlog(logging.INFO, "engine_resume_after_ws_open", info=info, note="engine re-enabled and kicked")

    # ---------------- I/O helpers: executor + retry + timeout + timing ----------------
    async def _io_call(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._io_pool, partial(func, *args, **kwargs))

    async def _io_once_guarded(self, func, *args, **kwargs):
        async with self._io_sem:
            return await self._io_call(func, *args, **kwargs)

    async def _io_retry(
        self, func, *args, tries: Optional[int] = None, base_sleep: Optional[float] = None, **kwargs
    ):
        tries = self._io_tries if tries is None else max(1, int(tries))
        base_sleep = self._io_backoff_s if base_sleep is None else float(base_sleep)
        for i in range(tries):
            try:
                return await self._io_once_guarded(func, *args, **kwargs)
            except Exception:
                if i + 1 >= tries:
                    raise
                await asyncio.sleep(base_sleep * (1.0 + 0.5 * random.random()))

    async def _timed(self, label: str, coro: asyncio.Future, slug: str, salt: str):
        t0 = time.perf_counter()
        try:
            return await asyncio.wait_for(coro, timeout=self._io_timeout_s)
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            task = asyncio.current_task()
            jlog(
                logging.INFO,
                "timing",
                slug=slug,
                op=label,
                ms=round(dt_ms, 2),
                task=(task.get_name() if task else None),
                salt=salt,
            )

    def _remove_live_orders_by_ids(self, slug: str, cancel_ids: Sequence[str]) -> None:
        if not cancel_ids:
            return
        cancel_set = {str(x) for x in cancel_ids if x}
        meta = self._live_orders_by_slug.get(slug, []) or []
        keep = [m for m in meta if str(m.get("order_id") or "") not in cancel_set]
        self._live_orders_by_slug[slug] = keep
        self._live_order_ids_by_slug[slug] = [str(m.get("order_id")) for m in keep if m.get("order_id")]

    async def _cancel_live_orders(self, slug: str, salt: str) -> None:
        last_ids = self._live_order_ids_by_slug.get(slug, [])
        if not last_ids:
            self._live_orders_by_slug[slug] = []
            return
        jlog(logging.INFO, "order_cancel_start", slug=slug, n=len(last_ids), salt=salt)
        try:
            await self._timed("cancel_orders", self._io_retry(cancel_orders, last_ids), slug, salt)
        except Exception as e:
            jlog(logging.ERROR, "order_cancel_error", slug=slug, error=repr(e), salt=salt)
        else:
            jlog(logging.INFO, "order_cancel_done", slug=slug, n=len(last_ids), salt=salt)
            self._live_order_ids_by_slug[slug] = []
            self._live_orders_by_slug[slug] = []

    def _try_acquire_submit_slot(self, slug: str, salt: str) -> bool:
        max_rate = float(MAX_SUBMITS_PER_SEC)
        if max_rate <= 0.0:
            return True

        window = 1.0
        now = time.time()
        times = self._submit_times_by_slug.get(slug, [])

        cutoff = now - window
        times = [t for t in times if t > cutoff]

        if len(times) >= int(max_rate):
            jlog(
                logging.INFO,
                "submit_rate_limit_skip",
                slug=slug,
                max_submits_per_sec=max_rate,
                n_recent=len(times),
                salt=salt,
            )
            self._submit_times_by_slug[slug] = times
            return False

        times.append(now)
        self._submit_times_by_slug[slug] = times
        return True

    # ---------------- Optional per-order TTL cancellation (disabled by default) ----------------
    async def _cancel_ttl_orders(self, slug: str, now_s: float, salt: str) -> None:
        live_meta = self._live_orders_by_slug.get(slug, []) or []
        if not live_meta:
            return
        ttl = float(ORDER_TTL_S)
        if ttl <= 0:
            return

        cancel_ids: List[str] = []
        for m in live_meta:
            oid = str(m.get("order_id", "") or "")
            if not oid:
                continue
            try:
                submit_ts = float(m.get("submit_ts", now_s))
            except Exception:
                submit_ts = now_s
            age = max(0.0, now_s - submit_ts)
            if age >= ttl:
                cancel_ids.append(oid)

        if not cancel_ids:
            return

        jlog(logging.INFO, "order_cancel_ttl", slug=slug, n=len(cancel_ids), ttl=ttl, salt=salt)
        try:
            await self._timed("cancel_orders_ttl", self._io_retry(cancel_orders, cancel_ids), slug, salt)
        except Exception as e:
            jlog(logging.ERROR, "order_cancel_ttl_error", slug=slug, error=repr(e), salt=salt)
            return

        self._remove_live_orders_by_ids(slug, cancel_ids)

    async def _ttl_canceller_loop(self) -> None:
        try:
            while getattr(self, "_enabled", True):
                if not (ORDER_TTL_S and ORDER_TTL_S > 0):
                    await asyncio.sleep(0.25)
                    continue
                now_s = time.time()
                for slug, meta_list in list(self._live_orders_by_slug.items()):
                    if not meta_list:
                        continue
                    salt = f"ttl-{slug}"
                    try:
                        tok = self.tokens.by_slug.get(slug)
                        if tok:
                            yes_id = tok[0]
                            snap = self._last_snapshot_by_yes.get(yes_id) or {}
                            salt = str(snap.get("salt") or salt)
                    except Exception:
                        pass

                    await self._cancel_ttl_orders(slug=slug, now_s=now_s, salt=salt)

                await asyncio.sleep(max(0.05, min(TTL_POLL_INTERVAL_S, max(0.25, ORDER_TTL_S / 4.0))))
        except asyncio.CancelledError:
            return
        except Exception as e:
            jlog(logging.ERROR, "ttl_loop_error", error=repr(e))

    # ---------------- Inactivity canceller (ORDER_TTS_CANCEL_S) ----------------
    async def _inactivity_canceller_loop(self) -> None:
        """
        If a slug has live orders, but no cb_tick_pred (on_tick) arrives for ORDER_TTS_CANCEL_S,
        cancel ALL orders for that slug.
        """
        try:
            while True:
                if not getattr(self, "_enabled", True):
                    await asyncio.sleep(0.2)
                    continue

                now_s = time.time()
                for slug, ids in list(self._live_order_ids_by_slug.items()):
                    if not ids:
                        continue
                    last = self._last_tick_ts_by_slug.get(slug)
                    if last is None:
                        # no tick seen for slug; treat as stale immediately (but don't spam cancels)
                        last = 0.0
                    dt = now_s - float(last)
                    if dt <= float(ORDER_TTS_CANCEL_S):
                        continue

                    salt = f"inactivity-{slug}"
                    jlog(
                        logging.ERROR,
                        "inactivity_timeout_cancel",
                        slug=slug,
                        dt=dt,
                        timeout_s=float(ORDER_TTS_CANCEL_S),
                        n_live=len(ids),
                        salt=salt,
                    )
                    try:
                        await self._cancel_live_orders(slug, salt)
                    except Exception as e:
                        jlog(logging.ERROR, "inactivity_cancel_error", slug=slug, error=repr(e), salt=salt)

                await asyncio.sleep(max(0.05, float(INACTIVITY_POLL_S)))
        except asyncio.CancelledError:
            return
        except Exception as e:
            jlog(logging.ERROR, "inactivity_loop_error", error=repr(e))

    # ---------------- Session quiet window ----------------
    def _quiet_window_action(self, start_ts: int, now_s: float) -> Optional[str]:
        try:
            t0 = float(start_ts)
            t = float(now_s)
        except Exception:
            return None
        if t < t0 + DO_NOT_TRADE_HEAD_SEC:
            return "head_skip"
        if t >= (t0 + WINDOW_SEC - DO_NOT_TRADE_TAIL_SEC):
            return "tail_cancel"
        return None

    async def shutdown(self, reason: str = "shutdown") -> None:
        if not getattr(self, "_enabled", True) and self._disabled_reason == reason:
            return

        self._disabled_reason = reason
        self._enabled = False

        jlog(logging.INFO, "engine_shutdown_begin", reason=reason)

        slug_keys = list(self._live_order_ids_by_slug.keys())
        per_slug_tasks = []
        for slug in slug_keys:
            per_slug_tasks.append(self._cancel_live_orders(slug, str(reason)))

        if per_slug_tasks:
            try:
                await asyncio.gather(*per_slug_tasks, return_exceptions=True)
            except Exception as e:
                jlog(logging.ERROR, "engine_shutdown_per_slug_error", error=repr(e), reason=reason)

        try:
            await self._timed(
                "cancel_all_orders",
                self._io_retry(cancel_all_orders),
                slug="*",
                salt=str(reason),
            )
        except Exception as e:
            jlog(logging.ERROR, "engine_shutdown_cancel_all_error", error=repr(e), reason=reason)

        self._live_order_ids_by_slug.clear()
        self._live_orders_by_slug.clear()
        jlog(logging.INFO, "engine_shutdown_done", reason=reason)

        try:
            if (
                reason not in ("user_ws_dead",)
                and self._loop_thread
                and self._loop
                and self._loop.is_running()
            ):
                self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass

    def _kick_current_slugs(self) -> None:
        try:
            self.tokens.refresh()
        except Exception:
            pass

        bases = self.allowed_bases or {"BTC", "ETH", "SOL", "XRP"}
        bases = {str(b).upper() for b in bases}

        last_by_base: Dict[str, Dict[str, Any]] = {}
        for snap in self._last_snapshot_by_yes.values():
            b = str(snap.get("base", "")).upper()
            if b and b in bases:
                last_by_base[b] = snap

        for base in bases:
            slug = self.tokens.active_slug_for_base(base)
            if not slug:
                continue
            tok = self.tokens.by_slug.get(slug) or (self.tokens.refresh() or self.tokens.by_slug.get(slug))
            if not tok:
                continue
            yes_id = str(tok[0])

            snap_src = last_by_base.get(base)
            if not snap_src:
                continue

            snap = dict(snap_src)
            snap["slug"] = slug
            snap["salt"] = _salt_from(snap)
            self._enqueue_update(yes_id, snap)

    def _get_submit_state(self, slug: str) -> _SubmitState:
        st = self._submit_state_by_slug.get(slug)
        if st is None:
            st = _SubmitState()
            self._submit_state_by_slug[slug] = st
        return st

    def _update_live_orders_after_submit(
        self,
        slug: str,
        order_list: Sequence[Dict[str, Any]],
        order_ids: Sequence[str],
    ) -> None:
        if not order_ids:
            return

        now_submit = time.time()
        valid_ids: List[str] = []
        new_meta: List[Dict[str, Any]] = []

        # order_ids may be shorter; map by index safely
        n = min(len(order_list), len(order_ids))
        for i in range(n):
            order = order_list[i]
            oid = order_ids[i]
            if not isinstance(oid, str) or not oid:
                continue
            try:
                price = float(order["price"])
                size = float(order["size"])
            except Exception:
                continue
            valid_ids.append(oid)
            new_meta.append(
                {
                    "order_id": oid,
                    "token_id": str(order["token_id"]),
                    "side": str(order["side"]).upper(),
                    "price": price,
                    "size": size,
                    "submit_ts": now_submit,
                }
            )

        if not valid_ids:
            return

        prev_ids = self._live_order_ids_by_slug.get(slug, [])
        prev_meta = self._live_orders_by_slug.get(slug, [])
        self._live_order_ids_by_slug[slug] = prev_ids + valid_ids
        self._live_orders_by_slug[slug] = prev_meta + new_meta

    async def _schedule_submit(
        self,
        slug: str,
        new_orders: Sequence[Dict[str, Any]],
        version: int,
        salt: str,
    ) -> None:
        if not new_orders:
            return
        st = self._get_submit_state(slug)

        if st.task is None or st.task.done():
            st.stale = False
            st.version = version

            async def _runner() -> None:
                try:
                    await self._submit_runner(slug, new_orders, version, salt)
                finally:
                    cur = self._get_submit_state(slug)
                    cur.task = None

            st.task = asyncio.create_task(_runner())
            return

        if version > st.version:
            st.stale = True
            st.version = version
            jlog(logging.INFO, "submit_mark_stale", slug=slug, latest_version=version)

    # ---------- push-based inventory callback ----------
    def _on_inventory_delta(self, snap_positions: Dict[str, float], delta_event: Dict[str, Any]) -> None:
        asset_id = str(delta_event.get("asset_id", ""))
        if not asset_id:
            return
        slug_out = self.tokens.by_token.get(asset_id)
        if not slug_out:
            self.tokens.refresh()
            slug_out = self.tokens.by_token.get(asset_id)
            if not slug_out:
                return
        slug = slug_out[0]
        tok = self.tokens.by_slug.get(slug) or (self.tokens.refresh() or self.tokens.by_slug.get(slug))
        if not tok:
            return

        try:
            vol_slug = delta_event.get("volume_slug", None)
            if isinstance(vol_slug, (int, float)):
                with self._vol_lock:
                    cur = float(self._vol_used_by_slug.get(slug, 0.0))
                    self._vol_used_by_slug[slug] = max(cur, float(vol_slug))
        except Exception as e:
            jlog(logging.WARNING, "ws_vol_update_error", slug=slug, error=repr(e))

        yes_id = tok[0]
        snap = self._last_snapshot_by_yes.get(yes_id)
        if snap:
            self._enqueue_update(yes_id, dict(snap))

    async def _submit_runner(self, slug: str, new_orders: Sequence[Dict[str, Any]], version: int, salt: str) -> None:
        async def _do_submit() -> List[str]:
            jlog(logging.INFO, "order_submit_start", slug=slug, n=len(new_orders), seq_version=version, salt=salt)
            order_ids = await self._timed(
                "submit_batch",
                self._io_retry(submit_orders_batch, new_orders),
                slug,
                salt,
            )
            self._update_live_orders_after_submit(slug, new_orders, order_ids or [])
            jlog(
                logging.INFO,
                "order_submit_done",
                slug=slug,
                n=len(order_ids or []),
                seq_version=version,
                order_ids=list(order_ids or []),
                salt=salt,
            )
            return list(order_ids or [])

        try:
            if not self._try_acquire_submit_slot(slug, salt):
                jlog(logging.INFO, "submit_runner_skipped_by_rate_limit", slug=slug, seq_version=version, n_orders=len(new_orders), salt=salt)
                return
            order_ids = await _do_submit()
        except Exception as e:
            jlog(logging.ERROR, "submit_runner_error", slug=slug, seq_version=version, error=repr(e), salt=salt)
            return

        st_after = self._get_submit_state(slug)
        need_post_price_check = st_after.stale and st_after.version > version

        if need_post_price_check and order_ids:
            fair_state = self._fair_state_by_slug.get(slug)
            if fair_state is not None:
                try:
                    fair_yes_lo = float(fair_state.get("fair_yes_lo", fair_state.get("fair_yes_hi", 0.5)))
                    fair_yes_hi = float(fair_state.get("fair_yes_hi", fair_yes_lo))
                except Exception:
                    fair_yes_lo = fair_yes_hi = 0.5

                yes_id_f = str(fair_state.get("yes_id", ""))
                no_id_f = str(fair_state.get("no_id", ""))

                if yes_id_f and no_id_f:
                    cancel_ids = self._select_stale_and_crossed_orders(
                        slug=slug,
                        yes_id=yes_id_f,
                        no_id=no_id_f,
                        fair_yes_lo=fair_yes_lo,
                        fair_yes_hi=fair_yes_hi,
                    )
                    if cancel_ids:
                        jlog(
                            logging.INFO,
                            "cancel_post_submit_unfavorable",
                            slug=slug,
                            seq_version=version,
                            latest_version=st_after.version,
                            n_orders=len(cancel_ids),
                            fair_yes_lo=fair_yes_lo,
                            fair_yes_hi=fair_yes_hi,
                            salt=salt,
                        )

                        async def _cancel_post_submit() -> None:
                            try:
                                await self._timed(
                                    "cancel_post_submit_unfavorable",
                                    self._io_retry(cancel_orders, cancel_ids),
                                    slug,
                                    salt,
                                )
                                self._remove_live_orders_by_ids(slug, cancel_ids)
                            except Exception as e:
                                jlog(logging.ERROR, "cancel_post_submit_error", slug=slug, error=repr(e), salt=salt)

                        asyncio.create_task(_cancel_post_submit())
            else:
                jlog(logging.INFO, "post_submit_no_fair_state", slug=slug, seq_version=version, salt=salt)

        st_after.stale = False

    # ---------- enqueue update (coalesced + version bump) ----------
    def _enqueue_update(self, yes_id: str, snapshot: Dict[str, Any]) -> None:
        self._pending_snap[yes_id] = snapshot
        self._desired_seq[yes_id] = self._desired_seq.get(yes_id, 0) + 1
        self._schedule(self._ensure_worker(yes_id))

    # ---- internal scheduler ----
    def _schedule(self, coro) -> None:
        if not self._loop or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._loop_thread.start()

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is self._loop:
            self._loop.create_task(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ---------- external tick entry ----------
    def on_tick(self, snapshot: Dict[str, Any]) -> None:
        """
        snapshot keys:
          base, slug(optional), prev_close, tau, model_iv, model_df,
          coinbase (S_now), bid_px, ask_px,
          precomputed quote signals:
            iv_mid_pred, iv_hs_pred, iv_delta_mid,
            seq_quote_score, seq_quote_skew, seq_quote_side, seq_quote_one_sided
          realized-vol inputs:
            cb_sigma_ewma, cb_rv_3s
        """
        baseU = str(snapshot.get("base", "")).upper()
        if self.allowed_bases is not None and baseU not in self.allowed_bases:
            return

        slug = snapshot.get("slug") or self.tokens.active_slug_for_base(baseU)
        if not slug:
            return
        tok = self.tokens.by_slug.get(slug) or (self.tokens.refresh() or self.tokens.by_slug.get(slug))
        if not tok:
            return

        yes_id = tok[0]
        snap = dict(snapshot)
        snap["slug"] = slug
        if "salt" not in snap or not isinstance(snap["salt"], str) or not snap["salt"]:
            snap["salt"] = _salt_from(snap)

        # record last tick pred time for inactivity cancel
        self._last_tick_ts_by_slug[slug] = time.time()

        self._last_snapshot_by_yes[yes_id] = snap

        if not getattr(self, "_enabled", True):
            return

        self._enqueue_update(yes_id, snap)

    async def _ensure_worker(self, yes_id: str) -> None:
        if yes_id in self._workers and not self._workers[yes_id].done():
            return
        self._workers[yes_id] = asyncio.create_task(self._worker_loop(yes_id))

    # ---------- worker loop ----------
    async def _worker_loop(self, yes_id: str) -> None:
        try:
            while True:
                if not getattr(self, "_enabled", True):
                    break
                desired = self._desired_seq.get(yes_id, 0)
                applied = self._applied_seq.get(yes_id, 0)
                if desired <= applied:
                    break

                snap = self._pending_snap.get(yes_id)
                if not snap:
                    self._applied_seq[yes_id] = desired
                    break

                seq_started = desired
                slug = str(snap.get("slug") or "")

                try:
                    await self._quote_and_trade_atomic(snap, seq_started)
                except Exception as e:
                    salt = _salt_from(snap)
                    jlog(logging.ERROR, "mm_worker_error", error=repr(e), tb=traceback.format_exc(), slug=slug, salt=salt)
                    try:
                        if slug:
                            await self._cancel_live_orders(slug, salt)
                    except Exception:
                        pass

                self._applied_seq[yes_id] = seq_started
        finally:
            self._workers.pop(yes_id, None)

    # ---------- math helpers ----------
    @staticmethod
    def _sigma_from_iv_tau_model(iv_model_15m: float, tau: float) -> float:
        return float(iv_model_15m) * _sqrt(max(float(tau), 1.0) / 900.0)

    @staticmethod
    def _robust_S_now(snapshot: Dict[str, Any], prev_close: float) -> Optional[float]:
        S = snapshot.get("coinbase")
        if isinstance(S, (int, float)):
            return float(S)
        for k in ("proxy", "spot", "chainlink", "S_now"):
            v = snapshot.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        x = snapshot.get("model_x_input")
        if isinstance(x, (int, float)):
            try:
                return float(prev_close / math.exp(float(x)))
            except Exception:
                return None
        return None

    # ---------- cancellation helpers ----------
    def _select_stale_and_crossed_orders(
        self,
        slug: str,
        yes_id: str,
        no_id: str,
        fair_yes_lo: float,
        fair_yes_hi: float,
    ) -> list[str]:
        live_meta = self._live_orders_by_slug.get(slug, []) or []
        if not live_meta:
            return []

        try:
            lo = float(fair_yes_lo)
            hi = float(fair_yes_hi)
        except Exception:
            return []

        lo = _clip01(lo)
        hi = _clip01(hi)
        if hi < lo:
            lo, hi = hi, lo

        fair_yes_lo = lo
        fair_yes_hi = hi

        fair_no_lo = 1.0 - fair_yes_hi
        fair_no_hi = 1.0 - fair_yes_lo

        cancel_ids: list[str] = []
        keep_meta: list[dict] = []

        for m in live_meta:
            oid = str(m.get("order_id") or "")
            if not oid:
                continue
            side = str(m.get("side") or "").upper()
            token_id = str(m.get("token_id") or "")
            price = float(m.get("price", math.nan))

            if not math.isfinite(price):
                cancel_ids.append(oid)
                continue

            if token_id == str(yes_id):
                fair_lo = fair_yes_lo
                fair_hi = fair_yes_hi
            elif token_id == str(no_id):
                fair_lo = fair_no_lo
                fair_hi = fair_no_hi
            else:
                cancel_ids.append(oid)
                continue

            cancel = False
            if side == "BUY":
                # if BUY is inside/above "too aggressive" bound, cancel
                if price >= fair_lo:
                    cancel = True
            elif side == "SELL":
                # if SELL is inside/below "too aggressive" bound, cancel
                if price <= fair_hi:
                    cancel = True

            if cancel:
                cancel_ids.append(oid)
            else:
                keep_meta.append(m)

        self._live_orders_by_slug[slug] = keep_meta
        self._live_order_ids_by_slug[slug] = [m["order_id"] for m in keep_meta if m.get("order_id")]

        return cancel_ids

    # ---------- order limit helpers (NEW) ----------
    @staticmethod
    def _implied_yes_prob(token_id: str, yes_id: str, no_id: str, price: float) -> float:
        p = _clip01(float(price))
        if str(token_id) == str(yes_id):
            return p
        if str(token_id) == str(no_id):
            return _clip01(1.0 - p)
        return p

    def _count_same_price(
        self,
        slug: str,
        token_id: str,
        side: str,
        price: float,
    ) -> int:
        sideU = str(side).upper()
        tok = str(token_id)
        px = float(price)
        n = 0
        for m in (self._live_orders_by_slug.get(slug, []) or []):
            if str(m.get("token_id") or "") != tok:
                continue
            if str(m.get("side") or "").upper() != sideU:
                continue
            try:
                if float(m.get("price")) == px:
                    n += 1
            except Exception:
                continue
        return n

    def _select_furthest_to_cancel(
        self,
        slug: str,
        *,
        fair_center: float,
        yes_id: str,
        no_id: str,
        side_filter: str,
        n_cancel: int,
    ) -> List[str]:
        if n_cancel <= 0:
            return []
        sideU = str(side_filter).upper()
        fc = _clip01(float(fair_center))

        candidates: List[Tuple[float, float, str]] = []  # (dist, submit_ts, oid)
        for m in (self._live_orders_by_slug.get(slug, []) or []):
            oid = str(m.get("order_id") or "")
            if not oid:
                continue
            if str(m.get("side") or "").upper() != sideU:
                continue
            token_id = str(m.get("token_id") or "")
            try:
                px = float(m.get("price"))
            except Exception:
                continue
            p_yes = self._implied_yes_prob(token_id, yes_id, no_id, px)
            dist = abs(float(p_yes) - fc)
            try:
                submit_ts = float(m.get("submit_ts", 0.0))
            except Exception:
                submit_ts = 0.0
            # cancel furthest first; if tie, cancel newest first (keep older queue priority)
            candidates.append((dist, submit_ts, oid))

        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [oid for (_, __, oid) in candidates[: int(n_cancel)]]

    async def _ensure_capacity_then_submit(
        self,
        slug: str,
        *,
        proposed_orders: Sequence[Dict[str, Any]],
        fair_center: float,
        yes_id: str,
        no_id: str,
        seq_version: int,
        salt: str,
    ) -> None:
        """
        Apply stacking + max-live rules:
          - allow stacking up to MAX_SAME_PRICE at same (token, side, price)
          - enforce max live per BUY/SELL; if exceeded, cancel furthest (same side) then submit
        """
        if not proposed_orders:
            return

        # 1) Filter by max-same-price (stacking allowed up to MAX_SAME_PRICE)
        submit_list: List[Dict[str, Any]] = []
        for o in proposed_orders:
            try:
                tok = str(o["token_id"])
                side = str(o["side"]).upper()
                px = float(o["price"])
            except Exception:
                continue

            cur_n = self._count_same_price(slug, tok, side, px)
            if cur_n >= int(MAX_SAME_PRICE):
                continue
            submit_list.append(o)

        if not submit_list:
            jlog(logging.INFO, "order_submit_skip_all_due_to_max_same_price", slug=slug, salt=salt, max_same=int(MAX_SAME_PRICE))
            return

        # 2) Enforce max-live-per-side (BUY/SELL) AFTER these submissions
        live_meta = self._live_orders_by_slug.get(slug, []) or []
        live_buy = sum(1 for m in live_meta if str(m.get("side") or "").upper() == "BUY" and m.get("order_id"))
        live_sell = sum(1 for m in live_meta if str(m.get("side") or "").upper() == "SELL" and m.get("order_id"))

        new_buy = sum(1 for o in submit_list if str(o.get("side") or "").upper() == "BUY")
        new_sell = sum(1 for o in submit_list if str(o.get("side") or "").upper() == "SELL")

        need_cancel_buy = max(0, (live_buy + new_buy) - int(MAX_LIVE_PER_SIDE))
        need_cancel_sell = max(0, (live_sell + new_sell) - int(MAX_LIVE_PER_SIDE))

        cancel_ids: List[str] = []
        if need_cancel_buy > 0:
            cancel_ids.extend(
                self._select_furthest_to_cancel(
                    slug,
                    fair_center=fair_center,
                    yes_id=yes_id,
                    no_id=no_id,
                    side_filter="BUY",
                    n_cancel=need_cancel_buy,
                )
            )
        if need_cancel_sell > 0:
            cancel_ids.extend(
                self._select_furthest_to_cancel(
                    slug,
                    fair_center=fair_center,
                    yes_id=yes_id,
                    no_id=no_id,
                    side_filter="SELL",
                    n_cancel=need_cancel_sell,
                )
            )

        if cancel_ids:
            cancel_ids = list(dict.fromkeys(cancel_ids))  # stable unique
            jlog(
                logging.INFO,
                "order_cancel_make_room",
                slug=slug,
                n=len(cancel_ids),
                need_cancel_buy=int(need_cancel_buy),
                need_cancel_sell=int(need_cancel_sell),
                max_live_per_side=int(MAX_LIVE_PER_SIDE),
                salt=salt,
            )
            try:
                await self._timed("cancel_make_room", self._io_retry(cancel_orders, cancel_ids), slug, salt)
                self._remove_live_orders_by_ids(slug, cancel_ids)
            except Exception as e:
                # If we can't cancel to make room, don't submit (risk control).
                jlog(logging.ERROR, "order_cancel_make_room_error", slug=slug, error=repr(e), salt=salt)
                return

        # 3) Submit
        await self._schedule_submit(slug=slug, new_orders=submit_list, version=seq_version, salt=salt)

    # ---------- atomic quote & trade for a single snapshot ----------
    async def _quote_and_trade_atomic(self, snapshot: Dict[str, Any], seq_version: int) -> None:
        """
        Atomic quote + trade for a single snapshot.

        Spread logic (your corrected version):
          - fair_center computed from pred_mid + seq_skew + iv_skew (t_fair_mid vs center0).
          - invert fair_center -> spot S_center using Student-t model with (df, model_iv, tau, prev_close).
          - realized 1s sigma in log-space:
                sigma_1s = max(cb_rv_3s/sqrt(3), cb_sigma_ewma)
            => S_dn = S_center*exp(-sigma_1s), S_up = S_center*exp(+sigma_1s)
            => p_dn/p_up from Student-t with prev_close threshold
            => hs_realized_prob = 0.5*(p_up - p_dn)
          - quote_half = hs_pred + hs_realized_prob
        """
        baseU = str(snapshot.get("base", "")).upper()
        slug = snapshot.get("slug") or self.tokens.active_slug_for_base(baseU)
        if not slug:
            return
        salt = _salt_from(snapshot)

        tok = self.tokens.by_slug.get(slug) or (self.tokens.refresh() or self.tokens.by_slug.get(slug))
        if not tok:
            return

        # --- local helpers ---
        def _read_num(
            key: str,
            *,
            required: bool = False,
            positive: bool = False,
            default: Any = None,
        ) -> Any:
            raw = snapshot.get(key, default)
            if isinstance(raw, (int, float)):
                v = float(raw)
                if not math.isfinite(v):
                    return default
                if positive and v <= 0.0:
                    return default
                return v
            return default

        def _read_str(key: str, default: str = "") -> str:
            v = snapshot.get(key, default)
            return v if isinstance(v, str) else default

        def _read_bool(key: str, default: bool = False) -> bool:
            v = snapshot.get(key, default)
            return bool(v) if isinstance(v, bool) else default

        def _net_yes_effect(side: str, which: str) -> int:
            # +1 increases net YES exposure, -1 decreases net YES exposure
            side = side.upper()
            which = which.upper()
            if which == "YES":
                return +1 if side == "BUY" else -1
            # NO: buying NO decreases net YES, selling NO increases net YES
            return -1 if side == "BUY" else +1

        def _desired_sign_from_side(side: str, score: float) -> int:
            s = (side or "flat").lower()
            if s not in ("bid", "ask", "flat"):
                if score > 0.0:
                    s = "bid"
                elif score < 0.0:
                    s = "ask"
                else:
                    s = "flat"
            if s == "bid":
                return +1
            if s == "ask":
                return -1
            return 0

        # ----- token fields -----
        yes_id, no_id, tick_sz, min_size, start_ts, _ = tok
        tick_sz = float(tick_sz)
        min_size = float(min_size)
        start_ts = int(start_ts)

        # ---------------- Inputs / guards ----------------
        prev_close = _read_num("prev_close", required=True, positive=True, default=None)
        if prev_close is None:
            jlog(logging.INFO, "mm_skip_invalid_snapshot", slug=slug, base=baseU, salt=salt, reason="prev_close")
            await self._cancel_live_orders(slug, salt)
            return
        prev_close = float(prev_close)

        S_now_raw = self._robust_S_now(snapshot, prev_close)
        if not (isinstance(S_now_raw, (int, float)) and math.isfinite(float(S_now_raw)) and float(S_now_raw) > 0.0):
            jlog(logging.INFO, "mm_skip_invalid_snapshot", slug=slug, base=baseU, salt=salt, reason="S_now")
            await self._cancel_live_orders(slug, salt)
            return
        S_now = float(S_now_raw)

        df = _read_num("model_df", required=True, positive=True, default=None)
        tau_mix = _read_num("tau", required=True, positive=True, default=None)
        iv_model_15m = _read_num("model_iv", required=True, positive=True, default=None)
        if df is None or tau_mix is None or iv_model_15m is None:
            jlog(logging.INFO, "mm_skip_invalid_snapshot", slug=slug, base=baseU, salt=salt, reason="df/tau/iv_model")
            await self._cancel_live_orders(slug, salt)
            return
        df = float(df)
        tau_mix = float(tau_mix)
        iv_model_15m = float(iv_model_15m)

        pm_iv_mult = _read_num("pm_iv_mult", required=True, positive=True, default=None)
        if pm_iv_mult is None:
            jlog(logging.INFO, "mm_skip_invalid_snapshot", slug=slug, base=baseU, salt=salt, reason="pm_iv_mult")
            await self._cancel_live_orders(slug, salt)
            return
        pm_iv_mult = float(pm_iv_mult)

        cb_px = _read_num("coinbase", required=True, positive=True, default=None)
        if cb_px is None:
            jlog(logging.INFO, "mm_skip_invalid_snapshot", slug=slug, base=baseU, salt=salt, reason="coinbase")
            await self._cancel_live_orders(slug, salt)
            return
        cb_px = float(cb_px)

        prev_low = _read_num("prev_low", required=True, positive=False, default=None)
        prev_high = _read_num("prev_high", required=True, positive=False, default=None)
        if prev_low is None or prev_high is None:
            jlog(logging.INFO, "mm_skip_invalid_snapshot", slug=slug, base=baseU, salt=salt, reason="prev_low/high")
            await self._cancel_live_orders(slug, salt)
            return
        prev_low = float(prev_low)
        prev_high = float(prev_high)

        # market TOB is REQUIRED for interval mismatch check
        mkt_bid = _read_num("bid_px", required=True, positive=False, default=None)
        mkt_ask = _read_num("ask_px", required=True, positive=False, default=None)
        if mkt_bid is None or mkt_ask is None:
            jlog(logging.INFO, "mm_skip_missing_market_tob", slug=slug, base=baseU, salt=salt, bid=mkt_bid, ask=mkt_ask)
            await self._cancel_live_orders(slug, salt)
            return
        mkt_bid = float(mkt_bid)
        mkt_ask = float(mkt_ask)
        if not (math.isfinite(mkt_bid) and math.isfinite(mkt_ask)) or mkt_ask <= mkt_bid or mkt_ask <= 0.0:
            jlog(logging.INFO, "mm_skip_invalid_market_tob", slug=slug, base=baseU, salt=salt, bid=mkt_bid, ask=mkt_ask)
            await self._cancel_live_orders(slug, salt)
            return
        mkt_mid = 0.5 * (mkt_bid + mkt_ask)

        # Quote signals (preds) are REQUIRED
        iv_mid_pred = _read_num("iv_mid_pred", required=True, positive=False, default=None)
        iv_hs_pred = _read_num("iv_hs_pred", required=True, positive=False, default=None)
        if iv_mid_pred is None or iv_hs_pred is None:
            jlog(logging.INFO, "mm_skip_missing_iv_preds", slug=slug, base=baseU, salt=salt, iv_mid_pred=iv_mid_pred, iv_hs_pred=iv_hs_pred)
            await self._cancel_live_orders(slug, salt)
            return

        pred_mid = _clip01(float(iv_mid_pred))
        hs_pred = abs(float(iv_hs_pred))

        # Realized short-term sigma inputs (OPTIONAL)
        cb_sigma_ewma = _read_num("cb_sigma_ewma", required=False, positive=True, default=0.0) or 0.0
        cb_rv_3s = _read_num("cb_rv_3s", required=False, positive=True, default=0.0) or 0.0

        seq_score = _read_num("seq_quote_score", required=False, positive=False, default=0.0) or 0.0
        seq_skew = _read_num("seq_quote_skew", required=False, positive=False, default=0.0) or 0.0
        seq_side = _read_str("seq_quote_side", default="flat")
        seq_one_sided = _read_bool("seq_quote_one_sided", default=False)

        # legacy fallbacks
        if seq_score == 0.0 and isinstance(snapshot.get("quote_score"), (int, float)):
            try:
                seq_score = float(snapshot.get("quote_score"))
            except Exception:
                pass
        if seq_skew == 0.0 and isinstance(snapshot.get("quote_skew"), (int, float)):
            try:
                seq_skew = float(snapshot.get("quote_skew"))
            except Exception:
                pass
        if seq_side == "flat" and isinstance(snapshot.get("quote_side"), str):
            seq_side = str(snapshot.get("quote_side"))

        seq_score = float(seq_score)
        seq_skew = float(seq_skew)

        iv_delta_mid = _read_num("iv_delta_mid", required=False, positive=False, default=None)
        if iv_delta_mid is None:
            iv_delta_mid = pred_mid - mkt_mid
        iv_delta_mid = float(iv_delta_mid)

        # ---------------- (2) model/vol bug checks ----------------
        if abs(iv_delta_mid) > 0.03:
            jlog(logging.INFO, "mm_cancel_all_iv_delta_mid_too_large", slug=slug, base=baseU, salt=salt, iv_delta_mid=iv_delta_mid)
            await self._cancel_live_orders(slug, salt)
            return
        if hs_pred > 0.04:
            jlog(logging.INFO, "mm_cancel_all_iv_hs_pred_too_large", slug=slug, base=baseU, salt=salt, iv_hs_pred=hs_pred)
            await self._cancel_live_orders(slug, salt)
            return

        # ---------------- (1) interval mismatch vs market TOB ----------------
        p_chk_lo = _clip01(pred_mid - 2.0 * hs_pred)
        p_chk_hi = _clip01(pred_mid + 2.0 * hs_pred)
        if (p_chk_hi < mkt_bid) or (p_chk_lo > mkt_ask):
            jlog(logging.INFO, "mm_skip_interval_mismatch", slug=slug, base=baseU, salt=salt, pred_mid=pred_mid, hs=hs_pred, p_lo=p_chk_lo, p_hi=p_chk_hi, mkt_bid=mkt_bid, mkt_ask=mkt_ask)
            await self._cancel_live_orders(slug, salt)
            return

        # ---------------- prev_low / prev_high bounds (rounded) ----------------
        spot_decimals = 4 if baseU == "XRP" else 2
        scale = 10 ** spot_decimals
        prev_low_floor = math.floor(prev_low * scale) / scale
        prev_high_ceil = math.ceil(prev_high * scale) / scale
        if prev_low_floor <= 0.0:
            prev_low_floor = prev_close
        if prev_high_ceil <= 0.0:
            prev_high_ceil = prev_close
        if prev_low_floor > prev_high_ceil:
            prev_low_floor, prev_high_ceil = prev_high_ceil, prev_low_floor

        # ---------------- Inventory snapshot ----------------
        pos = get_ws_positions() or {}
        yes_bal = float(pos.get(str(yes_id), 0.0))
        no_bal = float(pos.get(str(no_id), 0.0))
        net_yes = yes_bal - no_bal
        inv_cap = float(BASE_CFG.get(baseU, {}).get("inv_cap", 0.0))
        rho = 0.0
        if inv_cap > 0.0:
            rho = _clip(net_yes / inv_cap, -1.0, 1.0)

        # ---------------- Student-t / sigma ----------------
        sigma_real = self._sigma_from_iv_tau_model(iv_model_15m, tau_mix)
        if (not math.isfinite(sigma_real)) or sigma_real <= 0.0:
            jlog(logging.INFO, "mm_skip_invalid_sigma_real", slug=slug, base=baseU, salt=salt, sigma_real=sigma_real)
            await self._cancel_live_orders(slug, salt)
            return
        inv_sigma = 1.0 / max(sigma_real, 1e-12)

        # p_yes for an arbitrary "threshold" prev_px, using CURRENT spot S_now (used for t_fair_mid)
        def _p_yes_from_prev(prev_px: float) -> float:
            prev_eff = max(float(prev_px), 1e-12)
            Sq_eff = max(float(S_now), 1e-12)
            z = (_log(prev_eff) - _log(Sq_eff)) * inv_sigma
            u = float(student_t_cdf_np(z, df))
            return _clip01(1.0 - u)

        # p_yes as a function of spot S (threshold is prev_close) — used for inversion + realized spread
        def _p_yes_from_spot(spot_S: float) -> float:
            S_eff = max(float(spot_S), 1e-12)
            z = (_log(prev_close) - _log(S_eff)) * inv_sigma
            u = float(student_t_cdf_np(z, df))
            return _clip01(1.0 - u)

        def _invert_p_to_spot(p_target: float) -> float:
            eps = 1e-6
            p_t = _clip(float(p_target), eps, 1.0 - eps)

            lo = max(prev_close * 0.05, 1e-9)
            hi = max(prev_close * 20.0, lo * 2.0)

            for _ in range(32):
                p_lo = _p_yes_from_spot(lo)
                p_hi = _p_yes_from_spot(hi)
                if p_lo <= p_t <= p_hi:
                    break
                if p_lo > p_t:
                    lo *= 0.5
                if p_hi < p_t:
                    hi *= 2.0
                lo = max(lo, 1e-12)
                hi = max(hi, lo * 2.0)

            for _ in range(48):
                mid = 0.5 * (lo + hi)
                p_mid = _p_yes_from_spot(mid)
                if p_mid < p_t:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        # ---------------- (5) Student-t fair mid + IV skew ----------------
        p_now_low = _p_yes_from_prev(prev_low_floor)
        p_now_high = _p_yes_from_prev(prev_high_ceil)
        t_fair_mid = 0.5 * (min(p_now_low, p_now_high) + max(p_now_low, p_now_high))

        center0 = _clip01(pred_mid + seq_skew)
        iv_skew = _clip((t_fair_mid - center0) * 0.15, -0.01, 0.01)
        fair_center = _clip01(center0 + iv_skew)
        p_now = fair_center

        # ---------------- (4) fair band for cancels (kept as predicted-band) ----------------
        fair_half = 0.5 * hs_pred
        fair_yes_lo = _clip01(fair_center - fair_half)
        fair_yes_hi = _clip01(fair_center + fair_half)
        if fair_yes_hi < fair_yes_lo:
            fair_yes_lo, fair_yes_hi = fair_yes_hi, fair_yes_lo

        self._fair_state_by_slug[slug] = {
            "fair_yes_lo": float(fair_yes_lo),
            "fair_yes_hi": float(fair_yes_hi),
            "yes_id": str(yes_id),
            "no_id": str(no_id),
            "seq_version": int(seq_version),
            "ts": time.time(),
        }

        # ---------------- (3) quote half-spread via RV->prob inversion ----------------
        allow_cross = abs(seq_score) > 0.5
        PAD = 0.01

        try:
            cand1 = float(cb_sigma_ewma) if math.isfinite(float(cb_sigma_ewma)) else 0.0
        except Exception:
            cand1 = 0.0
        try:
            rv = float(cb_rv_3s)
            cand2 = (rv / math.sqrt(3.0)) if (math.isfinite(rv) and rv > 0.0) else 0.0
        except Exception:
            cand2 = 0.0
        sigma_1s = max(cand1, cand2, 0.0)

        S_center = None
        S_dn = None
        S_up = None
        p_dn = None
        p_up = None
        spread_realized_prob = 0.0
        hs_realized_prob = 0.0

        try:
            if sigma_1s > 0.0:
                S_center = _invert_p_to_spot(fair_center)
                S_dn = S_center * math.exp(-sigma_1s)
                S_up = S_center * math.exp(+sigma_1s)
                p_dn = _p_yes_from_spot(S_dn)
                p_up = _p_yes_from_spot(S_up)
                spread_realized_prob = max(0.0, float(p_up) - float(p_dn))
                hs_realized_prob = 0.5 * spread_realized_prob
        except Exception as e:
            jlog(
                logging.WARNING,
                "mm_realized_spread_compute_error",
                slug=slug,
                base=baseU,
                salt=salt,
                error=repr(e),
                sigma_1s=sigma_1s,
                fair_center=fair_center,
            )
            spread_realized_prob = 0.0
            hs_realized_prob = 0.0

        quote_half = float(hs_pred) + float(hs_realized_prob)
        if not math.isfinite(quote_half) or quote_half <= 0.0:
            jlog(
                logging.INFO,
                "mm_skip_invalid_quote_half",
                slug=slug,
                base=baseU,
                salt=salt,
                hs_pred=hs_pred,
                hs_realized_prob=hs_realized_prob,
                sigma_1s=sigma_1s,
            )
            await self._cancel_live_orders(slug, salt)
            return

        # minimum: at least one tick
        quote_half = max(quote_half, float(tick_sz))

        p_bid = _clip01(fair_center - quote_half)
        p_ask = _clip01(fair_center + quote_half)

        # predicted "book" from model outputs (still used for no-cross clamp)
        pred_book_bid = _clip01(pred_mid - hs_pred)
        pred_book_ask = _clip01(pred_mid + hs_pred)

        if not allow_cross:
            p_bid = min(p_bid, pred_book_ask - PAD)
            p_ask = max(p_ask, pred_book_bid + PAD)
            p_bid = _clip01(p_bid)
            p_ask = _clip01(p_ask)

        if p_bid < 0.01 or p_ask > 0.99:
            jlog(logging.INFO, "mm_skip_edges", slug=slug, base=baseU, salt=salt, p_bid=p_bid, p_ask=p_ask, p_now=p_now)
            await self._cancel_live_orders(slug, salt)
            return

        tick = float(tick_sz)
        yes_bid_q = _floor_to_tick(p_bid, tick)
        yes_ask_q = _ceil_to_tick(p_ask, tick)

        if not allow_cross:
            max_bid = _floor_to_tick(pred_book_ask - PAD, tick)
            min_ask = _ceil_to_tick(pred_book_bid + PAD, tick)
            if yes_bid_q > max_bid:
                yes_bid_q = max_bid
            if yes_ask_q < min_ask:
                yes_ask_q = min_ask

        if yes_ask_q <= yes_bid_q:
            pm_mid = 0.5 * (p_bid + p_ask)
            yes_bid_q = _floor_to_tick(pm_mid - tick * 0.5, tick)
            yes_ask_q = _ceil_to_tick(pm_mid + tick * 0.5, tick)
            if yes_ask_q <= yes_bid_q:
                jlog(
                    logging.INFO,
                    "mm_skip_collapsed_spread",
                    slug=slug,
                    base=baseU,
                    salt=salt,
                    p_bid=p_bid,
                    p_ask=p_ask,
                    pred_mid=pred_mid,
                    hs_pred=hs_pred,
                    hs_realized_prob=hs_realized_prob,
                    sigma_1s=sigma_1s,
                )
                await self._cancel_live_orders(slug, salt)
                return

        no_bid_q = _floor_to_tick(1.0 - yes_ask_q, tick)
        no_ask_q = _ceil_to_tick(1.0 - yes_bid_q, tick)

        # ---------------- cancels: stale + one-sided ----------------
        now_s = time.time()

        cancel_ids = self._select_stale_and_crossed_orders(
            slug=slug,
            yes_id=str(yes_id),
            no_id=str(no_id),
            fair_yes_lo=fair_yes_lo,
            fair_yes_hi=fair_yes_hi,
        )

        desired_sign = _desired_sign_from_side(seq_side, seq_score)
        strong_one_side = (desired_sign != 0) and (abs(seq_score) > 0.1) and (seq_side or "").lower() != "flat"
        if strong_one_side:
            live_meta = self._live_orders_by_slug.get(slug, []) or []
            cancel_other: List[str] = []
            keep_meta: List[Dict[str, Any]] = []
            for m in live_meta:
                oid = str(m.get("order_id", "") or "")
                if not oid:
                    continue
                token_id = str(m.get("token_id", "") or "")
                which = "YES" if token_id == str(yes_id) else ("NO" if token_id == str(no_id) else "")
                if not which:
                    cancel_other.append(oid)
                    continue
                eff = _net_yes_effect(str(m.get("side", "")), which)
                if eff != desired_sign:
                    cancel_other.append(oid)
                else:
                    keep_meta.append(m)

            if cancel_other:
                self._live_orders_by_slug[slug] = keep_meta
                self._live_order_ids_by_slug[slug] = [m.get("order_id") for m in keep_meta if m.get("order_id")]
                cancel_ids = list({*cancel_ids, *cancel_other})
                jlog(logging.INFO, "mm_one_side_cancel_other", slug=slug, base=baseU, salt=salt, seq_side=seq_side, seq_score=seq_score, n=len(cancel_other))

        if cancel_ids:
            jlog(logging.INFO, "order_cancel_stale", slug=slug, base=baseU, n=len(cancel_ids), fair_yes=p_now, salt=salt)

            async def _run_cancel() -> None:
                try:
                    await self._timed("cancel_orders_stale", self._io_retry(cancel_orders, cancel_ids), slug, salt)
                    self._remove_live_orders_by_ids(slug, cancel_ids)
                except Exception as e:
                    jlog(logging.ERROR, "order_cancel_stale_error", slug=slug, error=repr(e), salt=salt)

            asyncio.create_task(_run_cancel())

        action = self._quiet_window_action(int(start_ts), now_s)
        if action == "head_skip":
            jlog(logging.INFO, "mm_skip_head_do_not_trade", slug=slug, salt=salt)
            return
        if action == "tail_cancel":
            if int(start_ts) not in self._session_cancel_fired:
                self._session_cancel_fired.add(int(start_ts))
                jlog(logging.INFO, "tail_cancel_all_start", slug=slug, start_ts=int(start_ts), salt=salt)
                try:
                    _ = await self._timed("cancel_all_orders", self._io_retry(cancel_all_orders), slug, salt)
                except Exception as e:
                    jlog(logging.ERROR, "tail_cancel_all_error", slug=slug, error=repr(e), salt=salt)
                else:
                    jlog(logging.INFO, "tail_cancel_all_done", slug=slug, salt=salt)
                self._live_order_ids_by_slug.clear()
                self._live_orders_by_slug.clear()
            return

        # --- Volume cap per slug ---
        vol_cap = float(BASE_CFG.get(baseU, {}).get("vol_cap", 0.0))
        with self._vol_lock:
            vol_used = float(self._vol_used_by_slug.get(slug, 0.0))
        if vol_cap > 0.0 and vol_used >= vol_cap:
            jlog(logging.INFO, "mm_skip_vol_cap", slug=slug, base=baseU, vol_used=float(vol_used), vol_cap=float(vol_cap), salt=salt)
            await self._cancel_live_orders(slug, salt)
            return

        # ---------------- Inventory-driven action logic (kept) ----------------
        actions: List[Tuple[str, float, str]] = []
        over_cap_flag = None

        mm_cfg = BASE_CFG.get(baseU, {})
        try:
            qsize_cfg = float(mm_cfg.get("quote_size", min_size))
        except Exception:
            qsize_cfg = float(min_size)
        effective_qsize = max(qsize_cfg, float(min_size))

        bal_thresh = 5.0 * effective_qsize

        actions = [
            ("BUY", yes_bid_q, "YES"),
            ("BUY", no_bid_q, "NO"),
        ]

        if yes_bal > bal_thresh:
            new_actions: List[Tuple[str, float, str]] = []
            for side, price, which in actions:
                if side == "BUY" and which == "NO":
                    new_actions.append(("SELL", yes_ask_q, "YES"))
                else:
                    new_actions.append((side, price, which))
            actions = new_actions

        if no_bal > bal_thresh:
            new_actions = []
            for side, price, which in actions:
                if side == "BUY" and which == "YES":
                    new_actions.append(("SELL", no_ask_q, "NO"))
                else:
                    new_actions.append((side, price, which))
            actions = new_actions

        inv_cap_f = float(inv_cap)
        lower_cap = -inv_cap_f
        upper_cap = inv_cap_f

        if inv_cap_f > 0.0:
            inv_tilt_sign = 0
            inv_tilt_progress = 0.0

            if cb_px > 0.0 and prev_close > 0.0 and pm_iv_mult > 0.0:
                if cb_px > prev_close:
                    d_price = 1.0
                elif cb_px < prev_close:
                    d_price = -1.0
                else:
                    d_price = 0.0

                if pm_iv_mult > 1.0:
                    d_mult = 1.0
                elif pm_iv_mult < 1.0:
                    d_mult = -1.0
                else:
                    d_mult = 0.0

                if d_price != 0.0 and d_mult != 0.0:
                    inv_tilt_sign = 1 if (d_price * d_mult) > 0.0 else -1
                    tau_clamped = max(0.0, min(float(tau_mix), float(WINDOW_SEC)))
                    inv_tilt_progress = 1.0 - tau_clamped / float(WINDOW_SEC)
                    inv_tilt_progress = max(0.0, min(inv_tilt_progress, 1.0))
                    p_prog = inv_tilt_progress

                    if inv_tilt_sign > 0:
                        lower_cap = inv_cap_f * (1.2 * p_prog - 1.0)
                        upper_cap = inv_cap_f
                    else:
                        upper_cap = inv_cap_f * (1.0 - 1.2 * p_prog)
                        lower_cap = -inv_cap_f

            lower_cap = max(-inv_cap_f, min(lower_cap, inv_cap_f))
            upper_cap = max(-inv_cap_f, min(upper_cap, inv_cap_f))
            if lower_cap > upper_cap:
                lower_cap, upper_cap = upper_cap, lower_cap

            if net_yes > upper_cap:
                over_cap_flag = "YES_LONG"

                def _would_increase_net_yes(side: str, which: str) -> bool:
                    if which == "YES":
                        return side.upper() == "BUY"
                    else:
                        return side.upper() == "SELL"

                actions = [(side, price, which) for (side, price, which) in actions if not _would_increase_net_yes(side, which)]

            elif net_yes < lower_cap:
                over_cap_flag = "NO_LONG"

                def _would_decrease_net_yes(side: str, which: str) -> bool:
                    if which == "YES":
                        return side.upper() == "SELL"
                    else:
                        return side.upper() == "BUY"

                actions = [(side, price, which) for (side, price, which) in actions if not _would_decrease_net_yes(side, which)]

        # one-sided filter for new submissions unless over-cap (risk > signal)
        if strong_one_side and over_cap_flag is None:
            filtered: List[Tuple[str, float, str]] = []
            for side, price, which in actions:
                if _net_yes_effect(side, which) == desired_sign:
                    filtered.append((side, price, which))
            actions = filtered

        # ---------------- Submit with NEW order management ----------------
        qsize = effective_qsize
        size_to_use = max(qsize, float(min_size))

        order_list: List[Dict[str, Any]] = []
        for side, price, which in actions:
            token_id = yes_id if which == "YES" else no_id
            order_list.append(
                {"token_id": str(token_id), "side": str(side).upper(), "price": float(price), "size": float(size_to_use)}
            )

        # Apply stacking + max-live rules (cancel furthest if needed, then submit)
        await self._ensure_capacity_then_submit(
            slug=slug,
            proposed_orders=order_list,
            fair_center=fair_center,
            yes_id=str(yes_id),
            no_id=str(no_id),
            seq_version=seq_version,
            salt=salt,
        )

        await _pause(ATOMIC_POST_SUBMIT_WAIT_S, "post_submit", slug, salt)

        with self._vol_lock:
            vol_used_log = float(self._vol_used_by_slug.get(slug, 0.0))

        jlog(
            logging.INFO,
            "mm_quote_update",
            slug=slug,
            base=baseU,
            salt=salt,
            mkt_bid=mkt_bid,
            mkt_ask=mkt_ask,
            pred_mid=pred_mid,
            iv_delta_mid=iv_delta_mid,
            iv_hs_pred=hs_pred,
            cb_sigma_ewma=float(cb_sigma_ewma),
            cb_rv_3s=float(cb_rv_3s),
            sigma_1s=float(sigma_1s),
            S_center=(float(S_center) if isinstance(S_center, (int, float)) else None),
            S_dn=(float(S_dn) if isinstance(S_dn, (int, float)) else None),
            S_up=(float(S_up) if isinstance(S_up, (int, float)) else None),
            p_dn=(float(p_dn) if isinstance(p_dn, (int, float)) else None),
            p_up=(float(p_up) if isinstance(p_up, (int, float)) else None),
            spread_realized_prob=float(spread_realized_prob),
            hs_realized_prob=float(hs_realized_prob),
            quote_half_used=float(quote_half),
            p_chk_lo=p_chk_lo,
            p_chk_hi=p_chk_hi,
            seq_score=float(seq_score),
            seq_skew=float(seq_skew),
            seq_side=str(seq_side),
            seq_one_sided=bool(seq_one_sided),
            t_fair_mid=float(t_fair_mid),
            iv_skew=float(iv_skew),
            fair_center=float(fair_center),
            fair_yes_lo=float(fair_yes_lo),
            fair_yes_hi=float(fair_yes_hi),
            p_bid=float(p_bid),
            p_ask=float(p_ask),
            p_now=float(p_now),
            yes_bid=float(yes_bid_q),
            yes_ask=float(yes_ask_q),
            no_bid=float(no_bid_q),
            no_ask=float(no_ask_q),
            size=float(qsize),
            df=float(df),
            tau=float(tau_mix),
            model_iv_15m=float(iv_model_15m),
            pm_iv_mult=float(pm_iv_mult),
            inv_yes=float(yes_bal),
            inv_no=float(no_bal),
            net_yes=float(net_yes),
            rho=float(rho),
            inv_cap=float(inv_cap),
            inv_lower_cap=float(lower_cap),
            inv_upper_cap=float(upper_cap),
            over_cap_flag=over_cap_flag,
            vol_used=float(vol_used_log),
            vol_cap=float(BASE_CFG.get(baseU, {}).get("vol_cap", 0.0)),
            n_live=int(len(self._live_order_ids_by_slug.get(slug, []))),
            max_live_per_side=int(MAX_LIVE_PER_SIDE),
            max_same_price=int(MAX_SAME_PRICE),
            inactivity_cancel_s=float(ORDER_TTS_CANCEL_S),
        )
