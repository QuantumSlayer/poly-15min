#!/usr/bin/env python3
"""
quote_iv.py — Online event-sequence builder + IVTransformer inference for Polymarket quote prediction.

FIX (Dec 2025)
--------------
Your ingestion can be out-of-order (server streaming + network jitter), so appending into a deque
creates a NON-monotone timestamp sequence. That breaks:
  - segment building (max_gap_sec checks)
  - time_lag (can become negative)
  - and carry-forward semantics when a late/older event should have affected later events.

This version fixes it by:
  1) Maintaining per-base events SORTED by timestamp (stable for same-ts: later arrivals go after).
  2) Storing raw per-event updates, and REBUILDING snapshots from the insertion point forward
     whenever an out-of-order event arrives, so carry-forward rules remain correct.
  3) Keeping a "base_state" that represents the carried-forward state from dropped (older) events,
     so bounded storage remains consistent without full rebuild every time.
  4) Robust mid_base extraction: use the LAST valid bid/ask in the (unpadded) tail, not blindly the
     last token (which might be a ticker event).

Also added (requested)
----------------------
- clear_all_states(): clears all per-base sequences/states so you can switch contracts every 15m.
- More "complete" prediction metadata:
    preds include: mid_base, last_best_bid/ask, pad_len, used_len
    status includes: segment_len_raw, segment_len_used, n_events_total, cb/pm ts, t_last_s
"""

from __future__ import annotations

import asyncio
import bisect
import json
import logging
import math
import threading
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

# Import model definition + base ordering from train_iv.py (keeps mapping consistent with training)
from train_iv import IVTransformer, BASES  # type: ignore


# -----------------------------
# Defaults (match build_iv.py)
# -----------------------------
MIN_SEQ_LEN_DEFAULT = 16
MAX_SEQ_LEN_DEFAULT = 64
MAX_GAP_SEC_DEFAULT = 1.0


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _softplus(x: float) -> float:
    # numerically stable softplus
    if x > 20:
        return x
    if x < -20:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _resolve_run_dir(model_dir: Path) -> Path:
    """
    Accept either:
      - a run directory containing args.json + model_best.pt
      - a runs root containing multiple timestamped subdirs
    Pick the newest subdir if needed.
    """
    if (model_dir / "args.json").exists() and (model_dir / "model_best.pt").exists():
        return model_dir

    cands: List[Path] = []
    if model_dir.exists() and model_dir.is_dir():
        for p in model_dir.iterdir():
            if not p.is_dir():
                continue
            if (p / "args.json").exists() and (p / "model_best.pt").exists():
                cands.append(p)

    if not cands:
        raise FileNotFoundError(f"No run dir found under: {model_dir}")

    cands.sort(key=lambda x: x.name)
    return cands[-1]


def _json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_norm_stats_iv(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Robust parser for norm_stats_iv.json. Accept a few possible schemas:
      A) {"feat_mean": {"BTC":[...], ...}, "feat_std": {"BTC":[...], ...}, ...}
      B) {"bases": {"BTC":{"mean":[...],"std":[...]}, ...}, ...}
      C) {"BTC":{"mean":[...],"std":[...]}, "ETH":{...}, ...}
    Returns dict: base -> {"mean": np.ndarray, "std": np.ndarray}
    """
    raw = _json_load(path)
    out: Dict[str, Dict[str, np.ndarray]] = {}

    if isinstance(raw.get("feat_mean"), dict) and isinstance(raw.get("feat_std"), dict):
        for b, arr in raw["feat_mean"].items():
            if b in raw["feat_std"]:
                out[b.upper()] = {
                    "mean": np.asarray(arr, dtype=np.float32),
                    "std": np.asarray(raw["feat_std"][b], dtype=np.float32),
                }
        return out

    if isinstance(raw.get("bases"), dict):
        for b, st in raw["bases"].items():
            if isinstance(st, dict) and "mean" in st and "std" in st:
                out[b.upper()] = {
                    "mean": np.asarray(st["mean"], dtype=np.float32),
                    "std": np.asarray(st["std"], dtype=np.float32),
                }
        if out:
            return out

    # fallback: direct base keys
    bases_up = {x.upper() for x in BASES}
    for b in list(raw.keys()):
        if b.upper() in bases_up and isinstance(raw[b], dict):
            st = raw[b]
            if "mean" in st and "std" in st:
                out[b.upper()] = {
                    "mean": np.asarray(st["mean"], dtype=np.float32),
                    "std": np.asarray(st["std"], dtype=np.float32),
                }
    return out


def _book_summary_from_levels(
    bids: Any,
    asks: Any,
    top_n: int = 5,
) -> Dict[str, Optional[float]]:
    """
    Match build_iv.py's book summary features:
      pm_best_bid, pm_best_ask, pm_mid, pm_spread,
      pm_size_bid_top, pm_size_ask_top, pm_imbalance_top
    """
    best_bid = best_ask = None
    size_bid_top = size_ask_top = None

    try:
        if isinstance(bids, list) and bids:
            best_bid = _safe_float(bids[0][0])
            s = 0.0
            ok = False
            for lvl in bids[:top_n]:
                sz = _safe_float(lvl[1])
                if sz is None:
                    continue
                s += float(sz)
                ok = True
            size_bid_top = s if ok else None
    except Exception:
        pass

    try:
        if isinstance(asks, list) and asks:
            best_ask = _safe_float(asks[0][0])
            s = 0.0
            ok = False
            for lvl in asks[:top_n]:
                sz = _safe_float(lvl[1])
                if sz is None:
                    continue
                s += float(sz)
                ok = True
            size_ask_top = s if ok else None
    except Exception:
        pass

    mid = spread = imb_top = None
    if best_bid is not None and best_ask is not None:
        mid = 0.5 * (float(best_bid) + float(best_ask))
        spread = float(best_ask) - float(best_bid)

    if size_bid_top is not None and size_ask_top is not None:
        denom = float(size_bid_top) + float(size_ask_top)
        if denom > 0:
            imb_top = (float(size_bid_top) - float(size_ask_top)) / denom

    return {
        "pm_best_bid": best_bid,
        "pm_best_ask": best_ask,
        "pm_mid": mid,
        "pm_spread": spread,
        "pm_size_bid_top": size_bid_top,
        "pm_size_ask_top": size_ask_top,
        "pm_imbalance_top": imb_top,
    }


# kind tags for events
_KIND_TICKER = 1
_KIND_BOOK = 2


@dataclass
class _Event:
    t_s: float
    kind: int
    updates: Tuple[Tuple[int, float], ...]  # (feature_idx, value_or_nan)
    x: np.ndarray  # snapshot after applying updates at t_s (time_lag=0; flags set per kind)


class _BaseSeq:
    """
    Per-base ordered event store with correct carry-forward semantics under out-of-order arrival.

    Invariants:
      - events sorted by t_s ascending (stable for equal ts)
      - each event.x is the state snapshot AFTER applying its updates, with event flags set for that row
      - base_state is carried-forward state prior to events[0], reflecting dropped history
    """

    def __init__(self, base: str, feature_names: List[str], max_store: int):
        self.base = base
        self.feature_names = feature_names
        self.D = len(feature_names)
        self.max_store = int(max_store)

        self.fidx: Dict[str, int] = {k: i for i, k in enumerate(feature_names)}
        self.idx_time_lag = self.fidx["time_lag"]
        self.idx_is_ticker = self.fidx["is_ticker_update"]
        self.idx_is_book = self.fidx["is_book_update"]
        self._protected_idxs = {self.idx_time_lag, self.idx_is_ticker, self.idx_is_book}

        self.ts_list: List[float] = []
        self.events: List[_Event] = []

        self.base_state = np.full(self.D, np.nan, dtype=np.float32)
        self.base_state[self.idx_time_lag] = 0.0
        self.base_state[self.idx_is_ticker] = 0.0
        self.base_state[self.idx_is_book] = 0.0
        self.base_state_t_s: float = float("-inf")

        self.tail_state = self.base_state.copy()

        self.last_cb_ts_server_ms: Optional[int] = None
        self.last_pm_ts_server_ms: Optional[int] = None

    def clear(self) -> None:
        self.ts_list.clear()
        self.events.clear()
        self.base_state = np.full(self.D, np.nan, dtype=np.float32)
        self.base_state[self.idx_time_lag] = 0.0
        self.base_state[self.idx_is_ticker] = 0.0
        self.base_state[self.idx_is_book] = 0.0
        self.base_state_t_s = float("-inf")
        self.tail_state = self.base_state.copy()
        self.last_cb_ts_server_ms = None
        self.last_pm_ts_server_ms = None

    def _pack_updates(self, updates: Dict[str, Optional[float]]) -> Tuple[Tuple[int, float], ...]:
        out: List[Tuple[int, float]] = []
        for k, v in updates.items():
            i = self.fidx.get(k)
            if i is None:
                continue
            if i in self._protected_idxs:
                continue
            out.append((i, np.nan if v is None else float(v)))
        return tuple(out)

    def _state_from_x(self, x: np.ndarray) -> np.ndarray:
        st = x.copy()
        st[self.idx_time_lag] = 0.0
        st[self.idx_is_ticker] = 0.0
        st[self.idx_is_book] = 0.0
        return st

    def _apply_updates(self, state: np.ndarray, ups: Tuple[Tuple[int, float], ...]) -> None:
        for i, v in ups:
            state[i] = v

    def _make_x(self, state: np.ndarray, kind: int) -> np.ndarray:
        x = state.copy()
        x[self.idx_time_lag] = 0.0
        x[self.idx_is_ticker] = 1.0 if kind == _KIND_TICKER else 0.0
        x[self.idx_is_book] = 1.0 if kind == _KIND_BOOK else 0.0
        return x

    def _rebuild_from(self, start_idx: int) -> None:
        if start_idx < 0:
            start_idx = 0
        if start_idx >= len(self.events):
            self.tail_state = self._state_from_x(self.events[-1].x) if self.events else self.base_state.copy()
            return

        if start_idx == 0:
            state = self.base_state.copy()
        else:
            state = self._state_from_x(self.events[start_idx - 1].x)

        for j in range(start_idx, len(self.events)):
            ev = self.events[j]
            self._apply_updates(state, ev.updates)
            ev.x = self._make_x(state, ev.kind)

        self.tail_state = self._state_from_x(self.events[-1].x) if self.events else self.base_state.copy()

    def _drop_oldest_to_fit(self) -> None:
        if self.max_store <= 0:
            self.clear()
            return

        extra = len(self.events) - self.max_store
        if extra <= 0:
            return

        dropped_last = self.events[extra - 1]
        self.base_state = self._state_from_x(dropped_last.x)
        self.base_state_t_s = float(dropped_last.t_s)

        del self.events[:extra]
        del self.ts_list[:extra]

        self.tail_state = self._state_from_x(self.events[-1].x) if self.events else self.base_state.copy()

    def add_event(self, *, t_s: float, kind: int, updates: Dict[str, Optional[float]]) -> None:
        t_s = float(t_s)

        # too-old vs already dropped history => cannot incorporate consistently
        if t_s < self.base_state_t_s:
            return

        ups = self._pack_updates(updates)

        # empty
        if not self.events:
            state = self.base_state.copy()
            self._apply_updates(state, ups)
            x = self._make_x(state, kind)
            self.events.append(_Event(t_s=t_s, kind=kind, updates=ups, x=x))
            self.ts_list.append(t_s)
            self.tail_state = self._state_from_x(x)
            self._drop_oldest_to_fit()
            return

        last_t = self.ts_list[-1]

        # in-order append (stable for same-ts)
        if t_s >= last_t:
            state = self.tail_state.copy()
            self._apply_updates(state, ups)
            x = self._make_x(state, kind)
            self.events.append(_Event(t_s=t_s, kind=kind, updates=ups, x=x))
            self.ts_list.append(t_s)
            self.tail_state = self._state_from_x(x)
            self._drop_oldest_to_fit()
            return

        # out-of-order insert: rightmost for stability among equal timestamps
        i = bisect.bisect_right(self.ts_list, t_s)

        # if full and would be dropped immediately, ignore
        if len(self.events) >= self.max_store and i == 0:
            return

        dummy_x = np.zeros((self.D,), dtype=np.float32)
        self.events.insert(i, _Event(t_s=t_s, kind=kind, updates=ups, x=dummy_x))
        self.ts_list.insert(i, t_s)

        self._rebuild_from(i)
        self._drop_oldest_to_fit()


class IVQuoteEngine:
    """
    Live IV quote predictor.

    Sources:
      - add_cb_tick(snapshot): ticker-ish / cb_tick_pred derived event
      - add_pm_book_update(...): polymarket book recompute event

    Prediction:
      - predict_for_base(base): sync
      - predict_for_base_async(base): async-friendly
      - optional async workers via start_predict_workers()

    Clearing:
      - clear_all_states(): clears all built sequences (for 15m contract roll)
      - clear_base(base): clears a single base
    """

    def __init__(
        self,
        model_dir: Path | str,
        data_dir: Optional[Path | str] = None,
        device: Optional[str] = None,
        min_seq_len: int = MIN_SEQ_LEN_DEFAULT,
        max_seq_len: int = MAX_SEQ_LEN_DEFAULT,
        max_gap_sec: float = MAX_GAP_SEC_DEFAULT,
        max_store: int = 4096,
        enable_async_predict: bool = True,
    ) -> None:
        self.model_dir = _resolve_run_dir(Path(model_dir))
        self.args = _json_load(self.model_dir / "args.json")

        if data_dir is None:
            dd = self.args.get("data_dir")
            if not dd:
                raise ValueError("data_dir not provided and args.json has no 'data_dir'")
            self.data_dir = Path(dd)
        else:
            self.data_dir = Path(data_dir)

        self.min_seq_len = int(min_seq_len)
        self.max_seq_len = int(max_seq_len)
        self.max_gap_sec = float(max_gap_sec)
        self.max_store = int(max_store)

        # dataset stats (feature ordering + decoding config)
        self.dataset_stats = _json_load(self.model_dir / "dataset_stats.json")
        self.feature_names: List[str] = list(self.dataset_stats["feature_names"])
        self.D = len(self.feature_names)
        self._fidx = {k: i for i, k in enumerate(self.feature_names)}

        # indices for mid_base extraction
        self.idx_feat_bid = int(self.dataset_stats.get("idx_feat_bid", self._fidx.get("pm_best_bid", -1)))
        self.idx_feat_ask = int(self.dataset_stats.get("idx_feat_ask", self._fidx.get("pm_best_ask", -1)))
        self.idx_feat_mid = int(self.dataset_stats.get("idx_feat_mid", self._fidx.get("pm_mid", -1)))

        # output decoding mode detection
        self.MID_SCALE = float(self.dataset_stats.get("MID_SCALE", 100.0))
        self.HS_SCALE = float(self.dataset_stats.get("HS_SCALE", 1.0))
        self.spread_activation = str(self.dataset_stats.get("spread_activation", "")).lower().strip()
        self.has_new_scales = ("MID_SCALE" in self.dataset_stats) or ("HS_SCALE" in self.dataset_stats)

        # norm stats
        norm_path = self.data_dir / "norm_stats_iv.json"
        self.norm = _parse_norm_stats_iv(norm_path)
        if not self.norm:
            raise FileNotFoundError(f"Could not parse norm stats: {norm_path}")

        # base mapping consistent with training
        self.bases: List[str] = [b.upper() for b in BASES]
        self.base_to_id: Dict[str, int] = {b: i for i, b in enumerate(self.bases)}

        # device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # model hyperparams
        d_model = int(self.args.get("d_model", 192))
        n_heads = int(self.args.get("n_heads", 8))
        n_layers = int(self.args.get("n_layers", 4))
        dim_feedforward = int(self.args.get("dim_feedforward", 384))
        dropout = float(self.args.get("dropout", 0.10))

        self.model = IVTransformer(
            seq_dim=self.D,
            n_bases=len(self.bases),
            seq_len=self.max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        ckpt = torch.load(self.model_dir / "model_best.pt", map_location="cpu")
        # robust checkpoint handling
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        # per-base buffers
        self._bases: Dict[str, _BaseSeq] = {
            b: _BaseSeq(b, self.feature_names, max_store=self.max_store)
            for b in self.bases
        }

        # torch norm tensors per base
        self._mean_t: Dict[str, torch.Tensor] = {}
        self._std_t: Dict[str, torch.Tensor] = {}
        for b in self.bases:
            st = self.norm.get(b)
            if st is None:
                raise KeyError(f"Missing norm stats for base={b} in {norm_path}")
            mean = np.asarray(st["mean"], dtype=np.float32)
            std = np.asarray(st["std"], dtype=np.float32)
            if mean.shape[0] != self.D or std.shape[0] != self.D:
                raise ValueError(f"Norm dim mismatch for {b}: mean/std={mean.shape}/{std.shape}, D={self.D}")
            std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
            self._mean_t[b] = torch.from_numpy(mean)
            self._std_t[b] = torch.from_numpy(std)

        # base_id tensors
        self._base_id_t: Dict[str, torch.Tensor] = {
            b: torch.tensor([self.base_to_id[b]], dtype=torch.long, device=self.device)
            for b in self.bases
        }

        # async prediction queues (drop intermediate prediction requests)
        self.enable_async_predict = bool(enable_async_predict)
        self._predict_q: Dict[str, asyncio.Queue] = {}
        self._predict_tasks: List[asyncio.Task] = []
        self._on_pred: Optional[Callable[[str, Dict[str, Any], Dict[str, Any]], None]] = None

        self._lock = threading.RLock()

        if self.enable_async_predict:
            for b in self.bases:
                self._predict_q[b] = asyncio.Queue(maxsize=1)

        LOGGER.info(
            "[IVQuoteEngine] run=%s device=%s bases=%s D=%d min/max=%d/%d max_gap=%.3f max_store=%d act=%s",
            str(self.model_dir),
            str(self.device),
            ",".join(self.bases),
            self.D,
            self.min_seq_len,
            self.max_seq_len,
            self.max_gap_sec,
            self.max_store,
            self.spread_activation or "relu",
        )

    # -------------------------
    # Clearing (requested)
    # -------------------------
    def clear_base(self, base: str) -> None:
        baseU = str(base).upper()
        if baseU not in self._bases:
            return
        with self._lock:
            self._bases[baseU].clear()

    def clear_all_states(self) -> None:
        """
        Clear all built sequences/states for all bases.
        Use this at each 15-min contract roll (expiry switch).
        """
        with self._lock:
            for bs in self._bases.values():
                bs.clear()

        # also drain prediction queues to avoid immediately predicting on stale queued signals
        if self.enable_async_predict:
            for q in self._predict_q.values():
                try:
                    while q.full():
                        _ = q.get_nowait()
                        q.task_done()
                except Exception:
                    pass

    # -------------------------
    # Optional async prediction
    # -------------------------
    def start_predict_workers(
        self,
        on_pred: Optional[Callable[[str, Dict[str, Any], Dict[str, Any]], None]] = None,
    ) -> None:
        if not self.enable_async_predict:
            raise RuntimeError("enable_async_predict=False (set it True to use workers).")

        self._on_pred = on_pred
        for b in self.bases:
            self._predict_tasks.append(asyncio.create_task(self._predict_worker(b)))

    async def stop_predict_workers(self) -> None:
        for t in self._predict_tasks:
            t.cancel()
        await asyncio.gather(*self._predict_tasks, return_exceptions=True)
        self._predict_tasks.clear()

    def _signal_predict(self, base: str) -> None:
        if not self.enable_async_predict:
            return
        q = self._predict_q.get(base)
        if q is None:
            return
        if q.full():
            try:
                _ = q.get_nowait()
                q.task_done()
            except Exception:
                pass
        try:
            q.put_nowait(1)
        except Exception:
            pass

    async def _predict_worker(self, base: str) -> None:
        q = self._predict_q[base]
        while True:
            _ = await q.get()
            try:
                preds, status = await self.predict_for_base_async(base)
                if status.get("ok") and preds is not None and self._on_pred is not None:
                    self._on_pred(base, preds, status)
            finally:
                q.task_done()

    # -------------------------
    # Event ingestion (2 sources)
    # -------------------------
    @staticmethod
    def _ts_from_cb_snapshot(snap: Dict[str, Any]) -> float:
        v = _safe_float(snap.get("cb_last_ts"))
        if v is not None:
            return float(v)

        v2 = snap.get("cb_ts_server_ms")
        if v2 is not None:
            try:
                return float(int(v2)) / 1000.0
            except Exception:
                pass

        v3 = snap.get("log_ts_ms")
        if v3 is not None:
            try:
                return float(int(v3)) / 1000.0
            except Exception:
                pass

        return float(_time.time())

    @staticmethod
    def _ts_from_pm_update(ts_server_ms: Any = None, ts_s: Any = None) -> float:
        v = _safe_float(ts_s)
        if v is not None:
            return float(v)
        if ts_server_ms is not None:
            try:
                return float(int(ts_server_ms)) / 1000.0
            except Exception:
                pass
        return float(_time.time())

    def add_cb_tick(self, snapshot: Dict[str, Any], schedule_predict: bool = True) -> None:
        """
        Coinbase tick push event (cb_tick_pred or cb_tick push).
        Stored timestamp-sorted; late/older ticks are inserted and may trigger rebuild.
        """
        base = str(snapshot.get("base", "")).upper()
        if base not in self._bases:
            return

        t_s = self._ts_from_cb_snapshot(snapshot)
        bs = self._bases[base]

        cb_ms = snapshot.get("cb_ts_server_ms")
        if cb_ms is not None:
            try:
                bs.last_cb_ts_server_ms = int(cb_ms)
            except Exception:
                pass
        pm_ms = snapshot.get("pm_ts_server_ms")
        if pm_ms is not None:
            try:
                bs.last_pm_ts_server_ms = int(pm_ms)
            except Exception:
                pass

        updates: Dict[str, Optional[float]] = {}

        # log_rel_px = log(coinbase/prev_close)
        coinbase = _safe_float(snapshot.get("coinbase"))
        prev_close = _safe_float(snapshot.get("prev_close"))
        log_rel_px = None
        if coinbase is not None and prev_close is not None and prev_close > 0 and coinbase > 0:
            log_rel_px = math.log(float(coinbase) / float(prev_close))
        if "log_rel_px" in self._fidx:
            updates["log_rel_px"] = log_rel_px

        # common numeric inputs
        for k in [
            "cb_sigma_ewma",
            "cb_rv_3s",
            "cb_buy_frac_1s",
            "cb_buy_frac_5s",
            "cb_flow1s_net",
            "cb_depth_imb_1bp",
            "cb_spread_bp",
            "pm_iv_implied_900",
            "model_iv",
            "tau",
        ]:
            if k in self._fidx:
                updates[k] = _safe_float(snapshot.get(k))

        # Optional: if snapshot includes book TOB, refresh these too
        bid_px = _safe_float(snapshot.get("bid_px"))
        ask_px = _safe_float(snapshot.get("ask_px"))
        if bid_px is not None and "pm_best_bid" in self._fidx:
            updates["pm_best_bid"] = bid_px
        if ask_px is not None and "pm_best_ask" in self._fidx:
            updates["pm_best_ask"] = ask_px
        if "pm_mid" in self._fidx and bid_px is not None and ask_px is not None:
            updates["pm_mid"] = 0.5 * (bid_px + ask_px)
        if "pm_spread" in self._fidx and bid_px is not None and ask_px is not None:
            updates["pm_spread"] = ask_px - bid_px

        with self._lock:
            bs.add_event(t_s=t_s, kind=_KIND_TICKER, updates=updates)

        if schedule_predict:
            self._signal_predict(base)

    def add_pm_book_update(
        self,
        base: str,
        bids: Any,
        asks: Any,
        ts_server_ms: Any = None,
        ts_s: Any = None,
        schedule_predict: bool = True,
    ) -> None:
        """
        Polymarket orderbook update event.
        Stored timestamp-sorted; late/older book updates are inserted and rebuild forward.
        """
        baseU = str(base).upper()
        if baseU not in self._bases:
            return

        t_s2 = self._ts_from_pm_update(ts_server_ms=ts_server_ms, ts_s=ts_s)
        bs = self._bases[baseU]
        if ts_server_ms is not None:
            try:
                bs.last_pm_ts_server_ms = int(ts_server_ms)
            except Exception:
                pass

        book = _book_summary_from_levels(bids=bids, asks=asks, top_n=5)

        updates: Dict[str, Optional[float]] = {}
        for k, v in book.items():
            if k in self._fidx:
                updates[k] = v

        with self._lock:
            bs.add_event(t_s=t_s2, kind=_KIND_BOOK, updates=updates)

        if schedule_predict:
            self._signal_predict(baseU)

    # -------------------------
    # Prediction helpers
    # -------------------------
    def _build_latest_segment(self, base: str) -> Tuple[List[_Event], int, float]:
        """
        Walk backward from the latest event and collect the contiguous segment
        where consecutive gaps <= max_gap_sec (events are timestamp-sorted).
        Returns (events_in_segment_chrono, segment_len_raw, t_last).
        """
        bs = self._bases[base]
        if not bs.events:
            return [], 0, float("nan")

        evs = bs.events
        t_last = float(evs[-1].t_s)

        seg_rev: List[_Event] = [evs[-1]]
        prev_t = t_last

        for ev in reversed(evs[:-1]):
            dt = prev_t - float(ev.t_s)
            if math.isfinite(dt) and (0.0 <= dt <= self.max_gap_sec):
                seg_rev.append(ev)
                prev_t = float(ev.t_s)
            else:
                break

        seg = list(reversed(seg_rev))
        return seg, len(seg), t_last

    def _make_model_inputs(
        self, base: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[float], Optional[float], Optional[float], Dict[str, Any]]:
        """
        Create (x_seq, mask, mid_base, last_bid, last_ask, status_stub).
        x_seq: (1, T, D), T=max_seq_len
        mask: (1, T) bool
        """
        bs = self._bases[base]
        n_total = len(bs.events)

        status: Dict[str, Any] = {
            "base": base,
            "ok": False,
            "reason": "",
            "n_events_total": n_total,
            "segment_len_raw": 0,
            "segment_len_used": 0,
            "min_seq_len": self.min_seq_len,
            "max_seq_len": self.max_seq_len,
            "max_gap_sec": self.max_gap_sec,
            "t_last_s": None,
            "cb_ts_server_ms": bs.last_cb_ts_server_ms,
            "pm_ts_server_ms": bs.last_pm_ts_server_ms,
        }

        seg, seg_len_raw, t_last = self._build_latest_segment(base)
        status["segment_len_raw"] = seg_len_raw
        status["t_last_s"] = float(t_last) if math.isfinite(t_last) else None

        if seg_len_raw < self.min_seq_len:
            status["reason"] = "segment_too_short"
            return None, None, None, None, None, status

        tail = seg[-self.max_seq_len :]
        n = len(tail)
        status["segment_len_used"] = n

        X = np.zeros((self.max_seq_len, self.D), dtype=np.float32)
        M = np.zeros((self.max_seq_len,), dtype=np.bool_)

        # right-align
        start = self.max_seq_len - n
        for i, ev in enumerate(tail):
            X[start + i, :] = ev.x
            M[start + i] = True

        # fill time_lag on valid tokens BEFORE sanitization
        idx_time_lag = self._fidx["time_lag"]
        for i, ev in enumerate(tail):
            lag = float(t_last) - float(ev.t_s)
            if not math.isfinite(lag) or lag < 0:
                lag = 0.0
            X[start + i, idx_time_lag] = float(lag)

        # mid_base from LAST VALID bid/ask in the tail (not padded)
        mid_base: Optional[float] = None
        last_bid: Optional[float] = None
        last_ask: Optional[float] = None

        if self.idx_feat_bid >= 0 and self.idx_feat_ask >= 0:
            for i in range(n - 1, -1, -1):
                row = X[start + i, :]
                bid = float(row[self.idx_feat_bid])
                ask = float(row[self.idx_feat_ask])
                if math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
                    last_bid, last_ask = bid, ask
                    mid_base = 0.5 * (bid + ask)
                    break

        if mid_base is None and self.idx_feat_mid >= 0:
            for i in range(n - 1, -1, -1):
                row = X[start + i, :]
                mid_v = float(row[self.idx_feat_mid])
                # polymarket probs are in [0,1], but allow slightly outside due to noise
                if math.isfinite(mid_v) and (-0.25 < mid_v < 1.25):
                    mid_base = mid_v
                    break

        # sanitize AFTER mid_base extraction
        X[~np.isfinite(X)] = 0.0

        # normalize
        mean_t = self._mean_t[base]
        std_t = self._std_t[base]

        x_t = torch.from_numpy(X)  # (T,D)
        m_t = torch.from_numpy(M)  # (T,)
        x_t = (x_t - mean_t) / std_t

        x_t = x_t.unsqueeze(0).to(self.device)  # (1,T,D)
        m_t = m_t.unsqueeze(0).to(self.device)  # (1,T)

        status["ok"] = True
        status["reason"] = "ok"
        return x_t, m_t, mid_base, last_bid, last_ask, status

    def _decode_outputs(self, base: str, mid_base: Optional[float], z0: float, z1: float) -> Dict[str, Any]:
        """
        Convert raw model outputs (z0,z1) into mid/hs/bid/ask predictions.
        """
        mid_scale = float(self.MID_SCALE if self.MID_SCALE else 100.0)
        hs_scale = float(self.HS_SCALE if self.HS_SCALE else 1.0)

        delta_mid = z0 / mid_scale

        if self.has_new_scales or self.spread_activation == "softplus":
            hs = _softplus(z1) / hs_scale
        else:
            hs = max(0.0, z1)

        mb = float(mid_base) if (mid_base is not None and math.isfinite(float(mid_base))) else 0.0
        mid_pred = mb + float(delta_mid)
        bid_pred = mid_pred - hs
        ask_pred = mid_pred + hs

        return {
            "mid_pred": float(mid_pred),
            "hs_pred": float(hs),
            "spread_pred": float(2.0 * hs),
            "bid_pred": float(bid_pred),
            "ask_pred": float(ask_pred),
            "delta_mid": float(delta_mid),
            "z0": float(z0),
            "z1": float(z1),
        }

    # -------------------------
    # Prediction API
    # -------------------------
    def predict_for_base(self, base: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Synchronous prediction.
        """
        baseU = str(base).upper()
        if baseU not in self._bases:
            return None, {"base": baseU, "ok": False, "reason": "unknown_base"}

        with self._lock:
            x_t, m_t, mid_base, last_bid, last_ask, status = self._make_model_inputs(baseU)
            if not status.get("ok") or x_t is None or m_t is None:
                return None, status

        base_id_t = self._base_id_t[baseU]
        with torch.no_grad():
            z = self.model(x_t, m_t, base_id_t)  # (1,2)

        z0 = float(z[0, 0].detach().cpu().item())
        z1 = float(z[0, 1].detach().cpu().item())

        preds = self._decode_outputs(baseU, mid_base, z0, z1)

        # extend outputs for logging/debugging completeness
        used_len = int(status.get("segment_len_used", 0))
        pad_len = max(0, self.max_seq_len - used_len)
        preds.update(
            {
                "mid_base": float(mid_base) if (mid_base is not None and math.isfinite(float(mid_base))) else None,
                "last_best_bid": float(last_bid) if (last_bid is not None and math.isfinite(float(last_bid))) else None,
                "last_best_ask": float(last_ask) if (last_ask is not None and math.isfinite(float(last_ask))) else None,
                "used_len": int(used_len),
                "pad_len": int(pad_len),
            }
        )

        return preds, status

    async def predict_for_base_async(self, base: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Async-friendly prediction.
        """
        baseU = str(base).upper()
        if baseU not in self._bases:
            return None, {"base": baseU, "ok": False, "reason": "unknown_base"}

        with self._lock:
            x_t, m_t, mid_base, last_bid, last_ask, status = self._make_model_inputs(baseU)
            if not status.get("ok") or x_t is None or m_t is None:
                return None, status

        base_id_t = self._base_id_t[baseU]
        with torch.no_grad():
            z = self.model(x_t, m_t, base_id_t)

        z0 = float(z[0, 0].detach().cpu().item())
        z1 = float(z[0, 1].detach().cpu().item())

        preds = self._decode_outputs(baseU, mid_base, z0, z1)

        used_len = int(status.get("segment_len_used", 0))
        pad_len = max(0, self.max_seq_len - used_len)
        preds.update(
            {
                "mid_base": float(mid_base) if (mid_base is not None and math.isfinite(float(mid_base))) else None,
                "last_best_bid": float(last_bid) if (last_bid is not None and math.isfinite(float(last_bid))) else None,
                "last_best_ask": float(last_ask) if (last_ask is not None and math.isfinite(float(last_ask))) else None,
                "used_len": int(used_len),
                "pad_len": int(pad_len),
            }
        )

        return preds, status


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    eng = IVQuoteEngine(model_dir="runs_iv_delta", enable_async_predict=False)
    print("IVQuoteEngine initialized.")
