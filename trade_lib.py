# trade_lib.py (singleton/session-based, efficient reuse) — with live inventory callbacks & soft WS shutdown
from __future__ import annotations
import os, json, time, threading, ssl, uuid, ast, glob, re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Sequence, Union, Callable

import requests
from websocket import WebSocketApp
from dataclasses import dataclass
from collections import OrderedDict
import hashlib, json

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, RequestArgs, PostOrdersArgs
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.headers.headers import create_level_2_headers  # L2 header builder

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
WS_URL = "wss://ws-subscriptions-clob.polymarket.com"
USER_CH = "user"
_re_slug_ts = re.compile(r"(\d{10})$")
WINDOW_SEC = 900

# -----------------------------------------------------------------------------
# Quote engine + trading with inventory skew, ATOMIC per-slug updates
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class MMParams:
    min_spread_bps: float   # floor spread (underlying) in bps
    gamma: float            # risk aversion (Δ term)
    raw_skew_bps: float     # raw-mid skew per full-cap inventory (bps)

# ---- MM param constants (drives DEFAULT_MM_BY_BASE) ----
SPREAD_BPS = {  # per-base baseline spreads (bps)
    "BTC": 1.4,
    "ETH": 2.3,
    "SOL": 2.8,
    "XRP": 2.5,
}

REF_PRICE = {   # reference crypto prices
    "BTC": 85_000.0,
    "ETH": 2_700.0,
    "SOL": 130.0,
    "XRP": 1.95,
}

MM_MULT = {
    "min_spread": 0.5,
    "gamma": 6,
    "raw_skew": 0.1,
}

# One parameter set per base (no ensemble)
DEFAULT_MM_BY_BASE: Dict[str, MMParams] = {
    base: MMParams(
        # min_spread_bps = min_spread * SPREAD_BPS
        min_spread_bps=MM_MULT["min_spread"] * SPREAD_BPS[base],
        # gamma = gamma / SPREAD_BPS / REF_PRICE
        gamma=MM_MULT["gamma"] / SPREAD_BPS[base] / REF_PRICE[base],
        # raw_skew_bps = raw_skew * SPREAD_BPS
        raw_skew_bps=MM_MULT["raw_skew"] * SPREAD_BPS[base],
    )
    for base in ("BTC", "ETH", "SOL", "XRP")
}

BASE_CFG = {
    "BTC": {
        "quote_size": float(os.getenv("MM_BTC_QSIZE", "5")),
        "inv_cap":    float(os.getenv("MM_BTC_INVCAP", "20")),
        "vol_cap":    float(os.getenv("MM_BTC_VOLCAP", "20000")),
    },
    "ETH": {
        "quote_size": float(os.getenv("MM_BTC_QSIZE", "5")),
        "inv_cap":    float(os.getenv("MM_BTC_INVCAP", "15")),
        "vol_cap":    float(os.getenv("MM_BTC_VOLCAP", "2500")),
    },
    "SOL": {
        "quote_size": float(os.getenv("MM_BTC_QSIZE", "5")),
        "inv_cap":    float(os.getenv("MM_BTC_INVCAP", "15")),
        "vol_cap":    float(os.getenv("MM_BTC_VOLCAP", "1500")),
    },
    "XRP": {
        "quote_size": float(os.getenv("MM_BTC_QSIZE", "5")),
        "inv_cap":    float(os.getenv("MM_BTC_INVCAP", "15")),
        "vol_cap":    float(os.getenv("MM_BTC_VOLCAP", "1500")),
    },
}

# ---------------- file + util helpers ----------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _write_json(path: Path, obj: Any) -> None:
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        path.write_text(str(obj), encoding="utf-8")

def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + f"-{int((time.time()%1)*1e6):06d}"

def _minify_json(body: Any) -> Optional[str]:
    if body is None:
        return None
    if isinstance(body, str):
        return body
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False)

# ---------------- env / creds ----------------
def _load_env_keys() -> Tuple[str, Optional[str], int]:
    # Load key.env if present
    if os.path.exists("key.env"):
        with open("key.env", "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s: continue
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    key = os.environ.get("PRIVATE_KEY") or os.environ.get("PM_PRIVATE_KEY") or os.environ.get("PK")
    funder = os.environ.get("PM_FUNDER") or os.environ.get("FUNDER")
    if not key:
        raise RuntimeError("Missing PRIVATE_KEY/PM_PRIVATE_KEY/PK in env/key.env")
    sig_type = 1 if funder else 0  # 1 = proxy/Magic, 0 = EOA
    return key, funder, sig_type

# -----------------------------------------------------------------------------
# Token / Market indexing (temp/*.json)
# -----------------------------------------------------------------------------

def _start_from_slug(slug: str) -> Optional[int]:
    m = _re_slug_ts.search(slug or "")
    return int(m.group(1)) if m else None

def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _parse_market_json(obj: dict) -> Optional[Tuple[str, str, str, str, float, float, int]]:
    slug = obj.get("slug")
    if not slug:
        return None

    outs_raw = obj.get("outcomes")
    toks_raw = obj.get("clobTokenIds")

    outs = json.loads(outs_raw) if isinstance(outs_raw, str) else outs_raw
    toks = json.loads(toks_raw) if isinstance(toks_raw, str) else toks_raw

    if not (isinstance(outs, list) and isinstance(toks, list) and len(outs) == 2 and len(toks) == 2):
        return None

    lbl0, lbl1 = str(outs[0]).lower(), str(outs[1]).lower()
    t0, t1 = str(toks[0]), str(toks[1])

    # Map Up/Down -> YES/NO
    if ("up" in lbl0) and ("down" in lbl1):
        yes_id, no_id = t0, t1
    elif ("down" in lbl0) and ("up" in lbl1):
        yes_id, no_id = t1, t0
    else:
        yes_id, no_id = t0, t1

    tick = float(obj.get("orderPriceMinTickSize", 0.01) or 0.01)
    min_size = float(obj.get("orderMinSize", 1) or 1)
    baseU = slug.split("-")[0].upper()
    start_ts = _start_from_slug(slug)
    if start_ts is None:
        return None
    return baseU, slug, yes_id, no_id, tick, min_size, start_ts

class TokenIndex:
    """
    Indexes temp/*.json:
      by_slug:  slug -> (yes_id, no_id, tick, min_size, start_ts, baseU)
      by_base:  baseU -> {start_ts: slug}
      by_token: token_id -> (slug, "YES"/"NO")
    """
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = temp_dir
        self.by_slug: Dict[str, Tuple[str, str, float, float, int, str]] = {}
        self.by_base: Dict[str, Dict[int, str]] = {}
        self.by_token: Dict[str, Tuple[str, str]] = {}
        self.refresh()

    def refresh(self) -> None:
        by_slug: Dict[str, Tuple[str, str, float, float, int, str]] = {}
        by_base: Dict[str, Dict[int, str]] = {}
        by_token: Dict[str, Tuple[str, str]] = {}
        for path in glob.glob(os.path.join(self.temp_dir, "*.json")):
            obj = _read_json(path)
            if not isinstance(obj, dict):
                continue
            parsed = _parse_market_json(obj)
            if not parsed:
                continue
            baseU, slug, yes_id, no_id, tick, min_size, start_ts = parsed
            by_slug[slug] = (yes_id, no_id, tick, min_size, start_ts, baseU)
            by_base.setdefault(baseU, {})[start_ts] = slug
            by_token[yes_id] = (slug, "YES")
            by_token[no_id]  = (slug, "NO")

        self.by_slug = by_slug
        self.by_base = {b: dict(sorted(d.items())) for b, d in by_base.items()}
        self.by_token = by_token

    def active_slug_for_base(self, baseU: str, now_s: Optional[float] = None) -> Optional[str]:
        t = float(now_s if now_s is not None else time.time())
        canonical_start = int(t // WINDOW_SEC * WINDOW_SEC)
        m = self.by_base.get(baseU)
        if not m:
            return None
        if canonical_start in m:
            return m[canonical_start]
        starts = [st for st in m.keys() if st <= t]
        if starts:
            st = max(starts)
            if t < st + WINDOW_SEC:
                return m[st]
        return None

# ---------------- core session ----------------
class _WSHandle:
    def __init__(self, ws: Optional[WebSocketApp] = None, thread: Optional[threading.Thread] = None):
        self.ws = ws
        self.thread = thread

    def close(self):
        # Close the underlying WebSocket if present
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass

        # Join the thread if it’s still running
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

class TradeSession:
    """
    Holds a single ClobClient + HTTP Session + (optional) user-channel websocket.
    Reuse this across your app to avoid re-deriving creds & reconnecting.

    New in this version:
      - ws_positions (live balances from TRADE/MATCHED)
      - position listeners (callbacks) for push-based inventory updates
      - ws status listeners (open/error/close/retrying/dead)
      - soft shutdown on final WS failure (no hard exit)

    Added (simple inventory watchdog):
      - Polls Data-API /positions every 5s (timeout 2.5s)
      - Logs 8 token sizes of *current* BTC/ETH/SOL/XRP slugs to logs/inv_watch
      - Hard-exits if any token exceeds 3× inv_cap for its base, or after 3 consecutive fetch failures
    """
    def __init__(self, outdir: str = "logs"):
        self.outdir = outdir
        key, funder, sig_type = _load_env_keys()
        if funder:
            self.client = ClobClient(HOST, key=key, chain_id=CHAIN_ID, signature_type=sig_type, funder=funder)
        else:
            self.client = ClobClient(HOST, key=key, chain_id=CHAIN_ID, signature_type=sig_type)
        # Ensure L2 creds exist once (used by REST + WSS user auth)
        self.client.set_api_creds(self.client.create_or_derive_api_creds())
        # Owner id used to attribute fills on the user channel. Derived from the
        # configured L2 creds so inventory tracking follows whichever key is in use.
        self._my_owner = os.environ.get("PM_OWNER_ID") or self._extract_api_key()
        if not self._my_owner:
            raise RuntimeError(
                "Could not resolve owner id; inventory tracking would silently "
                "report zero. Set PM_OWNER_ID or check API creds."
            )
        # Pooled HTTP session
        self.http = requests.Session()

        # Active WS (optional)
        self._ws_handle: Optional[_WSHandle] = None
        self._lock = threading.Lock()

        # WS positions tracker: asset_id -> net size (BUY +, SELL -)
        self.ws_positions: Dict[str, float] = {}
        self.ws_positions_lock = threading.Lock()
        self.ws_volume_total: float = 0.0
        self.ws_volume_by_asset: Dict[str, float] = {}
        self.ws_volume_by_slug: Dict[str, float] = {}
        self._vol_lock = threading.Lock()
        self._ws_seen_msgs = OrderedDict()         # msg_key -> None
        self._ws_seen_lock = threading.Lock()
        self._ws_seen_limit = int(os.getenv("WS_SEEN_LIMIT", "20000"))  # tune as desired

        # Position listeners
        self._pos_listeners: List[Callable[[Dict[str, float], Dict[str, Any]], None]] = []
        self._pos_listeners_lock = threading.Lock()

        # WS status listeners
        self._ws_status_listeners: List[Callable[[str, Dict[str, Any]], None]] = []
        self._ws_status_lock = threading.Lock()

        # Termination event/signals for WS lifecycle
        self._ws_dead_event = threading.Event()

        # WS activity tracking: last time we saw a USER-channel message
        self._ws_last_msg_ts = 0.0          # monotonic seconds
        self._ws_last_msg_lock = threading.Lock()

        # WS health: last time we submitted a batch of orders
        self._ws_last_submit_ts = 0.0      # monotonic seconds

        # Ensure log dirs
        self.base = _ensure_dir(Path(self.outdir))

        # Defaults for "fatal" behavior are disabled; we do soft shutdown
        self._fatal_on_error_default = False
        self._fatal_on_close_default = False

        # Final WS status captured on last failure
        self._ws_final_info: Dict[str, Any] = {}

        # ---------------- Inventory watchdog state ----------------
        self._inv_watch_thread: Optional[threading.Thread] = None
        self._inv_watch_stop = threading.Event()
        self._inv_watch_fail_count = 0
        self._inv_dir = _ensure_dir(self.base / "inv_watch")

        # WS health watchdog state
        self._ws_health_thread: Optional[threading.Thread] = None
        self._ws_health_stop = threading.Event()

        # Deferred reconnect scheduling state
        self._ws_reconnect_scheduled = False

        # Optional tokens registry (if your file sets it later, that's fine)
        # Users can call set_tokens_provider(...) to inject one.
        self.tokens = getattr(self, "tokens", None)

        temp_dir = os.getenv("PM_TEMP_DIR", "temp")
        self.tokens = TokenIndex(temp_dir=temp_dir)

    # ---------- optional dependency injection ----------
    def set_tokens_provider(self, tokens_obj: Any) -> None:
        """If your tokens registry lives elsewhere, inject it here."""
        self.tokens = tokens_obj

    # ---------- shared helpers ----------
    def _extract_api_key(self) -> str:
        """
        L2 API key (the `owner` value stamped on our own user-channel trades).
        Handles creds being either an object or a dict, same as _ws_auth.
        """
        creds = self.client.creds
        v = (getattr(creds, "api_key", None)
             or getattr(creds, "apiKey", None)
             or (creds.get("api_key") if isinstance(creds, dict) else None))
        return str(v or "")

    def _ws_auth(self) -> Dict[str, str]:
        """
        WSS user auth: {"apiKey","secret","passphrase"}.
        """
        creds = self.client.creds
        api_key = getattr(creds, "api_key", None) or getattr(creds, "apiKey", None) or (creds.get("api_key") if isinstance(creds, dict) else None)
        api_secret = getattr(creds, "api_secret", None) or getattr(creds, "apiSecret", None) or (creds.get("api_secret") if isinstance(creds, dict) else None)
        api_passphrase = getattr(creds, "api_passphrase", None) or getattr(creds, "apiPassphrase", None) or (creds.get("api_passphrase") if isinstance(creds, dict) else None)
        if not (api_key and api_secret and api_passphrase):
            raise RuntimeError("Could not extract API credentials for WS auth")
        return {"apiKey": str(api_key), "secret": str(api_secret), "passphrase": str(api_passphrase)}

    def _signed_request_l2(self, method: str, request_path: str, body: Any = None) -> Dict[str, Any]:
        """
        Build L2 headers and send the exact JSON string as `data=...` so the signature matches.
        Accepts dict, list, str, or None for body.
        """
        payload_str = _minify_json(body)
        req_args = RequestArgs(method=method.upper(), request_path=request_path, body=payload_str)
        headers = create_level_2_headers(self.client.signer, self.client.creds, req_args)
        if payload_str is not None:
            headers["Content-Type"] = "application/json"
        url = f"{HOST}{request_path}"
        resp = self.http.request(method=method.upper(), url=url, headers=headers, data=payload_str)
        try:
            data = resp.json()
        except Exception:
            data = {"text": resp.text}
        if resp.status_code != 200:
            raise RuntimeError(f"L2 {method} {request_path} failed: status={resp.status_code}, body={data}")
        return data

    # ---------- listener management ----------
    def register_position_listener(self, cb: Callable[[Dict[str, float], Dict[str, Any]], None]) -> None:
        with self._pos_listeners_lock:
            self._pos_listeners.append(cb)

    def unregister_position_listener(self, cb: Callable[[Dict[str, float], Dict[str, Any]], None]) -> None:
        with self._pos_listeners_lock:
            self._pos_listeners = [f for f in self._pos_listeners if f is not cb]

    def _emit_position_update(self, delta_event: Dict[str, Any]) -> None:
        snap = self.get_ws_positions()
        with self._pos_listeners_lock:
            for cb in list(self._pos_listeners):
                try:
                    cb(snap, delta_event)
                except Exception as e:
                    # Never let listener exceptions kill the WS thread
                    _write_json(self.base / "ws_user" / f"{_ts()}-pos_listener_error.json",
                                {"error": str(e)})

    def register_ws_status_listener(self, cb: Callable[[str, Dict[str, Any]], None]) -> None:
        with self._ws_status_lock:
            self._ws_status_listeners.append(cb)

    def unregister_ws_status_listener(self, cb: Callable[[str, Dict[str, Any]], None]) -> None:
        with self._ws_status_lock:
            self._ws_status_listeners = [f for f in self._ws_status_listeners if f is not cb]

    def _emit_ws_status(self, status: str, info: Dict[str, Any]) -> None:
        with self._ws_status_lock:
            for cb in list(self._ws_status_listeners):
                try:
                    cb(status, info)
                except Exception:
                    pass

    def wait_ws_termination(self, timeout: Optional[float] = None) -> bool:
        """
        Block until the user WS errors/closes (returns True if it happened).
        """
        return self._ws_dead_event.wait(timeout=timeout)

    # ---------- WS reconnect helpers ----------
    def _next_session_start_s(self) -> int:
        """
        Return the UNIX second of the next 15m session start, using WINDOW_SEC grid.
        """
        now = int(time.time())
        return ((now // WINDOW_SEC) + 1) * WINDOW_SEC

    def _schedule_ws_reconnect_before_next_session(self) -> None:
        """
        After we've marked the WS as dead, schedule a *one-shot* reconnect
        ~5 seconds before the start of the next 15m session.

        - Does NOT clear the 'dead' flag immediately.
        - When the reconnect actually happens, subscribe_user_ws()
        will clear the dead flag and emit a new 'open' status.
        """
        if self._ws_reconnect_scheduled:
            return  # already scheduled

        self._ws_reconnect_scheduled = True

        # Start a background worker to sleep & reconnect.
        t = threading.Thread(
            target=self._ws_reconnect_worker,
            daemon=True,
        )
        t.start()


    def _ws_reconnect_worker(self) -> None:
        """
        Background worker that waits until ~5s before the next 15m session start
        and then calls subscribe_user_ws(), unless the WS is already alive again.
        """
        try:
            while True:
                # If someone already brought the WS back manually, stop.
                if self.is_user_ws_alive():
                    # clear the flag so future deaths can schedule again
                    self._ws_reconnect_scheduled = False
                    return

                next_start = self._next_session_start_s()
                target = next_start - 5  # 5 seconds before session start
                now = time.time()
                delay = target - now

                if delay <= 0:
                    # We're already at/after target; reconnect immediately.
                    break

                # Sleep in smallish chunks so we can notice a manual reconnect.
                time.sleep(min(delay, 5.0))

                if self.is_user_ws_alive():
                    self._ws_reconnect_scheduled = False
                    return

            # At this point we're at or past target time
            try:
                self.subscribe_user_ws(
                    condition_ids=None,
                    fatal_on_error=self._fatal_on_error_default,
                    fatal_on_close=self._fatal_on_close_default,
                )
            finally:
                self._ws_reconnect_scheduled = False
        except Exception:
            # In case of unexpected error, clear the flag so future schedules work
            self._ws_reconnect_scheduled = False

    # ---------- Inventory watchdog helpers ----------
    def _positions_user_addr(self) -> str:
        """
        Resolve the wallet to query for /positions.
        Prefer the funder/proxy (browser/Magic accounts keep balances there).
        Fallbacks try common attributes on the client.
        """
        addr = os.environ.get("PM_FUNDER") or os.environ.get("PM_ADDRESS")
        if not addr:
            for attr in ("funder", "signer", "owner", "address"):
                obj = getattr(self.client, attr, None)
                if hasattr(obj, "address"):
                    addr = obj.address
                    break
                if isinstance(obj, str) and obj.startswith("0x") and len(obj) == 42:
                    addr = obj
                    break
        if not addr:
            raise RuntimeError("inv_watch: cannot determine user address (set PM_FUNDER)")
        return str(addr)

    def _current_slug_token_map(self) -> Dict[str, Dict[str, str]]:
        """
        Returns: { slug: {"base": "BTC", "yes": "<tokenId>", "no": "<tokenId>"} }
        Uses TokenIndex.active_slug_for_base() to select the canonical 15m window.
        """
        out: Dict[str, Dict[str, str]] = {}
        tokreg = getattr(self, "tokens", None)
        if tokreg is None:
            return out

        for base in ("BTC", "ETH", "SOL", "XRP"):
            slug = tokreg.active_slug_for_base(base)
            if not slug:
                continue
            rec = tokreg.by_slug.get(slug)
            if not rec:
                continue
            yes_id, no_id, _, _, _, baseU = rec  # (yes, no, tick, min_size, start_ts, baseU)
            out[slug] = {"base": baseU, "yes": str(yes_id), "no": str(no_id)}
        return out

    def _inv_watch_log(self, kind: str, payload: Dict[str, Any]) -> None:
        """Write a JSON log file into logs/inv_watch."""
        try:
            _write_json(self._inv_dir / f"{_ts()}-{kind}.json", payload)
        except Exception:
            # Make best effort to not crash on logging
            pass

    def _inv_watch_loop(self) -> None:
        POLL_S = 5.0
        TIMEOUT_S = 2.5
        MAX_FAILS = 3

        try:
            user_addr = self._positions_user_addr()
        except Exception as e:
            self._inv_watch_log("init_error", {"error": str(e)})
            os._exit(3)  # cannot determine address -> hard fail

        while not self._inv_watch_stop.is_set():
            t_start = time.time()
            try:
                # in _inv_watch_loop(), right before you call _current_slug_token_map()
                try:
                    # pick up new 15m sessions and symbol rotations
                    self.tokens.refresh()
                except Exception as e:
                    self._inv_watch_log("tokens_refresh_error", {"error": str(e)})
                # Resolve current slugs/tokens (expecting 4 slugs × 2 tokens)
                slug_map = self._current_slug_token_map()
                if not slug_map:
                    # Not a network failure; just log and skip this tick without bumping fail count
                    self._inv_watch_log("warn_no_slugs", {"note": "no slugs resolved; tokens registry missing?"})
                    self._inv_watch_stop.wait(timeout=5.0)
                    continue

                # Fetch positions (Data-API)
                url = "https://data-api.polymarket.com/positions"
                params = {"user": user_addr, "limit": 500, "offset": 0, "sizeThreshold": 0}
                resp = self.http.get(url, params=params, timeout=TIMEOUT_S)
                if resp.status_code != 200:
                    raise RuntimeError(f"positions HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                if not isinstance(data, list):
                    raise RuntimeError("positions payload is not a list")

                # Map asset -> size
                size_by_asset: Dict[str, float] = {}
                for row in data:
                    a = row.get("asset"); s = row.get("size")
                    if a is None or s is None:
                        continue
                    try:
                        size_by_asset[str(a)] = float(s)
                    except Exception:
                        pass

                # Build debug details and threshold checks
                details: List[Dict[str, Any]] = []
                breach: Optional[Dict[str, Any]] = None

                for slug, meta in slug_map.items():
                    base = meta["base"]
                    inv_cap = float(BASE_CFG.get(base, {}).get("inv_cap", 0.0))
                    thr = (inv_cap * 4) if inv_cap > 0 else None

                    for side_key, label in (("yes","YES"), ("no","NO")):
                        tok = meta[side_key]
                        bal = float(size_by_asset.get(tok, 0.0))
                        details.append({
                            "slug": slug, "base": base, "side": label,
                            "token": tok, "size": bal, "threshold": thr
                        })
                        if thr is not None and abs(bal) > thr and breach is None:
                            breach = {
                                "reason": "threshold",
                                "slug": slug, "base": base, "side": label,
                                "token": tok, "size": bal, "threshold": thr
                            }

                # Snapshot log
                self._inv_watch_log("snapshot", {
                    "user": user_addr, "poll_s": POLL_S, "tokens": len(details), "details": details
                })

                if breach:
                    # Breach log then immediate hard exit
                    self._inv_watch_log("hard_fail_threshold", breach)
                    os._exit(2)

                # success -> reset fail counter
                self._inv_watch_fail_count = 0

            except requests.Timeout as e:
                self._inv_watch_fail_count += 1
                self._inv_watch_log("fetch_timeout", {
                    "error": str(e), "fails": self._inv_watch_fail_count,
                    "timeout_s": TIMEOUT_S, "max_fails": MAX_FAILS
                })
            except Exception as e:
                self._inv_watch_fail_count += 1
                self._inv_watch_log("fetch_error", {
                    "error": str(e), "fails": self._inv_watch_fail_count,
                    "timeout_s": TIMEOUT_S, "max_fails": MAX_FAILS
                })

            if self._inv_watch_fail_count >= MAX_FAILS:
                self._inv_watch_log("hard_fail_consecutive_fetch", {
                    "fails": self._inv_watch_fail_count, "max_fails": MAX_FAILS
                })
                os._exit(3)

            # Sleep to keep ~5s cadence (accounting for time spent)
            elapsed = time.time() - t_start
            remaining = max(0.0, 5.0 - elapsed)
            self._inv_watch_stop.wait(timeout=remaining)

    def _start_inventory_watchdog(self) -> None:
        if self._inv_watch_thread and self._inv_watch_thread.is_alive():
            return
        self._inv_watch_stop.clear()
        self._inv_watch_thread = threading.Thread(target=self._inv_watch_loop, daemon=True)
        self._inv_watch_thread.start()
    
    def _ws_health_loop(self) -> None:
        """
        Background watchdog:

        If the user WS is alive and more than TIMEOUT_S seconds have passed
        since the *last batch submit*, and no USER-WS message has arrived
        after that submit, we mark the WS as dead.
        """
        CHECK_INTERVAL = 0.1
        TIMEOUT_S = 5

        wdir = _ensure_dir(self.base / "ws_user")

        while not self._ws_health_stop.is_set():
            # If WS is already dead or never started, just sleep a bit.
            if not self.is_user_ws_alive():
                self._ws_health_stop.wait(timeout=CHECK_INTERVAL)
                continue

            now = time.monotonic()
            with self._ws_last_msg_lock:
                last_submit = getattr(self, "_ws_last_submit_ts", 0.0)
                last_msg    = self._ws_last_msg_ts

            # Only check if we’ve actually submitted a batch
            if last_submit > 0.0:
                delta = now - last_submit
                if delta > TIMEOUT_S and last_msg < last_submit:
                    info = {
                        "timeout_s": TIMEOUT_S,
                        "last_submit": last_submit,
                        "last_msg": last_msg,
                        "delta_since_submit": delta,
                    }
                    try:
                        _write_json(
                            wdir / f"{_ts()}-ws_health_timeout.json",
                            info,
                        )
                    except Exception:
                        pass

                    # Soft-flag WS as dead; does not kill the process.
                    self._soft_mark_ws_dead(
                        "no_user_ws_message_after_batch",
                        info,
                    )

                    # Prevent repeated triggers for the same submit
                    with self._ws_last_msg_lock:
                        self._ws_last_submit_ts = 0.0

            self._ws_health_stop.wait(timeout=CHECK_INTERVAL)

    def _start_ws_health_watchdog(self) -> None:
        if self._ws_health_thread and self._ws_health_thread.is_alive():
            return
        self._ws_health_stop.clear()
        self._ws_health_thread = threading.Thread(
            target=self._ws_health_loop,
            daemon=True,
        )
        self._ws_health_thread.start()

    # ---------- WS helpers ----------
    def _msg_fingerprint(self, obj: dict) -> str:
        for k in ("id", "event_id", "trade_id", "tx_hash", "match_id"):
            v = obj.get(k)
            if isinstance(v, (str, int)):
                return f"msg:{k}:{v}"

        # Optional: strip known-noisy fields if present
        noisy = {"server_time", "receive_ts", "seq", "sequence", "ingest_ts"}
        try:
            slim = {k: v for k, v in obj.items() if k not in noisy}
        except Exception:
            slim = obj

        try:
            blob = json.dumps(slim, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            return "msg:sha1:" + hashlib.sha1(blob).hexdigest()
        except Exception:
            return "msg:sha1:" + hashlib.sha1(repr(slim).encode("utf-8")).hexdigest()

    def _seen_msg_add(self, key: str) -> bool:
        """
        Returns True if we've already processed this message key.
        Maintains a small LRU to bound memory.
        """
        with self._ws_seen_lock:
            if key in self._ws_seen_msgs:
                return True
            self._ws_seen_msgs[key] = None
            # LRU trim
            if len(self._ws_seen_msgs) > self._ws_seen_limit:
                self._ws_seen_msgs.popitem(last=False)
            return False

    def _ws_update_positions(self, message_obj: Dict[str, Any]) -> None:
        """
        Update ws_positions on matched trades and notify listeners.

        We may be either the taker (top-level "owner") or one of the makers
        (inside "maker_orders"). Logic:

        * Must be a matched trade:
            - type == "TRADE" (or event_type == "trade")
            - status == "MATCHED"
        * If top-level owner == MY_OWNER -> use top-level fields:
            asset_id, side, size
        * Else scan maker_orders and for each entry with owner == MY_OWNER:
            asset_id, side, matched_amount

        Each matching role produces a position delta:
            delta = +amount if side == "BUY" else -amount
        We also accumulate absolute volume (shares).
        """
        if not isinstance(message_obj, dict):
            return

        typ = str(message_obj.get("type", "")).upper()
        evt = str(message_obj.get("event_type", "")).lower()
        status = str(message_obj.get("status", "")).upper()
        if not ((typ == "TRADE" or evt == "trade") and status == "MATCHED"):
            return
        
        # ---- De-dupe whole message (replays / retries) ----
        msg_key = self._msg_fingerprint(message_obj)
        if self._seen_msg_add(msg_key):
            return  # already applied this TRADE; skip

        MY_OWNER = self._my_owner
        processed_any = False

        # ------- Case A: top-level owner (usually taker) -------
        top_owner = str(message_obj.get("owner", ""))
        if top_owner == MY_OWNER:
            asset_id = message_obj.get("asset_id")
            side = str(message_obj.get("side", "")).upper()
            size = message_obj.get("size")
            if isinstance(asset_id, str) and asset_id and side in ("BUY", "SELL"):
                try:
                    amount = float(size)
                except Exception:
                    amount = None
                if amount is not None:
                    delta = amount if side == "BUY" else -amount
                    with self.ws_positions_lock:
                        new_bal = float(self.ws_positions.get(asset_id, 0.0) + delta)
                        self.ws_positions[asset_id] = new_bal
                    processed_any = True

                    # [VOL] accumulate absolute volume; also per-asset & per-slug
                    abs_delta = abs(delta)
                    slug = None
                    if self.tokens is not None:
                        try:
                            slug = self.tokens.by_token.get(asset_id, [None])[0]
                        except Exception:
                            slug = None
                    with self._vol_lock:
                        self.ws_volume_total += abs_delta
                        self.ws_volume_by_asset[asset_id] = self.ws_volume_by_asset.get(asset_id, 0.0) + abs_delta
                        if slug:
                            self.ws_volume_by_slug[slug] = self.ws_volume_by_slug.get(slug, 0.0) + abs_delta
                        vol_total_now = float(self.ws_volume_total)
                        vol_slug_now = float(self.ws_volume_by_slug.get(slug, 0.0)) if slug else None

                    self._emit_position_update({
                        "asset_id": asset_id,
                        "side": side,
                        "delta": delta,
                        "abs_delta": abs_delta,               # [VOL]
                        "new_balance": new_bal,
                        "role": "taker_or_top",
                        "slug": slug,                          # [VOL]
                        "volume_total": vol_total_now,         # [VOL]
                        "volume_slug": vol_slug_now,           # [VOL]
                        "raw": message_obj,
                    })

        # ------- Case B: we are one (or more) makers -------
        maker_orders = message_obj.get("maker_orders")
        if isinstance(maker_orders, list):
            for mo in maker_orders:
                if not isinstance(mo, dict):
                    continue
                if str(mo.get("owner", "")) != MY_OWNER:
                    continue  # not our maker fill
                asset_id = mo.get("asset_id")
                side = str(mo.get("side", "")).upper()
                matched_amount = mo.get("matched_amount")
                if not (isinstance(asset_id, str) and asset_id and side in ("BUY", "SELL")):
                    continue
                try:
                    amount = float(matched_amount)
                except Exception:
                    amount = None
                if amount is None:
                    continue

                delta = amount if side == "BUY" else -amount
                with self.ws_positions_lock:
                    new_bal = float(self.ws_positions.get(asset_id, 0.0) + delta)
                    self.ws_positions[asset_id] = new_bal
                processed_any = True

                # [VOL] accumulate absolute volume; also per-asset & per-slug
                abs_delta = abs(delta)
                slug = None
                if self.tokens is not None:
                    try:
                        slug = self.tokens.by_token.get(asset_id, [None])[0]
                    except Exception:
                        slug = None
                with self._vol_lock:
                    self.ws_volume_total += abs_delta
                    self.ws_volume_by_asset[asset_id] = self.ws_volume_by_asset.get(asset_id, 0.0) + abs_delta
                    if slug:
                        self.ws_volume_by_slug[slug] = self.ws_volume_by_slug.get(slug, 0.0) + abs_delta
                    vol_total_now = float(self.ws_volume_total)
                    vol_slug_now = float(self.ws_volume_by_slug.get(slug, 0.0)) if slug else None

                self._emit_position_update({
                    "asset_id": asset_id,
                    "side": side,
                    "delta": delta,
                    "abs_delta": abs_delta,                   # [VOL]
                    "new_balance": new_bal,
                    "role": "maker",
                    "slug": slug,                              # [VOL]
                    "volume_total": vol_total_now,             # [VOL]
                    "volume_slug": vol_slug_now,               # [VOL]
                    "raw": message_obj,
                })

        if not processed_any:
            return

        # Log every successfully processed matched trade message
        try:
            match_dir = _ensure_dir(self.base / "ws_user" / "matches")
            _write_json(match_dir / f"{_ts()}-match.json", message_obj)
        except Exception:
            pass

    def get_ws_positions(self) -> Dict[str, float]:
        """Return a shallow copy of current WS-tracked balances by asset_id."""
        with self.ws_positions_lock:
            return dict(self.ws_positions)

    # ---------- public ops ----------
    def subscribe_user_ws(
        self,
        condition_ids: Optional[list] = None,
        *,
        fatal_on_error: Optional[bool] = None,
        fatal_on_close: Optional[bool] = None,
    ) -> _WSHandle:
        """
        Connect once to the authenticated USER channel and keep it alive in a background thread.

        Retries: if the connection drops/fails, we retry up to MAX_RETRIES times with RETRY_DELAY seconds
        between attempts. After the final failed attempt we mark the channel 'dead' (soft shutdown).
        """
        fatal_on_error = self._fatal_on_error_default if fatal_on_error is None else bool(fatal_on_error)
        fatal_on_close = self._fatal_on_close_default if fatal_on_close is None else bool(fatal_on_close)

        MAX_RETRIES = 2                    # reconnect at most 2 times
        RETRY_DELAY = 1                    # 1 second between attempts
        PING_INTERVAL = 10                 # Polymarket recommends ~10s pings
        PING_TIMEOUT  = 5                  # allow 1–2 missed pongs before drop

        # Shared container for error/close reasons across attempts
        last_reason: Dict[str, Any] = {}

        with self._lock:
            # Close prior WS if any
            if self._ws_handle:
                try:
                    self._ws_handle.close()
                except Exception:
                    pass
                self._ws_handle = None

            # New lifecycle: clear 'dead' flag & previous failure info
            self._ws_dead_event.clear()
            self._ws_final_info.clear()
            self._ws_reconnect_scheduled = False

            wdir = _ensure_dir(self.base / "ws_user")
            sdir = _ensure_dir(wdir / "sent")
            rdir = _ensure_dir(wdir / "recv")

            submsg = {"markets": condition_ids or [], "type": USER_CH, "auth": self._ws_auth()}
            _write_json(sdir / f"{_ts()}-subscribe.json", submsg)

            url = f"{WS_URL}/ws/{USER_CH}"

            # ---------- WebSocket factory (per attempt) ----------
            def _make_ws(attempt_idx: int) -> WebSocketApp:
                # per-attempt info (optional, mainly for logging)
                info_last: Dict[str, Any] = {"attempt": attempt_idx}

                def on_open(ws):
                    try:
                        ws.send(json.dumps(submsg))
                    finally:
                        # treat a successful open as "activity"
                        with self._ws_last_msg_lock:
                            self._ws_last_msg_ts = time.monotonic()

                        info_last["open"] = True
                        _write_json(
                            rdir / f"{_ts()}-open.json",
                            {"attempt": attempt_idx, "sent_subscribe": True},
                        )
                        self._emit_ws_status(
                            "open",
                            {"attempt": attempt_idx, "note": "user ws opened"},
                        )
                        # Start inventory watchdog at first successful open
                        try:
                            self._start_inventory_watchdog()
                        except Exception:
                            pass
                        # Start WS health watchdog
                        try:
                            self._start_ws_health_watchdog()
                        except Exception:
                            pass

                def on_message(ws, message):
                    # bump activity marker first
                    with self._ws_last_msg_lock:
                        self._ws_last_msg_ts = time.monotonic()

                    try:
                        obj = json.loads(message)
                    except Exception:
                        obj = {"raw": message}
                    # _write_json(rdir / f"{_ts()}-msg.json",
                    #             {"attempt": attempt_idx, "data": obj})
                    if isinstance(obj, dict):
                        self._ws_update_positions(obj)

                def on_error(ws, error):
                    info = {"attempt": attempt_idx, "error": str(error)}
                    info_last.update(info)
                    _write_json(rdir / f"{_ts()}-error.json", info)
                    self._emit_ws_status("error", info)
                    # record last reason for this lifecycle
                    try:
                        last_reason.update(info)
                    except Exception:
                        pass
                    try:
                        ws.close()
                    except Exception:
                        pass

                def on_close(ws, code, msg):
                    info = {"attempt": attempt_idx, "code": code, "msg": msg}
                    info_last.update(info)
                    _write_json(rdir / f"{_ts()}-close.json", info)
                    self._emit_ws_status("close", info)
                    # record last reason for this lifecycle
                    try:
                        last_reason.update(info)
                    except Exception:
                        pass

                return WebSocketApp(
                    url,
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                )

            # ---------- Runner thread with bounded retries ----------
            def _runner():
                attempts = 0
                # reset reason at start of lifecycle
                last_reason.clear()
                while attempts <= MAX_RETRIES:
                    ws = _make_ws(attempt_idx=attempts)

                    # Attach the *real* ws to the existing handle under the same lock.
                    with self._lock:
                        if self._ws_handle is not None:
                            self._ws_handle.ws = ws

                    ws.run_forever(
                        sslopt={"cert_reqs": ssl.CERT_REQUIRED},
                        ping_interval=PING_INTERVAL,
                        ping_timeout=PING_TIMEOUT,
                    )

                    if attempts < MAX_RETRIES:
                        _write_json(
                            rdir / f"{_ts()}-retry.json",
                            {"attempt": attempts, "next_attempt_in_sec": RETRY_DELAY},
                        )
                        self._emit_ws_status(
                            "retrying",
                            {"attempt": attempts, "delay": RETRY_DELAY},
                        )
                        time.sleep(RETRY_DELAY)
                        attempts += 1
                        continue

                    # ---- Out of retries: SOFT SHUTDOWN (no process exit) ----
                    self._ws_final_info = dict(last_reason)
                    dead_info = {"attempts": attempts, **self._ws_final_info}
                    _write_json(rdir / f"{_ts()}-dead.json", dead_info)
                    self._emit_ws_status("dead", dead_info)

                    # *** important line ***
                    self._ws_dead_event.set()

                    # Schedule reconnect before next session
                    try:
                        self._schedule_ws_reconnect_before_next_session()
                    except Exception:
                        pass

                    return

            # Create the handle *before* starting the thread, with ws=None for now.
            t = threading.Thread(target=_runner, daemon=True)
            self._ws_handle = _WSHandle(ws=None, thread=t)
            t.start()
            return self._ws_handle

    # ---- Single submit (GTC) ----
    def submit_order(self, token_id: str, side: str, price: float, size: float,
                     wait_poll: bool = False, poll_secs: float = 0.25, timeout: float = 10.0) -> Optional[str]:
        """
        Place a single GTC order; optionally poll GET /data/order/<id> until visible.
        Returns the exact `orderID` if present.
        """
        odir = _ensure_dir(self.base / "submit")
        sid = uuid.uuid4().hex[:10]

        args = OrderArgs(
            token_id=str(token_id),
            side=(BUY if str(side).upper() == "BUY" else SELL),
            price=float(price),
            size=float(size),
        )

        # request log: log only the input you provided
        req_dump = {"token_id": str(token_id), "side": str(side).upper(), "price": float(price), "size": float(size), "order_type": "GTC"}
        # _write_json(odir / f"{_ts()}-{sid}-request.json", req_dump)

        signed = self.client.create_order(args)
        # _write_json(odir / f"{_ts()}-{sid}-signed.json", {"signed": True})

        resp = self.client.post_order(signed, OrderType.GTC)
        # _write_json(odir / f"{_ts()}-{sid}-response.json", resp if isinstance(resp, (dict, list, str)) else {"resp": str(resp)})

        # Exact extraction: orderID (primary), fallback orderId
        order_id = None
        if isinstance(resp, dict):
            if isinstance(resp.get("orderID"), str) and resp["orderID"]:
                order_id = resp["orderID"]
            elif isinstance(resp.get("orderId"), str) and resp["orderId"]:
                order_id = resp["orderId"]

        if wait_poll and order_id:
            t0 = time.time()
            last_err = None
            while time.time() - t0 < timeout:
                try:
                    od = self.client.get_order(order_id)  # GET /data/order/<id>
                    # _write_json(odir / f"{_ts()}-{sid}-polled.json", od if isinstance(od, (dict, list, str)) else {"resp": str(od)})
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(poll_secs)
            if last_err:
                _write_json(odir / f"{_ts()}-{sid}-poll_error.json", {"error": str(last_err)})

        return order_id

    # ---- Batch submit (GTC) ----
    def submit_orders_batch(self, orders: Sequence[Union[Tuple[str, str, float, float], Dict[str, Any]]]) -> List[str]:
        """
        Submit multiple GTC orders in one call.
        Accepts:
          - list of tuples: (token_id, side, price, size)
          - or list of dicts: {"token_id":..., "side":..., "price":..., "size":...}
        Returns list of exact orderIDs extracted from response.
        """

        bdir = _ensure_dir(self.base / "submit_batch")
        sid = uuid.uuid4().hex[:10]

        post_args: List[PostOrdersArgs] = []
        req_dump: List[Dict[str, Any]] = []
        for item in orders:
            if isinstance(item, dict):
                token_id = item["token_id"]
                side = item["side"]
                price = float(item["price"])
                size = float(item["size"])
            else:
                token_id, side, price, size = item  # tuple-like
                price = float(price); size = float(size)
            req_dump.append({"token_id": str(token_id), "side": str(side).upper(), "price": price, "size": size, "order_type": "GTC"})
            order = self.client.create_order(OrderArgs(
                token_id=str(token_id),
                side=(BUY if str(side).upper() == "BUY" else SELL),
                price=price,
                size=size,
            ))
            post_args.append(PostOrdersArgs(order=order, orderType=OrderType.GTC))

        # _write_json(bdir / f"{_ts()}-{sid}-request.json", req_dump)

        resp = self.client.post_orders(post_args)
        # _write_json(bdir / f"{_ts()}-{sid}-response_raw.json", resp if isinstance(resp, (dict, list, str)) else {"resp": str(resp)})

        order_ids: List[str] = []

        def _maybe_add(d: Dict[str, Any]):
            oid = d.get("orderID") or d.get("orderId")
            if isinstance(oid, str) and oid:
                order_ids.append(oid)

        if isinstance(resp, list):
            for elem in resp:
                if isinstance(elem, dict):
                    _maybe_add(elem)
        elif isinstance(resp, dict):
            # Some client versions wrap under "resp" as a string; try JSON first, then literal_eval
            r = resp.get("resp")
            if isinstance(r, str):
                parsed = None
                try:
                    parsed = json.loads(r)
                except Exception:
                    try:
                        parsed = ast.literal_eval(r)
                    except Exception:
                        parsed = None
                if isinstance(parsed, list):
                    for elem in parsed:
                        if isinstance(elem, dict):
                            _maybe_add(elem)
            else:
                _maybe_add(resp)

        # _write_json(bdir / f"{_ts()}-{sid}-parsed_ids.json", {"order_ids": order_ids})

        # ---- record submit time only if we actually got at least one orderID ----
        if order_ids:
            try:
                with self._ws_last_msg_lock:
                    self._ws_last_submit_ts = time.monotonic()
            except Exception:
                pass

        return order_ids

    # ---------- Cancels ----------
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        DELETE /order with body {"orderID": "..."} (L2 header).
        """
        cdir = _ensure_dir(self.base / "cancel_single")
        rid = uuid.uuid4().hex[:10]
        body = {"orderID": str(order_id)}
        # _write_json(cdir / f"{_ts()}-{rid}-request.json", body)
        resp = self._signed_request_l2("DELETE", "/order", body)
        # _write_json(cdir / f"{_ts()}-{rid}-response.json", resp if isinstance(resp, (dict, list, str)) else {"resp": str(resp)})
        return resp

    def cancel_orders(self, order_ids: Sequence[str]) -> Dict[str, Any]:
        """
        DELETE /orders with body being a JSON array of orderIDs (L2 header).
        """
        cdir = _ensure_dir(self.base / "cancel_orders")
        rid = uuid.uuid4().hex[:10]
        ids = [str(x) for x in order_ids]
        # _write_json(cdir / f"{_ts()}-{rid}-request.json", {"ids": ids})
        # Body must be the array itself, not an object:
        resp = self._signed_request_l2("DELETE", "/orders", ids)
        # _write_json(cdir / f"{_ts()}-{rid}-response.json", resp if isinstance(resp, (dict, list, str)) else {"resp": str(resp)})
        return resp

    def cancel_all_orders(self) -> Dict[str, Any]:
        """
        DELETE /cancel-all (no body) with L2 header.
        """
        cdir = _ensure_dir(self.base / "cancel_all")
        rid = uuid.uuid4().hex[:10]
        # _write_json(cdir / f"{_ts()}-{rid}-request.json", {"note": "no body"})
        resp = self._signed_request_l2("DELETE", "/cancel-all", None)
        # _write_json(cdir / f"{_ts()}-{rid}-response.json", resp if isinstance(resp, (dict, list, str)) else {"resp": str(resp)})
        return resp

    def cancel_market_orders(self, market: Optional[str] = None, asset_id: Optional[str] = None) -> Dict[str, Any]:
        """
        DELETE /cancel-market-orders (L2). Supports filtering by market (condition id) and/or asset_id (token id).
        """
        cdir = _ensure_dir(self.base / "cancel_market")
        rid = uuid.uuid4().hex[:10]
        body: Dict[str, Any] = {}
        if market: body["market"] = str(market)
        if asset_id: body["asset_id"] = str(asset_id)
        # _write_json(cdir / f"{_ts()}-{rid}-request.json", body or {"note": "empty body"})
        resp = self._signed_request_l2("DELETE", "/cancel-market-orders", body if body else {})
        # _write_json(cdir / f"{_ts()}-{rid}-response.json", resp if isinstance(resp, (dict, list, str)) else {"resp": str(resp)})
        return resp

    def close(self):
        # stop inventory watchdog
        try:
            if self._inv_watch_thread and self._inv_watch_thread.is_alive():
                self._inv_watch_stop.set()
                self._inv_watch_thread.join(timeout=1.0)
        except Exception:
            pass

        # stop WS health watchdog
        try:
            if self._ws_health_thread and self._ws_health_thread.is_alive():
                self._ws_health_stop.set()
                self._ws_health_thread.join(timeout=1.0)
        except Exception:
            pass

        # close network resources
        with self._lock:
            if self._ws_handle:
                try:
                    self._ws_handle.close()
                except Exception:
                    pass
                self._ws_handle = None
            try:
                self.http.close()
            except Exception:
                pass

    # ---------- WS health helpers ----------
    def is_user_ws_alive(self) -> bool:
        with self._lock:
            t_alive = bool(self._ws_handle and self._ws_handle.thread.is_alive())
        return t_alive and not self._ws_dead_event.is_set()

    def last_user_ws_failure(self) -> Dict[str, Any]:
        return dict(self._ws_final_info)

    def _soft_mark_ws_dead(self, reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark the user WS as 'dead' without touching the WS thread.
        This is a soft flag: sets _ws_dead_event, updates _ws_final_info, and emits a 'dead' status.
        Also schedules a deferred reconnect before the next 15m session.
        """
        info: Dict[str, Any] = {"reason": reason}
        if extra:
            info.update(extra)
        self._ws_final_info.update(info)
        self._ws_dead_event.set()
        self._emit_ws_status("dead", dict(info))

        # IMPORTANT: make sure the current WS is really closed
        try:
            with self._lock:
                if getattr(self, "_ws_handle", None) is not None:
                    try:
                        self._ws_handle.close()
                    except Exception:
                        pass
                    self._ws_handle = None
        except Exception:
            pass

        # Reset WS-tracked state (so the next lifecycle starts clean)
        try:
            with self.ws_positions_lock:
                self.ws_positions.clear()
            with self._vol_lock:
                self.ws_volume_total = 0.0
                self.ws_volume_by_asset.clear()
                self.ws_volume_by_slug.clear()
        except Exception:
            pass

        # >>> NEW: clear last-submit so the health watchdog can't retrip immediately
        try:
            with self._ws_last_msg_lock:
                self._ws_last_submit_ts = 0.0
        except Exception:
            pass

        # Try to reconnect in time for the next session
        try:
            self._schedule_ws_reconnect_before_next_session()
        except Exception:
            pass

# ---------------- singleton wiring + convenience wrappers ----------------
_SESSION: Optional[TradeSession] = None
_SESSION_LOCK = threading.Lock()

def init_session(outdir: str = "logs") -> TradeSession:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            _SESSION = TradeSession(outdir=outdir)
        return _SESSION

def _get_session() -> TradeSession:
    if _SESSION is None:
        return init_session(outdir="logs")
    return _SESSION

# Convenience wrappers
def subscribe_user_ws(outdir: str = "logs", ws_url: str = WS_URL, condition_ids: Optional[list] = None,
                      *, fatal_on_error: Optional[bool] = None, fatal_on_close: Optional[bool] = None) -> _WSHandle:
    s = init_session(outdir=outdir)
    # ws_url currently not overridable (single endpoint); kept for backward compat
    return s.subscribe_user_ws(condition_ids=condition_ids, fatal_on_error=fatal_on_error, fatal_on_close=fatal_on_close)

def register_position_listener(cb: Callable[[Dict[str, float], Dict[str, Any]], None]) -> None:
    _get_session().register_position_listener(cb)

def unregister_position_listener(cb: Callable[[Dict[str, float], Dict[str, Any]], None]) -> None:
    _get_session().unregister_position_listener(cb)

def register_ws_status_listener(cb: Callable[[str, Dict[str, Any]], None]) -> None:
    _get_session().register_ws_status_listener(cb)

def unregister_ws_status_listener(cb: Callable[[str, Dict[str, Any]], None]) -> None:
    _get_session().unregister_ws_status_listener(cb)

def wait_ws_termination(timeout: Optional[float] = None) -> bool:
    return _get_session().wait_ws_termination(timeout)

def is_user_ws_alive() -> bool:
    return _get_session().is_user_ws_alive()

def last_user_ws_failure() -> Dict[str, Any]:
    return _get_session().last_user_ws_failure()

def get_ws_positions() -> Dict[str, float]:
    return _get_session().get_ws_positions()

def submit_order(token_id: str, side: str, price: float, size: float,
                 outdir: str = "logs", wait_poll: bool = False, poll_secs: float = 0.25, timeout_secs: float = 10.0) -> Optional[str]:
    s = init_session(outdir=outdir)
    return s.submit_order(token_id, side, price, size, wait_poll=wait_poll, poll_secs=poll_secs, timeout=timeout_secs)

def submit_orders_batch(order_list: Sequence[Union[Tuple[str, str, float, float], Dict[str, Any]]], outdir: str = "logs") -> List[str]:
    s = init_session(outdir=outdir)
    return s.submit_orders_batch(order_list)

def cancel_order(order_id: str, outdir: str = "logs") -> Dict[str, Any]:
    s = init_session(outdir=outdir)
    return s.cancel_order(order_id)

def cancel_orders(order_ids: Sequence[str], outdir: str = "logs") -> Dict[str, Any]:
    s = init_session(outdir=outdir)
    return s.cancel_orders(order_ids)

def cancel_all_orders(outdir: str = "logs") -> Dict[str, Any]:
    s = init_session(outdir=outdir)
    return s.cancel_all_orders()

def cancel_market_orders(market: Optional[str] = None, asset_id: Optional[str] = None, outdir: str = "logs") -> Dict[str, Any]:
    s = init_session(outdir=outdir)
    return s.cancel_market_orders(market=market, asset_id=asset_id)
