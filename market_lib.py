# market_lib.py
# Streams + dashboard + shared state for Chainlink, Coinbase, and Polymarket (CLOB)
# - Chainlink RTDS (topic=crypto_prices_chainlink): keep latest tick per base and 15-minute close
# - Coinbase public WS: keep latest ticker price per product
# - Polymarket slug refresher: save ./temp/{slug}.json for updown-15m markets
# - Polymarket market WS: subscribe to END=B-900 (switch at B-5s), track top-3 bids/asks per active slug
# - Dashboard: event-driven terminal panel that refreshes on *every* update
# - Model snapshot: a live area showing latest model outputs (iv, df) per base
#
# NOTE: Use your main script (e.g., live_prediction.py) to:
#   - set up logging to file (keep console free for the dashboard)
#   - call hub.set_model_prediction(base, iv, df, exch_sec) whenever you have a new prediction

from __future__ import annotations
import os
import json
import math
import asyncio
import logging
import glob
import heapq
import time
import numpy as np
import datetime as _dt
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Sequence, TYPE_CHECKING, Callable
from collections import defaultdict, deque

if TYPE_CHECKING:
    # Imported only for type checking; no runtime dependency or cycles
    from fast_oracle import FastChainlinkOracle

import aiohttp
import websockets
import pandas as pd
from live_lib import jlog

from scipy.stats import t as _student_t_dist


def _student_t_cdf(z: float, df: float) -> float:
    # SciPy's C-implemented CDF: fast and accurate
    return float(_student_t_dist.cdf(z, df))


def _digital_prob_student_t(
    spot: float,
    prev_close: float,
    tau: float,
    iv_900: float,
    df: float,
) -> float:
    """
    Digital Up/Down probability under Student-t(df) return model.

    Assumptions:
      - iv_900 is the scale parameter for a 900s horizon.
      - tau is time-to-expiry in seconds.
      - Log-return from now to expiry ~ StudentT(0, scale=sigma_tau),
        with sigma_tau = iv_900 * sqrt(tau / 900).
      - Event is S_T >= prev_close.
    """
    tau = float(tau)
    if tau <= 0 or iv_900 <= 0 or spot <= 0 or prev_close <= 0:
        return 0.5

    sigma_tau = iv_900 * math.sqrt(max(tau, 1.0) / 900.0)
    r_star = math.log(prev_close / spot)
    z = r_star / sigma_tau

    return 1.0 - _student_t_cdf(z, df)


def implied_iv_from_price_student_t(
    target_p: float,
    *,
    spot: float,
    prev_close: float,
    tau: float,
    df: float,
    iv_lo: float = 1e-5,
    iv_hi: float = 5e-3,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Optional[float]:
    """
    Invert the Student-t digital price to get a per-900s implied IV.
    Solves for iv_900 in:
        _digital_prob_student_t(spot, prev_close, tau, iv_900, df) = target_p
    """
    if not (0.0 < target_p < 1.0):
        return None
    if tau <= 0 or spot <= 0 or prev_close <= 0:
        return None

    p_lo = _digital_prob_student_t(spot, prev_close, tau, iv_lo, df)
    p_hi = _digital_prob_student_t(spot, prev_close, tau, iv_hi, df)

    if not (math.isfinite(p_lo) and math.isfinite(p_hi)):
        return None
    if abs(p_hi - p_lo) < 1e-9:
        return None

    p_min, p_max = (p_lo, p_hi) if p_lo <= p_hi else (p_hi, p_lo)
    if not (p_min - 1e-12 <= target_p <= p_max + 1e-12):
        return None

    lo, hi = iv_lo, iv_hi
    increasing = p_hi > p_lo

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = _digital_prob_student_t(spot, prev_close, tau, mid, df)
        if not math.isfinite(p_mid):
            break

        if abs(p_mid - target_p) < tol:
            return mid

        if increasing:
            if p_mid < target_p:
                lo = mid
            else:
                hi = mid
        else:
            if p_mid > target_p:
                lo = mid
            else:
                hi = mid

    return 0.5 * (lo + hi)

# ----------------------------- Paths & constants -----------------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
PM_DIR = os.path.join(DATA_DIR, "polymarket")
PM_RES_DIR = os.path.join(PM_DIR, "resolution")
PM_RAW_CHAINLINK = os.path.join(PM_DIR, "crypto_prices-chainlink.jsonl")
PM_RAW_OTHER = os.path.join(PM_DIR, "rtds-other.jsonl")
CB_DIR = os.path.join(DATA_DIR, "coinbase")
MARKET_DIR = os.path.join(DATA_DIR, "market")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

for d in (DATA_DIR, PM_DIR, PM_RES_DIR, CB_DIR, MARKET_DIR, TEMP_DIR):
    os.makedirs(d, exist_ok=True)

RTDS_URL = "wss://ws-live-data.polymarket.com"
COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"
WINDOW_SEC = 900         # 15 minutes
SWITCH_LEAD_MS = 5_000   # switch at B - 5s
DEFAULT_PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
BASES = ("BTC", "ETH", "SOL", "XRP")
CB_ROTATE_MAX_BYTES = int(os.getenv("CB_ROTATE_MAX_BYTES", str(64 * 1024 * 1024)))  # rotate at 64MB
CB_FLUSH_EVERY = int(os.getenv("CB_FLUSH_EVERY", "200"))                     # flush every N lines
CB_L2_FEATURE_INTERVAL_S = float(os.environ.get("CB_L2_FEATURE_INTERVAL_S", "0.10"))

# ============================= Shared State (Hub) =============================

def _base_from_product(pid: str) -> str:
    # "BTC-USD" -> "BTC"
    try:
        return str(pid).split("-", 1)[0].upper()
    except Exception:
        return str(pid).upper()

def _as_epoch_s(ts: str) -> float:
    # Coinbase WS timestamps are RFC3339 with nanos, e.g. "2025-11-13T02:47:20.533308839Z"
    # We only need seconds resolution for rolling windows.
    if not ts:
        return 0.0
    ts = ts.rstrip("Z")
    # Trim any sub-second part longer than microseconds for Python parsing
    if "." in ts:
        head, frac = ts.split(".", 1)
        frac = "".join(ch for ch in frac if ch.isdigit())
        frac = (frac + "000000")[:6]  # microseconds
        ts = f"{head}.{frac}"
    try:
        dt = _dt.datetime.fromisoformat(ts).replace(tzinfo=_dt.timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0

@dataclass
class _FlowWindow:
    buy: float = 0.0
    sell: float = 0.0
    net: float = 0.0
    asof: float = 0.0  # epoch seconds

class StateHub:
    """
    Shared in-memory state across services.
    Existing fields are kept; new Coinbase-specific analytics are added:
      - coinbase_microprice_latest[BASE], coinbase_microprice_prev[BASE]
      - coinbase_flow_1s[BASE], coinbase_flow_3s[BASE]  (buy/sell/net in the last 1s/3s)

    All mutations should occur on the same asyncio loop (no external locking needed).
    """

    # --------- core (existing) ---------
    def __init__(self) -> None:
        # Chainlink
        self.chainlink_latest: Dict[str, float] = {}  # "BTC" -> last px
        self.chainlink_15m_close: Dict[str, Tuple[int, float]] = {}  # "BTC" -> (end_ts, px)

        # >>> NEW: Coinbase prev close per base (prev 15m close from Coinbase) <<<
        self.coinbase_prev_close: Dict[str, float] = {}
        # and short window stats around that close (from live_prediction)
        self.coinbase_prev_high: Dict[str, float] = {}
        self.coinbase_prev_low: Dict[str, float] = {}

        # Polymarket order books (per slug)
        self.clob_current_end: Optional[int] = None

        # Model predictions (per base)
        self.model_preds: Dict[str, Dict[str, Any]] = {}

        # Polymarket IV state (per base)
        self.pm_iv_implied_900: Dict[str, float] = defaultdict(float)   # smoothed IV (posterior)
        self.pm_iv_mult: Dict[str, float] = defaultdict(lambda: 1.0)    # smoothed multiplicity
        self._pm_iv_log_mult: Dict[str, float] = defaultdict(float)     # state mean in log-space
        self._pm_iv_log_mult_var: Dict[str, float] = defaultdict(lambda: 0.1)  # state variance P_t

        # Polymarket orderbook state (per base) for IV calibrator + trading
        # base -> {bids, asks, spread, imb1, imb3, imb5, trade_imb_3s, ts_server_ms}
        self.pm_base_state: Dict[str, Dict[str, Any]] = {}

        # base -> deque[(ts_ms, buy, sell)] for 3s trade imbalance
        self._pm_trades: Dict[str, deque] = defaultdict(lambda: deque())

        # --- Coinbase microprice + flows (per base) ---
        self.coinbase_microprice_latest: Dict[str, float] = {}
        self.coinbase_microprice_prev: Dict[str, float] = {}

        # Rolling trade flow buffers (per base) — store raw trades for <=3s
        # deque entries: (ts_seconds, buy_size, sell_size)
        self._cb_trades: Dict[str, deque] = defaultdict(lambda: deque())

        # Exposed aggregates per base
        self.coinbase_flow_1s: Dict[str, _FlowWindow] = defaultdict(_FlowWindow)
        self.coinbase_flow_3s: Dict[str, _FlowWindow] = defaultdict(_FlowWindow)
        self.coinbase_flow_5s: Dict[str, _FlowWindow] = defaultdict(_FlowWindow)

        # --- Coinbase L2 + EWMA analytics (per base) ---
        # Latest Coinbase *server* timestamp (ms)
        self.coinbase_ts_server_ms: Dict[str, int] = {}

        # Microprice skew in ±1/2 bps bands
        self.coinbase_mp_skew_bps1: Dict[str, float] = defaultdict(float)
        self.coinbase_mp_skew_bps2: Dict[str, float] = defaultdict(float)
        self.coinbase_mp_skew_bps5: Dict[str, float] = defaultdict(float)
        self.coinbase_mp_skew_bps10: Dict[str, float] = defaultdict(float)

        # Per-second EWMA sigma (no /dt in var increment), 30s half-life
        self.coinbase_sigma_ewma: Dict[str, float] = defaultdict(float)
        self.coinbase_jump_flag: Dict[str, int] = defaultdict(int)  # 0/1

        # Private EWMA state per base
        self._cb_ewma_state: Dict[str, Dict[str, float]] = defaultdict(dict)

        # --- NEW: rich L2 *snapshot* features (per base) ---

        # 1.1 Top-of-book stats
        self.coinbase_tob_bid_px: Dict[str, float] = defaultdict(float)
        self.coinbase_tob_ask_px: Dict[str, float] = defaultdict(float)
        self.coinbase_tob_bid_qty: Dict[str, float] = defaultdict(float)
        self.coinbase_tob_ask_qty: Dict[str, float] = defaultdict(float)
        self.coinbase_spread_abs: Dict[str, float] = defaultdict(float)
        self.coinbase_spread_bp: Dict[str, float] = defaultdict(float)

        # 1.2 Depth in ±1/2/5 bps bands around microprice
        self.coinbase_depth_bid_1bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_ask_1bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_tot_1bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_imb_1bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_lr_1bp: Dict[str, float] = defaultdict(float)

        self.coinbase_depth_bid_2bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_ask_2bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_tot_2bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_imb_2bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_lr_2bp: Dict[str, float] = defaultdict(float)

        self.coinbase_depth_bid_5bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_ask_5bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_tot_5bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_imb_5bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_lr_5bp: Dict[str, float] = defaultdict(float)

        self.coinbase_depth_bid_10bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_ask_10bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_tot_10bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_imb_10bp: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_lr_10bp: Dict[str, float] = defaultdict(float)

        # 1.3 Near vs far depth ratios
        self.coinbase_depth_near_far_ratio_bid: Dict[str, float] = defaultdict(float)
        self.coinbase_depth_near_far_ratio_ask: Dict[str, float] = defaultdict(float)

        # 2. Book slope / convexity (top K levels)
        self.coinbase_book_avg_dist_bid_bp: Dict[str, float] = defaultdict(float)
        self.coinbase_book_avg_dist_ask_bp: Dict[str, float] = defaultdict(float)
        self.coinbase_book_slope_bid: Dict[str, float] = defaultdict(float)
        self.coinbase_book_slope_ask: Dict[str, float] = defaultdict(float)
        self.coinbase_book_convexity: Dict[str, float] = defaultdict(float)

        # 3. Liquidity walls
        self.coinbase_wall_bid_size: Dict[str, float] = defaultdict(float)
        self.coinbase_wall_bid_dist_bp: Dict[str, float] = defaultdict(float)
        self.coinbase_wall_ask_size: Dict[str, float] = defaultdict(float)
        self.coinbase_wall_ask_dist_bp: Dict[str, float] = defaultdict(float)
        self.coinbase_wall_imbalance: Dict[str, float] = defaultdict(float)

        # 4. L2 dynamics: depth & quote behaviour
        self.coinbase_depth_imb_1bp_diff: Dict[str, float] = defaultdict(float)
        self.coinbase_net_add_bid_1bp_1s: Dict[str, float] = defaultdict(float)
        self.coinbase_net_add_ask_1bp_1s: Dict[str, float] = defaultdict(float)

        self.coinbase_n_bid_improve_1s: Dict[str, int] = defaultdict(int)
        self.coinbase_n_bid_worsen_1s: Dict[str, int] = defaultdict(int)
        self.coinbase_n_ask_improve_1s: Dict[str, int] = defaultdict(int)
        self.coinbase_n_ask_worsen_1s: Dict[str, int] = defaultdict(int)
        self.coinbase_n_spread_tighten_1s: Dict[str, int] = defaultdict(int)
        self.coinbase_n_spread_widen_1s: Dict[str, int] = defaultdict(int)

        # 5. Trade microstructure features
        # last_trade_side: +1=buy, -1=sell, 0=none/unknown
        self.coinbase_last_trade_side: Dict[str, int] = defaultdict(int)
        self.coinbase_last_trade_ts_s: Dict[str, float] = defaultdict(float)
        self.coinbase_last_trade_px: Dict[str, float] = defaultdict(float)
        self.coinbase_last_trade_vs_mid_bp: Dict[str, float] = defaultdict(float)
        self.coinbase_last_trade_at_bid: Dict[str, int] = defaultdict(int)
        self.coinbase_last_trade_at_ask: Dict[str, int] = defaultdict(int)
        self.coinbase_buy_frac_1s: Dict[str, float] = defaultdict(float)
        self.coinbase_buy_frac_3s: Dict[str, float] = defaultdict(float)
        self.coinbase_buy_frac_5s: Dict[str, float] = defaultdict(float)

        # 6. Short-horizon microprice returns / realized variance
        # (per base; store last ~5s of microprices)
        self._cb_micro_hist: Dict[str, deque] = defaultdict(lambda: deque())
        self.coinbase_ret_1s: Dict[str, float] = defaultdict(float)
        self.coinbase_ret_3s: Dict[str, float] = defaultdict(float)
        self.coinbase_ret_5s: Dict[str, float] = defaultdict(float)
        self.coinbase_ret_10s: Dict[str, float] = defaultdict(float)   
        self.coinbase_rv_3s: Dict[str, float] = defaultdict(float)
        self.coinbase_rv_up_3s: Dict[str, float] = defaultdict(float)
        self.coinbase_rv_dn_3s: Dict[str, float] = defaultdict(float)

        # Notifier
        self.refresh_event: asyncio.Event = asyncio.Event()

    # --------- tiny helpers (existing compat) ---------
    def mark_updated(self) -> None:
        if not self.refresh_event.is_set():
            self.refresh_event.set()

    def set_model_prediction(self, base: str, iv: float, df: float, exch_sec: Optional[int]) -> None:
        self.model_preds[str(base).upper()] = {"iv": float(iv), "df": float(df), "exch_sec": exch_sec}
        self.mark_updated()

    # ========================= NEW: Coinbase microprice =========================
    def set_coinbase_prev_close(
        self,
        base: str,
        prev_close: Optional[float],
        prev_high: Optional[float] = None,
        prev_low: Optional[float] = None,
    ) -> None:
        baseU = str(base).upper()

        if prev_close is None or prev_high is None or prev_low is None:
            # wipe all three if any is missing
            self.coinbase_prev_close.pop(baseU, None)
            self.coinbase_prev_high.pop(baseU, None)
            self.coinbase_prev_low.pop(baseU, None)
            self.mark_updated()
            return

        self.coinbase_prev_close[baseU] = float(prev_close)
        self.coinbase_prev_high[baseU] = float(prev_high)
        self.coinbase_prev_low[baseU] = float(prev_low)
        self.mark_updated()

    def set_cb_microprice(self, product_id: str, microprice: float) -> None:
        """
        Set latest Coinbase microprice for the product's BASE,
        shifting the previous value into `coinbase_microprice_prev`.
        """
        base = _base_from_product(product_id)
        micro = float(microprice)
        prev = self.coinbase_microprice_latest.get(base)
        if prev is not None:
            self.coinbase_microprice_prev[base] = prev
        self.coinbase_microprice_latest[base] = micro
        self.mark_updated()
    
    def set_cb_server_ts(self, product_id: str, exch_ts: str) -> None:
        base = _base_from_product(product_id)
        try:
            # Use integer nanoseconds for accuracy, then convert to ms
            ts_ns = pd.Timestamp(exch_ts).value  # ns since epoch, int
            # Guard weird values (e.g., pandas can't parse): raises to except below
            if not isinstance(ts_ns, (int, np.integer)):
                return
            ms = int(ts_ns // 1_000_000)
            self.coinbase_ts_server_ms[base] = ms
            self.mark_updated()
        except Exception:
            pass

    def set_cb_mp_skews(self, product_id: str, skew1: float, skew2: float, skew5: float, skew10: float) -> None:
        base = _base_from_product(product_id)
        self.coinbase_mp_skew_bps1[base] = float(skew1)
        self.coinbase_mp_skew_bps2[base] = float(skew2)
        self.coinbase_mp_skew_bps5[base] = float(skew5)
        self.coinbase_mp_skew_bps10[base] = float(skew10)
        self.mark_updated()

    def set_cb_l2_features(
        self,
        product_id: str,
        *,
        micro: Optional[float],
        best_bid: Optional[float],
        best_ask: Optional[float],
        best_bid_sz: float,
        best_ask_sz: float,
        spread_abs: float,
        spread_bp: float,
        depth_bands: Dict[float, Dict[str, float]],
        near_far_ratio_bid: float,
        near_far_ratio_ask: float,
        avg_dist_bid_bp: float,
        avg_dist_ask_bp: float,
        slope_bid: float,
        slope_ask: float,
        wall_bid_size: float,
        wall_bid_dist_bp: float,
        wall_ask_size: float,
        wall_ask_dist_bp: float,
        wall_imbalance: float,
        depth_imb_1bp_diff: float,
        net_add_bid_1bp_1s: float,
        net_add_ask_1bp_1s: float,
        n_bid_improve_1s: int,
        n_bid_worsen_1s: int,
        n_ask_improve_1s: int,
        n_ask_worsen_1s: int,
        n_spread_tighten_1s: int,
        n_spread_widen_1s: int,
    ) -> None:
        base = _base_from_product(product_id)

        # Top-of-book
        if best_bid is not None:
            self.coinbase_tob_bid_px[base] = float(best_bid)
        else:
            self.coinbase_tob_bid_px.pop(base, None)
        if best_ask is not None:
            self.coinbase_tob_ask_px[base] = float(best_ask)
        else:
            self.coinbase_tob_ask_px.pop(base, None)

        self.coinbase_tob_bid_qty[base] = float(best_bid_sz)
        self.coinbase_tob_ask_qty[base] = float(best_ask_sz)
        self.coinbase_spread_abs[base] = float(spread_abs)
        self.coinbase_spread_bp[base] = float(spread_bp)

        # Depth bands (only 1,2,5,10 bps are used)
        for bps, vals in depth_bands.items():
            k = int(bps)
            bid = float(vals.get("bid", 0.0))
            ask = float(vals.get("ask", 0.0))
            tot = float(vals.get("tot", 0.0))
            imb = float(vals.get("imb", 0.0))
            lr = float(vals.get("lr", 0.0))
            if k == 1:
                self.coinbase_depth_bid_1bp[base] = bid
                self.coinbase_depth_ask_1bp[base] = ask
                self.coinbase_depth_tot_1bp[base] = tot
                self.coinbase_depth_imb_1bp[base] = imb
                self.coinbase_depth_lr_1bp[base] = lr
            elif k == 2:
                self.coinbase_depth_bid_2bp[base] = bid
                self.coinbase_depth_ask_2bp[base] = ask
                self.coinbase_depth_tot_2bp[base] = tot
                self.coinbase_depth_imb_2bp[base] = imb
                self.coinbase_depth_lr_2bp[base] = lr
            elif k == 5:
                self.coinbase_depth_bid_5bp[base] = bid
                self.coinbase_depth_ask_5bp[base] = ask
                self.coinbase_depth_tot_5bp[base] = tot
                self.coinbase_depth_imb_5bp[base] = imb
                self.coinbase_depth_lr_5bp[base] = lr
            elif k == 10:
                self.coinbase_depth_bid_10bp[base] = bid
                self.coinbase_depth_ask_10bp[base] = ask
                self.coinbase_depth_tot_10bp[base] = tot
                self.coinbase_depth_imb_10bp[base] = imb
                self.coinbase_depth_lr_10bp[base] = lr

        self.coinbase_depth_near_far_ratio_bid[base] = float(near_far_ratio_bid)
        self.coinbase_depth_near_far_ratio_ask[base] = float(near_far_ratio_ask)

        # Book slope / convexity
        self.coinbase_book_avg_dist_bid_bp[base] = float(avg_dist_bid_bp)
        self.coinbase_book_avg_dist_ask_bp[base] = float(avg_dist_ask_bp)
        self.coinbase_book_slope_bid[base] = float(slope_bid)
        self.coinbase_book_slope_ask[base] = float(slope_ask)
        self.coinbase_book_convexity[base] = float(slope_bid + slope_ask)

        # Liquidity walls
        self.coinbase_wall_bid_size[base] = float(wall_bid_size)
        self.coinbase_wall_bid_dist_bp[base] = float(wall_bid_dist_bp)
        self.coinbase_wall_ask_size[base] = float(wall_ask_size)
        self.coinbase_wall_ask_dist_bp[base] = float(wall_ask_dist_bp)
        self.coinbase_wall_imbalance[base] = float(wall_imbalance)

        # L2 dynamics
        self.coinbase_depth_imb_1bp_diff[base] = float(depth_imb_1bp_diff)
        self.coinbase_net_add_bid_1bp_1s[base] = float(net_add_bid_1bp_1s)
        self.coinbase_net_add_ask_1bp_1s[base] = float(net_add_ask_1bp_1s)

        self.coinbase_n_bid_improve_1s[base] = int(n_bid_improve_1s)
        self.coinbase_n_bid_worsen_1s[base] = int(n_bid_worsen_1s)
        self.coinbase_n_ask_improve_1s[base] = int(n_ask_improve_1s)
        self.coinbase_n_ask_worsen_1s[base] = int(n_ask_worsen_1s)
        self.coinbase_n_spread_tighten_1s[base] = int(n_spread_tighten_1s)
        self.coinbase_n_spread_widen_1s[base] = int(n_spread_widen_1s)

        self.mark_updated()

    def update_cb_ewma(self, product_id: str, px: float, exch_ts: str,
                    half_life_sec: float = 30.0, jump_k: float = 3.0) -> None:
        """
        Per-second EWMA of log-returns with *no division by dt* in the increment.
        decay = exp(-dt/HL);  var <- decay*var + (1-decay)*(log(px/px_prev))^2
        sigma = sqrt(var). Jump flag = 1{ |r| > jump_k * sigma }.

        Also updates short-horizon microprice features:
        - coinbase_ret_1s / coinbase_ret_3s
        - coinbase_rv_3s, coinbase_rv_up_3s, coinbase_rv_dn_3s
        """
        base = _base_from_product(product_id)
        try:
            t = float(pd.Timestamp(exch_ts).timestamp())
        except Exception:
            return
        if not (isinstance(px, (int, float)) and px > 0):
            return

        lp = float(px)
        st = self._cb_ewma_state[base]

        # --- long-ish horizon EWMA sigma + jump flag (existing behaviour) ---
        if "last_px" in st and "last_ts" in st and st["last_px"] > 0 and st["last_ts"] > 0:
            dt = max(0.0, t - st["last_ts"])
            r = math.log(lp / st["last_px"]) if st["last_px"] > 0 else 0.0
            decay = math.exp(-dt / float(half_life_sec)) if dt > 0 else 0.0
            var_prev = float(st.get("var", 0.0))
            var_new = decay * var_prev + (1.0 - decay) * (r * r)  # **no /dt**
            self.coinbase_sigma_ewma[base] = math.sqrt(max(var_new, 0.0))
            self.coinbase_jump_flag[base] = 1 if (
                self.coinbase_sigma_ewma[base] > 0
                and abs(r) > jump_k * self.coinbase_sigma_ewma[base]
            ) else 0
            st["var"] = var_new
        else:
            # First observation: initialize softly
            st["var"] = 0.0
            self.coinbase_sigma_ewma[base] = 0.0
            self.coinbase_jump_flag[base] = 0

        st["last_px"] = lp
        st["last_ts"] = t

        # --- NEW: short-horizon realized returns / volatility (1–3s) ---
        hist = self._cb_micro_hist[base]
        hist.append((t, lp))

        # keep only last ~5s of history (enough for 3s window + margin)
        cutoff_hist = t - 11.0
        while hist and hist[0][0] < cutoff_hist:
            hist.popleft()

        # 1s / 3s / 5s / 10s log-returns
        ret_1s = 0.0
        ret_3s = 0.0
        ret_5s = 0.0
        ret_10s = 0.0
        t1_cut = t - 1.0
        t3_cut = t - 3.0
        t5_cut = t - 5.0
        t10_cut = t - 10.0
        p1 = p3 = p5 = p10 = None

        for ts0, px0 in hist:
            if p1 is None and ts0 >= t1_cut:
                p1 = px0
            if p3 is None and ts0 >= t3_cut:
                p3 = px0
            if p5 is None and ts0 >= t5_cut:
                p5 = px0
            if p10 is None and ts0 >= t10_cut:
                p10 = px0

        if p1 is not None and p1 > 0:
            ret_1s = math.log(lp / p1)
        if p3 is not None and p3 > 0:
            ret_3s = math.log(lp / p3)
        if p5 is not None and p5 > 0:
            ret_5s = math.log(lp / p5)
        if p10 is not None and p10 > 0:
            ret_10s = math.log(lp / p10)

        self.coinbase_ret_1s[base] = ret_1s
        self.coinbase_ret_3s[base] = ret_3s
        self.coinbase_ret_5s[base] = ret_5s
        self.coinbase_ret_10s[base] = ret_10s

        # Realized variance over last 3 seconds (total / up / down)
        rv = rv_up = rv_dn = 0.0
        hist_list = list(hist)

        if len(hist_list) >= 2:
            prev_t, prev_px = hist_list[0]
            for cur_t, cur_px in hist_list[1:]:
                if cur_t < t3_cut:
                    prev_t, prev_px = cur_t, cur_px
                    continue
                if prev_px > 0 and cur_px > 0:
                    r = math.log(cur_px / prev_px)
                    r2 = r * r
                    rv += r2
                    if r > 0:
                        rv_up += r2
                    elif r < 0:
                        rv_dn += r2
                prev_t, prev_px = cur_t, cur_px

        self.coinbase_rv_3s[base]    = math.sqrt(rv)    if rv    > 0 else 0.0
        self.coinbase_rv_up_3s[base] = math.sqrt(rv_up) if rv_up > 0 else 0.0
        self.coinbase_rv_dn_3s[base] = math.sqrt(rv_dn) if rv_dn > 0 else 0.0

        self.mark_updated()

    # ========================= NEW: Coinbase trade flow =========================
    def register_cb_trade(
        self,
        product_id: str,
        *,
        side: str,
        size: float,
        exch_ts: str,
        price: Optional[float] = None,
    ) -> None:
        """
        Record a single trade and refresh rolling 1s/3s buy/sell/net aggregates (per BASE).
        Also updates:
          - last trade side / price / timestamp
          - last_trade_vs_mid_bp, last_trade_at_bid/ask
          - buy_frac_1s / buy_frac_3s
        `exch_ts` should be the Coinbase exchange timestamp string from the WS message
        (top-level "timestamp", not local receipt time).
        """
        base = _base_from_product(product_id)
        ts_s = _as_epoch_s(exch_ts)

        side_up = str(side).upper()
        sz = float(size)

        buy = sz if side_up == "BUY" else 0.0
        sell = sz if side_up == "SELL" else 0.0

        dq = self._cb_trades[base]
        dq.append((ts_s, buy, sell))

        # prune to the last 5 seconds
        cutoff = ts_s - 5.0
        while dq and dq[0][0] < cutoff:
            dq.popleft()

        # recompute 1s & 3s aggregates
        buy1 = sell1 = 0.0
        buy3 = sell3 = 0.0
        buy5 = sell5 = 0.0
        one_sec_cut = ts_s - 1.0
        three_sec_cut = ts_s - 3.0
        five_sec_cut = ts_s - 5.0
        for t, b, s in dq:
            if t >= one_sec_cut:
                buy1 += b
                sell1 += s
            if t >= three_sec_cut:
                buy3 += b
                sell3 += s
            if t >= five_sec_cut:
                buy5 += b
                sell5 += s

        self.coinbase_flow_1s[base] = _FlowWindow(
            buy=buy1, sell=sell1, net=buy1 - sell1, asof=ts_s
        )
        self.coinbase_flow_3s[base] = _FlowWindow(
            buy=buy3, sell=sell3, net=buy3 - sell3, asof=ts_s
        )
        self.coinbase_flow_5s[base] = _FlowWindow(
            buy=buy5, sell=sell5, net=buy5 - sell5, asof=ts_s
        )

        # Buy fractions (how much of recent flow is buy-initiated)
        tot1 = buy1 + sell1
        self.coinbase_buy_frac_1s[base] = float(buy1 / tot1) if tot1 > 0 else 0.5
        tot3 = buy3 + sell3
        self.coinbase_buy_frac_3s[base] = float(buy3 / tot3) if tot3 > 0 else 0.5
        tot5 = buy5 + sell5
        self.coinbase_buy_frac_5s[base] = float(buy5 / tot5) if tot5 > 0 else 0.5

        # Last-trade microstructure features
        if sz > 0.0:
            px = None
            if price is not None:
                try:
                    px = float(price)
                except Exception:
                    px = None

            side_int = 1 if side_up == "BUY" else -1 if side_up == "SELL" else 0
            self.coinbase_last_trade_side[base] = side_int
            self.coinbase_last_trade_ts_s[base] = ts_s
            if px is not None and px > 0.0:
                self.coinbase_last_trade_px[base] = px

                micro = self.coinbase_microprice_latest.get(base)
                if micro and micro > 0.0:
                    self.coinbase_last_trade_vs_mid_bp[base] = (
                        (px - micro) / micro * 1e4
                    )

                # Approx: was this trade at current bid/ask?
                bid_px = self.coinbase_tob_bid_px.get(base)
                ask_px = self.coinbase_tob_ask_px.get(base)
                at_bid = 1 if (bid_px is not None and abs(px - bid_px) <= 1e-8) else 0
                at_ask = 1 if (ask_px is not None and abs(px - ask_px) <= 1e-8) else 0
                self.coinbase_last_trade_at_bid[base] = at_bid
                self.coinbase_last_trade_at_ask[base] = at_ask

    def update_pm_iv_multiplicity(
        self,
        base: str,
        pm_iv_900_obs: float,
        quality: float = 1.0,
        q_process: float = 1e-5,   # process noise in log-space
        r_base: float = 0.04,      # base measurement noise var (~0.2^2) in log-space
    ) -> None:
        """
        Kalman-style update for IV multiplicity (log-space).

        pm_iv_900_obs : raw implied IV from Polymarket (per 900s).
        quality       : ∈ [0,1], higher = more trusted observation.
        """
        baseU = str(base).upper()
        mp = self.model_preds.get(baseU)
        if not mp:
            return

        model_iv = float(mp.get("iv") or 0.0)
        if not (model_iv > 0.0 and pm_iv_900_obs > 0.0 and math.isfinite(pm_iv_900_obs)):
            return

        # ----- Observation in log-space -----
        ratio_raw = pm_iv_900_obs / model_iv
        ratio = max(0.2, min(5.0, ratio_raw))  # clamp extremes
        y = math.log(ratio)

        # ----- Prior state -----
        k_prev = float(self._pm_iv_log_mult[baseU])
        P_prev = float(self._pm_iv_log_mult_var[baseU])

        if not (P_prev > 0.0 and math.isfinite(P_prev)):
            # Initialize from first observation
            k_prev = y
            P_prev = r_base

        # ----- Measurement noise R_t -----
        quality = max(0.0, min(1.0, quality))
        if quality > 0.0:
            R_t = r_base / quality
        else:
            return

        if not math.isfinite(R_t):
            return

        # ----- Kalman gain -----
        S_t = P_prev + R_t
        if not (S_t > 0.0 and math.isfinite(S_t)):
            return

        K_t = P_prev / S_t

        # ----- Posterior mean/var -----
        k_post = k_prev + K_t * (y - k_prev)
        P_post = (1.0 - K_t) * P_prev + q_process

        self._pm_iv_log_mult[baseU] = k_post
        self._pm_iv_log_mult_var[baseU] = P_post

        m_post = math.exp(k_post)
        self.pm_iv_mult[baseU] = m_post
        self.pm_iv_implied_900[baseU] = m_post * model_iv

        self.mark_updated()

    # ----------------- convenience getters (optional) -----------------
    def get_cb_microprices(self, base: str) -> Tuple[Optional[float], Optional[float]]:
        base = str(base).upper()
        return (
            self.coinbase_microprice_latest.get(base),
            self.coinbase_microprice_prev.get(base),
        )

    def get_cb_flow(self, base: str) -> Tuple[_FlowWindow, _FlowWindow]:
        base = str(base).upper()
        return (self.coinbase_flow_1s.get(base, _FlowWindow()),
                self.coinbase_flow_3s.get(base, _FlowWindow()))

async def run_pm_iv_calibrator(
    hub: StateHub,
    bases: Sequence[str] = ("BTC", "ETH", "XRP", "SOL"),
    interval_s: float = 0.1,
) -> None:
    """
    Periodic (e.g. 100ms) updater for Polymarket IV multiplicity.
    """
    basesU = [str(b).upper() for b in bases]

    while True:
        try:
            for baseU in basesU:
                # 1) Latest PM book for this base
                st = hub.pm_base_state.get(baseU)
                if not st:
                    continue

                bids: List[Tuple[float, float]] = st.get("bids") or []
                asks: List[Tuple[float, float]] = st.get("asks") or []

                if not bids and not asks:
                    continue

                # Best prices + sizes; treat empty side price as 0 or 1, size = 0
                if bids:
                    best_bid_px, best_bid_sz = bids[0]
                else:
                    best_bid_px, best_bid_sz = 0.0, 0.0
                if asks:
                    best_ask_px, best_ask_sz = asks[0]
                else:
                    best_ask_px, best_ask_sz = 1.0, 0.0

                denom = best_bid_sz + best_ask_sz
                if denom <= 0.0:
                    continue

                pm_mid = (best_bid_px * best_bid_sz + best_ask_px * best_ask_sz) / denom
                if not (0.0 < pm_mid < 1.0):
                    continue

                pm_spread = (best_ask_px - best_bid_px) if (bids and asks) else None
                if pm_spread is None or pm_spread <= 0.0:
                    continue

                # 2) Underlying + prev close + tau
                spot = hub.coinbase_microprice_latest.get(baseU)
                if spot is None or not (spot > 0.0):
                    continue

                prev_close_px = hub.coinbase_prev_close.get(baseU)
                now_ms = hub.coinbase_ts_server_ms.get(baseU)

                if prev_close_px is None or now_ms is None:
                    continue
                if prev_close_px <= 0.0:
                    continue

                # Use the SAME horizon as live_prediction: B = next 15m boundary
                now_s = now_ms / 1000.0
                B = _next_15m_boundary_s(int(now_s))
                tau = float(B - now_s - 1.1)
                if tau <= 0.0:
                    continue

                tau = max(tau, 1.0)

                # 3) Skip 0.5-mismatch cases (impossible under symmetric, zero-drift Student-t)
                r_star = math.log(prev_close_px / spot)
                if (r_star < 0.0 and pm_mid <= 0.5) or (r_star > 0.0 and pm_mid >= 0.5):
                    continue

                # 4) Quality weights: spread + downweight near 0.5
                w_spread = 1.0 / (1.0 + (pm_spread / 0.02) ** 2)
                w_atm = 2.0 * abs(pm_mid - 0.5)
                if w_atm < 0.0:
                    w_atm = 0.0
                elif w_atm > 1.0:
                    w_atm = 1.0

                quality = w_spread * w_atm
                if quality <= 0.0:
                    continue

                # 5) Model prediction for this base
                mp = hub.model_preds.get(baseU)
                if not mp:
                    continue
                df = float(mp.get("df") or 5.0)

                # 6) Implied IV under Student-t (per 900s)
                pm_iv_900 = implied_iv_from_price_student_t(
                    target_p=pm_mid,
                    spot=spot,
                    prev_close=prev_close_px,
                    tau=tau,
                    df=df,
                )
                if not (pm_iv_900 and pm_iv_900 > 0.0):
                    continue

                # 7) Update multiplicity in hub (Kalman update)
                hub.update_pm_iv_multiplicity(baseU, pm_iv_900, quality=quality)

        except Exception:
            # Don't let a single failure kill the loop
            pass

        await asyncio.sleep(interval_s)

# ----------------------------- time helpers -----------------------------
def _now_utc_s() -> int:
    import time
    return int(time.time())

def _next_15m_boundary_s(now_s: Optional[int] = None) -> int:
    t = _now_utc_s() if now_s is None else now_s
    return math.ceil(t / WINDOW_SEC) * WINDOW_SEC

# ----------------------------- Chainlink Resolution Service -----------------------------
@dataclass
class Tick:
    ts_ms: int
    price: float

def _to_base(sym: str) -> str:
    s = (sym or "").lower()
    if s.startswith("btc/"): return "BTC"
    if s.startswith("eth/"): return "ETH"
    if s.startswith("sol/"): return "SOL"
    if s.startswith("xrp/"): return "XRP"
    su = (sym or "").upper()
    for b in BASES:
        if su.startswith(b): return b
    return su

class ChainlinkResolutionService:
    """
    Subscribe to topic=crypto_prices_chainlink (type="*"):
      - Maintain small tick deques per base
      - Keep latest tick price in hub.chainlink_latest
      - Every 15 minutes at boundary B, select close as:
          • last tick with ts <= B*1000; else first tick within +5s
      - Persist resolution to data/polymarket/resolution/{BASE}.jsonl
      - Update hub.chainlink_15m_close

    Additionally, we flush a 15m close *on tick boundary crossing per base* so
    every market (BTC/ETH/SOL/XRP) emits a close even if ticks arrive late.
    """
    def __init__(self,
                 hub: StateHub,
                 bases: List[str] = list(BASES),
                 raw_path: str = PM_RAW_CHAINLINK,
                 other_path: str = PM_RAW_OTHER,
                 out_dir: str = PM_RES_DIR,
                 lookback_sec: int = 3600,
                 after_grace_sec: int = 5,
                 oracles: Optional[Dict[str, 'FastChainlinkOracle']] = None):
        self.hub = hub
        self.bases = list(bases)
        self.raw_path = raw_path
        self.other_path = other_path
        self.out_dir = out_dir
        # keep at least a full 15m window + slack
        self.lookback_sec = max(1200, lookback_sec)
        self.after_grace_sec = after_grace_sec
        self.oracles = oracles

        self._buf: Dict[str, deque[Tick]] = {b: deque() for b in self.bases}
        self._ws_task: Optional[asyncio.Task] = None
        self._sched_15m: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

        # Per-base 15m bucket trackers (tick-driven flush)
        self._bucket_idx: Dict[str, int] = {}        # base -> current 15m bucket index (ts//900)
        self._bucket_last_px: Dict[str, float] = {}  # base -> last price seen in current bucket
        self._bucket_last_ts: Dict[str, int] = {}    # base -> last tick ts_ms in current bucket

    def _prune(self, base: str, now_ms: int) -> None:
        lb_ms = now_ms - self.lookback_sec * 1000
        dq = self._buf[base]
        while dq and dq[0].ts_ms < lb_ms:
            dq.popleft()

    def _write_line(self, path: str, obj: Dict[str, Any]) -> None:
        line = f"{pd.Timestamp.utcnow().isoformat()}Z {json.dumps(obj, separators=(',',':'))}\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    def _write_raw(self, obj: Dict[str, Any]) -> None:
        self._write_line(self.raw_path, obj)

    def _write_other(self, obj: Dict[str, Any]) -> None:
        self._write_line(self.other_path, obj)

    def _persist_resolution(self, base: str, boundary_s: int, close_px: float, src_ts_ms: int, method: str) -> None:
        rec = {
            "event": "resolution",
            "base": base,
            "end_unix": int(boundary_s),
            "end_iso": pd.to_datetime(boundary_s, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": float(close_px),
            "source_ts_ms": int(src_ts_ms),
            "source_iso": pd.to_datetime(src_ts_ms, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "method": method,
        }
        path = os.path.join(self.out_dir, f"{base}.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except Exception as e:
            jlog(logging.ERROR, "resolution_write_error", symbol=base, error=str(e))
        self.hub.chainlink_15m_close[base] = (boundary_s, float(close_px))
        self.hub.mark_updated()
        jlog(logging.INFO, "resolution", **rec)

    def _ingest_tick(self, symbol: str, ts_ms: int, price: float) -> None:
        base = _to_base(symbol)
        if base not in self._buf:
            return

        # append & prune
        self._buf[base].append(Tick(ts_ms=ts_ms, price=float(price)))
        self._prune(base, ts_ms)

        # update latest
        self.hub.chainlink_latest[base] = float(price)
        if self.oracles:
            o = self.oracles.get(base)
            if o:
                o.on_cl_tick(px=float(price), ts_ms=int(ts_ms))

        # per-base bucket detection (tick-driven close flush)
        idx = (ts_ms // 1000) // 900  # 15m bucket index
        prev_idx = self._bucket_idx.get(base)
        if prev_idx is None:
            # initialize trackers for this base
            self._bucket_idx[base] = idx
            self._bucket_last_px[base] = float(price)
            self._bucket_last_ts[base] = int(ts_ms)
            self.hub.mark_updated()
            return

        if idx != prev_idx:
            # crossed into a new bucket -> flush previous bucket close if not already recorded
            end_s = (prev_idx + 1) * 900
            already = self.hub.chainlink_15m_close.get(base)
            if not (already and isinstance(already, tuple) and already[0] == end_s):
                close_px = self._bucket_last_px.get(base, float(price))
                src_ts = self._bucket_last_ts.get(base, int(ts_ms))
                self._persist_resolution(base, end_s, close_px, src_ts, method="tick_boundary_flush")

            # start tracking new bucket
            self._bucket_idx[base] = idx
            self._bucket_last_px[base] = float(price)
            self._bucket_last_ts[base] = int(ts_ms)
        else:
            # still in same bucket, update last
            self._bucket_last_px[base] = float(price)
            self._bucket_last_ts[base] = int(ts_ms)

        self.hub.mark_updated()

    def _pick_close(self, dq: deque[Tick], boundary_ms: int, grace_after_ms: int) -> Optional[Tick]:
        # prefer last <= boundary
        for t in reversed(dq):
            if t.ts_ms <= boundary_ms:
                return t
        # else first within +grace
        for t in dq:
            if boundary_ms <= t.ts_ms <= boundary_ms + grace_after_ms:
                return t
        return None

    def _resolve_for_base(self, base: str, boundary_s: int) -> Optional[Dict[str, Any]]:
        dq = self._buf.get(base)
        if not dq:
            return None
        B_ms = boundary_s * 1000
        chosen = self._pick_close(dq, B_ms, self.after_grace_sec * 1000)
        if not chosen:
            return None
        rec = {
            "event": "resolution",
            "base": base,
            "end_unix": boundary_s,
            "end_iso": pd.to_datetime(boundary_s, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": float(chosen.price),
            "source_ts_ms": int(chosen.ts_ms),
            "source_iso": pd.to_datetime(chosen.ts_ms, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "method": "close<=boundary_or_after_grace",
        }
        return rec

    def _extract_chainlink_ticks(self, msg: Dict[str, Any]) -> List[Tuple[str, int, float]]:
        out: List[Tuple[str, int, float]] = []

        # Primary format (your example): top-level 'payload'
        pld = msg.get("payload")
        if isinstance(pld, dict):
            sym = pld.get("symbol")
            val = pld.get("value")
            ts = pld.get("timestamp")  # ms
            if sym is not None and val is not None and ts is not None:
                try:
                    out.append((str(sym), int(ts), float(val)))
                except Exception:
                    pass

        # Fallback flat / nested / arrays
        for candidate in (msg, msg.get("data") if isinstance(msg.get("data"), dict) else None):
            if not isinstance(candidate, dict):  # skip
                continue
            sym = candidate.get("symbol") or candidate.get("pair")
            val = candidate.get("value") or candidate.get("price") or candidate.get("p")
            ts = candidate.get("timestamp") or candidate.get("ts") or candidate.get("t")
            if sym is not None and val is not None and ts is not None:
                try:
                    ts_i = int(ts)
                    ts_ms = ts_i if ts_i > 10_000_000_000 else ts_i * 1000
                    out.append((str(sym), ts_ms, float(val)))
                except Exception:
                    pass

        for key in ("prices", "updates", "events"):
            arr = msg.get(key)
            if isinstance(arr, list):
                for it in arr:
                    if not isinstance(it, dict):
                        continue
                    sym = it.get("symbol") or it.get("pair")
                    val = it.get("value") or it.get("price") or it.get("p")
                    ts = it.get("timestamp") or it.get("ts") or it.get("t")
                    if sym is not None and val is not None and ts is not None:
                        try:
                            ts_i = int(ts)
                            ts_ms = ts_i if ts_i > 10_000_000_000 else ts_i * 1000
                            out.append((str(sym), ts_ms, float(val)))
                        except Exception:
                            continue

        return out

    async def _ws_loop(self) -> None:
        subs = {
            "action": "subscribe",
            "subscriptions": [{
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": ""
            }]
        }
        while not self._stop.is_set():
            try:
                async with websockets.connect(RTDS_URL, ping_interval=20, ping_timeout=20) as ws:
                    jlog(logging.INFO, "rtds_open", url=RTDS_URL, topic="crypto_prices_chainlink")
                    await ws.send(json.dumps(subs))
                    jlog(logging.INFO, "rtds_subscribed", topic="crypto_prices_chainlink")

                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            self._write_other({"raw": str(raw)})
                            continue
                        if msg.get("topic") == "crypto_prices_chainlink":
                            # self._write_raw(msg)
                            for sym, ts_ms, price in self._extract_chainlink_ticks(msg):
                                self._ingest_tick(sym, ts_ms, price)
                        else:
                            self._write_other(msg)
            except Exception as e:
                jlog(logging.ERROR, "rtds_error", error=str(e))
                await asyncio.sleep(1.0)

    async def _scheduler_15m(self) -> None:
        while not self._stop.is_set():
            now = _now_utc_s()
            B = _next_15m_boundary_s(now)
            delay = max(0.0, B - now)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                pass
            # tiny grace
            await asyncio.sleep(0.25)
            for base in self.bases:
                # skip if already flushed by tick-boundary
                existing = self.hub.chainlink_15m_close.get(base)
                if existing and isinstance(existing, tuple) and existing[0] == B:
                    continue
                rec = self._resolve_for_base(base, B)
                if rec:
                    try:
                        path = os.path.join(self.out_dir, f"{base}.jsonl")
                        with open(path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    except Exception as e:
                        jlog(logging.ERROR, "resolution_write_error", symbol=base, error=str(e))
                    self.hub.chainlink_15m_close[base] = (B, float(rec["price"]))
                    self.hub.mark_updated()
                    evt = rec.get("event", "resolution")
                    fields = {k: v for k, v in rec.items() if k != "event"}
                    jlog(logging.INFO, evt, **fields)
                else:
                    jlog(logging.WARNING, "resolution_miss",
                         base=base, end_unix=B,
                         end_iso=pd.to_datetime(B, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"))

    async def start(self) -> None:
        self._stop.clear()
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._sched_15m = asyncio.create_task(self._scheduler_15m())

    async def stop(self) -> None:
        self._stop.set()
        for t in (self._ws_task, self._sched_15m):
            if t:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass


# ----------------------------- Coinbase public recorder -----------------------------
async def run_coinbase_recorder(hub: StateHub, products: List[str] = None, write_files: bool = False) -> None:
    products = [p.upper() for p in (products or list(DEFAULT_PRODUCTS)) if p]
    os.makedirs(CB_DIR, exist_ok=True)

    class Writers:
        def __init__(self, base_dir: str, max_bytes: int, flush_every: int):
            self.base_dir = base_dir
            self.max_bytes = max_bytes
            self.flush_every = flush_every
            self.queue: asyncio.Queue[Tuple[str, str, str]] = asyncio.Queue(maxsize=20000)
            self.state: Dict[Tuple[str, str], Dict[str, Any]] = {}
            self.task = asyncio.create_task(self._worker())

        def _dir(self, product: str, channel: str) -> str:
            d = os.path.join(self.base_dir, product, channel)
            os.makedirs(d, exist_ok=True)
            return d

        def _open_new(self, product: str, channel: str):
            d = self._dir(product, channel)
            existing = sorted(glob.glob(os.path.join(d, f"{channel}-*.jsonl")))
            idx = 0
            if existing:
                try:
                    last = os.path.basename(existing[-1])
                    idx = int(last.split("-")[-1].split(".")[0]) + 1
                except Exception:
                    idx = len(existing)
            path = os.path.join(d, f"{channel}-{idx:05d}.jsonl")
            fh = open(path, "a", encoding="utf-8")
            self.state[(product, channel)] = {"fh": fh, "bytes": 0, "index": idx, "n_since_flush": 0}

        async def write(self, product: str, channel: str, obj: Dict[str, Any]):
            line = f"{pd.Timestamp.utcnow().isoformat()}Z {json.dumps(obj, separators=(',',':'))}\n"
            try:
                self.queue.put_nowait((product or "ALL", channel, line))
            except asyncio.QueueFull:
                await self.queue.put((product or "ALL", channel, line))
            if self.queue.qsize() > self.queue.maxsize * 0.8:
                await asyncio.sleep(0)

        async def _worker(self):
            while True:
                product, channel, line = await self.queue.get()
                key = (product, channel)
                st = self.state.get(key)
                if st is None:
                    self._open_new(product, channel)
                    st = self.state[key]
                if st["bytes"] + len(line) > self.max_bytes:
                    try:
                        st["fh"].close()
                    except Exception:
                        pass
                    st["index"] += 1
                    d = self._dir(product, channel)
                    path = os.path.join(d, f"{channel}-{st['index']:05d}.jsonl")
                    st["fh"] = open(path, "a", encoding="utf-8")
                    st["bytes"] = 0
                    st["n_since_flush"] = 0
                st["fh"].write(line)
                st["bytes"] += len(line)
                st["n_since_flush"] += 1
                if st["n_since_flush"] >= self.flush_every:
                    try:
                        st["fh"].flush()
                    except Exception:
                        pass
                    st["n_since_flush"] = 0
                self.queue.task_done()

    writers = Writers(CB_DIR, CB_ROTATE_MAX_BYTES, CB_FLUSH_EVERY) if write_files else None

    class L2Book:
        __slots__ = ("bids", "asks", "last_seq")
        def __init__(self):
            self.bids: Dict[float, float] = {}
            self.asks: Dict[float, float] = {}
            self.last_seq: int = -1

        def apply_snapshot(self, updates: List[Dict[str, Any]]):
            self.bids.clear(); self.asks.clear()
            for u in updates or []:
                side = str(u.get("side","")).lower()
                try:
                    px = float(u.get("price_level")); qty = float(u.get("new_quantity"))
                except Exception:
                    continue
                if qty <= 0.0:
                    continue
                (self.bids if side == "bid" else self.asks)[px] = qty

        def apply_update(self, updates: List[Dict[str, Any]]):
            for u in updates or []:
                side = str(u.get("side","")).lower()
                try:
                    px = float(u.get("price_level")); qty = float(u.get("new_quantity"))
                except Exception:
                    continue
                book = self.bids if side == "bid" else self.asks
                if qty <= 0.0:
                    book.pop(px, None)
                else:
                    book[px] = qty

        def top_n(self, n: int = 20):
            bids = heapq.nlargest(n, self.bids.items(), key=lambda kv: kv[0])
            asks = heapq.nsmallest(n, self.asks.items(), key=lambda kv: kv[0])
            return bids, asks

    class L2Processor:
        """
        Coinbase L2 event processor that NEVER drops events.
        - Uses an *unbounded* asyncio.Queue (no maxsize) and does not evict.
        - Applies *every* L2 event to the in-memory book.
        - Computes rich L2 features (depth bands, slopes, walls, dynamics) and
        stores them into StateHub, but only at most once every `interval_s`
        seconds (interval-gated).
        """
        def __init__(self, product: str, interval_s: float = CB_L2_FEATURE_INTERVAL_S):
            self.product = product
            self.book: L2Book = L2Book()  # keep in-memory only; no hub book cache
            # Unbounded queue → no drops (beware: backpressure is memory).
            self.q: asyncio.Queue[Tuple[str, int, Dict[str, Any]]] = asyncio.Queue()
            self.task = asyncio.create_task(self._loop())

            # Dirty flag + last timestamp for interval gating
            self._dirty: bool = False
            self._last_ts: Optional[str] = None

            # Interval gating
            self._interval_s: float = float(interval_s)
            self._last_flush_s: Optional[float] = None

            # State for short-horizon L2 dynamics
            self._last_best_bid: Optional[float] = None
            self._last_best_ask: Optional[float] = None
            self._last_micro: Optional[float] = None
            self._last_depth_bid_1bp: float = 0.0
            self._last_depth_ask_1bp: float = 0.0
            self._last_depth_imb_1bp: float = 0.0
            # (ts, Δbid_1bp, Δask_1bp)
            self._depth_changes_1bp: deque = deque()
            # (ts, bid_imp, bid_worsen, ask_imp, ask_worsen, spread_tighten, spread_widen)
            self._quote_events_1s: deque = deque()

        def _mp_skew(self, micro: float, bps: float) -> float:
            if not (isinstance(micro, (int, float)) and micro > 0):
                return 0.0
            band = bps / 1e4
            lo = micro * (1.0 - band)
            hi = micro * (1.0 + band)
            # bids >= lo ; asks <= hi (books are dicts: price -> size)
            buy_sz  = sum(q for px, q in self.book.bids.items() if px >= lo)
            sell_sz = sum(q for px, q in self.book.asks.items() if px <= hi)
            tot = buy_sz + sell_sz
            return float((buy_sz - sell_sz) / tot) if tot > 0 else 0.0

        def submit_nowait(self, ts: str, seq: int, ev: Dict[str, Any]) -> None:
            """
            Enqueue without ever dropping events.
            With an unbounded queue, put_nowait won't raise; keep a defensive fallback.
            """
            try:
                self.q.put_nowait((ts, seq, ev))
            except asyncio.QueueFull:
                # Shouldn't happen with unbounded queue; defensive backpressure
                asyncio.create_task(self.q.put((ts, seq, ev)))

        def _compute_and_push_features(self, ts: str) -> None:
            """
            Heavy path: compute microprice, mp skews, depth bands, slopes, walls,
            short-horizon dynamics, and push to StateHub.

            This is called at most once every `_interval_s` seconds by `_loop`.
            """
            # Compute microprice from *current book* (robust to empty sides)
            best_bid = None
            best_ask = None

            # Compute microprice from *current book* (robust to empty sides)
            try:
                best_bid = max(self.book.bids.keys()) if self.book.bids else None
                best_ask = min(self.book.asks.keys()) if self.book.asks else None

                if (best_bid is None) and (best_ask is None):
                    micro = hub.coinbase_microprice_latest.get(
                        _base_from_product(self.product), None
                    )
                elif best_bid is None:
                    micro = best_ask
                elif best_ask is None:
                    micro = best_bid
                elif best_bid > best_ask:
                    # crossed book: fall back to last known micro
                    micro = hub.coinbase_microprice_latest.get(
                        _base_from_product(self.product), None
                    )
                else:
                    micro = 0.5 * (best_bid + best_ask)
            except Exception:
                micro = None

            # Skew
            s1  = self._mp_skew(micro, 1.0)  if micro else 0.0
            s2  = self._mp_skew(micro, 2.0)  if micro else 0.0
            s5  = self._mp_skew(micro, 5.0)  if micro else 0.0
            s10 = self._mp_skew(micro, 10.0) if micro else 0.0
            hub.set_cb_mp_skews(self.product, s1, s2, s5, s10)

            # Also keep the latest server timestamp for this base
            hub.set_cb_server_ts(self.product, ts)

            # ================== rich L2 features ==================
            now_s = _as_epoch_s(ts) if ts else 0.0
            base = _base_from_product(self.product)  # currently unused, kept for parity

            best_bid_sz = float(self.book.bids.get(best_bid, 0.0)) if best_bid is not None else 0.0
            best_ask_sz = float(self.book.asks.get(best_ask, 0.0)) if best_ask is not None else 0.0

            spread_abs = 0.0
            spread_bp = 0.0
            if best_bid is not None and best_ask is not None and micro and micro > 0.0:
                spread_abs = max(0.0, best_ask - best_bid)
                spread_bp = (spread_abs / micro) * 1e4 if spread_abs > 0 else 0.0

            # 1.2 Depth in ±1/2/5 bps bands around micro
            depth_bands: Dict[float, Dict[str, float]] = {
                1.0: {"bid": 0.0, "ask": 0.0, "tot": 0.0, "imb": 0.0, "lr": 0.0},
                2.0: {"bid": 0.0, "ask": 0.0, "tot": 0.0, "imb": 0.0, "lr": 0.0},
                5.0: {"bid": 0.0, "ask": 0.0, "tot": 0.0, "imb": 0.0, "lr": 0.0},
                10.0: {"bid": 0.0, "ask": 0.0, "tot": 0.0, "imb": 0.0, "lr": 0.0},
            }

            if micro and micro > 0.0:
                for bps in (1.0, 2.0, 5.0, 10.0):
                    band = bps / 1e4
                    lo = micro * (1.0 - band)
                    hi = micro * (1.0 + band)
                    bid_depth = sum(q for px, q in self.book.bids.items() if px >= lo)
                    ask_depth = sum(q for px, q in self.book.asks.items() if px <= hi)
                    tot = bid_depth + ask_depth
                    imb = (bid_depth - ask_depth) / tot if tot > 0 else 0.0
                    lr = math.log(bid_depth + 1e-9) - math.log(ask_depth + 1e-9)
                    depth_bands[bps] = {
                        "bid": float(bid_depth),
                        "ask": float(ask_depth),
                        "tot": float(tot),
                        "imb": float(imb),
                        "lr": float(lr),
                    }

            # 1.3 Near vs far depth ratios
            near_far_ratio_bid = 0.0
            near_far_ratio_ask = 0.0
            if 1.0 in depth_bands and 5.0 in depth_bands:
                near_bid = depth_bands[1.0]["bid"]
                far_bid = depth_bands[5.0]["bid"]
                if far_bid > 0:
                    near_far_ratio_bid = float(near_bid / far_bid)
                else:
                    near_far_ratio_bid = 1.0  # neutral if both near/far zero

                near_ask = depth_bands[1.0]["ask"]
                far_ask = depth_bands[5.0]["ask"]
                if far_ask > 0:
                    near_far_ratio_ask = float(near_ask / far_ask)
                else:
                    near_far_ratio_ask = 1.0  # neutral

            # 2. Book slope / convexity (top 5 levels)
            avg_dist_bid_bp = 0.0
            avg_dist_ask_bp = 0.0
            slope_bid = 0.0
            slope_ask = 0.0

            if micro and micro > 0.0:
                bids_top, asks_top = self.book.top_n(5)

                def _avg_and_slope(levels, is_bid: bool) -> Tuple[float, float]:
                    if not levels:
                        return 0.0, 0.0
                    ds: List[float] = []
                    vs: List[float] = []
                    for px, sz in levels:
                        if sz <= 0:
                            continue
                        if is_bid:
                            if px > micro:
                                continue
                            d_bp = (micro - px) / micro * 1e4
                        else:
                            if px < micro:
                                continue
                            d_bp = (px - micro) / micro * 1e4
                        ds.append(float(d_bp))
                        vs.append(float(sz))
                    if not vs:
                        return 0.0, 0.0
                    tot_v = sum(vs)
                    avg_dist = sum(v * d for v, d in zip(vs, ds)) / max(tot_v, 1e-9)
                    mean_d = sum(ds) / len(ds)
                    mean_v = tot_v / len(vs)
                    num = den = 0.0
                    for d, v in zip(ds, vs):
                        dd = d - mean_d
                        dv = v - mean_v
                        num += dd * dv
                        den += dd * dd
                    slope = num / den if den > 0 else 0.0
                    return float(avg_dist), float(slope)

                avg_dist_bid_bp, slope_bid = _avg_and_slope(bids_top, True)
                avg_dist_ask_bp, slope_ask = _avg_and_slope(asks_top, False)

            # 3. Liquidity walls: biggest order within ~25 bps of micro
            wall_bid_size = 0.0
            wall_bid_dist_bp = 0.0
            wall_ask_size = 0.0
            wall_ask_dist_bp = 0.0
            wall_imbalance = 0.0

            if micro and micro > 0.0:
                max_bid_sz = 0.0
                for px, sz in self.book.bids.items():
                    if sz <= 0 or px > micro:
                        continue
                    dist_bp = (micro - px) / micro * 1e4
                    if 0.0 < dist_bp <= 25.0 and sz > max_bid_sz:
                        max_bid_sz = sz
                        wall_bid_size = float(sz)
                        wall_bid_dist_bp = float(dist_bp)

                max_ask_sz = 0.0
                for px, sz in self.book.asks.items():
                    if sz <= 0 or px < micro:
                        continue
                    dist_bp = (px - micro) / micro * 1e4
                    if 0.0 < dist_bp <= 25.0 and sz > max_ask_sz:
                        max_ask_sz = sz
                        wall_ask_size = float(sz)
                        wall_ask_dist_bp = float(dist_bp)

                denom_wall = wall_bid_size + wall_ask_size
                wall_imbalance = (
                    (wall_bid_size - wall_ask_size) / denom_wall
                    if denom_wall > 0
                    else 0.0
                )

            # 4. Dynamics: depth & quote behaviour
            depth_imb_1bp_diff = 0.0
            net_add_bid_1bp_1s = 0.0
            net_add_ask_1bp_1s = 0.0
            n_bid_improve_1s = n_bid_worsen_1s = 0
            n_ask_improve_1s = n_ask_worsen_1s = 0
            n_spread_tighten_1s = n_spread_widen_1s = 0

            # depth dynamics (±1bp band)
            d1 = depth_bands[1.0]
            cur_bid_1bp = d1["bid"]
            cur_ask_1bp = d1["ask"]
            cur_imb_1bp = d1["imb"]

            delta_bid_1bp = cur_bid_1bp - self._last_depth_bid_1bp
            delta_ask_1bp = cur_ask_1bp - self._last_depth_ask_1bp
            depth_imb_1bp_diff = cur_imb_1bp - self._last_depth_imb_1bp

            if now_s > 0.0:
                self._depth_changes_1bp.append((now_s, delta_bid_1bp, delta_ask_1bp))
                cut = now_s - 1.0
                while self._depth_changes_1bp and self._depth_changes_1bp[0][0] < cut:
                    self._depth_changes_1bp.popleft()
                net_add_bid_1bp_1s = float(
                    sum(db for _, db, _ in self._depth_changes_1bp)
                )
                net_add_ask_1bp_1s = float(
                    sum(da for _, _, da in self._depth_changes_1bp)
                )

            self._last_depth_bid_1bp = cur_bid_1bp
            self._last_depth_ask_1bp = cur_ask_1bp
            self._last_depth_imb_1bp = cur_imb_1bp

            # quote event counts (best bid/ask moves, spread changes)
            bid_imp = bid_worsen = 0
            ask_imp = ask_worsen = 0
            spread_tighten = spread_widen = 0

            prev_bid = self._last_best_bid
            prev_ask = self._last_best_ask

            if prev_bid is not None and best_bid is not None:
                if best_bid > prev_bid:
                    bid_imp = 1
                elif best_bid < prev_bid:
                    bid_worsen = 1

            if prev_ask is not None and best_ask is not None:
                if best_ask < prev_ask:
                    ask_imp = 1
                elif best_ask > prev_ask:
                    ask_worsen = 1

            prev_spread = (
                prev_ask - prev_bid
                if (prev_bid is not None and prev_ask is not None)
                else None
            )
            if prev_spread is not None and spread_abs > 0:
                if spread_abs < prev_spread:
                    spread_tighten = 1
                elif spread_abs > prev_spread:
                    spread_widen = 1

            if now_s > 0.0:
                self._quote_events_1s.append(
                    (now_s, bid_imp, bid_worsen, ask_imp, ask_worsen, spread_tighten, spread_widen)
                )
                cut_q = now_s - 1.0
                while self._quote_events_1s and self._quote_events_1s[0][0] < cut_q:
                    self._quote_events_1s.popleft()

                for _, bi, bw, ai, aw, st, sw in self._quote_events_1s:
                    n_bid_improve_1s += bi
                    n_bid_worsen_1s += bw
                    n_ask_improve_1s += ai
                    n_ask_worsen_1s += aw
                    n_spread_tighten_1s += st
                    n_spread_widen_1s += sw

            # update last best prices / micro for next iteration
            self._last_best_bid = best_bid
            self._last_best_ask = best_ask
            self._last_micro = micro

            # push all L2 features into hub
            hub.set_cb_l2_features(
                self.product,
                micro=micro,
                best_bid=best_bid,
                best_ask=best_ask,
                best_bid_sz=best_bid_sz,
                best_ask_sz=best_ask_sz,
                spread_abs=spread_abs,
                spread_bp=spread_bp,
                depth_bands=depth_bands,
                near_far_ratio_bid=near_far_ratio_bid,
                near_far_ratio_ask=near_far_ratio_ask,
                avg_dist_bid_bp=avg_dist_bid_bp,
                avg_dist_ask_bp=avg_dist_ask_bp,
                slope_bid=slope_bid,
                slope_ask=slope_ask,
                wall_bid_size=wall_bid_size,
                wall_bid_dist_bp=wall_bid_dist_bp,
                wall_ask_size=wall_ask_size,
                wall_ask_dist_bp=wall_ask_dist_bp,
                wall_imbalance=wall_imbalance,
                depth_imb_1bp_diff=depth_imb_1bp_diff,
                net_add_bid_1bp_1s=net_add_bid_1bp_1s,
                net_add_ask_1bp_1s=net_add_ask_1bp_1s,
                n_bid_improve_1s=n_bid_improve_1s,
                n_bid_worsen_1s=n_bid_worsen_1s,
                n_ask_improve_1s=n_ask_improve_1s,
                n_ask_worsen_1s=n_ask_worsen_1s,
                n_spread_tighten_1s=n_spread_tighten_1s,
                n_spread_widen_1s=n_spread_widen_1s,
            )

            if hasattr(hub, "mark_updated"):
                hub.mark_updated()

        async def _loop(self):
            while True:
                ts, seq, ev = await self.q.get()
                etype = ev.get("type")
                ups = ev.get("updates") or []
                try:
                    # Apply book updates for *every* event (no drops)
                    if etype == "snapshot":
                        self.book.apply_snapshot(ups)
                    elif etype == "update":
                        self.book.apply_update(ups)

                    # Update last_seq robustly (seq might be str from some feeds)
                    try:
                        self.book.last_seq = int(seq)
                    except Exception:
                        pass

                    # Fast path: mark dirty and remember latest ts
                    self._last_ts = ts
                    self._dirty = True

                    # Interval gating: only compute heavy features occasionally
                    now_s = _as_epoch_s(ts) if ts else 0.0
                    if self._dirty and now_s > 0.0:
                        if (self._last_flush_s is None) or (now_s - self._last_flush_s >= self._interval_s):
                            self._compute_and_push_features(ts)
                            self._last_flush_s = now_s
                            self._dirty = False

                except Exception as e:
                    jlog(logging.ERROR, "cb_l2_apply_error", product=self.product, error=str(e))
                finally:
                    self.q.task_done()

    l2_processors: Dict[str, L2Processor] = {}

    channels = ["heartbeats", "ticker", "level2", "market_trades"]

    while True:
        try:
            async with websockets.connect(
                COINBASE_WS_URL,
                ping_interval=25,
                ping_timeout=25,
                max_size=16 * 1024 * 1024,
                write_limit=16 * 1024 * 1024,
            ) as ws:
                jlog(logging.INFO, "cb_ws_open", url=COINBASE_WS_URL, products=",".join(products))
                for ch in channels:
                    sub = {"type": "subscribe", "channel": ch}
                    if ch not in ("heartbeats",):
                        sub["product_ids"] = products
                    await ws.send(json.dumps(sub))

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        jlog(logging.ERROR, "cb_parse_error")
                        continue

                    ch = msg.get("channel")
                    seq = msg.get("sequence_num")
                    ts  = msg.get("timestamp")
                    evs = msg.get("events", [])
                    if not isinstance(evs, list):
                        continue

                    if ch == "heartbeats":
                        for ev in evs:
                            if write_files:
                                asyncio.create_task(
                                    writers.write("ALL", "heartbeats",
                                                  {"channel": ch, "timestamp": ts, "sequence_num": seq, "event": ev})
                                )

                    elif ch == "market_trades":
                        for ev in evs:
                            for tr in ev.get("trades") or []:
                                pid = (tr.get("product_id") or "ALL").upper()
                                if write_files:
                                    asyncio.create_task(
                                        writers.write(pid, "market_trades",
                                                      {"channel": ch, "timestamp": ts, "sequence_num": seq, "trade": tr})
                                    )
                                # update hub rolling flows (1s/3s) using top-level Coinbase timestamp
                                try:
                                    side = str(tr.get("side", "")).upper()
                                    size = float(tr.get("size"))
                                    px = tr.get("price")
                                    price_f = float(px) if px is not None else None
                                    hub.register_cb_trade(pid, side=side, size=size, exch_ts=ts, price=price_f)
                                    hub.set_cb_server_ts(pid, ts)
                                except Exception:
                                    pass

                    elif ch == "ticker":
                        for ev in evs:
                            for tk in ev.get("tickers") or []:
                                pid = (tk.get("product_id") or "ALL").upper()
                                if write_files:
                                    asyncio.create_task(
                                        writers.write(pid, "ticker",
                                                      {"channel": ch, "timestamp": ts, "sequence_num": seq, "ticker": tk})
                                    )
                                # microprice from best bid/ask & quantities — guard for EWMA
                                micro_val = None
                                try:
                                    bid = float(tk.get("best_bid"))
                                    ask = float(tk.get("best_ask"))
                                    q_bid = float(tk.get("best_bid_quantity") or 0.0)
                                    q_ask = float(tk.get("best_ask_quantity") or 0.0)
                                    denom = q_bid + q_ask
                                    if denom > 0 and bid > 0 and ask > 0:
                                        micro_val = (ask * q_bid + bid * q_ask) / denom
                                    elif bid > 0 and ask > 0:
                                        micro_val = 0.5 * (bid + ask)
                                except Exception:
                                    pass
                                if micro_val is None:
                                    try:
                                        micro_val = float(tk.get("price"))
                                    except Exception:
                                        micro_val = None

                                if micro_val is not None and math.isfinite(micro_val):
                                    hub.set_cb_microprice(pid, float(micro_val))
                                    # keep Coinbase *server* ts and update EWMA/jump off microprice
                                    hub.set_cb_server_ts(pid, ts)
                                    try:
                                        hub.update_cb_ewma(pid, float(micro_val), ts)
                                    except Exception:
                                        pass
                                    if hasattr(hub, "mark_updated"):
                                        hub.mark_updated()                                

                    elif ch in ("l2_data", "level2"):
                        for ev in evs:
                            pid = (ev.get("product_id") or "ALL").upper()
                            if write_files:
                                asyncio.create_task(
                                    writers.write(pid, "l2_data",
                                                  {"channel": "l2_data", "timestamp": ts, "sequence_num": seq, "level2": ev})
                                )
                            if pid not in l2_processors:
                                l2_processors[pid] = L2Processor(pid)
                            l2_processors[pid].submit_nowait(ts, int(seq) if isinstance(seq, int) else -1, ev)

        except Exception as e:
            jlog(logging.ERROR, "cb_ws_error", error=str(e))
            await asyncio.sleep(1.0)

# ----------------------------- Polymarket slug refresher -----------------------------
SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
def MARKET_BY_SLUG(s: str) -> str:
    from urllib.parse import quote
    return f"https://gamma-api.polymarket.com/markets/slug/{quote(s)}"

async def _fetch_json(session: aiohttp.ClientSession, url: str):
    try:
        async with session.get(url, timeout=15) as r:
            txt = await r.text()
            try:
                body = json.loads(txt)
            except Exception:
                body = None
            return r.ok, (body if body is not None else txt)
    except Exception as e:
        return False, str(e)

async def _search(session: aiohttp.ClientSession, q: str):
    from urllib.parse import quote
    url = f"{SEARCH_URL}?q={quote(q)}&limit_per_type=200"
    return await _fetch_json(session, url)

async def run_slug_refresher() -> None:
    bases = ["btc", "eth", "xrp", "sol"]
    while True:
        try:
            now = _now_utc_s()
            B = _next_15m_boundary_s(now)
            jlog(logging.INFO, "slug_refresh_begin", boundary=B)
            slugs = set()
            async with aiohttp.ClientSession() as session:
                for q in ["updown-15m", "updown 15m", "updown-15m-", "updown-15m"]:
                    ok, body = await _search(session, q)
                    if ok and isinstance(body, dict):
                        for m in body.get("markets", []) or []:
                            s = str(m.get("slug", ""))
                            if "updown-15m" in s:
                                slugs.add(s)
                if not slugs:
                    for q in ["updown", "15m"]:
                        ok, body = await _search(session, q)
                        if ok and isinstance(body, dict):
                            for m in body.get("markets", []) or []:
                                s = str(m.get("slug", ""))
                                if "updown-15m" in s:
                                    slugs.add(s)
                center = B
                for b in bases:
                    base_slug = f"{b}-updown-15m"
                    for off in range(-1, 3):
                        pe = center + off * WINDOW_SEC
                        s = f"{base_slug}-{pe}"
                        if s in slugs:
                            continue
                        ok, body = await _fetch_json(session, MARKET_BY_SLUG(s))
                        if ok and isinstance(body, dict):
                            slugs.add(s)
                            jlog(logging.INFO, "slug_probe_discovered", slug=s)

                for s in sorted(slugs):
                    ok, body = await _fetch_json(session, MARKET_BY_SLUG(s))
                    if ok and isinstance(body, dict):
                        path = os.path.join(TEMP_DIR, f"{s}.json")
                        try:
                            with open(path, "w", encoding="utf-8") as f:
                                json.dump(body, f, indent=2)
                            jlog(logging.INFO, "slug_saved", slug=s, path=path)
                        except Exception as e:
                            jlog(logging.ERROR, "slug_save_error", slug=s, error=str(e))
            jlog(logging.INFO, "slug_refresh_end", count=len(slugs))
        except Exception as e:
            jlog(logging.ERROR, "slug_refresh_error", error=str(e))
        now = _now_utc_s()
        next_q = _next_15m_boundary_s(now) + WINDOW_SEC
        await asyncio.sleep(max(30, next_q - now))

# ----------------------------- Polymarket market stream (CLOB) -----------------------------
def _parse_maybe_array(v: Any) -> List[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        try:
            arr = json.loads(v)
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except Exception:
            pass
        return [s.strip() for s in v.split(",") if s.strip()]
    return []

async def _read_local_market_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(TEMP_DIR, f"{slug}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        clobs = _parse_maybe_array(m.get("clobTokenIds") or m.get("clob_token_ids"))
        shorts = _parse_maybe_array(m.get("shortOutcomes") or m.get("short_outcomes"))
        if not clobs:
            return None
        return {
            "slug": str(m.get("slug", slug)),
            "clobTokenIds": clobs,
            "shortOutcomes": shorts,
            "condition": m.get("conditionId") or m.get("conditionID") or m.get("condition"),
            "startIso": m.get("startDateIso") or m.get("startDate"),
            "endIso": m.get("endDateIso") or m.get("endDate"),
        }
    except Exception:
        return None

def _slug_for(sym: str, end_ts: int) -> str:
    return f"{sym}-updown-15m-{end_ts}"

async def run_polymarket_market_stream(
    hub: StateHub,
    symbols: List[str] = ["btc", "eth", "xrp", "sol"],
    on_book_event: Optional[
        Callable[[str, List[Tuple[float, float]], List[Tuple[float, float]], Optional[int]], None]
    ] = None,
) -> None:
    """Subscribe to END=B-900; at (B-5s) switch to END=B.
    Keep ONE order book per base (BTC/ETH/SOL/XRP), storing the FULL book (all levels).
    Features per base: spread, depth-imbalance @ N∈{1,3,5}, 3s trade-imbalance (all using server timestamps).
    """

    # ---------- tiny helpers ----------
    def _infer_base_from_slug(slug: str) -> Optional[str]:
        s = (slug or "").lower()
        for b in symbols:
            if s.startswith(f"{b}-") or f"-{b}-" in s:
                return b.upper()
        return None

    def _canon(px: float) -> float:
        # avoid float-equality mismatches across updates
        return float(round(px, 6))

    def _levels_from_booklike(book: Dict[str, Any]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Parse bids/asks arrays (each item may be {'price','size'} or [price,size]).
        Return full L2 lists sorted: bids DESC, asks ASC (no level limit)."""
        def _to_levels(arr, side):
            lvls = []
            if isinstance(arr, list):
                for it in arr:
                    try:
                        if isinstance(it, dict):
                            px = _canon(float(it.get("price")))
                            sz = float(it.get("size") or it.get("amount") or it.get("qty") or it.get("quantity"))
                        elif isinstance(it, (list, tuple)) and len(it) >= 2:
                            px, sz = _canon(float(it[0])), float(it[1])
                        else:
                            continue
                        if math.isfinite(px) and math.isfinite(sz):
                            lvls.append((px, sz))
                    except Exception:
                        continue
            lvls.sort(key=lambda x: x[0], reverse=(side == "BUY"))
            return lvls

        bids = _to_levels(book.get("bids") or book.get("buy") or book.get("bid") or [], "BUY")
        asks = _to_levels(book.get("asks") or book.get("sell") or book.get("ask") or [], "SELL")
        return bids, asks

    def _apply_price_change(side: str, price: float, size: float,
                            bids: List[Tuple[float, float]],
                            asks: List[Tuple[float, float]]) -> None:
        """price_change sets the aggregate size at `price`; if size==0 remove the level. Keep full book."""
        price = _canon(price)
        lvls = bids if side == "BUY" else asks
        i = next((i for i, (px, _) in enumerate(lvls) if px == price), -1)
        if size <= 0.0:
            if i >= 0:
                lvls.pop(i)
        else:
            if i >= 0:
                lvls[i] = (price, size)
            else:
                lvls.append((price, size))
        # keep sorted
        lvls.sort(key=lambda x: x[0], reverse=(side == "BUY"))

    def _spread_and_imbalances(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Tuple[Optional[float], float, float, float]:
        """Spread and depth-imb @ N=1,3,5. If a side is empty, spread=None; imbalances well-defined."""
        spread = None
        if bids and asks:
            spread = asks[0][0] - bids[0][0]

        def _imb(N: int) -> float:
            bsum = sum(q for _, q in bids[:N])
            asum = sum(q for _, q in asks[:N])
            denom = bsum + asum
            return (bsum - asum) / denom if denom > 0 else 0.0

        return spread, _imb(1), _imb(3), _imb(5)

    # ---------- hub state (single OB per base) ----------
    if not hasattr(hub, "pm_base_state"):
        hub.pm_base_state = {}   # base -> {bids, asks, spread, imb1, imb3, imb5, trade_imb_3s, ts_server_ms}
    if not hasattr(hub, "_pm_trades"):
        hub._pm_trades = defaultdict(lambda: deque())  # base -> deque[(ts_ms, buy, sell)]

    # ---------- YES-token resolution for each base/end ----------
    async def resolve_assets_for_end(end_ts: int):
        def _is_yes_name(x: Any) -> bool:
            try:
                return str(x).strip().lower() in {"yes", "y", "true"}
            except Exception:
                return False

        def _maybe_get_token_id(d: Dict[str, Any]) -> Optional[str]:
            for k in (
                "clobTokenId", "clob_token_id",
                "instrumentId", "instrument_id",
                "asset_id", "id", "tokenId", "token_id"
            ):
                v = d.get(k)
                if v is not None:
                    return str(v)
            tok = d.get("token")
            if isinstance(tok, dict):
                for k in (
                    "clobTokenId", "clob_token_id",
                    "instrumentId", "instrument_id",
                    "asset_id", "id", "tokenId", "token_id"
                ):
                    v = tok.get(k)
                    if v is not None:
                        return str(v)
            return None

        ids: List[str] = []
        asset_to_slug: Dict[str, str] = {}
        cond_to_slug: Dict[str, str] = {}
        yes_for_slug: Dict[str, str] = {}

        for s in symbols:
            slug = _slug_for(s, end_ts)
            m = await _read_local_market_by_slug(slug)
            if not m:
                jlog(logging.WARNING, "clob_local_missing", slug=slug, hint="ensure temp/{slug}.json exists")
                continue

            yes_id: Optional[str] = None

            for key in ("outcomes", "tokens", "contracts", "assets", "markets"):
                arr = m.get(key)
                if isinstance(arr, list):
                    for it in arr:
                        if not isinstance(it, dict):
                            continue
                        name = (
                            it.get("name")
                            or it.get("outcome")
                            or it.get("label")
                            or it.get("side")
                            or (it.get("token") or {}).get("ticker")
                            or (it.get("token") or {}).get("symbol")
                        )
                        if _is_yes_name(name):
                            maybe = _maybe_get_token_id(it)
                            if maybe:
                                yes_id = maybe
                                break
                if yes_id:
                    break

            if not yes_id:
                for k in ("yesInstrumentId", "yes_instrument_id", "yesTokenId", "yes_token_id"):
                    v = m.get(k)
                    if v is not None:
                        yes_id = str(v)
                        break

            if not yes_id and isinstance(m.get("clobTokenIds"), list) and len(m["clobTokenIds"]) >= 1:
                yes_id = str(m["clobTokenIds"][0])
                jlog(logging.INFO, "assume_yes_first_token", slug=slug)

            if yes_id:
                asset_to_slug[yes_id] = slug
                yes_for_slug[slug] = yes_id
                ids.append(yes_id)
            else:
                jlog(logging.WARNING, "yes_token_not_found", slug=slug)

            if m.get("condition"):
                cond_to_slug[str(m["condition"])] = slug

        return ids, asset_to_slug, cond_to_slug, yes_for_slug

    # ---------- routing & logging ----------
    url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    reader_task: Optional[asyncio.Task] = None
    ws: Optional[websockets.WebSocketClientProtocol] = None
    boundary_end = _next_15m_boundary_s()
    asset_to_slug: Dict[str, str] = {}
    cond_to_slug: Dict[str, str] = {}
    yes_for_slug: Dict[str, str] = {}  # slug -> YES token id

    def route_slug(obj: Dict[str, Any]) -> str:
        a = obj.get("asset_id")
        c = obj.get("market")
        if c and c in cond_to_slug:
            return cond_to_slug[c]
        if a and a in asset_to_slug:
            return asset_to_slug[a]
        return "polymarket"

    async def write_msg(slug: str, obj: Dict[str, Any]) -> None:
        path = os.path.join(MARKET_DIR, f"{slug}.jsonl")
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    def _best_px(
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> Tuple[Optional[float], Optional[float]]:
        bb = bids[0][0] if bids else None
        ba = asks[0][0] if asks else None
        return bb, ba

    def _prev_best_px_from_state(st: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        # Prefer explicit keys if present, else derive from stored bids/asks.
        pb = st.get("best_bid_px", None)
        pa = st.get("best_ask_px", None)

        if pb is None:
            try:
                bids0 = st.get("bids") or []
                pb = bids0[0][0] if bids0 else None
            except Exception:
                pb = None

        if pa is None:
            try:
                asks0 = st.get("asks") or []
                pa = asks0[0][0] if asks0 else None
            except Exception:
                pa = None

        return pb, pa

    def _recompute_base(
        baseU: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        ts_ms: Optional[int],
    ) -> None:
        """
        Always refresh hub.pm_base_state (keeps state accurate + stale-ts watchdog happy),
        but only EMIT an update (mark_updated + on_book_event) iff best bid/ask PRICE changes.
        """
        # --- compute new TOB prices ---
        new_bb, new_ba = _best_px(bids, asks)

        # --- prior state / prior TOB prices ---
        st_prev = hub.pm_base_state.get(baseU) or {}
        prev_bb, prev_ba = _prev_best_px_from_state(st_prev)

        tob_price_changed = (new_bb != prev_bb) or (new_ba != prev_ba)

        # --- derived features (same as before) ---
        spread, imb1, imb3, imb5 = _spread_and_imbalances(bids, asks)

        dq = hub._pm_trades.setdefault(baseU, deque())
        if ts_ms is not None:
            cutoff = ts_ms - 3000
            while dq and dq[0][0] < cutoff:
                dq.popleft()
        buy3 = sum(x[1] for x in dq)
        sell3 = sum(x[2] for x in dq)
        trade_imb_3s = ((buy3 - sell3) / (buy3 + sell3)) if (buy3 + sell3) > 0 else 0.0

        # --- timestamp ---
        last_ts = st_prev.get("ts_server_ms")
        latest_ts = ts_ms if ts_ms is not None else (int(last_ts) if isinstance(last_ts, (int, float)) else None)

        # --- ALWAYS update shared state (even if no TOB change) ---
        hub.pm_base_state[baseU] = {
            "bids": bids,
            "asks": asks,
            "spread": spread,
            "imb1": imb1,
            "imb3": imb3,
            "imb5": imb5,
            "trade_imb_3s": trade_imb_3s,
            "ts_server_ms": latest_ts,
            # store explicit TOB prices for quick comparisons
            "best_bid_px": new_bb,
            "best_ask_px": new_ba,
        }

        # --- ONLY EMIT update when TOB PRICE changes ---
        if tob_price_changed:
            hub.mark_updated()

            if on_book_event is not None:
                try:
                    on_book_event(baseU, bids, asks, latest_ts)
                except Exception as e:
                    jlog(logging.ERROR, "pm_on_book_event_error", base=baseU, error=str(e))

    def _handle_book(slug: str, msg: Dict[str, Any]) -> None:
        book = msg.get("book") or msg.get("order_book") or msg
        bids, asks = _levels_from_booklike(book)

        baseU = _infer_base_from_slug(slug)
        if not baseU:
            return

        ts_ms = int(msg.get("timestamp")) if str(msg.get("timestamp") or "").isdigit() else None
        _recompute_base(baseU, bids, asks, ts_ms)

    def _handle_price_change(slug: str, msg: Dict[str, Any]) -> None:
        baseU = _infer_base_from_slug(slug)
        if not baseU:
            return

        yes_id = yes_for_slug.get(slug)
        if not yes_id:
            return

        st = hub.pm_base_state.get(baseU)
        bids = list(st.get("bids", [])) if st else []
        asks = list(st.get("asks", [])) if st else []

        ts_ms = int(msg.get("timestamp")) if str(msg.get("timestamp") or "").isdigit() else None

        for ch in msg.get("price_changes", []):
            try:
                if str(ch.get("asset_id")) != yes_id:
                    continue
                side = str(ch.get("side", "")).upper()
                px = float(ch["price"])
                sz = float(ch["size"])
            except Exception:
                continue

            if side not in ("BUY", "SELL"):
                continue

            _apply_price_change(side, px, sz, bids, asks)

        # _recompute_base will gate emission by TOB PRICE change
        _recompute_base(baseU, bids, asks, ts_ms)

    def _handle_last_trade_price(slug: str, msg: Dict[str, Any]) -> None:
        baseU = _infer_base_from_slug(slug)
        if not baseU:
            return
        try:
            side = str(msg.get("side", "")).upper()
            size = float(msg.get("size", 0.0))
            ts_ms = int(msg.get("timestamp"))
        except Exception:
            return

        dq = hub._pm_trades.setdefault(baseU, deque())
        if side == "BUY":
            dq.append((ts_ms, size, 0.0))
        elif side == "SELL":
            dq.append((ts_ms, 0.0, size))

        cutoff = ts_ms - 3000
        while dq and dq[0][0] < cutoff:
            dq.popleft()

        # Keep state consistent, but emission is still gated by TOB PRICE change.
        st = hub.pm_base_state.get(baseU) or {}
        _recompute_base(baseU, list(st.get("bids", [])), list(st.get("asks", [])), ts_ms)

    # ---------- helper to get freshest server ts across bases ----------
    def _latest_pm_ts_ms() -> Optional[int]:
        best = None
        for st in (hub.pm_base_state or {}).values():
            ts_ms = st.get("ts_server_ms")
            if isinstance(ts_ms, (int, float)):
                if best is None or ts_ms > best:
                    best = int(ts_ms)
        return best

    # ---------- subscribe & reader with auto-resubscribe + stale-ts watchdog ----------
    async def subscribe_to_end(end_ts: int, boundary_for_reader: int) -> bool:
        nonlocal ws, reader_task, asset_to_slug, cond_to_slug, yes_for_slug

        ids, a2s, c2s, yfs = await resolve_assets_for_end(end_ts)
        if not ids:
            jlog(logging.ERROR, "clob_no_assets_for_end", end=end_ts)
            return False
        asset_to_slug, cond_to_slug, yes_for_slug = a2s, c2s, yfs
        hub.clob_current_end = end_ts
        hub.mark_updated()

        # cancel previous reader (if any)
        try:
            if reader_task and not reader_task.done():
                reader_task.cancel()
                try:
                    await reader_task
                except asyncio.CancelledError:
                    pass
        except Exception:
            pass

        # close previous socket (if any)
        try:
            if ws:
                await ws.close()
        except Exception:
            pass
        ws = None

        async def _reader():
            nonlocal ws
            # Outer loop: (re)connect if needed while we are still >10s from boundary
            while True:
                secs_to_boundary = boundary_for_reader - time.time()
                if secs_to_boundary <= 10.0:
                    jlog(
                        logging.INFO,
                        "clob_reader_stop_near_boundary",
                        end=end_ts,
                        secs_to_boundary=secs_to_boundary,
                    )
                    break

                # Attempt to connect
                try:
                    ws = await websockets.connect(
                        url,
                        ping_interval=10,
                        ping_timeout=20,
                    )
                    await ws.send(json.dumps({"type": "market", "assets_ids": ids}))
                    jlog(logging.INFO, "clob_open", end=end_ts, assets=len(ids))
                except Exception as e:
                    jlog(
                        logging.ERROR,
                        "clob_open_error",
                        end=end_ts,
                        error=str(e),
                        phase="reader_connect",
                    )
                    ws = None
                    await asyncio.sleep(1.0)
                    continue

                # Inner loop: receive with timeout, detect stale timestamps, or handle disconnect
                try:
                    assert ws is not None
                    while True:
                        secs_to_boundary = boundary_for_reader - time.time()
                        if secs_to_boundary <= 10.0:
                            jlog(
                                logging.INFO,
                                "clob_reader_stop_near_boundary_inner",
                                end=end_ts,
                                secs_to_boundary=secs_to_boundary,
                            )
                            raise asyncio.CancelledError  # break out cleanly

                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            # No message for 5s -> check staleness vs server timestamps
                            now_ms = int(time.time() * 1000)
                            last_ts = _latest_pm_ts_ms()
                            if last_ts is not None:
                                age_ms = now_ms - last_ts
                                if age_ms > 10_000 and secs_to_boundary > 10.0:
                                    jlog(
                                        logging.WARNING,
                                        "clob_stale_ts_resubscribe",
                                        end=end_ts,
                                        age_ms=age_ms,
                                        max_age_ms=10_000,
                                    )
                                    # Break out to outer loop to reconnect
                                    break
                            continue  # keep waiting on same connection

                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            jlog(
                                logging.WARNING,
                                "clob_reader_recv_error",
                                end=end_ts,
                                error=str(e),
                            )
                            # Break to reconnect, if time allows
                            break

                        # Got data
                        t = str(raw)
                        if t in ("PING", "PONG"):
                            continue
                        try:
                            obj = json.loads(t)
                        except Exception:
                            continue

                        msgs = obj if isinstance(obj, list) else [obj] if isinstance(obj, dict) else []
                        for msg in (m for m in msgs if isinstance(m, dict)):
                            slug = route_slug(msg)
                            # optional persist:
                            # await write_msg(slug, msg)

                            ev = str(msg.get("event_type") or "").lower()
                            if ev == "book" or ("bids" in msg and "asks" in msg):
                                _handle_book(slug, msg)
                                continue
                            if ev == "price_change" or "price_changes" in msg:
                                _handle_price_change(slug, msg)
                                continue
                            if ev == "last_trade_price":
                                _handle_last_trade_price(slug, msg)
                                continue

                except asyncio.CancelledError:
                    # graceful shutdown due to boundary or external cancel
                    break
                finally:
                    # Ensure socket is closed before possibly reconnecting
                    try:
                        if ws:
                            await ws.close()
                    except Exception:
                        pass
                    ws = None

                # We exited inner loop due to stale data or recv error; decide whether to reconnect
                secs_to_boundary = boundary_for_reader - time.time()
                if secs_to_boundary <= 10.0:
                    jlog(
                        logging.INFO,
                        "clob_reader_no_reconnect_near_boundary",
                        end=end_ts,
                        secs_to_boundary=secs_to_boundary,
                    )
                    break

                jlog(logging.INFO, "clob_reader_disconnected_reconnect", end=end_ts, secs_to_boundary=secs_to_boundary)
                await asyncio.sleep(0.5)

        reader_task = asyncio.create_task(_reader())
        return True

    # initial subscribe: END = B - 900 (boundary_end = B)
    boundary_end = _next_15m_boundary_s()
    ok = await subscribe_to_end(boundary_end - WINDOW_SEC, boundary_for_reader=boundary_end)
    if not ok:
        jlog(logging.ERROR, "clob_initial_subscribe_failed", end=boundary_end - WINDOW_SEC)

    # switch at (B - 5s) to END=B, then roll boundary_end forward by 900s
    while True:
        switch_ms = boundary_end * 1000 - SWITCH_LEAD_MS
        now_ms = int(time.time() * 1000)
        delay = max(0.0, (switch_ms - now_ms) / 1000.0)
        await asyncio.sleep(delay)
        next_end = boundary_end
        jlog(logging.INFO, "clob_switch", next_end=next_end)
        # For the new end, use the *next* boundary as the reader's boundary_for_reader
        # so the stream keeps running until ~10s before the NEXT switch.
        boundary_for_reader = boundary_end + WINDOW_SEC
        await subscribe_to_end(next_end, boundary_for_reader=boundary_for_reader)
        boundary_end += WINDOW_SEC
