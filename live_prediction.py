#!/usr/bin/env python3
# live_prediction.py — store every tick; predict latest-only; log seq quantiles + seq quote score; IVQuoteEngine w/ 15m roll clear;
# push EVERYTHING (including seq/iv predictions + scores) into TradeEngine snapshot.

from __future__ import annotations

import os
import math
import time as _time
import asyncio
import logging
import signal
import secrets
from typing import Dict, Any, List, Optional, Tuple
from collections.abc import Mapping
from collections import deque

import torch

torch.set_num_threads(3)
torch.set_num_interop_threads(3)

import live_lib as live_mod
from live_lib import (
    setup_logging,
    SYMBOLS,
    load_checkpoint_for_symbol,
    LoadedModel,
    MinuteFeatureBuilder,
    SecRing,
    LoadedAndState,
    backfill_and_seed,
    minute_incremental_updater,
    stream_symbol,
    base_key,
)

from market_lib import (
    StateHub,
    run_slug_refresher,
    run_coinbase_recorder,
    run_polymarket_market_stream,
    run_pm_iv_calibrator,
    ChainlinkResolutionService,
    _next_15m_boundary_s,
)

from hub_debug import run_hub_debug_dumper
from trade import TradeEngine
import trade as _trade_mod

# train_seq live inference engine
from quote_seq import SeqQuoteEngine

# train_iv live inference engine (new version with clear_all_states / clear_base)
from quote_iv import IVQuoteEngine


# ---- Config ----
WARMUP_SECONDS = 5
MINUTE_STAGGER_SECONDS = 2

# Microprice change thresholds (absolute) to throttle ticks
MP_THRESH = {"BTC": 0.1, "ETH": 0.01, "SOL": 0.01, "XRP": 0.0001}

# Startup logging controls (quiet missing-microprice logs initially)
CB_MISSING_LOG_GRACE_SEC = float(os.environ.get("CB_MISSING_LOG_GRACE_SEC", "2.0"))
CB_MISSING_LOG_RATE_SEC = float(os.environ.get("CB_MISSING_LOG_RATE_SEC", "2.0"))

# Hard safety: require very recent model (Binance) prediction
MODEL_TS_MAX_AGE_SEC = 20.0

# Hard safety: require recent Polymarket book timestamp (server-side)
PM_TS_MAX_AGE_SEC = 10.0

# Heartbeat: ensure at least 1 emit per base per HEARTBEAT_SEC
HEARTBEAT_SEC = 0.32

# ---- Seq (train_seq) live prediction controls ----
ENABLE_SEQ = bool(int(os.environ.get("ENABLE_SEQ", "1")))
SEQ_RUN_DIR = os.environ.get("SEQ_RUN_DIR", "runs_seq_t")
SEQ_DATA_DIR = os.environ.get("SEQ_DATA_DIR", "data_seq32")
SEQ_CKPT = os.environ.get("SEQ_CKPT", "model_best.pt")
SEQ_PRED_MIN_INTERVAL_MS = int(os.environ.get("SEQ_PRED_MIN_INTERVAL_MS", "50"))

# ---- IV (train_iv) live prediction controls ----
ENABLE_IV = bool(int(os.environ.get("ENABLE_IV", "1")))
IV_RUN_DIR = os.environ.get("IV_RUN_DIR", "runs_iv_delta")
IV_DATA_DIR = os.environ.get("IV_DATA_DIR", "data_iv64")
IV_PRED_MIN_INTERVAL_MS = int(os.environ.get("IV_PRED_MIN_INTERVAL_MS", "50"))

# ---- IV roll clearing ----
# Clear IV state ONLY at contract roll boundary (15m switch).
IV_ROLL_CLEAR_EPS_SEC = float(os.environ.get("IV_ROLL_CLEAR_EPS_SEC", "1"))


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _parse_exch_sec(ex_time: Optional[str]) -> Optional[int]:
    """Parse 'YYYY-mm-ddTHH:MM:SSZ' to epoch seconds (Binance server time)."""
    if not ex_time:
        return None
    try:
        import datetime as _dt

        return int(
            _dt.datetime.strptime(ex_time, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=_dt.timezone.utc)
            .timestamp()
        )
    except Exception:
        return None


def _seq_quote_from_status(status: Dict[str, Any]) -> Dict[str, Any]:
    """
    quote_seq.py puts quote fields directly on `status`.

    Current quote_seq outputs (see quote_seq._compute_quote_signal):
      - quote_score          (float)
      - quote_delta_p        (float)
      - quote_one_sided      (bool)
      - quote_mode           (str)
      - quote_debug          (list[dict])  # per-horizon diagnostics incl z/sigma/q50

    Map into fields that live_prediction/trader expects.
    """
    score = _safe_float(status.get("quote_score")) or 0.0
    delta_p = _safe_float(status.get("quote_delta_p")) or 0.0
    one_sided = bool(status.get("quote_one_sided") or False)
    mode = status.get("quote_mode")
    mode_s = str(mode) if isinstance(mode, str) and mode else ("one_sided" if one_sided else "two_sided")

    if delta_p > 0:
        side = "buy"
    elif delta_p < 0:
        side = "sell"
    else:
        side = "flat"

    z_agg = 0.0
    mu_agg = 0.0
    sig_agg = 0.0
    w_sum = 0.0
    dbg = status.get("quote_debug")
    if isinstance(dbg, list) and dbg:
        active_entries = [e for e in dbg if isinstance(e, dict) and bool(e.get("active"))]
        entries = active_entries if active_entries else [e for e in dbg if isinstance(e, dict)]
        for e in entries:
            w = _safe_float(e.get("w"))
            z = _safe_float(e.get("z"))
            q50 = _safe_float(e.get("q50"))
            sig = _safe_float(e.get("sigma"))
            if w is None or w <= 0:
                continue
            if z is not None:
                z_agg += w * z
            if q50 is not None:
                mu_agg += w * q50
            if sig is not None:
                sig_agg += w * sig
            w_sum += w

        if w_sum > 0:
            z_agg /= w_sum
            mu_agg /= w_sum
            sig_agg /= w_sum

    return {
        "quote_score": float(score),
        "quote_delta_p": float(delta_p),
        "quote_skew": float(delta_p),
        "quote_one_sided": bool(one_sided),
        "quote_mode": mode_s,
        "quote_side": side,
        "quote_z": float(z_agg),
        "quote_mu": float(mu_agg),
        "quote_sigma": float(sig_agg),
    }


async def run_live_prediction() -> None:
    setup_logging()
    live_mod.jlog(logging.INFO, "service_boot", version="4.4.0-ivrollclear-newiv")

    hub = StateHub()
    bases = ["BTC", "ETH", "SOL", "XRP"]

    # ensure these exist; StateHub doesn't define them, but we want them in hub snapshot
    if not isinstance(getattr(hub, "seq_quote", None), dict):
        hub.seq_quote = {}  # type: ignore[attr-defined]
    if not isinstance(getattr(hub, "iv_pred", None), dict):
        hub.iv_pred = {}  # type: ignore[attr-defined]

    # Background dump of hub for observability
    hub_dump_task = asyncio.create_task(
        run_hub_debug_dumper(
            hub,
            out_dir=os.path.join(os.getcwd(), "data", "debug"),
            interval_sec=1.0,
            rotate_mb=50,
        )
    )

    # IV calibrator (keep)
    asyncio.create_task(run_pm_iv_calibrator(hub, bases=bases, interval_s=0.1))

    trader = TradeEngine(temp_dir="temp", allowed_bases={"BTC"})

    # --------- jlog bridge: capture model (Binance) predictions into hub ----------
    _orig_jlog = live_mod.jlog

    def _get_fresh_model_state(h: StateHub, base: str, now_s: float) -> Optional[Dict[str, Any]]:
        baseU = base.upper()
        st = getattr(h, "model_preds", {}).get(baseU)
        if not isinstance(st, dict):
            return None
        exch_sec = st.get("exch_sec")
        if not isinstance(exch_sec, (int, float)):
            return None
        age = now_s - float(exch_sec)
        if age > MODEL_TS_MAX_AGE_SEC:
            try:
                h.model_preds.pop(baseU, None)
            except Exception:
                pass
            live_mod.jlog(
                logging.INFO,
                "model_prediction_stale_reset",
                base=baseU,
                exch_sec=float(exch_sec),
                age=float(age),
                max_age=MODEL_TS_MAX_AGE_SEC,
            )
            return None
        return st

    def _get_any_model_state(h: StateHub, base: str) -> Optional[Dict[str, Any]]:
        baseU = base.upper()
        st = getattr(h, "model_preds", {}).get(baseU)
        return st if isinstance(st, dict) else None

    def _jlog_bridge(level: int, event: str, **fields):
        _orig_jlog(level, event, **fields)

        if event == "prediction":
            try:
                sym = fields.get("symbol")
                iv = _safe_float(fields.get("iv"))
                df = _safe_float(fields.get("df"))
                exs = _parse_exch_sec(fields.get("ex_time"))
                if sym is None or iv is None or df is None:
                    return
                b = base_key(sym)  # e.g. "BTCUSDT" -> "BTC"
                hub.set_model_prediction(base=b, iv=float(iv), df=float(df), exch_sec=exs)
            except Exception as e:
                _orig_jlog(logging.ERROR, "jlog_bridge_error", error=str(e))

        elif event in ("binance_ws_dead", "binance_model_disconnected"):
            try:
                if hasattr(hub, "model_preds") and isinstance(hub.model_preds, dict):
                    hub.model_preds.clear()
                _orig_jlog(logging.WARNING, "model_prediction_reset_on_disconnect", reason=event)
            except Exception as e:
                _orig_jlog(logging.ERROR, "model_prediction_reset_error", error=str(e))

        elif event in ("binance_ws_open", "binance_model_reconnected"):
            _orig_jlog(logging.INFO, "model_prediction_stream_reconnected")

    live_mod.jlog = _jlog_bridge
    jlog = live_mod.jlog
    _trade_mod.jlog = live_mod.jlog

    # --------- Services ----------
    cl_svc = ChainlinkResolutionService(hub, bases=bases)
    await cl_svc.start()
    jlog(logging.INFO, "chainlink_resolution_started")

    cb_task = asyncio.create_task(
        run_coinbase_recorder(
            hub, products=["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"], write_files=False
        )
    )
    jlog(logging.INFO, "coinbase_recorder_started", products="BTC-USD,ETH-USD,SOL-USD,XRP-USD")

    # Trading queue stays last-only; seq/iv storage + prediction are separate paths.
    tick_queues: Dict[str, asyncio.Queue] = {b: asyncio.Queue(maxsize=1) for b in bases}
    base_locks: Dict[str, asyncio.Lock] = {b: asyncio.Lock() for b in bases}

    # --------- Utilities ----------
    def _read_micro_pair(h: StateHub, base: str) -> Tuple[Optional[float], Optional[float]]:
        latest = prev = None
        d_latest = getattr(h, "coinbase_microprice_latest", None)
        d_prev = getattr(h, "coinbase_microprice_prev", None)
        if isinstance(d_latest, dict):
            latest = _safe_float(d_latest.get(base))
        if isinstance(d_prev, dict):
            prev = _safe_float(d_prev.get(base))
        return latest, prev

    def _read_ts_fields(h: StateHub, base: str) -> Tuple[Optional[int], Optional[int]]:
        cb_ms = None
        dcb = getattr(h, "coinbase_ts_server_ms", None)
        if isinstance(dcb, dict):
            try:
                v = dcb.get(base)
                if v is not None:
                    cb_ms = int(v)
            except Exception:
                cb_ms = None

        pm_ms = None
        pmd = getattr(h, "pm_base_state", None)
        if isinstance(pmd, dict):
            x = pmd.get(base)
            if isinstance(x, dict):
                try:
                    v2 = x.get("ts_server_ms")
                    if v2 is not None:
                        pm_ms = int(v2)
                except Exception:
                    pm_ms = None
        return cb_ms, pm_ms

    # 1h ring for prev_close derivation
    _cb_hist: Dict[str, deque[Tuple[float, float]]] = {b: deque(maxlen=65535) for b in bases}

    def _cb_hist_add(base: str, px: float, ts_s: Optional[float] = None) -> None:
        t = float(ts_s if ts_s is not None else _time.time())
        dq = _cb_hist[base]
        dq.append((t, float(px)))
        cutoff = t - 3600.0
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _coinbase_prev_close(base: str, now_s: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        B = _next_15m_boundary_s(int(now_s))
        S = B - 900
        target = S - 1.1
        band_lo = S - 2.0
        band_hi = S - 1.0

        dq = _cb_hist[base]
        prev_close: Optional[float] = None
        prev_hi: Optional[float] = None
        prev_lo: Optional[float] = None

        for ts, px in dq:
            if band_lo <= ts <= band_hi:
                p = float(px)
                if prev_hi is None or p > prev_hi:
                    prev_hi = p
                if prev_lo is None or p < prev_lo:
                    prev_lo = p

        for ts, px in reversed(dq):
            if ts <= target:
                prev_close = float(px)
                break

        if prev_close is not None:
            if prev_hi is None:
                prev_hi = prev_close
            if prev_lo is None:
                prev_lo = prev_close

        return prev_close, prev_hi, prev_lo

    def _flow_net(entry) -> Optional[float]:
        if entry is None:
            return None
        try:
            v = getattr(entry, "net", None)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return float(v)
        except Exception:
            pass
        if isinstance(entry, Mapping):
            try:
                v = entry.get("net")
                return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None
            except Exception:
                return None
        return None

    def _cb_read(
        h: StateHub, base: str
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[int],
    ]:
        skew1 = skew2 = skew5 = skew10 = None
        d1 = getattr(h, "coinbase_mp_skew_bps1", None)
        if isinstance(d1, Mapping):
            skew1 = _safe_float(d1.get(base))
        d2 = getattr(h, "coinbase_mp_skew_bps2", None)
        if isinstance(d2, Mapping):
            skew2 = _safe_float(d2.get(base))
        d5 = getattr(h, "coinbase_mp_skew_bps5", None)
        if isinstance(d5, Mapping):
            skew5 = _safe_float(d5.get(base))
        d10 = getattr(h, "coinbase_mp_skew_bps10", None)
        if isinstance(d10, Mapping):
            skew10 = _safe_float(d10.get(base))

        f1 = f3 = f5 = None
        f1_all = getattr(h, "coinbase_flow_1s", None)
        if isinstance(f1_all, Mapping):
            f1 = _flow_net(f1_all.get(base))
        f3_all = getattr(h, "coinbase_flow_3s", None)
        if isinstance(f3_all, Mapping):
            f3 = _flow_net(f3_all.get(base))
        f5_all = getattr(h, "coinbase_flow_5s", None)
        if isinstance(f5_all, Mapping):
            f5 = _flow_net(f5_all.get(base))

        sig = None
        sd = getattr(h, "coinbase_sigma_ewma", None)
        if isinstance(sd, Mapping):
            sig = _safe_float(sd.get(base))

        jump: Optional[int] = None
        jd = getattr(h, "coinbase_jump_flag", None)
        if isinstance(jd, Mapping):
            jv = jd.get(base)
            if jv is not None:
                try:
                    jump = int(jv)
                except Exception:
                    jump = None

        return (skew1, skew2, skew5, skew10, f1, f3, f5, sig, jump)

    def _cb_l2_read(h: StateHub, base: str) -> Dict[str, Optional[float]]:
        attr_to_key = {
            "coinbase_tob_bid_px": "cb_tob_bid_px",
            "coinbase_tob_ask_px": "cb_tob_ask_px",
            "coinbase_tob_bid_qty": "cb_tob_bid_qty",
            "coinbase_tob_ask_qty": "cb_tob_ask_qty",
            "coinbase_spread_abs": "cb_spread_abs",
            "coinbase_spread_bp": "cb_spread_bp",
            "coinbase_depth_bid_1bp": "cb_depth_bid_1bp",
            "coinbase_depth_ask_1bp": "cb_depth_ask_1bp",
            "coinbase_depth_tot_1bp": "cb_depth_tot_1bp",
            "coinbase_depth_imb_1bp": "cb_depth_imb_1bp",
            "coinbase_depth_lr_1bp": "cb_depth_lr_1bp",
            "coinbase_depth_bid_2bp": "cb_depth_bid_2bp",
            "coinbase_depth_ask_2bp": "cb_depth_ask_2bp",
            "coinbase_depth_tot_2bp": "cb_depth_tot_2bp",
            "coinbase_depth_imb_2bp": "cb_depth_imb_2bp",
            "coinbase_depth_lr_2bp": "cb_depth_lr_2bp",
            "coinbase_depth_bid_5bp": "cb_depth_bid_5bp",
            "coinbase_depth_ask_5bp": "cb_depth_ask_5bp",
            "coinbase_depth_tot_5bp": "cb_depth_tot_5bp",
            "coinbase_depth_imb_5bp": "cb_depth_imb_5bp",
            "coinbase_depth_lr_5bp": "cb_depth_lr_5bp",
            "coinbase_depth_bid_10bp": "cb_depth_bid_10bp",
            "coinbase_depth_ask_10bp": "cb_depth_ask_10bp",
            "coinbase_depth_tot_10bp": "cb_depth_tot_10bp",
            "coinbase_depth_imb_10bp": "cb_depth_imb_10bp",
            "coinbase_depth_lr_10bp": "cb_depth_lr_10bp",
            "coinbase_depth_near_far_ratio_bid": "cb_depth_near_far_ratio_bid",
            "coinbase_depth_near_far_ratio_ask": "cb_depth_near_far_ratio_ask",
            "coinbase_book_avg_dist_bid_bp": "cb_book_avg_dist_bid_bp",
            "coinbase_book_avg_dist_ask_bp": "cb_book_avg_dist_ask_bp",
            "coinbase_book_slope_bid": "cb_book_slope_bid",
            "coinbase_book_slope_ask": "cb_book_slope_ask",
            "coinbase_book_convexity": "cb_book_convexity",
            "coinbase_wall_bid_size": "cb_wall_bid_size",
            "coinbase_wall_bid_dist_bp": "cb_wall_bid_dist_bp",
            "coinbase_wall_ask_size": "cb_wall_ask_size",
            "coinbase_wall_ask_dist_bp": "cb_wall_ask_dist_bp",
            "coinbase_wall_imbalance": "cb_wall_imbalance",
            "coinbase_depth_imb_1bp_diff": "cb_depth_imb_1bp_diff",
            "coinbase_net_add_bid_1bp_1s": "cb_net_add_bid_1bp_1s",
            "coinbase_net_add_ask_1bp_1s": "cb_net_add_ask_1bp_1s",
            "coinbase_n_bid_improve_1s": "cb_n_bid_improve_1s",
            "coinbase_n_bid_worsen_1s": "cb_n_bid_worsen_1s",
            "coinbase_n_ask_improve_1s": "cb_n_ask_improve_1s",
            "coinbase_n_ask_worsen_1s": "cb_n_ask_worsen_1s",
            "coinbase_n_spread_tighten_1s": "cb_n_spread_tighten_1s",
            "coinbase_n_spread_widen_1s": "cb_n_spread_widen_1s",
            "coinbase_last_trade_side": "cb_last_trade_side",
            "coinbase_last_trade_ts_s": "cb_last_trade_ts_s",
            "coinbase_last_trade_px": "cb_last_trade_px",
            "coinbase_last_trade_vs_mid_bp": "cb_last_trade_vs_mid_bp",
            "coinbase_last_trade_at_bid": "cb_last_trade_at_bid",
            "coinbase_last_trade_at_ask": "cb_last_trade_at_ask",
            "coinbase_buy_frac_1s": "cb_buy_frac_1s",
            "coinbase_buy_frac_3s": "cb_buy_frac_3s",
            "coinbase_buy_frac_5s": "cb_buy_frac_5s",
            "coinbase_ret_1s": "cb_ret_1s",
            "coinbase_ret_3s": "cb_ret_3s",
            "coinbase_ret_5s": "cb_ret_5s",
            "coinbase_ret_10s": "cb_ret_10s",
            "coinbase_rv_3s": "cb_rv_3s",
            "coinbase_rv_up_3s": "cb_rv_up_3s",
            "coinbase_rv_dn_3s": "cb_rv_dn_3s",
        }

        res: Dict[str, Optional[float]] = {out_key: None for out_key in attr_to_key.values()}
        for attr_name, out_key in attr_to_key.items():
            d = getattr(h, attr_name, None)
            if isinstance(d, Mapping):
                res[out_key] = _safe_float(d.get(base))
        return res

    def _pm_read(
        h: StateHub, base: str
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        st = getattr(h, "pm_base_state", {}) or {}
        d = st.get(base) if isinstance(st, dict) else None
        if not isinstance(d, dict):
            return (None, None, None, None, None, None, None, None, None)

        pm_spread = _safe_float(d.get("spread"))
        pm_imb1 = _safe_float(d.get("imb1"))
        pm_imb3 = _safe_float(d.get("imb3"))
        pm_imb5 = _safe_float(d.get("imb5"))
        pm_trade_imb3s = _safe_float(d.get("trade_imb_3s"))

        bids = d.get("bids") or []
        asks = d.get("asks") or []
        bid_px = bid_sz = ask_px = ask_sz = None
        try:
            if isinstance(bids, list) and len(bids) > 0:
                bid_px = _safe_float(bids[0][0])
                bid_sz = _safe_float(bids[0][1])
            if isinstance(asks, list) and len(asks) > 0:
                ask_px = _safe_float(asks[0][0])
                ask_sz = _safe_float(asks[0][1])
        except Exception:
            pass

        return (pm_spread, pm_imb1, pm_imb3, pm_imb5, pm_trade_imb3s, bid_px, bid_sz, ask_px, ask_sz)

    def _pm_iv_read(h: StateHub, base: str) -> Tuple[Optional[float], Optional[float]]:
        implied = mult = None
        d_implied = getattr(h, "pm_iv_implied_900", None)
        if isinstance(d_implied, Mapping):
            implied = _safe_float(d_implied.get(base))
        d_mult = getattr(h, "pm_iv_mult", None)
        if isinstance(d_mult, Mapping):
            mult = _safe_float(d_mult.get(base))
        return implied, mult

    def _hub_set_seq_quote(base: str, payload: Dict[str, Any]) -> None:
        try:
            d = getattr(hub, "seq_quote", None)
            if not isinstance(d, dict):
                hub.seq_quote = {}  # type: ignore[attr-defined]
                d = hub.seq_quote  # type: ignore[attr-defined]
            d[str(base).upper()] = payload
            hub.mark_updated()
        except Exception:
            pass

    def _hub_set_iv_pred(base: str, payload: Dict[str, Any]) -> None:
        try:
            d = getattr(hub, "iv_pred", None)
            if not isinstance(d, dict):
                hub.iv_pred = {}  # type: ignore[attr-defined]
                d = hub.iv_pred  # type: ignore[attr-defined]
            d[str(base).upper()] = payload
            hub.mark_updated()
        except Exception:
            pass

    # =========================================================================
    # Seq engine + LATEST-ONLY prediction coalescer
    # =========================================================================
    seq_engine: Optional[SeqQuoteEngine] = None
    _seq_last_meta: Dict[str, Dict[str, Any]] = {b: {} for b in bases}
    _seq_pending: Dict[str, bool] = {b: False for b in bases}
    _seq_running: Dict[str, bool] = {b: False for b in bases}
    _seq_last_pred_ms: Dict[str, int] = {b: 0 for b in bases}

    def _flatten_seq_quantiles(qobj: Any) -> Dict[str, float]:
        """status['quantiles'][target][q] -> seq_<target>_q10/q25/q50/q75/q90"""
        out: Dict[str, float] = {}
        if not isinstance(qobj, dict):
            return out
        want = [0.10, 0.25, 0.50, 0.75, 0.90]
        for tname, qm in qobj.items():
            if not isinstance(tname, str) or not isinstance(qm, dict):
                continue
            for q in want:
                v = qm.get(f"{q:.2f}")
                if v is None:
                    v = qm.get(q)
                fv = _safe_float(v)
                if fv is None:
                    continue
                qtag = int(round(q * 100))
                out[f"seq_{tname}_q{qtag:02d}"] = float(fv)
        return out

    if ENABLE_SEQ:
        try:
            seq_engine = SeqQuoteEngine(
                run_dir=SEQ_RUN_DIR,
                data_dir=SEQ_DATA_DIR,
                checkpoint_name=SEQ_CKPT,
                device="cpu",
            )
            jlog(
                logging.INFO,
                "seq_engine_loaded",
                run_dir=str(SEQ_RUN_DIR),
                data_dir=str(SEQ_DATA_DIR),
                ckpt=str(SEQ_CKPT),
            )
        except Exception as e:
            seq_engine = None
            jlog(logging.ERROR, "seq_engine_load_failed", error=str(e))

    def _build_seq_snapshot(base: str, micro: float, ts_s: float, now_ms: int) -> Dict[str, Any]:
        (
            cb_mp_skew_1bp,
            cb_mp_skew_2bp,
            cb_mp_skew_5bp,
            cb_mp_skew_10bp,
            cb_flow1s_net,
            cb_flow3s_net,
            cb_flow5s_net,
            cb_sigma_ewma,
            cb_jump_flag,
        ) = _cb_read(hub, base)

        cb_l2_fields = _cb_l2_read(hub, base)
        cb_ts_server_ms, _ = _read_ts_fields(hub, base)

        cb_fields: Dict[str, Any] = {
            "cb_mp_skew_1bp": cb_mp_skew_1bp,
            "cb_mp_skew_2bp": cb_mp_skew_2bp,
            "cb_mp_skew_5bp": cb_mp_skew_5bp,
            "cb_mp_skew_10bp": cb_mp_skew_10bp,
            "cb_flow1s_net": cb_flow1s_net,
            "cb_flow3s_net": cb_flow3s_net,
            "cb_flow5s_net": cb_flow5s_net,
            "cb_sigma_ewma": cb_sigma_ewma,
            "cb_jump_flag": cb_jump_flag,
        }
        cb_fields.update(cb_l2_fields)

        snap: Dict[str, Any] = {
            "base": base,
            "coinbase": float(micro),
            "cb_last_ts": float(ts_s),
            "cb_ts_server_ms": int(cb_ts_server_ms) if cb_ts_server_ms is not None else None,
            "log_ts_ms": int(now_ms),
        }
        snap.update({k: (float(v) if isinstance(v, (int, float)) else None) for k, v in cb_fields.items()})
        return snap

    def _seq_store_tick(base: str, micro: float, ts_s: float) -> None:
        """FAST: always store tick into seq buffer (no heavy work)."""
        nonlocal seq_engine
        if seq_engine is None:
            return
        now_ms = int(_time.time() * 1000)
        snap = _build_seq_snapshot(base, micro=micro, ts_s=ts_s, now_ms=now_ms)
        seq_engine.add_tick(snap)
        _seq_last_meta[base] = {
            "cb_ts_server_ms": snap.get("cb_ts_server_ms"),
            "cb_last_ts": snap.get("cb_last_ts"),
            "log_ts_ms": snap.get("log_ts_ms"),
        }

    async def _seq_pred_drain(base: str) -> None:
        """Heavy: coalescing predictor (latest-only)."""
        nonlocal seq_engine
        if seq_engine is None:
            _seq_running[base] = False
            _seq_pending[base] = False
            return

        try:
            while True:
                _seq_pending[base] = False

                if SEQ_PRED_MIN_INTERVAL_MS > 0:
                    now_ms = int(_time.time() * 1000)
                    dt = now_ms - _seq_last_pred_ms.get(base, 0)
                    if dt < SEQ_PRED_MIN_INTERVAL_MS:
                        await asyncio.sleep((SEQ_PRED_MIN_INTERVAL_MS - dt) / 1000.0)

                preds, status = seq_engine.predict_for_base(base)
                now_ms2 = int(_time.time() * 1000)

                qobj = status.get("quantiles") if isinstance(status, dict) else None
                if not isinstance(qobj, dict) and isinstance(preds, dict):
                    qobj = preds.get("quantiles")

                if status.get("ok"):
                    meta = _seq_last_meta.get(base, {})
                    flat = _flatten_seq_quantiles(qobj)
                    quote_payload = _seq_quote_from_status(status)

                    _hub_set_seq_quote(
                        base,
                        {
                            "cb_ts_server_ms": meta.get("cb_ts_server_ms"),
                            "cb_last_ts": meta.get("cb_last_ts"),
                            "segment_len": int(status.get("segment_len", 0)),
                            "n_ticks": int(status.get("n_ticks", 0)),
                            **quote_payload,
                        },
                    )

                    jlog(
                        logging.INFO,
                        "seq_pred",
                        base=base,
                        cb_ts_server_ms=meta.get("cb_ts_server_ms"),
                        cb_last_ts=meta.get("cb_last_ts"),
                        n_ticks=int(status.get("n_ticks", 0)),
                        segment_len=int(status.get("segment_len", 0)),
                        **flat,
                        **quote_payload,
                    )
                    _seq_last_pred_ms[base] = now_ms2
                else:
                    jlog(
                        logging.DEBUG,
                        "seq_pred_not_ready",
                        base=base,
                        reason=status.get("reason"),
                        n_ticks=int(status.get("n_ticks", 0)),
                        segment_len=int(status.get("segment_len", 0)),
                    )

                if not _seq_pending.get(base, False):
                    break

        except Exception as e:
            jlog(logging.ERROR, "seq_pred_task_error", base=base, error=str(e))
        finally:
            _seq_running[base] = False
            if _seq_pending.get(base, False):
                if not _seq_running[base]:
                    _seq_running[base] = True
                    asyncio.create_task(_seq_pred_drain(base))

    def _seq_request_predict(base: str) -> None:
        if seq_engine is None:
            return
        _seq_pending[base] = True
        if not _seq_running[base]:
            _seq_running[base] = True
            asyncio.create_task(_seq_pred_drain(base))

    # =========================================================================
    # IV engine + LATEST-ONLY prediction coalescer
    #   - compatible with NEW quote_iv:
    #       status: segment_len_raw / segment_len_used / n_events_total / t_last_s / cb_ts_server_ms / pm_ts_server_ms
    #       preds: mid_pred/hs_pred/bid_pred/ask_pred/spread_pred/delta_mid/z0/z1 + mid_base/last_best_bid/ask/used_len/pad_len
    #   - AND clear IV state ONLY at 15m contract roll.
    # =========================================================================
    iv_engine: Optional[IVQuoteEngine] = None
    _iv_last_meta: Dict[str, Dict[str, Any]] = {b: {} for b in bases}
    _iv_pending: Dict[str, bool] = {b: False for b in bases}
    _iv_running: Dict[str, bool] = {b: False for b in bases}
    _iv_last_pred_ms: Dict[str, int] = {b: 0 for b in bases}

    # latest payload for trade snapshots
    _iv_latest: Dict[str, Dict[str, Any]] = {b: {} for b in bases}

    if ENABLE_IV:
        try:
            if IV_DATA_DIR.strip():
                iv_engine = IVQuoteEngine(
                    model_dir=IV_RUN_DIR,
                    data_dir=IV_DATA_DIR,
                    device="cpu",
                    min_seq_len=16,
                    max_seq_len=64,
                    max_gap_sec=1.0,
                    enable_async_predict=False,
                )
            else:
                iv_engine = IVQuoteEngine(
                    model_dir=IV_RUN_DIR,
                    device="cpu",
                    min_seq_len=16,
                    max_seq_len=64,
                    max_gap_sec=1.0,
                    enable_async_predict=False,
                )
            jlog(logging.INFO, "iv_engine_loaded", run_dir=str(IV_RUN_DIR), data_dir=str(IV_DATA_DIR))
        except Exception as e:
            iv_engine = None
            jlog(logging.ERROR, "iv_engine_load_failed", error=str(e))

    def _build_iv_snapshot(base: str, micro: float, ts_s: float, now_ms: int) -> Dict[str, Any]:
        B = _next_15m_boundary_s(int(ts_s))
        tau = max(1.0, float(B - ts_s - 1.1))

        prev_close_cb, _, _ = _coinbase_prev_close(base, ts_s)
        (_, _, _, _, _, bid_px, _bid_sz, ask_px, _ask_sz) = _pm_read(hub, base)

        cb_l2_fields = _cb_l2_read(hub, base)
        (
            _cb_mp_skew_1bp,
            _cb_mp_skew_2bp,
            _cb_mp_skew_5bp,
            _cb_mp_skew_10bp,
            cb_flow1s_net,
            _cb_flow3s_net,
            _cb_flow5s_net,
            cb_sigma_ewma,
            _cb_jump_flag,
        ) = _cb_read(hub, base)

        pm_iv_implied_900, _pm_iv_mult = _pm_iv_read(hub, base)
        st_any = _get_any_model_state(hub, base)
        model_iv = _safe_float(st_any.get("iv")) if isinstance(st_any, dict) else None

        cb_ts_server_ms, pm_ts_server_ms = _read_ts_fields(hub, base)

        snap: Dict[str, Any] = {
            "base": base,
            "coinbase": float(micro),
            "prev_close": float(prev_close_cb) if prev_close_cb is not None else None,
            "tau": float(tau),
            "model_iv": float(model_iv) if model_iv is not None else None,
            "pm_iv_implied_900": float(pm_iv_implied_900) if pm_iv_implied_900 is not None else None,
            "bid_px": float(bid_px) if bid_px is not None else None,
            "ask_px": float(ask_px) if ask_px is not None else None,
            "cb_last_ts": float(ts_s),
            "cb_ts_server_ms": int(cb_ts_server_ms) if cb_ts_server_ms is not None else None,
            "pm_ts_server_ms": int(pm_ts_server_ms) if pm_ts_server_ms is not None else None,
            "log_ts_ms": int(now_ms),
            # build_iv-required cb features (if present)
            "cb_sigma_ewma": float(cb_sigma_ewma) if cb_sigma_ewma is not None else None,
            "cb_flow1s_net": float(cb_flow1s_net) if cb_flow1s_net is not None else None,
            "cb_rv_3s": float(cb_l2_fields.get("cb_rv_3s")) if cb_l2_fields.get("cb_rv_3s") is not None else None,
            "cb_buy_frac_1s": float(cb_l2_fields.get("cb_buy_frac_1s")) if cb_l2_fields.get("cb_buy_frac_1s") is not None else None,
            "cb_buy_frac_5s": float(cb_l2_fields.get("cb_buy_frac_5s")) if cb_l2_fields.get("cb_buy_frac_5s") is not None else None,
            "cb_depth_imb_1bp": float(cb_l2_fields.get("cb_depth_imb_1bp")) if cb_l2_fields.get("cb_depth_imb_1bp") is not None else None,
            "cb_spread_bp": float(cb_l2_fields.get("cb_spread_bp")) if cb_l2_fields.get("cb_spread_bp") is not None else None,
        }
        return snap

    def _iv_store_tick(base: str, micro: float, ts_s: float) -> None:
        nonlocal iv_engine
        if iv_engine is None:
            return
        now_ms = int(_time.time() * 1000)
        snap = _build_iv_snapshot(base, micro=micro, ts_s=ts_s, now_ms=now_ms)
        iv_engine.add_cb_tick(snap, schedule_predict=False)
        _iv_last_meta[base] = {
            "cb_ts_server_ms": snap.get("cb_ts_server_ms"),
            "pm_ts_server_ms": snap.get("pm_ts_server_ms"),
            "cb_last_ts": snap.get("cb_last_ts"),
            "log_ts_ms": snap.get("log_ts_ms"),
        }

    async def _iv_pred_drain(base: str) -> None:
        nonlocal iv_engine
        if iv_engine is None:
            _iv_running[base] = False
            _iv_pending[base] = False
            return

        try:
            while True:
                _iv_pending[base] = False

                if IV_PRED_MIN_INTERVAL_MS > 0:
                    now_ms = int(_time.time() * 1000)
                    dt = now_ms - _iv_last_pred_ms.get(base, 0)
                    if dt < IV_PRED_MIN_INTERVAL_MS:
                        await asyncio.sleep((IV_PRED_MIN_INTERVAL_MS - dt) / 1000.0)

                preds, status = iv_engine.predict_for_base(base)
                now_ms2 = int(_time.time() * 1000)

                if status.get("ok") and isinstance(preds, dict):
                    meta = _iv_last_meta.get(base, {})

                    # keep latest for trade snapshots
                    _iv_latest[base] = {
                        "ts_ms": int(now_ms2),
                        "preds": dict(preds),
                        "status": dict(status),
                        "meta": dict(meta),
                    }

                    # also store in hub for debug
                    _hub_set_iv_pred(
                        base,
                        {
                            "ts_ms": int(now_ms2),
                            "cb_ts_server_ms": meta.get("cb_ts_server_ms"),
                            "pm_ts_server_ms": meta.get("pm_ts_server_ms"),
                            "cb_last_ts": meta.get("cb_last_ts"),
                            **{k: preds.get(k) for k in preds.keys()},
                            **{
                                "segment_len_raw": int(status.get("segment_len_raw", 0)),
                                "segment_len_used": int(status.get("segment_len_used", 0)),
                                "n_events_total": int(status.get("n_events_total", 0)),
                                "t_last_s": status.get("t_last_s"),
                            },
                        },
                    )

                    jlog(
                        logging.INFO,
                        "iv_pred",
                        base=base,
                        cb_ts_server_ms=meta.get("cb_ts_server_ms"),
                        pm_ts_server_ms=meta.get("pm_ts_server_ms"),
                        cb_last_ts=meta.get("cb_last_ts"),
                        segment_len_raw=int(status.get("segment_len_raw", 0)),
                        segment_len_used=int(status.get("segment_len_used", 0)),
                        n_events_total=int(status.get("n_events_total", 0)),
                        t_last_s=status.get("t_last_s"),
                        mid_base=preds.get("mid_base"),
                        last_best_bid=preds.get("last_best_bid"),
                        last_best_ask=preds.get("last_best_ask"),
                        used_len=preds.get("used_len"),
                        pad_len=preds.get("pad_len"),
                        mid_pred=float(preds.get("mid_pred", 0.0)),
                        hs_pred=float(preds.get("hs_pred", 0.0)),
                        bid_pred=float(preds.get("bid_pred", 0.0)),
                        ask_pred=float(preds.get("ask_pred", 0.0)),
                        spread_pred=float(preds.get("spread_pred", 0.0)),
                        delta_mid=float(preds.get("delta_mid", 0.0)),
                        z0=float(preds.get("z0", 0.0)),
                        z1=float(preds.get("z1", 0.0)),
                    )
                    _iv_last_pred_ms[base] = now_ms2
                else:
                    jlog(
                        logging.DEBUG,
                        "iv_pred_not_ready",
                        base=base,
                        reason=status.get("reason"),
                        segment_len_raw=int(status.get("segment_len_raw", 0)),
                        segment_len_used=int(status.get("segment_len_used", 0)),
                        n_events_total=int(status.get("n_events_total", 0)),
                    )

                if not _iv_pending.get(base, False):
                    break

        except Exception as e:
            jlog(logging.ERROR, "iv_pred_task_error", base=base, error=str(e))
        finally:
            _iv_running[base] = False
            if _iv_pending.get(base, False):
                if not _iv_running[base]:
                    _iv_running[base] = True
                    asyncio.create_task(_iv_pred_drain(base))

    def _iv_request_predict(base: str) -> None:
        if iv_engine is None:
            return
        _iv_pending[base] = True
        if not _iv_running[base]:
            _iv_running[base] = True
            asyncio.create_task(_iv_pred_drain(base))

    # ---- PM book callback into IV engine (requires market_lib to support on_book_event) ----
    def _on_pm_book_event(baseU: str, bids: Any, asks: Any, ts_server_ms: Optional[int]) -> None:
        nonlocal iv_engine
        if iv_engine is None:
            return
        try:
            iv_engine.add_pm_book_update(
                base=baseU,
                bids=bids,
                asks=asks,
                ts_server_ms=ts_server_ms,
                ts_s=None,
                schedule_predict=False,
            )
            _iv_request_predict(baseU)
        except Exception as e:
            jlog(logging.ERROR, "iv_on_book_event_error", base=baseU, error=str(e))

    # =========================================================================
    # IV roll-clear task (ONLY clears IV state at 15m contract switch)
    # =========================================================================
    async def _iv_roll_clear_task() -> None:
        nonlocal iv_engine
        # align on boundary (same for all bases) -> clear_all_states once per boundary
        last_cleared_boundary: Optional[int] = None

        while True:
            await asyncio.sleep(0.10)

            if iv_engine is None:
                continue

            now_s = _time.time()
            now_i = int(now_s)
            B = int(_next_15m_boundary_s(now_i))  # next boundary epoch seconds

            # sleep until boundary, then clear
            sleep_s = float(B) - now_s + IV_ROLL_CLEAR_EPS_SEC
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

            # recompute boundary at wake to be robust
            now2_i = int(_time.time())
            B2 = int(_next_15m_boundary_s(now2_i))
            # At/after boundary, the "next" boundary jumps forward; so the boundary we just crossed is B2-900.
            crossed_boundary = B2 - 900

            if last_cleared_boundary is not None and crossed_boundary <= last_cleared_boundary:
                continue

            try:
                iv_engine.clear_all_states()
            except Exception as e:
                jlog(logging.ERROR, "iv_roll_clear_error", error=str(e))
                last_cleared_boundary = crossed_boundary
                continue

            # reset iv local caches / meta so logs and trade snapshot don't accidentally carry stale preds
            for b in bases:
                _iv_last_meta[b] = {}
                _iv_pending[b] = False
                _iv_running[b] = False
                _iv_latest[b] = {}
            try:
                d = getattr(hub, "iv_pred", None)
                if isinstance(d, dict):
                    d.clear()
                    hub.mark_updated()
            except Exception:
                pass

            last_cleared_boundary = crossed_boundary
            jlog(
                logging.INFO,
                "iv_state_cleared_on_contract_roll",
                crossed_boundary_s=int(crossed_boundary),
                new_session_start_s=int(crossed_boundary),
                note="cleared iv state only; seq state untouched",
            )

    iv_roll_task: Optional[asyncio.Task] = None
    if iv_engine is not None:
        iv_roll_task = asyncio.create_task(_iv_roll_clear_task())

    # =========================================================================
    # Coinbase producer (emit rule stays same) + unconditional seq/iv store
    # =========================================================================
    last_emit_ts: Dict[str, float] = {b: 0.0 for b in bases}

    async def _coinbase_producer_gated() -> None:
        last_seen_latest: Dict[str, float] = {}
        missing_log_ts: Dict[str, float] = {}
        start_ts = _time.time()

        while True:
            now_s = _time.time()
            for b in bases:
                latest, prev = _read_micro_pair(hub, b)

                if latest is None or prev is None:
                    if now_s - start_ts >= CB_MISSING_LOG_GRACE_SEC:
                        t_prev = missing_log_ts.get(b, -1e9)
                        if (now_s - t_prev) >= CB_MISSING_LOG_RATE_SEC:
                            jlog(
                                logging.INFO,
                                "cb_micro_pair_missing",
                                base=b,
                                has_latest=bool(latest is not None),
                                has_prev=bool(prev is not None),
                            )
                            missing_log_ts[b] = now_s
                    continue

                latest = float(latest)
                prev_f = float(prev)

                if last_seen_latest.get(b) != latest:
                    _cb_hist_add(b, latest, ts_s=now_s)
                    last_seen_latest[b] = latest

                    if abs(latest - prev_f) >= MP_THRESH[b]:
                        _seq_store_tick(b, micro=latest, ts_s=now_s)
                        _seq_request_predict(b)

                        _iv_store_tick(b, micro=latest, ts_s=now_s)
                        _iv_request_predict(b)

                        q = tick_queues[b]
                        if q.full():
                            try:
                                _ = q.get_nowait()
                                q.task_done()
                            except Exception:
                                pass
                        await q.put((latest, now_s))
                        last_emit_ts[b] = now_s

            await asyncio.sleep(0.002)

    async def _heartbeat_emitter() -> None:
        while True:
            now_s = _time.time()
            for b in bases:
                if (now_s - last_emit_ts.get(b, 0.0)) >= HEARTBEAT_SEC:
                    latest, prev = _read_micro_pair(hub, b)
                    if latest is not None and prev is not None:
                        latest_f = float(latest)

                        _seq_store_tick(b, micro=latest_f, ts_s=now_s)
                        _seq_request_predict(b)

                        _iv_store_tick(b, micro=latest_f, ts_s=now_s)
                        _iv_request_predict(b)

                        q = tick_queues[b]
                        if q.full():
                            try:
                                _ = q.get_nowait()
                                q.task_done()
                            except Exception:
                                pass
                        await q.put((latest_f, now_s))
                        last_emit_ts[b] = now_s

            await asyncio.sleep(0.05)

    # =========================================================================
    # Model worker per base (TRADING PATH; still gated)
    #   - Pass EVERYTHING into trader snapshot:
    #       * raw features
    #       * model (binance) params
    #       * seq quote payload (scores etc)
    #       * iv prediction payload + status
    # =========================================================================
    async def _mm_worker(base: str) -> None:
        while True:
            micro, ts_s = await tick_queues[base].get()
            try:
                now_s = _time.time()

                pred = _get_fresh_model_state(hub, base, now_s)
                if pred is None:
                    jlog(logging.INFO, "skip_cb_tick_pred_no_fresh_model", base=base, now_s=now_s)
                    continue

                model_pred_ts = pred.get("exch_sec")
                if not isinstance(model_pred_ts, (int, float)):
                    jlog(logging.INFO, "skip_missing_model_exch_sec", base=base)
                    continue
                model_pred_ts = float(model_pred_ts)

                B = _next_15m_boundary_s(int(ts_s))
                S = B - 900
                tau = max(1.0, float(B - ts_s - 1.1))
                sqrt_tau = math.sqrt(tau)
                slug = f"{base.lower()}-updown-15m-{S}"

                prev_close_cb, prev_high_cb, prev_low_cb = _coinbase_prev_close(base, ts_s)
                if prev_close_cb is None:
                    jlog(
                        logging.INFO,
                        "waiting_prevclose_coinbase",
                        base=base,
                        hint="need history <= session_start-1.1s",
                    )
                    continue

                hub.set_coinbase_prev_close(base, prev_close_cb, prev_high=prev_high_cb, prev_low=prev_low_cb)

                (
                    pm_spread,
                    pm_imb1,
                    pm_imb3,
                    pm_imb5,
                    pm_trade_imb3s,
                    bid_px,
                    bid_sz,
                    ask_px,
                    ask_sz,
                ) = _pm_read(hub, base)

                (
                    cb_mp_skew_1bp,
                    cb_mp_skew_2bp,
                    cb_mp_skew_5bp,
                    cb_mp_skew_10bp,
                    cb_flow1s_net,
                    cb_flow3s_net,
                    cb_flow5s_net,
                    cb_sigma_ewma,
                    cb_jump_flag,
                ) = _cb_read(hub, base)

                cb_l2_fields = _cb_l2_read(hub, base)

                iv_param = _safe_float(pred.get("iv")) if isinstance(pred, dict) else None
                df_param = _safe_float(pred.get("df")) if isinstance(pred, dict) else None
                if iv_param is None or df_param is None:
                    jlog(logging.INFO, "skip_missing_model_params", base=base)
                    continue

                cb_ts_server_ms, pm_ts_server_ms = _read_ts_fields(hub, base)
                now_ms = int(_time.time() * 1000)
                pm_age_ms = None if pm_ts_server_ms is None else (now_ms - int(pm_ts_server_ms))
                if (pm_ts_server_ms is None) or (
                    pm_age_ms is not None and pm_age_ms > int(PM_TS_MAX_AGE_SEC * 1000)
                ):
                    jlog(
                        logging.WARNING,
                        "skip_stale_pm_ts",
                        base=base,
                        pm_ts_server_ms=(int(pm_ts_server_ms) if pm_ts_server_ms is not None else None),
                        age_ms=pm_age_ms,
                        max_age_ms=int(PM_TS_MAX_AGE_SEC * 1000),
                    )
                    continue

                ch_latest = None
                cl = getattr(hub, "chainlink_latest", None)
                if isinstance(cl, dict):
                    ch_latest = _safe_float(cl.get(base))
                ch_15m = None
                cl15 = getattr(hub, "chainlink_15m_close", None)
                if isinstance(cl15, dict):
                    v = cl15.get(base)
                    if isinstance(v, (list, tuple)) and len(v) == 2:
                        try:
                            ch_15m = [int(v[0]), float(v[1])]
                        except Exception:
                            ch_15m = None

                pm_iv_implied_900, pm_iv_mult = _pm_iv_read(hub, base)

                # latest seq quote payload from hub
                seq_quote = None
                try:
                    sd = getattr(hub, "seq_quote", None)
                    if isinstance(sd, dict):
                        seq_quote = sd.get(base)
                except Exception:
                    seq_quote = None

                # latest iv pred payload (prefer local cache; fallback hub)
                iv_payload = _iv_latest.get(base) or {}
                if not iv_payload:
                    try:
                        hd = getattr(hub, "iv_pred", None)
                        if isinstance(hd, dict) and isinstance(hd.get(base), dict):
                            iv_payload = {"preds": dict(hd.get(base) or {})}
                    except Exception:
                        iv_payload = {}

                # optional on-demand IV predict to ensure trade snapshot has *something*
                # (still respects IV_PRED_MIN_INTERVAL_MS)
                if iv_engine is not None:
                    want_predict = False
                    last_ms = _iv_last_pred_ms.get(base, 0)
                    if IV_PRED_MIN_INTERVAL_MS <= 0:
                        # if never predicted yet, or very stale cache
                        ts_ms_cache = None
                        if isinstance(iv_payload, dict):
                            ts_ms_cache = iv_payload.get("ts_ms")
                        if not isinstance(ts_ms_cache, int) or (now_ms - ts_ms_cache) > 700:
                            want_predict = True
                    else:
                        if (now_ms - last_ms) >= IV_PRED_MIN_INTERVAL_MS:
                            # stale cache -> predict
                            ts_ms_cache = None
                            if isinstance(iv_payload, dict):
                                ts_ms_cache = iv_payload.get("ts_ms")
                            if not isinstance(ts_ms_cache, int) or (now_ms - ts_ms_cache) > max(700, IV_PRED_MIN_INTERVAL_MS):
                                want_predict = True

                    if want_predict:
                        try:
                            preds_iv, status_iv = iv_engine.predict_for_base(base)
                            if status_iv.get("ok") and isinstance(preds_iv, dict):
                                _iv_last_pred_ms[base] = int(now_ms)
                                iv_payload = {
                                    "ts_ms": int(now_ms),
                                    "preds": dict(preds_iv),
                                    "status": dict(status_iv),
                                    "meta": dict(_iv_last_meta.get(base, {})),
                                }
                                _iv_latest[base] = dict(iv_payload)
                                _hub_set_iv_pred(base, dict(preds_iv))
                        except Exception as e:
                            jlog(logging.ERROR, "iv_predict_inline_error", base=base, error=str(e))

                salt = secrets.token_hex(8)

                cb_fields: Dict[str, Any] = {
                    "cb_mp_skew_1bp": cb_mp_skew_1bp,
                    "cb_mp_skew_2bp": cb_mp_skew_2bp,
                    "cb_mp_skew_5bp": cb_mp_skew_5bp,
                    "cb_mp_skew_10bp": cb_mp_skew_10bp,
                    "cb_flow1s_net": cb_flow1s_net,
                    "cb_flow3s_net": cb_flow3s_net,
                    "cb_flow5s_net": cb_flow5s_net,
                    "cb_sigma_ewma": cb_sigma_ewma,
                    "cb_jump_flag": cb_jump_flag,
                }
                cb_fields.update(cb_l2_fields)

                pm_iv_fields: Dict[str, Any] = {
                    "pm_iv_implied_900": pm_iv_implied_900,
                    "pm_iv_mult": pm_iv_mult,
                }

                # seq quote flattened for cb_tick_pred log
                sq_score = _safe_float(seq_quote.get("quote_score")) if isinstance(seq_quote, dict) else None
                sq_skew = _safe_float(seq_quote.get("quote_skew")) if isinstance(seq_quote, dict) else None
                sq_z = _safe_float(seq_quote.get("quote_z")) if isinstance(seq_quote, dict) else None
                sq_side = seq_quote.get("quote_side") if isinstance(seq_quote, dict) else None
                sq_one = bool(seq_quote.get("quote_one_sided")) if isinstance(seq_quote, dict) else None

                # iv flattened for cb_tick_pred log
                ivp = iv_payload.get("preds") if isinstance(iv_payload, dict) else None
                if not isinstance(ivp, dict):
                    ivp = None

                jlog(
                    logging.INFO,
                    "cb_tick_pred",
                    salt=salt,
                    base=base,
                    slug=slug,
                    coinbase=float(micro),
                    prev_close=float(prev_close_cb),
                    prev_high=float(prev_high_cb) if prev_high_cb is not None else None,
                    prev_low=float(prev_low_cb) if prev_low_cb is not None else None,
                    tau=float(tau),
                    sqrt_tau=float(sqrt_tau),
                    bid_px=float(bid_px) if bid_px is not None else None,
                    bid_sz=float(bid_sz) if bid_sz is not None else None,
                    ask_px=float(ask_px) if ask_px is not None else None,
                    ask_sz=float(ask_sz) if ask_sz is not None else None,
                    pm_spread=float(pm_spread) if pm_spread is not None else None,
                    pm_imb1=float(pm_imb1) if pm_imb1 is not None else None,
                    pm_imb3=float(pm_imb3) if pm_imb3 is not None else None,
                    pm_trade_imb3s=(float(pm_trade_imb3s) if pm_trade_imb3s is not None else None),
                    **{k: (float(v) if isinstance(v, (int, float)) else None) for k, v in cb_fields.items()},
                    model_iv=float(iv_param),
                    model_df=float(df_param),
                    model_pred_ts=int(model_pred_ts) if model_pred_ts is not None else None,
                    cb_ts_server_ms=int(cb_ts_server_ms) if cb_ts_server_ms is not None else None,
                    pm_ts_server_ms=int(pm_ts_server_ms) if pm_ts_server_ms is not None else None,
                    cb_last_ts=float(ts_s),
                    chainlink=float(ch_latest) if ch_latest is not None else None,
                    chainlink_15m_close=ch_15m,
                    **{k: (float(v) if isinstance(v, (int, float)) else None) for k, v in pm_iv_fields.items()},
                    seq_quote_score=float(sq_score) if sq_score is not None else None,
                    seq_quote_skew=float(sq_skew) if sq_skew is not None else None,
                    seq_quote_z=float(sq_z) if sq_z is not None else None,
                    seq_quote_one_sided=sq_one,
                    seq_quote_side=str(sq_side) if isinstance(sq_side, str) else None,
                    # IV pred fields (optional)
                    iv_mid_pred=(float(ivp.get("mid_pred")) if ivp is not None and _safe_float(ivp.get("mid_pred")) is not None else None),
                    iv_hs_pred=(float(ivp.get("hs_pred")) if ivp is not None and _safe_float(ivp.get("hs_pred")) is not None else None),
                    iv_bid_pred=(float(ivp.get("bid_pred")) if ivp is not None and _safe_float(ivp.get("bid_pred")) is not None else None),
                    iv_ask_pred=(float(ivp.get("ask_pred")) if ivp is not None and _safe_float(ivp.get("ask_pred")) is not None else None),
                    iv_delta_mid=(float(ivp.get("delta_mid")) if ivp is not None and _safe_float(ivp.get("delta_mid")) is not None else None),
                    iv_z0=(float(ivp.get("z0")) if ivp is not None and _safe_float(ivp.get("z0")) is not None else None),
                    iv_z1=(float(ivp.get("z1")) if ivp is not None and _safe_float(ivp.get("z1")) is not None else None),
                    log_ts_ms=int(now_ms),
                )

                snapshot: Dict[str, Any] = {
                    "salt": salt,
                    "base": base,
                    "slug": slug,
                    "coinbase": float(micro),
                    "prev_close": float(prev_close_cb),
                    "prev_high": float(prev_high_cb) if prev_high_cb is not None else None,
                    "prev_low": float(prev_low_cb) if prev_low_cb is not None else None,
                    "tau": float(tau),
                    "sqrt_tau": float(sqrt_tau),
                    "model_iv": float(iv_param),
                    "model_df": float(df_param),
                    "model_pred_ts": int(model_pred_ts) if model_pred_ts is not None else None,
                    "bid_px": float(bid_px) if bid_px is not None else None,
                    "bid_sz": float(bid_sz) if bid_sz is not None else None,
                    "ask_px": float(ask_px) if ask_px is not None else None,
                    "ask_sz": float(ask_sz) if ask_sz is not None else None,
                    "pm_spread": float(pm_spread) if pm_spread is not None else None,
                    "pm_imb1": float(pm_imb1) if pm_imb1 is not None else None,
                    "pm_imb3": float(pm_imb3) if pm_imb3 is not None else None,
                    "pm_imb5": float(pm_imb5) if pm_imb5 is not None else None,
                    "pm_trade_imb3s": (float(pm_trade_imb3s) if pm_trade_imb3s is not None else None),
                    "cb_ts_server_ms": int(cb_ts_server_ms) if cb_ts_server_ms is not None else None,
                    "pm_ts_server_ms": int(pm_ts_server_ms) if pm_ts_server_ms is not None else None,
                    "cb_last_ts": float(ts_s),
                    "chainlink": float(ch_latest) if ch_latest is not None else None,
                    "chainlink_15m_close": ch_15m,
                    "log_ts_ms": int(now_ms),
                }
                snapshot.update({k: (float(v) if isinstance(v, (int, float)) else None) for k, v in cb_fields.items()})
                snapshot.update({k: (float(v) if isinstance(v, (int, float)) else None) for k, v in pm_iv_fields.items()})

                # Attach seq quote payload for trader usage (full dict)
                if isinstance(seq_quote, dict):
                    snapshot["seq_quote"] = dict(seq_quote)
                    snapshot["seq_quote_score"] = _safe_float(seq_quote.get("quote_score"))
                    snapshot["seq_quote_skew"] = _safe_float(seq_quote.get("quote_skew"))
                    snapshot["seq_quote_delta_p"] = _safe_float(seq_quote.get("quote_delta_p"))
                    snapshot["seq_quote_z"] = _safe_float(seq_quote.get("quote_z"))
                    snapshot["seq_quote_mu"] = _safe_float(seq_quote.get("quote_mu"))
                    snapshot["seq_quote_sigma"] = _safe_float(seq_quote.get("quote_sigma"))
                    snapshot["seq_quote_side"] = seq_quote.get("quote_side")
                    snapshot["seq_quote_one_sided"] = bool(seq_quote.get("quote_one_sided", False))
                    snapshot["seq_quote_mode"] = seq_quote.get("quote_mode")

                # Attach IV pred payload for trader usage (preds + status + meta)
                if isinstance(iv_payload, dict) and isinstance(iv_payload.get("preds"), dict):
                    snapshot["iv_pred"] = dict(iv_payload.get("preds") or {})
                    if isinstance(iv_payload.get("status"), dict):
                        snapshot["iv_status"] = dict(iv_payload.get("status") or {})
                    if isinstance(iv_payload.get("meta"), dict):
                        snapshot["iv_meta"] = dict(iv_payload.get("meta") or {})
                    snapshot["iv_pred_ts_ms"] = iv_payload.get("ts_ms")

                    # convenient flattened aliases
                    p = snapshot["iv_pred"]
                    snapshot["iv_mid_pred"] = _safe_float(p.get("mid_pred"))
                    snapshot["iv_hs_pred"] = _safe_float(p.get("hs_pred"))
                    snapshot["iv_bid_pred"] = _safe_float(p.get("bid_pred"))
                    snapshot["iv_ask_pred"] = _safe_float(p.get("ask_pred"))
                    snapshot["iv_spread_pred"] = _safe_float(p.get("spread_pred"))
                    snapshot["iv_delta_mid"] = _safe_float(p.get("delta_mid"))
                    snapshot["iv_z0"] = _safe_float(p.get("z0"))
                    snapshot["iv_z1"] = _safe_float(p.get("z1"))
                    snapshot["iv_mid_base"] = _safe_float(p.get("mid_base"))
                    snapshot["iv_last_best_bid"] = _safe_float(p.get("last_best_bid"))
                    snapshot["iv_last_best_ask"] = _safe_float(p.get("last_best_ask"))
                    snapshot["iv_used_len"] = p.get("used_len")
                    snapshot["iv_pad_len"] = p.get("pad_len")

                async with base_locks[base]:
                    trader.on_tick(snapshot)

            finally:
                tick_queues[base].task_done()

    # --------- Start background tasks ----------
    producer_task = asyncio.create_task(_coinbase_producer_gated())
    heartbeat_task = asyncio.create_task(_heartbeat_emitter())
    worker_tasks = [asyncio.create_task(_mm_worker(b)) for b in bases]

    slug_task = asyncio.create_task(run_slug_refresher())
    jlog(logging.INFO, "slug_refresher_started")

    # ---- Polymarket stream (optionally provides book callback) ----
    try:
        clob_task = asyncio.create_task(
            run_polymarket_market_stream(
                hub,
                symbols=["btc", "eth", "xrp", "sol"],
                on_book_event=_on_pm_book_event,
            )
        )
        jlog(logging.INFO, "clob_stream_started", symbols="btc,eth,xrp,sol", on_book_event=True)
    except TypeError:
        clob_task = asyncio.create_task(run_polymarket_market_stream(hub, symbols=["btc", "eth", "xrp", "sol"]))
        jlog(logging.INFO, "clob_stream_started", symbols="btc,eth,xrp,sol", on_book_event=False)

    # --------- Load inference models & start streams ----------
    loaded: Dict[str, LoadedModel] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jlog(logging.INFO, "device_selected", device=device)
    for sym in SYMBOLS:
        lm = load_checkpoint_for_symbol(sym)
        lm.model.to(device)
        lm.device = device
        loaded[sym] = lm
        jlog(logging.INFO, "model_loaded", symbol=sym, L_sec=int(lm.L_sec), device=lm.device)

    states: Dict[str, LoadedAndState] = {}
    import aiohttp

    async with aiohttp.ClientSession() as session:
        for sym in SYMBOLS:
            jlog(logging.INFO, "seed_begin", symbol=sym)
            lm = loaded[sym]
            ring = SecRing(L=lm.L_sec)
            mfb = MinuteFeatureBuilder(symbol=sym, static_cols=[])
            st = LoadedAndState(lm=lm, ring=ring, mfb=mfb)
            await backfill_and_seed(st, session)
            states[sym] = st
            jlog(logging.INFO, "seed_complete", symbol=sym, L_sec=int(lm.L_sec))

    minute_tasks: List[asyncio.Task] = []
    for i, sym in enumerate(SYMBOLS):
        st = states[sym]
        jlog(logging.INFO, "minute_updater_start", symbol=sym, jitter=i * MINUTE_STAGGER_SECONDS)
        minute_tasks.append(
            asyncio.create_task(minute_incremental_updater(st, jitter_sec=i * MINUTE_STAGGER_SECONDS))
        )

    if WARMUP_SECONDS > 0:
        jlog(logging.INFO, "warmup_begin", seconds=WARMUP_SECONDS)
        await asyncio.sleep(WARMUP_SECONDS)
        jlog(logging.INFO, "warmup_end")

    stream_tasks: List[asyncio.Task] = []
    for sym in SYMBOLS:
        st = states[sym]

        async def _runner(s: str, state: LoadedAndState):
            try:
                await stream_symbol(state)
            except Exception as e:
                jlog(logging.ERROR, "stream_symbol_error", symbol=s, error=str(e))
                raise

        jlog(logging.INFO, "stream_symbol_spawn", symbol=sym)
        stream_tasks.append(asyncio.create_task(_runner(sym, st)))
    jlog(logging.INFO, "prediction_streams_started", symbols=",".join(SYMBOLS))

    # --------- Graceful shutdown ----------
    stop_event = asyncio.Event()

    def _graceful(*_args):
        try:
            jlog(logging.INFO, "signal", sig="terminate")
            stop_event.set()
        except Exception:
            pass

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _graceful)
        except NotImplementedError:
            pass

    await stop_event.wait()

    try:
        await trader.shutdown(reason="service_stop_event")
    except Exception as e:
        jlog(logging.ERROR, "trader_shutdown_error", error=str(e))

    await cl_svc.stop()

    # cancel tasks
    for t in stream_tasks + minute_tasks:
        t.cancel()
    for t in (cb_task, slug_task, clob_task, producer_task, heartbeat_task, hub_dump_task):
        t.cancel()
    for t in worker_tasks:
        t.cancel()
    if iv_roll_task is not None:
        iv_roll_task.cancel()

    await asyncio.gather(*stream_tasks, return_exceptions=True)
    await asyncio.gather(*minute_tasks, return_exceptions=True)
    await asyncio.gather(
        cb_task,
        slug_task,
        clob_task,
        producer_task,
        heartbeat_task,
        hub_dump_task,
        return_exceptions=True,
    )
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    if iv_roll_task is not None:
        await asyncio.gather(iv_roll_task, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(run_live_prediction())
    except KeyboardInterrupt:
        live_mod.jlog(logging.INFO, "service_exit", reason="KeyboardInterrupt")
    except Exception as e:
        live_mod.jlog(logging.ERROR, "service_crash", error=str(e))
        raise
