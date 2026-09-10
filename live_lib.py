# live_lib.py (student-t, compiled-ScaledModel checkpoint compatible)
# Helper library for live predictor: logging, HTTP/WS, models, feature builders, and streaming.

import os, sys, json, math, glob, time as _time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from collections import deque, OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler  # still allowed for legacy checkpoints

import asyncio
import aiohttp
import websockets
import logging

# ==== Project-provided utilities ====
from utils import (
    SeqTransformerT,
    build_minute_features,        # exact training static_cols/order
    end_of_current_15min,
    predict_params_t,             # inference helper -> returns (iv, df)
)

# ============================ Configuration constants ============================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
BASES = ["BTC", "ETH", "SOL", "XRP"]
RUNS_DIR = os.path.join(os.getcwd(), "runs")

BINANCE_REST = "https://api.binance.com"        # kept for possible legacy/aux calls
BINANCE_FREST = "https://fapi.binance.com"      # USDⓈ-M futures REST
# Live streaming now uses USDⓈ-M futures market streams (fstream)
BINANCE_WS   = "wss://fstream.binance.com/ws"

BACKFILL_MINUTES = 120         # initial 1m history for seeding
MINUTE_UPDATE_DELAY_SEC = 2.0  # wait after boundary to ensure closed bar
SAVE_MINUTE_HISTORY = 240      # keep last N minute rows in memory
MINUTE_FETCH_MAX_RETRIES = 6          # total attempts after the first wake
MINUTE_FETCH_RETRY_SLEEP_SEC = 1.0    # 1s between attempts
STATIC_FALLBACK_GRACE_SEC = 10.0      # seconds after minute boundary

# Per-second feature order — MUST match training (see utils.seconds_feature_cols).
SEC_COLS = [
    "sec_close","sec_vwap",
    "sec_vol","sec_signed_vol","sec_trades","sec_buy_vol","sec_sell_vol",
    "sec_ret1","sec_ret3","sec_ret5","sec_ret10","sec_ret15","sec_ret30",
    "sec_imb",
]

# ============================ Unified logging ============================
class UTCFormatter(logging.Formatter):
    converter = _time.gmtime
    def formatTime(self, record, datefmt=None):
        t = super().formatTime(record, datefmt)
        return t + "Z" if not t.endswith("Z") else t

_logger: Optional[logging.Logger] = None

def base_key(sym: str) -> str:
    s = (sym or "").upper()
    for b in BASES:
        if (
            s == b
            or s.startswith(b + "/")
            or s.startswith(b + "USD")
            or s.startswith(b + "USDT")
            or s.startswith(b + "-")
        ):
            return b
    return s

def setup_logging(
    log_path: str = os.path.join(os.getcwd(), "data", "logs", "live_predictor.log")
) -> logging.Logger:
    from logging.handlers import RotatingFileHandler
    global _logger

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.WARNING)

    logger = logging.getLogger("live_predictor")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = "%(asctime)s.%(msecs)03dZ [%(levelname)s] %(message)s"
    formatter = UTCFormatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S")

    fh = RotatingFileHandler(
        log_path, maxBytes=10_000_000, backupCount=4096, encoding="utf-8"
    )
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    root.addHandler(fh)

    if os.getenv("LIVE_LOG_STDOUT", "1") == "1":
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)

    for noisy in ("aiohttp", "websockets", "asyncio", "urllib3"):
        nlog = logging.getLogger(noisy)
        nlog.handlers.clear()
        nlog.propagate = True
        nlog.setLevel(logging.WARNING)

    _logger = logger
    return logger

def jlog(level: int, event: str, **fields):
    global _logger
    if _logger is None:
        setup_logging()
    payload = {"event": event, **fields}
    _logger.log(level, json.dumps(payload, separators=(",", ":"), ensure_ascii=False))

# ============================ HTTP helpers ============================
async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Dict[str, Any] = None,
    retries=3,
    timeout=10,
):
    for i in range(retries):
        try:
            async with session.get(url, params=params, timeout=timeout) as r:
                txt = await r.text()
                if r.status == 200:
                    try:
                        return True, json.loads(txt)
                    except Exception:
                        return True, txt
                return False, txt
        except Exception as e:
            if i == retries - 1:
                return False, str(e)
            await asyncio.sleep(0.5 * (i + 1))

async def fetch_spot_klines_1m(
    session: aiohttp.ClientSession, symbol: str, limit=BACKFILL_MINUTES
) -> pd.DataFrame:
    """
    Fetch 1m USDⓈ-M futures klines for `symbol`.

    Name kept for backward compatibility with older code that used spot klines.
    Training now uses Binance perp (futures) close/volume.
    """
    url = BINANCE_FREST + "/fapi/v1/klines"
    ok, data = await fetch_json(
        session,
        url,
        params={"symbol": symbol, "interval": "1m", "limit": limit},
    )
    if not ok:
        raise RuntimeError(f"futures klines 1m failed {symbol}: {data}")
    rows = []
    for k in data:
        rows.append(
            {
                "ts": pd.to_datetime(k[0], unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "number_of_trades": int(k[8]),
                "tb_base": float(k[9]),  # taker buy base volume
            }
        )
    return pd.DataFrame(rows).set_index("ts").sort_index()

async def fetch_index_klines_1m(
    session: aiohttp.ClientSession, symbol: str, limit=BACKFILL_MINUTES
) -> pd.DataFrame:
    url = BINANCE_FREST + "/fapi/v1/indexPriceKlines"
    ok, data = await fetch_json(
        session,
        url,
        params={"pair": symbol, "interval": "1m", "limit": limit},
    )
    if not ok:
        raise RuntimeError(f"index klines 1m failed {symbol}: {data}")
    rows = [
        {
            "ts": pd.to_datetime(k[0], unit="ms", utc=True),
            "index_close": float(k[4]),
        }
        for k in data
    ]
    return pd.DataFrame(rows).set_index("ts").sort_index()

async def fetch_spot_last2_1m(session: aiohttp.ClientSession, symbol: str):
    """
    Fetch the last 2×1m USDⓈ-M futures klines for `symbol` (for incremental updates).
    """
    url = BINANCE_FREST + "/fapi/v1/klines"
    ok, data = await fetch_json(
        session,
        url,
        params={"symbol": symbol, "interval": "1m", "limit": 2},
    )
    if not ok or not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"futures last2 klines failed {symbol}: {data}")
    return data

async def fetch_index_last2_1m(session: aiohttp.ClientSession, symbol: str):
    url = BINANCE_FREST + "/fapi/v1/indexPriceKlines"
    ok, data = await fetch_json(
        session,
        url,
        params={"pair": symbol, "interval": "1m", "limit": 2},
    )
    if not ok or not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"index last2 klines failed {symbol}: {data}")
    return data

# ============================ Checkpoint loading ============================
@dataclass
class LoadedModel:
    symbol: str
    model: SeqTransformerT
    sec_scaler: Any   # accepts sklearn or NumpyScaler
    stat_scaler: Any
    static_cols: List[str]
    L_sec: int
    device: str = field(default="cpu")

def _safe_torch_load(path: str) -> Dict[str, Any]:
    # legacy allow: sklearn scaler, though new ckpt uses arrays
    try:
        torch.serialization.add_safe_globals([StandardScaler])
    except Exception:
        pass
    return torch.load(path, map_location="cpu", weights_only=False)

def _infer_dims_from_state(sd: Dict[str, torch.Tensor]):
    # expect proj.weight and static_mlp.0.weight to exist
    for pref in ("", "base.", "_orig_mod.base."):
        k1 = pref + "proj.weight"
        k2 = pref + "static_mlp.0.weight"
        if k1 in sd and k2 in sd:
            return int(sd[k1].shape[1]), int(sd[k2].shape[1])
    # fallback
    return 14, 14

class NumpyScaler:
    """Lightweight scaler rebuilt from mean/scale arrays saved in checkpoint."""
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / (self.scale_ + 1e-9)

def _remap_state_for_seqtransformer(
    state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Accept state dicts from:
      - bare SeqTransformerT: 'proj.weight', ...
      - ScaledModel(base=SeqTransformerT): 'base.proj.weight', ...
      - compiled ScaledModel: '_orig_mod.base.proj.weight', ...
    Strip any wrapper prefixes so keys match SeqTransformerT.
    """
    keys = list(state.keys())
    if any(k.startswith("_orig_mod.base.") for k in keys):
        state = {
            k[len("_orig_mod.base.") :]: v
            for k, v in state.items()
            if k.startswith("_orig_mod.base.")
        }
    elif any(k.startswith("base.") for k in keys):
        state = {
            k[len("base.") :]: v
            for k, v in state.items()
            if k.startswith("base.")
        }
    # else: already bare
    return state

def load_checkpoint_for_symbol(
    symbol: str, runs_dir: str = RUNS_DIR
) -> LoadedModel:
    base = symbol.replace("USDT", "")
    all_ckpts = glob.glob(
        os.path.join(runs_dir, "**", "artifacts", "model_checkpoint.pt"),
        recursive=True,
    )
    ckpts = [
        p
        for p in all_ckpts
        if f"/{base}" in p
        or f"_{base}_" in p
        or p.upper().find(base.upper()) >= 0
    ] or all_ckpts
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {runs_dir} for {symbol}")
    ckpt_path = max(ckpts, key=lambda p: os.path.getmtime(p))

    blob = _safe_torch_load(ckpt_path)
    raw_state = (
        blob.get("model_state_dict") or blob.get("state_dict") or blob
    )
    state = _remap_state_for_seqtransformer(raw_state)

    # Build scalers (new format uses arrays under 'scalers')
    sec_scaler = None
    stat_scaler = None
    scalers = blob.get("scalers") or {}
    if (
        isinstance(scalers, dict)
        and {"sec_mean", "sec_scale", "stat_mean", "stat_scale"}
        <= scalers.keys()
    ):
        sec_scaler = NumpyScaler(scalers["sec_mean"], scalers["sec_scale"])
        stat_scaler = NumpyScaler(scalers["stat_mean"], scalers["stat_scale"])
    else:
        # legacy: sklearn objects
        sec_scaler = scalers.get("sec_scaler")
        stat_scaler = scalers.get("stat_scaler")
        if not (
            hasattr(sec_scaler, "transform")
            and hasattr(stat_scaler, "transform")
        ):
            raise RuntimeError(
                "Checkpoint missing usable scalers (need arrays or sklearn objects)."
            )

    mc = blob.get("model_config") or {}
    sec_d = int(mc.get("sec_d")) if "sec_d" in mc else None
    static_d = int(mc.get("static_d")) if "static_d" in mc else None
    if sec_d is None or static_d is None:
        sec_d, static_d = _infer_dims_from_state(state)

    model = SeqTransformerT(
        sec_d=sec_d,
        static_d=static_d,
        d_model=int(mc.get("d_model", 64)),
        nhead=int(mc.get("n_head", 4)),
        depth=int(mc.get("depth", 3)),
        dropout=float(mc.get("dropout", 0.20)),
    )
    # load only the inner SeqTransformerT weights
    model.load_state_dict(state, strict=True)
    model.eval()

    L_sec = int(mc.get("l_sec", 300))
    return LoadedModel(
        symbol=symbol,
        model=model,
        sec_scaler=sec_scaler,
        stat_scaler=stat_scaler,
        static_cols=[],
        L_sec=L_sec,
        device="cpu",
    )

# ============================ Rolling stats ============================
class RollingWindowStats:
    def __init__(self, window: int):
        self.w = window
        self.q = deque()
        self.s = 0.0
        self.s2 = 0.0
    def push(self, x: float):
        self.q.append(x)
        self.s += x
        self.s2 += x * x
        if len(self.q) > self.w:
            old = self.q.popleft()
            self.s -= old
            self.s2 -= old * old
    def mean(self) -> float:
        n = len(self.q)
        return self.s / n if n > 0 else 0.0
    def std(self) -> float:
        n = len(self.q)
        if n <= 1:
            return 0.0
        m = self.s / n
        var = (self.s2 - n * m * m) / (n - 1)
        return math.sqrt(max(var, 0.0))

# ============================ Incremental minute-feature builder ============================
class MinuteFeatureBuilder:
    """
    Incremental builder emitting static feature rows (training column order) for each closed 1-minute bar.
    Uses *futures* 1m klines (even though the parameter is still named `spot_1m` in build_minute_features).
    """
    def __init__(self, symbol: str, static_cols: List[str]):
        self.symbol = symbol
        self.static_cols = list(static_cols)
        self.closes = deque(maxlen=64)
        self.ret1_5 = RollingWindowStats(5)
        self.ret1_15 = RollingWindowStats(15)
        self.logv_60 = RollingWindowStats(60)
        self.last_minute_ts: Optional[pd.Timestamp] = None
        self._rows: "OrderedDict[pd.Timestamp, np.ndarray]" = OrderedDict()

    def _tod_terms(self, ts: pd.Timestamp):
        m = ts.tz_convert("UTC").hour * 60 + ts.tz_convert("UTC").minute
        phi = 2.0 * math.pi * (m / 1440.0)
        return math.sin(phi), math.cos(phi)

    def seed_from_history(
        self,
        spot_1m: pd.DataFrame,
        index_1m: pd.DataFrame,
        minute_df: pd.DataFrame,
    ):
        for ts, row in minute_df[self.static_cols].iterrows():
            self._rows[ts] = row.values.astype("float32")
        while len(self._rows) > SAVE_MINUTE_HISTORY:
            self._rows.popitem(last=False)

        spot = spot_1m.sort_index()
        self.last_minute_ts = spot.index.max() if len(spot) else None
        prev_close = None
        for _, srow in spot.iloc[-120:].iterrows():
            close = float(srow["close"])
            vol = float(srow["volume"])
            if prev_close is not None and prev_close > 0 and close > 0:
                r1 = math.log(close) - math.log(prev_close)
                self.ret1_5.push(r1)
                self.ret1_15.push(r1)
            self.closes.append(close)
            self.logv_60.push(math.log(max(vol, 1e-12)))
            prev_close = close
        jlog(
            logging.INFO,
            "seed_minute_builder",
            symbol=self.symbol,
            last_ts=str(self.last_minute_ts),
        )

    def _compute_feature_map(
        self,
        ts: pd.Timestamp,
        spot_bar: Dict[str, float],
        idx_close: Optional[float],
    ) -> Dict[str, float]:
        close = float(spot_bar["close"])
        high = float(spot_bar["high"])
        low = float(spot_bar["low"])
        vol = float(spot_bar["volume"])
        tb = float(spot_bar["tb_base"])
        trades = float(spot_bar["number_of_trades"])
        prev_close = self.closes[-1] if len(self.closes) else None
        ret_1 = (
            math.log(close) - math.log(prev_close)
            if (prev_close and prev_close > 0 and close > 0)
            else 0.0
        )
        ret_3 = (
            math.log(close) - math.log(self.closes[-3])
            if (len(self.closes) >= 3 and self.closes[-3] > 0 and close > 0)
            else 0.0
        )
        ret_5 = (
            math.log(close) - math.log(self.closes[-5])
            if (len(self.closes) >= 5 and self.closes[-5] > 0 and close > 0)
            else 0.0
        )
        self.ret1_5.push(ret_1)
        self.ret1_15.push(ret_1)
        self.logv_60.push(math.log(max(vol, 1e-12)))
        self.closes.append(close)
        rv_5, rv_15 = self.ret1_5.std(), self.ret1_15.std()
        rv_ratio = rv_5 / (rv_15 + 1e-12)
        hl_range = ((high - low) / close) if close > 0 else 0.0
        tbr = max(0.0, min(1.0, (tb / vol) if vol > 0 else 0.0))
        vol_mean, vol_std = self.logv_60.mean(), self.logv_60.std()
        vol_z_60 = (math.log(max(vol, 1e-12)) - vol_mean) / (vol_std + 1e-12)
        basis_rel = (
            ((close - idx_close) / idx_close)
            if (idx_close and idx_close > 0)
            else 0.0
        )
        tod_sin, tod_cos = self._tod_terms(ts)
        is_weekend = 1.0 if ts.tz_convert("UTC").dayofweek >= 5 else 0.0

        return {
            "ret_1": ret_1,
            "ret_3": ret_3,
            "ret_5": ret_5,
            "rv_5": rv_5,
            "rv_15": rv_15,
            "rv_ratio": rv_ratio,
            "hl_range": hl_range,
            "taker_buy_ratio": tbr,
            "trades": trades,
            "vol_z_60": vol_z_60,
            "basis_rel": basis_rel,
            "tod_sin": tod_sin,
            "tod_cos": tod_cos,
            "is_weekend": is_weekend,
        }

    def add_minute(
        self,
        ts: pd.Timestamp,
        spot_bar: Dict[str, float],
        idx_close: Optional[float],
    ):
        fmap = self._compute_feature_map(ts, spot_bar, idx_close)
        vec = np.array(
            [float(fmap.get(c, 0.0)) for c in self.static_cols], dtype="float32"
        )
        self._rows[ts] = vec
        self.last_minute_ts = ts
        while len(self._rows) > SAVE_MINUTE_HISTORY:
            self._rows.popitem(last=False)
        jlog(
            logging.INFO,
            "minute_added",
            symbol=self.symbol,
            ts=str(ts),
            close=float(spot_bar["close"]),
            vol=float(spot_bar["volume"]),
        )

    def get_row(self, t_floor: pd.Timestamp) -> Optional[np.ndarray]:
        return self._rows.get(t_floor, None)

# ============================ Per-second aggregation ============================
@dataclass
class SecAgg:
    open: float = math.nan
    high: float = -math.inf
    low: float = math.inf
    close: float = math.nan
    vol: float = 0.0
    pq_sum: float = 0.0
    signed_vol: float = 0.0
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    trades: int = 0

    def update(self, price: float, qty: float, is_buy: bool):
        if not math.isfinite(self.open):
            self.open = price
        self.close = price
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.vol += qty
        self.pq_sum += price * qty
        self.signed_vol += qty if is_buy else -qty
        if is_buy:
            self.buy_vol += qty
        else:
            self.sell_vol += qty
        self.trades += 1

    def finalize(self, prev_close: Optional[float]) -> Dict[str, float]:
        o = (
            self.open
            if math.isfinite(self.open)
            else (prev_close if prev_close is not None else math.nan)
        )
        c = (
            self.close
            if math.isfinite(self.close)
            else (prev_close if prev_close is not None else math.nan)
        )
        h = max(
            self.high,
            o if math.isfinite(o) else -math.inf,
            c if math.isfinite(c) else -math.inf,
        )
        l = min(
            self.low,
            o if math.isfinite(o) else math.inf,
            c if math.isfinite(c) else math.inf,
        )
        vwap = (
            self.pq_sum / self.vol
            if self.vol > 0
            else (c if math.isfinite(c) else (prev_close or math.nan))
        )
        ret1 = (
            float(math.log(c) - math.log(prev_close))
            if prev_close
            and c
            and prev_close > 0
            and c > 0
            and math.isfinite(prev_close)
            and math.isfinite(c)
            else 0.0
        )
        imb = (self.buy_vol - self.sell_vol) / (
            self.buy_vol + self.sell_vol + 1e-9
        )
        return {
            "sec_close": float(c),
            "sec_vwap": float(vwap),
            "sec_vol": float(self.vol),
            "sec_signed_vol": float(self.signed_vol),
            "sec_trades": float(self.trades),
            "sec_buy_vol": float(self.buy_vol),
            "sec_sell_vol": float(self.sell_vol),
            "sec_ret1": float(ret1),
            "sec_imb": float(imb),
        }

    def reset(self):
        self.__init__()

class SecRing:
    def __init__(self, L: int):
        self.L = L
        self.buf = deque(maxlen=L)
        self.prev_close: Optional[float] = None
        self.last_second: Optional[int] = None
        self._log_closes = deque(maxlen=64)  # for multi-second returns

    def push_finalized(self, d: Dict[str, float]):
        # fill NaNs for price-like, zero for counts
        for k in ("sec_close", "sec_vwap"):
            if not (isinstance(d[k], float) and math.isfinite(d[k])) and (
                self.prev_close is not None
            ):
                d[k] = float(self.prev_close)
        for k in (
            "sec_vol",
            "sec_signed_vol",
            "sec_trades",
            "sec_buy_vol",
            "sec_sell_vol",
            "sec_ret1",
            "sec_imb",
        ):
            if not (isinstance(d[k], float) and math.isfinite(d[k])):
                d[k] = 0.0

        # compute multi-second returns BEFORE appending current close to history
        lc_now = None
        if math.isfinite(d["sec_close"]) and d["sec_close"] > 0:
            lc_now = math.log(d["sec_close"])
        elif self.prev_close and self.prev_close > 0:
            lc_now = math.log(self.prev_close)

        for k in (3, 5, 10, 15, 30):
            key = f"sec_ret{k}"
            if lc_now is None or len(self._log_closes) < k:
                d[key] = 0.0
            else:
                d[key] = float(lc_now - self._log_closes[-k])

        # update histories
        if lc_now is not None:
            self._log_closes.append(lc_now)
        if math.isfinite(d["sec_close"]):
            self.prev_close = d["sec_close"]

        # append in training order
        self.buf.append([d[c] for c in SEC_COLS])

    def to_array(self) -> Optional[np.ndarray]:
        if len(self.buf) < self.L:
            return None
        a = np.array(self.buf, dtype="float32")
        return a if a.shape == (self.L, len(SEC_COLS)) else None

# ============================ Live container state ============================
@dataclass
class LoadedAndState:
    lm: LoadedModel
    ring: SecRing
    mfb: MinuteFeatureBuilder
    sec_agg: SecAgg = field(default_factory=SecAgg)
    # Whether this symbol needs a full reseed (minute history + 300s ring reset)
    needs_reseed: bool = False
    # Placeholder for future L2/orderbook state if you want to attach depth streams
    orderbook: Dict[str, Any] = field(default_factory=dict)

# ============================ Seed + minute updates ============================
async def backfill_and_seed(
    state: LoadedAndState,
    session: aiohttp.ClientSession,
    limit=BACKFILL_MINUTES,
):
    # Futures klines for price/volume; index klines for basis features
    spot = await fetch_spot_klines_1m(session, state.lm.symbol, limit=limit)
    idx = await fetch_index_klines_1m(session, state.lm.symbol, limit=limit)
    df, cols = build_minute_features(spot, idx)
    state.lm.static_cols = cols
    state.mfb.static_cols = list(cols)
    state.mfb.seed_from_history(spot, idx, df)
    jlog(
        logging.INFO,
        "minute_seed_done",
        symbol=state.lm.symbol,
        rows=int(len(df)),
        start=str(df.index.min()),
        end=str(df.index.max()),
    )

async def minute_incremental_updater(
    state: LoadedAndState, jitter_sec: float = 0.0
):
    """
    Periodically updates the per-minute static feature builder.

    Logic:
      - Sleep to the next minute boundary + MINUTE_UPDATE_DELAY_SEC.
      - Try up to MINUTE_FETCH_MAX_RETRIES times (1s apart) to obtain a *new*
        1m kline whose open time > state.mfb.last_minute_ts.
      - As soon as we see such a kline, we call add_minute(...) once and stop.
      - If all attempts either error or show no new minute (kline open time
        <= last_minute_ts), we give up for this minute. Static features remain
        at the previous minute; predictions will keep using them during
        STATIC_FALLBACK_GRACE_SEC and later become stale.
    """
    if jitter_sec > 0:
        await asyncio.sleep(jitter_sec)

    async with aiohttp.ClientSession() as session:
        while True:
            # Sleep until just after the next minute boundary.
            now = int(_time.time())
            next_minute = (now // 60 + 1) * 60
            await asyncio.sleep(
                max(0.0, next_minute - now) + MINUTE_UPDATE_DELAY_SEC
            )

            # Try multiple times to see the new minute
            success = False
            for attempt in range(MINUTE_FETCH_MAX_RETRIES):
                try:
                    spot2 = await fetch_spot_last2_1m(session, state.lm.symbol)
                    idx2 = await fetch_index_last2_1m(session, state.lm.symbol)

                    if not spot2 or not idx2:
                        raise RuntimeError("empty kline response")

                    s = spot2[-1]
                    i = idx2[-1]

                    ts_open_ms = int(s[0])
                    ts = pd.to_datetime(ts_open_ms, unit="ms", utc=True)

                    # If this is not strictly newer than what we already have,
                    # the new kline probably isn't ready yet.
                    if (
                        state.mfb.last_minute_ts is not None
                        and ts <= state.mfb.last_minute_ts
                    ):
                        jlog(
                            logging.INFO,
                            "minute_update_retry_no_new_bar",
                            symbol=state.lm.symbol,
                            attempt=int(attempt),
                            last_minute_ts=str(state.mfb.last_minute_ts),
                            latest_ts=str(ts),
                        )
                        # If more retries left, sleep and try again.
                        if attempt < MINUTE_FETCH_MAX_RETRIES - 1:
                            await asyncio.sleep(MINUTE_FETCH_RETRY_SLEEP_SEC)
                            continue
                        # No retries left -> give up for this minute.
                        break

                    # We have a new minute bar: build + add features.
                    spot_bar = {
                        "open": float(s[1]),
                        "high": float(s[2]),
                        "low": float(s[3]),
                        "close": float(s[4]),
                        "volume": float(s[5]),
                        "number_of_trades": int(s[8]),
                        "tb_base": float(s[9]),
                    }
                    idx_close = float(i[4]) if len(i) > 4 else None
                    state.mfb.add_minute(ts, spot_bar, idx_close)
                    success = True
                    break

                except Exception as e:
                    jlog(
                        logging.ERROR,
                        "minute_update_error_attempt",
                        symbol=state.lm.symbol,
                        attempt=int(attempt),
                        error=str(e),
                    )
                    if attempt < MINUTE_FETCH_MAX_RETRIES - 1:
                        await asyncio.sleep(MINUTE_FETCH_RETRY_SLEEP_SEC)
                        continue
                    else:
                        # Final failure for this minute.
                        break

            if not success:
                # We did not manage to add a new minute for this cycle.
                # Static features stay as-is; prediction side will
                # temporarily fall back to previous minute (see make_static_vec)
                # and eventually become stale.
                jlog(
                    logging.WARNING,
                    "minute_update_give_up",
                    symbol=state.lm.symbol,
                    last_minute_ts=str(state.mfb.last_minute_ts),
                )

# ============================ Scaling & prediction ============================
def transform_seq_with_scaler(X_seq: np.ndarray, scaler) -> np.ndarray:
    """
    Accepts sklearn StandardScaler or NumpyScaler (mean_/scale_).
    """
    B, L, D = X_seq.shape
    flat = X_seq.reshape(B * L, D)
    try:
        t = scaler.transform(flat)
        return t.reshape(B, L, D).astype("float32")
    except Exception:
        # manual path if mean_/scale_ present
        mean = getattr(scaler, "mean_", None)
        scale = getattr(scaler, "scale_", None)
        if mean is None or scale is None:
            return X_seq
        t = (flat - mean) / (scale + 1e-9)
        return t.reshape(B, L, D).astype("float32")

def transform_stat_with_scaler(X_stat: np.ndarray, scaler) -> np.ndarray:
    try:
        return scaler.transform(X_stat).astype("float32")
    except Exception:
        mean = getattr(scaler, "mean_", None)
        scale = getattr(scaler, "scale_", None)
        if mean is None or scale is None:
            return X_stat
        return ((X_stat - mean) / (scale + 1e-9)).astype("float32")

def call_predict_params_t(model, X_seq, X_stat, device: str):
    # utils.predict_params_t signature is (model, Xseq_s, Xstat_s, batch=...)
    try:
        return predict_params_t(model, X_seq, X_stat, batch=1)
    except TypeError:
        try:
            return predict_params_t(model, X_seq, X_stat, 1)
        except TypeError:
            return predict_params_t(model, X_seq, X_stat)

def make_static_vec(state: LoadedAndState, ex_sec: int) -> Optional[np.ndarray]:
    """
    Build the static feature vector at execution second `ex_sec`.

    Behavior:
      - Normally uses the minute row for floor(ex_sec, 1min).
      - If that row is missing but we are only a few seconds into a *new* minute
        and we do have the previous minute row, we temporarily reuse the
        previous minute's static features (fallback).
      - After STATIC_FALLBACK_GRACE_SEC seconds into a minute with no new
        static row, we return None so the caller treats the model as stale.
    """
    t = pd.to_datetime(ex_sec, unit="s", utc=True)
    t_floor = t.floor("min")

    # Try exact minute first
    base = state.mfb.get_row(t_floor)

    if base is None:
        lm_ts = state.mfb.last_minute_ts

        # We only consider fallback if we *do* have a previous minute row
        # and that row is exactly one minute before t_floor.
        if lm_ts is not None and lm_ts < t_floor:
            minute_gap = (t_floor - lm_ts).total_seconds()
            seconds_into_minute = (t - t_floor).total_seconds()

            if (
                abs(minute_gap - 60.0) <= 1.0  # previous minute
                and 0.0 <= seconds_into_minute <= STATIC_FALLBACK_GRACE_SEC
            ):
                prev_row = state.mfb.get_row(lm_ts)
                if prev_row is not None:
                    base = prev_row

        # If still None, we really have no usable static features.
        if base is None:
            return None

    # At this point `base` is the minute feature vector (possibly fallback).
    end_t = end_of_current_15min(t_floor)
    tau_sec = max(1, int((end_t - t).total_seconds()))
    tau_frac = tau_sec / 900.0
    tau_block = np.array(
        [
            float(tau_sec),
            math.sqrt(tau_frac),
            math.sqrt(900.0 / max(tau_sec, 1.0)),
            math.sin(2 * math.pi * tau_frac),
            math.cos(2 * math.pi * tau_frac),
        ],
        dtype="float32",
    )
    return np.concatenate([base.astype("float32"), tau_block], axis=0)

# ============================ Streaming & per-second prediction ============================
async def maybe_predict(state: LoadedAndState, ex_sec: int):
    # Raw sequence from SecRing
    Xseq_raw = state.ring.to_array()
    if Xseq_raw is None:
        # Not enough seconds yet
        return

    # Static vector (minute features + tau block)
    xstat = make_static_vec(state, ex_sec)
    if xstat is None:
        # Minute/static features not available for this time
        return

    # Shape into batch form
    X_seq = Xseq_raw[None, :, :].astype("float32")
    X_stat = xstat[None, :].astype("float32")

    # Scale
    X_seq_s = transform_seq_with_scaler(X_seq, state.lm.sec_scaler)
    X_stat_s = transform_stat_with_scaler(X_stat, state.lm.stat_scaler)

    # Call model
    iv, df = call_predict_params_t(
        state.lm.model, X_seq_s, X_stat_s, state.lm.device
    )
    iv_v, df_v = float(iv[0]), float(df[0])
    ex_time = (
        pd.to_datetime(ex_sec, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    jlog(
        logging.INFO,
        "prediction",
        ex_time=ex_time,
        symbol=state.lm.symbol,
        iv=iv_v,
        df=df_v,
    )

async def _stream_symbol_once(state: LoadedAndState):
    """
    Single Binance websocket session for one symbol.

    - Reads futures aggTrades
    - Maintains per-second aggregation -> SecRing
    - Calls maybe_predict() once per finalized second
    """
    symbol = state.lm.symbol.lower()
    url = f"{BINANCE_WS}/{symbol}@aggTrade"
    ring = state.ring
    sec_agg = state.sec_agg

    async with websockets.connect(
        url, ping_interval=20, ping_timeout=20
    ) as ws:
        jlog(
            logging.INFO,
            "stream_connected",
            symbol=state.lm.symbol,
            url=url,
        )
        async for msg in ws:
            try:
                j = json.loads(msg)
            except Exception:
                continue

            ts_ms = int(j.get("T") or j.get("E") or 0)
            t_sec = ts_ms // 1000
            price = float(j["p"])
            qty = float(j["q"])
            # buyer_is_maker => taker is seller
            is_buy = not bool(j.get("m", True))

            if ring.last_second is None:
                ring.last_second = t_sec

            if t_sec != ring.last_second:
                prev_sec = ring.last_second
                d = sec_agg.finalize(ring.prev_close)
                ring.push_finalized(d)
                sec_agg.reset()
                ring.last_second = t_sec
                await maybe_predict(state, prev_sec)

            sec_agg.update(price, qty, is_buy)

async def stream_symbol(state: LoadedAndState):
    """
    High-level Binance aggTrade streamer with reconnect + reseed:

    - On first call: assumes caller already did backfill_and_seed.
    - On any websocket error:
        * logs binance_ws_dead
        * marks state.needs_reseed = True
    - On next iteration:
        * resets SecRing + SecAgg (drop old 300s window)
        * runs backfill_and_seed(...) to rebuild minute/static context
        * logs binance_reseed_begin / binance_reseed_complete
        * logs binance_ws_open
        * resumes streaming with _stream_symbol_once(...)
    """
    sym = state.lm.symbol

    while True:
        try:
            # If previous run died, fully reseed the context.
            if state.needs_reseed:
                # Reset second-level state: drop old 300s window
                state.ring = SecRing(L=state.lm.L_sec)
                state.sec_agg = SecAgg()

                jlog(
                    logging.INFO,
                    "binance_reseed_begin",
                    symbol=sym,
                    L_sec=int(state.lm.L_sec),
                )
                try:
                    async with aiohttp.ClientSession() as session:
                        await backfill_and_seed(state, session)
                except Exception as e:
                    jlog(
                        logging.ERROR,
                        "binance_reseed_error",
                        symbol=sym,
                        error=str(e),
                    )
                    # Try again later
                    await asyncio.sleep(2.0)
                    continue

                state.needs_reseed = False
                jlog(
                    logging.INFO,
                    "binance_reseed_complete",
                    symbol=sym,
                    L_sec=int(state.lm.L_sec),
                )

            # At this point we have a fresh context; start websocket stream.
            jlog(logging.INFO, "binance_ws_open", symbol=sym)
            await _stream_symbol_once(state)

        except asyncio.CancelledError:
            jlog(logging.INFO, "binance_ws_cancelled", symbol=sym)
            raise

        except Exception as e:
            # Any unexpected error is treated as websocket death.
            jlog(
                logging.ERROR,
                "stream_error",
                symbol=sym,
                error=str(e),
            )
            jlog(
                logging.ERROR,
                "binance_ws_dead",
                symbol=sym,
                error=str(e),
            )

            # Mark that on the next loop we must reseed + reset 300s window.
            state.needs_reseed = True

            # Small backoff before attempting to reconnect & reseed.
            await asyncio.sleep(1.0)
