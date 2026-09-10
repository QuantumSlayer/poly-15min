# utils.py
# Reusable utilities for data loading, feature engineering, model definition,
# training, inference, and calibration for the 15-min (random-τ) Student-t model.
# NOTE: Using Binance USDⓈ-M futures data for both bars and aggTrades. No funding.

import os
import re
import sys
import gc
import math
import time
import zipfile
from typing import List, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler

# ---------------- Logging ----------------
def log(msg: str) -> None:
    """UTC-timestamped logger."""
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}")
    sys.stdout.flush()

# ---------------- Globals / defaults ----------------
SYMBOL_DEFAULT = "BTCUSDT"

# IV transform constants (shared by model + plots).
IV_FLOOR: float = 0.0
IV_MULT:  float = 0.0005

# Training toggles
USE_COSINE: bool = True

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    except Exception:
        pass

# Cache for seconds aggregation from aggTrades
CACHE_DIR = "cache/sec"

# ---------------- DAILY path helpers (futures, no funding) ----------------
def _day_range(start_date: str, end_date: str) -> List[str]:
    """Inclusive list of YYYY-MM-DD strings."""
    idx = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")
    return [d.strftime("%Y-%m-%d") for d in idx.tz_convert("UTC")]

def _spot_paths(start_date: str,
                end_date: str,
                symbol: str = SYMBOL_DEFAULT,
                interval: str = "1m") -> List[str]:
    """
    Paths for *futures* 1m klines used as underlying price series.
    (Name kept for backward-compat with older code.)
    Futures USDⓈ-M:
      futures/um/daily/klines/SYMBOL/1m/SYMBOL-1m-YYYY-MM-DD.zip
    """
    days = _day_range(start_date, end_date)
    return [
        f"futures/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{d}.zip"
        for d in days
    ]

def _index_paths(start_date: str,
                 end_date: str,
                 symbol: str = SYMBOL_DEFAULT,
                 interval: str = "1m") -> List[str]:
    """
    Futures USDⓈ-M indexPriceKlines paths:
      futures/um/daily/indexPriceKlines/SYMBOL/1m/SYMBOL-1m-YYYY-MM-DD.zip
    """
    days = _day_range(start_date, end_date)
    return [
        f"futures/um/daily/indexPriceKlines/{symbol}/{interval}/{symbol}-{interval}-{d}.zip"
        for d in days
    ]

def _agg_paths(start_date: str,
               end_date: str,
               symbol: str = SYMBOL_DEFAULT) -> List[str]:
    """
    Futures USDⓈ-M aggTrades paths:
      futures/um/daily/aggTrades/SYMBOL/SYMBOL-aggTrades-YYYY-MM-DD.zip
    (Name kept for backward-compat with older code.)
    """
    days = _day_range(start_date, end_date)
    return [
        f"futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{d}.zip"
        for d in days
    ]

# ---------------- Schemas ----------------
# For futures aggTrades (header present):
#   agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker
AGG_COLS = [
    "agg_trade_id", "price", "qty",
    "first_trade_id", "last_trade_id",
    "timestamp", "is_buyer_maker"
]

# Klines: we only care about a subset, but keep a canonical list for reference.
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]

IDX_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]

# ---------------- IO helpers ----------------
def _read_zip_first_csv(zp: str, header=None, names=None) -> pd.DataFrame:
    with zipfile.ZipFile(zp) as zf:
        inner = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])[0]
        with zf.open(inner) as f:
            return pd.read_csv(f, header=header, names=names)

def _read_first_csv_from_zip(zip_path: str, nrows: int = 200) -> pd.DataFrame:
    """Generic small preview helper (no assumptions on header)."""
    with zipfile.ZipFile(zip_path) as zf:
        inner = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])[0]
        with zf.open(inner) as f:
            return pd.read_csv(f, header=None, nrows=nrows)

def _to_utc_auto(ts: pd.Series) -> pd.Series:
    ts = ts.astype("int64")
    unit = "us" if ts.max() >= 10**15 else "ms"
    return pd.to_datetime(ts, unit=unit, utc=True)

def _existing(paths: List[str], tag: str) -> List[str]:
    out, miss = [], []
    for p in paths:
        if os.path.exists(p):
            out.append(p)
        else:
            miss.append(p)
    if miss:
        log(f"[warn] {tag}: {len(miss)} missing file(s) will be skipped")
        for m in miss:
            log(f"[warn]   missing -> {m}")
    return out

# ---------------- Loaders (DAILY zips, futures) ----------------
def indexify_samples(sec_index: pd.DatetimeIndex,
                     min_index: pd.DatetimeIndex,
                     samples: list[dict],
                     L_sec: int):
    """
    Map sampled (t, end, tau) triples into integer indices for seconds+minute frames.
    """
    sec_pos = pd.Series(np.arange(len(sec_index), dtype=np.int64), index=sec_index)
    min_pos = pd.Series(np.arange(len(min_index), dtype=np.int64), index=min_index)
    out = []
    for s in samples:
        t = s["t"]
        end = s["end"]
        tau = s["tau"]
        m_floor = t.floor("min")
        if t not in sec_pos.index or end not in sec_pos.index or m_floor not in min_pos.index:
            continue
        i_end = int(sec_pos.loc[t])
        i0    = i_end - (L_sec - 1)
        if i0 < 0:  # not enough history
            continue
        out.append({
            "i0": i0,                       # start row for the seconds window
            "i_end": i_end,                 # end row for the seconds window (time t)
            "i_end15": int(sec_pos.loc[end]),  # end of 15m
            "imin": int(min_pos.loc[m_floor]), # minute features row
            "tau": int(tau),
            "t_ns": int(t.value),           # stable key if you want
        })
    return out

def load_spot_1m_from_list(zips: List[str]) -> pd.DataFrame:
    """
    Load 1m *futures* klines as the underlying bar series.

    Expects futures paths:
      futures/um/daily/klines/SYMBOL/1m/SYMBOL-1m-YYYY-MM-DD.zip

    Output index: UTC minute timestamps (open_time).
    Columns used:
      open, high, low, close, volume, number_of_trades, tb_base
    """
    zips = _existing(zips, "fut_1m")
    log(f"[load_fut_1m] {len(zips)} file(s)")
    frames = []
    for zp in zips:
        log(f"  -> {zp}")
        # Futures klines have a header row; use it and then rename
        df = _read_zip_first_csv(zp, header=0)
        # Standard futures kline columns: open_time, open, high, low, close,
        # volume, close_time, quote_volume, count, taker_buy_volume, ...
        df = df[["open_time", "open", "high", "low", "close", "volume",
                 "count", "taker_buy_volume"]].copy()
        df["open_time"] = _to_utc_auto(df["open_time"])
        df = df.rename(columns={
            "open_time": "ts",
            "count": "number_of_trades",
            "taker_buy_volume": "tb_base",
        })
        frames.append(df)

    if not frames:
        raise RuntimeError("No futures 1m kline data loaded.")

    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["ts"])
            .sort_values("ts")
            .set_index("ts"))

    for c in ["open", "high", "low", "close", "volume", "tb_base"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce")

    log(f"[load_fut_1m] rows={len(df):,}  {df.index.min()} → {df.index.max()}")
    return df

def load_index_1m_from_list(zips: List[str]) -> pd.DataFrame:
    """
    Load 1m indexPriceKlines for the same symbol.
    Paths:
      futures/um/daily/indexPriceKlines/SYMBOL/1m/SYMBOL-1m-YYYY-MM-DD.zip
    """
    zips = _existing(zips, "index_1m")
    log(f"[load_index_1m] {len(zips)} file(s)")
    frames = []
    for zp in zips:
        log(f"  -> {zp}")
        try:
            df = _read_zip_first_csv(zp, header=0)
            if "open_time" not in df.columns:
                raise ValueError
        except Exception:
            # Fallback for older format
            df = _read_zip_first_csv(zp, header=None, names=IDX_COLS)
        df["open_time"] = _to_utc_auto(df["open_time"])
        frames.append(
            df[["open_time", "close"]]
            .rename(columns={"open_time": "ts", "close": "index_close"})
        )
    if not frames:
        raise RuntimeError("No index 1m data loaded.")
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["ts"])
            .sort_values("ts")
            .set_index("ts"))
    df["index_close"] = pd.to_numeric(df["index_close"], errors="coerce")
    log(f"[load_index_1m] rows={len(df):,}  {df.index.min()} → {df.index.max()}")
    return df

# ---------------- Minute features (no funding) ----------------
def build_minute_features(spot_1m: pd.DataFrame,
                          index_1m: pd.DataFrame):
    """
    Build minute-level static features from:
      - futures 1m klines (spot_1m)
      - index 1m (index_1m)

    Returns:
      df: minute-indexed feature frame
      static_cols: list of feature column names
    """
    log("[features] start minute features (no funding)")
    df = spot_1m.copy()

    # Returns & vols
    logp = np.log(df["close"])
    df["ret_1"]  = logp.diff(1)
    df["ret_3"]  = logp.diff(3)
    df["ret_5"]  = logp.diff(5)
    df["rv_5"]   = df["ret_1"].rolling(5,  min_periods=5).std()
    df["rv_15"]  = df["ret_1"].rolling(15, min_periods=15).std()
    df["rv_ratio"] = df["rv_5"]/(df["rv_15"]+1e-9)

    # Bar stats
    df["hl_range"] = (df["high"] - df["low"]) / (df["close"].replace(0, np.nan))
    tbb_col = "taker_buy_base_asset_volume"
    if "taker_buy_base_asset_volume" in df.columns:
        tbb_col = "taker_buy_base_asset_volume"
    elif "tb_base" in df.columns:
        tbb_col = "tb_base"
    elif "taker_buy_volume" in df.columns:
        tbb_col = "taker_buy_volume"

    df["taker_buy_ratio"] = (df[tbb_col]) / (df["volume"] + 1e-9)
    df["taker_buy_ratio"] = df["taker_buy_ratio"].clip(0, 1)
    df["trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce")
    lv = np.log(df["volume"] + 1e-12)
    df["vol_z_60"] = (lv - lv.rolling(60).mean()) / (lv.rolling(60).std() + 1e-9)

    # Index join → basis
    df = df.join(index_1m[["index_close"]], how="left")
    df["basis_rel"] = (df["close"] - df["index_close"]) / (df["index_close"] + 1e-9)

    # Time encodings
    mins = df.index.hour * 60 + df.index.minute
    df["tod_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    df["tod_cos"] = np.cos(2 * np.pi * mins / 1440.0)

    # Weekend flag (Sat/Sun)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype("float32")

    static_cols = [
        "ret_1", "ret_3", "ret_5",
        "rv_5", "rv_15", "rv_ratio",
        "hl_range", "taker_buy_ratio", "trades", "vol_z_60",
        "basis_rel", "tod_sin", "tod_cos",
        "is_weekend",
    ]

    df = df[static_cols + ["close"]].dropna()
    log(f"[features] minute df rows={len(df):,}  {df.index.min()} → {df.index.max()}")
    return df, static_cols

# ---------------- aggTrades → seconds (parallel, futures) ----------------
def _cache_name_for_zip(zp: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = os.path.basename(zp)
    base = re.sub(r"\.zip$", "", base, flags=re.IGNORECASE)
    return os.path.join(CACHE_DIR, f"{base}.pkl")

def _read_first_csv_from_zip_safe(zip_path: str, nrows: int = 200) -> pd.DataFrame:
    """
    For aggTrades: we know there is a header row; read with header=0.
    """
    with zipfile.ZipFile(zip_path) as zf:
        inner = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])[0]
        with zf.open(inner) as f:
            return pd.read_csv(f, header=0, nrows=nrows)

def _detect_ts_unit_for_agg(zip_path: str) -> str:
    head = _read_first_csv_from_zip_safe(zip_path, nrows=200)
    # futures aggTrades: 'transact_time' or 'timestamp'-like at col 5
    ts = head.iloc[:, 5].astype("int64")
    return "us" if ts.max() >= 10**15 else "ms"

def _aggregate_one_zip_to_pickle(zp: str) -> str:
    out_path = _cache_name_for_zip(zp)
    if os.path.exists(out_path):
        return out_path

    unit = _detect_ts_unit_for_agg(zp)
    chunk_aggs = []
    with zipfile.ZipFile(zp) as zf:
        inner = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])[0]
        with zf.open(inner) as f:
            # futures aggTrades have a header row; we override column names with AGG_COLS
            for chunk in pd.read_csv(
                f,
                header=0,
                names=AGG_COLS,
                chunksize=1_000_000,
                dtype={
                    "agg_trade_id": "int64",
                    "price": "float32",
                    "qty": "float32",
                    "first_trade_id": "int64",
                    "last_trade_id": "int64",
                    "timestamp": "int64",
                    "is_buyer_maker": "object",
                },
            ):
                # Normalize booleans if needed
                for c in ["is_buyer_maker", "is_best_match"]:
                    if c in chunk.columns and chunk[c].dtype != bool:
                        chunk[c] = chunk[c].astype(str).str.lower().isin(["true", "1"])

                sign = (~chunk["is_buyer_maker"]).astype(np.int8) * 2 - 1
                sec = (chunk["timestamp"] // (1_000_000 if unit == "us" else 1_000)).astype("int64")

                chunk = chunk.assign(
                    sec=sec.values,
                    pq=(chunk["price"] * chunk["qty"]).astype("float64"),
                    signed_qty=(chunk["qty"].astype("float64") * sign.astype("int8")).astype("float64"),
                    buy_qty=np.where(sign > 0, chunk["qty"], 0.0).astype("float32"),
                    sell_qty=np.where(sign < 0, chunk["qty"], 0.0).astype("float32"),
                ).sort_values(["sec", "timestamp"], kind="mergesort")

                grp = chunk.groupby("sec", sort=True)
                agg = grp.agg(
                    open=("price", "first"),
                    high=("price", "max"),
                    low=("price", "min"),
                    close=("price", "last"),
                    vol=("qty", "sum"),
                    pq_sum=("pq", "sum"),
                    signed_vol=("signed_qty", "sum"),
                    buy_vol=("buy_qty", "sum"),
                    sell_vol=("sell_qty", "sum"),
                    trades=("price", "count"),
                )
                chunk_aggs.append(agg)
                del chunk, grp, agg
                gc.collect()

    if not chunk_aggs:
        raise RuntimeError("No aggTrades data.")

    sec_df = pd.concat(chunk_aggs).sort_index()
    sec_df = sec_df.groupby(level=0, sort=True).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vol": "sum",
        "pq_sum": "sum",
        "signed_vol": "sum",
        "trades": "sum",
        "buy_vol": "sum",
        "sell_vol": "sum",
    })

    sec_df["sec_vwap"] = np.where(sec_df["vol"] > 0, sec_df["pq_sum"] / sec_df["vol"], np.nan)
    sec_df = sec_df.rename(columns={
        "open": "sec_open",
        "high": "sec_high",
        "low": "sec_low",
        "close": "sec_close",
        "vol": "sec_vol",
        "signed_vol": "sec_signed_vol",
        "trades": "sec_trades",
        "buy_vol": "sec_buy_vol",
        "sell_vol": "sec_sell_vol",
    })

    idx = pd.to_datetime(sec_df.index.values, unit="s", utc=True)
    sec_df.index = idx
    sec_df = sec_df.sort_index().astype({
        "sec_open": "float32",
        "sec_high": "float32",
        "sec_low": "float32",
        "sec_close": "float32",
        "sec_vwap": "float32",
        "sec_vol": "float32",
        "sec_signed_vol": "float32",
        "sec_trades": "float32",
        "sec_buy_vol": "float32",
        "sec_sell_vol": "float32",
    })
    sec_df.to_pickle(out_path)
    return out_path

def aggtrades_to_seconds_parallel(zip_list: List[str]) -> pd.DataFrame:
    """
    Aggregate futures aggTrades into a 1-second grid with microstructure features.
    """
    zips = _existing(zip_list, "agg (sec)")
    log(f"[aggtrades_to_seconds_parallel] files={len(zips)} workers={max(1, (os.cpu_count() or 4) - 1)}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    out_paths, futures = [], []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        for zp in zips:
            outp = _cache_name_for_zip(zp)
            if os.path.exists(outp):
                out_paths.append(outp)
            else:
                futures.append(ex.submit(_aggregate_one_zip_to_pickle, zp))
        for fut in as_completed(futures):
            out_paths.append(fut.result())

    out_paths = sorted(out_paths)
    log(f"[aggtrades_to_seconds_parallel] cached parts={len(out_paths)}")
    frames = [pd.read_pickle(p) for p in out_paths]
    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        raise RuntimeError("No seconds frames.")

    log("[aggtrades_to_seconds_parallel] concat & fill ...")
    sec_df = pd.concat(frames).sort_index()
    sec_df = sec_df.groupby(level=0, sort=True).agg({
        "sec_open": "first",
        "sec_high": "max",
        "sec_low": "min",
        "sec_close": "last",
        "sec_vol": "sum",
        "sec_signed_vol": "sum",
        "sec_trades": "sum",
        "sec_buy_vol": "sum",
        "sec_sell_vol": "sum",
        "sec_vwap": "last",
    })

    full_idx = pd.date_range(
        sec_df.index.min().floor("s"),
        sec_df.index.max().ceil("s"),
        freq="s",
        tz="UTC"
    )
    sec_df = sec_df.reindex(full_idx)

    # Fill prices forwards, volumes with zeros
    for c in ["sec_close", "sec_vwap", "sec_open", "sec_high", "sec_low"]:
        sec_df[c] = sec_df[c].ffill()
    for c in ["sec_vol", "sec_signed_vol", "sec_trades", "sec_buy_vol", "sec_sell_vol"]:
        sec_df[c] = sec_df[c].fillna(0.0)

    # 1s return
    logp_sec = np.log(sec_df["sec_close"])
    sec_df["sec_ret1"] = logp_sec.diff(1).fillna(0.0).astype("float32")

    # Multi-second returns
    for k in (3, 5, 10, 15, 30):
        sec_df[f"sec_ret{k}"] = logp_sec.diff(k).fillna(0.0).astype("float32")

    # Imbalance
    denom = (sec_df["sec_buy_vol"] + sec_df["sec_sell_vol"] + 1e-9)
    sec_df["sec_imb"] = ((sec_df["sec_buy_vol"] - sec_df["sec_sell_vol"]) / denom).astype("float32")

    cast_map = {
        "sec_open": "float32",
        "sec_high": "float32",
        "sec_low": "float32",
        "sec_close": "float32",
        "sec_vwap": "float32",
        "sec_vol": "float32",
        "sec_signed_vol": "float32",
        "sec_trades": "float32",
        "sec_buy_vol": "float32",
        "sec_sell_vol": "float32",
        "sec_ret1": "float32",
        "sec_imb": "float32",
        "sec_ret3": "float32",
        "sec_ret5": "float32",
        "sec_ret10": "float32",
        "sec_ret15": "float32",
        "sec_ret30": "float32",
    }
    sec_df = sec_df.astype({k: v for k, v in cast_map.items() if k in sec_df.columns})

    log(f"[aggtrades_to_seconds_parallel] seconds={len(sec_df):,}")
    return sec_df

# ---------------- Time helper ----------------
def end_of_current_15min(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.floor("15min") + pd.Timedelta(minutes=15) - pd.Timedelta(seconds=1)

# ---------------- Random-day split with purge ----------------
def random_day_time_split(idx_ts,
                          val_frac: float = 0.2,
                          seed: int = 42,
                          horizon_min: int = 15,
                          L_sec: int = 300,
                          min_val_samples: int = 1024,
                          reseed_attempts: int = 10):
    if not isinstance(idx_ts, pd.DatetimeIndex):
        idx_ts = pd.DatetimeIndex(idx_ts)
    if idx_ts.tz is None:
        idx_ts = idx_ts.tz_localize("UTC")
    else:
        idx_ts = idx_ts.tz_convert("UTC")

    days = pd.DatetimeIndex(idx_ts.normalize()).unique()
    rng = np.random.default_rng(seed)
    purge_gap = pd.Timedelta(seconds=max(horizon_min * 60, L_sec))

    for k in range(reseed_attempts):
        val_days = set(rng.choice(days, size=max(1, int(len(days) * val_frac)), replace=False))
        va_mask = np.array(
            [d in val_days for d in pd.DatetimeIndex(idx_ts.normalize())],
            dtype=bool
        )
        tr_mask = ~va_mask

        # Purge around validation days
        for d in val_days:
            d_ts = pd.Timestamp(d)
            day_start = d_ts.tz_localize("UTC") if d_ts.tz is None else d_ts.tz_convert("UTC")
            day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            start_purge = day_start - purge_gap
            end_purge   = day_end   + purge_gap
            mask = ((idx_ts >= start_purge) & (idx_ts <= end_purge))
            tr_mask = tr_mask & (~mask)

        if va_mask.sum() >= min_val_samples and tr_mask.sum() > 0:
            log(
                "[split-randdays] days={} val_days={} purge_gap={}s  "
                "train={:,} valid={:,} (seed={})".format(
                    len(days), len(val_days),
                    int(max(horizon_min * 60, L_sec)),
                    tr_mask.sum(), va_mask.sum(), seed + k
                )
            )
            try:
                sel_days = sorted(pd.to_datetime(list(val_days)))[:10]
                sel_days_str = ", ".join(pd.Series(sel_days).dt.strftime("%Y-%m-%d").tolist())
                log(f"[split-randdays] chosen validation days (UTC): {sel_days_str} ...")
            except Exception:
                pass
            return tr_mask, va_mask
        else:
            seed += 7

    raise RuntimeError("Could not create a non-empty random-day split with purge.")

# ---------------- Scalers ----------------
def fit_transform_scalers(Xseq_tr: np.ndarray, Xstat_tr: np.ndarray):
    B, L, D = Xseq_tr.shape
    sec_scaler = StandardScaler()
    stat_scaler = StandardScaler()
    Xseq_tr_scaled = sec_scaler.fit_transform(Xseq_tr.reshape(-1, D)).reshape(B, L, D).astype("float32")
    Xstat_tr_scaled = stat_scaler.fit_transform(Xstat_tr).astype("float32")
    return sec_scaler, stat_scaler, Xseq_tr_scaled, Xstat_tr_scaled

def apply_scalers(sec_scaler: StandardScaler,
                  stat_scaler: StandardScaler,
                  Xseq: np.ndarray,
                  Xstat: np.ndarray):
    if len(Xseq) == 0:
        return Xseq, Xstat
    B, L, D = Xseq.shape
    Xseq_scaled = sec_scaler.transform(Xseq.reshape(-1, D)).reshape(B, L, D).astype("float32")
    Xstat_scaled = stat_scaler.transform(Xstat).astype("float32")
    return Xseq_scaled, Xstat_scaled

# ---------------- Student-t PDF / CDF / Quantiles (NumPy) ----------------
def _betaln_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lgamma = np.frompyfunc(math.lgamma, 1, 1)
    return (
        np.asarray(lgamma(a), dtype=np.float64)
        + np.asarray(lgamma(b), dtype=np.float64)
        - np.asarray(lgamma(a + b), dtype=np.float64)
    )

def _betacf_vec(a, b, x, max_iter=200, eps=1e-12):
    # Vectorized Lentz algorithm for continued fraction of incomplete beta
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    FPMIN = np.finfo(np.float64).tiny
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = np.ones_like(x)
    d = 1.0 - (qab * x / qap)
    d = 1.0 / np.where(np.abs(d) < FPMIN, FPMIN, d)
    h = d.copy()

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / np.where(np.abs(d) < FPMIN, FPMIN, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < FPMIN, FPMIN, c)
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / np.where(np.abs(d) < FPMIN, FPMIN, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < FPMIN, FPMIN, c)
        delh = d * c
        h *= delh
        if np.all(np.abs(delh - 1.0) < eps):
            break
    return h

def _reg_incomplete_beta(a, b, x):
    """Regularized incomplete beta I_x(a,b), vectorized & broadcast-safe."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    a_b, b_b, x_b = np.broadcast_arrays(a, b, x)
    out = np.empty_like(x_b, dtype=np.float64)

    out[x_b <= 0.0] = 0.0
    out[x_b >= 1.0] = 1.0

    mask_mid = (x_b > 0.0) & (x_b < 1.0)
    if np.any(mask_mid):
        xv = x_b[mask_mid]
        av = a_b[mask_mid]
        bv = b_b[mask_mid]

        use_direct = xv < (av + 1.0) / (av + bv + 2.0)
        Iv = np.empty_like(xv, dtype=np.float64)

        if np.any(use_direct):
            xx = xv[use_direct]
            aa = av[use_direct]
            bb = bv[use_direct]
            bt = np.exp(
                aa * np.log(xx)
                + bb * np.log1p(-xx)
                - np.log(aa)
                - _betaln_np(aa, bb)
            )
            Iv[use_direct] = bt * _betacf_vec(aa, bb, xx)

        if np.any(~use_direct):
            xx = xv[~use_direct]
            aa = av[~use_direct]
            bb = bv[~use_direct]
            bt = np.exp(
                bb * np.log1p(-xx)
                + aa * np.log(xx)
                - np.log(bb)
                - _betaln_np(aa, bb)
            )
            Iv[~use_direct] = 1.0 - bt * _betacf_vec(bb, aa, 1.0 - xx)

        out[mask_mid] = Iv

    return np.clip(out, 0.0, 1.0)

def student_t_cdf_np(x, df):
    """Standard Student-t CDF with df=ν (scale=1). Broadcasts over x, df."""
    x  = np.asarray(x, dtype=np.float64)
    df = np.asarray(df, dtype=np.float64)

    df_b, x_b = np.broadcast_arrays(df, x)
    a = 0.5 * df_b
    b = 0.5
    z = df_b / np.maximum(df_b + x_b * x_b, 1e-300)
    I = _reg_incomplete_beta(a, b, z)

    out = np.empty_like(x_b, dtype=np.float64)
    pos = (x_b >= 0.0)
    out[pos]  = 1.0 - 0.5 * I[pos]
    out[~pos] = 0.5 * I[~pos]
    return out

def student_t_pdf_np(x, df, sigma):
    """t_ν(0, sigma) PDF."""
    x = np.asarray(x, dtype=np.float64)
    df = np.asarray(df, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    z = x / (sigma + 1e-15)
    return np.exp(
        np.frompyfunc(math.lgamma, 1, 1)((df + 1.0)/2.0).astype(np.float64)
        - np.frompyfunc(math.lgamma, 1, 1)(df/2.0).astype(np.float64)
        - 0.5 * np.log(df * np.pi)
        - np.log(sigma + 1e-15)
        - 0.5 * (df + 1.0) * np.log1p((z * z) / df)
    )

def t_pit_values_np(r_true, iv, df, tau):
    sigma = iv * np.sqrt(np.maximum(tau, 1.0) / 900.0)
    z = r_true / (sigma + 1e-15)
    return student_t_cdf_np(z, df).astype(np.float64)

def t_nll_np(r_true, iv, df, tau):
    sigma = iv * np.sqrt(np.maximum(tau, 1.0) / 900.0)
    pdf = student_t_pdf_np(r_true, df, sigma)
    return float(-np.log(pdf + 1e-24).mean())

def t_quantiles_bisect(alphas, iv, df, tau, max_iter=40):
    alphas = np.asarray(alphas, dtype=np.float64)
    N = len(iv)
    K = len(alphas)
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=np.float64)
    sigma = iv * np.sqrt(np.maximum(tau, 1.0) / 900.0)
    base = np.maximum(0.05, 25.0 * sigma)
    lo = np.tile((-base).reshape(-1, 1), (1, K))
    hi = np.tile(( base).reshape(-1, 1), (1, K))
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cdf = student_t_cdf_np(mid / (sigma[:, None] + 1e-15), df[:, None])
        lo = np.where(cdf < alphas[None, :], mid, lo)
        hi = np.where(cdf >= alphas[None, :], mid, hi)
    return 0.5 * (lo + hi)

def reliability_quantile_table_t(r_true, iv, df, tau, alphas, max_samples=200_000, seed=123):
    N = len(r_true)
    if N == 0:
        return pd.DataFrame({
            "alpha": alphas,
            "empirical_coverage": np.nan,
            "error": np.nan,
            "n_used": 0
        })
    if N > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=max_samples, replace=False)
        r_true, iv, df, tau = r_true[idx], iv[idx], df[idx], tau[idx]
        N = len(r_true)
    Q = t_quantiles_bisect(np.array(alphas, dtype=np.float64), iv, df, tau, max_iter=40)
    cov = (r_true[:, None] <= Q).mean(axis=0)
    df_out = pd.DataFrame({"alpha": alphas, "empirical_coverage": cov})
    df_out["error"] = df_out["empirical_coverage"] - df_out["alpha"]
    df_out["n_used"] = N
    return df_out

def ece_uniform(u, n_bins=20) -> Tuple[float, float, np.ndarray]:
    u = np.asarray(u, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    mids = 0.5 * (bins[:-1] + bins[1:])
    eces = []
    counts = []
    diffs = []
    for j in range(n_bins):
        if j < n_bins - 1:
            mask = (u >= bins[j]) & (u < bins[j+1])
        else:
            mask = (u >= bins[j]) & (u <= bins[j+1])
        if mask.any():
            avg = u[mask].mean()
            eces.append(abs(avg - mids[j]))
            counts.append(mask.sum())
            diffs.append(avg - mids[j])
        else:
            eces.append(0.0)
            counts.append(0)
            diffs.append(0.0)
    w = np.array(counts) / max(1, sum(counts))
    ece = float((w * np.array(eces)).sum())
    mce = float(np.max(np.abs(diffs))) if len(diffs) else 0.0
    return ece, mce, np.array(counts, dtype=int)

def ks_uniform_test(u):
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = len(u)
    if n == 0:
        return 0.0, 1.0
    i = np.arange(1, n+1)
    d_plus = np.max(i/n - u)
    d_minus = np.max(u - (i-1)/n)
    D = max(d_plus, d_minus)
    en = math.sqrt(n)
    lam = (en + 0.12 + 0.11/en) * D
    s = 0.0
    for k in range(1, 101):
        s += (-1)**(k-1) * math.exp(-2*(k*k)*(lam*lam))
    p = max(0.0, min(1.0, 2*s))
    return float(D), float(p)

# ---------------- Model ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.v(torch.tanh(self.w(x))), 1)
        return (a * x).sum(1)

class ResidualMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 96, dropout: float = 0.20):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_in)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.norm(x + h)

class SeqTransformerT(nn.Module):
    """
    Transformer encoder over seconds features + static minute features.
    Head outputs 2 params per sample: [iv_raw, df_raw]
      - iv = 0.0004 * softplus(iv_raw)           (per √15min)
      - df = 16.0   / softplus(df_raw)           (degrees of freedom)
    """
    def __init__(self,
                 sec_d: int,
                 static_d: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 depth: int = 3,
                 dropout: float = 0.20):
        super().__init__()
        self.proj = nn.Linear(sec_d, d_model)
        self.pe = PositionalEncoding(d_model, max_len=4096)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*3,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.pool = AttnPool(d_model)

        self.static_mlp = nn.Sequential(
            nn.Linear(static_d, 64), nn.SiLU(),
            ResidualMLP(64, d_hidden=96, dropout=dropout),
            nn.Linear(64, 32), nn.SiLU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_model + 32, 64), nn.SiLU(),
            ResidualMLP(64, d_hidden=96, dropout=dropout),
            nn.Dropout(dropout)
        )
        self.head = nn.Linear(64, 2)  # [iv_raw, df_raw]

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor):
        h = self.proj(x_seq)
        h = self.pe(h)
        h = self.encoder(h)
        h = self.pool(h)
        s = self.static_mlp(x_static)
        f = self.fuse(torch.cat([h, s], dim=1))
        p = self.head(f)
        iv = IV_MULT * F.softplus(p[:, 0])
        df = 16.0 / (F.softplus(p[:, 1]) + 1e-8)
        return iv, df

# ---------------- Dataset wrapper ----------------
class SeqDataset(Dataset):
    """Returns (X_seq, X_static, z_true, tau_sec, w0, idx) for weighted NLL."""
    def __init__(self,
                 Xseq: np.ndarray,
                 Xstat: np.ndarray,
                 z: np.ndarray,
                 tau_sec: np.ndarray,
                 w0: np.ndarray = None):
        self.Xseq = Xseq
        self.Xstat = Xstat
        self.z = z
        self.tau = tau_sec
        if w0 is None:
            w0 = np.ones((len(z),), dtype="float32")
        self.w0 = w0.astype("float32")
        self.idx = np.arange(len(self.z), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(self, i: int):
        return self.Xseq[i], self.Xstat[i], self.z[i], self.tau[i], self.w0[i], self.idx[i]

# ---------------- Device helper ----------------
def to_device(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    return torch.tensor(x, dtype=dtype, device=device)

# ---------------- Training (Student-t NLL) ----------------
def _torch_studentt_logpdf(x, sigma, df):
    # log f_t(x; 0, sigma, df)
    z = x / (sigma + 1e-12)
    return (
        torch.lgamma((df + 1.0)/2.0)
        - torch.lgamma(df/2.0)
        - 0.5 * torch.log(df * torch.pi)
        - torch.log(sigma + 1e-12)
        - 0.5 * (df + 1.0) * torch.log1p((z*z) / torch.clamp(df, min=1e-8))
    )

def train_student_t(model: nn.Module,
                    train_loader: DataLoader,
                    valid_loader: DataLoader,
                    epochs: int = 4,
                    lr: float = 1e-3,
                    wd: float = 5e-4,
                    patience: int = 3,
                    use_cosine: bool = USE_COSINE,
                    # probability weighting
                    r_thresh_bp: float = 0.0,      # barrier in basis points for p=Pr(r>barrier)
                    prob_update: str = "epoch",    # "epoch", "batch", or "off"
                    epoch_callback=None,
                    ):
    """
    Train Student-t model with weighted NLL and probability-based reweighting.

    Returns:
      model  : model with best-validation weights loaded
      history: list of dicts {"epoch", "train_nll", "valid_nll"}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[train] device={device} epochs={epochs} lr={lr} wd={wd}")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = None
    if use_cosine:
        total_steps = max(1, epochs * max(1, len(train_loader)))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr * 0.1)

    best_val = float("inf")
    best_state = None
    wait = 0

    # Global probability multipliers for the whole train set
    Ntrain = len(train_loader.dataset)
    prob_mult = np.ones((Ntrain,), dtype=np.float32)
    prob_next = prob_mult.copy()

    # ---- Q1 polynomial params ----
    A_Q1, B_Q1, C_Q1 = 0.20, 1.00, 0.00

    def _gamma_from_p(p_np: np.ndarray) -> np.ndarray:
        p = np.clip(p_np.astype(np.float32), 0.0, 1.0)
        x = 2.0 * p - 1.0
        g = p * (1.0 - p) * (A_Q1 + B_Q1 * (x * x) + C_Q1 * (x * x * x * x))
        g = np.clip(g, 0.0, 0.25)
        return g

    def _gamma_from_params(iv_t: torch.Tensor,
                           df_t: torch.Tensor,
                           tau_t: torch.Tensor) -> np.ndarray:
        iv = iv_t.detach().float().cpu().numpy()
        df = df_t.detach().float().cpu().numpy()
        tau = tau_t.detach().float().cpu().numpy()
        if r_thresh_bp == 0.0:
            p = np.full_like(iv, 0.5, dtype=np.float32)
        else:
            r_thr = (r_thresh_bp / 1e4)
            sigma = iv * np.sqrt(np.maximum(tau, 1.0) / 900.0)
            z = r_thr / (sigma + 1e-15)
            p = 1.0 - student_t_cdf_np(z, df).astype(np.float32)
        return _gamma_from_p(p)

    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        n = 0

        for batch in train_loader:
            if len(batch) == 4:
                xs, st, z_true, tau = batch
                w0_bt = None
                idx_bt = None
            else:
                xs, st, z_true, tau, w0_bt, idx_bt = batch

            xs = to_device(xs, device, torch.float32)
            st = to_device(st, device, torch.float32)
            z_true = to_device(z_true, device, torch.float32)
            tau = to_device(tau, device, torch.float32)

            if w0_bt is not None:
                w0_bt = to_device(w0_bt, device, torch.float32)
                idx_np = idx_bt.cpu().numpy()
                w_prob_bt = torch.tensor(prob_mult[idx_np], dtype=torch.float32, device=device)
                w_bt = torch.clamp(w0_bt * w_prob_bt, 1e-8, 1e3)
            else:
                w_bt = None

            r_true = z_true * torch.sqrt(torch.clamp(tau, min=1.0) / 900.0)

            opt.zero_grad(set_to_none=True)

            iv, df = model(xs, st)
            sigma = iv * torch.sqrt(torch.clamp(tau, min=1.0) / 900.0)
            logpdf = _torch_studentt_logpdf(r_true, sigma, df)
            nll_vec = -logpdf
            if w_bt is None:
                nll = nll_vec.mean()
            else:
                nll = (w_bt * nll_vec).sum() / (w_bt.sum() + 1e-12)

            nll.backward()
            opt.step()
            if sched is not None:
                sched.step()

            if (w0_bt is not None) and (prob_update in ("batch", "epoch")):
                gamma = _gamma_from_params(iv, df, tau)
                prob_next[idx_np] = gamma.astype(np.float32)
                if prob_update == "batch":
                    prob_mult[idx_np] = prob_next[idx_np]

            tr_loss += nll.detach().item() * len(z_true)
            n += len(z_true)

        if prob_update == "epoch":
            prob_mult[:] = prob_next

        tr_loss /= max(n, 1)

        # ---------- VALID ----------
        model.eval()
        va_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in valid_loader:
                xs, st, z_true, tau = batch[:4]
                xs = to_device(xs, device, torch.float32)
                st = to_device(st, device, torch.float32)
                z_true = to_device(z_true, device, torch.float32)
                tau = to_device(tau, device, torch.float32)
                r_true = z_true * torch.sqrt(torch.clamp(tau, min=1.0) / 900.0)

                iv, df = model(xs, st)
                sigma = iv * torch.sqrt(torch.clamp(tau, min=1.0) / 900.0)
                logpdf = _torch_studentt_logpdf(r_true, sigma, df)
                nll = -logpdf.mean()

                va_loss += nll.detach().item() * len(z_true)
                n += len(z_true)
        va_loss /= max(n, 1)
        dt = time.time() - t0
        log(f"[train] Epoch {ep:02d} | train_nll={tr_loss:.5f} | valid_nll={va_loss:.5f} | {dt:.1f}s")

        # Snapshot weights to CPU once per epoch
        state_dict_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": int(ep),
            "train_nll": float(tr_loss),
            "valid_nll": float(va_loss),
        })

        # Optional: let train.py save a checkpoint + metadata
        if epoch_callback is not None:
            try:
                epoch_callback(ep, float(tr_loss), float(va_loss), state_dict_cpu)
            except Exception as e:
                log(f"[train] epoch_callback failed at epoch {ep}: {e}")

        # Early-stopping tracking
        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = state_dict_cpu
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log("[train] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

# ---------------- Inference (Student-t) ----------------
@torch.no_grad()
def predict_params_t(model: nn.Module,
                     Xseq_s: np.ndarray,
                     Xstat_s: np.ndarray,
                     batch: int = 4096):
    """
    Simple FP32 inference helper on in-memory arrays.
    """
    device = next(model.parameters()).device
    ivs = []
    dfs = []
    model.eval()
    for i in range(0, len(Xseq_s), batch):
        xs = to_device(Xseq_s[i:i+batch], device, torch.float32)
        st = to_device(Xstat_s[i:i+batch], device, torch.float32)
        iv, df = model(xs, st)
        ivs.append(iv.float().cpu().numpy())
        dfs.append(df.float().cpu().numpy())
    if not ivs:
        return np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    return np.concatenate(ivs), np.concatenate(dfs)

# ---------------- Seconds feature columns ----------------
def seconds_feature_cols() -> list:
    # Must match the features produced in aggtrades_to_seconds_parallel(...)
    return [
        "sec_close", "sec_vwap",
        "sec_vol", "sec_signed_vol", "sec_trades", "sec_buy_vol", "sec_sell_vol",
        "sec_ret1", "sec_ret3", "sec_ret5", "sec_ret10", "sec_ret15", "sec_ret30",
        "sec_imb",
    ]

# ---------------- Sampling + Lazy dataset ----------------
def make_samples(min_df: pd.DataFrame,
                 L_sec: int = 300,
                 k_per_min: int = 1,
                 seed: int = 42) -> list[dict]:
    """
    Return a list of dicts: {"t": T_current, "end": T_end15, "tau": int}
    where T_current = end - tau.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for m in min_df.index:
        end = end_of_current_15min(m)
        taus = rng.integers(5, 901, size=int(k_per_min))
        for tau in taus:
            t = end - pd.Timedelta(seconds=int(tau))
            samples.append({"t": t, "end": end, "tau": int(tau)})
    return samples

class LazySeqDataset(Dataset):
    def __init__(self,
                 sec_df: pd.DataFrame,
                 min_df: pd.DataFrame,
                 static_cols: list[str],
                 samples_idx: list[dict],
                 L_sec: int = 300):
        self.L = int(L_sec)
        self.sec_cols = seconds_feature_cols()
        self.sec_arr = sec_df[self.sec_cols].to_numpy(dtype=np.float32, copy=False)
        self.min_arr = min_df[static_cols].to_numpy(dtype=np.float32, copy=False)
        self.log_close = np.log(sec_df["sec_close"].to_numpy(dtype=np.float64, copy=False))
        self.samples = samples_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        i0, i_end, i_end15, imin = s["i0"], s["i_end"], s["i_end15"], s["imin"]
        tau = float(s["tau"])

        Xs = self.sec_arr[i0:i_end+1]
        Xst_base = self.min_arr[imin]

        tau_frac = tau / 900.0
        Xst_tau = np.array([
            tau,
            math.sqrt(tau_frac),
            math.sqrt(900.0 / max(tau, 1.0)),
            math.sin(2 * math.pi * tau_frac),
            math.cos(2 * math.pi * tau_frac),
        ], dtype=np.float32)
        Xst = np.concatenate([Xst_base, Xst_tau], axis=0)

        r = self.log_close[i_end15] - self.log_close[i_end]
        z = np.float32(r / (math.sqrt(tau/900.0) + 1e-12))

        lc_now = self.log_close[i_end]
        w0 = 0.0
        for k in (1, 3, 5, 10, 15, 30):
            if self.L > k:
                w0 += math.sqrt(abs(lc_now - self.log_close[i_end - k]) / k + 1e-15)
        w0 = np.float32(np.clip(w0, 1e-6, 50.0))

        return Xs, Xst, np.float32(z), np.float32(tau), w0, np.int64(i)

def fit_scalers_lazy(ds: LazySeqDataset,
                     batch_size: int = 512,
                     num_workers: int = 0) -> tuple[StandardScaler, StandardScaler]:
    """
    Incrementally fit StandardScalers on (seconds, static) features using the lazy dataset.
    """
    sec_scaler = StandardScaler()
    stat_scaler = StandardScaler()

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
        drop_last=False
    )

    first = True
    for Xs, Xst, _, _, _, _ in loader:
        B, L, D = Xs.shape
        Xs2d = Xs.reshape(B * L, D).numpy()
        Xst2d = Xst.numpy()

        if first:
            sec_scaler.partial_fit(Xs2d)
            stat_scaler.partial_fit(Xst2d)
            first = False
        else:
            sec_scaler.partial_fit(Xs2d)
            stat_scaler.partial_fit(Xst2d)

    return sec_scaler, stat_scaler

@torch.no_grad()
def predict_params_t_dataset(model: nn.Module,
                             ds: Dataset,
                             batch: int = 4096,
                             num_workers: int = 0):
    """
    Iterate a dataset (lazy or not) and return iv, df as arrays.
    Expects dataset to already apply scaling internally (or via ScaledModel).
    """
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
        drop_last=False
    )
    ivs = []
    dfs = []
    for Xs, Xst, *_ in loader:
        xs = to_device(Xs, device, torch.float32)
        st = to_device(Xst, device, torch.float32)
        iv, df = model(xs, st)
        ivs.append(iv.float().cpu().numpy())
        dfs.append(df.float().cpu().numpy())
    if not ivs:
        return np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    return np.concatenate(ivs), np.concatenate(dfs)

# ---------------- Public exports ----------------
__all__ = [
    # Logging
    "log",
    # Path helpers (DAILY; futures, no funding)
    "_day_range", "_spot_paths", "_index_paths", "_agg_paths",
    # IO loaders / helpers
    "_read_zip_first_csv", "_read_first_csv_from_zip", "_to_utc_auto", "_existing",
    "load_spot_1m_from_list", "load_index_1m_from_list",
    "aggtrades_to_seconds_parallel",
    # Feature engineering & dataset
    "build_minute_features", "end_of_current_15min",
    "random_day_time_split", "indexify_samples",
    # Scalers
    "fit_transform_scalers", "apply_scalers",
    # Model & training
    "SeqTransformerT", "SeqDataset", "to_device", "train_student_t",
    "predict_params_t",
    # Calibration utilities (Student-t)
    "student_t_cdf_np", "student_t_pdf_np", "t_pit_values_np",
    "t_quantiles_bisect", "t_nll_np", "reliability_quantile_table_t",
    "ece_uniform", "ks_uniform_test",
    # Globals
    "IV_FLOOR", "IV_MULT", "USE_COSINE", "CACHE_DIR",
    # Seconds features & lazy dataset
    "seconds_feature_cols", "make_samples", "LazySeqDataset",
    "fit_scalers_lazy", "predict_params_t_dataset",
]
