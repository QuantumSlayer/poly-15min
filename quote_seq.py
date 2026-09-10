#!/usr/bin/env python3
"""
quote_seq.py — Online 32-tick sequence builder + *quantile* transformer inference (CPU-optimized)
+ deterministic "quote score" / "one-sided" decision for market making.

Adds (in predict_for_base status):
  - status["quote_score"]      : float in [-1, +1]  (direction + strength)
  - status["quote_delta_p"]    : suggested probability shift (e.g. +0.005 means +0.5 cents)
  - status["quote_one_sided"]  : bool (if True, consider quoting only one side / widen opposite)
  - status["quote_mode"]       : string ("normal" or "tail_strong")
  - status["quote_debug"]      : per-horizon diagnostics (p_up, z, iqr, active, tail_strong, score_h)

Policy is intentionally *lenient* by default (you said you want frequent small skew, since MM).
You can tighten by passing QuotePolicyParams(...) to SeqQuoteEngine.

Key training-time semantics mirrored here
---------------------------------------
(1) Sequence columns + order:
    meta_transformer.json -> meta["cb_cols"] (includes "log_ts_s")

(2) Price-like cb_* normalization (same as build_seq.py):
    If snapshot contains "coinbase" > 0, then for each cb_col:
      - if col endswith "_px" AND ("vs_" not in col) AND ("bp" not in col): divide by coinbase
      - if col == "cb_spread_abs": divide by coinbase

(3) Window recentering (same as build_seq.py):
    base_time = max(log_ts_s) over window (fallback: max(cb_last_ts))
    then recenter (if present in cb_cols):
      - log_ts_s           -= base_time
      - cb_last_ts         -= base_time
      - cb_last_trade_ts_s -= base_time
      - cb_ts_server_ms     = (cb_ts_server_ms/1000) - base_time

(4) Static vector (same as build_seq.py):
    x_static = [ base_id,
                 cb_last (n_cb),
                 mean(window) (n_cb),
                 std(window)  (n_cb),
                 static_extra_features... ]
    where static_extra_features are from meta["static_extra_features"]
    (default: ["tod_sin","tod_cos","is_weekend"] computed at the anchor tick)

(5) Non-finite handling (same as train_seq.py dataset loader):
    Before normalization: non-finite x_seq / x_static -> 0.0

(6) Model output:
    model(...) -> (B, n_targets * n_quantiles) in scaled space.
    Unscale to raw label space with train_seq.TARGET_SCALES:
      q_raw = q_scaled / target_scale[target]

Public API
----------
engine = SeqQuoteEngine(run_dir="runs_seq_t", data_dir="data_seq32")

engine.add_tick(snapshot)  # ALWAYS call this for each cb_tick_pred snapshot

preds_median, status = engine.predict_for_base("BTC")

- preds_median: {target: median_prediction_float}  (q=0.50)
- status["quantiles"]: {target: {"0.10":..., "0.25":..., "0.50":..., ...}}  (raw label space)
- status includes quote score fields described above.
"""

from __future__ import annotations

import json
import math
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# train_seq.py defines the model + quantile list + scaling used during training
from train_seq import SeqTransformerModel, QUANTILES, TARGET_SCALES  # type: ignore[import]

LOGGER = logging.getLogger(__name__)


# ----------------------------
# Timestamp parsing helpers
# ----------------------------
def _parse_log_ts_to_epoch(v: Any) -> Optional[float]:
    """
    Parse log_ts like '2025-11-25T22:37:37Z.245Z' or numeric seconds into Unix seconds (UTC).
    Mirrors build_seq.py.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)

    s = str(v).strip()
    if not s:
        return None

    # '2025-11-25T22:37:37Z.245Z' -> '2025-11-25T22:37:37.245+00:00'
    s_norm = s.replace("Z.", ".").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s_norm)
    except Exception:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return float(dt.timestamp())


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _sign(x: float, eps: float = 1e-18) -> float:
    if x > eps:
        return 1.0
    if x < -eps:
        return -1.0
    return 0.0


def p_up_from_quantiles(q10: float, q25: float, q50: float, q75: float, q90: float) -> float:
    """
    Estimate P(Y > 0) from 5 quantiles by linear interpolation of F(0).
    Robust to slight non-monotonic quantiles (enforces monotone).
    """
    taus = [0.10, 0.25, 0.50, 0.75, 0.90]
    qs = [float(q10), float(q25), float(q50), float(q75), float(q90)]

    # enforce monotone
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]

    # if everything above 0
    if 0.0 < qs[0]:
        return 1.0 - taus[0]

    # if everything below 0
    if 0.0 > qs[-1]:
        return 1.0 - taus[-1]

    # find first q >= 0
    k_hi = None
    for k, q in enumerate(qs):
        if q >= 0.0:
            k_hi = k
            break
    if k_hi is None:
        return 0.5
    if k_hi == 0:
        return 1.0 - taus[0]

    k_lo = k_hi - 1
    q_lo, q_hi = qs[k_lo], qs[k_hi]
    t_lo, t_hi = taus[k_lo], taus[k_hi]
    denom = (q_hi - q_lo) if abs(q_hi - q_lo) > 1e-18 else 1e-18
    frac = (0.0 - q_lo) / denom
    frac = _clip(frac, 0.0, 1.0)
    F0 = t_lo + frac * (t_hi - t_lo)
    return float(1.0 - _clip(F0, 0.0, 1.0))


def skew_norm_from_quantiles(q10: float, q50: float, q90: float) -> float:
    """
    Tail asymmetry measure in roughly [-1, +1]:
      >0 => fatter/right tail (bullish skew)
      <0 => fatter/left tail (bearish skew)
    """
    up = float(q90) - float(q50)
    dn = float(q50) - float(q10)
    denom = abs(up) + abs(dn) + 1e-18
    return float(_clip((up - dn) / denom, -1.0, 1.0))


@dataclass
class QuotePolicyParams:
    """
    Lenient defaults for market making:
      - pup_thr ~ 0.58 => allow mild directional imbalance
      - z_min   ~ 0.20 => allow small effect-size
      - z_scale ~ 2.0  => tanh(|z|/z_scale) sizing
      - base_delta_p ~ 0.02 => skew mid by up to ~2 cents at score=1.0
      - one_side_thr ~ 0.80  => only one-side when quite strong (or tail-strong)
    """
    # gating / sizing
    pup_thr: float = 0.58
    z_min: float = 0.20
    z_scale: float = 2.0

    # horizon weights (2s/5s/10s)
    w2: float = 0.50
    w5: float = 0.35
    w10: float = 0.15

    # how much to skew your binary fair mid
    base_delta_p: float = 0.02

    # one-sided decision
    one_side_thr: float = 0.80

    # “tail strong” override: if q10>0 or q90<0 AND |z| >= tail_z_min => mode="tail_strong"
    tail_z_min: float = 0.60


@dataclass
class BaseBuffer:
    base: str
    seq: deque[np.ndarray]
    window: np.ndarray
    static: np.ndarray
    last_seg_ts_s: Optional[float] = None
    segment_len: int = 0
    last_gap_sec: Optional[float] = None
    last_anchor_abs_ts_s: Optional[float] = None  # for tod/weekend (absolute, not recentered)


class SeqQuoteEngine:
    def __init__(
        self,
        run_dir: Path | str = "runs_seq_t",
        data_dir: Path | str = "data_seq32",
        device: Optional[str] = None,
        checkpoint_name: str = "model_best.pt",
        max_gap_sec_override: Optional[float] = None,
        quote_policy: Optional[QuotePolicyParams] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.quote_policy = quote_policy or QuotePolicyParams()

        # ---- Load meta ----
        meta_path = self.data_dir / "meta_transformer.json"
        with meta_path.open("r", encoding="utf-8") as f:
            self.meta: Dict[str, Any] = json.load(f)

        self.cb_cols: List[str] = list(self.meta["cb_cols"])
        self.targets: List[str] = list(self.meta["targets"])
        self.bases: List[str] = list(self.meta["bases"])
        self.seq_len: int = int(self.meta.get("seq_len", 32))
        self.max_gap_sec: float = float(
            max_gap_sec_override if max_gap_sec_override is not None else self.meta.get("max_gap_sec", 1.0)
        )

        self.static_extra: List[str] = list(self.meta.get("static_extra_features", ["tod_sin", "tod_cos", "is_weekend"]))
        self.base_to_id: Dict[str, int] = {b: i for i, b in enumerate(self.bases)}
        self.n_cb: int = len(self.cb_cols)

        # Static dim per build_seq.py: 1 + 3*n_cb + extra_dim
        self.feat_static_dim: int = 1 + 3 * self.n_cb + len(self.static_extra)

        # Indices for timestamp-like cb_* features (may be None if absent)
        self.idx_log_ts = self._safe_index("log_ts_s")
        self.idx_last_trade_ts = self._safe_index("cb_last_trade_ts_s")
        self.idx_last_ts = self._safe_index("cb_last_ts")
        self.idx_ts_server_ms = self._safe_index("cb_ts_server_ms")

        # Precompute indices for price normalization (same filter as build_seq.py)
        self._price_norm_idx: List[int] = []
        for i, c in enumerate(self.cb_cols):
            if c == "cb_spread_abs":
                self._price_norm_idx.append(i)
            elif c.endswith("_px") and ("vs_" not in c) and ("bp" not in c):
                self._price_norm_idx.append(i)

        # ---- Load normalization stats (per-base) ----
        norm_path = self.data_dir / "norm_stats.json"
        with norm_path.open("r", encoding="utf-8") as f:
            norm_stats_raw: Dict[str, Any] = json.load(f)

        self.norm_torch: Dict[str, Dict[str, torch.Tensor]] = {}
        for base in self.bases:
            st = norm_stats_raw[base]
            self.norm_torch[base] = {
                "seq_mean": torch.tensor(st["seq"]["mean"], dtype=torch.float32),
                "seq_std": torch.tensor(st["seq"]["std"], dtype=torch.float32),
                "static_mean": torch.tensor(st["static"]["mean"], dtype=torch.float32),
                "static_std": torch.tensor(st["static"]["std"], dtype=torch.float32),
            }

        # Infer dims from norms (source of truth)
        base0 = self.bases[0]
        self.seq_dim = int(self.norm_torch[base0]["seq_mean"].numel())
        self.static_dim = int(self.norm_torch[base0]["static_mean"].numel())
        if self.seq_dim != self.n_cb:
            LOGGER.warning("[SeqQuoteEngine] seq_dim(%d) != n_cb(%d); check meta/norm.", self.seq_dim, self.n_cb)
        if self.static_dim != self.feat_static_dim:
            LOGGER.warning(
                "[SeqQuoteEngine] static_dim(%d) != expected(%d); check meta/static features.",
                self.static_dim,
                self.feat_static_dim,
            )

        # ---- Device ----
        if device is None:
            device = "cpu"
        self.device = torch.device(device)

        # ---- Resolve run dir + load args/model ----
        self.run_dir = self._resolve_run_dir(Path(run_dir))
        args_path = self.run_dir / "args.json"
        if not args_path.exists():
            raise FileNotFoundError(f"args.json not found in run_dir: {self.run_dir}")

        with args_path.open("r", encoding="utf-8") as f:
            args: Dict[str, Any] = json.load(f)

        d_model = int(args.get("d_model", 256))
        n_heads = int(args.get("n_heads", 8))
        n_layers = int(args.get("n_layers", 4))
        dim_feedforward = int(args.get("dim_feedforward", 512))
        dropout = float(args.get("dropout", 0.1))

        self.quantiles: List[float] = list(QUANTILES)
        self.n_quantiles: int = len(self.quantiles)
        self.n_targets: int = len(self.targets)

        # Target scales (same as train_seq.py)
        self.target_scales_t = torch.tensor(
            [float(TARGET_SCALES.get(t, 1.0)) for t in self.targets],
            dtype=torch.float32,
            device=self.device,
        ).clamp_min(1e-12)

        model = SeqTransformerModel(
            seq_dim=self.seq_dim,
            static_dim=self.static_dim,
            n_bases=len(self.bases),
            n_targets=len(self.targets),
            seq_len=self.seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        ckpt_path = self.run_dir / checkpoint_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict) and "model_state" in obj:
            state = obj["model_state"]
        elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
            # raw state_dict OR training checkpoint dict without "model_state"
            if any(k.startswith("seq_proj.") or k.startswith("heads.") for k in obj.keys()):
                state = obj
            else:
                state = obj.get("state_dict", obj)
        else:
            state = obj

        model.load_state_dict(state)
        model.eval()
        self.model = model.to(self.device)

        # ---- Per-base buffers ----
        self.buffers: Dict[str, BaseBuffer] = {}
        for base in self.bases:
            self.buffers[base] = BaseBuffer(
                base=base,
                seq=deque(maxlen=self.seq_len),
                window=np.zeros((self.seq_len, self.n_cb), dtype=np.float32),
                static=np.zeros((self.static_dim,), dtype=np.float32),
            )

        # base_id tensors (avoid recreating every call)
        self.base_id_tensors: Dict[str, torch.Tensor] = {
            base: torch.tensor([idx], dtype=torch.long, device=self.device)
            for base, idx in self.base_to_id.items()
        }

        # target indices for the “quote score” policy (if present)
        self._tgt_idx: Dict[str, int] = {t: i for i, t in enumerate(self.targets)}
        self._idx_ret2 = self._tgt_idx.get("ret_2s")
        self._idx_ret5 = self._tgt_idx.get("ret_5s")
        self._idx_ret10 = self._tgt_idx.get("ret_10s")

        # normalize weights
        wsum = self.quote_policy.w2 + self.quote_policy.w5 + self.quote_policy.w10
        if wsum <= 1e-18:
            self._w2, self._w5, self._w10 = 1.0, 0.0, 0.0
        else:
            self._w2 = self.quote_policy.w2 / wsum
            self._w5 = self.quote_policy.w5 / wsum
            self._w10 = self.quote_policy.w10 / wsum

        LOGGER.info(
            "[SeqQuoteEngine] ready: device=%s run_dir=%s seq_len=%d n_cb=%d static_dim=%d max_gap_sec=%.3f",
            self.device,
            str(self.run_dir),
            self.seq_len,
            self.n_cb,
            self.static_dim,
            self.max_gap_sec,
        )

    # ----------------------------
    # Public API
    # ----------------------------
    def add_tick(self, snapshot: Dict[str, Any]) -> None:
        """
        Ingest one cb_tick_pred-like snapshot and update per-base rolling buffer.
        """
        base = str(snapshot.get("base", "")).upper()
        buf = self.buffers.get(base)
        if buf is None:
            return

        # Segment timestamp used for gap logic: prefer log_ts, else cb_last_ts, else cb_ts_server_ms
        seg_ts_s = self._extract_seg_ts_s(snapshot)

        # Gap / segment logic (mirror build_seq segmentation semantics)
        if buf.last_seg_ts_s is not None and seg_ts_s is not None:
            dt = seg_ts_s - buf.last_seg_ts_s
            if not (math.isfinite(dt) and 0.0 <= dt <= self.max_gap_sec):
                buf.seq.clear()
                buf.segment_len = 0
                buf.last_gap_sec = None
            else:
                buf.last_gap_sec = float(dt)
        else:
            # first tick or missing timestamps -> reset
            buf.seq.clear()
            buf.segment_len = 0
            buf.last_gap_sec = None

        buf.last_seg_ts_s = seg_ts_s

        # Anchor absolute ts for tod/weekend: prefer seg_ts_s, else cb_last_ts, else cb_ts_server_ms
        buf.last_anchor_abs_ts_s = self._extract_anchor_abs_ts_s(snapshot, fallback=seg_ts_s)

        # Build cb_* vector in meta["cb_cols"] order
        cb_vec = np.empty(self.n_cb, dtype=np.float32)

        # coinbase price for price normalization
        coinbase_px = _safe_float(snapshot.get("coinbase"))
        if coinbase_px is None or coinbase_px <= 0.0:
            coinbase_px = None

        # Provide log_ts_s feature (absolute) in the sequence
        log_ts_s = self._extract_log_ts_s(snapshot)

        for i, col in enumerate(self.cb_cols):
            if col == "log_ts_s":
                cb_vec[i] = float(log_ts_s) if log_ts_s is not None else np.nan
                continue
            v = snapshot.get(col)
            fv = _safe_float(v)
            cb_vec[i] = float(fv) if fv is not None else np.nan

        # Price-like normalization (same as build_seq.py)
        if coinbase_px is not None:
            for i in self._price_norm_idx:
                v = cb_vec[i]
                if math.isfinite(float(v)):
                    cb_vec[i] = float(v) / float(coinbase_px)

        buf.seq.append(cb_vec)
        buf.segment_len = min(buf.segment_len + 1, self.seq_len)

    def predict_for_base(self, base: str) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
        """
        Return median predictions per target (q=0.50) plus diagnostics and quote score.
        """
        base = str(base).upper()
        status: Dict[str, Any] = {
            "base": base,
            "ok": False,
            "reason": "",
            "n_ticks": 0,
            "segment_len": 0,
            "last_gap_sec": None,
            "max_gap_sec": self.max_gap_sec,
        }

        buf = self.buffers.get(base)
        if buf is None:
            status["reason"] = "unknown_base"
            return None, status

        status["n_ticks"] = len(buf.seq)
        status["segment_len"] = buf.segment_len
        status["last_gap_sec"] = buf.last_gap_sec

        if len(buf.seq) < self.seq_len or buf.segment_len < self.seq_len:
            status["reason"] = "seq_too_short"
            return None, status

        if buf.last_gap_sec is not None and not (0.0 <= buf.last_gap_sec <= self.max_gap_sec):
            status["reason"] = "gap_exceeds_max"
            return None, status

        q_raw = self._run_model_quantiles(base, buf)  # (n_targets, n_quantiles) in raw space

        # Build outputs
        preds_median: Dict[str, float] = {}
        quant_map: Dict[str, Dict[str, float]] = {}
        q50_idx = self._quantile_index(0.50)

        for ti, tname in enumerate(self.targets):
            q_list = q_raw[ti, :].tolist()
            quant_map[tname] = {f"{q:.2f}": float(q_list[qi]) for qi, q in enumerate(self.quantiles)}
            preds_median[tname] = float(q_raw[ti, q50_idx])

        status["ok"] = True
        status["reason"] = "ok"
        status["quantiles"] = quant_map

        # ---- Quote score / one-sided decision ----
        quote = self._compute_quote_signal(q_raw)
        status.update(quote)

        return preds_median, status

    # ----------------------------
    # Quote policy logic
    # ----------------------------
    def _compute_quote_signal(self, q_raw: np.ndarray) -> Dict[str, Any]:
        """
        Deterministic mapping from quantiles -> quote score for binary MM.

        Returns dict keys:
          quote_score, quote_delta_p, quote_one_sided, quote_mode, quote_debug
        """
        out: Dict[str, Any] = {
            "quote_score": 0.0,
            "quote_delta_p": 0.0,
            "quote_one_sided": False,
            "quote_mode": "none",
            "quote_debug": {},
        }

        if self._idx_ret2 is None or self._idx_ret5 is None or self._idx_ret10 is None:
            out["quote_mode"] = "targets_missing"
            return out

        # helper to compute horizon stats
        def horizon_stats(idx: int) -> Dict[str, float]:
            q10, q25, q50, q75, q90 = (float(q_raw[idx, 0]), float(q_raw[idx, 1]), float(q_raw[idx, 2]),
                                       float(q_raw[idx, 3]), float(q_raw[idx, 4]))
            # enforce monotone defensively for stats (don’t mutate original)
            qs = [q10, q25, q50, q75, q90]
            for i in range(1, 5):
                if qs[i] < qs[i - 1]:
                    qs[i] = qs[i - 1]
            q10, q25, q50, q75, q90 = qs

            iqr = max(0.0, q75 - q25)
            sigma = max(1e-12, iqr / 1.349)
            z = q50 / sigma
            pup = p_up_from_quantiles(q10, q25, q50, q75, q90)
            skew = skew_norm_from_quantiles(q10, q50, q90)
            tail_strong = (q10 > 0.0) or (q90 < 0.0)
            tail_dir = 1.0 if q10 > 0.0 else (-1.0 if q90 < 0.0 else 0.0)
            return {
                "q10": q10, "q25": q25, "q50": q50, "q75": q75, "q90": q90,
                "iqr": iqr, "sigma": sigma, "z": z, "p_up": pup,
                "skew": skew,
                "tail_strong": 1.0 if tail_strong else 0.0,
                "tail_dir": tail_dir,
            }

        hp = self.quote_policy
        s2 = horizon_stats(self._idx_ret2)
        s5 = horizon_stats(self._idx_ret5)
        s10 = horizon_stats(self._idx_ret10)

        # per-horizon score (lenient gating)
        def score_h(s: Dict[str, float]) -> Tuple[float, bool]:
            pup = s["p_up"]
            z = s["z"]
            # gating
            active_p = (pup >= hp.pup_thr) or (pup <= (1.0 - hp.pup_thr))
            active_z = abs(z) >= hp.z_min
            active = bool(active_p and active_z)
            if not active:
                return 0.0, False
            direction = _sign(pup - 0.5)
            mag = math.tanh(abs(z) / max(1e-12, hp.z_scale))
            return float(direction * mag), True

        sc2, a2 = score_h(s2)
        sc5, a5 = score_h(s5)
        sc10, a10 = score_h(s10)

        # tail-strong override (rare but decisive). Still lenient: only require |z| >= tail_z_min
        mode = "normal"
        tail = None
        for name, s in [("ret_2s", s2), ("ret_5s", s5), ("ret_10s", s10)]:
            if s["tail_strong"] > 0.5 and abs(s["z"]) >= hp.tail_z_min:
                tail = (name, float(s["tail_dir"]), float(s["z"]))
                break

        if tail is not None and abs(tail[1]) > 0.0:
            score = float(_clip(tail[1], -1.0, 1.0))
            mode = "tail_strong"
        else:
            score = float(_clip(self._w2 * sc2 + self._w5 * sc5 + self._w10 * sc10, -1.0, 1.0))

        one_sided = bool((abs(score) >= hp.one_side_thr) or (mode == "tail_strong"))

        delta_p = float(_clip(hp.base_delta_p * score, -hp.base_delta_p, hp.base_delta_p))

        out["quote_score"] = score
        out["quote_delta_p"] = delta_p
        out["quote_one_sided"] = one_sided
        out["quote_mode"] = mode

        out["quote_debug"] = {
            "params": {
                "pup_thr": hp.pup_thr,
                "z_min": hp.z_min,
                "z_scale": hp.z_scale,
                "w2": self._w2,
                "w5": self._w5,
                "w10": self._w10,
                "base_delta_p": hp.base_delta_p,
                "one_side_thr": hp.one_side_thr,
                "tail_z_min": hp.tail_z_min,
            },
            "ret_2s": {**s2, "score_h": sc2, "active": a2},
            "ret_5s": {**s5, "score_h": sc5, "active": a5},
            "ret_10s": {**s10, "score_h": sc10, "active": a10},
            "tail_override": {"horizon": tail[0], "dir": tail[1], "z": tail[2]} if tail is not None else None,
        }
        return out

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _safe_index(self, col: str) -> Optional[int]:
        try:
            return self.cb_cols.index(col)
        except ValueError:
            return None

    def _quantile_index(self, q: float) -> int:
        # Exact match preferred; fallback to closest
        try:
            return self.quantiles.index(q)
        except ValueError:
            best_i, best_d = 0, float("inf")
            for i, qq in enumerate(self.quantiles):
                d = abs(float(qq) - float(q))
                if d < best_d:
                    best_d, best_i = d, i
            return best_i

    def _extract_log_ts_s(self, snapshot: Dict[str, Any]) -> Optional[float]:
        """
        The log_ts_s sequence feature:
          prefer snapshot["log_ts_s"], else parse snapshot["log_ts"], else snapshot["log_ts_ms"]/1000.
        """
        v = snapshot.get("log_ts_s")
        fv = _safe_float(v)
        if fv is not None:
            return float(fv)

        v = snapshot.get("log_ts")
        ts = _parse_log_ts_to_epoch(v)
        if ts is not None:
            return float(ts)

        v = snapshot.get("log_ts_ms")
        fv = _safe_float(v)
        if fv is not None:
            return float(fv) / 1000.0

        return None

    def _extract_seg_ts_s(self, snapshot: Dict[str, Any]) -> Optional[float]:
        """
        Segmentation timestamp for gap logic:
          prefer log_ts_s (or parse log_ts/log_ts_ms), else cb_last_ts, else cb_ts_server_ms/1000.
        """
        ts = self._extract_log_ts_s(snapshot)
        if ts is not None:
            return ts

        v = snapshot.get("cb_last_ts")
        fv = _safe_float(v)
        if fv is not None:
            return float(fv)

        v = snapshot.get("cb_ts_server_ms")
        fv = _safe_float(v)
        if fv is not None:
            return float(fv) / 1000.0

        return None

    def _extract_anchor_abs_ts_s(self, snapshot: Dict[str, Any], fallback: Optional[float]) -> Optional[float]:
        """
        Anchor absolute timestamp for tod/weekend features (UTC).
        Prefer log_ts_s; fallback to cb_last_ts or cb_ts_server_ms.
        """
        ts = self._extract_log_ts_s(snapshot)
        if ts is not None:
            return ts

        v = snapshot.get("cb_last_ts")
        fv = _safe_float(v)
        if fv is not None:
            return float(fv)

        v = snapshot.get("cb_ts_server_ms")
        fv = _safe_float(v)
        if fv is not None:
            return float(fv) / 1000.0

        return fallback

    def _resolve_run_dir(self, p: Path) -> Path:
        """
        If p contains args.json, use it.
        Else if p is a root (e.g. runs_seq_t) with timestamp subdirs, pick the newest subdir containing args.json.
        """
        if p.is_file():
            return p.parent
        if (p / "args.json").exists():
            return p

        # choose latest subdir that has args.json
        if p.exists() and p.is_dir():
            subs = [d for d in p.iterdir() if d.is_dir() and (d / "args.json").exists()]
            if subs:
                subs_sorted = sorted(subs, key=lambda d: d.name)
                return subs_sorted[-1]

        raise FileNotFoundError(f"Could not resolve a run_dir with args.json from: {p}")

    def _compute_tod_weekend(self, abs_ts_s: Optional[float]) -> Tuple[float, float, float]:
        if abs_ts_s is None or not math.isfinite(abs_ts_s):
            return 0.0, 0.0, 0.0
        dt = datetime.fromtimestamp(float(abs_ts_s), tz=timezone.utc)
        seconds_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        angle = 2.0 * math.pi * (seconds_of_day / 86400.0)
        tod_sin = math.sin(angle)
        tod_cos = math.cos(angle)
        is_weekend = 1.0 if dt.weekday() >= 5 else 0.0
        return float(tod_sin), float(tod_cos), float(is_weekend)

    def _run_model_quantiles(self, base: str, buf: BaseBuffer) -> np.ndarray:
        """
        Build (x_seq, x_static) for the latest window and run quantile model.
        Returns q_raw: (n_targets, n_quantiles) in raw label space.
        """
        norm = self.norm_torch[base]

        # ---- Assemble window (seq_len, n_cb) ----
        window = buf.window
        # buf.seq is oldest->newest; len == seq_len here
        for t, vec in enumerate(buf.seq):
            window[t, :] = vec

        # ---- Recenter timestamps in-place (mirror build_seq.py) ----
        base_time: Optional[float] = None

        if self.idx_log_ts is not None:
            vals = window[:, self.idx_log_ts]
            mask = np.isfinite(vals)
            if mask.any():
                base_time = float(vals[mask].max())
                window[mask, self.idx_log_ts] = vals[mask] - base_time
        else:
            # fallback: use cb_last_ts if present
            if self.idx_last_ts is not None:
                vals = window[:, self.idx_last_ts]
                mask = np.isfinite(vals)
                if mask.any():
                    base_time = float(vals[mask].max())

        if base_time is not None:
            if self.idx_last_trade_ts is not None:
                vals = window[:, self.idx_last_trade_ts]
                mask = np.isfinite(vals)
                if mask.any():
                    window[mask, self.idx_last_trade_ts] = vals[mask] - base_time

            if self.idx_last_ts is not None:
                vals = window[:, self.idx_last_ts]
                mask = np.isfinite(vals)
                if mask.any():
                    window[mask, self.idx_last_ts] = vals[mask] - base_time

            if self.idx_ts_server_ms is not None:
                vals = window[:, self.idx_ts_server_ms]
                mask = np.isfinite(vals)
                if mask.any():
                    sec_vals = vals[mask] / 1000.0
                    window[mask, self.idx_ts_server_ms] = sec_vals - base_time

        # ---- Static features ----
        static = buf.static
        n_cb = self.n_cb

        # [ base_id, cb_last, mean, std, extras... ]
        static.fill(0.0)
        static[0] = float(self.base_to_id[base])

        cb_last = window[-1]          # (n_cb,)
        mean = window.mean(axis=0)    # (n_cb,)
        std = window.std(axis=0)      # (n_cb,)

        off = 1
        static[off: off + n_cb] = cb_last
        off += n_cb
        static[off: off + n_cb] = mean
        off += n_cb
        static[off: off + n_cb] = std
        off += n_cb

        # Extras (meta-driven order)
        tod_sin, tod_cos, is_weekend = self._compute_tod_weekend(buf.last_anchor_abs_ts_s)
        extra_values = {
            "tod_sin": tod_sin,
            "tod_cos": tod_cos,
            "is_weekend": is_weekend,
        }
        for i, name in enumerate(self.static_extra):
            static[off + i] = float(extra_values.get(name, 0.0))

        # ---- Sanitize non-finite to 0 before normalization (train_seq dataset behavior) ----
        window[~np.isfinite(window)] = 0.0
        static[~np.isfinite(static)] = 0.0

        # ---- Torch tensors + normalization ----
        x_seq_t = torch.from_numpy(window).to(self.device)           # (T, D)
        x_static_t = torch.from_numpy(static).to(self.device)        # (D_static,)

        seq_mean = norm["seq_mean"].to(self.device)
        seq_std = norm["seq_std"].to(self.device)
        static_mean = norm["static_mean"].to(self.device)
        static_std = norm["static_std"].to(self.device)

        x_seq_t = (x_seq_t - seq_mean) / seq_std
        x_static_t = (x_static_t - static_mean) / static_std

        x_seq_t = x_seq_t.unsqueeze(0)          # (1, T, D)
        x_static_t = x_static_t.unsqueeze(0)    # (1, D_static)
        base_id_t = self.base_id_tensors[base]  # (1,)

        with torch.no_grad():
            out = self.model(x_seq_t, x_static_t, base_id_t)  # (1, n_targets*n_quantiles)
            q_scaled = out.view(1, self.n_targets, self.n_quantiles)  # scaled label space
            q_raw = q_scaled / self.target_scales_t.view(1, self.n_targets, 1)  # raw label space
            q_raw = torch.nan_to_num(q_raw, nan=0.0, posinf=0.0, neginf=0.0)

        return q_raw[0].detach().cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    eng = SeqQuoteEngine()
    print("SeqQuoteEngine initialized.")
