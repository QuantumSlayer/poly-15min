#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
import logging

# ============================================================
# Config
# ============================================================

BASES: List[str] = ["BTC", "ETH", "SOL", "XRP"]

# Two outputs:
#   z[:, 0] -> predicted change in mid / 100  (so change itself is z[:,0] / 100)
#   z[:, 1] -> predicted half-spread (after ReLU)
LATENT_DIM: int = 2

# Loss / modelling hyper-params
HUBER_DELTA: float = 1.0
MID_LOSS_WEIGHT: float = 1.0
HS_LOSS_WEIGHT: float = 0.2   # half-spread is less important

# ============================================================
# Argument parsing / utils
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Train model to predict change in digital mid (and half-spread) from sequences"
    )
    p.add_argument("--data-dir", type=str, default="data_iv64")
    p.add_argument("--out-root", type=str, default="runs_iv_delta")

    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)

    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--max-steps-per-epoch", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)


def log(msg: str) -> None:
    logging.info(msg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_base_from_slug(slug: str) -> str:
    slug = slug.lower()
    if slug.startswith("btc"):
        return "BTC"
    if slug.startswith("eth"):
        return "ETH"
    if slug.startswith("sol"):
        return "SOL"
    if slug.startswith("xrp"):
        return "XRP"
    raise ValueError(f"Cannot infer base from slug={slug}")


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return 0.0 if not math.isfinite(v) else v
    if isinstance(obj, float):
        return 0.0 if not math.isfinite(obj) else obj
    return obj

# ============================================================
# Stats helpers
# ============================================================

def _init_vec_stats(D: int) -> Dict[str, np.ndarray]:
    return {
        "count": np.zeros(D, dtype=np.int64),
        "sum": np.zeros(D, dtype=np.float64),
        "sum_sq": np.zeros(D, dtype=np.float64),
    }


def _update_vec_stats(stats: Dict[str, np.ndarray], x: np.ndarray) -> None:
    if x.size == 0:
        return
    x = x.astype(np.float64)
    mask = np.isfinite(x)
    x_valid = np.where(mask, x, 0.0)
    stats["count"] += mask.sum(axis=0)
    stats["sum"] += x_valid.sum(axis=0)
    stats["sum_sq"] += (x_valid ** 2).sum(axis=0)


def _finalize_vec_stats(stats: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    cnt = stats["count"].astype(np.float64)
    mean = np.zeros_like(stats["sum"])
    std = np.ones_like(stats["sum"])
    mask = cnt > 0
    mean[mask] = stats["sum"][mask] / cnt[mask]
    var = np.zeros_like(stats["sum"])
    var[mask] = stats["sum_sq"][mask] / cnt[mask] - mean[mask] ** 2
    var = np.maximum(var, 0.0)
    std[mask] = np.sqrt(var[mask])
    std[std < 1e-6] = 1.0
    return mean, std


def compute_feature_and_label_stats(
    train_files: List[Path],
    D: int,
    idx_mid: int,
    idx_spread: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, float]:
    """
    Per-base feature mean/std (over masked timesteps),
    and global std for pm_mid and pm_spread/2.
    """
    feat_stats_per_base: Dict[str, Dict[str, np.ndarray]] = {
        b: _init_vec_stats(D) for b in BASES
    }

    mid_sum = mid_sum_sq = 0.0
    hs_sum = hs_sum_sq = 0.0
    mid_cnt = hs_cnt = 0

    for path in train_files:
        slug = path.stem
        base = infer_base_from_slug(slug)
        if base not in feat_stats_per_base:
            continue
        st = feat_stats_per_base[base]

        with np.load(path, allow_pickle=True) as z:
            X = z["X"]          # (N, T, D)
            Y = z["Y"]          # (N, L)
            mask = z["mask"]    # (N, T)

        N, T, D_ = X.shape
        assert D_ == D
        X_flat = X.reshape(N * T, D)
        mask_flat = mask.reshape(N * T)
        X_valid = X_flat[mask_flat.astype(bool)]
        _update_vec_stats(st, X_valid)

        mid = Y[:, idx_mid]
        spread = Y[:, idx_spread]
        hs = 0.5 * spread

        m_mask = np.isfinite(mid)
        hs_mask = np.isfinite(hs)

        mid_valid = mid[m_mask].astype(np.float64)
        hs_valid = hs[hs_mask].astype(np.float64)

        mid_cnt += mid_valid.size
        hs_cnt += hs_valid.size
        mid_sum += mid_valid.sum()
        hs_sum += hs_valid.sum()
        mid_sum_sq += (mid_valid ** 2).sum()
        hs_sum_sq += (hs_valid ** 2).sum()

    feat_mean: Dict[str, np.ndarray] = {}
    feat_std: Dict[str, np.ndarray] = {}
    for b in BASES:
        m, s = _finalize_vec_stats(feat_stats_per_base[b])
        feat_mean[b] = m
        feat_std[b] = s

    def _finish_label(sum_, sum_sq_, cnt_) -> float:
        if cnt_ <= 0:
            return 1.0
        mean = sum_ / cnt_
        var = sum_sq_ / cnt_ - mean * mean
        var = max(var, 0.0)
        std = math.sqrt(var)
        return std if std > 1e-6 else 1.0

    mid_std = _finish_label(mid_sum, mid_sum_sq, mid_cnt)
    hs_std = _finish_label(hs_sum, hs_sum_sq, hs_cnt)

    return feat_mean, feat_std, mid_std, hs_std

# ============================================================
# Dataset
# ============================================================

@dataclass
class NormPerBase:
    mean: np.ndarray
    std: np.ndarray


class IVDataset(IterableDataset):
    """
    Iterable over slug NPZ files.

    Each sample:
        x_seq:   (T, D)   normalized features
        mask:    (T,)     bool
        base_id: ()
        mid:     ()       pm_mid
        hs:      ()       pm_spread / 2
        bid:     ()       pm_best_bid
        ask:     ()       pm_best_ask
    """

    def __init__(
        self,
        files: List[Path],
        feature_names: List[str],
        feat_mean: Dict[str, np.ndarray],
        feat_std: Dict[str, np.ndarray],
        idx_mid: int,
        idx_spread: int,
        idx_bid: int,
        idx_ask: int,
        split: str,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.feature_names = feature_names
        self.idx_mid = idx_mid
        self.idx_spread = idx_spread
        self.idx_bid = idx_bid
        self.idx_ask = idx_ask
        self.split = split

        self.bases: List[str] = BASES
        self.base_to_id: Dict[str, int] = {b: i for i, b in enumerate(self.bases)}
        self.norm: Dict[str, NormPerBase] = {}
        for b in self.bases:
            self.norm[b] = NormPerBase(
                mean=feat_mean[b].astype(np.float32),
                std=feat_std[b].astype(np.float32),
            )

        if not self.files:
            raise RuntimeError(f"No NPZ files for split={split}")

        with np.load(self.files[0], allow_pickle=True) as z0:
            X0 = z0["X"]
            self.seq_len = int(X0.shape[1])
            self.seq_dim = int(X0.shape[2])

        total = 0
        for path in self.files:
            with np.load(path, allow_pickle=True) as z:
                total += int(z["X"].shape[0])
        self.total_samples = total

        log(
            f"[DATASET {split}] num_files={len(self.files)}, "
            f"total_samples={self.total_samples}, seq_len={self.seq_len}, seq_dim={self.seq_dim}"
        )

    def __len__(self) -> int:
        return self.total_samples

    def _iter_worker_files(self) -> List[Path]:
        worker = get_worker_info()
        if worker is None:
            indices = list(range(len(self.files)))
        else:
            per_worker = int(math.ceil(len(self.files) / worker.num_workers))
            start = worker.id * per_worker
            end = min(start + per_worker, len(self.files))
            indices = list(range(start, end))
        if self.split == "train":
            random.shuffle(indices)
        return [self.files[i] for i in indices]

    def __iter__(self):
        for path in self._iter_worker_files():
            slug = path.stem
            base = infer_base_from_slug(slug)
            base_id = self.base_to_id[base]
            norm = self.norm[base]

            with np.load(path, allow_pickle=True) as z:
                X = z["X"].astype(np.float32)        # (N, T, D)
                Y = z["Y"].astype(np.float32)        # (N, L)
                mask = z["mask"].astype(bool)        # (N, T)

            X[~np.isfinite(X)] = 0.0
            N = X.shape[0]

            mean = norm.mean.reshape(1, 1, -1)
            std = norm.std.reshape(1, 1, -1)
            X = (X - mean) / std

            mid = Y[:, self.idx_mid]
            spread = Y[:, self.idx_spread]
            hs = 0.5 * spread
            bid = Y[:, self.idx_bid]
            ask = Y[:, self.idx_ask]

            for i in range(N):
                yield (
                    torch.from_numpy(X[i]),
                    torch.from_numpy(mask[i]),
                    torch.tensor(base_id, dtype=torch.long),
                    torch.tensor(mid[i], dtype=torch.float32),
                    torch.tensor(hs[i], dtype=torch.float32),
                    torch.tensor(bid[i], dtype=torch.float32),
                    torch.tensor(ask[i], dtype=torch.float32),
                )


def collate_batch(batch):
    x_seq = torch.stack([b[0] for b in batch], dim=0)
    mask = torch.stack([b[1] for b in batch], dim=0)
    base_id = torch.stack([b[2] for b in batch], dim=0)
    mid = torch.stack([b[3] for b in batch], dim=0)
    hs = torch.stack([b[4] for b in batch], dim=0)
    bid = torch.stack([b[5] for b in batch], dim=0)
    ask = torch.stack([b[6] for b in batch], dim=0)
    return x_seq, mask, base_id, mid, hs, bid, ask

# ============================================================
# Model (with per-base heads)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class IVTransformer(nn.Module):
    """
    Transformer encoder -> per-base heads (2 outputs per base):
      z[:, 0] -> delta_mid_scaled  (change in mid / 100)
      z[:, 1] -> half-spread (ReLUed)
    """

    def __init__(
        self,
        seq_dim: int,
        n_bases: int,
        seq_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_bases = n_bases

        self.seq_proj = nn.Linear(seq_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=seq_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Base embedding + per-base heads
        self.base_embed = nn.Embedding(n_bases, d_model)

        self.latent_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, LATENT_DIM),
                )
                for _ in range(n_bases)
            ]
        )

    def forward(self, x_seq: torch.Tensor, mask: torch.Tensor, base_id: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (B, T, D)
        mask:  (B, T) bool
        base_id: (B,)
        returns z: (B, LATENT_DIM)
        """
        h = self.seq_proj(x_seq)
        h = self.pos_enc(h)

        key_padding_mask = ~mask
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        h_last = h[:, -1, :]
        h_last = h_last + self.base_embed(base_id)

        B = h_last.size(0)
        device = h_last.device
        dtype = h_last.dtype
        z = torch.zeros(B, LATENT_DIM, device=device, dtype=dtype)

        for b in range(self.n_bases):
            mask_b = (base_id == b)
            if mask_b.any():
                z[mask_b] = self.latent_heads[b](h_last[mask_b])

        return z

# ============================================================
# Loss / metrics
# ============================================================

def huber_loss(diff: torch.Tensor, delta: float = HUBER_DELTA) -> torch.Tensor:
    abs_diff = diff.abs()
    d = torch.as_tensor(delta, device=diff.device, dtype=diff.dtype)
    quad = torch.minimum(abs_diff, d)
    lin = abs_diff - quad
    return 0.5 * quad * quad + d * lin


@dataclass
class BatchLossOut:
    loss: torch.Tensor
    loss_mid: torch.Tensor
    loss_spread: torch.Tensor
    mid_rmse: torch.Tensor
    hs_rmse: torch.Tensor
    mid_pred: torch.Tensor
    hs_pred: torch.Tensor
    delta_mid: torch.Tensor  # actual delta mid (after /100)


def compute_batch_loss(
    z: torch.Tensor,
    x_seq: torch.Tensor,
    base_id: torch.Tensor,
    mid_true: torch.Tensor,
    hs_true: torch.Tensor,
    feat_mean_t: torch.Tensor,
    feat_std_t: torch.Tensor,
    idx_feat_bid: int,
    idx_feat_ask: int,
    mid_std: float,
    hs_std: float,
) -> BatchLossOut:
    """
    Use model outputs to predict:
      - change in mid (from latest sequence best bid/ask) / 100
      - half-spread (direct)

    Then reconstruct pm_mid and half-spread:
      base_mid = 0.5 * (best_bid_last + best_ask_last)
      mid_pred = base_mid + delta_mid
      hs_pred  = relu(z[:,1])
    """
    device = z.device
    B, T, D = x_seq.shape

    mean_b = feat_mean_t[base_id]   # (B, D)
    std_b = feat_std_t[base_id]     # (B, D)

    x_last = x_seq[:, -1, :]        # (B, D)

    # Unnormalize latest best bid / ask from features
    bid_norm = x_last[:, idx_feat_bid]
    ask_norm = x_last[:, idx_feat_ask]

    bid_last = bid_norm * std_b[:, idx_feat_bid] + mean_b[:, idx_feat_bid]
    ask_last = ask_norm * std_b[:, idx_feat_ask] + mean_b[:, idx_feat_ask]

    bid_last = torch.where(torch.isfinite(bid_last), bid_last, torch.zeros_like(bid_last))
    ask_last = torch.where(torch.isfinite(ask_last), ask_last, torch.zeros_like(ask_last))

    base_mid = 0.5 * (bid_last + ask_last)

    # Model outputs
    delta_mid_scaled = z[:, 0]                  # ~O(1)
    delta_mid = delta_mid_scaled / 100.0        # actual change in mid
    hs_pred = torch.relu(z[:, 1])               # half-spread >= 0

    mid_pred = base_mid + delta_mid

    # Standardized Huber loss
    mid_scale = torch.as_tensor(mid_std, device=device, dtype=z.dtype)
    hs_scale = torch.as_tensor(hs_std, device=device, dtype=z.dtype)

    mid_diff = (mid_pred - mid_true) / mid_scale
    hs_diff = (hs_pred - hs_true) / hs_scale

    loss_mid = huber_loss(mid_diff).mean()
    loss_spread = huber_loss(hs_diff).mean()

    loss = MID_LOSS_WEIGHT * loss_mid + HS_LOSS_WEIGHT * loss_spread

    mid_rmse = torch.sqrt(torch.mean((mid_pred - mid_true) ** 2))
    hs_rmse = torch.sqrt(torch.mean((hs_pred - hs_true) ** 2))

    return BatchLossOut(
        loss=loss,
        loss_mid=loss_mid,
        loss_spread=loss_spread,
        mid_rmse=mid_rmse,
        hs_rmse=hs_rmse,
        mid_pred=mid_pred,
        hs_pred=hs_pred,
        delta_mid=delta_mid,
    )

# ============================================================
# Train / eval loop
# ============================================================

def _summarize_scalar(count: int, s: float, s_sq: float, vmin: float, vmax: float) -> Dict[str, float]:
    if count <= 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = s / count
    var = s_sq / count - mean * mean
    var = max(var, 0.0)
    std = math.sqrt(var)
    return {
        "count": int(count),
        "mean": float(mean),
        "std": float(std),
        "min": float(vmin),
        "max": float(vmax),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    use_amp: bool,
    epoch: int,
    log_interval: int,
    feat_mean_t: torch.Tensor,
    feat_std_t: torch.Tensor,
    idx_feat_bid: int,
    idx_feat_ask: int,
    mid_std: float,
    hs_std: float,
    max_steps: Optional[int] = None,
    train: bool = True,
) -> Dict[str, Any]:
    phase = "TRAIN" if train else "EVAL"
    model.train(train)

    total_loss = 0.0
    total_mid_loss = 0.0
    total_spread_loss = 0.0

    # For global RMSE and crossing stats
    n_total = 0
    mid_sq_err_sum = 0.0
    hs_sq_err_sum = 0.0

    cross_bid_total = 0  # predicted bid > true ask
    cross_ask_total = 0  # predicted ask < true bid
    cross_any_total = 0

    # Per-base stats
    n_bases = len(BASES)
    base_counts = np.zeros(n_bases, dtype=np.int64)
    base_mid_sq_err = np.zeros(n_bases, dtype=np.float64)
    base_hs_sq_err = np.zeros(n_bases, dtype=np.float64)
    base_cross_bid = np.zeros(n_bases, dtype=np.int64)
    base_cross_ask = np.zeros(n_bases, dtype=np.int64)
    base_cross_any = np.zeros(n_bases, dtype=np.int64)

    # delta_mid stats (global + per-base)
    dmid_count = 0
    dmid_sum = 0.0
    dmid_sum_sq = 0.0
    dmid_min = float("inf")
    dmid_max = float("-inf")

    base_dmid_count = np.zeros(n_bases, dtype=np.int64)
    base_dmid_sum = np.zeros(n_bases, dtype=np.float64)
    base_dmid_sum_sq = np.zeros(n_bases, dtype=np.float64)
    base_dmid_min = np.full(n_bases, np.inf, dtype=np.float64)
    base_dmid_max = np.full(n_bases, -np.inf, dtype=np.float64)

    n_batches = 0

    for batch_idx, (x_seq, mask, base_id, mid, hs, bid, ask) in enumerate(loader):
        x_seq = x_seq.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        base_id = base_id.to(device, non_blocking=True)
        mid = mid.to(device, non_blocking=True)
        hs = hs.to(device, non_blocking=True)
        bid = bid.to(device, non_blocking=True)
        ask = ask.to(device, non_blocking=True)

        if train:
            assert optimizer is not None and scaler is not None
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                z = model(x_seq, mask, base_id)
                out = compute_batch_loss(
                    z=z,
                    x_seq=x_seq,
                    base_id=base_id,
                    mid_true=mid,
                    hs_true=hs,
                    feat_mean_t=feat_mean_t,
                    feat_std_t=feat_std_t,
                    idx_feat_bid=idx_feat_bid,
                    idx_feat_ask=idx_feat_ask,
                    mid_std=mid_std,
                    hs_std=hs_std,
                )
                loss = out.loss

            if not torch.isfinite(loss):
                log(f"[{phase}] epoch={epoch} batch={batch_idx}: non-finite loss {loss.item()}, skipping batch")
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
                z = model(x_seq, mask, base_id)
                out = compute_batch_loss(
                    z=z,
                    x_seq=x_seq,
                    base_id=base_id,
                    mid_true=mid,
                    hs_true=hs,
                    feat_mean_t=feat_mean_t,
                    feat_std_t=feat_std_t,
                    idx_feat_bid=idx_feat_bid,
                    idx_feat_ask=idx_feat_ask,
                    mid_std=mid_std,
                    hs_std=hs_std,
                )
                loss = out.loss

            if not torch.isfinite(loss):
                log(f"[{phase}] epoch={epoch} batch={batch_idx}: non-finite loss {loss.item()}, skipping batch")
                continue

        # --- aggregate loss ---
        total_loss += float(loss.detach().cpu().item())
        total_mid_loss += float(out.loss_mid.detach().cpu().item())
        total_spread_loss += float(out.loss_spread.detach().cpu().item())
        n_batches += 1

        # --- aggregate RMSE + crossing + delta_mid stats ---
        with torch.no_grad():
            mid_pred = out.mid_pred.detach().cpu().numpy()
            hs_pred = out.hs_pred.detach().cpu().numpy()
            mid_true = mid.detach().cpu().numpy()
            hs_true = hs.detach().cpu().numpy()
            bid_true = bid.detach().cpu().numpy()
            ask_true = ask.detach().cpu().numpy()
            base_id_np = base_id.detach().cpu().numpy()
            delta_mid_np = out.delta_mid.detach().cpu().numpy()

            # squared errors
            mid_sq = (mid_pred - mid_true) ** 2
            hs_sq = (hs_pred - hs_true) ** 2

            batch_n = mid_sq.size
            n_total += batch_n
            mid_sq_err_sum += float(mid_sq.sum())
            hs_sq_err_sum += float(hs_sq.sum())

            # predicted bid/ask
            pred_bid = mid_pred - hs_pred
            pred_ask = mid_pred + hs_pred

            cross_bid = pred_bid > ask_true
            cross_ask = pred_ask < bid_true
            cross_any = np.logical_or(cross_bid, cross_ask)

            cross_bid_total += int(cross_bid.sum())
            cross_ask_total += int(cross_ask.sum())
            cross_any_total += int(cross_any.sum())

            # delta_mid global stats
            dmid_count += delta_mid_np.size
            dmid_sum += float(delta_mid_np.sum())
            dmid_sum_sq += float((delta_mid_np ** 2).sum())
            if delta_mid_np.size > 0:
                dmid_min = min(dmid_min, float(np.min(delta_mid_np)))
                dmid_max = max(dmid_max, float(np.max(delta_mid_np)))

            # per-base
            for b_idx in range(n_bases):
                mask_b = base_id_np == b_idx
                if not np.any(mask_b):
                    continue
                n_b = int(mask_b.sum())
                base_counts[b_idx] += n_b
                base_mid_sq_err[b_idx] += float(mid_sq[mask_b].sum())
                base_hs_sq_err[b_idx] += float(hs_sq[mask_b].sum())
                base_cross_bid[b_idx] += int(cross_bid[mask_b].sum())
                base_cross_ask[b_idx] += int(cross_ask[mask_b].sum())
                base_cross_any[b_idx] += int(cross_any[mask_b].sum())

                dmid_b = delta_mid_np[mask_b]
                base_dmid_count[b_idx] += n_b
                base_dmid_sum[b_idx] += float(dmid_b.sum())
                base_dmid_sum_sq[b_idx] += float((dmid_b ** 2).sum())
                if dmid_b.size > 0:
                    base_dmid_min[b_idx] = min(base_dmid_min[b_idx], float(np.min(dmid_b)))
                    base_dmid_max[b_idx] = max(base_dmid_max[b_idx], float(np.max(dmid_b)))

        if train and (batch_idx % log_interval == 0):
            avg_loss = total_loss / max(n_batches, 1)
            log(
                f"[{phase}] epoch={epoch} batch={batch_idx} "
                f"loss={avg_loss:.6g}"
            )

        if max_steps is not None and n_batches >= max_steps:
            break

    if n_batches == 0 or n_total == 0:
        return {
            "loss": float("inf"),
            "loss_mid": float("inf"),
            "loss_spread": float("inf"),
            "mid_rmse": float("inf"),
            "hs_rmse": float("inf"),
            "cross_frac_any": 0.0,
            "cross_frac_bid": 0.0,
            "cross_frac_ask": 0.0,
            "n_total": 0,
            "per_base": {},
            "delta_mid_stats": {"global": _summarize_scalar(0, 0.0, 0.0, 0.0, 0.0), "per_base": {}},
        }

    loss = total_loss / n_batches
    loss_mid = total_mid_loss / n_batches
    loss_spread = total_spread_loss / n_batches

    mid_rmse = math.sqrt(mid_sq_err_sum / max(n_total, 1))
    hs_rmse = math.sqrt(hs_sq_err_sum / max(n_total, 1))

    cross_frac_any = cross_any_total / n_total
    cross_frac_bid = cross_bid_total / n_total
    cross_frac_ask = cross_ask_total / n_total

    per_base: Dict[str, Dict[str, float]] = {}
    for b_idx, base in enumerate(BASES):
        n_b = int(base_counts[b_idx])
        if n_b <= 0:
            continue
        mid_rmse_b = math.sqrt(base_mid_sq_err[b_idx] / n_b)
        hs_rmse_b = math.sqrt(base_hs_sq_err[b_idx] / n_b)
        per_base[base] = {
            "n": n_b,
            "mid_rmse": mid_rmse_b,
            "hs_rmse": hs_rmse_b,
            "cross_frac_any": base_cross_any[b_idx] / n_b,
            "cross_frac_bid": base_cross_bid[b_idx] / n_b,
            "cross_frac_ask": base_cross_ask[b_idx] / n_b,
        }

    # delta_mid summaries
    dmid_global = _summarize_scalar(dmid_count, dmid_sum, dmid_sum_sq, dmid_min, dmid_max)
    dmid_per_base: Dict[str, Dict[str, float]] = {}
    for b_idx, base in enumerate(BASES):
        dmid_per_base[base] = _summarize_scalar(
            int(base_dmid_count[b_idx]),
            float(base_dmid_sum[b_idx]),
            float(base_dmid_sum_sq[b_idx]),
            float(base_dmid_min[b_idx]) if np.isfinite(base_dmid_min[b_idx]) else 0.0,
            float(base_dmid_max[b_idx]) if np.isfinite(base_dmid_max[b_idx]) else 0.0,
        )

    return {
        "loss": loss,
        "loss_mid": loss_mid,
        "loss_spread": loss_spread,
        "mid_rmse": mid_rmse,
        "hs_rmse": hs_rmse,
        "cross_frac_any": cross_frac_any,
        "cross_frac_bid": cross_frac_bid,
        "cross_frac_ask": cross_frac_ask,
        "n_total": int(n_total),
        "per_base": per_base,
        "delta_mid_stats": {
            "global": dmid_global,
            "per_base": dmid_per_base,
        },
    }

# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"

    train_files = sorted(train_dir.glob("*.npz"))
    valid_files = sorted(valid_dir.glob("*.npz"))

    if not train_files:
        raise FileNotFoundError(f"No train NPZ files in {train_dir}")
    if not valid_files:
        raise FileNotFoundError(f"No valid NPZ files in {valid_dir}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    log(f"Run directory: {run_dir}")

    with (run_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    cpu_cores = os.cpu_count() or 1
    num_threads = min(cpu_cores, 4)
    try:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
    except Exception:
        pass
    log(f"CPU cores={cpu_cores}, torch num_threads={num_threads}")

    # Inspect one file for names
    with np.load(train_files[0], allow_pickle=True) as z_ex:
        feature_names = z_ex["feature_names"].tolist()
        label_names = z_ex["label_names"].tolist()
        D = int(z_ex["X"].shape[2])

    log(f"Feature dim D={D}")
    log(f"First 10 features: {feature_names[:10]}")
    log(f"Labels: {label_names}")

    # Label indices
    if "pm_mid" not in label_names or "pm_spread" not in label_names:
        raise RuntimeError("Expected 'pm_mid' and 'pm_spread' in label_names")
    if "pm_best_bid" not in label_names or "pm_best_ask" not in label_names:
        raise RuntimeError("Expected 'pm_best_bid' and 'pm_best_ask' in label_names")

    idx_mid = label_names.index("pm_mid")
    idx_spread = label_names.index("pm_spread")
    idx_bid = label_names.index("pm_best_bid")
    idx_ask = label_names.index("pm_best_ask")

    # Feature indices used as base bid/ask
    if "pm_best_bid" not in feature_names or "pm_best_ask" not in feature_names:
        raise RuntimeError("Expected 'pm_best_bid' and 'pm_best_ask' in feature_names for base quotes")
    idx_feat_bid = feature_names.index("pm_best_bid")
    idx_feat_ask = feature_names.index("pm_best_ask")

    # --------------------------------------------------------
    # Normalization stats (features + label stds)
    #   - If norm_stats_iv.json exists, load it.
    #   - Otherwise compute and save it.
    # --------------------------------------------------------
    norm_stats_path = data_dir / "norm_stats_iv.json"
    if norm_stats_path.exists():
        log(f"[STATS] Loading normalization stats from {norm_stats_path}")
        with norm_stats_path.open("r") as f:
            norm_stats = json.load(f)

        # Convert back to numpy
        feat_mean_np: Dict[str, np.ndarray] = {
            b: np.asarray(norm_stats["feat_mean"][b], dtype=np.float32) for b in BASES
        }
        feat_std_np: Dict[str, np.ndarray] = {
            b: np.asarray(norm_stats["feat_std"][b], dtype=np.float32) for b in BASES
        }
        mid_std = float(norm_stats["mid_std"])
        hs_std = float(norm_stats["half_spread_std"])
    else:
        log("[STATS] Computing per-base feature stats and label stds from train set ...")
        feat_mean_np, feat_std_np, mid_std, hs_std = compute_feature_and_label_stats(
            train_files=train_files,
            D=D,
            idx_mid=idx_mid,
            idx_spread=idx_spread,
        )
        raw_stats = {
            "feat_mean": {b: feat_mean_np[b].tolist() for b in BASES},
            "feat_std": {b: feat_std_np[b].tolist() for b in BASES},
            "mid_std": mid_std,
            "half_spread_std": hs_std,
        }
        with norm_stats_path.open("w") as f:
            json.dump(sanitize_for_json(raw_stats), f, indent=2)
        log(f"[STATS] Saved normalization stats to {norm_stats_path}")

    log(f"[STATS] mid_std={mid_std:.6g}, half_spread_std={hs_std:.6g}")

    # Save stats for this run (including indices and names)
    stats_out = {
        "feature_names": feature_names,
        "label_names": label_names,
        "bases": BASES,
        "feat_mean": {b: feat_mean_np[b].tolist() for b in BASES},
        "feat_std": {b: feat_std_np[b].tolist() for b in BASES},
        "mid_std": mid_std,
        "half_spread_std": hs_std,
        "idx_mid": idx_mid,
        "idx_spread": idx_spread,
        "idx_bid": idx_bid,
        "idx_ask": idx_ask,
        "idx_feat_bid": idx_feat_bid,
        "idx_feat_ask": idx_feat_ask,
    }

    with (run_dir / "dataset_stats.json").open("w") as f:
        json.dump(sanitize_for_json(stats_out), f, indent=2)

    # --------------------------------------------------------
    # Datasets / loaders
    # --------------------------------------------------------
    train_dataset = IVDataset(
        files=train_files,
        feature_names=feature_names,
        feat_mean=feat_mean_np,
        feat_std=feat_std_np,
        idx_mid=idx_mid,
        idx_spread=idx_spread,
        idx_bid=idx_bid,
        idx_ask=idx_ask,
        split="train",
    )
    valid_dataset = IVDataset(
        files=valid_files,
        feature_names=feature_names,
        feat_mean=feat_mean_np,
        feat_std=feat_std_np,
        idx_mid=idx_mid,
        idx_spread=idx_spread,
        idx_bid=idx_bid,
        idx_ask=idx_ask,
        split="valid",
    )

    num_workers = max(0, min(args.num_workers, cpu_cores))
    log(f"Dataloader num_workers={num_workers}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    seq_dim = train_dataset.seq_dim
    seq_len = train_dataset.seq_len
    n_bases = len(BASES)

    model = IVTransformer(
        seq_dim=seq_dim,
        n_bases=n_bases,
        seq_len=seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    log(
        f"Model: seq_dim={seq_dim}, seq_len={seq_len}, "
        f"n_bases={n_bases}, output_dim={LATENT_DIM}"
    )
    log(str(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device="cuda" if device.type == "cuda" else "cpu", enabled=args.use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # Per-base feature stats as tensors (for un-normalization)
    feat_mean_t = torch.stack(
        [torch.from_numpy(feat_mean_np[b].astype(np.float32)) for b in BASES],
        dim=0,
    ).to(device)
    feat_std_t = torch.stack(
        [torch.from_numpy(feat_std_np[b].astype(np.float32)) for b in BASES],
        dim=0,
    ).to(device)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        log(f"========== Epoch {epoch}/{args.epochs} (lr={current_lr:.6g}) ==========")

        # ---------------- TRAIN ----------------
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=args.use_amp,
            epoch=epoch,
            log_interval=args.log_interval,
            feat_mean_t=feat_mean_t,
            feat_std_t=feat_std_t,
            idx_feat_bid=idx_feat_bid,
            idx_feat_ask=idx_feat_ask,
            mid_std=mid_std,
            hs_std=hs_std,
            max_steps=args.max_steps_per_epoch,
            train=True,
        )
        log(
            f"[TRAIN] epoch={epoch} "
            f"loss={train_stats['loss']:.6g} "
            f"loss_mid={train_stats['loss_mid']:.6g} "
            f"loss_spread={train_stats['loss_spread']:.6g} "
            f"mid_rmse={train_stats['mid_rmse']:.3e} "
            f"hs_rmse={train_stats['hs_rmse']:.3e} "
            f"cross_any={train_stats['cross_frac_any']:.4f} "
            f"cross_bid={train_stats['cross_frac_bid']:.4f} "
            f"cross_ask={train_stats['cross_frac_ask']:.4f}"
        )

        dmid_g = train_stats["delta_mid_stats"]["global"]
        log(
            f"[TRAIN] epoch={epoch} delta_mid_global "
            f"count={dmid_g['count']} mean={dmid_g['mean']:.3e} "
            f"std={dmid_g['std']:.3e} min={dmid_g['min']:.3e} max={dmid_g['max']:.3e}"
        )

        # Per-base train stats
        for base, st in train_stats["per_base"].items():
            dmid_b = train_stats["delta_mid_stats"]["per_base"].get(base, {})
            log(
                f"[TRAIN] epoch={epoch} base={base} "
                f"n={st['n']} "
                f"mid_rmse={st['mid_rmse']:.3e} "
                f"hs_rmse={st['hs_rmse']:.3e} "
                f"cross_any={st['cross_frac_any']:.4f} "
                f"cross_bid={st['cross_frac_bid']:.4f} "
                f"cross_ask={st['cross_frac_ask']:.4f} "
                f"delta_mid_mean={dmid_b.get('mean',0.0):.3e} "
                f"delta_mid_std={dmid_b.get('std',0.0):.3e}"
            )

        with (run_dir / f"train_stats_epoch{epoch:02d}.json").open("w") as f:
            json.dump(sanitize_for_json(train_stats), f, indent=2, allow_nan=False)

        # ---------------- VALID ----------------
        val_stats = run_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=args.use_amp,
            epoch=epoch,
            log_interval=args.log_interval,
            feat_mean_t=feat_mean_t,
            feat_std_t=feat_std_t,
            idx_feat_bid=idx_feat_bid,
            idx_feat_ask=idx_feat_ask,
            mid_std=mid_std,
            hs_std=hs_std,
            max_steps=None,
            train=False,
        )
        log(
            f"[VALID] epoch={epoch} "
            f"loss={val_stats['loss']:.6g} "
            f"loss_mid={val_stats['loss_mid']:.6g} "
            f"loss_spread={val_stats['loss_spread']:.6g} "
            f"mid_rmse={val_stats['mid_rmse']:.3e} "
            f"hs_rmse={val_stats['hs_rmse']:.3e} "
            f"cross_any={val_stats['cross_frac_any']:.4f} "
            f"cross_bid={val_stats['cross_frac_bid']:.4f} "
            f"cross_ask={val_stats['cross_frac_ask']:.4f}"
        )

        dmid_g_v = val_stats["delta_mid_stats"]["global"]
        log(
            f"[VALID] epoch={epoch} delta_mid_global "
            f"count={dmid_g_v['count']} mean={dmid_g_v['mean']:.3e} "
            f"std={dmid_g_v['std']:.3e} min={dmid_g_v['min']:.3e} max={dmid_g_v['max']:.3e}"
        )

        # Per-base valid stats
        for base, st in val_stats["per_base"].items():
            dmid_b = val_stats["delta_mid_stats"]["per_base"].get(base, {})
            log(
                f"[VALID] epoch={epoch} base={base} "
                f"n={st['n']} "
                f"mid_rmse={st['mid_rmse']:.3e} "
                f"hs_rmse={st['hs_rmse']:.3e} "
                f"cross_any={st['cross_frac_any']:.4f} "
                f"cross_bid={st['cross_frac_bid']:.4f} "
                f"cross_ask={st['cross_frac_ask']:.4f} "
                f"delta_mid_mean={dmid_b.get('mean',0.0):.3e} "
                f"delta_mid_std={dmid_b.get('std',0.0):.3e}"
            )

        with (run_dir / f"valid_stats_epoch{epoch:02d}.json").open("w") as f:
            json.dump(sanitize_for_json(val_stats), f, indent=2, allow_nan=False)

        # ---------------- CHECKPOINTS ----------------
        ckpt_path = run_dir / f"model_epoch{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "feature_names": feature_names,
                "label_names": label_names,
                "bases": BASES,
                "latent_dim": LATENT_DIM,
                "mode": "delta_mid",  # important for live code
                "idx_mid": idx_mid,
                "idx_spread": idx_spread,
                "idx_bid": idx_bid,
                "idx_ask": idx_ask,
                "idx_feat_bid": idx_feat_bid,
                "idx_feat_ask": idx_feat_ask,
            },
            ckpt_path,
        )
        log(f"[CKPT] Saved checkpoint to {ckpt_path}")

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_path = run_dir / "model_best.pt"
            torch.save(model.state_dict(), best_path)
            log(f"[CKPT] New best model (val_loss={best_val_loss:.6g}) saved to {best_path}")

        scheduler.step()

    log("Training finished.")


if __name__ == "__main__":
    main()
