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

# ========================= Config =========================

LARGE_MOVE_SIGMA_MULT: float = 4.0
LARGE_MOVE_PROB_THRESHOLD: float = 0.10

QUANTILES: List[float] = [0.10, 0.25, 0.50, 0.75, 0.90]
N_QUANTILES: int = len(QUANTILES)
MEDIAN_INDEX: int = QUANTILES.index(0.50)

QUANTILE_WEIGHTS: Dict[float, float] = {0.10: 0.3, 0.25: 0.4, 0.50: 2.5, 0.75: 0.4, 0.90: 0.3}

# label → transformed scale (roughly O(1–10) range)
TARGET_SCALES: Dict[str, float] = {
    "ret_2s": 40000.0,
    "ret_5s": 30000.0,
    "ret_10s": 20000.0,
}

BASE_WEIGHTS: Dict[str, float] = {"BTC": 3.0, "ETH": 2.0, "XRP": 1.0, "SOL": 1.0}

MAX_TRANSFORMED_ABS: float = 1000.0


# ========================= Args / utils =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train quantile transformer on 32-tick sequence dataset")
    p.add_argument("--data-dir", type=str, default="data_seq32")
    p.add_argument("--meta-path", type=str, default=None)
    p.add_argument("--out-root", type=str, default="runs_seq_t")

    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--max-steps-per-epoch", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--norm-stats-path", type=str, default=None)
    p.add_argument("--force-recompute-norm", action="store_true")
    p.add_argument("--max-seq-samples", type=int, default=2_000_000)
    p.add_argument("--max-static-samples", type=int, default=2_000_000)
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


def load_meta(meta_path: Path) -> Dict[str, Any]:
    with meta_path.open("r") as f:
        return json.load(f)


def _sanitize_array(arr: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = fallback
    return arr


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


# ========================= Norm stats =========================

def compute_norm_stats_for_base(
    base: str,
    base_id: int,
    meta: Dict[str, Any],
    data_dir: Path,
    max_seq_samples: int,
    max_static_samples: int,
) -> Dict[str, Any]:
    train_dir = Path(meta.get("shards_train_dir", str(data_dir / "shards" / "train")))
    train_shards: List[str] = meta["train_shards"]

    seq_samples: List[np.ndarray] = []
    static_samples: List[np.ndarray] = []
    seq_count = static_count = 0

    for shard_name in train_shards:
        if not shard_name.startswith(base + "_"):
            continue
        shard_path = train_dir / shard_name
        if not shard_path.exists():
            continue

        with np.load(shard_path, allow_pickle=False) as z:
            x_seq = z["x_seq"]
            x_static = z["x_static"]
            base_idx = z["base_idx"]
            mask = (base_idx == base_id)
            if not np.any(mask):
                continue

            seq = x_seq[mask]
            static = x_static[mask]

            if seq_count < max_seq_samples:
                seq_flat = seq.reshape(-1, seq.shape[-1])
                n_flat, remaining = seq_flat.shape[0], max_seq_samples - seq_count
                k = min(remaining, n_flat)
                if k > 0:
                    idx = np.random.choice(n_flat, size=k, replace=False) if n_flat > k else slice(None)
                    seq_samples.append(seq_flat[idx])
                    seq_count += k

            if static_count < max_static_samples:
                n_win, remaining = static.shape[0], max_static_samples - static_count
                k = min(remaining, n_win)
                if k > 0:
                    idx = np.random.choice(n_win, size=k, replace=False) if n_win > k else slice(None)
                    static_samples.append(static[idx])
                    static_count += k

        if seq_count >= max_seq_samples and static_count >= max_static_samples:
            break

    if not seq_samples or not static_samples:
        raise RuntimeError(f"No samples collected for base {base}")

    seq_all = np.concatenate(seq_samples, axis=0).astype(np.float64)
    static_all = np.concatenate(static_samples, axis=0).astype(np.float64)
    seq_all[~np.isfinite(seq_all)] = np.nan
    static_all[~np.isfinite(static_all)] = np.nan

    seq_mean = np.nanmean(seq_all, axis=0)
    seq_std = np.nanstd(seq_all, axis=0)
    static_mean = np.nanmean(static_all, axis=0)
    static_std = np.nanstd(static_all, axis=0)

    seq_mean = _sanitize_array(seq_mean, 0.0)
    static_mean = _sanitize_array(static_mean, 0.0)
    seq_std = _sanitize_array(seq_std, 1.0)
    static_std = _sanitize_array(static_std, 1.0)
    seq_std[seq_std < 1e-6] = 1.0
    static_std[static_std < 1e-6] = 1.0

    log(
        f"[NORM] base={base}: seq_samples={seq_count}, static_samples={static_count}, "
        f"seq_dim={seq_all.shape[1]}, static_dim={static_all.shape[1]}"
    )

    return {
        "seq": {"mean": seq_mean.tolist(), "std": seq_std.tolist()},
        "static": {"mean": static_mean.tolist(), "std": static_std.tolist()},
    }


def compute_or_load_norm_stats(args: argparse.Namespace, meta: Dict[str, Any], data_dir: Path) -> Dict[str, Any]:
    norm_path = Path(args.norm_stats_path) if args.norm_stats_path else data_dir / "norm_stats.json"
    if norm_path.exists() and not args.force_recompute_norm:
        log(f"[NORM] Loading norm stats from {norm_path}")
        with norm_path.open("r") as f:
            return json.load(f)

    bases: List[str] = meta.get("bases", [])
    if not bases:
        raise RuntimeError("meta_transformer.json must contain 'bases'")

    base_to_id = {b: i for i, b in enumerate(bases)}
    log("[NORM] Computing per-base normalization stats ...")
    stats: Dict[str, Any] = {}
    for b in bases:
        stats[b] = compute_norm_stats_for_base(
            base=b,
            base_id=base_to_id[b],
            meta=meta,
            data_dir=data_dir,
            max_seq_samples=args.max_seq_samples,
            max_static_samples=args.max_static_samples,
        )

    with norm_path.open("w") as f:
        json.dump(stats, f, indent=2)
    log(f"[NORM] Saved norm stats to {norm_path}")
    return stats


# ========================= Dataset =========================

@dataclass
class NormPerBase:
    seq_mean: np.ndarray
    seq_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray


class SeqSlugDataset(IterableDataset):
    """
    Iterable over NPZ shards. Each sample:
        x_seq:   (T, D_seq)
        x_static:(D_static,)
        base_id: ()
        y:       (n_targets,)
        w:       (n_targets,)
    """

    def __init__(self, shard_paths: List[Path], meta: Dict[str, Any], norm_stats: Dict[str, Any], split: str) -> None:
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.meta = meta
        self.targets: List[str] = meta["targets"]
        self.n_targets = len(self.targets)
        self.bases: List[str] = meta["bases"]
        self.base_to_id = {b: i for i, b in enumerate(self.bases)}
        self.split = split

        self.norm: Dict[str, NormPerBase] = {}
        for b in self.bases:
            st = norm_stats[b]
            self.norm[b] = NormPerBase(
                seq_mean=np.asarray(st["seq"]["mean"], dtype=np.float32),
                seq_std=np.asarray(st["seq"]["std"], dtype=np.float32),
                static_mean=np.asarray(st["static"]["mean"], dtype=np.float32),
                static_std=np.asarray(st["static"]["std"], dtype=np.float32),
            )

        assert self.shard_paths, f"No shard paths for split={split}"
        with np.load(self.shard_paths[0], allow_pickle=False) as z0:
            x_seq0 = z0["x_seq"]
            x_static0 = z0["x_static"]
            self.seq_len = x_seq0.shape[1]
            self.seq_dim = x_seq0.shape[2]
            self.static_dim = x_static0.shape[1]

        total_samples = 0
        for path in self.shard_paths:
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as z:
                total_samples += int(z["x_seq"].shape[0])
        self.total_samples = total_samples

        log(
            f"[DATASET {split}] num_shards={len(self.shard_paths)}, "
            f"total_samples={self.total_samples}, seq_dim={self.seq_dim}, static_dim={self.static_dim}"
        )

    def __len__(self) -> int:
        return self.total_samples

    def _iter_worker_shards(self) -> List[Path]:
        worker = get_worker_info()
        if worker is None:
            indices = list(range(len(self.shard_paths)))
        else:
            per_worker = int(math.ceil(len(self.shard_paths) / worker.num_workers))
            start = worker.id * per_worker
            end = min(start + per_worker, len(self.shard_paths))
            indices = list(range(start, end))
        if self.split == "train":
            random.shuffle(indices)
        return [self.shard_paths[i] for i in indices]

    def __iter__(self):
        for path in self._iter_worker_shards():
            fname = path.name
            base_name = fname.split("_", 1)[0]
            base_id = self.base_to_id[base_name]
            ns = self.norm[base_name]

            with np.load(path, allow_pickle=False) as z:
                x_seq = z["x_seq"].astype(np.float32)
                x_static = z["x_static"].astype(np.float32)
                base_idx = z["base_idx"].astype(np.int16)

                y_list, w_list = [], []
                for t in self.targets:
                    y_list.append(z[f"y_{t}"].astype(np.float32))
                    w_list.append(z[f"w_{t}"].astype(np.float32))
                y = np.stack(y_list, axis=-1)
                w = np.stack(w_list, axis=-1)

            if base_idx.size > 0 and not np.all(base_idx == base_id):
                base_id = int(base_idx[0])

            x_seq[~np.isfinite(x_seq)] = 0.0
            x_static[~np.isfinite(x_static)] = 0.0

            x_seq = (x_seq - ns.seq_mean) / ns.seq_std
            x_static = (x_static - ns.static_mean) / ns.static_std

            mask_y = np.isfinite(y)
            y[~mask_y] = 0.0
            w[~mask_y] = 0.0
            w[~np.isfinite(w)] = 0.0

            N = x_seq.shape[0]
            for i in range(N):
                yield (
                    torch.from_numpy(x_seq[i]),
                    torch.from_numpy(x_static[i]),
                    torch.tensor(base_id, dtype=torch.long),
                    torch.from_numpy(y[i]),
                    torch.from_numpy(w[i]),
                )


def collate_batch(batch):
    x_seq = torch.stack([b[0] for b in batch], dim=0)
    x_static = torch.stack([b[1] for b in batch], dim=0)
    base_id = torch.stack([b[2] for b in batch], dim=0)
    y = torch.stack([b[3] for b in batch], dim=0)
    w = torch.stack([b[4] for b in batch], dim=0)
    return x_seq, x_static, base_id, y, w


# ========================= Model =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SeqTransformerModel(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        static_dim: int,
        n_bases: int,
        n_targets: int,
        seq_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_bases = n_bases
        self.n_targets = n_targets
        self.n_quantiles = N_QUANTILES
        self.out_dim = n_targets * self.n_quantiles

        self.seq_proj = nn.Linear(seq_dim, d_model)
        self.static_proj = nn.Linear(static_dim, d_model)
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

        self.body = nn.Sequential(
            nn.Linear(2 * d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(d_model, self.out_dim) for _ in range(n_bases)])

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor, base_id: torch.Tensor) -> torch.Tensor:
        h_seq = self.seq_proj(x_seq)
        h_seq = self.pos_enc(h_seq)
        h_seq = self.encoder(h_seq)
        h_last = h_seq[:, -1, :]
        h_static = self.static_proj(x_static)
        h = self.body(torch.cat([h_last, h_static], dim=-1))

        B = h.size(0)
        device = h.device
        out = torch.zeros(B, self.out_dim, device=device, dtype=h.dtype)
        for b in range(self.n_bases):
            mask = (base_id == b)
            if mask.any():
                out[mask] = self.heads[b](h[mask])
        return out


# ========================= Loss & stats =========================

def huber_base(diff: torch.Tensor, delta: float = 3.0) -> torch.Tensor:
    abs_diff = diff.abs()
    d = torch.as_tensor(delta, device=diff.device, dtype=diff.dtype)
    quad = torch.minimum(abs_diff, d)
    lin = abs_diff - quad
    return 0.5 * quad * quad + d * lin


def quantile_huber_loss(
    y: torch.Tensor,                 # (B, T)
    q_scaled: torch.Tensor,          # (B, T, K)
    w_eff: torch.Tensor,             # (B, T)
    target_scales: torch.Tensor,     # (T,)
    delta: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T = y.shape
    _, Tq, K = q_scaled.shape
    assert T == Tq and K == N_QUANTILES

    device, dtype = y.device, y.dtype
    scales = target_scales.view(1, T, 1).to(device=device, dtype=dtype)
    y_scaled = y.view(B, T, 1) * scales
    w_bt = w_eff.view(B, T, 1).to(device=device, dtype=dtype)

    max_z = float(MAX_TRANSFORMED_ABS)
    y_scaled = torch.clamp(y_scaled, -max_z, max_z)
    q_scaled = torch.clamp(q_scaled, -max_z, max_z)

    diff = y_scaled - q_scaled
    base = huber_base(diff, delta=delta)

    taus = torch.as_tensor(QUANTILES, device=device, dtype=dtype).view(1, 1, K)
    q_w_list = [QUANTILE_WEIGHTS[q] for q in QUANTILES]
    q_weights = torch.as_tensor(q_w_list, device=device, dtype=dtype).view(1, 1, K)

    mask_pos = (diff >= 0)
    weight_side = torch.where(mask_pos, taus, 1.0 - taus) * q_weights

    loss_all = weight_side * base
    finite_mask = (
        torch.isfinite(loss_all) & torch.isfinite(w_bt) &
        torch.isfinite(y_scaled) & torch.isfinite(q_scaled)
    )
    loss_all = torch.where(finite_mask, loss_all, torch.zeros_like(loss_all))
    w_mask = torch.where(finite_mask, w_bt, torch.zeros_like(w_bt))

    num = (loss_all * w_mask).sum()
    denom = w_mask.sum().clamp_min(1e-12)
    loss_scalar = num / denom

    loss_bt = (loss_all * w_mask).sum(dim=(0, 2))
    w_bt_sum = w_mask.sum(dim=(0, 2))
    loss_per_target = loss_bt / w_bt_sum.clamp_min(1e-12)

    w_sum_example = w_mask.sum(dim=2).clamp_min(1e-12)
    loss_per_example = (loss_all * w_mask).sum(dim=2) / w_sum_example

    has_finite = finite_mask.any(dim=2, keepdim=True).to(w_bt.dtype)
    w_for_stats = (w_bt * has_finite).squeeze(2)
    return loss_scalar, loss_per_target, w_bt_sum, loss_per_example, w_for_stats


def normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def init_epoch_stats(n_bases: int, n_targets: int, device: torch.device) -> Dict[str, torch.Tensor]:
    shape = (n_bases, n_targets)
    z = lambda v=float("inf"): torch.full(shape, v, device=device)
    nz = lambda v=float("-inf"): torch.full(shape, v, device=device)
    zeros = lambda: torch.zeros(shape, device=device)
    return {
        "w_sum": zeros(),
        "loss_sum": zeros(),
        "y_sum": zeros(),
        "y2_sum": zeros(),
        "mu_sum": zeros(),
        "mu2_sum": zeros(),
        "y_mu_sum": zeros(),
        "sign_correct_w": zeros(),
        "large_event_w": zeros(),
        "large_pred_w": zeros(),
        "large_tp_w": zeros(),
        "large_dir_correct_w": zeros(),
        "y_min": z(),
        "y_max": nz(),
        "mu_min": z(),
        "mu_max": nz(),
    }


def update_epoch_stats(
    stats: Dict[str, torch.Tensor],
    y: torch.Tensor, mu: torch.Tensor, w_eff: torch.Tensor,
    loss_per_example: torch.Tensor, base_id: torch.Tensor,
    is_large_true: torch.Tensor, is_large_pred: torch.Tensor,
) -> None:
    n_bases, n_targets = stats["w_sum"].shape
    B, T = y.shape
    target_ids = torch.arange(n_targets, device=y.device).view(1, -1)
    idx_flat = (base_id.view(-1, 1) * n_targets + target_ids).reshape(-1)
    w_flat = w_eff.reshape(-1)

    def add(name: str, val: torch.Tensor):
        stats[name].view(-1).index_add_(0, idx_flat, val.reshape(-1))

    add("w_sum", w_flat)
    add("loss_sum", (w_eff * loss_per_example).reshape(-1))
    add("y_sum", (w_eff * y).reshape(-1))
    add("y2_sum", (w_eff * y * y).reshape(-1))
    add("mu_sum", (w_eff * mu).reshape(-1))
    add("mu2_sum", (w_eff * mu * mu).reshape(-1))
    add("y_mu_sum", (w_eff * y * mu).reshape(-1))

    sign_correct = (torch.sign(y) == torch.sign(mu)).to(y.dtype)
    add("sign_correct_w", (w_eff * sign_correct).reshape(-1))

    large_event = is_large_true.to(y.dtype)
    large_pred = is_large_pred.to(y.dtype)
    tp = large_event * large_pred
    dir_correct = tp * (torch.sign(y) == torch.sign(mu)).to(y.dtype)

    add("large_event_w", (w_eff * large_event).reshape(-1))
    add("large_pred_w", (w_eff * large_pred).reshape(-1))
    add("large_tp_w", (w_eff * tp).reshape(-1))
    add("large_dir_correct_w", (w_eff * dir_correct).reshape(-1))

    for b in range(n_bases):
        mask_b = (base_id == b)
        if not torch.any(mask_b):
            continue
        y_b, mu_b = y[mask_b], mu[mask_b]
        y_min_b, _ = y_b.min(dim=0)
        y_max_b, _ = y_b.max(dim=0)
        mu_min_b, _ = mu_b.min(dim=0)
        mu_max_b, _ = mu_b.max(dim=0)

        stats["y_min"][b] = torch.minimum(stats["y_min"][b], y_min_b)
        stats["y_max"][b] = torch.maximum(stats["y_max"][b], y_max_b)
        stats["mu_min"][b] = torch.minimum(stats["mu_min"][b], mu_min_b)
        stats["mu_max"][b] = torch.maximum(stats["mu_max"][b], mu_max_b)


def _metrics_for_group(
    w_sum, loss_sum, y_sum, y2_sum, mu_sum, mu2_sum, y_mu_sum,
    sign_correct_w, large_event_w, large_pred_w, large_tp_w, large_dir_correct_w,
    y_min, y_max, mu_min, mu_max, target_names: List[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n_targets = len(target_names)

    def safe_div(num: float, denom: float) -> float:
        if denom <= 0.0 or not math.isfinite(num) or not math.isfinite(denom):
            return 0.0
        return num / denom

    def safe_val(x: float) -> float:
        return x if math.isfinite(x) else 0.0

    for j in range(n_targets):
        ws = float(w_sum[j])
        if ws <= 0.0:
            continue

        loss_sum_j = float(loss_sum[j])
        y_sum_j, y2_sum_j = float(y_sum[j]), float(y2_sum[j])
        mu_sum_j, mu2_sum_j = float(mu_sum[j]), float(mu2_sum[j])
        y_mu_sum_j = float(y_mu_sum[j])

        mean_loss = safe_div(loss_sum_j, ws)

        Ey, Ey2 = safe_div(y_sum_j, ws), safe_div(y2_sum_j, ws)
        Emu, Emu2 = safe_div(mu_sum_j, ws), safe_div(mu2_sum_j, ws)
        Eym = safe_div(y_mu_sum_j, ws)

        mse_raw = safe_val(Ey2 + Emu2 - 2.0 * Eym)
        mse = max(mse_raw, 0.0)
        rmse = math.sqrt(mse)

        var_y_raw = safe_val(Ey2 - Ey * Ey)
        var_y = max(var_y_raw, 0.0)
        var_mu_raw = safe_val(Emu2 - Emu * Emu)
        var_mu = max(var_mu_raw, 0.0)

        r2 = 1.0 - mse / var_y if var_y > 1e-12 else 0.0
        if var_y <= 1e-12 or var_mu <= 1e-12:
            corr = 0.0
        else:
            cov_raw = safe_val(Eym - Ey * Emu)
            denom = math.sqrt(max(var_y * var_mu, 0.0))
            corr = cov_raw / denom if denom > 0 else 0.0

        sign_acc = safe_div(float(sign_correct_w[j]), ws)
        le_w = float(large_event_w[j])
        lp_w = float(large_pred_w[j])
        tp_w = float(large_tp_w[j])
        ld_w = float(large_dir_correct_w[j])

        large_event_rate = safe_div(le_w, ws)
        large_pred_rate = safe_div(lp_w, ws)
        precision = safe_div(tp_w, lp_w)
        recall = safe_div(tp_w, le_w)
        dir_acc_large = safe_div(ld_w, tp_w)

        tname = target_names[j]
        out[tname] = {
            "weight_sum": ws,
            "mean_loss": mean_loss,
            "rmse": rmse,
            "r2": r2,
            "corr": corr,
            "sign_acc": sign_acc,
            "large_event_rate": large_event_rate,
            "large_pred_rate": large_pred_rate,
            "large_precision": precision,
            "large_recall": recall,
            "large_dir_acc": dir_acc_large,
            "y_mean": Ey,
            "y_std": math.sqrt(var_y),
            "p_mean": Emu,
            "p_std": math.sqrt(var_mu),
            "y_min": safe_val(float(y_min[j])),
            "y_max": safe_val(float(y_max[j])),
            "p_min": safe_val(float(mu_min[j])),
            "p_max": safe_val(float(mu_max[j])),
        }
    return out


def finalize_epoch_stats(stats: Dict[str, torch.Tensor], base_names: List[str], target_names: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"total": {}, "per_base": {}}
    s = {k: v.detach().cpu().numpy() for k, v in stats.items()}

    w_sum = s["w_sum"]; loss_sum = s["loss_sum"]
    y_sum = s["y_sum"]; y2_sum = s["y2_sum"]
    mu_sum = s["mu_sum"]; mu2_sum = s["mu2_sum"]; y_mu_sum = s["y_mu_sum"]
    sign_correct_w = s["sign_correct_w"]
    large_event_w = s["large_event_w"]; large_pred_w = s["large_pred_w"]
    large_tp_w = s["large_tp_w"]; large_dir_correct_w = s["large_dir_correct_w"]
    y_min = s["y_min"]; y_max = s["y_max"]
    mu_min = s["mu_min"]; mu_max = s["mu_max"]

    for b, base_name in enumerate(base_names):
        out["per_base"][base_name] = _metrics_for_group(
            w_sum[b], loss_sum[b], y_sum[b], y2_sum[b],
            mu_sum[b], mu2_sum[b], y_mu_sum[b],
            sign_correct_w[b], large_event_w[b], large_pred_w[b],
            large_tp_w[b], large_dir_correct_w[b],
            y_min[b], y_max[b], mu_min[b], mu_max[b],
            target_names,
        )

    w_sum_tot = w_sum.sum(axis=0)
    loss_sum_tot = loss_sum.sum(axis=0)
    y_sum_tot = y_sum.sum(axis=0)
    y2_sum_tot = y2_sum.sum(axis=0)
    mu_sum_tot = mu_sum.sum(axis=0)
    mu2_sum_tot = mu2_sum.sum(axis=0)
    y_mu_sum_tot = y_mu_sum.sum(axis=0)
    sign_correct_w_tot = sign_correct_w.sum(axis=0)
    large_event_w_tot = large_event_w.sum(axis=0)
    large_pred_w_tot = large_pred_w.sum(axis=0)
    large_tp_w_tot = large_tp_w.sum(axis=0)
    large_dir_correct_w_tot = large_dir_correct_w.sum(axis=0)
    y_min_tot = y_min.min(axis=0)
    y_max_tot = y_max.max(axis=0)
    mu_min_tot = mu_min.min(axis=0)
    mu_max_tot = mu_max.max(axis=0)

    out["total"] = _metrics_for_group(
        w_sum_tot, loss_sum_tot, y_sum_tot, y2_sum_tot,
        mu_sum_tot, mu2_sum_tot, y_mu_sum_tot,
        sign_correct_w_tot, large_event_w_tot, large_pred_w_tot,
        large_tp_w_tot, large_dir_correct_w_tot,
        y_min_tot, y_max_tot, mu_min_tot, mu_max_tot,
        target_names,
    )
    return out


# ========================= Train / eval =========================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    use_amp: bool,
    epoch: int,
    log_interval: int,
    target_names: List[str],
    base_names: List[str],
    large_move_thresholds: torch.Tensor,
    target_scales: torch.Tensor,
    base_weights: torch.Tensor,
    max_steps: Optional[int] = None,
    train: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    model.train(train)
    try:
        approx_batches = math.ceil(len(loader.dataset) / loader.batch_size)  # type: ignore[arg-type]
    except Exception:
        approx_batches = None

    phase = "TRAIN" if train else "EVAL"
    log(f"[RUN_EPOCH] {phase} epoch {epoch} starting, "
        f"approx_batches={approx_batches if approx_batches is not None else '?'}")

    n_bases = len(base_names)
    n_targets = len(target_names)
    stats_tensors = init_epoch_stats(n_bases, n_targets, device)

    total_loss = 0.0
    total_batches = 0
    step = 0

    loss_sum_global = torch.zeros(n_targets, device=device)
    w_sum_global = torch.zeros(n_targets, device=device)

    cov_in_sum = torch.zeros(n_targets, N_QUANTILES, device=device)
    cov_w_sum = torch.zeros_like(cov_in_sum)

    thr = large_move_thresholds.to(device=device, dtype=torch.float32).view(1, -1)
    sigma_fixed = (large_move_thresholds / LARGE_MOVE_SIGMA_MULT).to(device=device, dtype=torch.float32)
    sigma_fixed = sigma_fixed.clamp_min(1e-6).view(1, -1)

    for batch_idx, (x_seq, x_static, base_id, y, w) in enumerate(loader):
        x_seq = x_seq.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        base_id = base_id.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        B, T = y.shape
        bw = base_weights[base_id]
        w_eff = w * bw.view(-1, 1)

        if train:
            optimizer.zero_grad(set_to_none=True)  # type: ignore[arg-type]
            with autocast(device_type=device.type, enabled=use_amp):
                pred_raw = model(x_seq, x_static, base_id)
                q_scaled = pred_raw.view(B, T, N_QUANTILES)
                loss_scalar, loss_per_target, w_per_target, loss_example, w_stats = quantile_huber_loss(
                    y=y, q_scaled=q_scaled, w_eff=w_eff, target_scales=target_scales
                )
            if not torch.isfinite(loss_scalar):
                log(f"[ERROR] Non-finite loss at epoch={epoch}, batch={batch_idx}: {loss_scalar.item()}")
                continue

            scaler.scale(loss_scalar).backward()  # type: ignore[union-attr]
            if scaler is not None:
                scaler.unscale_(optimizer)  # type: ignore[arg-type]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)  # type: ignore[union-attr]
            scaler.update()         # type: ignore[union-attr]
        else:
            with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
                pred_raw = model(x_seq, x_static, base_id)
                q_scaled = pred_raw.view(B, T, N_QUANTILES)
                loss_scalar, loss_per_target, w_per_target, loss_example, w_stats = quantile_huber_loss(
                    y=y, q_scaled=q_scaled, w_eff=w_eff, target_scales=target_scales
                )
            if not torch.isfinite(loss_scalar):
                log(f"[ERROR] Non-finite eval loss at epoch={epoch}, batch={batch_idx}: {loss_scalar.item()}")
                continue

        with torch.no_grad():
            loss_sum_global += loss_per_target * w_per_target
            w_sum_global += w_per_target

            q_med_scaled = q_scaled[:, :, MEDIAN_INDEX]
            mu_raw = q_med_scaled / target_scales.view(1, -1)
            
            # NEW: sanitize non-finite preds so stats don't get polluted
            non_finite_mu = ~torch.isfinite(mu_raw)
            if non_finite_mu.any():
                n_bad = int(non_finite_mu.sum().item())
                log(f"[WARN] {n_bad} non-finite median predictions this batch; zeroing them for stats.")
                mu_raw = torch.nan_to_num(mu_raw, nan=0.0, posinf=0.0, neginf=0.0)

            z_hi = (thr - mu_raw) / sigma_fixed
            z_lo = (-thr - mu_raw) / sigma_fixed
            cdf_hi = normal_cdf(z_hi)
            cdf_lo = normal_cdf(z_lo)
            p_in = (cdf_hi - cdf_lo).clamp(0.0, 1.0)
            p_large = (1.0 - p_in).clamp(0.0, 1.0)

            is_large_true = (y.abs() >= thr).detach()
            is_large_pred = (p_large >= LARGE_MOVE_PROB_THRESHOLD).detach()

            update_epoch_stats(
                stats_tensors, y=y, mu=mu_raw, w_eff=w_stats,
                loss_per_example=loss_example, base_id=base_id,
                is_large_true=is_large_true, is_large_pred=is_large_pred,
            )

            scales_bt = target_scales.view(1, T, 1)
            y_scaled_cov = y.view(B, T, 1) * scales_bt
            w_cov_bt = w_eff.view(B, T, 1)
            finite_cov = (
                torch.isfinite(y_scaled_cov) &
                torch.isfinite(q_scaled) &
                torch.isfinite(w_cov_bt)
            )
            w_cov = torch.where(finite_cov, w_cov_bt, torch.zeros_like(w_cov_bt))
            below = (y_scaled_cov <= q_scaled) & finite_cov
            cov_in_sum += (w_cov * below.to(w_cov.dtype)).sum(dim=0)
            cov_w_sum += w_cov.sum(dim=0)

        total_loss += float(loss_scalar.detach().cpu().item())
        total_batches += 1
        step += 1

        if train and (batch_idx % log_interval == 0):
            log(
                f"[TRAIN] epoch={epoch} step={batch_idx}/"
                f"{approx_batches if approx_batches is not None else '?'} "
                f"loss={total_loss / max(total_batches, 1):.6g}"
            )

        if max_steps is not None and step >= max_steps:
            break

    avg_loss = total_loss / max(total_batches, 1)
    stats_dict = finalize_epoch_stats(stats_tensors, base_names=base_names, target_names=target_names)

    with torch.no_grad():
        mean_loss_vec = (loss_sum_global / w_sum_global.clamp_min(1e-8)).detach().cpu().tolist()
    stats_dict["mean_loss_per_target"] = {t: float(mean_loss_vec[i]) for i, t in enumerate(target_names)}

    with torch.no_grad():
        cov_ratio = (cov_in_sum / cov_w_sum.clamp_min(1e-12)).detach().cpu().numpy()
    qc: Dict[str, Dict[str, float]] = {}
    for ti, t in enumerate(target_names):
        qc_t = {f"{q:.2f}": float(cov_ratio[ti, qi]) for qi, q in enumerate(QUANTILES)}
        qc[t] = qc_t
    stats_dict["quantile_coverage"] = qc
    return avg_loss, stats_dict


# ========================= Main =========================

def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    meta_path = Path(args.meta_path) if args.meta_path else data_dir / "meta_transformer.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta file not found: {meta_path}")

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

    meta = load_meta(meta_path)
    norm_stats = compute_or_load_norm_stats(args, meta, data_dir)

    train_dir = Path(meta.get("shards_train_dir", str(data_dir / "shards" / "train")))
    valid_dir = Path(meta.get("shards_valid_dir", str(data_dir / "shards" / "valid")))
    train_shards = [train_dir / name for name in meta["train_shards"]]
    valid_shards = [valid_dir / name for name in meta["valid_shards"]]

    train_dataset = SeqSlugDataset(train_shards, meta, norm_stats, split="train")
    valid_dataset = SeqSlugDataset(valid_shards, meta, norm_stats, split="valid")

    cpu_cores = os.cpu_count() or 1
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
    static_dim = train_dataset.static_dim
    base_names = meta["bases"]
    target_names = meta["targets"]      # now expected: ["ret_2s","ret_5s","ret_10s"]
    n_bases = len(base_names)
    n_targets = len(target_names)
    seq_len = train_dataset.seq_len

    model = SeqTransformerModel(
        seq_dim=seq_dim,
        static_dim=static_dim,
        n_bases=n_bases,
        n_targets=n_targets,
        seq_len=seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    log(
        f"Model: seq_dim={seq_dim}, static_dim={static_dim}, "
        f"n_bases={n_bases}, n_targets={n_targets}, n_quantiles={N_QUANTILES}"
    )
    log(str(model))

    # ----- large-move thresholds per target (raw space) -----
    # meta["target_sigma"] from build_seq.py: {target -> {base -> sigma}}
    target_sigma_meta = meta.get("target_sigma", {})
    large_thr_list: List[float] = []
    for t in target_names:
        val = target_sigma_meta.get(t, 1.0)
        if isinstance(val, dict):
            vals = [float(v) for v in val.values() if np.isfinite(v)]
            sig = float(np.mean(vals)) if vals else 1.0
        else:
            sig = float(val)
        large_thr_list.append(LARGE_MOVE_SIGMA_MULT * sig)
    large_move_thresholds = torch.tensor(large_thr_list, dtype=torch.float32, device=device)
    log(
        "Large-move thresholds (raw): "
        f"{ {t: thr for t, thr in zip(target_names, large_thr_list)} }, "
        f"k={LARGE_MOVE_SIGMA_MULT}, prob_thr={LARGE_MOVE_PROB_THRESHOLD}"
    )

    # Target scales (per target)
    target_scale_list = [TARGET_SCALES.get(t, 1.0) for t in target_names]
    target_scales = torch.tensor(target_scale_list, dtype=torch.float32, device=device)
    log(f"Target scales: { {t: s for t, s in zip(target_names, target_scale_list)} }")

    # Base weights
    base_weight_list = [BASE_WEIGHTS.get(b, 1.0) for b in base_names]
    base_weights = torch.tensor(base_weight_list, dtype=torch.float32, device=device)
    log(f"Base weights: { {b: w for b, w in zip(base_names, base_weight_list)} }")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device="cuda" if device.type == "cuda" else "cpu", enabled=args.use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        log(f"========== Epoch {epoch}/{args.epochs} (lr={current_lr:.6g}) ==========")

        train_loss, train_stats = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=args.use_amp,
            epoch=epoch,
            log_interval=args.log_interval,
            target_names=target_names,
            base_names=base_names,
            large_move_thresholds=large_move_thresholds,
            target_scales=target_scales,
            base_weights=base_weights,
            max_steps=args.max_steps_per_epoch,
            train=True,
        )
        log(f"[TRAIN] epoch={epoch} avg_loss={train_loss:.6g}")

        for t in target_names:
            m = train_stats["total"].get(t, {})
            if m:
                log(
                    f"[TRAIN] epoch={epoch} target={t} "
                    f"mean_loss={m['mean_loss']:.6g} rmse={m['rmse']:.3e} "
                    f"r2={m['r2']:.4f} sign_acc={m['sign_acc']:.4f} "
                    f"large_prec={m['large_precision']:.4f} "
                    f"large_rec={m['large_recall']:.4f}"
                )

        train_stats_clean = sanitize_for_json(train_stats)
        with (run_dir / f"train_stats_epoch{epoch:02d}.json").open("w") as f:
            json.dump(train_stats_clean, f, indent=2, allow_nan=False)

        val_loss, val_stats = run_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=args.use_amp,
            epoch=epoch,
            log_interval=args.log_interval,
            target_names=target_names,
            base_names=base_names,
            large_move_thresholds=large_move_thresholds,
            target_scales=target_scales,
            base_weights=base_weights,
            max_steps=None,
            train=False,
        )
        log(f"[VALID] epoch={epoch} avg_loss={val_loss:.6g}")

        for t in target_names:
            m = val_stats["total"].get(t, {})
            if m:
                log(
                    f"[VALID] epoch={epoch} target={t} "
                    f"mean_loss={m['mean_loss']:.6g} rmse={m['rmse']:.3e} "
                    f"r2={m['r2']:.4f} sign_acc={m['sign_acc']:.4f} "
                    f"large_prec={m['large_precision']:.4f} "
                    f"large_rec={m['large_recall']:.4f}"
                )

        val_stats_clean = sanitize_for_json(val_stats)
        with (run_dir / f"valid_stats_epoch{epoch:02d}.json").open("w") as f:
            json.dump(val_stats_clean, f, indent=2, allow_nan=False)

        ckpt_path = run_dir / f"model_epoch{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "meta_path": str(meta_path),
                "norm_stats_path": str(
                    Path(args.norm_stats_path) if args.norm_stats_path else data_dir / "norm_stats.json"
                ),
                "targets": target_names,
                "bases": base_names,
                "quantiles": QUANTILES,
            },
            ckpt_path,
        )
        log(f"[CKPT] Saved checkpoint to {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = run_dir / "model_best.pt"
            torch.save(model.state_dict(), best_path)
            log(f"[CKPT] New best model (val_loss={val_loss:.6g}) saved to {best_path}")

        scheduler.step()

    log("Training finished.")


if __name__ == "__main__":
    main()
