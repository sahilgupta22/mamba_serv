"""
06_ablation_studies.py
──────────────────────
Surgical ablations that isolate each Mamba-Surv design decision.

Ablation variants
─────────────────
  A.  NoDeltaT          – Mamba-Surv with delta_t zeroed out at input.
                          Isolates the contribution of the "hours since last
                          observation" signal.
  B.  FixedA            – Mamba-Surv with A_t = exp(A_base) (gate = 1 always).
                          Isolates the input-dependent gating mechanism.
  C.  SingleLayer       – Mamba-Surv with n_layers = 1 (vs default 3).
                          Isolates the depth contribution.
  D.  LastStepPool      – Mamba-Surv replacing attention pooling with simple
                          last-valid-step extraction.
                          Isolates the temporal pooling strategy.

Each ablation is trained with IDENTICAL hyper-params as the full model
(same lr, epochs, etc.) to ensure a fair comparison.  Early stopping and
checkpointing are handled by the shared train loop in 04_train.py via
programmatic invocation.

Output
──────
  results/ablation_results.json        – C-index for full + each ablation
  results/figures/fig_ablation.png     – horizontal bar chart

Usage:
    python 06_ablation_studies.py --processed_dir ./processed \
                                  --results_dir   ./results \
                                  [--epochs 60] [--batch_size 32] [--seed 42]

Estimated GPU time: ~6-12 h for all 4 ablations on A100 (each ≈ same as
the full Mamba-Surv run).
"""

import argparse
import json
import logging
import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module as _imp
_models = _imp("03_model_definitions")
MambaSurv         = _models.MambaSurv
MambaBlock        = _models.MambaBlock
ICUSurvivalDataset = _models.ICUSurvivalDataset
cox_partial_loss   = _models.cox_partial_loss
concordance_index  = _models.concordance_index

# Re-use the scheduler from 04_train
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Ablation model wrappers
# ═════════════════════════════════════════════════════════════════════════════

class MambaSurv_NoDeltaT(MambaSurv):
    """Ablation A: zero out delta_t before it reaches MambaBlocks."""

    def forward(self, X, delta_t, mask):
        # Replace delta_t with zeros → gate sees no temporal staleness signal
        delta_t_zeroed = torch.zeros_like(delta_t)
        return super().forward(X, delta_t_zeroed, mask)


class MambaBlock_FixedA(MambaBlock):
    """Ablation B: A_t = exp(A_base) always (gate clamped to 1)."""

    def forward(self, x, delta_t, mask):
        B, T, D = x.shape
        N = self.d_state

        # Fixed A — no input-dependent gate
        A_base_exp = torch.exp(self.A_base).unsqueeze(0).unsqueeze(0)  # (1,1,D,N)
        A_t = A_base_exp.expand(B, T, D, N)

        B_t = self.B_proj(x).view(B, T, D, N)
        C_t = self.C_proj(x).view(B, T, D, N)
        x_exp = x.unsqueeze(-1)

        h = torch.zeros(B, D, N, device=x.device)
        y_list = []
        for t in range(T):
            m = mask[:, t].view(B, 1, 1)
            h = m * (A_t[:, t] * h + B_t[:, t] * x_exp[:, t]) + (1 - m) * h
            y_t = (C_t[:, t] * h).sum(dim=-1)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)
        y = self.out_proj(self.norm(y + x))
        return y


class MambaSurv_FixedA(MambaSurv):
    """Ablation B wrapper: replace all MambaBlocks with FixedA variants."""

    def __init__(self, input_dim, delta_t_dim, d_model=128, d_state=16, n_layers=3, dropout=0.1):
        super().__init__(input_dim, delta_t_dim, d_model, d_state, n_layers, dropout)
        # Overwrite mamba_layers with FixedA blocks
        self.mamba_layers = nn.ModuleList([
            MambaBlock_FixedA(d_model, d_state, delta_t_dim) for _ in range(n_layers)
        ])


class MambaSurv_LastStepPool(MambaSurv):
    """Ablation D: replace attention pooling with last-valid-step extraction."""

    def forward(self, X, delta_t, mask):
        import torch.nn.functional as F

        h = self.input_proj(X)
        for mamba_blk, drop in zip(self.mamba_layers, self.dropouts):
            h = mamba_blk(h, delta_t, mask)
            h = drop(h)

        # Last valid step per sample
        # lengths = number of valid time-steps
        lengths = mask.sum(dim=1).long().clamp(min=1) - 1   # 0-indexed
        batch_idx = torch.arange(h.size(0), device=h.device)
        pooled = h[batch_idx, lengths]                       # (B, d_model)

        risk = self.cox_head(pooled).squeeze(-1)
        return risk


# ═════════════════════════════════════════════════════════════════════════════
# Ablation registry
# ═════════════════════════════════════════════════════════════════════════════
ABLATION_REGISTRY = {
    "full":           MambaSurv,                # reference (re-trained for direct comparison)
    "no_delta_t":     MambaSurv_NoDeltaT,
    "fixed_A":        MambaSurv_FixedA,
    "single_layer":   None,                     # handled specially (n_layers=1)
    "last_step_pool": MambaSurv_LastStepPool,
}

ABLATION_LABELS = {
    "full":           "Full Mamba-Surv",
    "no_delta_t":     "A: No Δt Input",
    "fixed_A":        "B: Fixed A (no gate)",
    "single_layer":   "C: 1 Layer (vs 3)",
    "last_step_pool": "D: Last-Step Pool",
}


# ═════════════════════════════════════════════════════════════════════════════
# Shared training loop (self-contained, mirrors 04_train logic)
# ═════════════════════════════════════════════════════════════════════════════
import math

class _CosineWarmup:
    def __init__(self, optimizer, warmup, total, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup = warmup
        self.total = total
        self.lr_min = lr_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup:
            scale = self.step_count / max(self.warmup, 1)
        else:
            progress = (self.step_count - self.warmup) / max(self.total - self.warmup, 1)
            scale = self.lr_min + 0.5 * (1 - self.lr_min) * (1 + math.cos(math.pi * progress))
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = blr * scale


@torch.no_grad()
def _eval_loader(model, loader, device, use_amp):
    model.eval()
    risks, times, events = [], [], []
    for batch in loader:
        X    = batch["X"].to(device)
        dt   = batch["delta_t"].to(device)
        mask = batch["mask"].to(device)
        with autocast(enabled=use_amp):
            r = model(X, dt, mask)
        risks.append(r.cpu())
        times.append(batch["time_to_event"])
        events.append(batch["event"])
    return (torch.cat(risks), torch.cat(times), torch.cat(events))


def _train_ablation(model: nn.Module, train_loader, val_loader, device,
                    epochs: int = 60, lr: float = 3e-4, wd: float = 1e-4,
                    patience: int = 10, grad_clip: float = 1.0,
                    warmup_epochs: int = 3) -> float:
    """Train one ablation variant; return best val C-index."""
    use_amp = device.type == "cuda"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = len(train_loader)
    scheduler = _CosineWarmup(optimizer, warmup_epochs * steps_per_epoch,
                              epochs * steps_per_epoch)
    scaler = GradScaler(enabled=use_amp)

    best_c     = 0.0
    patience_c = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            X    = batch["X"].to(device)
            dt   = batch["delta_t"].to(device)
            mask = batch["mask"].to(device)
            t_ev = batch["time_to_event"].to(device)
            ev   = batch["event"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                risk = model(X, dt, mask)
                loss = cox_partial_loss(risk, t_ev, ev)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Validate
        risks, times, events = _eval_loader(model, val_loader, device, use_amp)
        c_idx = concordance_index(risks, times, events)

        logger.info("    Epoch %2d | val C=%.4f", epoch, c_idx)

        if c_idx > best_c:
            best_c = c_idx
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= patience:
                logger.info("    Early stop at epoch %d", epoch)
                break

    return best_c


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="./processed")
    parser.add_argument("--results_dir",   type=str, default="./results")
    parser.add_argument("--epochs",        type=int,   default=60)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--max_len",       type=int,   default=2160)
    parser.add_argument("--d_model",       type=int,   default=128)
    parser.add_argument("--d_state",       type=int,   default=16)
    parser.add_argument("--num_workers",   type=int,   default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Manifest ─────────────────────────────────────────────────────────────
    with open(os.path.join(args.processed_dir, "feature_manifest.json")) as fh:
        manifest = json.load(fh)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = ICUSurvivalDataset(
        os.path.join(args.processed_dir, "train.parquet"), manifest, max_len=args.max_len
    )
    val_ds = ICUSurvivalDataset(
        os.path.join(args.processed_dir, "val.parquet"),   manifest, max_len=args.max_len
    )
    
    # CRITICAL: Use actual dimensions from the Dataset, not the manifest
    # The Dataset filters to only columns that exist in the parquet file
    input_dim   = train_ds.input_dim
    delta_t_dim = train_ds.delta_t_dim
    logger.info("Input dim: %d | Delta-T dim: %d", input_dim, delta_t_dim)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Run each ablation ────────────────────────────────────────────────────
    results = {}

    for abl_name in ["full", "no_delta_t", "fixed_A", "single_layer", "last_step_pool"]:
        logger.info("\n══ Ablation: %s ══", ABLATION_LABELS[abl_name])
        t0 = time.time()

        n_layers = 1 if abl_name == "single_layer" else 3

        if abl_name == "single_layer":
            model = MambaSurv(input_dim, delta_t_dim,
                              d_model=args.d_model, d_state=args.d_state,
                              n_layers=1, dropout=0.1)
        else:
            cls = ABLATION_REGISTRY[abl_name]
            model = cls(input_dim, delta_t_dim,
                        d_model=args.d_model, d_state=args.d_state,
                        n_layers=3, dropout=0.1)

        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("  Parameters: %s", f"{n_params:,}")

        best_c = _train_ablation(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
        )

        elapsed = time.time() - t0
        results[abl_name] = {
            "label":       ABLATION_LABELS[abl_name],
            "val_cindex":  round(best_c, 4),
            "n_params":    n_params,
            "time_min":    round(elapsed / 60, 1),
        }
        logger.info("  → Best val C = %.4f  (%.1f min)", best_c, elapsed / 60)

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = os.path.join(args.results_dir, "ablation_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("\nAblation results written to %s", out_path)

    # ── Figure: horizontal bar chart ─────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})

    names  = [results[k]["label"]      for k in results]
    cvals  = [results[k]["val_cindex"] for k in results]
    colours = ["#2C5F8A" if k == "full" else "#A8C5DA" for k in results]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, cvals, color=colours, edgecolor="white", height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Validation C-Index", fontsize=12)
    ax.set_title("Ablation Study: Mamba-Surv Design Decisions", fontsize=14, fontweight="bold")
    ax.set_xlim(0.45, max(cvals) + 0.04)

    # Annotate
    for i, v in enumerate(cvals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=10, fontweight="bold")

    # Dashed line at full model performance
    full_c = results["full"]["val_cindex"]
    ax.axvline(full_c, color="#2C5F8A", linestyle="--", linewidth=1.2, alpha=0.7)

    plt.tight_layout()
    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "fig_ablation.png"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Ablation figure saved to %s", fig_dir)


if __name__ == "__main__":
    main()