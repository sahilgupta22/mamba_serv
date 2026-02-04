"""
04_train.py
───────────
Training loop for all models.  Handles:
    - Mixed-precision (AMP) for GPU speed
    - Gradient clipping (Cox losses can spike)
    - Early stopping on val C-index (patience = 10 epochs)
    - Cosine-annealing LR schedule with warm-up
    - Per-epoch JSON logging  →  results/logs/<model_name>.jsonl
    - Best-model checkpoints →  results/checkpoints/<model_name>_best.pt

All four models share identical training logic; only the model constructor
and (for Mamba-Surv) the delta_t input differ.

Usage:
    # Train all four models sequentially:
    python 04_train.py --model mamba_surv      --processed_dir ./processed --results_dir ./results
    python 04_train.py --model deepsurv_lstm   --processed_dir ./processed --results_dir ./results
    python 04_train.py --model deepsurv_mlp    --processed_dir ./processed --results_dir ./results
    python 04_train.py --model cox_linear      --processed_dir ./processed --results_dir ./results

    # Or train a single model with custom hyper-params:
    python 04_train.py --model mamba_surv --epochs 80 --lr 3e-4 --batch_size 64 ...
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module as _imp
_models = _imp("03_model_definitions")
ICUSurvivalDataset = _models.ICUSurvivalDataset
build_model        = _models.build_model
cox_partial_loss   = _models.cox_partial_loss
concordance_index  = _models.concordance_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Warm-up + Cosine Annealing LR Scheduler
class CosineWarmupScheduler:
    """Linear warm-up then cosine decay."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 lr_min: float = 1e-6):
        self.optimizer     = optimizer
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self.lr_min        = lr_min
        self.base_lrs      = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count    = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warm-up
            scale = self.step_count / max(self.warmup_steps, 1)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = self.lr_min + 0.5 * (1.0 - self.lr_min) * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# Evaluation helper
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             use_amp: bool = True) -> dict[str, float]:
    """Run full validation pass; return loss and C-index."""
    model.eval()
    all_risk, all_time, all_event, all_lengths = [], [], [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        X       = batch["X"].to(device)
        dt      = batch["delta_t"].to(device)
        mask    = batch["mask"].to(device)
        t_event = batch["time_to_event"].to(device)
        event   = batch["event"].to(device)

        with autocast(enabled=use_amp):
            risk = model(X, dt, mask)
            loss = cox_partial_loss(risk.float(), t_event.float(), event.float())

        if not torch.isnan(loss):
            total_loss += loss.item()
            n_batches  += 1
        
        # Store risk as float32 for C-index calculation
        all_risk.append(risk.detach().float())
        all_time.append(t_event.float())
        all_event.append(event.float())
        all_lengths.append(mask.sum(dim=1).detach().cpu().float())

    all_risk  = torch.cat(all_risk,  dim=0)
    all_time  = torch.cat(all_time,  dim=0)
    all_event = torch.cat(all_event, dim=0)
    all_lengths = torch.cat(all_lengths, dim=0)

    c_idx = concordance_index(all_risk, all_time, all_event)

    return {"val_loss": total_loss / max(n_batches, 1), "val_cindex": c_idx}


# Training loop
def train_one_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # load features
    manifest_path = os.path.join(args.processed_dir, "feature_manifest.json")
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    # dataset + loaders
    logger.info("Loading datasets …")
    train_ds = ICUSurvivalDataset(
        os.path.join(args.processed_dir, "train.parquet"), manifest, max_len=args.max_len
    )
    val_ds = ICUSurvivalDataset(
        os.path.join(args.processed_dir, "val.parquet"), manifest, max_len=args.max_len
    )

    input_dim = train_ds.input_dim
    delta_t_dim = train_ds.delta_t_dim
    logger.info("Input dim: %d | Delta-T dim: %d", input_dim, delta_t_dim)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # build model
    model_kwargs = dict(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    if args.model == "mamba_surv":
        model_kwargs["d_state"] = args.d_state

    model = build_model(args.model, input_dim, delta_t_dim, **model_kwargs)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s | Parameters: %s (%.2fM)", args.model, f"{n_params:,}", n_params / 1e6)

    # optimizer + sheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    steps_per_epoch  = len(train_loader)
    total_steps      = steps_per_epoch * args.epochs
    warmup_steps     = steps_per_epoch * args.warmup_epochs
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, lr_min=1e-6)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # output directories
    log_dir  = os.path.join(args.results_dir, "logs")
    ckpt_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_file = open(os.path.join(log_dir, f"{args.model}.jsonl"), "w")

    # training state
    best_cindex       = 0.0
    patience_counter  = 0
    t_start           = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_steps = 0.0, 0

        for batch in train_loader:
            X       = batch["X"].to(device)
            dt      = batch["delta_t"].to(device)
            mask    = batch["mask"].to(device)
            t_event = batch["time_to_event"].to(device)
            event   = batch["event"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                risk = model(X, dt, mask).float() # Force to FP32 for loss stability
                loss = cox_partial_loss(risk, t_event, event)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            epoch_steps += 1

        avg_train_loss = epoch_loss / max(epoch_steps, 1)

        # validation
        val_metrics = evaluate(model, val_loader, device, use_amp=(device.type == "cuda"))

        elapsed = time.time() - t_start
        lr_now  = scheduler.get_lr()[0]

        logger.info(
            "Epoch %3d/%d | train_loss %.4f | val_loss %.4f | val_C %.4f | "
            "lr %.2e | %s elapsed",
            epoch, args.epochs, avg_train_loss,
            val_metrics["val_loss"], val_metrics["val_cindex"],
            lr_now, f"{elapsed/60:.1f}m",
        )

        # log
        record = {
            "epoch":          epoch,
            "train_loss":     round(avg_train_loss,                 6),
            "val_loss":       round(val_metrics["val_loss"],        6),
            "val_cindex":     round(val_metrics["val_cindex"],      4),
            "lr":             lr_now,
            "elapsed_sec":    round(elapsed, 1),
        }
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

        # early stopping & checkpointing
        if val_metrics["val_cindex"] > best_cindex:
            best_cindex = val_metrics["val_cindex"]
            patience_counter = 0
            # Save best checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"{args.model}_best.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_cindex":  best_cindex,
                "args":        vars(args),
            }, ckpt_path)
            logger.info("  → New best C-index %.4f  saved to %s", best_cindex, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch, args.patience)
                break

    log_file.close()
    logger.info("Training complete.  Best val C-index: %.4f  |  Total time: %.1f min",
                best_cindex, (time.time() - t_start) / 60)


def main():
    parser = argparse.ArgumentParser(description="Train Mamba-Surv or baselines")

    parser.add_argument("--model", type=str, required=True,
                        choices=["mamba_surv", "deepsurv_lstm", "deepsurv_mlp", "transformer_cox", "cox_linear"],
                        help="Model architecture to train")

    parser.add_argument("--processed_dir", type=str, default="./processed",
                        help="Directory containing train/val/test parquets & manifest")
    parser.add_argument("--results_dir",   type=str, default="./results",
                        help="Root directory for logs and checkpoints")

    parser.add_argument("--epochs",         type=int,   default=60)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--wd",             type=float, default=1e-4,   help="Weight decay")
    parser.add_argument("--grad_clip",      type=float, default=1.0)
    parser.add_argument("--warmup_epochs",  type=int,   default=3)
    parser.add_argument("--patience",       type=int,   default=10,    help="Early-stopping patience")

    parser.add_argument("--d_model",   type=int, default=128)
    parser.add_argument("--d_state",   type=int, default=16,  help="Mamba state dimension N")
    parser.add_argument("--n_layers",  type=int, default=3)
    parser.add_argument("--dropout",   type=float, default=0.1)
    parser.add_argument("--max_len",   type=int, default=2160, help="Sequence length (90 days)")

    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    train_one_model(args)


if __name__ == "__main__":
    main()