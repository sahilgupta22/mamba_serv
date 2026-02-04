"""
05_evaluate_and_plot.py
───────────────────────
Loads every trained model's best checkpoint, evaluates on the held-out test
set, and produces figures.

Metrics table written to:
  results/test_metrics.json

Usage:
    python 05_evaluate_and_plot.py --processed_dir ./processed \
                                   --results_dir   ./results
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module as _imp
_models = _imp("03_model_definitions")
ICUSurvivalDataset = _models.ICUSurvivalDataset
build_model        = _models.build_model
cox_partial_loss   = _models.cox_partial_loss
concordance_index  = _models.concordance_index

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PALETTE = {
    "deepsurv_lstm":   "#81B29A",   # sage green
    "cox_linear":      "#C9A227",   # muted gold
    "deepsurv_mlp":    "#F01E2C",   # red
    # "transformer_cox": "#81B29A",   # sage green
    "mamba_surv":      "#2C5F8A",   # deep blue
}
PRETTY_NAMES = {
    "mamba_surv":      "Mamba-Surv",
    "deepsurv_lstm":   "DeepSurv (LSTM)",
    "deepsurv_mlp":    "DeepSurv (MLP)",
    # "transformer_cox": "Transformer-Cox",
    "cox_linear":      "Cox-Linear",
}
MODEL_NAMES = list(PALETTE.keys())


def _set_style():
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.size":       11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid":       True,
        "grid.alpha":      0.3,
        "figure.dpi":      150,
        "savefig.dpi":     300,
        "savefig.bbox":    "tight",
    })


# helper methods
def _load_model(model_name: str, manifest: dict, results_dir: str, device: torch.device) -> nn.Module:
    ckpt_path = os.path.join(results_dir, "checkpoints", f"{model_name}_best.pt")
    if not os.path.exists(ckpt_path):
        logger.warning("Checkpoint not found: %s — skipping %s", ckpt_path, model_name)
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})

    model = build_model(
        model_name,
        input_dim=106,
        delta_t_dim=len(manifest["delta_t_cols"]),
        d_model=saved_args.get("d_model", 128),
        d_state=saved_args.get("d_state", 16),
        n_layers=saved_args.get("n_layers", 3),
        dropout=0.0,                            # no dropout at eval
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    logger.info("  Loaded %s  (epoch %d, val C=%.4f)", model_name, ckpt["epoch"], ckpt["val_cindex"])
    return model


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device,
             use_amp: bool) -> dict[str, np.ndarray]:
    """Return risk scores, times, events, subject IDs as numpy arrays."""
    risks, times, events = [], [], []
    for batch in loader:
        X    = batch["X"].to(device)
        dt   = batch["delta_t"].to(device)
        mask = batch["mask"].to(device)
        with autocast(enabled=use_amp):
            risk = model(X, dt, mask)
        risks.append(risk.cpu().numpy())
        times.append(batch["time_to_event"].numpy())
        events.append(batch["event"].numpy())

    return {
        "risk":   np.concatenate(risks),
        "time":   np.concatenate(times),
        "event":  np.concatenate(events),
    }


# Brier Score (time-dependent, at t = median event time)
def _brier_score_at_t(risk_scores, times, events, t_eval):
    """
    Approximate Brier score at horizon t_eval.
    Patients who died before t_eval: true label = 1.
    Patients alive past t_eval:      true label = 0.
    Censored before t_eval:          excluded.
    """
    # Predicted probability approx. = 1 - exp(-exp(risk) * t_eval)
    # We use a simple sigmoid rescaling for calibration-agnostic comparison
    p_death = 1.0 / (1.0 + np.exp(-risk_scores))

    mask    = np.ones(len(times), dtype=bool)
    labels  = np.zeros(len(times))

    for i in range(len(times)):
        if events[i] == 1 and times[i] <= t_eval:
            labels[i] = 1.0
        elif events[i] == 1 and times[i] > t_eval:
            labels[i] = 0.0
        elif events[i] == 0 and times[i] > t_eval:
            labels[i] = 0.0
        else:
            mask[i] = False          # censored before t_eval — exclude

    bs = np.mean((p_death[mask] - labels[mask]) ** 2)
    return bs


# Kaplan-Meier (manual, no lifelines dependency)
def _kaplan_meier(times: np.ndarray, events: np.ndarray):
    """Return (sorted_times, survival_probs)."""
    order   = np.argsort(times)
    t_sort  = times[order]
    e_sort  = events[order]

    n_at_risk = len(times)
    surv      = 1.0
    t_out, s_out = [0.0], [1.0]

    prev_t = -1.0
    for i in range(len(t_sort)):
        if t_sort[i] != prev_t and prev_t >= 0:
            t_out.append(prev_t)
            s_out.append(surv)
        if e_sort[i] == 1:
            surv *= (n_at_risk - 1) / max(n_at_risk, 1)
        n_at_risk -= 1
        prev_t = t_sort[i]

    t_out.append(prev_t)
    s_out.append(surv)
    return np.array(t_out), np.array(s_out)


# Figure generation functions
def fig1_cindex_bar(all_preds: dict, fig_dir: str):
    """Grouped bar chart of C-index ± bootstrap SE."""
    _set_style()
    models  = [m for m in MODEL_NAMES if m in all_preds]
    cindexs = []
    ses     = []

    for m in models:
        p = all_preds[m]
        c = concordance_index(
            torch.tensor(p["risk"]), torch.tensor(p["time"]), torch.tensor(p["event"])
        )
        cindexs.append(c)

        # Bootstrap SE (100 resamples)
        n = len(p["risk"])
        boot_cs = []
        for _ in range(100):
            idx = np.random.choice(n, n, replace=True)
            boot_cs.append(concordance_index(
                torch.tensor(p["risk"][idx]),
                torch.tensor(p["time"][idx]),
                torch.tensor(p["event"][idx]),
            ))
        ses.append(np.std(boot_cs))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    bars = ax.bar(x, cindexs, yerr=ses, capsize=5, width=0.55,
                  color=[PALETTE[m] for m in models], edgecolor="white", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_NAMES[m] for m in models], fontsize=12)
    ax.set_ylabel("Harrell's C-Index", fontsize=12)
    ax.set_title("Test-Set Concordance Index", fontsize=14, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random")
    ax.legend(fontsize=9)

    # Annotate values
    for i, (c, se) in enumerate(zip(cindexs, ses)):
        ax.text(i, c + se + 0.015, f"{c:.3f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_cindex_comparison.png"))
    plt.close()
    logger.info("  Saved fig1_cindex_comparison.png")


def fig2_km_quartiles(all_preds: dict, fig_dir: str):
    """KM curves by predicted risk quartile (Mamba-Surv)."""
    _set_style()
    if "mamba_surv" not in all_preds:
        logger.warning("  Mamba-Surv preds not available — skipping fig2")
        return

    p = all_preds["mamba_surv"]
    quartiles = np.percentile(p["risk"], [25, 50, 75])
    labels    = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    colours   = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    cuts      = np.digitize(p["risk"], quartiles)   # 0,1,2,3

    fig, ax = plt.subplots(figsize=(8, 5))
    for q_idx in range(4):
        mask = cuts == q_idx
        if mask.sum() == 0:
            continue
        t_km, s_km = _kaplan_meier(p["time"][mask], p["event"][mask])
        # Convert hours to days for readability
        ax.step(t_km / 24, s_km, where="post",
                color=colours[q_idx], linewidth=2.0, label=labels[q_idx])

    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title("Kaplan-Meier by Predicted Risk Quartile (Mamba-Surv)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 14)
    ax.set_xlim(0, 90)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_km_risk_quartiles.png"))
    plt.close()
    logger.info("  Saved fig2_km_risk_quartiles.png")


def fig3_calibration(all_preds: dict, fig_dir: str):
    """Calibration plot: predicted decile risk vs observed event rate."""
    _set_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")

    for m in MODEL_NAMES:
        if m not in all_preds:
            continue
        p = all_preds[m]
        # Map risk scores to [0,1] via sigmoid
        p_pred = 1.0 / (1.0 + np.exp(-p["risk"]))
        # Bin into 10 deciles
        bins  = np.linspace(p_pred.min(), p_pred.max() + 1e-8, 11)
        mid   = (bins[:-1] + bins[1:]) / 2
        obs   = []
        pred_mean = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (p_pred >= lo) & (p_pred < hi)
            if sel.sum() > 0:
                obs.append(p["event"][sel].mean())
                pred_mean.append(p_pred[sel].mean())

        ax.plot(pred_mean, obs, "o-", color=PALETTE[m], linewidth=1.8,
                markersize=6, label=PRETTY_NAMES[m])

    ax.set_xlabel("Mean Predicted Risk", fontsize=12)
    ax.set_ylabel("Observed Event Rate", fontsize=12)
    ax.set_title("Calibration Plot", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_calibration.png"))
    plt.close()
    logger.info("  Saved fig3_calibration.png")


def fig4_training_curves(results_dir: str, fig_dir: str):
    """2×2 panel of train/val loss and val C-index from .jsonl logs."""
    _set_style()
    log_dir = os.path.join(results_dir, "logs")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax_idx, m in enumerate(MODEL_NAMES):
        log_path = os.path.join(log_dir, f"{m}.jsonl")
        if not os.path.exists(log_path):
            axes[ax_idx].set_title(f"{PRETTY_NAMES[m]} — no log", fontsize=11)
            continue

        records = [json.loads(line) for line in open(log_path)]
        epochs      = [r["epoch"]      for r in records]
        train_loss  = [r["train_loss"] for r in records]
        val_loss    = [r["val_loss"]   for r in records]
        val_c       = [r["val_cindex"] for r in records]

        ax = axes[ax_idx]
        ax2 = ax.twinx()

        ax.plot(epochs, train_loss, color=PALETTE[m], linewidth=1.8, label="Train Loss")
        ax.plot(epochs, val_loss,   color=PALETTE[m], linewidth=1.8, linestyle="--", label="Val Loss")
        ax2.plot(epochs, val_c,     color="gray",     linewidth=1.5, linestyle=":",  label="Val C-Index")

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Cox Loss", fontsize=10, color=PALETTE[m])
        ax2.set_ylabel("C-Index", fontsize=10, color="gray")
        ax.set_title(PRETTY_NAMES[m], fontsize=12, fontweight="bold")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

    plt.suptitle("Training Dynamics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_training_curves.png"))
    plt.close()
    logger.info("  Saved fig4_training_curves.png")

def fig5_temporal_attention_analysis(model: nn.Module, loader: DataLoader,
                                     device: torch.device,
                                     fig_dir: str):
    """
    Figure 5: Temporal Attention Analysis for MambaSurv.

    Extracts attention weights from the attention pooling layer,
    averages across patients, and plots importance over ICU time.
    """

    _set_style()
    logger.info("  Computing temporal attention analysis (Figure 5)…")

    model.eval()

    all_attn = []

    with torch.no_grad():
        for batch in loader:

            X    = batch["X"].to(device)
            dt   = batch["delta_t"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass with attention returned
            _, attn = model(X, dt, mask, return_attention=True)

            # Zero-out padded steps
            attn = attn * mask

            all_attn.append(attn.cpu())

    # (N,T)
    all_attn = torch.cat(all_attn, dim=0)

    # Mean + Std over patients
    mean_attn = all_attn.mean(dim=0).numpy()
    std_attn  = all_attn.std(dim=0).numpy()

    T = len(mean_attn)
    hours = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 5))

    mask = hours <= 336
    ax.plot(hours[mask], mean_attn[mask], linewidth=2)

    # Uncertainty band
    ax.fill_between(
        hours,
        mean_attn - std_attn,
        mean_attn + std_attn,
        alpha=0.3
    )

    ax.set_title("Temporal Attention Weights (MambaSurv)",
                 fontsize=13, fontweight="bold")

    ax.set_xlabel("Hours Since ICU Admission", fontsize=11)
    ax.set_ylabel("Mean Attention Weight", fontsize=11)

    ax.set_xlim(0, 336)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()

    outpath = os.path.join(fig_dir, "fig5_temporal_attention.png")
    plt.savefig(outpath)
    plt.close()

    logger.info("  Saved Figure 5 → fig5_temporal_attention.png")

def fig6_feature_attention_heatmap(model: nn.Module,
                                  loader: DataLoader,
                                  manifest: dict,
                                  device: torch.device,
                                  fig_dir: str,
                                  top_k: int = 25):
    """
    Figure 6 — Feature-level attribution heatmap for MambaSurv.

    Uses Gradient x Input averaged across time and patients
    to estimate which input features most influence risk.

    Produces:
        fig6_feature_importance_heatmap.png
    """

    _set_style()
    logger.info("  Computing feature-level attribution heatmap (Figure 6)…")

    model.eval()

    feat_cols = [c for c in manifest["all_input_features"]
                 if not c.endswith("__delta_t")]

    total_attr = None
    n_batches = 0

    for batch in loader:

        X    = batch["X"].to(device)
        dt   = batch["delta_t"].to(device)
        mask = batch["mask"].to(device)

        # Enable gradients on input
        X.requires_grad = True

        # Forward
        risk = model(X, dt, mask)

        # Scalar objective
        loss = risk.sum()

        # Backprop to inputs
        loss.backward()

        # Gradient × Input attribution
        attr = (X.grad * X).abs()   # (B,T,D)

        # Mask padded steps
        attr = attr * mask.unsqueeze(-1)

        # Aggregate across time + batch
        attr_feat = attr.mean(dim=(0, 1))  # (D,)

        if total_attr is None:
            total_attr = attr_feat.detach().cpu()
        else:
            total_attr += attr_feat.detach().cpu()

        n_batches += 1

        # Important: clear grads
        model.zero_grad()
        X.grad.zero_()

    # Mean attribution across batches
    total_attr /= n_batches
    scores = total_attr.numpy()

    # Top-k features
    top_idx = np.argsort(scores)[::-1][:top_k]
    top_feats = [feat_cols[i] for i in top_idx]
    top_scores = scores[top_idx]

    # Normalize for heatmap
    top_scores = top_scores / (top_scores.max() + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 7))

    heat_data = top_scores.reshape(-1, 1)

    im = ax.imshow(heat_data, aspect="auto")

    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_feats, fontsize=9)

    ax.set_xticks([0])
    ax.set_xticklabels(["Attribution Strength"], fontsize=10)

    ax.set_title("Feature Attribution Heatmap (MambaSurv)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()

    outpath = os.path.join(fig_dir, "fig6_feature_attention_heatmap.png")
    plt.savefig(outpath)
    plt.close()

    logger.info("  Saved Figure 6 → fig6_feature_attention_heatmap.png")

def fig7_risk_distribution(all_preds: dict,
                           fig_dir: str,
                           model_name: str = "mamba_surv"):
    """
    Figure 7 — Risk score distribution for survivors vs non-survivors.

    Plots histogram of predicted risk scores stratified by event outcome.
    Produces:
        fig7_risk_distribution.png
    """

    _set_style()

    if model_name not in all_preds:
        logger.warning("  %s predictions missing — skipping fig8", model_name)
        return

    logger.info("  Generating Figure 7 - Risk Score Distribution…")

    p = all_preds[model_name]

    risk   = p["risk"]
    event  = p["event"]

    # Split by outcome
    risk_dead  = risk[event == 1]
    risk_alive = risk[event == 0]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(risk_alive,
            bins=40,
            alpha=0.6,
            label="Survivors (censored/alive)",
            density=True)

    ax.hist(risk_dead,
            bins=40,
            alpha=0.6,
            label="Deaths (events)",
            density=True)

    ax.set_title("Predicted Risk Score Distribution (MambaSurv)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Risk Score (log hazard)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    ax.legend(fontsize=10)
    plt.tight_layout()

    outpath = os.path.join(fig_dir, "fig7_risk_distribution.png")
    plt.savefig(outpath)
    plt.close()

    logger.info("  Saved Figure 7 → fig7_risk_distribution.png")

# def fig8_subgroup_cindex(model: nn.Module,
#                          dataset: ICUSurvivalDataset,
#                          manifest: dict,
#                          device: torch.device,
#                          fig_dir: str,
#                          subgroup_col: str = "age"):
#     """
#     Figure 8 - Subgroup C-index analysis.

#     Computes Harrell C-index stratified by subgroup.
#     Produces:
#         fig8_subgroup_cindex.png
#     """

#     _set_style()
#     logger.info("  Generating Figure 8 — Subgroup C-index…")

#     # Load full dataframe backing dataset
#     df = dataset.df.copy()

#     if subgroup_col not in df.columns:
#         logger.warning("  Subgroup column '%s' not found — skipping fig9", subgroup_col)
#         return

#     # Example: Age split
#     if subgroup_col == "age":
#         df["subgroup"] = np.where(df["age"] >= 65, "Age ≥ 65", "Age < 65")

#     # Example: Sex split
#     elif subgroup_col == "sex":
#         df["subgroup"] = np.where(df["sex"] == 1, "Male", "Female")

#     else:
#         # Generic binary split
#         median_val = df[subgroup_col].median()
#         df["subgroup"] = np.where(df[subgroup_col] >= median_val,
#                                   f"{subgroup_col} High",
#                                   f"{subgroup_col} Low")

#     subgroup_labels = df["subgroup"].unique()

#     results = {}

#     for sg in subgroup_labels:

#         idx = df.index[df["subgroup"] == sg].tolist()

#         if len(idx) < 30:
#             continue

#         # Build loader for subgroup
#         sub_loader = DataLoader(
#             torch.utils.data.Subset(dataset, idx),
#             batch_size=64,
#             shuffle=False
#         )

#         preds = _predict(model, sub_loader, device, use_amp=False)

#         c_idx = concordance_index(
#             torch.tensor(preds["risk"]),
#             torch.tensor(preds["time"]),
#             torch.tensor(preds["event"])
#         )

#         results[sg] = c_idx
#         logger.info("    Subgroup %-12s  C-index = %.4f", sg, c_idx)

#     fig, ax = plt.subplots(figsize=(7, 5))

#     names = list(results.keys())
#     vals  = list(results.values())

#     ax.bar(names, vals)

#     ax.set_ylim(0.5, 1.0)
#     ax.set_ylabel("Harrell's C-index", fontsize=12)

#     ax.set_title("Figure 8 - Subgroup Performance (MambaSurv)",
#                  fontsize=13, fontweight="bold")

#     plt.tight_layout()

#     outpath = os.path.join(fig_dir, "fig8_subgroup_cindex.png")
#     plt.savefig(outpath)
#     plt.close()

#     logger.info("  Saved Figure 8 → fig8_subgroup_cindex.png")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="./processed")
    parser.add_argument("--results_dir",   type=str, default="./results")
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--max_len",       type=int, default=2160)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    with open(os.path.join(args.processed_dir, "feature_manifest.json")) as fh:
        manifest = json.load(fh)

    test_ds = ICUSurvivalDataset(
        os.path.join(args.processed_dir, "test.parquet"), manifest, max_len=args.max_len
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_preds: dict[str, dict] = {}
    all_models: dict[str, nn.Module] = {}

    for m_name in MODEL_NAMES:
        logger.info("Evaluating %s …", PRETTY_NAMES[m_name])
        model = _load_model(m_name, manifest, args.results_dir, device)
        if model is None:
            continue
        preds = _predict(model, test_loader, device, use_amp=(device.type == "cuda"))
        all_preds[m_name]  = preds
        all_models[m_name] = model

    logger.info("\n─── Test Set Metrics ───")
    metrics_out = {}

    # Find median event time for Brier score horizon
    all_times  = np.concatenate([p["time"] for p in all_preds.values()])
    all_events = np.concatenate([p["event"] for p in all_preds.values()])
    t_brier    = float(np.median(all_times[all_events == 1]))   # median event time

    for m_name, p in all_preds.items():
        c_idx  = concordance_index(torch.tensor(p["risk"]), torch.tensor(p["time"]), torch.tensor(p["event"]))
        b_score = _brier_score_at_t(p["risk"], p["time"], p["event"], t_brier)

        entry = {
            "c_index":     round(c_idx, 4),
            "brier_score": round(b_score, 4),
            "n_patients":  int(len(p["risk"])),
            "brier_t_eval_hours": round(t_brier, 1),
        }
        metrics_out[m_name] = entry
        logger.info("  %-20s  C=%.4f  Brier=%.4f  (n=%d)", PRETTY_NAMES[m_name],
                    entry["c_index"], entry["brier_score"], entry["n_patients"])

    with open(os.path.join(args.results_dir, "test_metrics.json"), "w") as fh:
        json.dump(metrics_out, fh, indent=2)
    logger.info("  Wrote test_metrics.json")

    # ── Generate figures ─────────────────────────────────────────────────────
    logger.info("\nGenerating figures …")
    # fig1_cindex_bar(all_preds, fig_dir)
    # fig2_km_quartiles(all_preds, fig_dir)
    # fig3_calibration(all_preds, fig_dir)
    # fig4_training_curves(args.results_dir, fig_dir)
    # fig7_risk_distribution(all_preds, fig_dir, model_name="mamba_surv")

    if "mamba_surv" in all_models:
        fig5_temporal_attention_analysis(
            model=all_models["mamba_surv"],
            loader=test_loader,
            device=device,
            fig_dir=fig_dir
        )
        # fig6_feature_attention_heatmap(
        #     all_models["mamba_surv"],
        #     test_loader,
        #     manifest,
        #     device,
        #     fig_dir
        # )
        # fig8_subgroup_cindex(
        #     all_models["mamba_surv"],
        #     test_ds,
        #     manifest,
        #     device,
        #     fig_dir,
        #     subgroup_col="age"   # or "sex"
        # )

    logger.info("All figures saved to %s", fig_dir)


if __name__ == "__main__":
    main()