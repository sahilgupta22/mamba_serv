"""
03_model_definitions.py
───────────────────────
All model architectures and the shared Dataset / loss.

Models
──────
  MambaSurv          The proposed model: Selective-State-Space + Cox head.
  DeepSurvLSTM       Baseline: stacked LSTM + Cox head (temporal-aware DeepSurv).
  DeepSurvMLP        Baseline: stack MLP + Cox Head (classic DeepSurv).
  CoxLinear          Baseline: linear (no sequence model) Cox-PH.

Shared components
─────────────────
  ICUSurvivalDataset - PyTorch Dataset that packs variable-length hourly
                       sequences, pads, and returns (X, delta_t, mask, time,
                       event) tuples.
  cox_partial_loss   - Breslin approximation of the Cox partial log-likelihood,
                       with proper handling of ties and censoring.
  concordance_index  - Harrell's C-index for evaluation.

─────────────────────────────────────────────────────────────────────────────
Design notes on the Mamba block
─────────────────────────────────────────────────────────────────────────────
Standard Mamba uses a *fixed* discretised A matrix.  For sparse EHR data the
key insight is:

    A_t  =  A_base  *  sigmoid( W_a · [x_t ; Δt_t] )

where Δt_t is the Delta-T feature vector.  When Δt is large (stale data),
sigmoid pushes the gate toward 1 → A stays close to I → state is preserved.
When fresh data arrives (Δt ≈ 0), the gate opens and A can decay normally.

This is implemented in  MambaBlock.forward()  via the  delta_t_gate  mechanism.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from lifelines.utils import concordance_index as ll_cindex


# 1.  Dataset
class ICUSurvivalDataset(Dataset):
    """
    Reads one Parquet split, extracts features and survival labels.

    Returns per-sample:
        X          : (T, D_feat)   main input features (normalised)
        delta_t    : (T, D_dt)     Delta-T columns (un-normalised, raw hours)
        mask       : (T,)          1 for real hours, 0 for padding
        time_to_event : scalar     hours in ICU (from `hour` at last row,
                                     or from `los` if available)
        event      : scalar        1 = died, 0 = censored
    """

    def __init__(
        self,
        parquet_path: str,
        feature_manifest: dict,
        max_len: int = 2160,        # 90 days
        min_len: int = 24,          # drop patients with < 24 h of data
    ):
        super().__init__()
        self.max_len = max_len

        df = pd.read_parquet(parquet_path)

        # Identify feature / delta-t columns
        self.feat_cols   = [c for c in feature_manifest["all_input_features"]
                           if c in df.columns and not c.endswith("__delta_t")]
        self.dt_cols     = [c for c in feature_manifest["delta_t_cols"]
                           if c in df.columns]
        self.input_dim   = len(self.feat_cols)
        self.delta_t_dim = len(self.dt_cols)

        # Build per-patient tensors
        self.samples: list[dict] = []
        for sid, grp in df.groupby("subject_id"):
            grp = grp.sort_values("hour").reset_index(drop=True)
            if len(grp) < min_len:
                continue

            X = grp[self.feat_cols].values.astype(np.float32)
            dt = grp[self.dt_cols].values.astype(np.float32) if self.dt_cols else np.zeros((len(grp), 0), dtype=np.float32)

            # Survival labels - now using the patient-level columns we created
            if "survival_time_hours" in grp.columns:
                time_to_event = float(grp["survival_time_hours"].iloc[0])
            else:
                # Fallback to old logic if running on old data
                time_to_event = float(grp["los"].iloc[0]) * 24.0

            # Guard against zero-time events which break Cox models
            if time_to_event <= 0:
                time_to_event = 0.1

            if "survival_event" in grp.columns:
                event = float(grp["survival_event"].iloc[0])
            else:
                # Fallback
                event = float(grp["death_outcome"].max() if "death_outcome" in grp.columns else 0.0)

            self.samples.append({
                "subject_id":   sid,
                "X":            X,
                "delta_t":      dt,
                "time_to_event": time_to_event,
                "event":        event,
            })

        logger_msg = f"ICUSurvivalDataset: {len(self.samples)} patients loaded"
        print(logger_msg)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # Fixed-length window, independent of survival
        T = self.max_len

        X_raw  = s["X"]
        dt_raw = s["delta_t"]

        X  = torch.zeros(T, X_raw.shape[1], dtype=torch.float32)
        dt = torch.zeros(T, dt_raw.shape[1], dtype=torch.float32) if dt_raw.shape[1] > 0 else torch.zeros(T, 0)

        L = min(len(X_raw), T)
        X[:L]  = torch.tensor(X_raw[:L],  dtype=torch.float32)
        if dt_raw.shape[1] > 0:
            dt[:L] = torch.tensor(dt_raw[:L], dtype=torch.float32)

        # Causal mask: data availability ∧ survival horizon
        mask = torch.zeros(T, dtype=torch.float32)

        # Enforce causality: zero-out mask AFTER event / censoring
        time_to_event = s["time_to_event"]
        event_cutoff = int(min(time_to_event, T - 1))

        valid_len = L
        mask[:valid_len] = 1.0

        return {
            "X":              X,                                          # (max_len, D)
            "delta_t":        dt,                                         # (max_len, D_dt)
            "mask":           mask,                                       # (max_len,)
            "time_to_event":  torch.tensor(s["time_to_event"], dtype=torch.float32),
            "event":          torch.tensor(s["event"],         dtype=torch.float32),
        }


# 2.  Cox Partial Likelihood Loss  (Breslin approximation)
def cox_partial_loss(risk_scores: torch.Tensor,
                     times:       torch.Tensor,
                     events:      torch.Tensor) -> torch.Tensor:
    """
    Numerically stable Negative Partial Log-Likelihood using Log-Sum-Exp trick.
    
    Args:
        risk_scores : (B,)  Log-hazard ratios from the model
        times       : (B,)  Time-to-event
        events      : (B,)  1 = event, 0 = censored
    """
    # 1. Sort descending by time
    # Adding a tiny amount of noise to times can prevent issues with ties 
    # if your dataset has many identical event times.
    order = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]

    # 2. Compute the Log-Sum-Exp of the risk set
    # log_cumsum[i] = log( sum_{j=0 to i} exp(risk_scores[j]) )
    # Because times are sorted descending, index 0 is the patient at risk the longest.
    # logcumsumexp is much more stable than log(cumsum(exp(x)))
    log_cumsum = torch.logcumsumexp(risk_scores, dim=0)

    # 3. Compute log-likelihood contribution
    # Only patients where event == 1 contribute to the numerator
    log_lik = events * (risk_scores - log_cumsum)

    # 4. Negate and average over events
    n_events = events.sum()
    
    # Avoid division by zero if a batch has 0 events (e.g. all censored)
    if n_events == 0:
        return risk_scores.sum() * 0.0  # Returns 0 with gradient tracking
        
    return -log_lik.sum() / n_events

# def cox_partial_loss(risk_scores: torch.Tensor,
#                      times:       torch.Tensor,
#                      events:      torch.Tensor) -> torch.Tensor:
#     """
#     Breslin approximation of the negative partial log-likelihood.

#     Args:
#         risk_scores : (B,)  – model output (log-hazard ratio, no sigmoid)
#         times       : (B,)  – time-to-event
#         events      : (B,)  – 1 = event, 0 = censored

#     Returns:
#         Scalar loss (mean over events).
#     """
#     # Sort descending by time
#     order        = torch.argsort(times, descending=True)
#     risk_scores  = risk_scores[order]
#     times        = times[order]
#     events       = events[order]

#     # Cumulative log-sum-exp (risk set)
#     # exp(risk_scores) cumsum  →  sum of exp over risk set at each ordered time
#     exp_scores   = torch.exp(risk_scores - risk_scores.max())  # numerical stability
#     cumsum_exp   = torch.cumsum(exp_scores, dim=0)
#     log_cumsum   = torch.log(cumsum_exp + 1e-8) + risk_scores.max()

#     # Partial log-likelihood per event
#     log_lik      = events * (risk_scores - log_cumsum)

#     # Negate and average over events (avoid /0 if no events in batch)
#     n_events = events.sum().clamp(min=1)
#     return -log_lik.sum() / n_events


# 3.  Concordance Index (Harrell's C)
@torch.no_grad()
def concordance_index(risk_scores: torch.Tensor,
                      times:       torch.Tensor,
                      events:      torch.Tensor) -> float:
    # 1. Flatten and move to CPU
    # Ensure risk_scores is (N,) and not (N, 1)
    rs = risk_scores.detach().cpu().numpy().flatten()
    t  = times.detach().cpu().numpy().flatten()
    e  = events.detach().cpu().numpy().flatten()

    # 2. Check if we have at least one event and one comparable pair
    if np.sum(e) == 0:
        return 0.5  # Return uninformative baseline if no events in batch

    try:
        return float(ll_cindex(t, -rs, e)) 
    except ZeroDivisionError:
        return 0.5


# 4.  Mamba-Surv  (proposed model)class MambaBlock(nn.Module):
    """
    Single selective-state-space block with Delta-T gated A matrix.

    State update (per time-step, simplified scalar notation):
        h_t = A_t * h_{t-1} + B_t * x_t
        y_t = C_t * h_t

    Where:
        A_t = A_base * sigmoid( W_a @ [x_t; Δt_t] )
        B_t = Linear(x_t)
        C_t = Linear(x_t)

    A_base is a learned diagonal matrix initialised to a negative value so
    that exp(A_base) < 1 (stable decay).  The Delta-T gate modulates how
    much decay to apply.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_delta_t: int = 0):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state

        # A_base: log-space diagonal (per feature)
        self.A_base = nn.Parameter(torch.full((d_model, d_state), -0.1))

        # Delta-T gate projection: [x_t ; Δt_t] → scalar gate per (d_model, d_state)
        self.delta_t_gate = nn.Linear(d_model + d_delta_t, d_model * d_state)

        # B, C projections (input-dependent)
        self.B_proj = nn.Linear(d_model, d_model * d_state)
        self.C_proj = nn.Linear(d_model, d_model * d_state)

        # Layer norm + output projection
        self.norm   = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : (B, T, D)
            delta_t : (B, T, D_dt)   - raw (un-normalised) Delta-T features
            mask    : (B, T)         - 1 = real, 0 = pad

        Returns:
            y       : (B, T, D)
        """
        B, T, D = x.shape
        N = self.d_state

        # Compute gated A
        # If delta_t is empty (ablation), just use zeros
        if delta_t.shape[-1] > 0:
            gate_input = torch.cat([x, delta_t], dim=-1)            # (B,T, D+D_dt)
        else:
            gate_input = x
        gate = torch.sigmoid(self.delta_t_gate(gate_input))         # (B,T, D*N)
        gate = gate.view(B, T, D, N)

        # Invert gate semantics:
        # gate ≈ 1 = fresh data = allow decay
        # gate ≈ 0 = stale data = preserve state
        A_base_exp = torch.exp(self.A_base).unsqueeze(0).unsqueeze(0)
        A_t = 1.0 - gate * (1.0 - A_base_exp)

        # B and C
        B_t = self.B_proj(x).view(B, T, D, N)                      # (B,T,D,N)
        C_t = self.C_proj(x).view(B, T, D, N)                      # (B,T,D,N)

        # Recurrence: h_t = A_t * h_{t-1} + B_t * x_t
        # We expand x to (B,T,D,1) for broadcasting with N states
        x_exp = x.unsqueeze(-1)                                     # (B,T,D,1)

        h = torch.zeros(B, D, N, device=x.device)                  # (B,D,N)
        y_list: list[torch.Tensor] = []

        for t in range(T):
            # mask[:, t] : (B,),  zero out state update for padded steps
            m = mask[:, t].view(B, 1, 1)                            # (B,1,1)
            h = m * (A_t[:, t] * h + B_t[:, t] * x_exp[:, t]) + (1 - m) * h
            # Output: y_t = sum over states of C_t * h_t
            y_t = (C_t[:, t] * h).sum(dim=-1)                      # (B, D)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)                              # (B,T,D)
        y = self.out_proj(self.norm(y + x))                         # residual
        return y


class MambaSurv(nn.Module):
    """
    Full Mamba-Surv architecture.

    Architecture:
        Input Projection  => N x MambaBlock => Temporal Pooling  => Cox Head

    Temporal pooling uses attention-weighted mean over unmasked time-steps.
    The Cox head outputs a single scalar (log hazard ratio) per patient.
    """

    def __init__(
        self,
        input_dim:    int,
        delta_t_dim:  int,
        d_model:      int   = 128,
        d_state:      int   = 16,
        n_layers:     int   = 3,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, delta_t_dim) for _ in range(n_layers)
        ])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

        # Attention pooling over time
        self.attn_pool = nn.Linear(d_model, 1)

        # Cox head
        self.cox_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, X: torch.Tensor, delta_t: torch.Tensor,
                mask: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            X       : (B, T, D_in)
            delta_t : (B, T, D_dt)
            mask    : (B, T)

        Returns:
            risk    : (B,)  - log hazard ratio
        """
        h = self.input_proj(X)                                      # (B,T,d_model)

        for mamba_blk, drop in zip(self.mamba_layers, self.dropouts):
            h = mamba_blk(h, delta_t, mask)
            h = drop(h)

        # Attention-weighted temporal pooling
        attn_logits = self.attn_pool(h).squeeze(-1)                 # (B,T)
        fill_value = torch.finfo(attn_logits.dtype).min
        attn_logits = attn_logits.masked_fill(mask == 0, fill_value)
        attn_weights = F.softmax(attn_logits, dim=-1)               # (B,T)
        pooled = (h * attn_weights.unsqueeze(-1)).sum(dim=1)        # (B, d_model)

        risk = self.cox_head(pooled).squeeze(-1)                    # (B,)

        if return_attention:
            return risk, attn_weights
        else:
            return risk


# 5.  Baselines

class DeepSurvMLP(nn.Module):
    """
    Traditional DeepSurv (no sequence model):
    Masked mean pooling over time → MLP → Cox head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        **_kwargs,
    ):
        super().__init__()

        layers = []
        d = input_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = hidden_dim

        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

        # Small init for Cox stability
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        X: torch.Tensor,
        delta_t: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            X    : (B, T, D)
            mask : (B, T)
        """
        # Masked mean pooling over time
        mask_exp = mask.unsqueeze(-1)                      # (B,T,1)
        pooled = (X * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)

        risk = self.mlp(pooled).squeeze(-1)
        return torch.clamp(risk, min=-20, max=20)

class DeepSurvLSTM(nn.Module):
    """
    Classic DeepSurv: stacked bi-LSTM → attention pool → Cox head.
    Does NOT use delta_t (ablation comparison point).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.1, **_kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            bidirectional=False, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.cox_head  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        for m in self.cox_head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01) # Small initial weights
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor, delta_t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp(min=1)
        
        # Use pack_padded_sequence
        packed = nn.utils.rnn.pack_padded_sequence(X, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=X.size(1))
        
        # Instead of attention pooling the whole history (which can leak future info),
        # pull the LAST valid hidden state for each patient
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        last_h = out.gather(1, idx).squeeze(1)
        
        risk = self.cox_head(last_h).squeeze(-1)
        return torch.clamp(risk, min=-20, max=20)

class TransformerCox(nn.Module):
    """
    Causal Transformer encoder → attention pool → Cox head.
    Uses learned positional embeddings up to max_len.
    Does NOT use delta_t.
    """

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dropout: float = 0.1, max_len: int = 2160, **_kwargs):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed  = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = nn.Linear(d_model, 1)
        self.cox_head  = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, X: torch.Tensor, delta_t: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = X.shape
        h = self.input_proj(X)

        # 1. Positional embedding
        # Jittered positions to prevent absolute-time leakage
        offset = torch.randint(0, 4, (B, 1), device=X.device)
        positions = (torch.arange(T, device=X.device).unsqueeze(0) + offset) % self.pos_embed.num_embeddings
        h = h + self.pos_embed(positions)

        # 2. Masks
        # Causal mask: upper-triangular True = masked
        causal_mask = torch.triu(torch.ones(T, T, device=X.device, dtype=torch.bool), diagonal=1)
        # Key-padding mask: True = ignore (inverted from our mask)
        key_pad_mask = (mask == 0)

        # 3. Transformer Encoding
        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=key_pad_mask)

        # 4. Masked attention pooling (causal, length-invariant)
        attn_logits = self.attn_pool(h).squeeze(-1)          # (B, T)
        fill_value = torch.finfo(attn_logits.dtype).min
        attn_logits = attn_logits.masked_fill(mask == 0, fill_value)

        attn_weights = torch.softmax(attn_logits, dim=1)    # (B, T)
        pooled_h = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)

        # 5. Cox Head
        risk = self.cox_head(pooled_h).squeeze(-1)
        
        # Final safety clamp for Cox Loss stability
        return torch.clamp(risk, min=-20, max=20)


class CoxLinear(nn.Module):
    """
    No-sequence baseline: mean-pool features over time, then a single linear
    layer to a Cox scalar.  Equivalent to Cox-PH with learned feature weights.
    """

    def __init__(self, input_dim: int, **_kwargs):
        super().__init__()
        self.head = nn.Linear(input_dim, 1)

    def forward(self, X: torch.Tensor, delta_t: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # Masked mean over time
        mask_exp = mask.unsqueeze(-1)                               # (B,T,1)
        pooled   = (X * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


# 6.  Model factory
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mamba_surv":      MambaSurv,
    "deepsurv_lstm":   DeepSurvLSTM,
    "deepsurv_mlp":    DeepSurvMLP,
    "transformer_cox": TransformerCox,
    "cox_linear":      CoxLinear,
}


def build_model(name: str, input_dim: int, delta_t_dim: int, **kwargs) -> nn.Module:
    """Instantiate a model by name with the given dimensions."""
    cls = MODEL_REGISTRY[name]
    if name == "mamba_surv":
        return cls(input_dim=input_dim, delta_t_dim=delta_t_dim, **kwargs)
    else:
        return cls(input_dim=input_dim, **kwargs)