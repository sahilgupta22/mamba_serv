"""
02_feature_engineering.py
─────────────────────────
Consumes the preprocessed Parquet from script 01 and produces:

    1.  Delta-T columns  — "hours since last observation" for each feature
        group (labs, vitals, vent).  This is the "Delta-T trick" that lets
        Mamba learn input-dependent decay of stale values.
    2.  Derived clinical ratios — P/F ratio, pulse-pressure, shock index,
        estimated dynamic lung compliance.
    3.  One-hot encoding of categorical statics.
    4.  StandardScaler normalisation of all numeric features (fit on TRAIN
        only, applied to val/test).
    5.  Stratified 70 / 15 / 15  train / val / test split on death_outcome
        at the *patient* level.

Outputs (all in --output_dir):
    train.parquet, val.parquet, test.parquet   — final model-ready splits
    scaler_stats.parquet                       — mean / std for reproducibility
    feature_manifest.json                      — ordered feature lists by group

Usage:
    python 02_feature_engineering.py --input_parquet ./processed/patients_preprocessed.parquet \
                                     --output_dir   ./processed \
                                     --seed         42
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Feature-group membership
LAB_FEATURES = ["calculated_bicarbonate", "so2", "pCO2", "pO2", "pH"]

VITALS_FEATURES = [
    "heart_rate", "sbp", "dbp", "mbp",
    "sbp_ni", "dbp_ni", "mbp_ni",
    "temperature", "spo2", "glucose",
]

VENT_FEATURES = [
    "ppeak", "set_peep", "total_peep",
    "rr", "set_rr", "total_rr",
    "set_tv", "total_tv",
    "set_fio2", "set_ie_ratio", "set_pc",
    "set_pc_draeger", "pinsp_draeger", "pinsp_hamilton", "pcv_level",
]

OTHER_FEATURES = ["gcs", "gcs_motor", "gcs_verbal", "gcs_eyes", "gcs_unable", "sofa_24hours"]

TREATMENT_FEATURES = ["invasive", "noninvasive", "highflow", "vasopressor", "crrt"]

# Static numeric columns that will be normalised
STATIC_NUMERIC = ["anchor_age", "pbw_kg", "height_inch", "elixhauser_vanwalraven"]

# Static categorical columns to one-hot encode
STATIC_CATEGORICAL = ["gender", "insurance", "language", "marital_status", "race", "first_careunit"]

# The primary survival outcome
OUTCOME_COL = "death_outcome"
# Secondary outcomes kept but not used as features
OUTCOME_COLS = ["death_outcome", "survival_event", "survival_time_hours", "discharge_outcome", "icuouttime_outcome", "sepsis", "los"]


# Helper: Delta-T computation (vectorised, group-aware)
def _compute_delta_t(df: pd.DataFrame, feature_cols: list[str], group_col: str = "subject_id") -> pd.DataFrame:
    """
    For each feature in `feature_cols`, add a column  <feat>__delta_t  that
    records how many hours have elapsed since the last non-NaN observation
    of that feature, within each patient's time series.

    At hour 0 (or before the first observation) the value is 0.
    """
    for feat in feature_cols:
        obs_col = f"{feat}__obs"
        if obs_col not in df.columns:
            # Fallback for derived features which don't have masks from Script 01
            # For these, we assume any non-zero value is a 'new' observation
            is_obs = (df[feat] != 0.0)
        else:
            is_obs = (df[obs_col] == 1.0)
        
        col_name = f"{feat}__delta_t"

        # Within each group compute cumulative time since last obs
        row_idx = df.groupby(group_col).cumcount()
        last_obs_idx = row_idx.where(is_obs).groupby(df[group_col]).ffill().fillna(0)
        delta_t = (row_idx - last_obs_idx).astype(np.float32)

        # Prevent post-event monotonic leakage
        if "survival_time_hours" in df.columns:
            delta_t = delta_t.clip(upper=336)

        df[col_name] = delta_t

    return df


# Derived clinical features
def _derive_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standard ICU ratios from existing columns.  All divisions are
    guarded against divide-by-zero with np.where.
    """
    # P/F ratio = pO2 / FiO2  (classic ARDS severity metric)
    # set_fio2 is a fraction in [0.21, 1.0]
    if "pO2" in df.columns and "set_fio2" in df.columns:
        df["pf_ratio"] = np.where(
            df["set_fio2"] > 0,
            df["pO2"] / df["set_fio2"],
            np.nan,
        ).astype(np.float32)

    # Pulse pressure = SBP - DBP
    if "sbp" in df.columns and "dbp" in df.columns:
        df["pulse_pressure"] = (df["sbp"] - df["dbp"]).astype(np.float32)

    # Shock index = HR / SBP
    if "heart_rate" in df.columns and "sbp" in df.columns:
        df["shock_index"] = np.where(
            df["sbp"] > 0,
            df["heart_rate"] / df["sbp"],
            np.nan,
        ).astype(np.float32)

    # Estimated dynamic compliance ≈ TV / (Ppeak - PEEP)
    # Using total_tv and total_peep; ppeak is peak inspiratory pressure
    if all(c in df.columns for c in ["total_tv", "ppeak", "total_peep"]):
        denom = df["ppeak"] - df["total_peep"]
        df["dyn_compliance"] = np.where(
            denom > 0,
            df["total_tv"] / denom,
            np.nan,
        ).astype(np.float32)

    # pH-corrected pCO2 deviation (acid-base imbalance proxy)
    if "pCO2" in df.columns and "pH" in df.columns:
        df["pco2_ph_deviation"] = ((df["pCO2"] - 40.0) * (df["pH"] - 7.4)).astype(np.float32)

    return df


DERIVED_FEATURES = ["pf_ratio", "pulse_pressure", "shock_index", "dyn_compliance", "pco2_ph_deviation"]


# Cumulative treatment-hour features (causal)
# def _cumulative_treatment_hours(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     For each binary treatment indicator, add a column that is the cumulative
#     sum of hours on that treatment up to and including hour t.  This is
#     causal (no future information leaks).
#     """
#     for t_col in TREATMENT_FEATURES:
#         if t_col not in df.columns:
#             continue
#         cum_col = f"{t_col}__cumhours"
#         df[cum_col] = df.groupby("subject_id")[t_col].cumsum().astype(np.float32)
#     return df


# One-hot encoding of static categoricals
def _onehot_statics(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return augmented df and list of new dummy column names."""
    dummies_list: list[str] = []
    for cat_col in STATIC_CATEGORICAL:
        if cat_col not in df.columns:
            continue
        dummies = pd.get_dummies(df[cat_col].astype(str), prefix=cat_col, dtype=np.float32)
        # Drop "unknown" / NaN category if present
        drop_cols = [c for c in dummies.columns if "nan" in c.lower() or "unknown" in c.lower()]
        dummies = dummies.drop(columns=drop_cols, errors="ignore")
        dummies_list.extend(dummies.columns.tolist())
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[cat_col])
    return df, dummies_list


def main():
    parser = argparse.ArgumentParser(description="Feature engineering for Mamba-Surv pipeline")
    parser.add_argument("--input_parquet", type=str, required=True)
    parser.add_argument("--output_dir",    type=str, default="./processed")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    t0 = time.time()

    # Load
    logger.info("Loading preprocessed parquet …")
    df = pd.read_parquet(args.input_parquet)
    logger.info("  Shape: %s", df.shape)

    # Derived clinical features
    logger.info("Deriving clinical ratio features …")
    df = _derive_clinical_features(df)
    df = df.fillna(0.0)

    # Delta-T for each feature group
    logger.info("Computing Delta-T (hours-since-last-obs) columns …")
    df = _compute_delta_t(df, LAB_FEATURES)
    df = _compute_delta_t(df, VITALS_FEATURES)
    df = _compute_delta_t(df, VENT_FEATURES)
    df = _compute_delta_t(df, OTHER_FEATURES)
    df = _compute_delta_t(df, DERIVED_FEATURES)

    delta_t_cols = [c for c in df.columns if c.endswith("__delta_t")]
    logger.info("  Created %d Delta-T columns", len(delta_t_cols))

    # One-hot static categoricals
    logger.info("One-hot encoding static categoricals …")
    df, onehot_cols = _onehot_statics(df)
    logger.info("  Created %d one-hot columns: %s …", len(onehot_cols), onehot_cols[:6])

    # Identify and remove the temporary observation masks to keep the feature set clean
    obs_cols = [c for c in df.columns if c.endswith("__obs")]
    df = df.drop(columns=obs_cols)

    # Collect ALL numeric feature columns (excludes IDs / outcomes / times)
    exclude = {"subject_id", "hour", "intime", "outtime", "anchor_year"} | set(OUTCOME_COLS)
    all_feature_cols = [c for c in df.columns if c not in exclude]

    # Identify numeric columns for StandardScaler
    numeric_features = [
        c for c in all_feature_cols
        if df[c].dtype in (np.float32, np.float64, np.int64, np.int32, np.float16)
    ]

    # Patient-level stratified split
    logger.info("Splitting patients (70 / 15 / 15) stratified on death_outcome …")

    # Compute per-patient outcome for stratification
    # A patient is "dead" if death_outcome == 1 at any hour
    patient_outcomes = (
        df.groupby("subject_id")[OUTCOME_COL]
        .max()
        .reset_index()
        .rename(columns={OUTCOME_COL: "patient_died"})
    )
    subject_ids = patient_outcomes["subject_id"].values
    strat_labels = patient_outcomes["patient_died"].values

    logger.info("  Mortality rate in cohort: %.2f%%", strat_labels.mean() * 100)

    # First split: 70 train vs 30 temp
    train_ids, temp_ids, _, temp_labels = train_test_split(
        subject_ids, strat_labels,
        test_size=0.30, stratify=strat_labels, random_state=args.seed,
    )
    # Second split: 30 → 15 val + 15 test
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids, temp_labels,
        test_size=0.50, stratify=temp_labels, random_state=args.seed,
    )

    logger.info("  Train / Val / Test patients: %d / %d / %d",
                len(train_ids), len(val_ids), len(test_ids))

    train_mask = df["subject_id"].isin(train_ids)
    val_mask   = df["subject_id"].isin(val_ids)
    test_mask  = df["subject_id"].isin(test_ids)

    # Fit StandardScaler on TRAIN only
    logger.info("Fitting StandardScaler on training set and filling NaNs ...")
    train_df = df.loc[train_mask, numeric_features]
    means = train_df.mean()
    stds  = train_df.std().replace(0, 1.0)  # avoid /0 for constant cols

    # Apply normalisation AND fill
    for col in numeric_features:
        df[col] = ((df[col] - means[col]) / stds[col]).astype(np.float32)
        df[col] = df[col].fillna(0.0)
    
    # Do the same for Delta-T columns if they have NaNs
    for col in delta_t_cols:
        df[col] = df[col].fillna(0.0)

    # Save scaler stats for inference / reproducibility
    scaler_stats = pd.DataFrame({"feature": numeric_features,
                                 "mean": means.values,
                                 "std":  stds.values})
    scaler_stats.to_parquet(os.path.join(args.output_dir, "scaler_stats.parquet"), index=False)

    # Write splits
    logger.info("Writing train / val / test Parquets …")
    df[train_mask].to_parquet(os.path.join(args.output_dir, "train.parquet"), index=False)
    df[val_mask].to_parquet(os.path.join(args.output_dir, "val.parquet"),   index=False)
    df[test_mask].to_parquet(os.path.join(args.output_dir, "test.parquet"), index=False)

    # Feature manifest (ordered lists, for the model to consume)
    manifest = {
        "lab_features":          LAB_FEATURES,
        "vitals_features":       VITALS_FEATURES,
        "vent_features":         VENT_FEATURES,
        "other_features":        OTHER_FEATURES,
        "treatment_features":    TREATMENT_FEATURES,
        "derived_features":      DERIVED_FEATURES,
        "delta_t_cols":          delta_t_cols,
        # "cumhour_cols":          cumhour_cols,
        "onehot_cols":           onehot_cols,
        "static_numeric":        STATIC_NUMERIC,
        "all_input_features":    [c for c in all_feature_cols],   # full ordered list
        "numeric_features":      numeric_features,
        "outcome_cols":          OUTCOME_COLS,
        "input_dim":             len(all_feature_cols),
    }
    with open(os.path.join(args.output_dir, "feature_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info("\n─── Feature Engineering Summary ───")
    logger.info("  Total input features      : %d", manifest["input_dim"])
    logger.info("    Lab + Delta-T           : %d + %d", len(LAB_FEATURES), len([c for c in delta_t_cols if any(l in c for l in LAB_FEATURES)]))
    logger.info("    Vitals + Delta-T        : %d + %d", len(VITALS_FEATURES), len([c for c in delta_t_cols if any(v in c for v in VITALS_FEATURES)]))
    logger.info("    Ventilation + Delta-T   : %d + %d", len(VENT_FEATURES), len([c for c in delta_t_cols if any(v in c for v in VENT_FEATURES)]))
    logger.info("    Derived clinical        : %d", len(DERIVED_FEATURES))
    # logger.info("    Treatment + cumhours    : %d + %d", len(TREATMENT_FEATURES), len(cumhour_cols))
    logger.info("    One-hot static          : %d", len(onehot_cols))
    logger.info("  Wall time: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()