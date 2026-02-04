"""
01_load_and_preprocess.py
─────────────────────────
Loads the nested-directory CSV dataset (one file per patient, grouped by
first-3-digit folders), applies clinical validity filters, forward-fills
sparse time-varying measurements, and writes a single Parquet file.

**Weaning-task filtering**: Only patients who were on invasive mechanical
ventilation at any point are included. Each patient's timeline is truncated
to 14 days from first occurrence of invasive ventilation. This reduces the 
dataset size by ~65% compared to the full 90-day cohort.

Usage:
    python 01_load_and_preprocess.py --dataset_root /path/to/dataset \
                                     --output_dir   /path/to/processed \
                                     --num_workers  8
"""

import argparse
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STATIC_COLS = [
    "intime", "outtime", "gender", "anchor_year", "anchor_age",
    "insurance", "language", "marital_status", "race",
    "first_careunit", "pbw_kg", "height_inch", "elixhauser_vanwalraven",
]

# ventilation parameters
VENT_COLS = [
    "ppeak", "set_peep", "total_peep",
    "rr", "set_rr", "total_rr",
    "set_tv", "total_tv",
    "set_fio2", "set_ie_ratio", "set_pc",
    "set_pc_draeger", "pinsp_draeger", "pinsp_hamilton", "pcv_level",
]

LAB_COLS = [
    "calculated_bicarbonate", "so2", "pCO2", "pO2", "pH",
]

VITALS_COLS = [
    "heart_rate", "sbp", "dbp", "mbp",
    "sbp_ni", "dbp_ni", "mbp_ni",
    "temperature", "spo2", "glucose",
]

OTHER_MEASUREMENT_COLS = [
    "gcs", "gcs_motor", "gcs_verbal", "gcs_eyes", "gcs_unable",
    "sofa_24hours",
]

TREATMENT_COLS = [
    "invasive", "noninvasive", "highflow", "vasopressor", "crrt",
]

OUTCOME_COLS = [
    "discharge_outcome", "icuouttime_outcome", "death_outcome",
    "sepsis", "los",
]

# union of every time-varying numeric column we will forward-fill
TIME_VARYING_NUMERIC = VENT_COLS + LAB_COLS + VITALS_COLS + OTHER_MEASUREMENT_COLS

# clinical validity windows (inclusive)
VALIDITY_BOUNDS: dict[str, tuple[float, float]] = {
    # Ventilation
    "ppeak":            (0, 80),
    "set_peep":        (0, 30),
    "total_peep":       (0, 40),
    "rr":               (1, 60),
    "set_rr":          (1, 60),
    "total_rr":         (1, 60),
    "set_tv":          (50, 2000),
    "total_tv":         (50, 2000),
    "set_fio2":        (0.21, 1.0),   # fraction, not percentage
    "set_ie_ratio":    (0.2, 4.0),    # I:E ratio
    "set_pc":          (0, 60),
    "set_pc_draeger":  (0, 60),
    "pinsp_draeger":  (0, 80),
    "pinsp_hamilton": (0, 80),
    "pcv_level":      (0, 60),
    # Labs
    "calculated_bicarbonate": (5, 45),
    "so2":              (50, 100),
    "pCO2":             (10, 150),
    "pO2":              (20, 600),
    "pH":               (6.5, 7.7),
    # Vitals
    "heart_rate":       (10, 250),
    "sbp":              (30, 300),
    "dbp":              (10, 200),
    "mbp":              (20, 250),
    "sbp_ni":           (30, 300),
    "dbp_ni":           (10, 200),
    "mbp_ni":           (20, 250),
    "temperature":      (25, 45),      # celsius
    "spo2":             (50, 100),
    "glucose":          (20, 700),
    # GCS / SOFA
    "gcs":              (3, 15),
    "gcs_motor":        (1, 6),
    "gcs_verbal":       (1, 5),
    "gcs_eyes":         (1, 4),
    "sofa_24hours":    (0, 24),
}


def _load_and_preprocess_patient(args: tuple[str, int]) -> Optional[pd.DataFrame]:
    """
    Worker entry-point: read one CSV, run the full per-patient pipeline,
    return the cleaned DataFrame (or None on any failure).

    **Weaning-task filtering**: Only patients who were ever on invasive
    mechanical ventilation are kept. Hours are truncated to 336 (14 days)
    since first intubation event.

    Accepts a tuple so it can be called via Pool.imap with a single iterable.
    """
    csv_path, max_gap_hours = args
    try:
        df = pd.read_csv(csv_path)

        # subject_id from filename
        df["subject_id"] = int(Path(csv_path).stem)

        # column-name normalisation (spaces → underscores)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]

        # WEANING FILTER: exclude patients never on invasive ventilation
        if "invasive" not in df.columns or df["invasive"].max() == 0:
            return None  # patient never intubated

        # re-align 'hour' so that 0 is the start of intubation
        df = _ensure_hourly_index(df)
        vent_start_hour = df[df["invasive"] == 1]["hour"].min()
        df["hours_since_vent"] = df["hour"] - vent_start_hour
        df["hours_since_vent"] = df["hours_since_vent"].clip(lower=0)
        
        # calculate actual event time RELATIVE to ventilation start
        actual_event_time_hours = (df["los"].iloc[0] * 24.0) - vent_start_hour

        # keep full fixed window [0, 336] for ALL patients
        df = df[(df["hour"] >= 0) & (df["hour"] <= 336)].copy()

        # reindex to ensure all hours exist (prevents variable-length leakage)
        full_hours = pd.DataFrame({"hour": np.arange(0, 337)})
        df = full_hours.merge(df, on="hour", how="left")
        
        # administrative censoring ONLY (no LOS lookahead)
        MAX_HOURS = 336.0

        # event_occurred = float(df["death_outcome"].max())

        # if event_occurred == 1.0:
        #     event_hour = actual_event_time_hours
        #     if 0 <= event_hour <= MAX_HOURS:
        #         survival_time = event_hour
        #         survival_event = 1.0
        #     else:
        #         survival_time = MAX_HOURS
        #         survival_event = 0.0
        # else:
        #     survival_time = MAX_HOURS
        #     survival_event = 0.0

        # df["survival_time_hours"] = survival_time
        # df["survival_event"] = survival_event
        # df["death_outcome"] = survival_event

        # Detect successful extubation:
        # 1. Find first hour where invasive goes from 1 → 0
        # 2. Verify it stays 0 for at least 48 hours (successful wean)
        # 3. If patient dies or gets reintubated within 48h, it's a failed wean (censored)

        invasive_series = df["invasive"].fillna(0)  # treat missing as not ventilated

        # find all transitions from ventilated (1) to not ventilated (0)
        was_vent = invasive_series.shift(1, fill_value=0) == 1
        now_not_vent = invasive_series == 0
        extubation_candidates = was_vent & now_not_vent

        if extubation_candidates.any():
            first_extubation_hour = df.loc[extubation_candidates, "hour"].min()
            
            # check if extubation was successful (stayed off vent for 48h)
            hours_after_extubation = df[df["hour"] >= first_extubation_hour]
            if len(hours_after_extubation) >= 48:
                # check if patient stayed off vent for 48h
                next_48h = hours_after_extubation.head(48)
                reintubation = (next_48h["invasive"] == 1).any()
                
                # check if patient died within 48h
                if "death_outcome" in df.columns:
                    died_within_48h = (next_48h["death_outcome"] > 0).any()
                else:
                    died_within_48h = False
                
                if not reintubation and not died_within_48h:
                    # success
                    event_occurred = 1.0
                    event_hour = first_extubation_hour
                else:
                    # fail
                    event_occurred = 0.0
                    event_hour = MAX_HOURS
            else:
                # not enough follow-up to confirm success
                event_occurred = 0.0
                event_hour = MAX_HOURS
        else:
            # never extubated
            event_occurred = 0.0
            event_hour = MAX_HOURS

        # set survival outcome
        if event_occurred == 1.0 and 0 <= event_hour <= MAX_HOURS:
            survival_time = event_hour
            survival_event = 1.0
        else:
            survival_time = MAX_HOURS
            survival_event = 0.0

        df["survival_time_hours"] = survival_time
        df["survival_event"] = survival_event
        
        # create a boolean mask: 1.0 if the value is present and valid, 0.0 otherwise
        df = _apply_validity_bounds(df)

        for col in TIME_VARYING_NUMERIC:
            if col in df.columns:
                # 1.0 = Observed, 0.0 = Missing
                df[f"{col}__obs"] = df[col].notna().astype(np.float32)

        # forward-fill with max-gap cap
        df = _forward_fill_time_varying(df, max_gap_hours)

        # numerical/type stability check
        # 1. time-varying numerics (ensure float32 and no NaNs)
        for col in TIME_VARYING_NUMERIC:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float32)

        # 2. static catgories
        for col in STATIC_COLS:
            if col in df.columns:
                # if a numeric static (like anchor_age), make it float
                if col in ["anchor_age", "pbw_kg", "height_inch", "elixhauser_vanwalraven"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float32)
                else:
                    # if a string, force to string and fill NaNs
                    df[col] = df[col].astype(str).replace(['nan', 'None', 'NaN'], 'Unknown')
        
        # 3. outcomes/treatments
        remaining_cols = TREATMENT_COLS + OUTCOME_COLS
        for col in remaining_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float32)

        return df
    except Exception as exc:
        # logger is not safe across processes; print instead
        print(f"[WARN] Failed to process {csv_path}: {exc}")
        return None


def _discover_csv_paths(dataset_root: str) -> list[str]:
    """Walk the nested 3-digit-folder structure and collect all .csv paths."""
    root = Path(dataset_root)
    paths: list[str] = []
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and folder.name.isdigit():
            for f in sorted(folder.glob("*.csv")):
                paths.append(str(f))
    logger.info("Discovered %d CSV files under %s", len(paths), dataset_root)
    return paths


def _apply_validity_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip or NaN-out values outside clinical validity windows."""
    for col, (lo, hi) in VALIDITY_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def _forward_fill_time_varying(df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
    """
    Forward-fill sparse time-varying measurements with a maximum carry-forward
    window.  Labs (ABG) are typically drawn every 4-6 h; vitals every 1 h.
    We cap carry-forward at `max_gap_hours` to avoid propagating stale labs.
    """
    for col in TIME_VARYING_NUMERIC:
        if col not in df.columns:
            continue
        series = df[col]
        # forward-fill then limit consecutive NaN fills to max_gap_hours
        filled = series.ffill()
        was_nan = series.isna()
        # group consecutive NaN runs; cumsum within each run = run length
        groups = (~was_nan).cumsum()
        run_lengths = was_nan.groupby(groups).cumsum()
        # wherever run length > max_gap_hours, revert fill to NaN
        filled[run_lengths > max_gap_hours] = np.nan
        df[col] = filled
    return df


def _ensure_hourly_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the CSV has an explicit 'hour' or row-index column use it;
    otherwise assume rows are already in hourly order starting at 0.
    """
    # try common hour-column names
    for candidate in ("hr", "hour", "hours", "t", "time_step"):
        if candidate in df.columns:
            df = df.rename(columns={candidate: "hour"})
            break
    else:
        # no explicit column, assign sequential hours
        df["hour"] = range(len(df))

    df = df.sort_values("hour").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Load & preprocess temporal respiratory dataset")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory containing the nested 3-digit folders of per-patient CSVs")
    parser.add_argument("--output_dir",   type=str, default="./processed",
                        help="Directory to write the consolidated Parquet file")
    parser.add_argument("--num_workers",  type=int, default=8,
                        help="Number of parallel workers for CSV loading + preprocessing")
    parser.add_argument("--max_gap_hours", type=int, default=6,
                        help="Maximum hours to forward-fill a missing measurement")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # 1. discover CSVs
    csv_paths = _discover_csv_paths(args.dataset_root)

    # biuld argument tuples the worker expects: (path, max_gap_hours)
    worker_args = [(p, args.max_gap_hours) for p in csv_paths]

    # 2. stream-process patients and write directly to Parquet
    # each worker returns one fully-preprocessed patient DF
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path   = os.path.join(args.output_dir, "patients_preprocessed.parquet")

    # 2a. pre-sample to discover unified schema
    # sample first 100 patients to build a unified schema, then cast all
    # subsequent patients to match it exactly
    logger.info("Pre-sampling 100 patients to infer unified schema …")
    sample_dfs = []
    with Pool(processes=args.num_workers) as pool:
        for i, patient_df in enumerate(pool.imap_unordered(
            _load_and_preprocess_patient, worker_args[:100]
        )):
            if patient_df is not None:
                sample_dfs.append(patient_df)
            if len(sample_dfs) >= 50:  # 50 successful samples is enough
                break

    if not sample_dfs:
        logger.error("Failed to process any sample patients. Aborting.")
        return

    # concat samples to get a representative schema
    sample_concat = pd.concat(sample_dfs, ignore_index=True)
    unified_schema = pa.Schema.from_pandas(sample_concat, preserve_index=False)
    del sample_dfs, sample_concat

    logger.info("  Unified schema has %d columns", len(unified_schema))

    # 2b. stream all patients with schema casting
    writer     = pq.ParquetWriter(out_path, unified_schema)
    n_written  = 0
    n_failed   = 0

    logger.info("Streaming %d patients through %d workers …", len(csv_paths), args.num_workers)

    with Pool(processes=args.num_workers) as pool:
        for patient_df in pool.imap_unordered(_load_and_preprocess_patient, worker_args):
            if patient_df is None:
                n_failed += 1
                continue

            # cast to same schema to handle dtype drift (int vs float, etc.)
            try:
                table = pa.Table.from_pandas(patient_df, schema=unified_schema, preserve_index=False)
                writer.write_table(table)
                n_written += 1

                if n_written % 5000 == 0:
                    logger.info("  … wrote %d / %d patients  (%.1f min elapsed)",
                                n_written, len(csv_paths), (time.time() - t0) / 60)
            except Exception as exc:
                logger.warning("Failed to write patient (schema mismatch?): %s", exc)
                n_failed += 1

    writer.close()

    logger.info("Finished writing.  %d written, %d failed.", n_written, n_failed)

    # 3. summart stats (stream over the finished file — no full load)
    # read back column-by-column with pyarrow to compute missingness without
    # pulling the whole table into a pandas DataFrame.
    logger.info("─── Preprocessing Summary ───")
    pf = pq.ParquetFile(out_path)
    n_rows = pf.metadata.num_rows

    logger.info("  Total patients        : %d", n_written)
    logger.info("  Total hourly rows     : %d", n_rows)
    logger.info("  Parquet file size     : %.1f MB", os.path.getsize(out_path) / 1e6)

    # read only the time-varying columns, one row-group at a time, and
    # accumulate null counts.
    cols_to_check = [c for c in TIME_VARYING_NUMERIC if c in pf.schema.names]
    if cols_to_check:
        null_counts = {c: 0 for c in cols_to_check}
        total_rows  = 0

        for rg_idx in range(pf.metadata.num_row_groups):
            chunk = pf.read_row_group(rg_idx, columns=cols_to_check).to_pandas()
            total_rows += len(chunk)
            for c in cols_to_check:
                null_counts[c] += int(chunk[c].isna().sum())
            del chunk

        miss_rate = pd.Series({c: null_counts[c] / max(total_rows, 1)
                               for c in cols_to_check}).sort_values(ascending=False)
        logger.info("\n  Missingness after forward-fill (top-10 sparsest):")
        for col_name, rate in miss_rate.head(10).items():
            logger.info("    %-30s %.2f%%", col_name, rate * 100)

    logger.info("\n  Wrote %s", out_path)
    logger.info("  Total wall time: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()