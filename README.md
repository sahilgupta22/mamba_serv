# Mamba-Surv: Selective-State-Space Survival Model for Sparse ICU Time-Series

End-to-end pipeline for training and evaluating Mamba-Surv on the
*Temporal Dataset for Respiratory Support in Critically Ill Patients*
(PhysioNet / MIMIC-IV derivative, 50,920 patients).

---

## Repository layout

```
mamba_surv/
├── 01_load_and_preprocess.py   ─  CSV to validated Parquet
├── 02_feature_engineering.py   ─  Delta-T, derived features, splits, scaling
├── 03_model_definitions.py     ─  MambaSurv + 3 baselines + Dataset + loss
├── 04_train.py                 ─  Training loop (shared by all models)
├── 05_evaluate_and_plot.py     ─  Test metrics + 5 figures
├── 06_ablation_studies.py      ─  Ablations + ablation figure
└── README.md                   ─  this file
```

---

## 1. Environment & dependencies

```bash
# Python 3.9+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn scipy matplotlib seaborn pyarrow
```

| Package        | Minimum version | Role                          |
|----------------|-----------------|-------------------------------|
| torch          | 2.1             | Model, CUDA, AMP              |
| pandas         | 2.0             | Parquet I/O, data wrangling   |
| numpy          | 1.24            | Array ops                     |
| scikit-learn   | 1.3             | train_test_split              |
| matplotlib     | 3.7             | All figures                   |
| pyarrow        | 12              | Parquet engine                |

No external survival-analysis library is required; Cox loss and C-index
are implemented from scratch in `03_model_definitions.py`.

---

## 2. Data access

The dataset requires **credentialed access** via PhysioNet.

1. Create a PhysioNet account and sign the Data Use Agreement.
2. Download the dataset and unpack it so the directory tree matches:

```
/path/to/dataset/
    100/
        10000032.csv
        ...
    101/
        ...
    ...
```

Set `DATASET_ROOT=/path/to/dataset` in the commands below.

---

## 3. Execution — step by step

All commands assume a single GPU workstation.  Total wall-clock for the
full pipeline: **~1.5–2.5 days** on an A100; ~2–3 days on a 3090.

### 3.1  Preprocessing  (~20-40 min, CPU)

```bash
python 01_load_and_preprocess.py \
    --dataset_root  $DATASET_ROOT \
    --output_dir    ./processed \
    --num_workers   8
```

**Outputs:**  `processed/patients_preprocessed.parquet`

### 3.2  Feature engineering  (~10-20 min, CPU)

```bash
python 02_feature_engineering.py \
    --input_parquet ./processed/patients_preprocessed.parquet \
    --output_dir    ./processed \
    --seed          42
```

**Outputs:**
- `processed/train.parquet`  (70 %)
- `processed/val.parquet`    (15 %)
- `processed/test.parquet`   (15 %)
- `processed/scaler_stats.parquet`
- `processed/feature_manifest.json`

### 3.3  Model training

Run the four models sequentially (or in parallel on multiple GPUs/nodes):

```bash
# Proposed model
python 04_train.py --model mamba_surv      --processed_dir ./processed --results_dir ./results

# Baselines
python 04_train.py --model deepsurv_lstm   --processed_dir ./processed --results_dir ./results
python 04_train.py --model transformer_cox --processed_dir ./processed --results_dir ./results
python 04_train.py --model cox_linear      --processed_dir ./processed --results_dir ./results
```

**Outputs:**
- `results/checkpoints/<model>_best.pt`
- `results/logs/<model>.jsonl`

### 3.4  Test evaluation & figures

```bash
python 05_evaluate_and_plot.py \
    --processed_dir ./processed \
    --results_dir   ./results
```

**Outputs:**
- `results/test_metrics.json`
- `results/figures/fig1_cindex_comparison.png`
- `results/figures/fig2_km_risk_quartiles.png`
- `results/figures/fig3_calibration.png`
- `results/figures/fig4_training_curves.png`
- `results/figures/fig5_feature_importance.png`

### 3.5  Ablation studies

```bash
python 06_ablation_studies.py \
    --processed_dir ./processed \
    --results_dir   ./results \
    --epochs        60
```

**Outputs:**
- `results/ablation_results.json`
- `results/figures/fig_ablation.png`

---

## 4. Architecture summary

### Why Mamba for sparse EHR?

| Property               | RNN / LSTM         | Transformer         | **Mamba-Surv**             |
|------------------------|--------------------|---------------------|----------------------------|
| Complexity per step    | O(1)               | O(L)                | **O(1)**                   |
| Total sequence cost    | O(L)               | O(L²)              | **O(L)**                   |
| Handles 2160-step seq  | ✓ (vanishing grad) | ✗ (OOM risk)        | **✓ (stable)**             |
| Input-dependent decay  | ✗                  | ✗                   | **✓ (Δt-gated A matrix)**  |

### Delta-T trick

Each feature group (labs, vitals, ventilation) gets a companion
`<feature>__delta_t` column recording *hours since last observation*.
This is fed alongside the feature value into the Mamba gate:

```
A_t  =  exp(A_base) × σ( W_a · [x_t ; Δt_t] )
```

When Δt is large (stale lab), σ → 1 and A ≈ I, so the hidden state is
carried forward with minimal decay.  When fresh data arrives (Δt ≈ 0),
the gate opens and the state updates normally.

### Cox head

The final layer outputs a single scalar *log-hazard ratio* per patient,
trained with the Breslin approximation of the Cox partial log-likelihood.
Evaluation uses Harrell's C-index.

---

## 5. Reproducibility notes

- All random seeds default to **42** and are set in NumPy, PyTorch, and
  scikit-learn's `train_test_split`.
- The scaler (mean / std) is fit exclusively on the training split and
  persisted in `scaler_stats.parquet`.
- Best checkpoints are selected on **validation C-index** with patience 10.
- The test set is never touched until `05_evaluate_and_plot.py`.

---

## 6. Citation
