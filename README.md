# An Explainable Age-Stratified Machine Learning Framework for Sleep Disorder Risk Prediction and Personalized Health Recommendation

## Overview

This project is a research framework that predicts **insomnia** and **sleep apnea** risk from smartwatch sensor data, using **age-stratified specialist models** instead of a single one-size-fits-all classifier. The framework trains 12 separate models — one for every combination of:

- **3 age cohorts**: Young Adult (18–35), Middle-Aged (36–55), Older Adult (56–79)
- **2 disorders**: Insomnia, Sleep Apnea
- **2 algorithms**: Random Forest, XGBoost

Each prediction is explained using **SHAP (SHapley Additive exPlanations)**, and paired with a **personalized health recommendation engine** that separates modifiable risk factors (e.g. stress, caffeine, alcohol) from non-modifiable ones (age, gender, height). Everything is served through an interactive **Streamlit dashboard**.

The project is built as a direct response to identified gaps in an existing base paper (ASHA — Ratnakar & Shetty, IEEE AIDE 2026), most notably: absence of true age-stratification, no explainability, and label leakage in the original modeling pipeline.

---

## Authors & Project Info

| Role | Name | SRN |
|---|---|---|
| Author | Harshith R Reddy | PES2UG24CS188 |
| Author | Aaryesh Karamsetty | PES2UG24CS215 |
| Guide | Ruby Dinakar J | — |
| Centre Head | Dr. Sandesh B J | — |

**Institution:** CoDMAV Lab, PES University, EC Campus, Bengaluru
**Type:** Summer Internship Research Project

---

## Research Motivation — Gaps Addressed vs. Base Paper (ASHA)

The base paper's baseline pipeline had 6 identified issues that this project explicitly fixes:

1. **No true age-stratification** — ASHA does not split modeling by age; this framework trains fully independent specialist models per cohort.
2. **No explainability** — ASHA gives no feature-level reasoning; this framework integrates SHAP (TreeExplainer) for every prediction.
3. **Label leakage** — the raw dataset's `sleep_score`, `insomnia_flag`, and `apnea_risk_score` directly encode the targets. These are dropped from the feature set entirely.
4. **Broken target label** — the raw `daily_label` column is unusable (`fair=18666, poor=1330, good=4`). It is discarded and rebuilt from `sleep_score` using clinical thresholds.
5. **Single multi-output model over specialist models** — literature (Ha et al., JMIR 2023; Wei et al., Oxford/medRxiv 2025) shows single-task models achieve higher AUROC (0.92) than multi-task equivalents, and that SHAP drivers differ meaningfully by disorder — justifying 6 independent specialist models over 1 shared model.
6. **No personalized, explainable recommendations** — the framework's recommendation engine is SHAP-driven and cohort-aware.

---

## Dataset

**Source:** Kaggle Smartwatch Sleep Tracking Dataset
**File:** `smartwatch_sleep_dataset.csv`
**Size:** 20,000 sessions across 2,000 users, 45 raw features (sleep architecture, heart rate/HRV, respiration, SpO2, movement/snoring, lifestyle factors, demographics)

---

## Repository / File Structure

```
project/
├── smartwatch_sleep_dataset.csv        # Raw Kaggle dataset (20,000 rows, 45 cols)
├── preprocess.py                # STEP 1 — preprocessing pipeline
├── agewise_cleaned_full.csv            # Output of preprocessing (used by all training scripts)
│
├── insomnia_rf_young_adult.py          # RF — Insomnia — Young Adult (18–35)
├── insomnia_rf_middle_age.py           # RF — Insomnia — Middle-Aged (36–55)
├── insomnia_rf_older_adult.py          # RF — Insomnia — Older Adult (55/56–79)
├── apnea_rf_young_adult.py             # RF — Apnea    — Young Adult (18–35)
├── apnea_rf_middle_aged.py             # RF — Apnea    — Middle-Aged (36–55)
├── apnea_rf_older_adult.py             # RF — Apnea    — Older Adult (56–79)
│
├── insomnia_xgb_young_adult.py         # XGBoost — Insomnia — Young Adult
├── insomnia_xgb_middle_age.py          # XGBoost — Insomnia — Middle-Aged
├── insomnia_xgb_older_adult.py         # XGBoost — Insomnia — Older Adult
├── apnea_xgb_young_adult.py            # XGBoost — Apnea    — Young Adult
├── apnea_xgb_middle_aged.py            # XGBoost — Apnea    — Middle-Aged
├── apnea_xgb_older_adult.py            # XGBoost — Apnea    — Older Adult
│
├── models_registry.py                  # Central lazy-loading registry for all 12 .pkl models
├── shap_explainer.py                   # SHAP TreeExplainer wrapper per (model_type, disorder, cohort)
├── recommendation_engine.py            # SHAP-driven personalized recommendation generator
├── dashboard.py                        # Streamlit app tying everything together
│
├── plots/                              # Preprocessing EDA plots (7 plots)
├── plots_young_adult/ (or plots/)      # Per-cohort training plots (feature importance, confusion matrix, ROC)
├── plots_middle_age/
├── plots_older_adult/
│
└── *.pkl                               # Serialized trained models (RF + XGB) + scalers, one set per cohort/disorder
```

> Note: exact `.pkl` / plot folder names depend on how each training script was last run — see the "Model Registry" section below for the authoritative mapping used by the dashboard.

---

## Requirements / Environment

- **Python:** 3.9+ (developed on 3.9.6 via Xcode Command Line Tools / Homebrew on macOS)
- **Package installer:** `pip3` (use `--break-system-packages` flag on macOS system Python if needed)

### Install dependencies

```bash
pip3 install pandas numpy scikit-learn xgboost shap imbalanced-learn matplotlib seaborn streamlit --break-system-packages
```

Core libraries used across the project:

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data handling |
| `scikit-learn` | Random Forest, preprocessing, metrics, train/test split |
| `xgboost` | XGBoost specialist models |
| `imbalanced-learn` (SMOTE) | Class imbalance handling |
| `shap` | Model explainability (TreeExplainer) |
| `matplotlib`, `seaborn` | Plots (feature importance, confusion matrix, ROC curves) |
| `streamlit` | Interactive dashboard |
| `pickle` | Model persistence (`.pkl` files) |

---

## How to Run — Full Pipeline (in order)

### Step 1 — Preprocess the raw dataset

```bash
python3 preprocess.py
```

This reads `smartwatch_sleep_dataset.csv` and performs:
- Missing value analysis (median/mode imputation <5%, drop >10%)
- Duplicate removal
- Target label rebuild: `sleep_quality` from `sleep_score` (≥65 good, 50–64 fair, <50 poor) — replaces the broken `daily_label`
- Drops identifiers (`user_id`, timestamps, `created_at`), administrative columns (`device_model`, `timezone`), and label leakers (`sleep_score`, `insomnia_flag`, `apnea_risk_score`, `daily_label`)
- Creates `age_cohort` (Young_Adult / Middle_Aged / Older_Adult)
- IQR-based outlier winsorization
- Feature engineering (deep/REM ratio, restorative sleep %, fragmentation index, recovery index, HR range, stimulant load, etc.)
- Categorical encoding
- Baseline SMOTE + 7 EDA plots saved to `plots/`

**Outputs:** `agewise_cleaned_full.csv` (used by every training script below), plus `agewise_X_smote.csv`, `agewise_y_smote.csv`, `agewise_X_raw.csv`, `agewise_y_raw.csv`, `agewise_cohort.csv`.

### Step 2 — Train the 6 Random Forest models

Run each script independently (they are self-contained; no other run order required among themselves, but each depends on `agewise_cleaned_full.csv` and, for apnea, `smartwatch_sleep_dataset.csv`):

```bash
python3 insomnia_rf_young_adult.py
python3 insomnia_rf_middle_age.py
python3 insomnia_rf_older_adult.py
python3 apnea_rf_young_adult.py
python3 apnea_rf_middle_aged.py
python3 apnea_rf_older_adult.py
```

Each script:
1. Loads `agewise_cleaned_full.csv`
2. Filters to its age cohort
3. Builds the binary target (insomnia: `sleep_quality == 'poor'`; apnea: `apnea_risk_score >= 15`, recovered from the raw CSV via row-aligned merge and used only to build the label, never as a feature)
4. 80/20 stratified train/test split (`random_state=42`)
5. SMOTE on the training set only
6. StandardScaler (fit on train, applied to both)
7. Trains a `RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, class_weight='balanced', random_state=42)`
8. Evaluates (accuracy, ROC-AUC, 5-fold CV) and saves feature importance / confusion matrix / ROC plots
9. Prints a full summary to console

### Step 3 — Train the 6 XGBoost models

```bash
python3 insomnia_xgb_young_adult.py
python3 insomnia_xgb_middle_age.py
python3 insomnia_xgb_older_adult.py
python3 apnea_xgb_young_adult.py
python3 apnea_xgb_middle_aged.py
python3 apnea_xgb_older_adult.py
```

Same data pipeline as the RF scripts (SMOTE + StandardScaler, same 80/20 stratified split, `random_state=42`), but with:

```python
XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Step 4 — Serialize models (if not already done inside training scripts)

Each training script should pickle its trained model + scaler (and feature list) to a `.pkl` file consumed by `models_registry.py`. Confirm each of the 12 models has a corresponding `.pkl` before launching the dashboard.

### Step 5 — Launch the dashboard

```bash
streamlit run dashboard.py
```

This opens the interactive app in your browser (default `http://localhost:8501`). From the UI you can:
- Select disorder (Insomnia / Apnea), cohort, and model type (RF / XGBoost) via dropdowns
- Enter/adjust feature values and get a live risk prediction
- View a SHAP bar chart explaining the individual prediction
- Compare accuracy across all 12 models
- Get a personalized recommendation broken into risk-increasing modifiable factors, protective factors, and non-modifiable factors, with a clinical disclaimer

---

## Model Registry (`models_registry.py`)

Central module that lazily loads and caches all 12 pickled models to avoid repeated disk reads. Exposes:

```python
predict(model_type, disorder, cohort, input_features)
```

- `model_type`: `"rf"` or `"xgb"`
- `disorder`: `"insomnia"` or `"apnea"`
- `cohort`: `"young_adult"`, `"middle_aged"`, or `"older_adult"`

`get_model(model_type, disorder, cohort)` returns the cached `(model, scaler, feature_list)` tuple. This three-argument signature is used consistently by `shap_explainer.py`, `recommendation_engine.py`, and `dashboard.py`.

---

## SHAP Explainability (`shap_explainer.py`)

Uses `shap.TreeExplainer` (compatible with both RandomForest and XGBoost) to compute per-prediction SHAP values against the correct `(model_type, disorder, cohort)` triple pulled from the registry. Outputs a ranked list of features by SHAP contribution (signed), consumed directly by the dashboard's SHAP bar chart and by the recommendation engine.

---

## Recommendation Engine (`recommendation_engine.py`)

Takes SHAP output for a given prediction and splits contributing features into three buckets:

1. **Risk-increasing, modifiable** (e.g. stress, caffeine, alcohol, screen time before bed) — the person can act on these
2. **Protective factors** — features currently reducing risk, worth reinforcing
3. **Non-modifiable factors** (age, gender, height) — surfaced for context only, no action prompted

Every output includes a **clinical disclaimer**: this is not a diagnostic tool and is not a substitute for professional medical advice.

---

## Key Modeling Decisions & Rationale

- **Six specialist models, not one multi-output model** — supported by Ha et al. (JMIR 2023, different SHAP drivers per disorder) and Wei et al. (Oxford/medRxiv 2025, single-task AUROC 0.92 outperforms multi-task).
- **Apnea models achieve very high accuracy (96–99%)** because `apnea_risk_score` correlates ~0.95–0.98 with BMI in this dataset — this is a **dataset-level property**, not label leakage, and is explicitly framed as a limitation (see Ha et al., JMIR 2023, PMC10557018).
- **Insomnia models show genuine age-stratified signal**: stress and alcohol dominate SHAP importance for Young Adults; WASO (wake-after-sleep-onset) and sleep architecture metrics rise in importance with age — empirically supporting the stratification premise.
- **Apnea threshold** (`apnea_risk_score >= 15`) mirrors the clinical AASM AHI cutoff for moderate obstructive sleep apnea.
- **Label leakage prevention**: `sleep_score`, `insomnia_flag`, `apnea_risk_score` are removed from all model features. `apnea_risk_score` is only recovered from the raw CSV (via a row-alignment assertion against the cleaned CSV) to construct the apnea target — never reintroduced as a predictor.
- All Random Forest models share identical hyperparameters across cohorts, and all XGBoost models share identical hyperparameters across cohorts, so that performance differences reflect the underlying data/age-cohort signal rather than model tuning differences.

---

## Key References

- Ratnakar & Shetty — **ASHA**, IEEE AIDE 2026 (base paper)
- Ha et al., **JMIR 2023** (PMC10557018) — OSA driven by BMI/weight/sex; disorder-specific SHAP drivers
- Wei et al., Oxford/medRxiv 2025 — single-task vs. multi-task AUROC comparison
- Kapur et al. — sleep apnea clinical grounding
- AASM (American Academy of Sleep Medicine) — AHI severity thresholds

---

## Reproducibility Notes

- All train/test splits use `random_state=42`, `test_size=0.2`, `stratify=y`
- SMOTE is applied to the training set only, per cohort, per disorder — test sets are left at their real-world (imbalanced) distribution
- `StandardScaler` is fit on the (post-SMOTE) training set and applied unchanged to the test set
- If Random Forest / XGBoost logic, hyperparameters, feature lists, or thresholds are ever modified, this should be done as a structural/mechanical refactor only — the underlying model logic and thresholds should be preserved exactly to keep results reproducible against the plots and metrics already generated.

---

## Troubleshooting

- **`pip: command not found` on macOS** → use `pip3` instead.
- **Permission errors installing packages** → add `--break-system-packages` to the `pip3 install` command.
- **Stale results after fixing a bug** → verify which physical file Python is actually importing (e.g. via `inspect.getsourcefile(module)`), especially if multiple copies of a script exist in different directories.
- **`shap_explainer.py` errors about `get_model()` arguments** → confirm you're calling it with the three-argument signature `(model_type, disorder, cohort)`, matching `models_registry.py`.
