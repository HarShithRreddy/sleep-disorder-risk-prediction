# models_registry.py
# Central loader + predictor for all AgeWise models.
#
# Each model is trained in its own separate script:
#   insomnia_rf_young_adult.py   -> insomnia_rf_young_adult.pkl
#   insomnia_rf_middle_age.py    -> insomnia_rf_middle_age.pkl
#   insomnia_rf_older_adult.py   -> insomnia_rf_older_adult.pkl
#   apnea_rf_young_adult.py      -> apnea_rf_young_adult.pkl
#   apnea_rf_middle_aged.py      -> apnea_rf_middle_aged.pkl
#   apnea_rf_older_adult.py      -> apnea_rf_older_adult.pkl
#   insomnia_xgb_young_adult.py  -> insomnia_xgb_young_adult.pkl  (done)
#   insomnia_xgb_middle_age.py   -> insomnia_xgb_middle_age.pkl   (done)
#   insomnia_xgb_older_adult.py  -> insomnia_xgb_older_adult.pkl  (done)
#   apnea_xgb_young_adult.py     -> apnea_xgb_young_adult.pkl     (done)
#   apnea_xgb_middle_aged.py     -> apnea_xgb_middle_aged.pkl     (done)
#   apnea_xgb_older_adult.py     -> apnea_xgb_older_adult.pkl     (done)
#
# Each .pkl contains: {'rf': model, 'scaler': scaler}  (RF scripts)
#                 or  {'xgb': model, 'scaler': scaler} (XGBoost scripts)

import pickle
import pandas as pd

# --------------------------------------------------
# Model file paths
# None = not yet trained (placeholder)
# --------------------------------------------------
MODEL_PATHS = {
    "Random Forest": {
        ("insomnia", "young_adult"): "insomnia_rf_young_adult.pkl",
        ("insomnia", "middle_aged"): "insomnia_rf_middle_age.pkl",
        ("insomnia", "older_adult"): "insomnia_rf_older_adult.pkl",
        ("apnea",    "young_adult"): "apnea_rf_young_adult.pkl",
        ("apnea",    "middle_aged"): "apnea_rf_middle_aged.pkl",
        ("apnea",    "older_adult"): "apnea_rf_older_adult.pkl",
    },
    "XGBoost": {
        ("insomnia", "young_adult"): "insomnia_xgb_young_adult.pkl",
        ("insomnia", "middle_aged"): "insomnia_xgb_middle_age.pkl",
        ("insomnia", "older_adult"): "insomnia_xgb_older_adult.pkl",
        ("apnea",    "young_adult"): "apnea_xgb_young_adult.pkl",
        ("apnea",    "middle_aged"): "apnea_xgb_middle_aged.pkl",
        ("apnea",    "older_adult"): "apnea_xgb_older_adult.pkl",
    },
}

# --------------------------------------------------
# Feature lists — copied exactly from training scripts.
# Order must match the order the scaler was fitted on.
# --------------------------------------------------
INSOMNIA_FEATURES = [
    "age", "gender", "duration_minutes", "sleep_latency_minutes",
    "wake_after_sleep_onset_minutes", "sleep_efficiency_pct",
    "sleep_stage_deep_pct", "sleep_stage_light_pct",
    "sleep_stage_rem_pct", "sleep_stage_awake_pct",
    "heart_rate_mean_bpm", "heart_rate_min_bpm", "heart_rate_max_bpm",
    "hrv_rmssd_ms", "stress_score", "screen_time_before_bed_min",
    "activity_before_bed_min", "bedtime_consistency_std_min",
    "caffeine_mg", "alcohol_units",
]

APNEA_FEATURES = [
    "age", "gender", "weight_kg", "height_cm", "duration_minutes",
    "sleep_efficiency_pct", "sleep_stage_deep_pct", "sleep_stage_light_pct",
    "sleep_stage_rem_pct", "sleep_stage_awake_pct",
    "wake_after_sleep_onset_minutes", "heart_rate_mean_bpm",
    "heart_rate_min_bpm", "heart_rate_max_bpm", "hrv_rmssd_ms",
    "respiration_rate_bpm", "spo2_mean_pct", "spo2_min_pct",
    "snore_events", "movement_count", "stress_score", "alcohol_units",
]

FEATURE_SETS = {
    "insomnia": INSOMNIA_FEATURES,
    "apnea":    APNEA_FEATURES,
}

# --------------------------------------------------
# Gender encoding (alphabetical: Female=0, Male=1)
# --------------------------------------------------
GENDER_ENCODING = {
    "Female": 0,
    "Male":   1,
}

RISK_LABELS = {
    "insomnia": {0: "No Insomnia Risk", 1: "Insomnia Risk"},
    "apnea":    {0: "No Apnea Risk",    1: "Apnea Risk"},
}

# --------------------------------------------------
# Known accuracy metrics for the comparison dashboard.
# Update these as each new model is trained.
# None = model not yet trained.
#
# All values are percentages (0-100).
# --------------------------------------------------
METRICS = {
    "Random Forest": {
        "insomnia": {
            "young_adult": {"accuracy": 93.43, "roc_auc": 95.83, "f1": 92.10, "recall": 91.50},
            "middle_aged": {"accuracy": 91.85, "roc_auc": 91.94, "f1": 90.20, "recall": 89.80},
            "older_adult": {"accuracy": 92.60, "roc_auc": 93.10, "f1": 91.40, "recall": 90.90},
        },
        "apnea": {
            # weighted-avg f1/recall pulled from each run's classification_report
            "young_adult": {"accuracy": 97.64, "roc_auc": 99.70, "f1": 98.00, "recall": 98.00},
            "middle_aged": {"accuracy": 96.34, "roc_auc": 99.48, "f1": 96.00, "recall": 96.00},
            "older_adult": {"accuracy": 96.98, "roc_auc": 99.62, "f1": 97.00, "recall": 97.00},
        },
    },
    "XGBoost": {
        "insomnia": {
            # weighted-avg f1/recall pulled from each run's classification_report
            "young_adult": {"accuracy": 94.31, "roc_auc": 95.08, "f1": 94.00, "recall": 94.00},
            "middle_aged": {"accuracy": 92.75, "roc_auc": 91.39, "f1": 93.00, "recall": 93.00},
            "older_adult": {"accuracy": 92.43, "roc_auc": 93.71, "f1": 93.00, "recall": 92.00},
        },
        "apnea": {
            "young_adult": {"accuracy": 99.30, "roc_auc": 99.94, "f1": 99.00, "recall": 99.00},
            "middle_aged": {"accuracy": 99.10, "roc_auc": 99.98, "f1": 99.00, "recall": 99.00},
            "older_adult": {"accuracy": 99.41, "roc_auc": 99.99, "f1": 99.00, "recall": 99.00},
        },
    },
}

# --------------------------------------------------
# Lazy-loaded model cache (avoids re-reading .pkl on every call)
# --------------------------------------------------
_cache = {}


def _load_bundle(path, label):
    """Loads and caches a .pkl bundle from disk."""
    if path not in _cache:
        try:
            with open(path, "rb") as f:
                _cache[path] = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{label} model file not found: '{path}'\n"
                "Run the corresponding training script first to generate it."
            )
    return _cache[path]


def get_model(model_type, disorder, cohort):
    """
    Loads the model bundle for a given model type, disorder, and cohort.
    Raises NotImplementedError for placeholder (None) paths.
    """
    key = (disorder, cohort)
    paths = MODEL_PATHS.get(model_type, {})

    if key not in paths:
        raise ValueError(f"Unknown key {model_type}/{disorder}/{cohort}")

    path = paths[key]
    if path is None:
        raise NotImplementedError(
            f"{model_type} model for {disorder}/{cohort} is not trained yet."
        )

    return _load_bundle(path, f"{model_type} {disorder}/{cohort}")


def _build_input_row(disorder, input_dict):
    """Builds a single-row DataFrame in the exact feature order the model expects."""
    features = FEATURE_SETS[disorder]
    missing  = [f for f in features if f not in input_dict]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")

    row = {}
    for feat in features:
        val = input_dict[feat]
        if feat == "gender" and isinstance(val, str):
            if val not in GENDER_ENCODING:
                raise ValueError(f"Unknown gender value '{val}'.")
            val = GENDER_ENCODING[val]
        row[feat] = val

    return pd.DataFrame([row])[features]


def predict(disorder, cohort, input_dict, model_type="Random Forest"):
    """
    Run a prediction for one patient.

    Parameters
    ----------
    disorder   : "insomnia" or "apnea"
    cohort     : "young_adult", "middle_aged", or "older_adult"
    input_dict : dict with all required feature values
    model_type : "Random Forest" or "XGBoost"

    Returns
    -------
    dict with keys: risk, risk_label, probability, model_type
    """
    bundle = get_model(model_type, disorder, cohort)

    # Support both RF bundles {'rf': ...} and XGBoost bundles {'xgb': ...}
    model  = bundle.get("rf") or bundle.get("xgb") or bundle.get("model")
    scaler = bundle.get("scaler")

    X       = _build_input_row(disorder, input_dict)
    X_input = scaler.transform(X) if scaler is not None else X.values

    pred = int(model.predict(X_input)[0])
    prob = float(model.predict_proba(X_input)[0][1])

    return {
        "risk":        pred,
        "risk_label":  RISK_LABELS[disorder][pred],
        "probability": prob,
        "model_type":  model_type,
    }


def get_feature_list(disorder):
    """Returns the ordered feature list for a given disorder."""
    return FEATURE_SETS[disorder]


def get_metrics(model_type, disorder, cohort):
    """
    Returns the known metrics dict for a model, or None if not yet trained.
    Keys: accuracy, roc_auc, f1, recall (all as percentages 0-100)
    """
    return METRICS.get(model_type, {}).get(disorder, {}).get(cohort)


def is_model_available(model_type, disorder, cohort):
    """Returns True if the model .pkl exists (not a placeholder)."""
    return MODEL_PATHS.get(model_type, {}).get((disorder, cohort)) is not None