# shap_explainer.py
# SHAP explainability for the AgeWise Random Forest models.
# Sits on top of models_registry.py — uses the same loaded model/scaler,
# so predictions and explanations always agree with each other.

import shap
import numpy as np

from models_registry import get_model, _build_input_row, FEATURE_SETS

# --------------------------------------------------
# Cache TreeExplainers per (disorder, cohort) — building an explainer
# from scratch is comparatively expensive, so reuse it across calls.
# --------------------------------------------------

_explainer_cache = {}


def _get_explainer(disorder, cohort):
    key = (disorder, cohort)
    if key not in _explainer_cache:
        # SHAP explanations are always generated from the Random Forest
        # model (see dashboard.py) — models_registry.get_model now takes
        # (model_type, disorder, cohort), so we pass "Random Forest" here.
        bundle = get_model("Random Forest", disorder, cohort)
        rf = bundle["rf"]
        _explainer_cache[key] = shap.TreeExplainer(rf)
    return _explainer_cache[key]


def explain_instance(disorder, cohort, input_dict, top_n=None):
    """
    Computes SHAP values for a single person's input.
    """
    bundle = get_model("Random Forest", disorder, cohort)
    scaler = bundle["scaler"]
    features = FEATURE_SETS[disorder]

    X = _build_input_row(disorder, input_dict)
    X_scaled = scaler.transform(X)

    explainer = _get_explainer(disorder, cohort)
    raw_shap = explainer.shap_values(X_scaled)

    if isinstance(raw_shap, list):
        instance_shap = raw_shap[1][0]
        base_value = explainer.expected_value[1]
    else:
        arr = np.array(raw_shap)
        if arr.ndim == 3:
            instance_shap = arr[0, :, 1]
            base_value = explainer.expected_value[1] if hasattr(
                explainer.expected_value, "__len__"
            ) else explainer.expected_value
        else:
            instance_shap = arr[0]
            base_value = explainer.expected_value

    shap_list = [
        {
            "feature": feat,
            "value": input_dict[feat],
            "shap_value": float(val),
        }
        for feat, val in zip(features, instance_shap)
    ]

    shap_list.sort(key=lambda d: abs(d["shap_value"]), reverse=True)

    if top_n is not None:
        shap_list = shap_list[:top_n]

    return {
        "base_value": float(base_value),
        "shap_values": shap_list,
    }


if __name__ == "__main__":
    example_input = {
        "age": 45,
        "gender": "Female",
        "duration_minutes": 410,
        "sleep_latency_minutes": 20,
        "wake_after_sleep_onset_minutes": 30,
        "sleep_efficiency_pct": 85,
        "sleep_stage_deep_pct": 15,
        "sleep_stage_light_pct": 55,
        "sleep_stage_rem_pct": 20,
        "sleep_stage_awake_pct": 10,
        "heart_rate_mean_bpm": 65,
        "heart_rate_min_bpm": 55,
        "heart_rate_max_bpm": 90,
        "hrv_rmssd_ms": 40,
        "stress_score": 35,
        "screen_time_before_bed_min": 30,
        "activity_before_bed_min": 10,
        "bedtime_consistency_std_min": 20,
        "caffeine_mg": 100,
        "alcohol_units": 1,
    }
    result = explain_instance("insomnia", "middle_aged", example_input, top_n=8)
    print("Base value:", result["base_value"])
    for item in result["shap_values"]:
        print(f"  {item['feature']:35s} value={item['value']!s:>8}  shap={item['shap_value']:+.4f}")