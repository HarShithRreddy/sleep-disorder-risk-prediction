# dashboard.py
# AgeWise Streamlit dashboard.
# Run with: streamlit run dashboard.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models_registry import (
    predict, get_feature_list, get_metrics,
    is_model_available, GENDER_ENCODING, METRICS,
)
from shap_explainer import explain_instance
from recommendation_engine import generate_recommendations

st.set_page_config(page_title="AgeWise — Sleep Disorder Risk", layout="wide")

st.title("AgeWise: Age-Stratified Sleep Disorder Risk Dashboard")
st.caption(
    "Explainable Random Forest and XGBoost models for insomnia and sleep apnea "
    "risk, stratified by age cohort."
)

# --------------------------------------------------
# Feature labels
# --------------------------------------------------
FEATURE_LABELS = {
    "age":                             "Age (years)",
    "gender":                          "Gender",
    "duration_minutes":                "Sleep Duration (minutes)",
    "sleep_latency_minutes":           "Sleep Latency (minutes)",
    "wake_after_sleep_onset_minutes":  "Wake After Sleep Onset (minutes)",
    "sleep_efficiency_pct":            "Sleep Efficiency (%)",
    "sleep_stage_deep_pct":            "Deep Sleep (%)",
    "sleep_stage_light_pct":           "Light Sleep (%)",
    "sleep_stage_rem_pct":             "REM Sleep (%)",
    "sleep_stage_awake_pct":           "Awake (%)",
    "heart_rate_mean_bpm":             "Mean Heart Rate (bpm)",
    "heart_rate_min_bpm":              "Min Heart Rate (bpm)",
    "heart_rate_max_bpm":              "Max Heart Rate (bpm)",
    "hrv_rmssd_ms":                    "HRV — RMSSD (ms)",
    "stress_score":                    "Stress Score",
    "screen_time_before_bed_min":      "Screen Time Before Bed (min)",
    "activity_before_bed_min":         "Activity Before Bed (min)",
    "bedtime_consistency_std_min":     "Bedtime Consistency (std, min)",
    "caffeine_mg":                     "Caffeine (mg)",
    "alcohol_units":                   "Alcohol (units)",
    "weight_kg":                       "Weight (kg)",
    "height_cm":                       "Height (cm)",
    "respiration_rate_bpm":            "Respiration Rate (breaths/min)",
    "spo2_mean_pct":                   "Mean SpO2 (%)",
    "spo2_min_pct":                    "Min SpO2 (%)",
    "snore_events":                    "Snore Events (count)",
    "movement_count":                  "Movement Count",
}

FEATURE_DEFAULTS = {
    "age":                            dict(min_value=18,    max_value=79,    value=40),
    "duration_minutes":               dict(min_value=0,     max_value=700,   value=420),
    "sleep_latency_minutes":          dict(min_value=0,     max_value=180,   value=15),
    "wake_after_sleep_onset_minutes": dict(min_value=0,     max_value=300,   value=25),
    "sleep_efficiency_pct":           dict(min_value=0.0,   max_value=100.0, value=85.0),
    "sleep_stage_deep_pct":           dict(min_value=0.0,   max_value=100.0, value=18.0),
    "sleep_stage_light_pct":          dict(min_value=0.0,   max_value=100.0, value=50.0),
    "sleep_stage_rem_pct":            dict(min_value=0.0,   max_value=100.0, value=20.0),
    "sleep_stage_awake_pct":          dict(min_value=0.0,   max_value=100.0, value=10.0),
    "heart_rate_mean_bpm":            dict(min_value=30.0,  max_value=150.0, value=65.0),
    "heart_rate_min_bpm":             dict(min_value=30.0,  max_value=150.0, value=55.0),
    "heart_rate_max_bpm":             dict(min_value=30.0,  max_value=200.0, value=95.0),
    "hrv_rmssd_ms":                   dict(min_value=0.0,   max_value=200.0, value=40.0),
    "stress_score":                   dict(min_value=0,     max_value=100,   value=35),
    "screen_time_before_bed_min":     dict(min_value=0,     max_value=300,   value=30),
    "activity_before_bed_min":        dict(min_value=0,     max_value=180,   value=10),
    "bedtime_consistency_std_min":    dict(min_value=0.0,   max_value=120.0, value=20.0),
    "caffeine_mg":                    dict(min_value=0,     max_value=800,   value=100),
    "alcohol_units":                  dict(min_value=0.0,   max_value=10.0,  value=1.0),
    "weight_kg":                      dict(min_value=30.0,  max_value=200.0, value=70.0),
    "height_cm":                      dict(min_value=120.0, max_value=220.0, value=170.0),
    "respiration_rate_bpm":           dict(min_value=5.0,   max_value=40.0,  value=15.0),
    "spo2_mean_pct":                  dict(min_value=70.0,  max_value=100.0, value=96.0),
    "spo2_min_pct":                   dict(min_value=50.0,  max_value=100.0, value=92.0),
    "snore_events":                   dict(min_value=0,     max_value=500,   value=20),
    "movement_count":                 dict(min_value=0,     max_value=500,   value=40),
}

COHORT_LABELS = {
    "young_adult": "Young Adult (18–35)",
    "middle_aged": "Middle-Aged (36–55)",
    "older_adult": "Older Adult (56–79)",
}

DISORDER_LABELS = {
    "insomnia": "Insomnia",
    "apnea":    "Sleep Apnea",
}

MODEL_COLORS = {
    "Random Forest": "#2563eb",
    "XGBoost":       "#27ae60",
}


def render_input_field(feature):
    """Renders one Streamlit input widget for a given feature name."""
    label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
    if feature == "gender":
        return st.selectbox(label, options=list(GENDER_ENCODING.keys()))
    cfg = FEATURE_DEFAULTS.get(feature, dict(value=0.0))
    return st.number_input(label, **cfg)


# --------------------------------------------------
# Sidebar — must stay at top level outside any tab
# --------------------------------------------------
with st.sidebar:
    st.header("Model Selection")

    model_type = st.selectbox(
        "Algorithm",
        options=["Random Forest", "XGBoost"],
    )

    disorder = st.selectbox(
        "Disorder",
        options=list(DISORDER_LABELS.keys()),
        format_func=lambda d: DISORDER_LABELS[d],
    )

    cohort = st.selectbox(
        "Age Cohort",
        options=list(COHORT_LABELS.keys()),
        format_func=lambda c: COHORT_LABELS[c],
    )

    # Warn if selected XGBoost model not trained yet
    if model_type == "XGBoost" and not is_model_available(model_type, disorder, cohort):
        st.warning(
            f"⚠️ XGBoost model for {DISORDER_LABELS[disorder]} / "
            f"{COHORT_LABELS[cohort]} is not trained yet. "
            "Switch to Random Forest or select Young Adult Insomnia."
        )

    st.markdown("---")
    st.caption(
        "Select an algorithm, disorder, and age cohort, "
        "fill in the inputs, then click Predict Risk."
    )


# --------------------------------------------------
# TABS
# --------------------------------------------------
tab_predict, tab_compare = st.tabs(["🔍 Risk Prediction", "📊 Model Comparison"])


# ==================================================
# TAB 1 — RISK PREDICTION
# ==================================================
with tab_predict:

    st.subheader(
        f"{model_type} — {DISORDER_LABELS[disorder]} Risk | {COHORT_LABELS[cohort]}"
    )

    # Show known metrics for selected model
    m = get_metrics(model_type, disorder, cohort)
    if m:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Test Accuracy", f"{m['accuracy']:.2f}%")
        mc2.metric("ROC-AUC",       f"{m['roc_auc']:.2f}%")
        mc3.metric("F1 (macro)",    f"{m['f1']:.2f}%")
        mc4.metric("Recall",        f"{m['recall']:.2f}%")
        st.caption("Metrics from training — on held-out test set.")
    else:
        st.info("Model metrics not yet available for this combination.")

    st.markdown("---")

    # Input form
    features   = get_feature_list(disorder)
    input_dict = {}
    cols       = st.columns(3)
    for i, feat in enumerate(features):
        with cols[i % 3]:
            input_dict[feat] = render_input_field(feat)

    predict_clicked = st.button("Predict Risk", type="primary")

    # --------------------------------------------------
    # Prediction + SHAP
    # --------------------------------------------------
    if predict_clicked:

        if model_type == "XGBoost" and not is_model_available(model_type, disorder, cohort):
            st.error(
                f"XGBoost model for {DISORDER_LABELS[disorder]} / "
                f"{COHORT_LABELS[cohort]} has not been trained yet. "
                "Please run the corresponding training script first, "
                "or switch to Random Forest."
            )
            st.stop()

        try:
            result = predict(disorder, cohort, input_dict, model_type=model_type)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.markdown("### Prediction")
        col1, col2 = st.columns(2)
        with col1:
            if result["risk"] == 1:
                st.error(f"**{result['risk_label']}**")
            else:
                st.success(f"**{result['risk_label']}**")
        with col2:
            st.metric("Risk Probability", f"{result['probability'] * 100:.1f}%")

        # --------------------------------------------------
        # SHAP — always uses RF model (shap_explainer only
        # supports RF). Show a note when XGBoost is selected.
        # --------------------------------------------------
        st.markdown("### Why this prediction? (SHAP)")

        if model_type == "XGBoost":
            st.caption(
                "ℹ️ SHAP explanations are generated using the Random Forest model. "
                "XGBoost SHAP support will be added once all models are complete."
            )

        try:
            explanation = explain_instance(disorder, cohort, input_dict, top_n=8)
        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")
            st.stop()

        shap_vals   = explanation["shap_values"]
        feat_names  = [FEATURE_LABELS.get(d["feature"], d["feature"])
                       for d in shap_vals]
        shap_scores = [d["shap_value"] for d in shap_vals]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors  = ["#e74c3c" if v > 0 else "#3498db" for v in shap_scores]
        ax.barh(feat_names[::-1], shap_scores[::-1], color=colors[::-1])
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel("SHAP value (→ increases risk | ← decreases risk)")
        ax.set_title(
            f"Top factors — {DISORDER_LABELS[disorder]} | {COHORT_LABELS[cohort]}"
        )
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Raw SHAP values"):
            for d in shap_vals:
                st.write(
                    f"**{FEATURE_LABELS.get(d['feature'], d['feature'])}** "
                    f"= {d['value']}  →  SHAP {d['shap_value']:+.4f}"
                )

        # --------------------------------------------------
        # Personalized Recommendations — SHAP-driven, built on
        # top of the same explanation shown above. Always uses
        # the RF model, same as the SHAP section (recommendation_engine
        # calls explain_instance() internally).
        # --------------------------------------------------
        st.markdown("### Personalized Recommendations")

        try:
            rec = generate_recommendations(disorder, cohort, input_dict, top_n=5)
        except Exception as e:
            st.error(f"Recommendation generation failed: {e}")
            rec = None

        if rec:
            if rec["risk_increasing"]:
                st.markdown("#### 🔴 Factors contributing to risk")
                for item in rec["risk_increasing"]:
                    st.markdown(f"- {item['message']}")
            else:
                st.success(
                    "No modifiable factors were found pushing risk upward "
                    "for this prediction."
                )

            if rec["protective"]:
                st.markdown("#### 🟢 Factors working in your favor")
                for item in rec["protective"]:
                    st.markdown(f"- {item['message']}")

            if rec["non_modifiable"]:
                with st.expander("ℹ️ Non-modifiable contributing factors"):
                    for item in rec["non_modifiable"]:
                        st.markdown(f"- {item['message']}")

            st.caption(rec["disclaimer"])


# ==================================================
# TAB 2 — MODEL COMPARISON
# ==================================================
with tab_compare:

    st.subheader("Random Forest vs XGBoost — Accuracy Comparison")
    st.caption(
        "Comparison across all 6 cohort × disorder models. "
        "Grey bars indicate models not yet trained (placeholder)."
    )

    # Metric selector
    metric_map = {
        "Accuracy (%)": "accuracy",
        "ROC-AUC (%)":  "roc_auc",
        "F1-macro (%)": "f1",
        "Recall (%)":   "recall",
    }
    selected_label  = st.selectbox("Metric", options=list(metric_map.keys()))
    selected_metric = metric_map[selected_label]

    # One chart per disorder
    for disorder_key, disorder_name in DISORDER_LABELS.items():
        st.markdown(f"#### {disorder_name}")

        cohorts      = list(COHORT_LABELS.keys())
        cohort_names = [COHORT_LABELS[c] for c in cohorts]

        rf_vals,  xgb_vals  = [], []
        rf_avail, xgb_avail = [], []

        for c in cohorts:
            rf_m  = get_metrics("Random Forest", disorder_key, c)
            xgb_m = get_metrics("XGBoost",       disorder_key, c)
            rf_vals.append(rf_m[selected_metric]   if rf_m  else 0)
            xgb_vals.append(xgb_m[selected_metric] if xgb_m else 0)
            rf_avail.append(rf_m  is not None)
            xgb_avail.append(xgb_m is not None)

        x     = np.arange(len(cohorts))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8f9fa")

        rf_bars = ax.bar(
            x - width / 2, rf_vals, width, label="Random Forest",
            color=[MODEL_COLORS["Random Forest"] if a else "#cccccc"
                   for a in rf_avail],
            alpha=0.88,
        )
        xgb_bars = ax.bar(
            x + width / 2, xgb_vals, width, label="XGBoost",
            color=[MODEL_COLORS["XGBoost"] if a else "#cccccc"
                   for a in xgb_avail],
            alpha=0.88,
        )

        for bar, val, avail, color in zip(
            rf_bars, rf_vals, rf_avail,
            [MODEL_COLORS["Random Forest"]] * len(rf_vals)
        ):
            label_text = f"{val:.1f}%" if avail else "N/A"
            label_color = color if avail else "#999999"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    label_text, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=label_color)

        for bar, val, avail, color in zip(
            xgb_bars, xgb_vals, xgb_avail,
            [MODEL_COLORS["XGBoost"]] * len(xgb_vals)
        ):
            label_text = f"{val:.1f}%" if avail else "N/A"
            label_color = color if avail else "#999999"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    label_text, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=label_color)

        ax.set_xticks(x)
        ax.set_xticklabels(cohort_names, fontsize=10)
        ax.set_ylabel(selected_label, fontsize=10)
        ax.set_ylim(0, 110)
        ax.set_title(
            f"{disorder_name} — {selected_label} by Cohort",
            fontweight="bold", pad=10,
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", color="#e0e0e0")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    # Summary table
    st.markdown("#### Full Metrics Table")
    st.caption("N/A = model not yet trained.")

    rows = []
    for mt in ["Random Forest", "XGBoost"]:
        for dk, dn in DISORDER_LABELS.items():
            for ck, cn in COHORT_LABELS.items():
                m = get_metrics(mt, dk, ck)
                rows.append({
                    "Model":    mt,
                    "Disorder": dn,
                    "Cohort":   cn,
                    "Accuracy": f"{m['accuracy']:.2f}%" if m else "N/A",
                    "ROC-AUC":  f"{m['roc_auc']:.2f}%"  if m else "N/A",
                    "F1-macro": f"{m['f1']:.2f}%"        if m else "N/A",
                    "Recall":   f"{m['recall']:.2f}%"    if m else "N/A",
                })

    def highlight_na(val):
        if val == "N/A":
            return "color: #aaaaaa; font-style: italic"
        return ""

    comparison_df = pd.DataFrame(rows)
    st.dataframe(
        comparison_df.style.applymap(highlight_na),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.caption(
        "**N/A models in progress:** XGBoost Middle Aged Insomnia, "
        "XGBoost Older Adult Insomnia, and all three XGBoost Apnea cohorts. "
        "Train each model and update METRICS in models_registry.py to populate."
    )