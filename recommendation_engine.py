# recommendation_engine.py
# Personalized health recommendations for AgeWise, derived from
# per-individual SHAP values.
#
# Sits on top of shap_explainer.py (which itself sits on models_registry.py),
# so recommendations are always generated from the same SHAP explanation
# shown in the dashboard — nothing is recomputed differently here.
#
# Approach: for a given person's prediction, take the SHAP values from
# explain_instance(), and turn the top RISK-INCREASING contributors
# (positive SHAP == pushed this person's prediction toward risk) into
# personalized, feature-specific recommendations, ranked by |shap_value|.
# Protective (negative SHAP) features are also surfaced as positive
# reinforcement, not just warnings.
#
# This is NOT a diagnostic tool. It does not replace clinical judgment —
# see DISCLAIMER at the bottom of every recommendation set.

from shap_explainer import explain_instance

# --------------------------------------------------
# Feature -> recommendation template.
# Each entry is a function(value) -> str, written to read naturally
# regardless of the person's actual measured value.
#
# Covers the union of INSOMNIA_FEATURES and APNEA_FEATURES from
# models_registry.py.
# --------------------------------------------------

FEATURE_RECOMMENDATIONS = {
    "sleep_efficiency_pct": lambda v: (
        f"Your sleep efficiency ({v:.1f}%) is a key factor here. Time in bed "
        "spent actually asleep, not just lying down, matters — going to bed "
        "only when sleepy and getting up at a consistent time can help raise this."
    ),
    "wake_after_sleep_onset_minutes": lambda v: (
        f"You're waking up for about {v:.0f} minutes total after initially "
        "falling asleep. Reducing evening screen exposure and keeping the "
        "bedroom cool and dark can reduce these awakenings."
    ),
    "sleep_latency_minutes": lambda v: (
        f"It's taking around {v:.0f} minutes to fall asleep. A consistent "
        "wind-down routine (dim lights, no screens) 30–60 minutes before bed "
        "can help shorten this."
    ),
    "stress_score": lambda v: (
        f"Stress score ({v:.0f}) is contributing meaningfully here. "
        "Relaxation techniques before bed — breathing exercises, light "
        "stretching, journaling — are well-supported for reducing pre-sleep stress."
    ),
    "hrv_rmssd_ms": lambda v: (
        f"HRV (RMSSD = {v:.1f} ms) reflects autonomic recovery. Lower values "
        "can be improved with regular moderate exercise and consistent sleep "
        "timing, though HRV is also influenced by many non-sleep factors."
    ),
    "sleep_stage_deep_pct": lambda v: (
        f"Deep sleep is at {v:.1f}% of total sleep. Avoiding alcohol close to "
        "bedtime and keeping a consistent sleep schedule are both linked to "
        "healthier deep sleep proportions."
    ),
    "sleep_stage_rem_pct": lambda v: (
        f"REM sleep is at {v:.1f}%. REM is sensitive to sleep disruption — "
        "protecting total sleep duration (not just efficiency) tends to help this."
    ),
    "sleep_stage_light_pct": lambda v: (
        f"Light sleep makes up {v:.1f}% of your sleep. On its own this is less "
        "actionable, but it often shifts as deep and REM sleep improve."
    ),
    "sleep_stage_awake_pct": lambda v: (
        f"About {v:.1f}% of your time in bed is spent awake. This overlaps "
        "with fragmentation — a cooler room and avoiding late caffeine can help."
    ),
    "caffeine_mg": lambda v: (
        f"Caffeine intake ({v:.0f} mg) is a contributing factor. Caffeine has "
        "a half-life of several hours — shifting your last intake earlier in "
        "the day is a well-supported way to reduce its effect on sleep."
    ),
    "alcohol_units": lambda v: (
        f"Alcohol intake ({v:.1f} units) is contributing here. Alcohol can "
        "relax the upper airway (worsening apnea risk) and fragments sleep "
        "later in the night even when it helps you fall asleep faster."
    ),
    "screen_time_before_bed_min": lambda v: (
        f"About {v:.0f} minutes of screen time before bed is a factor. Blue "
        "light and mental stimulation from screens can delay sleep onset — "
        "a 30–60 minute screen-free wind-down is a common recommendation."
    ),
    "activity_before_bed_min": lambda v: (
        f"{v:.0f} minutes of activity close to bedtime is contributing. "
        "Vigorous exercise raises core body temperature and alertness — "
        "shifting intense activity earlier in the day can help."
    ),
    "bedtime_consistency_std_min": lambda v: (
        f"Your bedtime varies by about {v:.0f} minutes night-to-night. "
        "Circadian rhythm responds strongly to consistency — keeping bed and "
        "wake times within a similar window daily (even on weekends) helps."
    ),
    "duration_minutes": lambda v: (
        f"Total sleep duration is about {v/60:.1f} hours. Adults generally "
        "need 7–9 hours — consistently short sleep compounds many of the "
        "other risk factors here."
    ),
    "weight_kg": lambda v: (
        f"Weight ({v:.0f} kg) is the single strongest physiological factor "
        "for apnea risk in general. Even modest weight reduction (5–10%) is "
        "well-documented to meaningfully reduce OSA severity."
    ),
    "height_cm": lambda v: (
        "Height is a structural factor in airway anatomy and isn't something "
        "actionable — it's included for model completeness rather than as "
        "personal guidance."
    ),
    "snore_events": lambda v: (
        f"Snore events ({v:.0f}) are a direct clinical marker of airway "
        "obstruction. Sleeping on your side rather than your back, and "
        "avoiding alcohol/sedatives before bed, can reduce snoring."
    ),
    "spo2_mean_pct": lambda v: (
        f"Average blood oxygen ({v:.1f}%) is contributing to this result. "
        "Persistently reduced SpO2 during sleep is a core apnea marker and "
        "worth discussing with a doctor if it's consistently below ~95%."
    ),
    "spo2_min_pct": lambda v: (
        f"Minimum blood oxygen dipped to {v:.1f}%. Repeated desaturation "
        "events like this are one of the clearest clinical signs of apnea "
        "and are worth flagging to a sleep specialist."
    ),
    "respiration_rate_bpm": lambda v: (
        f"Respiration rate ({v:.1f} breaths/min) is a contributing factor — "
        "irregular breathing during sleep is a hallmark of apnea and is best "
        "assessed with a clinical sleep study if this persists."
    ),
    "movement_count": lambda v: (
        f"Movement count during sleep ({v:.0f}) is elevated. Frequent "
        "movement often reflects underlying sleep fragmentation rather than "
        "being a cause on its own."
    ),
    "heart_rate_mean_bpm": lambda v: (
        f"Average heart rate during sleep ({v:.0f} bpm) is a contributing "
        "factor. This is influenced by fitness, stress, and sleep quality "
        "together rather than any single behavior."
    ),
    "heart_rate_min_bpm": lambda v: (
        f"Minimum heart rate ({v:.0f} bpm) during sleep is contributing here. "
        "This reflects cardiovascular recovery — regular aerobic exercise "
        "tends to improve it over time."
    ),
    "heart_rate_max_bpm": lambda v: (
        f"Peak heart rate during sleep ({v:.0f} bpm) is a factor — spikes can "
        "relate to arousals or breathing disruptions rather than activity."
    ),
    "age": lambda v: (
        f"Age ({v:.0f}) is a non-modifiable factor — it's included because "
        "sleep architecture genuinely changes with age, not because anything "
        "can or should be done about it."
    ),
    "gender": lambda v: (
        "Gender is a non-modifiable structural factor in this model — "
        "included because it's an established epidemiological risk factor, "
        "not something to act on."
    ),
}

DEFAULT_TEMPLATE = lambda feat, v: (
    f"{feat.replace('_', ' ').title()} (value: {v}) is contributing to this "
    "prediction. No specific guidance is available for this factor yet."
)

DISCLAIMER = (
    "These recommendations are generated from a machine learning model's "
    "explanation of its own prediction (SHAP values), not a clinical "
    "diagnosis. They highlight statistical contributors to the model's "
    "output, not medical advice. Persistent sleep issues — especially "
    "recurring low blood oxygen, loud snoring, or ongoing insomnia — should "
    "be evaluated by a physician or sleep specialist, not managed from this "
    "dashboard alone."
)

# Features that are non-modifiable — flagged separately even if they're a
# top contributor, so recommendations don't imply someone can "fix" their age.
NON_MODIFIABLE = {"age", "gender", "height_cm"}


def generate_recommendations(disorder, cohort, input_dict, top_n=5):
    """
    Generates personalized, SHAP-driven recommendations for one person.

    Parameters
    ----------
    disorder   : "insomnia" or "apnea"
    cohort     : "young_adult", "middle_aged", or "older_adult"
    input_dict : dict with all required feature values (same as predict())
    top_n      : max number of risk-increasing recommendations to return

    Returns
    -------
    dict with keys:
        risk_increasing : list of {feature, shap_value, message} — top
                           contributors that pushed risk UP, ranked by
                           |shap_value|, modifiable factors only.
        non_modifiable   : list of {feature, shap_value, message} — top
                           contributors that pushed risk UP but aren't
                           actionable (age, gender, height).
        protective       : list of {feature, shap_value, message} — top
                           factors that pushed risk DOWN, for positive
                           reinforcement.
        disclaimer       : str
    """
    explanation = explain_instance(disorder, cohort, input_dict, top_n=None)
    shap_values = explanation["shap_values"]  # already sorted by |shap_value|

    risk_increasing = [d for d in shap_values if d["shap_value"] > 0
                        and d["feature"] not in NON_MODIFIABLE]
    non_modifiable = [d for d in shap_values if d["shap_value"] > 0
                       and d["feature"] in NON_MODIFIABLE]
    protective = [d for d in shap_values if d["shap_value"] < 0]

    def build(items, limit):
        out = []
        for d in items[:limit]:
            feat, val = d["feature"], d["value"]
            template = FEATURE_RECOMMENDATIONS.get(feat)
            try:
                message = template(val) if template else DEFAULT_TEMPLATE(feat, val)
            except (TypeError, ValueError):
                message = DEFAULT_TEMPLATE(feat, val)
            out.append({
                "feature": feat,
                "shap_value": d["shap_value"],
                "message": message,
            })
        return out

    return {
        "risk_increasing": build(risk_increasing, top_n),
        "non_modifiable":  build(non_modifiable, top_n),
        "protective":      build(protective, min(top_n, 3)),
        "disclaimer":      DISCLAIMER,
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
    result = generate_recommendations("insomnia", "middle_aged", example_input, top_n=5)

    print("RISK-INCREASING (modifiable):")
    for r in result["risk_increasing"]:
        print(f"  [{r['shap_value']:+.4f}] {r['message']}")

    print("\nNON-MODIFIABLE CONTRIBUTORS:")
    for r in result["non_modifiable"]:
        print(f"  [{r['shap_value']:+.4f}] {r['message']}")

    print("\nPROTECTIVE FACTORS:")
    for r in result["protective"]:
        print(f"  [{r['shap_value']:+.4f}] {r['message']}")

    print("\nDISCLAIMER:")
    print(result["disclaimer"])