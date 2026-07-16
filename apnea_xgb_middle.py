# apnea_xgb_middle_aged.py
# XGBoost — Sleep Apnea Risk Prediction
# Cohort: Middle-Aged (Age 36-55)
# Mirrors apnea_rf_middle_aged.py's data/target logic exactly,
# and insomnia_xgb_young_adult.py's XGBoost config exactly.
# Saves: apnea_xgb_middle_aged.pkl -> {'xgb': model, 'scaler': scaler}

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#f8f9fa",
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("agewise_cleaned_full.csv")
print("Full dataset shape:", df.shape)
print("Age range:", df["age"].min(), "to", df["age"].max())

# ---------------------------------------------------------------
# 2. RECOVER apnea_risk_score FOR TARGET CONSTRUCTION
#    apnea_risk_score was dropped in preprocessing (label leaker —
#    see claude_preprocess.py, Section 5). It is recovered here
#    ONLY to build the binary target, never reintroduced as a feature.
#    Row alignment verified: raw and cleaned CSVs share identical
#    row order (no rows dropped/reordered in preprocessing).
# ---------------------------------------------------------------
raw = pd.read_csv("smartwatch_sleep_dataset.csv")
assert len(raw) == len(df), "Row count mismatch between raw and cleaned data"
assert (raw["age"].values == df["age"].values).all(), "Row order mismatch between raw and cleaned data"

df["apnea_risk_score"] = raw["apnea_risk_score"].values

# ---------------------------------------------------------------
# 3. FILTER TO YOUNG ADULT COHORT (18-35)
#    Matches apnea_rf_middle_aged.py filter: (age > 35) & (age <= 55)
# ---------------------------------------------------------------
mid = df[(df["age"] > 35) & (df["age"] <= 55)].copy()
mid.reset_index(drop=True, inplace=True)

print("\nMiddle-Aged cohort (36-55):")
print("  Records  :", len(mid))
print("  Age range:", mid["age"].min(), "to", mid["age"].max())

# ---------------------------------------------------------------
# 4. BUILD BINARY APNEA TARGET
#    AASM cutoff: AHI >= 15 (at least moderate OSA) -> apnea risk = 1
# ---------------------------------------------------------------
mid["apnea_risk"] = (mid["apnea_risk_score"] >= 15).astype(int)

print("\nApnea risk distribution (Middle-Aged Adults):")
print(mid["apnea_risk"].value_counts())
print("\nClass percentages:")
print(round(mid["apnea_risk"].value_counts(normalize=True) * 100, 1))

# ---------------------------------------------------------------
# 5. FEATURE SELECTION
#    Same 22 features as apnea_rf_middle_aged.py
#    apnea_risk_score excluded (used only to build the target)
# ---------------------------------------------------------------
FEATURES = [
    "age",
    "gender",
    "weight_kg",
    "height_cm",
    "duration_minutes",
    "sleep_efficiency_pct",
    "sleep_stage_deep_pct",
    "sleep_stage_light_pct",
    "sleep_stage_rem_pct",
    "sleep_stage_awake_pct",
    "wake_after_sleep_onset_minutes",
    "heart_rate_mean_bpm",
    "heart_rate_min_bpm",
    "heart_rate_max_bpm",
    "hrv_rmssd_ms",
    "respiration_rate_bpm",
    "spo2_mean_pct",
    "spo2_min_pct",
    "snore_events",
    "movement_count",
    "stress_score",
    "alcohol_units",
]

X = mid[FEATURES].copy()
y = mid["apnea_risk"].copy()

print("\nFeature matrix shape:", X.shape)

# ---------------------------------------------------------------
# 6. ENCODE GENDER
# ---------------------------------------------------------------
le_gender = LabelEncoder()
X["gender"] = le_gender.fit_transform(X["gender"])
print("Gender encoding:", dict(zip(le_gender.classes_,
                                   le_gender.transform(le_gender.classes_))))

# ---------------------------------------------------------------
# 7. STRATIFIED 80/20 TRAIN/TEST SPLIT
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
print("\nTrain size:", X_train.shape)
print("Test size :", X_test.shape)
print("\nTrain class distribution:")
print(y_train.value_counts())

# ---------------------------------------------------------------
# 8. SMOTE — TRAINING SET ONLY
#    Test set stays imbalanced — reflects real-world distribution
# ---------------------------------------------------------------
print("\nBefore SMOTE:", dict(y_train.value_counts()))
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("After SMOTE :", dict(pd.Series(y_train_bal).value_counts()))
print("Train shape after SMOTE:", X_train_bal.shape)

# ---------------------------------------------------------------
# 9. STANDARD SCALING — FIT ON TRAIN ONLY
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled  = scaler.transform(X_test)

# ---------------------------------------------------------------
# 10. TRAIN XGBOOST
#     Same hyperparameters as insomnia_xgb_young_adult.py
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING XGBOOST — APNEA (MIDDLE-AGED 36-55)")
print("=" * 60)

xgb = XGBClassifier(
    n_estimators     = 400,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = "logloss",
    random_state     = 42,
    n_jobs           = -1,
)
xgb.fit(X_train_scaled, y_train_bal)
print("Model trained.")

# ---------------------------------------------------------------
# 11. EVALUATION
# ---------------------------------------------------------------
y_pred      = xgb.predict(X_test_scaled)
y_pred_prob = xgb.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)
cv_scores = cross_val_score(
    xgb, X_train_scaled, y_train_bal, cv=5, scoring="accuracy"
)

print("\n" + "=" * 60)
print("RESULTS — APNEA RISK (MIDDLE-AGED 36-55)")
print("=" * 60)
print(f"  Test Accuracy      : {accuracy * 100:.2f}%")
print(f"  ROC-AUC Score      : {roc_auc:.4f}")
print(f"  5-Fold CV Accuracy : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["No Apnea Risk", "Apnea Risk"]))

# ---------------------------------------------------------------
# 12. FEATURE IMPORTANCE
# ---------------------------------------------------------------
feat_imp_df = pd.DataFrame({
    "feature":    FEATURES,
    "importance": xgb.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("Top 10 Feature Importances:")
print(feat_imp_df.head(10).to_string(index=False))

# ---------------------------------------------------------------
# 13. PLOTS
# ---------------------------------------------------------------

# Plot 1 — Feature Importance
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#27ae60" if i < 3 else "#2ecc71" if i < 7 else "#95a5a6"
          for i in range(len(feat_imp_df))]
ax.barh(feat_imp_df["feature"][::-1], feat_imp_df["importance"][::-1],
        color=colors[::-1])
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title(
    "XGBoost — Feature Importance\nApnea Risk | Middle-Aged (36-55)",
    fontweight="bold", pad=12,
)
for i, (val, _) in enumerate(zip(feat_imp_df["importance"][::-1],
                                  feat_imp_df["feature"][::-1])):
    ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("plots/xgb_apnea_feature_importance_middle.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("\n  Plot saved: plots/xgb_apnea_feature_importance_middle.png")

# Plot 2 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=["No Apnea Risk", "Apnea Risk"],
    yticklabels=["No Apnea Risk", "Apnea Risk"],
    linewidths=0.5, linecolor="white", ax=ax,
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(
    "Confusion Matrix\nApnea Risk | Middle-Aged (36-55)",
    fontweight="bold", pad=12,
)
plt.tight_layout()
plt.savefig("plots/xgb_apnea_confusion_matrix_middle.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: plots/xgb_apnea_confusion_matrix_middle.png")

# Plot 3 — ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#27ae60", lw=2,
        label=f"ROC Curve (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="#bdc3c7", lw=1.5,
        linestyle="--", label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color="#27ae60")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(
    "ROC Curve — XGBoost\nApnea Risk | Middle-Aged (36-55)",
    fontweight="bold", pad=12,
)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig("plots/xgb_apnea_roc_curve_middle.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: plots/xgb_apnea_roc_curve_middle.png")

# ---------------------------------------------------------------
# 14. SAVE MODEL
#     Saves {'xgb': model, 'scaler': scaler} — same bundle
#     structure as RF .pkl files so models_registry.py can load
#     both with the same interface.
# ---------------------------------------------------------------
bundle = {"xgb": xgb, "scaler": scaler}
with open("apnea_xgb_middle_aged.pkl", "wb") as f:
    pickle.dump(bundle, f)
print("\n  Model saved: apnea_xgb_middle_aged.pkl")

# ---------------------------------------------------------------
# 15. SUMMARY
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Cohort             : Middle-Aged (Age 36-55)")
print(f"  Disorder Target    : Apnea Risk (binary)")
print(f"  Target source      : apnea_risk_score >= 15 (AASM AHI-moderate cutoff)")
print(f"  Training samples   : {X_train_scaled.shape[0]} (after SMOTE)")
print(f"  Test samples       : {X_test_scaled.shape[0]}")
print(f"  Features used      : {len(FEATURES)}")
print(f"  Test Accuracy      : {accuracy * 100:.2f}%")
print(f"  ROC-AUC            : {roc_auc:.4f}")
print(f"  CV Accuracy (5-fold): {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
print(f"  Top feature        : {feat_imp_df.iloc[0]['feature']} ({feat_imp_df.iloc[0]['importance']:.4f})")
print(f"  2nd feature        : {feat_imp_df.iloc[1]['feature']} ({feat_imp_df.iloc[1]['importance']:.4f})")
print(f"  3rd feature        : {feat_imp_df.iloc[2]['feature']} ({feat_imp_df.iloc[2]['importance']:.4f})")
print()
print("Next step: SHAP explainability on this model.")