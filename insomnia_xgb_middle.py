# insomnia_xgb_middle_age.py
# XGBoost — Insomnia Risk Prediction
# Cohort: Middle Age (Age 36-55)
# Mirrors the RF training scripts exactly — same data source,
# same feature set, same split + SMOTE protocol.
# Saves: insomnia_xgb_middle_age.pkl -> {'xgb': model, 'scaler': scaler}

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
# 2. FILTER TO MIDDLE AGE COHORT (36-55)
#    Matches insomnia_rf_middle_age.py filter: (age >= 36) & (age <= 55)
# ---------------------------------------------------------------
middle = df[(df["age"] >= 36) & (df["age"] <= 55)].copy()
middle.reset_index(drop=True, inplace=True)

print("\nMiddle Age cohort (36-55):")
print("  Records  :", len(middle))
print("  Age range:", middle["age"].min(), "to", middle["age"].max())

# ---------------------------------------------------------------
# 3. BUILD BINARY INSOMNIA TARGET
#    Consistent with RF scripts:
#    sleep_quality == 'poor' -> insomnia_risk = 1
#    sleep_quality == 'fair' or 'good' -> insomnia_risk = 0
# ---------------------------------------------------------------
middle["insomnia_risk"] = (middle["sleep_quality"] == "poor").astype(int)

print("\nInsomnia risk distribution (Middle Age):")
print(middle["insomnia_risk"].value_counts())
print("\nClass percentages:")
print(round(middle["insomnia_risk"].value_counts(normalize=True) * 100, 1))

# ---------------------------------------------------------------
# 4. FEATURE SELECTION
#    Same 20 features as insomnia_rf_middle_age.py
#    sleep_score excluded (label leaker — used to derive sleep_quality)
# ---------------------------------------------------------------
FEATURES = [
    "age",
    "gender",
    "duration_minutes",
    "sleep_latency_minutes",
    "wake_after_sleep_onset_minutes",
    "sleep_efficiency_pct",
    "sleep_stage_deep_pct",
    "sleep_stage_light_pct",
    "sleep_stage_rem_pct",
    "sleep_stage_awake_pct",
    "heart_rate_mean_bpm",
    "heart_rate_min_bpm",
    "heart_rate_max_bpm",
    "hrv_rmssd_ms",
    "stress_score",
    "screen_time_before_bed_min",
    "activity_before_bed_min",
    "bedtime_consistency_std_min",
    "caffeine_mg",
    "alcohol_units",
]

X = middle[FEATURES].copy()
y = middle["insomnia_risk"].copy()

print("\nFeature matrix shape:", X.shape)

# ---------------------------------------------------------------
# 5. ENCODE GENDER
# ---------------------------------------------------------------
le_gender = LabelEncoder()
X["gender"] = le_gender.fit_transform(X["gender"])
print("Gender encoding:", dict(zip(le_gender.classes_,
                                   le_gender.transform(le_gender.classes_))))

# ---------------------------------------------------------------
# 6. STRATIFIED 80/20 TRAIN/TEST SPLIT
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
# 7. SMOTE — TRAINING SET ONLY
#    Test set stays imbalanced — reflects real-world distribution
# ---------------------------------------------------------------
print("\nBefore SMOTE:", dict(y_train.value_counts()))
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("After SMOTE :", dict(pd.Series(y_train_bal).value_counts()))
print("Train shape after SMOTE:", X_train_bal.shape)

# ---------------------------------------------------------------
# 8. STANDARD SCALING — FIT ON TRAIN ONLY
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled  = scaler.transform(X_test)

# ---------------------------------------------------------------
# 9. TRAIN XGBOOST
#    Same hyperparameters as insomnia_xgb_young_adult.py
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING XGBOOST — INSOMNIA (MIDDLE AGE 36-55)")
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
# 10. EVALUATION
# ---------------------------------------------------------------
y_pred      = xgb.predict(X_test_scaled)
y_pred_prob = xgb.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)
cv_scores = cross_val_score(
    xgb, X_train_scaled, y_train_bal, cv=5, scoring="accuracy"
)

print("\n" + "=" * 60)
print("RESULTS — INSOMNIA RISK (MIDDLE AGE 36-55)")
print("=" * 60)
print(f"  Test Accuracy      : {accuracy * 100:.2f}%")
print(f"  ROC-AUC Score      : {roc_auc:.4f}")
print(f"  5-Fold CV Accuracy : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["No Insomnia", "Insomnia Risk"]))

# ---------------------------------------------------------------
# 11. FEATURE IMPORTANCE
# ---------------------------------------------------------------
feat_imp_df = pd.DataFrame({
    "feature":    FEATURES,
    "importance": xgb.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("Top 10 Feature Importances:")
print(feat_imp_df.head(10).to_string(index=False))

# ---------------------------------------------------------------
# 12. PLOTS
# ---------------------------------------------------------------

# Plot 1 — Feature Importance
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#27ae60" if i < 3 else "#2ecc71" if i < 7 else "#95a5a6"
          for i in range(len(feat_imp_df))]
ax.barh(feat_imp_df["feature"][::-1], feat_imp_df["importance"][::-1],
        color=colors[::-1])
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title(
    "XGBoost — Feature Importance\nInsomnia Risk | Middle Age (36-55)",
    fontweight="bold", pad=12,
)
for i, (val, _) in enumerate(zip(feat_imp_df["importance"][::-1],
                                  feat_imp_df["feature"][::-1])):
    ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("plots/xgb_insomnia_feature_importance_middle.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("\n  Plot saved: plots/xgb_insomnia_feature_importance_middle.png")

# Plot 2 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=["No Insomnia", "Insomnia Risk"],
    yticklabels=["No Insomnia", "Insomnia Risk"],
    linewidths=0.5, linecolor="white", ax=ax,
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(
    "Confusion Matrix\nInsomnia Risk | Middle Age (36-55)",
    fontweight="bold", pad=12,
)
plt.tight_layout()
plt.savefig("plots/xgb_insomnia_confusion_matrix_middle.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: plots/xgb_insomnia_confusion_matrix_middle.png")

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
    "ROC Curve — XGBoost\nInsomnia Risk | Middle Age (36-55)",
    fontweight="bold", pad=12,
)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig("plots/xgb_insomnia_roc_curve_middle.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: plots/xgb_insomnia_roc_curve_middle.png")

# ---------------------------------------------------------------
# 13. SAVE MODEL
#     Saves {'xgb': model, 'scaler': scaler} — same bundle
#     structure as RF .pkl files so models_registry.py can load
#     both with the same interface.
# ---------------------------------------------------------------
bundle = {"xgb": xgb, "scaler": scaler}
with open("insomnia_xgb_middle_age.pkl", "wb") as f:
    pickle.dump(bundle, f)
print("\n  Model saved: insomnia_xgb_middle_age.pkl")

# ---------------------------------------------------------------
# 14. SUMMARY
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Cohort             : Middle Age (Age 36-55)")
print(f"  Disorder Target    : Insomnia Risk (binary)")
print(f"  Target source      : sleep_quality == 'poor'")
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