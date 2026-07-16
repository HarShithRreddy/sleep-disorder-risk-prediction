# apnea_rf_young_adult.py
# Random Forest — Sleep Apnea Risk Prediction
# Cohort: Young Adults (Age 18–35)
# Author: Aaryesh

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os


def train_apnea_young_adult():

    os.makedirs("plots", exist_ok=True)

    plt.rcParams.update({
        'figure.facecolor' : '#ffffff',
        'axes.facecolor'   : '#f8f9fa',
        'axes.grid'        : True,
        'grid.color'       : '#e0e0e0',
        'grid.linestyle'   : '--',
        'axes.spines.top'  : False,
        'axes.spines.right': False,
        'font.family'      : 'DejaVu Sans',
        'axes.titlesize'   : 13,
        'axes.labelsize'   : 11,
    })


    # --------------------------------------------------
    # Load cleaned dataset from preprocessing output
    # --------------------------------------------------

    df = pd.read_csv("agewise_cleaned_full.csv")

    print("Full cleaned dataset shape:", df.shape)
    print("Age range:", df['age'].min(), "to", df['age'].max())


    # --------------------------------------------------
    # Recover apnea_risk_score for target construction
    #
    # apnea_risk_score was correctly dropped in preprocessing because it is
    # a disorder risk score, not a predictor (see claude_preprocess.py,
    # Section 5 — Label Leakers). It therefore does NOT exist in
    # agewise_cleaned_full.csv.
    #
    # However, exactly like sleep_score was used ONCE to derive the
    # sleep_quality label before being dropped, apnea_risk_score is used
    # here ONLY to derive a binary apnea target — it is never reintroduced
    # as a feature. We pull it back in from the original raw CSV.
    #
    # Row alignment: claude_preprocess.py does not drop or reorder any rows
    # (only columns are added/removed; outlier handling only clips values),
    # so row i of the raw file corresponds exactly to row i of
    # agewise_cleaned_full.csv. This was verified by comparing the
    # unmodified 'age' column across both files (100% match, n=20000).
    # --------------------------------------------------

    raw = pd.read_csv("smartwatch_sleep_dataset.csv")
    assert len(raw) == len(df), "Row count mismatch between raw and cleaned data"
    assert (raw['age'].values == df['age'].values).all(), "Row order mismatch between raw and cleaned data"

    df['apnea_risk_score'] = raw['apnea_risk_score'].values


    # --------------------------------------------------
    # Filter to Young Adult cohort (18–35)
    # --------------------------------------------------

    young = df[df['age'] <= 35].copy()
    young.reset_index(drop=True, inplace=True)

    print("\nYoung Adult cohort (18–35):")
    print("  Records:", len(young))
    print("  Age range:", young['age'].min(), "to", young['age'].max())


    # --------------------------------------------------
    # Build apnea target variable
    #
    # apnea_risk_score behaves like a simulated Apnea-Hypopnea Index (AHI),
    # the clinical metric (events/hour) used to grade obstructive sleep
    # apnea severity:
    #   AHI < 5        → normal
    #   5  <= AHI < 15  → mild
    #   15 <= AHI < 30  → moderate
    #   AHI >= 30       → severe
    #
    # We use the standard AASM cutoff of AHI >= 15 (at least moderate OSA)
    # as the positive class threshold — the same style of clinically
    # grounded thresholding used for sleep_score → sleep_quality.
    #
    # Mapping: apnea_risk_score >= 15 → apnea risk = 1
    #          apnea_risk_score <  15 → apnea risk = 0
    # --------------------------------------------------

    young['apnea_risk'] = (young['apnea_risk_score'] >= 15).astype(int)

    print("\nApnea risk distribution (Young Adults):")
    print(young['apnea_risk'].value_counts())
    print("\nClass percentages:")
    print(round(young['apnea_risk'].value_counts(normalize=True) * 100, 1))


    # --------------------------------------------------
    # Select features
    #
    # apnea_risk_score itself is excluded (used only to build the target).
    # Feature choices are grounded in OSA literature: oxygen desaturation
    # and respiration irregularity are the core physiological markers of
    # apnea; snoring is the primary clinical symptom; BMI-related measures
    # (weight, height) and sex are established structural risk factors
    # (Ha et al., JMIR 2023 — OSA driven by BMI/weight/sex, distinct from
    # insomnia's behavioral drivers). Alcohol is included since it relaxes
    # upper-airway muscles and is a known apnea-worsening factor.
    # --------------------------------------------------

    features = [
        'age',
        'gender',
        'weight_kg',
        'height_cm',
        'duration_minutes',
        'sleep_efficiency_pct',
        'sleep_stage_deep_pct',
        'sleep_stage_light_pct',
        'sleep_stage_rem_pct',
        'sleep_stage_awake_pct',
        'wake_after_sleep_onset_minutes',
        'heart_rate_mean_bpm',
        'heart_rate_min_bpm',
        'heart_rate_max_bpm',
        'hrv_rmssd_ms',
        'respiration_rate_bpm',
        'spo2_mean_pct',
        'spo2_min_pct',
        'snore_events',
        'movement_count',
        'stress_score',
        'alcohol_units',
    ]

    X = young[features].copy()
    y = young['apnea_risk'].copy()

    print("\nFeature matrix shape:", X.shape)


    # --------------------------------------------------
    # Encode gender
    # --------------------------------------------------

    le = LabelEncoder()
    X['gender'] = le.fit_transform(X['gender'])
    print("\nGender encoding:", dict(zip(le.classes_, le.transform(le.classes_))))


    # --------------------------------------------------
    # Train / test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain size:", X_train.shape)
    print("Test size:", X_test.shape)
    print("\nTrain class distribution:")
    print(y_train.value_counts())


    # --------------------------------------------------
    # SMOTE — train set only
    # --------------------------------------------------

    print("\nBefore SMOTE:", dict(y_train.value_counts()))

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print("After SMOTE:", dict(pd.Series(y_train_bal).value_counts()))
    print("Train shape after SMOTE:", X_train_bal.shape)


    # --------------------------------------------------
    # Standard scaling — fit on train, transform both
    # --------------------------------------------------

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled  = scaler.transform(X_test)


    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST — APNEA (YOUNG ADULTS 18–35)")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train_scaled, y_train_bal)
    print("Model trained.")


    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    y_pred      = rf.predict(X_test_scaled)
    y_pred_prob = rf.predict_proba(X_test_scaled)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_pred_prob)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train_bal, cv=5, scoring='accuracy')

    print("\n" + "=" * 60)
    print("RESULTS — APNEA RISK (YOUNG ADULTS 18–35)")
    print("=" * 60)
    print(f"  Test Accuracy       : {accuracy * 100:.2f}%")
    print(f"  ROC-AUC Score       : {roc_auc:.4f}")
    print(f"  5-Fold CV Accuracy  : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Apnea Risk', 'Apnea Risk']))


    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    importances = rf.feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature'   : features,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print("Top 10 Feature Importances:")
    print(feat_imp_df.head(10).to_string(index=False))


    # --------------------------------------------------
    # PLOT 1 — Feature Importance
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#e74c3c' if i < 3 else '#3498db' if i < 7 else '#95a5a6'
              for i in range(len(feat_imp_df))]

    ax.barh(feat_imp_df['feature'][::-1], feat_imp_df['importance'][::-1], color=colors[::-1])
    ax.set_xlabel('Feature Importance (Gini)')
    ax.set_title(
        'Random Forest — Feature Importance\nApnea Risk | Young Adults (18–35)',
        fontweight='bold', pad=12
    )

    for i, (val, name) in enumerate(zip(feat_imp_df['importance'][::-1], feat_imp_df['feature'][::-1])):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('plots/rf_apnea_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  Plot saved: plots/rf_apnea_feature_importance.png")


    # --------------------------------------------------
    # PLOT 2 — Confusion Matrix
    # --------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Apnea Risk', 'Apnea Risk'],
        yticklabels=['No Apnea Risk', 'Apnea Risk'],
        linewidths=0.5, linecolor='white',
        ax=ax
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(
        'Confusion Matrix\nApnea Risk | Young Adults (18–35)',
        fontweight='bold', pad=12
    )
    plt.tight_layout()
    plt.savefig('plots/rf_apnea_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Plot saved: plots/rf_apnea_confusion_matrix.png")


    # --------------------------------------------------
    # PLOT 3 — ROC Curve
    # --------------------------------------------------

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#bdc3c7', lw=1.5, linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#e74c3c')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(
        'ROC Curve — Random Forest\nApnea Risk | Young Adults (18–35)',
        fontweight='bold', pad=12
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig('plots/rf_apnea_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Plot saved: plots/rf_apnea_roc_curve.png")


    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Cohort              : Young Adults (Age 18–35)")
    print(f"  Disorder Target     : Apnea Risk (binary)")
    print(f"  Apnea source        : apnea_risk_score >= 15 (AASM AHI-moderate cutoff)")
    print(f"  Training samples    : {X_train_scaled.shape[0]} (after SMOTE)")
    print(f"  Test samples        : {X_test_scaled.shape[0]}")
    print(f"  Features used       : {len(features)}")
    print(f"  Test Accuracy       : {accuracy * 100:.2f}%")
    print(f"  ROC-AUC             : {roc_auc:.4f}")
    print(f"  CV Accuracy (5-fold): {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
    print(f"  Top feature         : {feat_imp_df.iloc[0]['feature']} ({feat_imp_df.iloc[0]['importance']:.4f})")
    print(f"  2nd feature         : {feat_imp_df.iloc[1]['feature']} ({feat_imp_df.iloc[1]['importance']:.4f})")
    print(f"  3rd feature         : {feat_imp_df.iloc[2]['feature']} ({feat_imp_df.iloc[2]['importance']:.4f})")
    print()
    print("Next step: SHAP explainability on this model.")

    # save model and scaler for SHAP
    with open("apnea_rf_young_adult.pkl", "wb") as f:
        pickle.dump({'rf': rf, 'scaler': scaler}, f)
    print("Model saved: apnea_rf_young_adult.pkl")


if __name__ == "__main__":
    train_apnea_young_adult()