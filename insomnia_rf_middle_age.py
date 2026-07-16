# insomnia_rf_middle_age.py
# Random Forest — Insomnia Risk Prediction
# Cohort: Middle Age (Age 36–55)
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


def train_insomnia_middle_age():

    os.makedirs("plots_middle_age", exist_ok=True)

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
    # Load cleaned dataset
    # --------------------------------------------------

    df = pd.read_csv("agewise_cleaned_full.csv")

    print("Full cleaned dataset shape:", df.shape)
    print("Age range in dataset:", df['age'].min(), "to", df['age'].max())


    # --------------------------------------------------
    # Filter to Middle Age cohort (36–55)
    # --------------------------------------------------

    middle = df[(df['age'] >= 36) & (df['age'] <= 55)].copy()
    middle.reset_index(drop=True, inplace=True)

    print("\nMiddle Age cohort (36–55):")
    print("  Records:", len(middle))
    print("  Age range:", middle['age'].min(), "to", middle['age'].max())


    # --------------------------------------------------
    # Build insomnia target variable
    #
    # Same approach as Young Adult model for consistency:
    #   sleep_quality == 'poor'  → insomnia risk = 1
    #   sleep_quality == 'fair' or 'good' → insomnia risk = 0
    #
    # insomnia_flag was dropped in preprocessing (label leakage).
    # sleep_quality is derived from sleep_score thresholds —
    # a separate source from the physiological features in X.
    # --------------------------------------------------

    middle['insomnia_risk'] = (middle['sleep_quality'] == 'poor').astype(int)

    print("\nInsomnia risk distribution (Middle Age):")
    print(middle['insomnia_risk'].value_counts())
    print("\nClass percentages:")
    print(round(middle['insomnia_risk'].value_counts(normalize=True) * 100, 1))


    # --------------------------------------------------
    # Select features
    # sleep_score dropped in preprocessing — not included
    # all other requested features are present
    # --------------------------------------------------

    features = [
        'age',
        'gender',
        'duration_minutes',
        'sleep_latency_minutes',
        'wake_after_sleep_onset_minutes',
        'sleep_efficiency_pct',
        'sleep_stage_deep_pct',
        'sleep_stage_light_pct',
        'sleep_stage_rem_pct',
        'sleep_stage_awake_pct',
        'heart_rate_mean_bpm',
        'heart_rate_min_bpm',
        'heart_rate_max_bpm',
        'hrv_rmssd_ms',
        'stress_score',
        'screen_time_before_bed_min',
        'activity_before_bed_min',
        'bedtime_consistency_std_min',
        'caffeine_mg',
        'alcohol_units',
    ]

    X = middle[features].copy()
    y = middle['insomnia_risk'].copy()

    print("\nFeature matrix shape:", X.shape)


    # --------------------------------------------------
    # Encode gender
    # --------------------------------------------------

    le = LabelEncoder()
    X['gender'] = le.fit_transform(X['gender'])
    print("\nGender encoding:", dict(zip(le.classes_, le.transform(le.classes_))))


    # --------------------------------------------------
    # Train / test split
    # stratify=y to preserve class ratio in both splits
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
    print("\nTest class distribution:")
    print(y_test.value_counts())


    # --------------------------------------------------
    # SMOTE — train set only
    # ~94% no insomnia vs ~6% insomnia in this cohort
    # test set is intentionally left imbalanced so
    # evaluation reflects real world distribution
    # --------------------------------------------------

    print("\nBefore SMOTE:", dict(y_train.value_counts()))

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print("After SMOTE:", dict(pd.Series(y_train_bal).value_counts()))
    print("Train shape after SMOTE:", X_train_bal.shape)


    # --------------------------------------------------
    # Standard scaling
    # fit on train only — transform both with same scaler
    # --------------------------------------------------

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled  = scaler.transform(X_test)


    # --------------------------------------------------
    # Random Forest
    # same hyperparameters as young adult model
    # keeps comparison between cohorts fair
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST — INSOMNIA (MIDDLE AGE 36–55)")
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
    print("RESULTS — INSOMNIA RISK (MIDDLE AGE 36–55)")
    print("=" * 60)
    print(f"  Test Accuracy       : {accuracy * 100:.2f}%")
    print(f"  ROC-AUC Score       : {roc_auc:.4f}")
    print(f"  5-Fold CV Accuracy  : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Insomnia', 'Insomnia Risk']))


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

    colors = ['#e67e22' if i < 3 else '#e74c3c' if i < 7 else '#95a5a6'
              for i in range(len(feat_imp_df))]

    ax.barh(feat_imp_df['feature'][::-1], feat_imp_df['importance'][::-1],
            color=colors[::-1])
    ax.set_xlabel('Feature Importance (Gini)')
    ax.set_title(
        'Random Forest — Feature Importance\nInsomnia Risk | Middle Age (36–55)',
        fontweight='bold', pad=12
    )

    for i, (val, name) in enumerate(zip(
            feat_imp_df['importance'][::-1],
            feat_imp_df['feature'][::-1])):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('plots_middle_age/rf_insomnia_feature_importance.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  Plot saved: plots_middle_age/rf_insomnia_feature_importance.png")


    # --------------------------------------------------
    # PLOT 2 — Confusion Matrix
    # --------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Oranges',
        xticklabels=['No Insomnia', 'Insomnia Risk'],
        yticklabels=['No Insomnia', 'Insomnia Risk'],
        linewidths=0.5, linecolor='white', ax=ax
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(
        'Confusion Matrix\nInsomnia Risk | Middle Age (36–55)',
        fontweight='bold', pad=12
    )
    plt.tight_layout()
    plt.savefig('plots_middle_age/rf_insomnia_confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("  Plot saved: plots_middle_age/rf_insomnia_confusion_matrix.png")


    # --------------------------------------------------
    # PLOT 3 — ROC Curve
    # --------------------------------------------------

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#e67e22', lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#bdc3c7', lw=1.5,
            linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#e67e22')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(
        'ROC Curve — Random Forest\nInsomnia Risk | Middle Age (36–55)',
        fontweight='bold', pad=12
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig('plots_middle_age/rf_insomnia_roc_curve.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("  Plot saved: plots_middle_age/rf_insomnia_roc_curve.png")


    # --------------------------------------------------
    # PLOT 4 — Cross-Cohort Feature Importance Comparison
    # comparing top features between Middle Age and Young Adult
    # --------------------------------------------------

    young_importance = {
        'sleep_efficiency_pct'           : 0.3360,
        'stress_score'                   : 0.1660,
        'wake_after_sleep_onset_minutes' : 0.1620,
        'hrv_rmssd_ms'                   : 0.0680,
        'sleep_latency_minutes'          : 0.0580,
    }

    top5_features = feat_imp_df['feature'].head(5).tolist()
    middle_vals   = feat_imp_df.set_index('feature').loc[top5_features, 'importance'].values

    # get young adult values for the same features (0 if not in top)
    young_vals = [young_importance.get(f, 0.0) for f in top5_features]

    x     = np.arange(len(top5_features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width/2, young_vals,  width, label='Young Adult (18–35)',
           color='#3498db', alpha=0.85)
    ax.bar(x + width/2, middle_vals, width, label='Middle Age (36–55)',
           color='#e67e22', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(top5_features, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Feature Importance (Gini)')
    ax.set_title(
        'Feature Importance Comparison — Young Adult vs Middle Age\nInsomnia Risk Prediction',
        fontweight='bold', pad=12
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('plots_middle_age/rf_cohort_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("  Plot saved: plots_middle_age/rf_cohort_comparison.png")


    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Cohort              : Middle Age (Age 36–55)")
    print(f"  Disorder Target     : Insomnia Risk (binary)")
    print(f"  Insomnia source     : sleep_quality == 'poor'")
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
    print("Next step: Older Adult cohort (56–79) insomnia model.")

    # save model and scaler for SHAP
    with open("insomnia_rf_middle_age.pkl", "wb") as f:
        pickle.dump({'rf': rf, 'scaler': scaler}, f)
    print("Model saved: insomnia_rf_middle_age.pkl")


if __name__ == "__main__":
    train_insomnia_middle_age()