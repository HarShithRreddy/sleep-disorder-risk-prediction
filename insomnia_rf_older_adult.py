# insomnia_rf_older_adult.py
# Random Forest — Insomnia Risk Prediction
# Cohort: Older Adults (Age 55–79)
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


def train_insomnia_older_adult():

    os.makedirs("plots_older_adult", exist_ok=True)

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
    print("Age range:", df['age'].min(), "to", df['age'].max())


    # --------------------------------------------------
    # Filter to Older Adult cohort (55–79)
    #
    # Dataset max age is 79, not 80 — cohort is 55–79.
    # Using >= 55 since age 55 marks the clinical boundary
    # where sleep architecture changes become more pronounced:
    # deep sleep declines sharply, HRV drops, fragmentation
    # increases significantly from this age onward.
    # --------------------------------------------------

    older = df[df['age'] >= 55].copy()
    older.reset_index(drop=True, inplace=True)

    print("\nOlder Adult cohort (55–79):")
    print("  Records  :", len(older))
    print("  Age range:", older['age'].min(), "to", older['age'].max())


    # --------------------------------------------------
    # Build insomnia target variable
    #
    # Consistent with Young Adult and Middle Age models:
    #   sleep_quality == 'poor'  → insomnia_risk = 1
    #   sleep_quality == 'fair' or 'good' → insomnia_risk = 0
    #
    # insomnia_flag dropped in preprocessing (direct label leakage).
    # sleep_quality is derived from sleep_score thresholds —
    # a separate source from the physiological features in X.
    # --------------------------------------------------

    older['insomnia_risk'] = (older['sleep_quality'] == 'poor').astype(int)

    print("\nInsomnia risk distribution (Older Adults):")
    print(older['insomnia_risk'].value_counts())
    print("\nClass percentages:")
    print(round(older['insomnia_risk'].value_counts(normalize=True) * 100, 1))


    # --------------------------------------------------
    # Select features
    # sleep_score dropped in preprocessing (leakage) — not included
    # same 20-feature set as Young Adult and Middle Age models
    # for consistent cross-cohort comparison
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

    X = older[features].copy()
    y = older['insomnia_risk'].copy()

    print("\nFeature matrix shape:", X.shape)


    # --------------------------------------------------
    # Encode gender
    # --------------------------------------------------

    le = LabelEncoder()
    X['gender'] = le.fit_transform(X['gender'])
    print("\nGender encoding:", dict(zip(le.classes_, le.transform(le.classes_))))


    # --------------------------------------------------
    # Train / test split
    # stratify=y preserves 93/7 class ratio in both splits
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain size:", X_train.shape)
    print("Test size :", X_test.shape)
    print("\nTrain class distribution:")
    print(y_train.value_counts())
    print("\nTest class distribution:")
    print(y_test.value_counts())


    # --------------------------------------------------
    # SMOTE — train set only
    # ~93% no insomnia vs ~7% insomnia in this cohort
    # test set stays imbalanced — reflects real-world distribution
    # --------------------------------------------------

    print("\nBefore SMOTE:", dict(y_train.value_counts()))

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print("After  SMOTE:", dict(pd.Series(y_train_bal).value_counts()))
    print("Train shape after SMOTE:", X_train_bal.shape)


    # --------------------------------------------------
    # Standard scaling
    # fit on train only, transform both with same scaler
    # --------------------------------------------------

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled  = scaler.transform(X_test)


    # --------------------------------------------------
    # Random Forest
    #
    # Same hyperparameters across all three cohorts so
    # performance differences reflect the data, not the model config.
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST — INSOMNIA (OLDER ADULTS 55–79)")
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
    cv_scores = cross_val_score(
        rf, X_train_scaled, y_train_bal, cv=5, scoring='accuracy'
    )

    print("\n" + "=" * 60)
    print("RESULTS — INSOMNIA RISK | OLDER ADULTS (55–79)")
    print("=" * 60)
    print(f"  Test Accuracy       : {accuracy * 100:.2f}%")
    print(f"  ROC-AUC Score       : {roc_auc:.4f}")
    print(f"  5-Fold CV Accuracy  : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
    print()
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['No Insomnia', 'Insomnia Risk']
    ))


    # --------------------------------------------------
    # Cross-cohort comparison
    # --------------------------------------------------

    print("=" * 60)
    print("CROSS-COHORT COMPARISON — INSOMNIA RISK (RF MODEL)")
    print("=" * 60)
    print(f"{'Metric':<25} {'18–35':>10} {'36–55':>10} {'55–79':>10}")
    print("-" * 55)
    print(f"{'Test Accuracy':<25} {'93.43%':>10} {'91.85%':>10} {accuracy*100:>9.2f}%")
    print(f"{'ROC-AUC':<25} {'0.9583':>10} {'0.9194':>10} {roc_auc:>10.4f}")
    print(f"{'CV Accuracy':<25} {'95.85%':>10} {'95.67%':>10} {cv_scores.mean()*100:>9.2f}%")
    print(f"{'Cohort size':<25} {'5,708':>10} {'6,688':>10} {'7,923':>10}")
    print(f"{'Insomnia % (real)':<25} {'7.0%':>10} {'6.3%':>10} {'6.8%':>10}")
    print("=" * 60)


    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    feat_imp_df = pd.DataFrame({
        'feature'   : features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print("\nTop 10 Feature Importances:")
    print(feat_imp_df.head(10).to_string(index=False))


    # --------------------------------------------------
    # PLOT 1 — Feature Importance
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#8e44ad' if i < 3 else '#9b59b6' if i < 7 else '#95a5a6'
              for i in range(len(feat_imp_df))]

    ax.barh(
        feat_imp_df['feature'][::-1],
        feat_imp_df['importance'][::-1],
        color=colors[::-1]
    )
    ax.set_xlabel('Feature Importance (Gini)')
    ax.set_title(
        'Random Forest — Feature Importance\nInsomnia Risk | Older Adults (55–79)',
        fontweight='bold', pad=12
    )
    for i, (val, name) in enumerate(zip(
            feat_imp_df['importance'][::-1],
            feat_imp_df['feature'][::-1])):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('plots_older_adult/rf_insomnia_feature_importance.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Plot saved: rf_insomnia_feature_importance.png")


    # --------------------------------------------------
    # PLOT 2 — Confusion Matrix
    # --------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Purples',
        xticklabels=['No Insomnia', 'Insomnia Risk'],
        yticklabels=['No Insomnia', 'Insomnia Risk'],
        linewidths=0.5, linecolor='white', ax=ax
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(
        'Confusion Matrix\nInsomnia Risk | Older Adults (55–79)',
        fontweight='bold', pad=12
    )
    plt.tight_layout()
    plt.savefig('plots_older_adult/rf_insomnia_confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: rf_insomnia_confusion_matrix.png")


    # --------------------------------------------------
    # PLOT 3 — ROC Curve
    # --------------------------------------------------

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#8e44ad', lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#bdc3c7', lw=1.5,
            linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#8e44ad')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(
        'ROC Curve — Random Forest\nInsomnia Risk | Older Adults (55–79)',
        fontweight='bold', pad=12
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig('plots_older_adult/rf_insomnia_roc_curve.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: rf_insomnia_roc_curve.png")


    # --------------------------------------------------
    # PLOT 4 — Cross-Cohort Feature Importance Comparison
    # top 5 features for all three age cohorts side by side
    # --------------------------------------------------

    # feature importances from the two previous models
    young_imp = {
        'sleep_efficiency_pct'           : 0.3360,
        'stress_score'                   : 0.1660,
        'wake_after_sleep_onset_minutes' : 0.1620,
        'hrv_rmssd_ms'                   : 0.0680,
        'sleep_latency_minutes'          : 0.0580,
    }
    middle_imp = {
        'sleep_efficiency_pct'           : 0.3456,
        'wake_after_sleep_onset_minutes' : 0.1879,
        'stress_score'                   : 0.1488,
        'hrv_rmssd_ms'                   : 0.0713,
        'sleep_latency_minutes'          : 0.0601,
    }

    # top 5 from this model
    top5 = feat_imp_df['feature'].head(5).tolist()
    older_vals  = feat_imp_df.set_index('feature').loc[top5, 'importance'].values
    young_vals  = [young_imp.get(f, 0.0)  for f in top5]
    middle_vals = [middle_imp.get(f, 0.0) for f in top5]

    x     = np.arange(len(top5))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width,     young_vals,  width, label='Young Adult (18–35)',
           color='#3498db', alpha=0.85)
    ax.bar(x,             middle_vals, width, label='Middle Age (36–55)',
           color='#e67e22', alpha=0.85)
    ax.bar(x + width,     older_vals,  width, label='Older Adult (55–79)',
           color='#8e44ad', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(top5, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Feature Importance (Gini)')
    ax.set_title(
        'Feature Importance Across All Three Age Cohorts\nInsomnia Risk Prediction — Random Forest',
        fontweight='bold', pad=12
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('plots_older_adult/rf_all_cohorts_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: rf_all_cohorts_comparison.png")


    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Cohort              : Older Adults (Age 55–79)")
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
    print("All three cohort RF models for insomnia are now complete.")
    print("Next step: SHAP analysis across all three cohorts to compare")
    print("           how feature importance shifts with age.")

    # save model and scaler for SHAP
    with open("insomnia_rf_older_adult.pkl", "wb") as f:
        pickle.dump({'rf': rf, 'scaler': scaler}, f)
    print("Model saved: insomnia_rf_older_adult.pkl")


if __name__ == "__main__":
    train_insomnia_older_adult()