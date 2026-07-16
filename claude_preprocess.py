 # =============================================================================
# AgeWise: Explainable Age-Stratified ML Framework for Sleep Disorder Risk
# STEP 1 — Data Preprocessing
# =============================================================================
# This script corrects all issues present in baseline Kaggle preprocessing:
#   1. Removes label-leaking features (sleep_score, insomnia_flag, apnea_risk_score)
#   2. Properly remaps the broken daily_label before using it as target
#   3. Creates age cohort column — core of the research contribution
#   4. Handles outliers via IQR method
#   5. Applies SMOTE for class imbalance
#   6. Engineers research-relevant derived features
#   7. Saves separate X (features) and y (target) files
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Global plot style
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

PALETTE = {
    'good'        : '#2ecc71',
    'fair'        : '#f39c12',
    'poor'        : '#e74c3c',
    'Young_Adult' : '#3498db',
    'Middle_Aged' : '#9b59b6',
    'Older_Adult' : '#e67e22',
}

PLOT_DIR = "plots/"
import os
os.makedirs(PLOT_DIR, exist_ok=True)

# =============================================================================
# SECTION 1 — LOAD DATASET
# =============================================================================

print("=" * 70)
print("SECTION 1 — LOADING DATASET")
print("=" * 70)

DATA_PATH = "smartwatch_sleep_dataset.csv"
df = pd.read_csv(DATA_PATH)

print(f"Raw Dataset Shape     : {df.shape}")
print(f"Total Missing Values  : {df.isnull().sum().sum()}")
print(f"Duplicate Rows        : {df.duplicated().sum()}")

# =============================================================================
# SECTION 2 — MISSING VALUE ANALYSIS
# (Per ASHA methodology: median imputation for <5%, drop if >10%)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2 — MISSING VALUE ANALYSIS")
print("=" * 70)

missing_pct = (df.isnull().mean() * 100).round(2)
missing_report = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing %": missing_pct
}).sort_values("Missing %", ascending=False)

cols_above_10 = missing_report[missing_report["Missing %"] > 10].index.tolist()
cols_below_5  = missing_report[
    (missing_report["Missing %"] > 0) & (missing_report["Missing %"] <= 5)
].index.tolist()

print(f"Columns with >10% missing (DROP)  : {cols_above_10 if cols_above_10 else 'None'}")
print(f"Columns with <5% missing (IMPUTE) : {cols_below_5 if cols_below_5 else 'None'}")
print(f"Columns with 0% missing           : {(missing_pct == 0).sum()}")

# Drop columns with >10% missing
if cols_above_10:
    df.drop(columns=cols_above_10, inplace=True)
    print(f"Dropped {len(cols_above_10)} columns with >10% missing.")

# Median imputation for numeric columns with <5% missing
for col in cols_below_5:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("Median imputation applied for columns with <5% missing values.")
print(f"Remaining Missing Values: {df.isnull().sum().sum()}")

# =============================================================================
# SECTION 3 — DUPLICATE REMOVAL
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3 — DUPLICATE REMOVAL")
print("=" * 70)

dupes = df.duplicated().sum()
if dupes > 0:
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Removed {dupes} duplicate rows. New shape: {df.shape}")
else:
    print(f"No duplicates found. Shape unchanged: {df.shape}")

# =============================================================================
# SECTION 4 — LABEL REMAPPING
# (MUST happen BEFORE dropping sleep_score or daily_label)
#
# Problem: original daily_label has fair=18666, poor=1330, good=4
# This is broken and unusable for classification.
#
# Fix: Remap using sleep_score with clinically grounded thresholds:
#   score >= 65  → "good"
#   score 50-64  → "fair"
#   score < 50   → "poor"
#
# These thresholds align with clinical sleep scoring literature
# and produce a reasonably balanced 3-class distribution.
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4 — LABEL REMAPPING (Clinical Threshold-Based)")
print("=" * 70)

print("Original daily_label distribution:")
print(df['daily_label'].value_counts())
print(f"\nSleep score range: {df['sleep_score'].min()} – {df['sleep_score'].max()}")
print(f"Sleep score mean : {df['sleep_score'].mean():.2f}")
print(f"Sleep score std  : {df['sleep_score'].std():.2f}")

def remap_label(score):
    if score >= 65:
        return 'good'
    elif score >= 50:
        return 'fair'
    else:
        return 'poor'

df['sleep_quality'] = df['sleep_score'].apply(remap_label)

print("\nRemapped sleep_quality distribution:")
print(df['sleep_quality'].value_counts())
print("\nRemapped sleep_quality percentages:")
print(df['sleep_quality'].value_counts(normalize=True).round(3) * 100)

# =============================================================================
# SECTION 5 — DROP IDENTIFIER, TIMESTAMP, AND LABEL-LEAKING COLUMNS
#
# IDENTIFIERS (no predictive value):
#   user_id, date_recorded, sleep_start_timestamp,
#   sleep_end_timestamp, created_at
#
# ADMINISTRATIVE (irrelevant to physiology):
#   device_model, timezone
#
# LABEL LEAKERS (directly encode the target — cause artificial 100% accuracy):
#   sleep_score    — used to derive the label in Section 4
#   insomnia_flag  — IS a disorder label, not a predictor
#   apnea_risk_score — IS a disorder risk score, not a predictor
#   daily_label    — original broken label, replaced by sleep_quality
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5 — DROPPING IRRELEVANT AND LABEL-LEAKING COLUMNS")
print("=" * 70)

drop_identifiers = [
    'user_id', 'date_recorded', 'sleep_start_timestamp',
    'sleep_end_timestamp', 'created_at'
]

drop_administrative = [
    'device_model', 'timezone'
]

drop_leakers = [
    'sleep_score',       # used to derive label — leaks target
    'insomnia_flag',     # disorder label masquerading as feature
    'apnea_risk_score',  # disorder score masquerading as feature
    'daily_label'        # original broken label — replaced by sleep_quality
]

all_drop = drop_identifiers + drop_administrative + drop_leakers
existing_drop = [c for c in all_drop if c in df.columns]

df.drop(columns=existing_drop, inplace=True)

print("Dropped Columns:")
print(f"  Identifiers      : {[c for c in drop_identifiers if c in existing_drop]}")
print(f"  Administrative   : {[c for c in drop_administrative if c in existing_drop]}")
print(f"  Label Leakers    : {[c for c in drop_leakers if c in existing_drop]}")
print(f"\nDataset shape after dropping: {df.shape}")

# =============================================================================
# SECTION 6 — AGE COHORT CREATION
# (Core contribution of AgeWise — must be created BEFORE any scaling)
#
# Cohort boundaries based on sleep physiology literature:
#   Young Adults  : 18–35  (circadian phase delay, high REM dependency)
#   Middle-Aged   : 36–55  (deep sleep decline begins, stress patterns shift)
#   Older Adults  : 56–79  (fragmented sleep, reduced HRV, apnea risk rises)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6 — AGE COHORT CREATION")
print("=" * 70)

def assign_cohort(age):
    if age <= 35:
        return 'Young_Adult'
    elif age <= 55:
        return 'Middle_Aged'
    else:
        return 'Older_Adult'

df['age_cohort'] = df['age'].apply(assign_cohort)

print("Age cohort distribution:")
print(df['age_cohort'].value_counts())
print("\nAge cohort percentages:")
print(df['age_cohort'].value_counts(normalize=True).round(3) * 100)

print("\nAge statistics per cohort:")
print(df.groupby('age_cohort')['age'].describe().round(2))

# =============================================================================
# SECTION 7 — OUTLIER DETECTION AND HANDLING (IQR METHOD)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7 — OUTLIER HANDLING (IQR Method)")
print("=" * 70)

# Only apply to physiological numeric columns — not binary flags
outlier_cols = [
    'duration_minutes', 'sleep_latency_minutes', 'wake_after_sleep_onset_minutes',
    'sleep_efficiency_pct', 'sleep_stage_deep_pct', 'sleep_stage_light_pct',
    'sleep_stage_rem_pct', 'sleep_stage_awake_pct', 'heart_rate_mean_bpm',
    'heart_rate_min_bpm', 'heart_rate_max_bpm', 'hrv_rmssd_ms',
    'respiration_rate_bpm', 'spo2_mean_pct', 'spo2_min_pct',
    'movement_count', 'ambient_noise_db', 'room_temperature_c',
    'room_humidity_pct', 'step_count_day', 'caffeine_mg', 'alcohol_units',
    'bedtime_consistency_std_min', 'stress_score', 'activity_before_bed_min',
    'screen_time_before_bed_min', 'nap_duration_minutes', 'weight_kg', 'height_cm'
]

outlier_cols = [c for c in outlier_cols if c in df.columns]

total_outliers = 0
outlier_report = []

for col in outlier_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    total_outliers += n_outliers

    # Cap outliers at boundary values (Winsorization)
    df[col] = df[col].clip(lower=lower, upper=upper)

    outlier_report.append({
        'Column': col,
        'Q1': round(Q1, 2), 'Q3': round(Q3, 2),
        'Lower Bound': round(lower, 2), 'Upper Bound': round(upper, 2),
        'Outliers Capped': n_outliers
    })

outlier_df = pd.DataFrame(outlier_report)
outlier_df_nonzero = outlier_df[outlier_df['Outliers Capped'] > 0]

print(f"Total outlier values capped (Winsorization): {total_outliers}")
print(f"Columns with outliers:")
print(outlier_df_nonzero[['Column', 'Lower Bound', 'Upper Bound', 'Outliers Capped']].to_string(index=False))

# =============================================================================
# SECTION 8 — FEATURE ENGINEERING
# (Derived features aligned with ASHA paper + AgeWise research contributions)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 8 — FEATURE ENGINEERING")
print("=" * 70)

# 8a. Sleep Architecture Metrics
df['deep_rem_ratio'] = (
    df['sleep_stage_deep_pct'] / (df['sleep_stage_rem_pct'] + 1e-6)
)
df['restorative_sleep_pct'] = (
    df['sleep_stage_deep_pct'] + df['sleep_stage_rem_pct']
)
df['fragmentation_index'] = (
    df['sleep_stage_awake_pct'] + df['wake_after_sleep_onset_minutes'] / 60
)

# 8b. Cardiovascular Indicators
df['recovery_index'] = (
    df['hrv_rmssd_ms'] * df['sleep_efficiency_pct'] / 100
)
df['hr_range'] = df['heart_rate_max_bpm'] - df['heart_rate_min_bpm']
df['hrv_sleep_duration_interaction'] = (
    df['hrv_rmssd_ms'] * (df['duration_minutes'] / 60)
)

# 8c. Activity-Sleep Interactions
df['total_disruption_min'] = (
    df['sleep_latency_minutes'] + df['wake_after_sleep_onset_minutes']
)

# 8d. Lifestyle Load
df['stimulant_load'] = df['caffeine_mg'] + df['alcohol_units'] * 14

print("Engineered Features Created:")
eng_features = [
    'deep_rem_ratio', 'restorative_sleep_pct', 'fragmentation_index',
    'recovery_index', 'hr_range', 'hrv_sleep_duration_interaction',
    'total_disruption_min', 'stimulant_load'
]
for f in eng_features:
    print(f"  + {f}")

# =============================================================================
# SECTION 9 — ENCODE CATEGORICAL VARIABLES
# (Using separate encoders per column — NOT single reused encoder)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 9 — ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c not in ['sleep_quality', 'age_cohort']]

print(f"Categorical columns to encode: {cat_cols}")

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"  Encoded: {col} → {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Encode age_cohort as ordinal (preserve order for analysis)
cohort_map = {'Young_Adult': 0, 'Middle_Aged': 1, 'Older_Adult': 2}
df['age_cohort_encoded'] = df['age_cohort'].map(cohort_map)
print(f"  Encoded: age_cohort → {cohort_map}")

# =============================================================================
# SECTION 10 — SEPARATE FEATURES AND TARGET
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 10 — SEPARATING FEATURES AND TARGET")
print("=" * 70)

TARGET = 'sleep_quality'

# Keep age_cohort string column for cohort-based splitting later
# Keep age_cohort_encoded as numeric for models
exclude_from_X = [TARGET, 'age_cohort']

X = df[[c for c in df.columns if c not in exclude_from_X]].copy()
y = df[TARGET].copy()
cohort = df['age_cohort'].copy()   # for stratified training later

print(f"Feature matrix X shape : {X.shape}")
print(f"Target vector y shape  : {y.shape}")
print(f"\nTarget distribution (remapped):")
print(y.value_counts())
print(y.value_counts(normalize=True).round(3) * 100)

print(f"\nFeature columns ({len(X.columns)}):")
for c in X.columns:
    print(f"  {c}")

# =============================================================================
# SECTION 11 — CLASS IMBALANCE HANDLING (SMOTE)
# Note: SMOTE applied to overall dataset here for baseline model.
# For age-stratified models, SMOTE is applied PER COHORT separately.
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 11 — CLASS IMBALANCE HANDLING (SMOTE)")
print("=" * 70)

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print(f"Before SMOTE:")
unique, counts = np.unique(y_encoded, return_counts=True)
for cls, cnt in zip(le_target.classes_, counts):
    print(f"  {cls}: {cnt} ({cnt/len(y_encoded)*100:.1f}%)")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

print(f"\nAfter SMOTE:")
unique_r, counts_r = np.unique(y_resampled, return_counts=True)
for cls, cnt in zip(le_target.classes_, counts_r):
    print(f"  {cls}: {cnt} ({cnt/len(y_resampled)*100:.1f}%)")

X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
y_resampled_series = pd.Series(
    le_target.inverse_transform(y_resampled), name='sleep_quality'
)

# =============================================================================
# SECTION 12 — EDA AND PREPROCESSING VALIDATION PLOTS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 12 — GENERATING EDA PLOTS")
print("=" * 70)

# -----------------------------------------------------------------------------
# PLOT 1 — Target Label Distribution: Before vs After Remapping
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Plot 1: Target Label Distribution — Before vs After Remapping',
             fontsize=14, fontweight='bold', y=1.02)

# Before remapping — original daily_label (reconstructed from sleep_score ranges)
original_counts = {'fair': 18666, 'poor': 1330, 'good': 4}
axes[0].bar(original_counts.keys(),
            original_counts.values(),
            color=[PALETTE['fair'], PALETTE['poor'], PALETTE['good']],
            edgecolor='white', linewidth=1.5, width=0.5)
axes[0].set_title('Original daily_label\n(Broken — good=4 unusable)')
axes[0].set_ylabel('Number of Records')
axes[0].set_xlabel('Sleep Quality Class')
for i, (k, v) in enumerate(original_counts.items()):
    axes[0].text(i, v + 100, f'{v:,}\n({v/20000*100:.1f}%)',
                 ha='center', fontsize=10, fontweight='bold')

# After remapping — sleep_quality
remap_counts = y.value_counts()
colors_remap = [PALETTE.get(c, '#999') for c in remap_counts.index]
axes[1].bar(remap_counts.index, remap_counts.values,
            color=colors_remap, edgecolor='white', linewidth=1.5, width=0.5)
axes[1].set_title('Remapped sleep_quality\n(Clinically grounded thresholds)')
axes[1].set_ylabel('Number of Records')
axes[1].set_xlabel('Sleep Quality Class')
for i, (cls, cnt) in enumerate(remap_counts.items()):
    axes[1].text(i, cnt + 100, f'{cnt:,}\n({cnt/len(y)*100:.1f}%)',
                 ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}plot1_label_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot 1 saved: plot1_label_distribution.png")

# -----------------------------------------------------------------------------
# PLOT 2 — Age Distribution with Cohort Boundaries
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 5))

# Shade cohort regions
ax.axvspan(18, 35, alpha=0.12, color=PALETTE['Young_Adult'], label='Young Adult (18–35)')
ax.axvspan(35, 55, alpha=0.12, color=PALETTE['Middle_Aged'], label='Middle-Aged (36–55)')
ax.axvspan(55, 80, alpha=0.12, color=PALETTE['Older_Adult'], label='Older Adult (56–79)')

# Histogram
ax.hist(df['age'], bins=62, color='#2c3e50', edgecolor='white',
        linewidth=0.5, alpha=0.85)

# Boundary lines
for x, label in [(35, 'Age 35'), (55, 'Age 55')]:
    ax.axvline(x, color='#c0392b', linestyle='--', linewidth=2)
    ax.text(x + 0.5, ax.get_ylim()[1] * 0.92, label,
            color='#c0392b', fontsize=10, fontweight='bold')

# Cohort sample size annotations
cohort_counts = df['age_cohort'].value_counts()
for cohort_name, x_center, color in [
    ('Young_Adult',  26, PALETTE['Young_Adult']),
    ('Middle_Aged',  45, PALETTE['Middle_Aged']),
    ('Older_Adult',  67, PALETTE['Older_Adult']),
]:
    n = cohort_counts.get(cohort_name, 0)
    ax.text(x_center, ax.get_ylim()[1] * 0.75,
            f'n = {n:,}', ha='center', fontsize=11,
            fontweight='bold', color=color)

ax.set_xlabel('Age (years)')
ax.set_ylabel('Number of Records')
ax.set_title('Plot 2: Age Distribution with Cohort Boundaries\n'
             'AgeWise Stratification: Young Adult | Middle-Aged | Older Adult',
             fontweight='bold')
ax.legend(loc='upper right', framealpha=0.8)
ax.set_xlim(17, 80)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}plot2_age_cohort_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot 2 saved: plot2_age_cohort_distribution.png")

# -----------------------------------------------------------------------------
# PLOT 3 — Class Distribution Per Age Cohort
# -----------------------------------------------------------------------------

cohort_label_df = df[['age_cohort', 'sleep_quality']].copy()
cohort_order    = ['Young_Adult', 'Middle_Aged', 'Older_Adult']
class_order     = ['good', 'fair', 'poor']

fig, ax = plt.subplots(figsize=(12, 5))

x        = np.arange(len(cohort_order))
width    = 0.25
offsets  = [-width, 0, width]

for i, cls in enumerate(class_order):
    counts = [
        cohort_label_df[
            (cohort_label_df['age_cohort'] == c) &
            (cohort_label_df['sleep_quality'] == cls)
        ].shape[0]
        for c in cohort_order
    ]
    bars = ax.bar(x + offsets[i], counts, width,
                  label=cls.capitalize(), color=PALETTE[cls],
                  edgecolor='white', linewidth=1.2)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 20,
                str(cnt), ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(['Young Adult\n(18–35)', 'Middle-Aged\n(36–55)', 'Older Adult\n(56–79)'],
                   fontsize=11)
ax.set_ylabel('Number of Records')
ax.set_title('Plot 3: Sleep Quality Class Distribution Per Age Cohort\n'
             'Justifies per-cohort SMOTE application', fontweight='bold')
ax.legend(title='Sleep Quality', framealpha=0.8)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}plot3_class_per_cohort.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot 3 saved: plot3_class_per_cohort.png")

# -----------------------------------------------------------------------------
# PLOT 4 — Before vs After SMOTE Class Distribution
# -----------------------------------------------------------------------------

before_counts = y.value_counts().reindex(class_order)
after_counts  = y_resampled_series.value_counts().reindex(class_order)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Plot 4: Class Distribution — Before vs After SMOTE',
             fontsize=14, fontweight='bold', y=1.02)

for ax, counts, title in [
    (axes[0], before_counts, 'Before SMOTE\n(Imbalanced)'),
    (axes[1], after_counts,  'After SMOTE\n(Balanced)')
]:
    colors = [PALETTE[c] for c in class_order]
    bars   = ax.bar(class_order, counts.values,
                    color=colors, edgecolor='white', linewidth=1.5, width=0.5)
    ax.set_title(title)
    ax.set_ylabel('Number of Records')
    ax.set_xlabel('Sleep Quality Class')
    total = counts.sum()
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + total * 0.005,
                f'{cnt:,}\n({cnt/total*100:.1f}%)',
                ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}plot4_smote_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot 4 saved: plot4_smote_comparison.png")

# -----------------------------------------------------------------------------
# PLOT 5 — Feature Correlation Heatmap (post preprocessing)
# -----------------------------------------------------------------------------

# Select key physiological + engineered features for readability
heatmap_features = [
    'duration_minutes', 'sleep_efficiency_pct',
    'sleep_stage_deep_pct', 'sleep_stage_rem_pct',
    'sleep_stage_awake_pct', 'hrv_rmssd_ms',
    'heart_rate_mean_bpm', 'respiration_rate_bpm',
    'spo2_mean_pct', 'stress_score',
    'deep_rem_ratio', 'restorative_sleep_pct',
    'fragmentation_index', 'recovery_index',
    'total_disruption_min', 'stimulant_load'
]
heatmap_features = [f for f in heatmap_features if f in X.columns]

corr_matrix = X[heatmap_features].corr()

fig, ax = plt.subplots(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f',
    cmap='RdBu_r', center=0, square=True,
    linewidths=0.4, linecolor='white',
    cbar_kws={'shrink': 0.7, 'label': 'Pearson Correlation'},
    ax=ax, annot_kws={'size': 8}
)
ax.set_title('Plot 5: Feature Correlation Heatmap\n'
             'Key Physiological + Engineered Features (Post Preprocessing)',
             fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}plot5_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot 5 saved: plot5_correlation_heatmap.png")

# -----------------------------------------------------------------------------
# PLOT 6 — Key Physiological Features by Age Cohort (Box Plots)
# The most important plot — proves age stratification is empirically justified
# -----------------------------------------------------------------------------

key_features = {
    'hrv_rmssd_ms'         : 'HRV — RMSSD (ms)',
    'sleep_stage_deep_pct' : 'Deep Sleep (%)',
    'sleep_stage_rem_pct'  : 'REM Sleep (%)',
    'sleep_efficiency_pct' : 'Sleep Efficiency (%)',
    'fragmentation_index'  : 'Fragmentation Index',
    'recovery_index'       : 'Recovery Index',
}

cohort_plot_df         = X[list(key_features.keys())].copy()
cohort_plot_df['cohort'] = cohort.values
cohort_colors          = [PALETTE[c] for c in cohort_order]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, (feat, label) in enumerate(key_features.items()):
    ax = axes[i]
    data_by_cohort = [
        cohort_plot_df[cohort_plot_df['cohort'] == c][feat].dropna()
        for c in cohort_order
    ]
    bp = ax.boxplot(
        data_by_cohort,
        labels=['Young\nAdult', 'Middle\nAged', 'Older\nAdult'],
        patch_artist=True,
        medianprops={'color': 'white', 'linewidth': 2.5},
        flierprops={'marker': 'o', 'markerfacecolor': '#bdc3c7',
                    'markersize': 3, 'alpha': 0.5},
        whiskerprops={'linewidth': 1.5},
        capprops={'linewidth': 2}
    )
    for patch, color in zip(bp['boxes'], cohort_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Annotate medians
    for j, data in enumerate(data_by_cohort):
        median = data.median()
        ax.text(j + 1, median, f'{median:.1f}',
                ha='center', va='bottom', fontsize=9,
                fontweight='bold', color='#2c3e50')

    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('Age Cohort')

fig.suptitle(
    'Plot 6: Key Physiological Features by Age Cohort\n'
    'Empirical evidence that sleep physiology differs significantly across age groups',
    fontsize=14, fontweight='bold', y=1.01
)

legend_patches = [
    mpatches.Patch(color=PALETTE['Young_Adult'], alpha=0.75, label='Young Adult (18–35)'),
    mpatches.Patch(color=PALETTE['Middle_Aged'], alpha=0.75, label='Middle-Aged (36–55)'),
    mpatches.Patch(color=PALETTE['Older_Adult'], alpha=0.75, label='Older Adult (56–79)'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3,
           framealpha=0.9, fontsize=11, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}plot6_features_by_cohort.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot 6 saved: plot6_features_by_cohort.png")

# -----------------------------------------------------------------------------
# PLOT 7 — Outlier Detection: Before vs After Winsorization
# (shown for 4 most clinically relevant features)
# -----------------------------------------------------------------------------

# Reload raw values for "before" comparison
df_raw = pd.read_csv(DATA_PATH)

outlier_vis_features = {
    'duration_minutes'             : 'Sleep Duration (min)',
    'hrv_rmssd_ms'                 : 'HRV — RMSSD (ms)',
    'sleep_latency_minutes'        : 'Sleep Latency (min)',
    'wake_after_sleep_onset_minutes': 'WASO (min)',
}

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle('Plot 7: Outlier Detection — Before vs After IQR Winsorization',
             fontsize=14, fontweight='bold', y=1.02)

for col_idx, (feat, label) in enumerate(outlier_vis_features.items()):
    if feat not in df_raw.columns or feat not in X.columns:
        continue

    raw_data    = df_raw[feat].dropna()
    cleaned_data = X[feat].dropna()

    # Before
    ax_before = axes[0][col_idx]
    ax_before.boxplot(raw_data, patch_artist=True,
                      boxprops={'facecolor': '#e74c3c', 'alpha': 0.6},
                      medianprops={'color': 'white', 'linewidth': 2},
                      flierprops={'marker': 'o', 'markerfacecolor': '#c0392b',
                                  'markersize': 3, 'alpha': 0.4})
    ax_before.set_title(f'{label}\n(Before)', fontsize=10)
    ax_before.set_ylabel('Value' if col_idx == 0 else '')

    # After
    ax_after = axes[1][col_idx]
    ax_after.boxplot(cleaned_data, patch_artist=True,
                     boxprops={'facecolor': '#2ecc71', 'alpha': 0.6},
                     medianprops={'color': 'white', 'linewidth': 2},
                     flierprops={'marker': 'o', 'markerfacecolor': '#27ae60',
                                 'markersize': 3, 'alpha': 0.4})
    ax_after.set_title(f'{label}\n(After)', fontsize=10)
    ax_after.set_ylabel('Value' if col_idx == 0 else '')

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}plot7_outlier_before_after.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot 7 saved: plot7_outlier_before_after.png")

print(f"\n  All 7 plots saved to '{PLOT_DIR}' directory.")

# =============================================================================
# SECTION 13 — FINAL SUMMARY AND SAVE
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 13 — FINAL SUMMARY")
print("=" * 70)

print(f"Original dataset shape         : (20000, 45)")
print(f"After cleaning shape           : {df.shape}")
print(f"After SMOTE shape (X)          : {X_resampled_df.shape}")
print(f"After SMOTE shape (y)          : {y_resampled_series.shape}")
print(f"Features retained              : {len(X.columns)}")
print(f"Label-leaking features removed : sleep_score, insomnia_flag, apnea_risk_score")
print(f"Age cohorts created            : Young_Adult, Middle_Aged, Older_Adult")
print(f"Engineered features added      : {len(eng_features)}")
print(f"Outlier method                 : IQR Winsorization")
print(f"Imbalance method               : SMOTE")
print(f"Target variable                : sleep_quality (remapped from sleep_score)")

# Save files
df.to_csv("agewise_cleaned_full.csv", index=False)
X_resampled_df.to_csv("agewise_X_smote.csv", index=False)
y_resampled_series.to_csv("agewise_y_smote.csv", index=False)

# Also save pre-SMOTE for cohort-level analysis
X.to_csv("agewise_X_raw.csv", index=False)
y.to_csv("agewise_y_raw.csv", index=False)
cohort.to_csv("agewise_cohort.csv", index=False)

print("\nFiles saved:")
print("  agewise_cleaned_full.csv  — full cleaned dataset with all columns")
print("  agewise_X_smote.csv       — features after SMOTE (use for baseline RF)")
print("  agewise_y_smote.csv       — target after SMOTE")
print("  agewise_X_raw.csv         — features pre-SMOTE (use for cohort splitting)")
print("  agewise_y_raw.csv         — target pre-SMOTE")
print("  agewise_cohort.csv        — cohort labels for stratified model training")

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)
