# ================================================================
# STEP 14 - CIC-IDS2018 FEATURE SELECTION
# ================================================================
#
# Dataset:
#   CIC-IDS2018
#
# Input:
#   CICIDS2018_train.csv
#
# Important:
#   ONLY TRAINING DATA is used for feature selection.
#   Validation and Test data are NOT used.
#
# Feature Selection:
#   1. Remove non-numeric features
#   2. Remove constant features
#   3. Remove highly correlated features
#   4. Mutual Information
#   5. Random Forest importance
#   6. Combined ranking
#   7. Select final 30 features
#
# Output:
#   experiments/step14_cicids2018_selected_features.csv
#   experiments/step14_cicids2018_feature_scores.csv
#   experiments/step14_cicids2018_summary.json
#
# ================================================================

import os
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "data",
    "modeling",
    "CICIDS2018_train.csv"
)

EXPERIMENT_DIR = os.path.join(
    BASE_DIR,
    "experiments"
)

SELECTED_FEATURE_FILE = os.path.join(
    EXPERIMENT_DIR,
    "step14_cicids2018_selected_features.csv"
)

FEATURE_SCORE_FILE = os.path.join(
    EXPERIMENT_DIR,
    "step14_cicids2018_feature_scores.csv"
)

SUMMARY_FILE = os.path.join(
    EXPERIMENT_DIR,
    "step14_cicids2018_summary.json"
)

TARGET_COLUMN = "Label"

FINAL_FEATURE_COUNT = 30

# Number of samples used for expensive feature-selection methods.
# The complete training dataset is NOT needed for MI/RF ranking.
SAMPLE_SIZE = 200_000

RANDOM_STATE = 42


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("STEP 14 - CIC-IDS2018 FEATURE SELECTION")
print("=" * 70)

print()
print("Dataset:")
print("CIC-IDS2018")

print()
print("Input:")
print("CICIDS2018_train.csv")

print()
print("IMPORTANT:")
print("ONLY TRAINING DATA will be used.")
print("Validation and Test data will NOT be used.")

print()
print("Feature selection methods:")
print("- Numeric feature validation")
print("- Constant feature removal")
print("- Correlation analysis")
print("- Mutual Information")
print("- Random Forest importance")
print("- Combined ranking")

print()
print("Final target:")
print(f"{FINAL_FEATURE_COUNT} features")

print("=" * 70)


# ================================================================
# CREATE EXPERIMENT DIRECTORY
# ================================================================

os.makedirs(EXPERIMENT_DIR, exist_ok=True)


# ================================================================
# CHECK INPUT FILE
# ================================================================

print()
print("=" * 70)
print("CHECKING INPUT FILE")
print("=" * 70)

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

print("[OK] Training file found:")
print(TRAIN_FILE)


# ================================================================
# LOAD TRAINING DATA
# ================================================================

print()
print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

start_time = time.time()

print("Loading CICIDS2018_train.csv...")
print("This may take some time because the dataset is large.")

df = pd.read_csv(TRAIN_FILE, low_memory=False)

load_time = time.time() - start_time

print()
print("[OK] Training dataset loaded.")
print(f"Shape: {df.shape}")
print(f"Loading time: {load_time:.2f} seconds")


# ================================================================
# BASIC VALIDATION
# ================================================================

print()
print("=" * 70)
print("BASIC DATA VALIDATION")
print("=" * 70)

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found."
    )

print(f"[OK] Target column found: {TARGET_COLUMN}")

print(f"Training rows    : {len(df):,}")
print(f"Training columns : {len(df.columns)}")
print(f"Total classes    : {df[TARGET_COLUMN].nunique()}")


# ================================================================
# TARGET INFORMATION
# ================================================================

print()
print("=" * 70)
print("TARGET INFORMATION")
print("=" * 70)

print("Classes:")

classes = sorted(df[TARGET_COLUMN].astype(str).unique())

for i, cls in enumerate(classes):
    count = (df[TARGET_COLUMN].astype(str) == cls).sum()
    print(f"{i:02d}. {cls:<35} : {count:,}")


# ================================================================
# IDENTIFY NUMERIC FEATURES
# ================================================================

print()
print("=" * 70)
print("IDENTIFYING NUMERIC FEATURES")
print("=" * 70)

feature_columns = [
    col for col in df.columns
    if col != TARGET_COLUMN
]

print(f"Total candidate features: {len(feature_columns)}")

numeric_features = []
non_numeric_features = []

for col in feature_columns:

    converted = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    # If at least 95% of values can be converted,
    # consider it a numeric feature.
    valid_ratio = converted.notna().mean()

    if valid_ratio >= 0.95:
        numeric_features.append(col)
    else:
        non_numeric_features.append(col)

print()
print(f"Numeric features     : {len(numeric_features)}")
print(f"Non-numeric features : {len(non_numeric_features)}")

if non_numeric_features:

    print()
    print("Non-numeric features removed:")

    for col in non_numeric_features:
        print(f" - {col}")


# ================================================================
# CONVERT NUMERIC FEATURES
# ================================================================

print()
print("=" * 70)
print("CONVERTING NUMERIC FEATURES")
print("=" * 70)

X = df[numeric_features].apply(
    pd.to_numeric,
    errors="coerce"
)

# Replace infinity values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Replace missing numeric values with 0.
X = X.fillna(0)

print("[OK] Numeric feature matrix prepared.")
print(f"Feature matrix shape: {X.shape}")


# ================================================================
# REMOVE CONSTANT FEATURES
# ================================================================

print()
print("=" * 70)
print("REMOVING CONSTANT FEATURES")
print("=" * 70)

constant_features = []

for col in X.columns:

    if X[col].nunique(dropna=False) <= 1:
        constant_features.append(col)

print(f"Constant features found: {len(constant_features)}")

if constant_features:

    for col in constant_features:
        print(f" - {col}")

    X = X.drop(
        columns=constant_features
    )

print()
print(f"Remaining features: {X.shape[1]}")


# ================================================================
# REMOVE HIGHLY CORRELATED FEATURES
# ================================================================

print()
print("=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

print("Calculating feature correlations...")

start_corr = time.time()

correlation_matrix = X.corr(
    method="spearman"
).abs()

upper_triangle = correlation_matrix.where(
    np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)
)

CORRELATION_THRESHOLD = 0.95

high_corr_features = []

for column in upper_triangle.columns:

    if any(
        upper_triangle[column] > CORRELATION_THRESHOLD
    ):
        high_corr_features.append(column)

correlation_time = time.time() - start_corr

print(
    f"Correlation analysis completed in "
    f"{correlation_time:.2f} seconds."
)

print()
print(
    f"Highly correlated features "
    f"(>{CORRELATION_THRESHOLD}): "
    f"{len(high_corr_features)}"
)

if high_corr_features:

    print()
    print("Removing highly correlated features:")

    for col in high_corr_features:
        print(f" - {col}")

    X = X.drop(
        columns=high_corr_features
    )

print()
print(f"Remaining features: {X.shape[1]}")


# ================================================================
# PREPARE TARGET
# ================================================================

print()
print("=" * 70)
print("PREPARING TARGET")
print("=" * 70)

y = df[TARGET_COLUMN].astype(str)

print(f"Number of classes: {y.nunique()}")

# Encode target
class_mapping = {
    cls: idx
    for idx, cls in enumerate(
        sorted(y.unique())
    )
}

y_encoded = y.map(class_mapping).astype(np.int32)

print("[OK] Target encoded.")


# ================================================================
# CREATE FEATURE-SELECTION SAMPLE
# ================================================================

print()
print("=" * 70)
print("CREATING FEATURE SELECTION SAMPLE")
print("=" * 70)

actual_sample_size = min(
    SAMPLE_SIZE,
    len(X)
)

print(
    f"Requested sample size: "
    f"{SAMPLE_SIZE:,}"
)

print(
    f"Actual sample size   : "
    f"{actual_sample_size:,}"
)

print()
print("Creating stratified sample...")

# Use a random sample first.
# The dataset is very large, so this keeps computation practical.
sample_indices = (
    pd.Series(
        np.arange(len(X))
    )
    .sample(
        n=actual_sample_size,
        random_state=RANDOM_STATE
    )
    .values
)

X_sample = X.iloc[sample_indices].copy()
y_sample = y_encoded.iloc[sample_indices].copy()

print()
print("[OK] Feature selection sample created.")
print(f"Sample shape: {X_sample.shape}")


# ================================================================
# MUTUAL INFORMATION
# ================================================================

print()
print("=" * 70)
print("MUTUAL INFORMATION FEATURE SELECTION")
print("=" * 70)

print(
    "Calculating Mutual Information scores..."
)

start_mi = time.time()

mi_scores = mutual_info_classif(
    X_sample,
    y_sample,
    random_state=RANDOM_STATE,
    n_neighbors=3
)

mi_time = time.time() - start_mi

mi_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "MI_Score": mi_scores
})

mi_df = mi_df.sort_values(
    "MI_Score",
    ascending=False
).reset_index(drop=True)

mi_df["MI_Rank"] = (
    np.arange(len(mi_df)) + 1
)

print()
print(
    f"Mutual Information completed in "
    f"{mi_time:.2f} seconds."
)

print()
print("Top 20 features by Mutual Information:")

for i, row in mi_df.head(20).iterrows():

    print(
        f"{i + 1:02d}. "
        f"{row['Feature']:<35} "
        f"{row['MI_Score']:.6f}"
    )


# ================================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ================================================================

print()
print("=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

print(
    "Training Random Forest for feature importance..."
)

start_rf = time.time()

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=2
)

rf_model.fit(
    X_sample,
    y_sample
)

rf_time = time.time() - start_rf

rf_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "RF_Importance": rf_model.feature_importances_
})

rf_df = rf_df.sort_values(
    "RF_Importance",
    ascending=False
).reset_index(drop=True)

rf_df["RF_Rank"] = (
    np.arange(len(rf_df)) + 1
)

print()
print(
    f"Random Forest completed in "
    f"{rf_time:.2f} seconds."
)

print()
print("Top 20 features by Random Forest importance:")

for i, row in rf_df.head(20).iterrows():

    print(
        f"{i + 1:02d}. "
        f"{row['Feature']:<35} "
        f"{row['RF_Importance']:.6f}"
    )


# ================================================================
# COMBINE FEATURE SCORES
# ================================================================

print()
print("=" * 70)
print("COMBINING FEATURE RANKINGS")
print("=" * 70)

scores = pd.merge(
    mi_df,
    rf_df,
    on="Feature",
    how="inner"
)

# Normalize MI scores
mi_min = scores["MI_Score"].min()
mi_max = scores["MI_Score"].max()

if mi_max > mi_min:

    scores["MI_Normalized"] = (
        (scores["MI_Score"] - mi_min)
        /
        (mi_max - mi_min)
    )

else:

    scores["MI_Normalized"] = 0.0


# Normalize RF importance
rf_min = scores["RF_Importance"].min()
rf_max = scores["RF_Importance"].max()

if rf_max > rf_min:

    scores["RF_Normalized"] = (
        (scores["RF_Importance"] - rf_min)
        /
        (rf_max - rf_min)
    )

else:

    scores["RF_Normalized"] = 0.0


# Equal weighting
scores["Combined_Score"] = (
    0.5 * scores["MI_Normalized"]
    +
    0.5 * scores["RF_Normalized"]
)

scores = scores.sort_values(
    "Combined_Score",
    ascending=False
).reset_index(drop=True)

scores["Combined_Rank"] = (
    np.arange(len(scores)) + 1
)


# ================================================================
# FINAL FEATURE SELECTION
# ================================================================

print()
print("=" * 70)
print("FINAL FEATURE SELECTION")
print("=" * 70)

if len(scores) < FINAL_FEATURE_COUNT:

    raise ValueError(
        f"Only {len(scores)} usable features available. "
        f"Cannot select {FINAL_FEATURE_COUNT}."
    )

selected_features = (
    scores
    .head(FINAL_FEATURE_COUNT)
    .copy()
)

print()
print(
    f"Final selected feature count: "
    f"{len(selected_features)}"
)

print()
print("Selected features:")

for i, row in selected_features.iterrows():

    print(
        f"{i + 1:02d}. "
        f"{row['Feature']}"
    )


# ================================================================
# SAVE SELECTED FEATURES
# ================================================================

print()
print("=" * 70)
print("SAVING SELECTED FEATURES")
print("=" * 70)

selected_output = selected_features[
    [
        "Combined_Rank",
        "Feature",
        "MI_Score",
        "MI_Rank",
        "RF_Importance",
        "RF_Rank",
        "Combined_Score"
    ]
]

selected_output.to_csv(
    SELECTED_FEATURE_FILE,
    index=False
)

print("[OK] Selected feature file saved:")
print(SELECTED_FEATURE_FILE)


# ================================================================
# SAVE ALL FEATURE SCORES
# ================================================================

scores_output = scores[
    [
        "Combined_Rank",
        "Feature",
        "MI_Score",
        "MI_Rank",
        "RF_Importance",
        "RF_Rank",
        "MI_Normalized",
        "RF_Normalized",
        "Combined_Score"
    ]
]

scores_output.to_csv(
    FEATURE_SCORE_FILE,
    index=False
)

print()
print("[OK] Complete feature score file saved:")
print(FEATURE_SCORE_FILE)


# ================================================================
# SAVE SUMMARY
# ================================================================

summary = {
    "step": 14,
    "dataset": "CIC-IDS2018",
    "input": "CICIDS2018_train.csv",
    "training_rows": int(len(df)),
    "original_columns": int(len(df.columns)),
    "target_column": TARGET_COLUMN,
    "number_of_classes": int(y.nunique()),
    "numeric_features_before_cleaning": int(len(numeric_features)),
    "non_numeric_features_removed": non_numeric_features,
    "constant_features_removed": constant_features,
    "high_correlation_features_removed": high_corr_features,
    "correlation_threshold": CORRELATION_THRESHOLD,
    "features_after_filtering": int(X.shape[1]),
    "feature_selection_sample_size": int(actual_sample_size),
    "mutual_information_used": True,
    "random_forest_used": True,
    "random_forest_estimators": 200,
    "final_feature_count": int(len(selected_features)),
    "selected_features": selected_features[
        "Feature"
    ].tolist(),
    "random_state": RANDOM_STATE,
    "validation_used": False,
    "test_used": False,
    "smote_used": False,
    "model_training_for_final_ids": False,
    "outputs": {
        "selected_features": SELECTED_FEATURE_FILE,
        "feature_scores": FEATURE_SCORE_FILE,
        "summary": SUMMARY_FILE
    }
}

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )

print()
print("[OK] Summary saved:")
print(SUMMARY_FILE)


# ================================================================
# FINAL SUMMARY
# ================================================================

total_time = time.time() - start_time

print()
print("=" * 70)
print("STEP 14 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("Dataset:")
print("CIC-IDS2018")

print()
print("Feature selection data:")
print("TRAINING ONLY")

print()
print(f"Original feature count : {len(feature_columns)}")
print(f"Numeric features       : {len(numeric_features)}")
print(
    f"Constant removed       : "
    f"{len(constant_features)}"
)
print(
    f"High correlation removed: "
    f"{len(high_corr_features)}"
)

print()
print(
    f"Final selected features: "
    f"{len(selected_features)}"
)

print()
print("Validation used:")
print("NO")

print("Test used:")
print("NO")

print("SMOTE used:")
print("NO")

print()
print(
    f"Total processing time: "
    f"{total_time:.2f} seconds"
)

print()
print("Next:")
print("STEP 15 - Apply Selected Features to")
print("CIC-IDS2018 Train / Validation / Test")

print("=" * 70)