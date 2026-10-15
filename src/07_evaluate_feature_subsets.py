"""
STEP 7 - VALIDATION-BASED FEATURE SUBSET EVALUATION

Purpose:
Evaluate different feature subset sizes using validation data.

Candidate subsets:
39, 30, 25, 20, 15, 12 features

Important:
- Feature ranking comes from Step 6
- Only training data was used for feature ranking
- Validation data is used only for model comparison
- Test data is NOT used
- No SMOTE
- No scaling
"""

import os
import time
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "modeling",
    "CICIoT2023_train_clean.csv"
)

VAL_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "modeling",
    "CICIoT2023_validation_clean.csv"
)

FEATURE_RANKING_PATH = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "step6_feature_selection_consensus.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# SETTINGS
# ================================================================

RANDOM_STATE = 42

# Candidate feature subset sizes
SUBSET_SIZES = [39, 30, 25, 20, 15, 12]

# Random Forest configuration
N_ESTIMATORS = 100
N_JOBS = -1


# ================================================================
# HEADER
# ================================================================

print("=" * 75)
print("STEP 7 - VALIDATION-BASED FEATURE SUBSET EVALUATION")
print("=" * 75)

print()
print("Candidate feature subset sizes:")
print(SUBSET_SIZES)

print()
print("Target:")
print("Raw Label - 34 multiclass attack categories")

print()
print("Model:")
print("Random Forest")

print()
print("Validation metric:")
print("Macro-F1")

print()
print("IMPORTANT:")
print("Training data is used for model training.")
print("Validation data is used for subset evaluation.")
print("Test data is NOT used.")
print("No SMOTE is applied.")
print("No scaling is applied.")


# ================================================================
# CHECK FILES
# ================================================================

print()
print("=" * 75)
print("CHECKING INPUT FILES")
print("=" * 75)

required_files = [
    TRAIN_PATH,
    VAL_PATH,
    FEATURE_RANKING_PATH
]

for path in required_files:

    if os.path.exists(path):
        print("[OK]", path)

    else:
        print("[ERROR] File not found:", path)
        raise FileNotFoundError(path)


# ================================================================
# LOAD DATA
# ================================================================

print()
print("=" * 75)
print("LOADING DATA")
print("=" * 75)

print()
print("Loading training data...")

train_df = pd.read_csv(TRAIN_PATH)

print("Training shape:", train_df.shape)

print()
print("Loading validation data...")

val_df = pd.read_csv(VAL_PATH)

print("Validation shape:", val_df.shape)


# ================================================================
# LOAD FEATURE RANKING
# ================================================================

print()
print("=" * 75)
print("LOADING STEP 6 FEATURE RANKING")
print("=" * 75)

ranking_df = pd.read_csv(FEATURE_RANKING_PATH)

print()
print("Ranking columns:")
print(list(ranking_df.columns))

print()
print("Total ranked features:", len(ranking_df))


# ================================================================
# SORT BY CONSENSUS RANK
# ================================================================

ranking_df = ranking_df.sort_values(
    "consensus_rank"
).reset_index(drop=True)


all_ranked_features = ranking_df["feature"].tolist()


print()
print("Top ranked features:")

for i, feature in enumerate(all_ranked_features[:20], start=1):
    print(f"{i:02d}. {feature}")


# ================================================================
# VALIDATE FEATURES
# ================================================================

print()
print("=" * 75)
print("VALIDATING FEATURE COLUMNS")
print("=" * 75)

missing_train_features = [
    f for f in all_ranked_features
    if f not in train_df.columns
]

missing_val_features = [
    f for f in all_ranked_features
    if f not in val_df.columns
]

if missing_train_features:
    print("[ERROR] Missing features in training data:")
    print(missing_train_features)
    raise ValueError("Training feature mismatch")

if missing_val_features:
    print("[ERROR] Missing features in validation data:")
    print(missing_val_features)
    raise ValueError("Validation feature mismatch")

print("[OK] All ranked features exist in training data.")
print("[OK] All ranked features exist in validation data.")


# ================================================================
# TARGET
# ================================================================

TARGET = "Label"

if TARGET not in train_df.columns:
    raise ValueError("Label column not found in training data.")

if TARGET not in val_df.columns:
    raise ValueError("Label column not found in validation data.")


y_train = train_df[TARGET]
y_val = val_df[TARGET]


print()
print("=" * 75)
print("TARGET VALIDATION")
print("=" * 75)

print()
print("Training classes:", y_train.nunique())
print("Validation classes:", y_val.nunique())

if set(y_train.unique()) != set(y_val.unique()):

    print()
    print("[WARNING] Training and validation classes are not identical.")

else:

    print("[OK] Training and validation contain the same classes.")


# ================================================================
# RESULTS STORAGE
# ================================================================

results = []

feature_details = {}


# ================================================================
# EVALUATION LOOP
# ================================================================

for subset_size in SUBSET_SIZES:

    print()
    print("=" * 75)
    print(f"EVALUATING TOP {subset_size} FEATURES")
    print("=" * 75)

    selected_features = all_ranked_features[:subset_size]

    feature_details[subset_size] = selected_features

    print()
    print("Selected features:")

    for i, feature in enumerate(selected_features, start=1):
        print(f"{i:02d}. {feature}")


    # ------------------------------------------------------------
    # Prepare X
    # ------------------------------------------------------------

    print()
    print("Preparing training data...")

    X_train = train_df[selected_features]

    print("Preparing validation data...")

    X_val = val_df[selected_features]


    # ------------------------------------------------------------
    # Safety validation
    # ------------------------------------------------------------

    if X_train.isna().sum().sum() != 0:
        raise ValueError(
            f"Missing values detected in training subset {subset_size}"
        )

    if X_val.isna().sum().sum() != 0:
        raise ValueError(
            f"Missing values detected in validation subset {subset_size}"
        )


    # ------------------------------------------------------------
    # Train Random Forest
    # ------------------------------------------------------------

    print()
    print("Training Random Forest...")

    start_train = time.time()

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        class_weight="balanced_subsample"
    )

    model.fit(X_train, y_train)

    train_time = time.time() - start_train

    print(
        f"Training completed in {train_time:.2f} seconds."
    )


    # ------------------------------------------------------------
    # Validation prediction
    # ------------------------------------------------------------

    print()
    print("Predicting validation set...")

    start_predict = time.time()

    y_pred = model.predict(X_val)

    prediction_time = time.time() - start_predict

    print(
        f"Prediction completed in {prediction_time:.2f} seconds."
    )


    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    balanced_acc = balanced_accuracy_score(
        y_val,
        y_pred
    )

    precision_macro = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # ------------------------------------------------------------
    # Store result
    # ------------------------------------------------------------

    result = {

        "feature_count": subset_size,

        "accuracy": accuracy,

        "balanced_accuracy": balanced_acc,

        "precision_macro": precision_macro,

        "recall_macro": recall_macro,

        "f1_macro": f1_macro,

        "f1_weighted": f1_weighted,

        "training_time_seconds": train_time,

        "prediction_time_seconds": prediction_time
    }

    results.append(result)


    # ------------------------------------------------------------
    # Print result
    # ------------------------------------------------------------

    print()
    print("-" * 75)
    print("RESULT")
    print("-" * 75)

    print(f"Features          : {subset_size}")
    print(f"Accuracy          : {accuracy:.6f}")
    print(f"Balanced Accuracy : {balanced_acc:.6f}")
    print(f"Macro Precision   : {precision_macro:.6f}")
    print(f"Macro Recall      : {recall_macro:.6f}")
    print(f"Macro F1          : {f1_macro:.6f}")
    print(f"Weighted F1       : {f1_weighted:.6f}")
    print(f"Training Time     : {train_time:.2f} sec")
    print(f"Prediction Time   : {prediction_time:.2f} sec")


# ================================================================
# RESULTS DATAFRAME
# ================================================================

results_df = pd.DataFrame(results)


# ================================================================
# FIND BEST SUBSET
# ================================================================

print()
print("=" * 75)
print("FEATURE SUBSET COMPARISON")
print("=" * 75)

display_columns = [
    "feature_count",
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "training_time_seconds",
    "prediction_time_seconds"
]

print()

print(
    results_df[display_columns].to_string(
        index=False
    )
)


# ================================================================
# BEST BY MACRO F1
# ================================================================

best_index = results_df["f1_macro"].idxmax()

best_result = results_df.loc[best_index]

best_feature_count = int(
    best_result["feature_count"]
)

best_features = feature_details[
    best_feature_count
]


print()
print("=" * 75)
print("BEST FEATURE SUBSET")
print("=" * 75)

print()
print(
    f"Best subset size: {best_feature_count} features"
)

print(
    f"Best Macro-F1: {best_result['f1_macro']:.6f}"
)

print()
print("Selected features:")

for i, feature in enumerate(best_features, start=1):

    print(
        f"{i:02d}. {feature}"
    )


# ================================================================
# SAVE RESULTS
# ================================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "step7_feature_subset_validation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print()
print("[OK] Results saved:")
print(results_path)


# ================================================================
# SAVE SELECTED FEATURES
# ================================================================

selected_features_path = os.path.join(
    OUTPUT_DIR,
    "step7_selected_features.csv"
)

selected_features_df = pd.DataFrame({

    "rank": range(
        1,
        len(best_features) + 1
    ),

    "feature": best_features
})

selected_features_df.to_csv(
    selected_features_path,
    index=False
)

print()
print("[OK] Selected feature list saved:")
print(selected_features_path)


# ================================================================
# SAVE JSON SUMMARY
# ================================================================

summary = {

    "step": 7,

    "dataset": "CICIoT2023",

    "target": "Label",

    "number_of_classes": int(
        y_train.nunique()
    ),

    "candidate_feature_counts":
        SUBSET_SIZES,

    "best_feature_count":
        best_feature_count,

    "best_macro_f1":
        float(
            best_result["f1_macro"]
        ),

    "best_features":
        best_features,

    "model":
        "RandomForestClassifier",

    "n_estimators":
        N_ESTIMATORS,

    "class_weight":
        "balanced_subsample",

    "random_state":
        RANDOM_STATE,

    "test_data_used":
        False,

    "smote_used":
        False,

    "scaling_used":
        False
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "step7_feature_subset_validation_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print()
print("[OK] Summary saved:")
print(summary_path)


# ================================================================
# FINAL
# ================================================================

print()
print("=" * 75)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 75)

print()
print("Important:")
print("1. Feature ranking came from training data.")
print("2. Validation data was used for subset comparison.")
print("3. Test data was NOT used.")
print("4. Multiple feature subset sizes were evaluated.")
print("5. Macro-F1 was used as the primary selection metric.")
print("6. No SMOTE was applied.")
print("7. No scaling was applied.")
print("8. Final feature subset was selected based on validation Macro-F1.")

print()
print("Next stage:")
print("Use the selected feature subset for baseline model comparison.")