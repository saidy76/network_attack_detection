import os
import pandas as pd
import json
import time

# ============================================================
# STEP 15 - APPLY SELECTED CIC-IDS2018 FEATURES
# ============================================================

print("=" * 70)
print("STEP 15 - APPLY SELECTED CIC-IDS2018 FEATURES")
print("=" * 70)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "modeling")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiments")

FEATURE_FILE = os.path.join(
    EXPERIMENT_DIR,
    "step14_cicids2018_selected_features.csv"
)

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "CICIDS2018_train.csv"
)

VAL_FILE = os.path.join(
    DATA_DIR,
    "CICIDS2018_validation.csv"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "CICIDS2018_test.csv"
)

OUTPUT_TRAIN = os.path.join(
    DATA_DIR,
    "CICIDS2018_train_selected.csv"
)

OUTPUT_VAL = os.path.join(
    DATA_DIR,
    "CICIDS2018_validation_selected.csv"
)

OUTPUT_TEST = os.path.join(
    DATA_DIR,
    "CICIDS2018_test_selected.csv"
)

SUMMARY_FILE = os.path.join(
    EXPERIMENT_DIR,
    "step15_cicids2018_summary.json"
)


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

required_files = {
    "Selected feature file": FEATURE_FILE,
    "Training file": TRAIN_FILE,
    "Validation file": VAL_FILE,
    "Test file": TEST_FILE
}

for name, path in required_files.items():

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] {name} not found:\n{path}"
        )

    print(f"[OK] {name}:")
    print(path)


# ============================================================
# 2. LOAD SELECTED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("LOADING SELECTED FEATURES")
print("=" * 70)

feature_df = pd.read_csv(FEATURE_FILE)

print(f"Feature file rows: {len(feature_df)}")
print(f"Feature file columns: {list(feature_df.columns)}")


# ------------------------------------------------------------
# Automatically detect feature-name column
# ------------------------------------------------------------

possible_columns = [
    "Feature",
    "feature",
    "Feature Name",
    "feature_name",
    "Selected Feature",
    "selected_feature"
]

feature_column = None

for col in possible_columns:

    if col in feature_df.columns:
        feature_column = col
        break


if feature_column is None:

    # If only one column exists, use it
    if len(feature_df.columns) == 1:
        feature_column = feature_df.columns[0]

    else:
        raise ValueError(
            "Could not identify the feature-name column "
            "inside selected feature file."
        )


selected_features = (
    feature_df[feature_column]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)

# Remove accidental duplicates while preserving order
selected_features = list(dict.fromkeys(selected_features))


print("\nSelected feature column:")
print(feature_column)

print("\nNumber of selected features:")
print(len(selected_features))

print("\nSelected features:")

for i, feature in enumerate(selected_features, start=1):
    print(f"{i:02d}. {feature}")


# ============================================================
# 3. CHECK TARGET
# ============================================================

TARGET = "Label"

print("\n" + "=" * 70)
print("TARGET INFORMATION")
print("=" * 70)

print(f"Target column: {TARGET}")


# ============================================================
# 4. PROCESS DATASET FUNCTION
# ============================================================

def apply_features(input_file, output_file, dataset_name):

    print("\n" + "-" * 70)
    print(f"PROCESSING: {dataset_name}")
    print("-" * 70)

    start_time = time.time()

    print("Loading dataset...")
    df = pd.read_csv(input_file)

    print(f"[OK] Loaded.")
    print(f"Original shape: {df.shape}")

    # --------------------------------------------------------
    # Check Label
    # --------------------------------------------------------

    if TARGET not in df.columns:

        raise ValueError(
            f"Target column '{TARGET}' not found in {dataset_name}"
        )

    # --------------------------------------------------------
    # Check selected features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in selected_features
        if feature not in df.columns
    ]

    if missing_features:

        print("\n[ERROR] Selected features missing:")

        for feature in missing_features:
            print(f" - {feature}")

        raise ValueError(
            f"{len(missing_features)} selected features "
            f"are missing from {dataset_name}."
        )

    print(
        f"[OK] All {len(selected_features)} selected features "
        "are present."
    )

    # --------------------------------------------------------
    # Select features + target
    # --------------------------------------------------------

    output_columns = selected_features + [TARGET]

    selected_df = df[output_columns].copy()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nValidating selected dataset...")

    print(
        f"Selected feature count: "
        f"{len(selected_features)}"
    )

    print(
        f"Final column count: "
        f"{len(selected_df.columns)}"
    )

    print(
        f"Rows: "
        f"{len(selected_df):,}"
    )

    # Check missing values
    missing_values = int(
        selected_df.isna().sum().sum()
    )

    print(
        f"Missing values: "
        f"{missing_values:,}"
    )

    # Check infinity only on numeric columns
    numeric_columns = selected_df.select_dtypes(
        include="number"
    ).columns

    infinity_values = int(
        selected_df[numeric_columns]
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    print(
        f"Infinity values: "
        f"{infinity_values:,}"
    )

    # Check target
    empty_labels = int(
        selected_df[TARGET]
        .isna()
        .sum()
    )

    print(
        f"Empty labels: "
        f"{empty_labels:,}"
    )

    if missing_values != 0:

        raise ValueError(
            f"{dataset_name} contains missing values."
        )

    if infinity_values != 0:

        raise ValueError(
            f"{dataset_name} contains Infinity values."
        )

    if empty_labels != 0:

        raise ValueError(
            f"{dataset_name} contains empty labels."
        )

    # --------------------------------------------------------
    # Check class count
    # --------------------------------------------------------

    class_count = selected_df[TARGET].nunique()

    print(
        f"Number of classes: "
        f"{class_count}"
    )

    print("\nClasses:")

    for label in sorted(
        selected_df[TARGET]
        .astype(str)
        .unique()
    ):

        print(f" - {label}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\nSaving selected dataset...")

    selected_df.to_csv(
        output_file,
        index=False
    )

    print("[OK] Saved:")
    print(output_file)

    elapsed = time.time() - start_time

    print(
        f"Processing time: "
        f"{elapsed:.2f} seconds"
    )

    return {
        "dataset": dataset_name,
        "input_file": input_file,
        "output_file": output_file,
        "rows": int(len(selected_df)),
        "features": int(len(selected_features)),
        "columns": int(len(selected_df.columns)),
        "classes": int(class_count),
        "missing_values": missing_values,
        "infinity_values": infinity_values,
        "empty_labels": empty_labels,
        "processing_time_seconds": round(elapsed, 2)
    }


# ============================================================
# 5. PROCESS TRAINING DATA
# ============================================================

train_summary = apply_features(
    TRAIN_FILE,
    OUTPUT_TRAIN,
    "CIC-IDS2018 TRAINING"
)


# ============================================================
# 6. PROCESS VALIDATION DATA
# ============================================================

validation_summary = apply_features(
    VAL_FILE,
    OUTPUT_VAL,
    "CIC-IDS2018 VALIDATION"
)


# ============================================================
# 7. PROCESS TEST DATA
# ============================================================

test_summary = apply_features(
    TEST_FILE,
    OUTPUT_TEST,
    "CIC-IDS2018 TEST"
)


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 15 FINAL SUMMARY")
print("=" * 70)

print(
    f"Selected features : {len(selected_features)}"
)

print(
    f"Training rows     : "
    f"{train_summary['rows']:,}"
)

print(
    f"Validation rows   : "
    f"{validation_summary['rows']:,}"
)

print(
    f"Test rows         : "
    f"{test_summary['rows']:,}"
)

print("\nOutput files:")

print(
    f"[OK] Train:\n{OUTPUT_TRAIN}"
)

print(
    f"[OK] Validation:\n{OUTPUT_VAL}"
)

print(
    f"[OK] Test:\n{OUTPUT_TEST}"
)


# ============================================================
# 9. SAVE SUMMARY JSON
# ============================================================

summary = {
    "step": 15,
    "dataset": "CIC-IDS2018",

    "feature_selection_source":
        "Step 14 - Training data only",

    "selected_feature_count":
        len(selected_features),

    "selected_features":
        selected_features,

    "training": train_summary,
    "validation": validation_summary,
    "test": test_summary,

    "feature_selection_repeated":
        False,

    "validation_used_for_feature_selection":
        False,

    "test_used_for_feature_selection":
        False,

    "smote_used":
        False,

    "model_training":
        False
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


print("\n[OK] Summary saved:")
print(SUMMARY_FILE)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 15 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
CIC-IDS2018 selected features have been
successfully applied to:

- Training
- Validation
- Test

Selected features:
30

No new feature selection was performed.

No SMOTE was performed.

No model training was performed.

Next:
STEP 16 - CIC-IDS2018 CLASS IMBALANCE ANALYSIS
""")

print("=" * 70)