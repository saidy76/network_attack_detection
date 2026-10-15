"""
STEP 13 - CIC-IDS2018 TRAIN / VALIDATION / TEST SPLIT

Purpose:
    Split the cleaned CIC-IDS2018 dataset into:
        - Training set      = 70%
        - Validation set    = 15%
        - Test set          = 15%

Dataset:
    CIC-IDS2018

Input:
    data/modeling/CICIDS2018_clean.csv

Target:
    Label

Important:
    - Stratified splitting is used.
    - All 14 classes are preserved as much as possible.
    - Test data is kept completely separate.
    - No feature selection.
    - No SMOTE.
    - No model training.
    - No scaling.
    - No tuning.

Random state:
    42

Outputs:
    CICIDS2018_train.csv
    CICIDS2018_validation.csv
    CICIDS2018_test.csv

Also saves:
    step13_cicids2018_split_summary.json
"""

import os
import json
import time

import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\DELL\OneDrive\Desktop\network_attack_detection"

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "modeling",
    "CICIDS2018_clean.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "modeling"
)

TRAIN_PATH = os.path.join(
    OUTPUT_DIR,
    "CICIDS2018_train.csv"
)

VALIDATION_PATH = os.path.join(
    OUTPUT_DIR,
    "CICIDS2018_validation.csv"
)

TEST_PATH = os.path.join(
    OUTPUT_DIR,
    "CICIDS2018_test.csv"
)

SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "experiments",
    "step13_cicids2018_split_summary.json"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "Label"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 13 - CIC-IDS2018 TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

print("""
Dataset:
CIC-IDS2018

Input:
CICIDS2018_clean.csv

Split:
70% Training
15% Validation
15% Test

Method:
Stratified Random Split

Target:
Label

Important:
Test data will remain completely separate.
No feature selection.
No SMOTE.
No model training.
""")

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(
    os.path.dirname(SUMMARY_PATH),
    exist_ok=True
)


# ============================================================
# CHECK INPUT
# ============================================================

print("=" * 70)
print("CHECKING INPUT DATASET")
print("=" * 70)

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"""
[ERROR] Clean CIC-IDS2018 dataset not found:

{INPUT_PATH}

Please complete Step 12 first.
"""
    )

print(f"[OK] Input file found:")
print(INPUT_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING CLEAN DATASET")
print("=" * 70)

print("Loading CICIDS2018_clean.csv...")
print("This may take some time because the dataset is large.")

start_time = time.time()

df = pd.read_csv(INPUT_PATH)

load_time = time.time() - start_time

print()
print(f"[OK] Dataset loaded.")
print(f"Shape: {df.shape}")
print(f"Loading time: {load_time:.2f} seconds")


# ============================================================
# BASIC VALIDATION
# ============================================================

print()
print("=" * 70)
print("BASIC DATA VALIDATION")
print("=" * 70)

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

print(f"[OK] Target column found: {TARGET}")

print(
    f"Total rows    : {len(df):,}"
)

print(
    f"Total columns : {len(df.columns)}"
)

print(
    f"Total classes : {df[TARGET].nunique()}"
)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print()
print("Checking missing values...")

missing_values = int(
    df.isnull().sum().sum()
)

if missing_values != 0:
    raise ValueError(
        f"Dataset contains {missing_values:,} missing values."
    )

print("[OK] No missing values found.")


# ============================================================
# CHECK INFINITY
# ============================================================

print()
print("Checking Infinity values...")

numeric_df = df.drop(
    columns=[TARGET]
).select_dtypes(
    include=["number"]
)

infinity_values = int(
    numeric_df.isin(
        [float("inf"), float("-inf")]
    ).sum().sum()
)

if infinity_values != 0:
    raise ValueError(
        f"Dataset contains {infinity_values:,} Infinity values."
    )

print("[OK] No Infinity values found.")


# ============================================================
# ORIGINAL CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("ORIGINAL CLASS DISTRIBUTION")
print("=" * 70)

original_distribution = (
    df[TARGET]
    .value_counts()
    .sort_index()
)

for class_name, count in original_distribution.items():

    percentage = (
        count / len(df)
    ) * 100

    print(
        f"{class_name:<35} "
        f": {count:>10,} "
        f"({percentage:>7.3f}%)"
    )


# ============================================================
# FIRST SPLIT
# ============================================================
#
# 70% TRAIN
# 30% TEMPORARY
#
# The temporary set will later become:
# 15% VALIDATION
# 15% TEST
#
# Therefore:
# validation/test each need 50% of temporary data.
# ============================================================

print()
print("=" * 70)
print("CREATING TRAINING SET")
print("=" * 70)

print("Creating 70% training set...")

split_start = time.time()

train_df, temp_df = train_test_split(
    df,
    test_size=(VALIDATION_SIZE + TEST_SIZE),
    stratify=df[TARGET],
    random_state=RANDOM_STATE
)

print()
print(
    f"Training rows : {len(train_df):,}"
)

print(
    f"Temporary rows: {len(temp_df):,}"
)


# ============================================================
# SECOND SPLIT
# ============================================================

print()
print("=" * 70)
print("CREATING VALIDATION AND TEST SETS")
print("=" * 70)

print(
    "Splitting temporary data into "
    "Validation 15% and Test 15%..."
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df[TARGET],
    random_state=RANDOM_STATE
)

split_time = time.time() - split_start

print()
print(
    f"Validation rows: {len(validation_df):,}"
)

print(
    f"Test rows      : {len(test_df):,}"
)

print(
    f"Total split time: {split_time:.2f} seconds"
)


# ============================================================
# RELEASE TEMPORARY DATA
# ============================================================

del temp_df


# ============================================================
# CHECK SPLIT SIZES
# ============================================================

print()
print("=" * 70)
print("CHECKING SPLIT SIZES")
print("=" * 70)

total_rows = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

print(
    f"Original rows : {len(df):,}"
)

print(
    f"Train rows    : {len(train_df):,}"
)

print(
    f"Validation    : {len(validation_df):,}"
)

print(
    f"Test          : {len(test_df):,}"
)

print(
    f"Total         : {total_rows:,}"
)

if total_rows != len(df):

    raise ValueError(
        "Split row counts do not match original dataset."
    )

print()
print("[OK] All rows accounted for.")


# ============================================================
# CHECK CLASS COUNTS
# ============================================================

print()
print("=" * 70)
print("CHECKING CLASS PRESENCE")
print("=" * 70)

original_classes = set(
    df[TARGET].unique()
)

train_classes = set(
    train_df[TARGET].unique()
)

validation_classes = set(
    validation_df[TARGET].unique()
)

test_classes = set(
    test_df[TARGET].unique()
)

print(
    f"Original classes   : {len(original_classes)}"
)

print(
    f"Training classes   : {len(train_classes)}"
)

print(
    f"Validation classes : {len(validation_classes)}"
)

print(
    f"Test classes       : {len(test_classes)}"
)


if original_classes != train_classes:
    missing = original_classes - train_classes

    raise ValueError(
        f"Classes missing from training set: {missing}"
    )

if original_classes != validation_classes:
    missing = original_classes - validation_classes

    raise ValueError(
        f"Classes missing from validation set: {missing}"
    )

if original_classes != test_classes:
    missing = original_classes - test_classes

    raise ValueError(
        f"Classes missing from test set: {missing}"
    )

print()
print(
    "[OK] All 14 classes are present in "
    "Train, Validation and Test."
)


# ============================================================
# CLASS DISTRIBUTION COMPARISON
# ============================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION AFTER SPLIT")
print("=" * 70)

train_distribution = (
    train_df[TARGET]
    .value_counts()
)

validation_distribution = (
    validation_df[TARGET]
    .value_counts()
)

test_distribution = (
    test_df[TARGET]
    .value_counts()
)

distribution_df = pd.DataFrame({
    "Total": original_distribution,
    "Train": train_distribution,
    "Validation": validation_distribution,
    "Test": test_distribution
}).fillna(0).astype(int)

distribution_df["Train_%"] = (
    distribution_df["Train"]
    / distribution_df["Total"]
    * 100
)

distribution_df["Validation_%"] = (
    distribution_df["Validation"]
    / distribution_df["Total"]
    * 100
)

distribution_df["Test_%"] = (
    distribution_df["Test"]
    / distribution_df["Total"]
    * 100
)

print(
    distribution_df.to_string()
)


# ============================================================
# CHECK DATA LEAKAGE BY EXACT DUPLICATES
# ============================================================

print()
print("=" * 70)
print("CHECKING EXACT DUPLICATE LEAKAGE")
print("=" * 70)

print(
    "Checking whether identical rows exist "
    "between Train and Validation..."
)

train_val_overlap = len(
    pd.merge(
        train_df,
        validation_df,
        how="inner"
    )
)

print(
    f"Train ↔ Validation exact duplicate rows: "
    f"{train_val_overlap:,}"
)

print(
    "Checking Train and Test..."
)

train_test_overlap = len(
    pd.merge(
        train_df,
        test_df,
        how="inner"
    )
)

print(
    f"Train ↔ Test exact duplicate rows: "
    f"{train_test_overlap:,}"
)

print(
    "Checking Validation and Test..."
)

validation_test_overlap = len(
    pd.merge(
        validation_df,
        test_df,
        how="inner"
    )
)

print(
    f"Validation ↔ Test exact duplicate rows: "
    f"{validation_test_overlap:,}"
)


if (
    train_val_overlap == 0
    and train_test_overlap == 0
    and validation_test_overlap == 0
):

    print()
    print(
        "[OK] No exact duplicate rows found "
        "between splits."
    )

else:

    print()
    print(
        "[WARNING] Exact duplicate rows detected "
        "between splits."
    )


# ============================================================
# SAVE TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("SAVING TRAINING DATA")
print("=" * 70)

save_start = time.time()

train_df.to_csv(
    TRAIN_PATH,
    index=False
)

print(
    f"[OK] Training dataset saved:"
)

print(TRAIN_PATH)

print(
    f"Training rows: {len(train_df):,}"
)


# ============================================================
# SAVE VALIDATION DATA
# ============================================================

print()
print("=" * 70)
print("SAVING VALIDATION DATA")
print("=" * 70)

validation_df.to_csv(
    VALIDATION_PATH,
    index=False
)

print(
    f"[OK] Validation dataset saved:"
)

print(VALIDATION_PATH)

print(
    f"Validation rows: {len(validation_df):,}"
)


# ============================================================
# SAVE TEST DATA
# ============================================================

print()
print("=" * 70)
print("SAVING TEST DATA")
print("=" * 70)

test_df.to_csv(
    TEST_PATH,
    index=False
)

print(
    f"[OK] Test dataset saved:"
)

print(TEST_PATH)

print(
    f"Test rows: {len(test_df):,}"
)

save_time = time.time() - save_start


# ============================================================
# SAVE SUMMARY
# ============================================================

print()
print("=" * 70)
print("SAVING STEP 13 SUMMARY")
print("=" * 70)

summary = {

    "step": 13,

    "dataset": "CIC-IDS2018",

    "input": "CICIDS2018_clean.csv",

    "target": TARGET,

    "random_state": RANDOM_STATE,

    "split_method": "stratified_random_split",

    "split_ratio": {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15
    },

    "original_rows": int(len(df)),

    "train_rows": int(len(train_df)),

    "validation_rows": int(len(validation_df)),

    "test_rows": int(len(test_df)),

    "num_classes": int(len(original_classes)),

    "classes": sorted(
        list(original_classes)
    ),

    "duplicate_overlap": {

        "train_validation": int(
            train_val_overlap
        ),

        "train_test": int(
            train_test_overlap
        ),

        "validation_test": int(
            validation_test_overlap
        )
    },

    "feature_selection": False,

    "smote": False,

    "model_training": False,

    "data_leakage_policy": {

        "test_used_for_feature_selection": False,

        "test_used_for_tuning": False,

        "test_used_for_model_selection": False
    },

    "output_files": {

        "train": TRAIN_PATH,

        "validation": VALIDATION_PATH,

        "test": TEST_PATH
    }
}


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

print(
    f"[OK] Summary saved:"
)

print(SUMMARY_PATH)


# ============================================================
# RELEASE MEMORY
# ============================================================

del df
del train_df
del validation_df
del test_df
del distribution_df


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("STEP 13 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
CIC-IDS2018 has been split successfully.

Train:
70%

Validation:
15%

Test:
15%

Method:
Stratified split

Classes:
14

Important:
Test data is completely separated and must NOT be
used for feature selection or model tuning.

Next:
STEP 14 - CIC-IDS2018 Feature Cleaning / Feature Selection
""")

print("=" * 70)
