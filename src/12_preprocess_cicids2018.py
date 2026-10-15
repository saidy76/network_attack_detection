"""
STEP 12 - CORRECTED CIC-IDS2018 PREPROCESSING

Dataset:
    CIC-IDS2018

Input:
    9 raw CICFlowMeter CSV files

Operations:
    1. Load raw CSV files
    2. Remove accidental header/Label rows
    3. Clean Label values
    4. Convert feature columns to numeric
    5. Replace Infinity with NaN
    6. Replace NaN / invalid numeric values with 0
    7. Remove duplicate rows
    8. Merge all 9 files
    9. Final validation
    10. Save cleaned dataset

No:
    - Feature selection
    - Train/Validation/Test split
    - SMOTE
    - Model training
"""

import os
import gc
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\DELL\OneDrive\Desktop\network_attack_detection"

INPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "cicids2018"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "modeling"
)

EXPERIMENT_DIR = os.path.join(
    BASE_DIR,
    "experiments"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "CICIDS2018_clean.csv"
)

SUMMARY_PATH = os.path.join(
    EXPERIMENT_DIR,
    "step12_cicids2018_summary.json"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 12 - CORRECTED CIC-IDS2018 PREPROCESSING")
print("=" * 70)

print("""
Dataset:
CIC-IDS2018

Input:
9 raw CICFlowMeter CSV files

Cleaning:
- Remove accidental header/Label rows
- Convert numeric features
- Replace Infinity with NaN
- Replace NaN/invalid values with 0
- Remove duplicates
- Standardize labels
- Validate final dataset

No:
- Feature selection
- Train/Validation/Test split
- SMOTE
- Model training
""")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


# ============================================================
# FIND CSV FILES
# ============================================================

print("=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

csv_files = sorted(
    [
        os.path.join(INPUT_DIR, file)
        for file in os.listdir(INPUT_DIR)
        if file.lower().endswith(".csv")
    ]
)

print(f"CSV files found: {len(csv_files)}")

if len(csv_files) != 9:
    raise ValueError(
        f"Expected exactly 9 CSV files, "
        f"but found {len(csv_files)}."
    )

for i, path in enumerate(csv_files, start=1):
    size_mb = os.path.getsize(path) / (1024 ** 2)

    print(
        f"{i:02d}. {os.path.basename(path)} "
        f"({size_mb:.2f} MB)"
    )

print("[OK] Exactly 9 CIC-IDS2018 CSV files found.")


# ============================================================
# REFERENCE COLUMNS
# ============================================================

print()
print("=" * 70)
print("CHECKING REFERENCE COLUMNS")
print("=" * 70)

reference_columns = pd.read_csv(
    csv_files[0],
    nrows=0
).columns.tolist()

print(
    f"Reference column count: "
    f"{len(reference_columns)}"
)

if "Label" not in reference_columns:
    raise ValueError(
        "Label column was not found."
    )

print("[OK] Label column exists.")


# ============================================================
# STORAGE
# ============================================================

cleaned_dataframes = []

total_original_rows = 0
total_removed_header_rows = 0
total_replaced_nan = 0
total_removed_duplicates = 0


# ============================================================
# PROCESS EACH FILE
# ============================================================

for file_number, file_path in enumerate(csv_files, start=1):

    file_start = time.time()

    filename = os.path.basename(file_path)

    print()
    print("-" * 70)
    print(
        f"[{file_number}/9] Processing: {filename}"
    )
    print("-" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("Loading CSV...")

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    original_rows = len(df)

    total_original_rows += original_rows

    print(
        f"Rows loaded: {original_rows:,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # COLUMN CHECK
    # --------------------------------------------------------

    if df.columns.tolist() != reference_columns:

        raise ValueError(
            f"Column mismatch detected in:\n{filename}"
        )

    print("[OK] Columns match reference.")

    # --------------------------------------------------------
    # REMOVE ACCIDENTAL HEADER ROWS
    # --------------------------------------------------------

    print(
        "Removing accidental header/Label rows..."
    )

    before = len(df)

    # Remove rows where Label itself is literally "Label"
    df = df[
        df["Label"].astype(str).str.strip().str.lower()
        != "label"
    ]

    removed_header = before - len(df)

    total_removed_header_rows += removed_header

    print(
        f"Removed accidental header rows: "
        f"{removed_header:,}"
    )

    # --------------------------------------------------------
    # CLEAN LABEL
    # --------------------------------------------------------

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    # Remove empty labels
    before = len(df)

    df = df[
        df["Label"].notna()
        & (df["Label"].str.len() > 0)
    ]

    removed_empty_labels = before - len(df)

    print(
        f"Removed empty labels: "
        f"{removed_empty_labels:,}"
    )

    # --------------------------------------------------------
    # STANDARDIZE LABELS
    # --------------------------------------------------------

    label_mapping = {
        "Infilteration": "Infiltration",
        "infilteration": "Infiltration",
        "INFILTERATION": "Infiltration",

        "Benign": "BENIGN",
        "BENIGN": "BENIGN",
        "benign": "BENIGN",
    }

    df["Label"] = df["Label"].replace(
        label_mapping
    )

    # --------------------------------------------------------
    # FEATURE COLUMNS
    # --------------------------------------------------------

    feature_columns = [
        col
        for col in df.columns
        if col != "Label"
    ]

    print(
        f"Converting "
        f"{len(feature_columns)} feature columns "
        f"to numeric..."
    )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    for col in feature_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # INFINITY -> NaN
    # --------------------------------------------------------

    print(
        "Handling Infinity values..."
    )

    infinity_count = np.isinf(
        df[feature_columns].to_numpy(
            dtype=np.float64
        )
    ).sum()

    print(
        f"Infinity values found: "
        f"{infinity_count:,}"
    )

    df[feature_columns] = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    # --------------------------------------------------------
    # NaN -> 0
    # --------------------------------------------------------

    print(
        "Handling missing/invalid numeric values..."
    )

    nan_before = int(
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing/invalid numeric values: "
        f"{nan_before:,}"
    )

    # IMPORTANT:
    # CICFlowMeter can contain undefined values
    # for some flow statistics.
    #
    # Instead of deleting the entire row,
    # replace these values with 0.

    df[feature_columns] = (
        df[feature_columns]
        .fillna(0)
    )

    nan_after = int(
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Remaining missing values: "
        f"{nan_after:,}"
    )

    total_replaced_nan += nan_before

    # --------------------------------------------------------
    # REMOVE DUPLICATES WITHIN FILE
    # --------------------------------------------------------

    print(
        "Checking duplicate rows..."
    )

    before = len(df)

    df = df.drop_duplicates(
        keep="first"
    )

    duplicates_removed = (
        before - len(df)
    )

    total_removed_duplicates += (
        duplicates_removed
    )

    print(
        f"Duplicate rows removed: "
        f"{duplicates_removed:,}"
    )

    # --------------------------------------------------------
    # FINAL FILE INFO
    # --------------------------------------------------------

    print(
        f"Final rows from file: "
        f"{len(df):,}"
    )

    print(
        f"Classes in file: "
        f"{df['Label'].nunique()}"
    )

    for label in sorted(
        df["Label"].unique()
    ):
        print(f"  - {label}")

    cleaned_dataframes.append(df)

    elapsed = time.time() - file_start

    print(
        f"Processing time: "
        f"{elapsed:.2f} sec"
    )

    # Free temporary objects
    gc.collect()


# ============================================================
# MERGE
# ============================================================

print()
print("=" * 70)
print("MERGING CLEANED DATA")
print("=" * 70)

print(
    f"Merging {len(cleaned_dataframes)} "
    f"processed files..."
)

merged_df = pd.concat(
    cleaned_dataframes,
    ignore_index=True
)

print(
    f"Merged shape: "
    f"{merged_df.shape}"
)

del cleaned_dataframes

gc.collect()


# ============================================================
# FINAL DUPLICATE CHECK
# ============================================================

print()
print("=" * 70)
print("FINAL DUPLICATE CHECK")
print("=" * 70)

before = len(merged_df)

merged_df = merged_df.drop_duplicates(
    keep="first"
)

final_duplicates_removed = (
    before - len(merged_df)
)

print(
    f"Duplicate rows removed after merge: "
    f"{final_duplicates_removed:,}"
)


# ============================================================
# FINAL LABEL CLEANING
# ============================================================

print()
print("=" * 70)
print("FINAL LABEL CLEANING")
print("=" * 70)

before = len(merged_df)

merged_df = merged_df[
    merged_df["Label"].notna()
]

merged_df = merged_df[
    merged_df["Label"].astype(str).str.strip() != ""
]

merged_df = merged_df[
    merged_df["Label"].astype(str).str.lower()
    != "label"
]

removed_final_labels = (
    before - len(merged_df)
)

print(
    f"Accidental/empty label rows removed: "
    f"{removed_final_labels:,}"
)


# ============================================================
# FINAL CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("FINAL CLASS DISTRIBUTION")
print("=" * 70)

class_distribution = (
    merged_df["Label"]
    .value_counts()
    .sort_index()
)

print(
    f"Total classes: "
    f"{len(class_distribution)}"
)

for label, count in class_distribution.items():

    print(
        f"{label:<35}: "
        f"{count:,}"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("FINAL DATA VALIDATION")
print("=" * 70)

feature_columns = [
    col
    for col in merged_df.columns
    if col != "Label"
]

final_rows = len(merged_df)
final_columns = len(merged_df.columns)

missing_values = int(
    merged_df[feature_columns]
    .isna()
    .sum()
    .sum()
)

infinity_values = int(
    np.isinf(
        merged_df[feature_columns].to_numpy(
            dtype=np.float64
        )
    ).sum()
)

duplicate_rows = int(
    merged_df.duplicated().sum()
)

empty_labels = int(
    merged_df["Label"]
    .isna()
    .sum()
)

print(
    f"Final rows       : "
    f"{final_rows:,}"
)

print(
    f"Final columns    : "
    f"{final_columns}"
)

print(
    f"Total classes    : "
    f"{merged_df['Label'].nunique()}"
)

print(
    f"Missing values   : "
    f"{missing_values:,}"
)

print(
    f"Infinity values  : "
    f"{infinity_values:,}"
)

print(
    f"Duplicate rows   : "
    f"{duplicate_rows:,}"
)

print(
    f"Empty labels     : "
    f"{empty_labels:,}"
)


# ============================================================
# VALIDATION CONDITIONS
# ============================================================

if final_rows == 0:
    raise ValueError(
        "Final dataset contains zero rows."
    )

if missing_values != 0:
    raise ValueError(
        "Missing values still exist."
    )

if infinity_values != 0:
    raise ValueError(
        "Infinity values still exist."
    )

if duplicate_rows != 0:
    raise ValueError(
        "Duplicate rows still exist."
    )

if empty_labels != 0:
    raise ValueError(
        "Empty labels still exist."
    )

if merged_df["Label"].nunique() < 2:
    raise ValueError(
        "Less than 2 classes remain."
    )

print()
print(
    "[OK] Final dataset validation passed."
)


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

print()
print("=" * 70)
print("SAVING CLEAN DATASET")
print("=" * 70)

print(
    "Saving cleaned CIC-IDS2018 dataset..."
)

save_start = time.time()

merged_df.to_csv(
    OUTPUT_PATH,
    index=False
)

save_time = time.time() - save_start

print(
    "[OK] Clean dataset saved:"
)

print(OUTPUT_PATH)

print(
    f"Save time: "
    f"{save_time:.2f} sec"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

print()
print("=" * 70)
print("SAVING STEP 12 SUMMARY")
print("=" * 70)

summary = {

    "step": 12,

    "dataset": "CIC-IDS2018",

    "input_files": len(csv_files),

    "input_rows": int(
        total_original_rows
    ),

    "final_rows": int(
        final_rows
    ),

    "final_columns": int(
        final_columns
    ),

    "feature_count": int(
        len(feature_columns)
    ),

    "num_classes": int(
        merged_df["Label"].nunique()
    ),

    "classes": [
        str(x)
        for x in sorted(
            merged_df["Label"].unique()
        )
    ],

    "missing_values": int(
        missing_values
    ),

    "infinity_values": int(
        infinity_values
    ),

    "duplicate_rows": int(
        duplicate_rows
    ),

    "removed_header_rows": int(
        total_removed_header_rows
    ),

    "replaced_invalid_numeric_values": int(
        total_replaced_nan
    ),

    "duplicates_removed_within_files": int(
        total_removed_duplicates
    ),

    "final_duplicate_rows_removed_after_merge": int(
        final_duplicates_removed
    ),

    "feature_selection": False,

    "train_validation_test_split": False,

    "smote": False,

    "model_training": False,

    "output_file": OUTPUT_PATH
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
    "[OK] Summary saved:"
)

print(SUMMARY_PATH)


# ============================================================
# CLEANUP
# ============================================================

del merged_df

gc.collect()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("STEP 12 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Dataset:
CIC-IDS2018

Input:
9 raw CICFlowMeter CSV files

Output:
CICIDS2018_clean.csv

Cleaning completed:
- Header/Label rows removed
- Numeric conversion completed
- Infinity handled
- Missing/invalid numeric values replaced with 0
- Duplicate rows removed
- Labels standardized
- Final validation completed

No:
- Feature selection
- Train/Validation/Test split
- SMOTE
- Model training

Next:
STEP 13 - CIC-IDS2018 Train / Validation / Test Split
""")

print("=" * 70)