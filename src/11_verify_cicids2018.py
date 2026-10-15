"""
STEP 11 - CIC-IDS2018 DATASET VERIFICATION

Purpose:
    Verify all downloaded CIC-IDS2018 CSV files before preprocessing.

Checks:
    1. Number of CSV files
    2. File sizes
    3. CSV readability
    4. Column consistency
    5. Label column
    6. Number of rows
    7. Attack classes
    8. Missing values
"""

import os
import glob
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = r"C:\Users\DELL\OneDrive\Desktop\network_attack_detection"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "cicids2018"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 11 - CIC-IDS2018 DATASET VERIFICATION")
print("=" * 70)

print()
print("Dataset folder:")
print(DATA_DIR)


# ============================================================
# FIND CSV FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING CSV FILES")
print("=" * 70)

csv_files = sorted(
    glob.glob(
        os.path.join(DATA_DIR, "*.csv")
    )
)

print(
    f"CSV files found: {len(csv_files)}"
)

if len(csv_files) == 0:
    raise FileNotFoundError(
        "No CSV files found in data/cicids2018/"
    )

for i, file_path in enumerate(csv_files, start=1):

    file_size_mb = (
        os.path.getsize(file_path)
        / (1024 * 1024)
    )

    print(
        f"{i:02d}. "
        f"{os.path.basename(file_path)} "
        f"({file_size_mb:.2f} MB)"
    )


# ============================================================
# EXPECTED FILE COUNT
# ============================================================

print()
print("=" * 70)
print("FILE COUNT CHECK")
print("=" * 70)

if len(csv_files) == 9:

    print("[OK] Exactly 9 CSV files found.")

else:

    print(
        f"[WARNING] Expected 9 CSV files, "
        f"but found {len(csv_files)}."
    )


# ============================================================
# READ HEADERS
# ============================================================

print()
print("=" * 70)
print("CHECKING CSV HEADERS")
print("=" * 70)

reference_columns = None
reference_file = None

for file_path in csv_files:

    print()
    print(
        "Checking:",
        os.path.basename(file_path)
    )

    try:

        df_sample = pd.read_csv(
            file_path,
            nrows=5
        )

        columns = list(
            df_sample.columns
        )

        print(
            f"Columns found: {len(columns)}"
        )

        # First file becomes reference
        if reference_columns is None:

            reference_columns = columns
            reference_file = file_path

            print(
                "[OK] Reference columns established."
            )

        else:

            if columns == reference_columns:

                print(
                    "[OK] Columns match reference."
                )

            else:

                print(
                    "[ERROR] Columns DO NOT match!"
                )

    except Exception as e:

        print(
            "[ERROR] Could not read file."
        )

        print(
            "Reason:",
            e
        )


# ============================================================
# LABEL CHECK
# ============================================================

print()
print("=" * 70)
print("CHECKING LABEL COLUMN")
print("=" * 70)

if reference_columns is None:

    raise RuntimeError(
        "Could not establish reference columns."
    )

if "Label" in reference_columns:

    print(
        "[OK] Label column exists."
    )

else:

    raise ValueError(
        "Label column not found!"
    )


# ============================================================
# COUNT ROWS AND LABELS
# ============================================================

print()
print("=" * 70)
print("CHECKING ROWS AND LABELS")
print("=" * 70)

total_rows = 0

all_labels = set()

file_results = []

for file_path in csv_files:

    filename = os.path.basename(
        file_path
    )

    print()
    print(
        "Processing:",
        filename
    )

    try:

        # Read only Label column
        label_df = pd.read_csv(
            file_path,
            usecols=["Label"]
        )

        row_count = len(
            label_df
        )

        labels = (
            label_df["Label"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        total_rows += row_count

        all_labels.update(
            labels
        )

        print(
            f"Rows: {row_count:,}"
        )

        print(
            f"Labels: {len(labels)}"
        )

        print(
            "Classes:"
        )

        for label in sorted(labels):

            print(
                f"  - {label}"
            )

        file_results.append(
            {
                "file": filename,
                "rows": row_count,
                "classes": len(labels),
            }
        )

        del label_df

    except Exception as e:

        print(
            "[ERROR] Could not process file."
        )

        print(
            "Reason:",
            e
        )


# ============================================================
# TOTAL DATA
# ============================================================

print()
print("=" * 70)
print("TOTAL DATASET INFORMATION")
print("=" * 70)

print(
    f"Total rows across all files: "
    f"{total_rows:,}"
)

print(
    f"Total unique classes: "
    f"{len(all_labels)}"
)

print()
print("All classes found:")

for label in sorted(all_labels):

    print(
        f"  - {label}"
    )


# ============================================================
# MISSING VALUE CHECK
# ============================================================

print()
print("=" * 70)
print("CHECKING MISSING VALUES")
print("=" * 70)

print(
    "Checking Label column only..."
)

total_missing_labels = 0

for file_path in csv_files:

    filename = os.path.basename(
        file_path
    )

    try:

        label_df = pd.read_csv(
            file_path,
            usecols=["Label"]
        )

        missing = (
            label_df["Label"]
            .isna()
            .sum()
        )

        total_missing_labels += missing

        print(
            f"{filename}: "
            f"{missing:,} missing labels"
        )

        del label_df

    except Exception as e:

        print(
            f"[ERROR] {filename}: {e}"
        )


if total_missing_labels == 0:

    print()
    print(
        "[OK] No missing Label values found."
    )

else:

    print()
    print(
        f"[WARNING] Total missing labels: "
        f"{total_missing_labels:,}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 11 VERIFICATION SUMMARY")
print("=" * 70)

print(
    f"CSV files              : {len(csv_files)}"
)

print(
    f"Total rows             : {total_rows:,}"
)

print(
    f"Total classes          : {len(all_labels)}"
)

print(
    f"Label column           : {'YES' if 'Label' in reference_columns else 'NO'}"
)

print(
    f"Missing labels         : {total_missing_labels:,}"
)

print()
print("Reference file:")

print(
    os.path.basename(reference_file)
)

print()
print("Column count:")

print(
    len(reference_columns)
)

print()
print("=" * 70)
print("STEP 11 VERIFICATION COMPLETED")
print("=" * 70)

print("""
IMPORTANT:
No files were modified.
No data was merged.
No preprocessing was performed.

Next:
CIC-IDS2018 preprocessing.
""")

print("=" * 70)
