"""
STEP 2 - CICIoT2023 DATASET AUDIT

Purpose:
- Discover all CICIoT2023 CSV files
- Inspect dataset structure
- Count rows and columns
- Identify label column
- Analyze attack classes
- Check missing values
- Check infinite values
- Check duplicate rows
- Inspect data types
- Save an audit report

No preprocessing is performed in this step.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "CICIoT2023"
REPORT_DIR = PROJECT_ROOT / "experiments"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 2 - CICIoT2023 DATASET AUDIT")
print("=" * 70)


# ============================================================
# CHECK DATASET DIRECTORY
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATASET DIRECTORY")
print("=" * 70)

if not RAW_DIR.exists():
    raise FileNotFoundError(
        f"\nCICIoT2023 directory not found:\n{RAW_DIR}\n\n"
        "Please place the extracted CSV files inside:\n"
        "data/raw/CICIoT2023/"
    )

print(f"[OK] Dataset directory found:")
print(RAW_DIR)


# ============================================================
# FIND CSV FILES
# ============================================================

csv_files = sorted(RAW_DIR.glob("*.csv"))

print("\n" + "=" * 70)
print("CSV FILE DISCOVERY")
print("=" * 70)

print(f"CSV files found: {len(csv_files)}")

if len(csv_files) == 0:
    raise FileNotFoundError(
        "No CSV files found inside data/raw/CICIoT2023/"
    )

for i, file in enumerate(csv_files, start=1):
    print(f"{i:02d}. {file.name}")


# ============================================================
# BASIC FILE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FILE SIZE INFORMATION")
print("=" * 70)

file_info = []

for file in csv_files:
    size_mb = file.stat().st_size / (1024 ** 2)

    file_info.append({
        "file": file.name,
        "size_MB": round(size_mb, 2)
    })

    print(f"{file.name:20s} : {size_mb:10.2f} MB")


file_info_df = pd.DataFrame(file_info)


# ============================================================
# INSPECT FIRST FILE
# ============================================================

print("\n" + "=" * 70)
print("INSPECTING FIRST CSV")
print("=" * 70)

first_file = csv_files[0]

sample_df = pd.read_csv(first_file, nrows=5000)

print(f"File: {first_file.name}")
print(f"Rows inspected: {len(sample_df)}")
print(f"Columns: {len(sample_df.columns)}")

print("\nColumn names:")

for i, column in enumerate(sample_df.columns, start=1):
    print(f"{i:02d}. {column}")


# ============================================================
# IDENTIFY LABEL COLUMN
# ============================================================

print("\n" + "=" * 70)
print("LABEL COLUMN DETECTION")
print("=" * 70)

possible_labels = [
    "label",
    "Label",
    "LABEL",
    "attack",
    "Attack",
    "class",
    "Class"
]

label_candidates = [
    col for col in sample_df.columns
    if col in possible_labels
]

if len(label_candidates) == 0:
    print("[WARNING] No obvious label column found.")
    label_column = None
else:
    print("Possible label columns:")

    for col in label_candidates:
        print(f"[FOUND] {col}")

    label_column = label_candidates[0]

    print(f"\nSelected label column: {label_column}")


# ============================================================
# DATA TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPE ANALYSIS")
print("=" * 70)

dtype_summary = pd.DataFrame({
    "column": sample_df.columns,
    "dtype": sample_df.dtypes.astype(str).values
})

print(dtype_summary.to_string(index=False))


# ============================================================
# SAMPLE MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE MISSING VALUE CHECK")
print("=" * 70)

missing_sample = sample_df.isna().sum()

total_missing_sample = int(missing_sample.sum())

print(f"Total missing values in inspected sample: {total_missing_sample}")

if total_missing_sample > 0:
    print("\nColumns containing missing values:")

    print(
        missing_sample[missing_sample > 0]
        .sort_values(ascending=False)
        .to_string()
    )
else:
    print("[OK] No missing values found in inspected sample.")


# ============================================================
# SAMPLE INFINITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE INFINITY CHECK")
print("=" * 70)

numeric_sample = sample_df.select_dtypes(include=[np.number])

if numeric_sample.shape[1] > 0:

    infinity_count = np.isinf(numeric_sample).sum().sum()

    print(
        f"Infinity values in numeric sample: "
        f"{int(infinity_count)}"
    )

    if infinity_count == 0:
        print("[OK] No infinity values found.")

else:

    print("No numeric columns detected in sample.")


# ============================================================
# SAMPLE DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE DUPLICATE CHECK")
print("=" * 70)

duplicate_count = int(sample_df.duplicated().sum())

print(
    f"Duplicate rows in inspected sample: "
    f"{duplicate_count}"
)


# ============================================================
# LABEL DISTRIBUTION FROM SAMPLE
# ============================================================

if label_column is not None:

    print("\n" + "=" * 70)
    print("SAMPLE LABEL DISTRIBUTION")
    print("=" * 70)

    label_counts = sample_df[label_column].value_counts(
        dropna=False
    )

    print(label_counts.to_string())

    print(
        f"\nNumber of unique labels in sample: "
        f"{sample_df[label_column].nunique(dropna=False)}"
    )


# ============================================================
# READ ALL FILES FOR LABEL AUDIT
# ============================================================

print("\n" + "=" * 70)
print("FULL DATASET LABEL AUDIT")
print("=" * 70)

all_labels = []

total_rows = 0

file_row_counts = []

for i, file in enumerate(csv_files, start=1):

    print(
        f"\n[{i}/{len(csv_files)}] Reading "
        f"{file.name} ..."
    )

    df = pd.read_csv(file)

    rows = len(df)

    total_rows += rows

    file_row_counts.append({
        "file": file.name,
        "rows": rows,
        "columns": len(df.columns)
    })

    if label_column is not None and label_column in df.columns:

        labels = df[label_column].value_counts(
            dropna=False
        )

        for label, count in labels.items():

            all_labels.append({
                "file": file.name,
                "label": str(label),
                "count": int(count)
            })

    del df


# ============================================================
# TOTAL DATASET SIZE
# ============================================================

print("\n" + "=" * 70)
print("TOTAL DATASET SIZE")
print("=" * 70)

print(f"Total CSV files : {len(csv_files)}")
print(f"Total rows      : {total_rows:,}")


# ============================================================
# COMBINED LABEL DISTRIBUTION
# ============================================================

if len(all_labels) > 0:

    label_df = pd.DataFrame(all_labels)

    combined_labels = (
        label_df
        .groupby("label", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )

    print("\n" + "=" * 70)
    print("COMBINED LABEL DISTRIBUTION")
    print("=" * 70)

    print(combined_labels.to_string(index=False))

    print(
        f"\nTotal unique attack/traffic classes: "
        f"{len(combined_labels)}"
    )

else:

    combined_labels = pd.DataFrame()

    print(
        "[WARNING] Could not create combined label distribution."
    )


# ============================================================
# FILE ROW SUMMARY
# ============================================================

file_rows_df = pd.DataFrame(file_row_counts)

print("\n" + "=" * 70)
print("ROWS PER FILE")
print("=" * 70)

print(file_rows_df.to_string(index=False))


# ============================================================
# SAVE REPORTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING AUDIT REPORTS")
print("=" * 70)

file_info_df.to_csv(
    REPORT_DIR / "ciciot2023_file_information.csv",
    index=False
)

file_rows_df.to_csv(
    REPORT_DIR / "ciciot2023_file_row_counts.csv",
    index=False
)

dtype_summary.to_csv(
    REPORT_DIR / "ciciot2023_sample_dtypes.csv",
    index=False
)

if len(combined_labels) > 0:

    combined_labels.to_csv(
        REPORT_DIR / "ciciot2023_label_distribution.csv",
        index=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nDataset:")
print("CICIoT2023")

print(f"\nCSV files: {len(csv_files)}")
print(f"Total rows: {total_rows:,}")

if len(combined_labels) > 0:
    print(
        f"Number of classes: "
        f"{len(combined_labels)}"
    )

print("\nReports saved to:")
print(REPORT_DIR)

print("\nIMPORTANT:")
print("No data preprocessing was performed.")
print("No rows were removed.")
print("No SMOTE was applied.")
print("No scaling was applied.")
print("No feature selection was performed.")

print(
    "\nThe purpose of this step is only to "
    "understand and validate the raw dataset."
)