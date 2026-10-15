import pandas as pd
import numpy as np
import os

print("=" * 70)
print("NSL-KDD DATASET PREPROCESSING")
print("=" * 70)

# ============================================================
# 1. FILE PATHS
# ============================================================

train_file = "data/KDDTrain+.txt"
test_file = "data/KDDTest+.txt"

# ============================================================
# 2. NSL-KDD COLUMN NAMES
# ============================================================

columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty"
]

# ============================================================
# 3. CHECK FILES
# ============================================================

if not os.path.exists(train_file):
    print("ERROR: KDDTrain+.txt not found!")
    print("Please put KDDTrain+.txt inside data folder.")
    exit()

if not os.path.exists(test_file):
    print("ERROR: KDDTest+.txt not found!")
    print("Please put KDDTest+.txt inside data folder.")
    exit()

# ============================================================
# 4. LOAD TRAINING DATA
# ============================================================

print("\nLoading KDDTrain+.txt...")

train_df = pd.read_csv(
    train_file,
    header=None,
    names=columns
)

print("Training dataset shape:", train_df.shape)

# ============================================================
# 5. LOAD TEST DATA
# ============================================================

print("\nLoading KDDTest+.txt...")

test_df = pd.read_csv(
    test_file,
    header=None,
    names=columns
)

print("Testing dataset shape:", test_df.shape)

# ============================================================
# 6. COMBINE TRAIN + TEST
# ============================================================

print("\nCombining training and testing datasets...")

df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)

print("Combined dataset shape:", df.shape)

# ============================================================
# 7. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

print(
    "Total missing values:",
    df.isnull().sum().sum()
)

# ============================================================
# 9. REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE REMOVAL")
print("=" * 70)

print(
    "Duplicates before removal:",
    df.duplicated().sum()
)

df = df.drop_duplicates()

print(
    "Shape after duplicate removal:",
    df.shape
)

# ============================================================
# 10. CHECK LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL LABEL DISTRIBUTION")
print("=" * 70)

print(df["label"].value_counts())

# ============================================================
# 11. CLEAN LABEL
# ============================================================

df["label"] = df["label"].astype(str).str.strip()

# Remove trailing period if present
df["label"] = df["label"].str.replace(
    ".", "",
    regex=False
)

# ============================================================
# 12. BINARY CLASSIFICATION
# ============================================================

print("\n" + "=" * 70)
print("CONVERTING LABELS TO BINARY")
print("=" * 70)

df["Label"] = df["label"].apply(
    lambda x: 0 if x.lower() == "normal" else 1
)

# Remove original label
df = df.drop(columns=["label"])

print("\nBinary label distribution:")

print(df["Label"].value_counts())

# ============================================================
# 13. REMOVE DIFFICULTY COLUMN
# ============================================================

if "difficulty" in df.columns:

    df = df.drop(
        columns=["difficulty"]
    )

    print(
        "\nDifficulty column removed."
    )

# ============================================================
# 14. CHECK DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ============================================================
# 15. SAVE COMBINED CLEAN DATASET
# ============================================================

output_file = "data/NSLKDD_combined_clean.csv"

df.to_csv(
    output_file,
    index=False
)

# ============================================================
# 16. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("NSL-KDD PREPROCESSING COMPLETED")
print("=" * 70)

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal columns:")

for i, column in enumerate(df.columns, 1):
    print(i, ":", column)

print("\nFinal label distribution:")
print(df["Label"].value_counts())

print("\nDataset saved to:")
print(output_file)

print("\n" + "=" * 70)