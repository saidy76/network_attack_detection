import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

print("=" * 70)
print("STEP 28 - PREPARING MODELING DATASETS")
print("=" * 70)


# ============================================================
# FILE PATHS
# ============================================================

CICIDS_FILE = "data/CICIDS2017_final_features.csv"
NSLKDD_FILE = "data/NSLKDD_final_features.csv"

# Output files
CICIDS_TRAIN = "data/CICIDS2017_model_train.csv"
CICIDS_TEST = "data/CICIDS2017_model_test.csv"

NSLKDD_TRAIN = "data/NSLKDD_model_train.csv"
NSLKDD_TEST = "data/NSLKDD_model_test.csv"

# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading CICIDS2017...")
cicids = pd.read_csv(CICIDS_FILE)

print("Loading NSL-KDD...")
nslkdd = pd.read_csv(NSLKDD_FILE)

print("\nOriginal dataset shapes:")
print("CICIDS2017:", cicids.shape)
print("NSL-KDD:", nslkdd.shape)


# ============================================================
# REQUIRED FEATURE CHECK
# ============================================================

cicids_features = [
    "Average Packet Size",
    "Bwd Packet Length Mean",
    "Packet Length Variance",
    "Destination Port",
    "Bwd Packet Length Std",
    "Packet Length Std",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Total Length of Fwd Packets",
    "Avg Fwd Segment Size",
    "Fwd IAT Max",
    "Fwd Header Length.1",
    "Bwd Header Length"
]

nslkdd_features = [
    "src_bytes",
    "dst_bytes",
    "flag",
    "service",
    "diff_srv_rate",
    "same_srv_rate",
    "dst_host_srv_count",
    "logged_in",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "count",
    "dst_host_srv_serror_rate",
    "dst_host_serror_rate",
    "srv_serror_rate",
    "serror_rate"
]


# ============================================================
# CHECK FEATURES
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING CICIDS2017 FEATURES")
print("=" * 70)

missing_cicids = [
    f for f in cicids_features
    if f not in cicids.columns
]

if missing_cicids:
    print("\nMissing CICIDS2017 features:")
    for f in missing_cicids:
        print("-", f)

    raise ValueError(
        "CICIDS2017 required features are missing."
    )

print("[OK] CICIDS2017 features found.")


print("\n" + "=" * 70)
print("VALIDATING NSL-KDD FEATURES")
print("=" * 70)

missing_nslkdd = [
    f for f in nslkdd_features
    if f not in nslkdd.columns
]

if missing_nslkdd:
    print("\nMissing NSL-KDD features:")
    for f in missing_nslkdd:
        print("-", f)

    raise ValueError(
        "NSL-KDD required features are missing."
    )

print("[OK] NSL-KDD features found.")


# ============================================================
# KEEP ONLY SELECTED FEATURES + LABEL
# ============================================================

cicids_model = cicids[
    cicids_features + ["Label"]
].copy()

nslkdd_model = nslkdd[
    nslkdd_features + ["Label"]
].copy()


# ============================================================
# DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)


def clean_dataset(df, name):

    print(f"\nCleaning {name}...")

    # Replace infinity
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    before = len(df)

    # Remove rows containing missing values
    df = df.dropna()

    after = len(df)

    print("Rows before cleaning:", before)
    print("Rows after cleaning :", after)
    print("Rows removed        :", before - after)

    print(
        "Missing values:",
        df.isnull().sum().sum()
    )

    print(
        "Infinity values:",
        np.isinf(
            df.select_dtypes(include=np.number)
        ).sum().sum()
    )

    return df


cicids_model = clean_dataset(
    cicids_model,
    "CICIDS2017"
)

nslkdd_model = clean_dataset(
    nslkdd_model,
    "NSL-KDD"
)


# ============================================================
# LABEL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("LABEL VALIDATION")
print("=" * 70)

print("\nCICIDS2017:")
print(cicids_model["Label"].value_counts())

print("\nNSL-KDD:")
print(nslkdd_model["Label"].value_counts())


# ============================================================
# TRAIN / TEST SPLIT FUNCTION
# ============================================================

def split_dataset(df, name):

    print("\n" + "=" * 70)
    print(f"TRAIN / TEST SPLIT - {name}")
    print("=" * 70)

    X = df.drop(
        columns=["Label"]
    )

    y = df["Label"]

    print("\nFeature shape:")
    print(X.shape)

    print("\nLabel shape:")
    print(y.shape)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    train_df = X_train.copy()
    train_df["Label"] = y_train.values

    test_df = X_test.copy()
    test_df["Label"] = y_test.values

    print("\nTraining shape:")
    print(train_df.shape)

    print("Testing shape:")
    print(test_df.shape)

    print("\nTraining label distribution:")
    print(
        train_df["Label"].value_counts()
    )

    print("\nTesting label distribution:")
    print(
        test_df["Label"].value_counts()
    )

    print("\nTraining percentage:")
    print(
        train_df["Label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nTesting percentage:")
    print(
        test_df["Label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    return train_df, test_df


# ============================================================
# SPLIT CICIDS2017
# ============================================================

cicids_train, cicids_test = split_dataset(
    cicids_model,
    "CICIDS2017"
)


# ============================================================
# SPLIT NSL-KDD
# ============================================================

nslkdd_train, nslkdd_test = split_dataset(
    nslkdd_model,
    "NSL-KDD"
)


# ============================================================
# SAVE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODELING DATASETS")
print("=" * 70)

cicids_train.to_csv(
    CICIDS_TRAIN,
    index=False
)

cicids_test.to_csv(
    CICIDS_TEST,
    index=False
)

nslkdd_train.to_csv(
    NSLKDD_TRAIN,
    index=False
)

nslkdd_test.to_csv(
    NSLKDD_TEST,
    index=False
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

datasets = {
    "CICIDS2017 Train": cicids_train,
    "CICIDS2017 Test": cicids_test,
    "NSL-KDD Train": nslkdd_train,
    "NSL-KDD Test": nslkdd_test
}

for name, df in datasets.items():

    missing = df.isnull().sum().sum()

    infinity = np.isinf(
        df.select_dtypes(include=np.number)
    ).sum().sum()

    print(f"\n{name}")
    print("Shape:", df.shape)
    print("Missing values:", missing)
    print("Infinity values:", infinity)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 28 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSaved files:")

print("CICIDS2017 training:")
print(CICIDS_TRAIN)

print("\nCICIDS2017 testing:")
print(CICIDS_TEST)

print("\nNSL-KDD training:")
print(NSLKDD_TRAIN)

print("\nNSL-KDD testing:")
print(NSLKDD_TEST)

print("\nIMPORTANT:")
print("Stratified 80/20 train-test split was used.")
print("SMOTE has NOT been applied in this step.")
print("SMOTE will only be applied to training data.")
print("Test data will remain untouched.")