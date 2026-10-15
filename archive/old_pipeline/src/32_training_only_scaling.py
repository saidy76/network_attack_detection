import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


print("=" * 70)
print("STEP 32 - TRAINING-ONLY FEATURE SCALING")
print("=" * 70)


# ================================================================
# PATHS
# ================================================================

CIC_TRAIN_PATH = "data/CICIDS2017_model_train.csv"
CIC_TEST_PATH = "data/CICIDS2017_cross_dataset_common.csv"

NSL_TRAIN_PATH = "data/NSLKDD_model_train.csv"
NSL_TEST_PATH = "data/NSLKDD_cross_dataset_common.csv"

os.makedirs("models", exist_ok=True)


# ================================================================
# COMMON FEATURES
# ================================================================

COMMON_FEATURES = [
    "forward_bytes",
    "backward_bytes",
    "total_bytes",
    "traffic_ratio"
]

TARGET = "Label"


# ================================================================
# LOAD DATA
# ================================================================

print("\nLoading training and testing datasets...")

cic_train = pd.read_csv(CIC_TRAIN_PATH)
cic_test = pd.read_csv(CIC_TEST_PATH)

nsl_train = pd.read_csv(NSL_TRAIN_PATH)
nsl_test = pd.read_csv(NSL_TEST_PATH)


print("\nOriginal shapes:")

print("CICIDS2017 Train:", cic_train.shape)
print("CICIDS2017 Test :", cic_test.shape)

print("NSL-KDD Train   :", nsl_train.shape)
print("NSL-KDD Test    :", nsl_test.shape)


# ================================================================
# FEATURE VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("VALIDATING SOURCE FEATURES")
print("=" * 70)


CIC_REQUIRED = [
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets"
]

NSL_REQUIRED = [
    "src_bytes",
    "dst_bytes"
]


for feature in CIC_REQUIRED:

    if feature not in cic_train.columns:
        raise ValueError(
            f"CICIDS2017 training feature missing: {feature}"
        )

    print(f"[OK] CICIDS2017: {feature}")


for feature in NSL_REQUIRED:

    if feature not in nsl_train.columns:
        raise ValueError(
            f"NSL-KDD training feature missing: {feature}"
        )

    print(f"[OK] NSL-KDD: {feature}")


for feature in COMMON_FEATURES:

    if feature not in cic_test.columns:
        raise ValueError(
            f"CICIDS2017 test common feature missing: {feature}"
        )

    if feature not in nsl_test.columns:
        raise ValueError(
            f"NSL-KDD test common feature missing: {feature}"
        )


# ================================================================
# CREATE COMMON FEATURES FOR TRAINING DATA
# ================================================================

print("\n" + "=" * 70)
print("CREATING COMMON FEATURES FOR TRAINING DATA")
print("=" * 70)


# ------------------------------------------------
# CICIDS2017 TRAIN
# ------------------------------------------------

print("\nProcessing CICIDS2017 training data...")


cic_train_common = pd.DataFrame()

cic_train_common["forward_bytes"] = (
    pd.to_numeric(
        cic_train["Total Length of Fwd Packets"],
        errors="coerce"
    )
)

cic_train_common["backward_bytes"] = (
    pd.to_numeric(
        cic_train["Total Length of Bwd Packets"],
        errors="coerce"
    )
)

cic_train_common["total_bytes"] = (
    cic_train_common["forward_bytes"]
    + cic_train_common["backward_bytes"]
)

cic_train_common["traffic_ratio"] = (
    cic_train_common["forward_bytes"]
    / (cic_train_common["backward_bytes"] + 1)
)

cic_train_common[TARGET] = cic_train[TARGET].values


# ------------------------------------------------
# NSL-KDD TRAIN
# ------------------------------------------------

print("Processing NSL-KDD training data...")


nsl_train_common = pd.DataFrame()

nsl_train_common["forward_bytes"] = (
    pd.to_numeric(
        nsl_train["src_bytes"],
        errors="coerce"
    )
)

nsl_train_common["backward_bytes"] = (
    pd.to_numeric(
        nsl_train["dst_bytes"],
        errors="coerce"
    )
)

nsl_train_common["total_bytes"] = (
    nsl_train_common["forward_bytes"]
    + nsl_train_common["backward_bytes"]
)

nsl_train_common["traffic_ratio"] = (
    nsl_train_common["forward_bytes"]
    / (nsl_train_common["backward_bytes"] + 1)
)

nsl_train_common[TARGET] = nsl_train[TARGET].values


# ================================================================
# DATA CLEANING
# ================================================================

print("\n" + "=" * 70)
print("CLEANING TRAINING DATA")
print("=" * 70)


def clean_dataset(df, name):

    print(f"\n{name}")

    before = len(df)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=COMMON_FEATURES + [TARGET]
    )

    # Remove negative byte values if any
    for feature in [
        "forward_bytes",
        "backward_bytes",
        "total_bytes",
        "traffic_ratio"
    ]:
        df = df[df[feature] >= 0]

    after = len(df)

    print("Rows before cleaning:", before)
    print("Rows after cleaning :", after)
    print("Rows removed        :", before - after)

    print(
        "Missing values:",
        int(df[COMMON_FEATURES].isnull().sum().sum())
    )

    print(
        "Infinity values:",
        int(
            np.isinf(
                df[COMMON_FEATURES].values
            ).sum()
        )
    )

    return df.reset_index(drop=True)


cic_train_common = clean_dataset(
    cic_train_common,
    "CICIDS2017 TRAIN"
)

nsl_train_common = clean_dataset(
    nsl_train_common,
    "NSL-KDD TRAIN"
)


# ================================================================
# LOG1P TRANSFORMATION
# ================================================================

print("\n" + "=" * 70)
print("APPLYING LOG1P TRANSFORMATION")
print("=" * 70)

print("\nTraining data only.")

for feature in COMMON_FEATURES:

    cic_train_common[feature] = np.log1p(
        cic_train_common[feature]
    )

    nsl_train_common[feature] = np.log1p(
        nsl_train_common[feature]
    )

    print(f"[LOG1P] {feature}")


# ================================================================
# PREPARE TEST DATA
# ================================================================

print("\n" + "=" * 70)
print("PREPARING TEST DATA")
print("=" * 70)


cic_test_common = cic_test[
    COMMON_FEATURES + [TARGET]
].copy()

nsl_test_common = nsl_test[
    COMMON_FEATURES + [TARGET]
].copy()


# Clean test data WITHOUT fitting anything
def clean_test_dataset(df, name):

    print(f"\n{name}")

    before = len(df)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=COMMON_FEATURES + [TARGET]
    )

    for feature in [
        "forward_bytes",
        "backward_bytes",
        "total_bytes",
        "traffic_ratio"
    ]:
        df = df[df[feature] >= 0]

    after = len(df)

    print("Rows before cleaning:", before)
    print("Rows after cleaning :", after)
    print("Rows removed        :", before - after)

    return df.reset_index(drop=True)


cic_test_common = clean_test_dataset(
    cic_test_common,
    "CICIDS2017 TEST"
)

nsl_test_common = clean_test_dataset(
    nsl_test_common,
    "NSL-KDD TEST"
)


# ================================================================
# APPLY LOG1P TO TEST
# ================================================================

print("\nApplying log1p transformation to test data...")

for feature in COMMON_FEATURES:

    cic_test_common[feature] = np.log1p(
        cic_test_common[feature]
    )

    nsl_test_common[feature] = np.log1p(
        nsl_test_common[feature]
    )


# ================================================================
# STANDARD SCALING
# ================================================================

print("\n" + "=" * 70)
print("STANDARD SCALING")
print("=" * 70)


print("\nIMPORTANT:")
print("Scaler is FIT ONLY on training data.")
print("Test data is TRANSFORMED using the training scaler.")


# ------------------------------------------------
# CICIDS2017 SCALER
# ------------------------------------------------

print("\nFitting CICIDS2017 scaler...")

cic_scaler = StandardScaler()

cic_scaler.fit(
    cic_train_common[COMMON_FEATURES]
)

cic_train_scaled = cic_scaler.transform(
    cic_train_common[COMMON_FEATURES]
)

cic_test_scaled = cic_scaler.transform(
    cic_test_common[COMMON_FEATURES]
)


# ------------------------------------------------
# NSL-KDD SCALER
# ------------------------------------------------

print("Fitting NSL-KDD scaler...")

nsl_scaler = StandardScaler()

nsl_scaler.fit(
    nsl_train_common[COMMON_FEATURES]
)

nsl_train_scaled = nsl_scaler.transform(
    nsl_train_common[COMMON_FEATURES]
)

nsl_test_scaled = nsl_scaler.transform(
    nsl_test_common[COMMON_FEATURES]
)


# ================================================================
# CREATE FINAL DATAFRAMES
# ================================================================

print("\n" + "=" * 70)
print("CREATING SCALED DATASETS")
print("=" * 70)


cic_train_final = pd.DataFrame(
    cic_train_scaled,
    columns=COMMON_FEATURES
)

cic_train_final[TARGET] = (
    cic_train_common[TARGET].values
)


cic_test_final = pd.DataFrame(
    cic_test_scaled,
    columns=COMMON_FEATURES
)

cic_test_final[TARGET] = (
    cic_test_common[TARGET].values
)


nsl_train_final = pd.DataFrame(
    nsl_train_scaled,
    columns=COMMON_FEATURES
)

nsl_train_final[TARGET] = (
    nsl_train_common[TARGET].values
)


nsl_test_final = pd.DataFrame(
    nsl_test_scaled,
    columns=COMMON_FEATURES
)

nsl_test_final[TARGET] = (
    nsl_test_common[TARGET].values
)


# ================================================================
# VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


datasets = {
    "CICIDS2017 Train": cic_train_final,
    "CICIDS2017 Test": cic_test_final,
    "NSL-KDD Train": nsl_train_final,
    "NSL-KDD Test": nsl_test_final
}


for name, df in datasets.items():

    print(f"\n{name}")

    print("Shape:", df.shape)

    print(
        "Missing values:",
        int(df[COMMON_FEATURES].isnull().sum().sum())
    )

    print(
        "Infinity values:",
        int(
            np.isinf(
                df[COMMON_FEATURES].values
            ).sum()
        )
    )


# ================================================================
# CHECK STANDARDIZATION
# ================================================================

print("\n" + "=" * 70)
print("TRAINING FEATURE STATISTICS AFTER SCALING")
print("=" * 70)


print("\nCICIDS2017 TRAIN:")
print(
    cic_train_final[COMMON_FEATURES].agg(
        ["mean", "std"]
    ).round(4)
)


print("\nNSL-KDD TRAIN:")
print(
    nsl_train_final[COMMON_FEATURES].agg(
        ["mean", "std"]
    ).round(4)
)


# ================================================================
# LABEL DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)


print("\nCICIDS2017 TRAIN:")
print(cic_train_final[TARGET].value_counts())


print("\nCICIDS2017 TEST:")
print(cic_test_final[TARGET].value_counts())


print("\nNSL-KDD TRAIN:")
print(nsl_train_final[TARGET].value_counts())


print("\nNSL-KDD TEST:")
print(nsl_test_final[TARGET].value_counts())


# ================================================================
# SAVE DATASETS
# ================================================================

print("\n" + "=" * 70)
print("SAVING SCALED DATASETS")
print("=" * 70)


cic_train_path = "data/CICIDS2017_scaled_train.csv"
cic_test_path = "data/CICIDS2017_scaled_test.csv"

nsl_train_path = "data/NSLKDD_scaled_train.csv"
nsl_test_path = "data/NSLKDD_scaled_test.csv"


cic_train_final.to_csv(
    cic_train_path,
    index=False
)

cic_test_final.to_csv(
    cic_test_path,
    index=False
)

nsl_train_final.to_csv(
    nsl_train_path,
    index=False
)

nsl_test_final.to_csv(
    nsl_test_path,
    index=False
)


# ================================================================
# SAVE SCALERS
# ================================================================

joblib.dump(
    cic_scaler,
    "models/CICIDS2017_standard_scaler.joblib"
)

joblib.dump(
    nsl_scaler,
    "models/NSLKDD_standard_scaler.joblib"
)


# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("STEP 32 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nScaled datasets saved to:")

print("CICIDS2017 Train:")
print(cic_train_path)

print("\nCICIDS2017 Test:")
print(cic_test_path)

print("\nNSL-KDD Train:")
print(nsl_train_path)

print("\nNSL-KDD Test:")
print(nsl_test_path)

print("\nScalers saved to:")

print("models/CICIDS2017_standard_scaler.joblib")
print("models/NSLKDD_standard_scaler.joblib")

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print("""
1. log1p transformation was applied to train and test.
2. StandardScaler was FIT ONLY on training data.
3. Test data was NOT used to fit the scaler.
4. No SMOTE was applied in this step.
5. Four common semantic features are used.
6. Label was never scaled.
7. This prevents test-set data leakage.
""")