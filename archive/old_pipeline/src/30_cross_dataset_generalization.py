import pandas as pd
import numpy as np
from pathlib import Path

# ================================================================
# STEP 30 - ROBUST CROSS-DATASET COMMON REPRESENTATION
# ================================================================

print("=" * 70)
print("STEP 30 - ROBUST CROSS-DATASET COMMON REPRESENTATION")
print("=" * 70)

# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CICIDS_FILE = DATA_DIR / "CICIDS2017_model_test.csv"
NSLKDD_FILE = DATA_DIR / "NSLKDD_model_test.csv"

CICIDS_OUTPUT = DATA_DIR / "CICIDS2017_cross_dataset_common.csv"
NSLKDD_OUTPUT = DATA_DIR / "NSLKDD_cross_dataset_common.csv"

MAPPING_OUTPUT = DATA_DIR / "cross_dataset_common_feature_mapping.csv"


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

print("\nLoading CICIDS2017 test data...")
cicids = pd.read_csv(CICIDS_FILE)

print("Loading NSL-KDD test data...")
nslkdd = pd.read_csv(NSLKDD_FILE)

print("\nOriginal shapes:")
print("CICIDS2017:", cicids.shape)
print("NSL-KDD   :", nslkdd.shape)


# ------------------------------------------------
# REQUIRED FEATURES
# ------------------------------------------------

cicids_required = [
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets"
]

nslkdd_required = [
    "src_bytes",
    "dst_bytes"
]


# ------------------------------------------------
# CHECK REQUIRED FEATURES
# ------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING REQUIRED FEATURES")
print("=" * 70)

for feature in cicids_required:

    if feature not in cicids.columns:
        raise ValueError(
            f"CICIDS2017 required feature missing: {feature}"
        )

    print(f"[OK] CICIDS2017: {feature}")


for feature in nslkdd_required:

    if feature not in nslkdd.columns:
        raise ValueError(
            f"NSL-KDD required feature missing: {feature}"
        )

    print(f"[OK] NSL-KDD: {feature}")


if "Label" not in cicids.columns:
    raise ValueError("CICIDS2017 Label column missing.")

if "Label" not in nslkdd.columns:
    raise ValueError("NSL-KDD Label column missing.")


# ------------------------------------------------
# CREATE COMMON FEATURES
# ------------------------------------------------

print("\n" + "=" * 70)
print("CREATING COMMON SEMANTIC FEATURES")
print("=" * 70)


# ================================================================
# CICIDS2017
# ================================================================

print("\nProcessing CICIDS2017...")

cicids_common = pd.DataFrame()

cicids_common["forward_bytes"] = (
    pd.to_numeric(
        cicids["Total Length of Fwd Packets"],
        errors="coerce"
    )
)

cicids_common["backward_bytes"] = (
    pd.to_numeric(
        cicids["Total Length of Bwd Packets"],
        errors="coerce"
    )
)

cicids_common["total_bytes"] = (
    cicids_common["forward_bytes"]
    + cicids_common["backward_bytes"]
)

cicids_common["traffic_ratio"] = (
    cicids_common["forward_bytes"]
    / (cicids_common["backward_bytes"] + 1.0)
)

cicids_common["Label"] = pd.to_numeric(
    cicids["Label"],
    errors="coerce"
)


# ================================================================
# NSL-KDD
# ================================================================

print("Processing NSL-KDD...")

nslkdd_common = pd.DataFrame()

nslkdd_common["forward_bytes"] = (
    pd.to_numeric(
        nslkdd["src_bytes"],
        errors="coerce"
    )
)

nslkdd_common["backward_bytes"] = (
    pd.to_numeric(
        nslkdd["dst_bytes"],
        errors="coerce"
    )
)

nslkdd_common["total_bytes"] = (
    nslkdd_common["forward_bytes"]
    + nslkdd_common["backward_bytes"]
)

nslkdd_common["traffic_ratio"] = (
    nslkdd_common["forward_bytes"]
    / (nslkdd_common["backward_bytes"] + 1.0)
)

nslkdd_common["Label"] = pd.to_numeric(
    nslkdd["Label"],
    errors="coerce"
)


# ------------------------------------------------
# DATA CLEANING
# ------------------------------------------------

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)


def clean_dataset(df, name):

    print(f"\nCleaning {name}...")

    before = len(df)

    # Replace infinity
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove invalid rows
    df = df.dropna()

    # Keep valid labels only
    df = df[df["Label"].isin([0, 1])]

    # Ensure numerical types
    feature_columns = [
        "forward_bytes",
        "backward_bytes",
        "total_bytes",
        "traffic_ratio"
    ]

    for col in feature_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Final NaN removal
    df = df.dropna()

    after = len(df)

    print("Rows before cleaning:", before)
    print("Rows after cleaning :", after)
    print("Rows removed        :", before - after)

    print(
        "Missing values:",
        int(df.isnull().sum().sum())
    )

    print(
        "Infinity values:",
        int(
            np.isinf(
                df[
                    [
                        "forward_bytes",
                        "backward_bytes",
                        "total_bytes",
                        "traffic_ratio"
                    ]
                ].to_numpy()
            ).sum()
        )
    )

    return df


cicids_common = clean_dataset(
    cicids_common,
    "CICIDS2017"
)

nslkdd_common = clean_dataset(
    nslkdd_common,
    "NSL-KDD"
)


# ------------------------------------------------
# FEATURE VALIDATION
# ------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATING COMMON FEATURES")
print("=" * 70)

common_features = [
    "forward_bytes",
    "backward_bytes",
    "total_bytes",
    "traffic_ratio"
]

for feature in common_features:

    print(
        f"[OK] {feature} "
        f"available in both datasets"
    )


# ------------------------------------------------
# FEATURE STATISTICS
# ------------------------------------------------

print("\n" + "=" * 70)
print("COMMON FEATURE STATISTICS")
print("=" * 70)

print("\nCICIDS2017:")

print(
    cicids_common[
        common_features
    ].describe()
)


print("\nNSL-KDD:")

print(
    nslkdd_common[
        common_features
    ].describe()
)


# ------------------------------------------------
# LABEL DISTRIBUTION
# ------------------------------------------------

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print("\nCICIDS2017:")

print(
    cicids_common["Label"]
    .value_counts()
    .sort_index()
)

print("\nNSL-KDD:")

print(
    nslkdd_common["Label"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------
# DATASET SOURCE
# ------------------------------------------------

cicids_common["Dataset"] = "CICIDS2017"
nslkdd_common["Dataset"] = "NSL-KDD"


# ------------------------------------------------
# FINAL COLUMN ORDER
# ------------------------------------------------

final_columns = [
    "forward_bytes",
    "backward_bytes",
    "total_bytes",
    "traffic_ratio",
    "Dataset",
    "Label"
]

cicids_common = cicids_common[
    final_columns
]

nslkdd_common = nslkdd_common[
    final_columns
]


# ------------------------------------------------
# SAVE
# ------------------------------------------------

print("\n" + "=" * 70)
print("SAVING CROSS-DATASET DATASETS")
print("=" * 70)

cicids_common.to_csv(
    CICIDS_OUTPUT,
    index=False
)

nslkdd_common.to_csv(
    NSLKDD_OUTPUT,
    index=False
)


# ------------------------------------------------
# FEATURE MAPPING REPORT
# ------------------------------------------------

mapping = pd.DataFrame({

    "Concept": [
        "Forward Traffic Volume",
        "Backward Traffic Volume",
        "Total Traffic Volume",
        "Traffic Direction Ratio"
    ],

    "CICIDS2017_Feature": [
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd + Bwd Packet Length",
        "Fwd Bytes / (Bwd Bytes + 1)"
    ],

    "NSLKDD_Feature": [
        "src_bytes",
        "dst_bytes",
        "src_bytes + dst_bytes",
        "src_bytes / (dst_bytes + 1)"
    ],

    "Unified_Feature": [
        "forward_bytes",
        "backward_bytes",
        "total_bytes",
        "traffic_ratio"
    ],

    "Comparable": [
        "YES",
        "YES",
        "YES",
        "YES"
    ]
})

mapping.to_csv(
    MAPPING_OUTPUT,
    index=False
)


# ------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print("\nCICIDS2017 final shape:")
print(cicids_common.shape)

print("\nNSL-KDD final shape:")
print(nslkdd_common.shape)

print("\nCICIDS2017 columns:")
print(list(cicids_common.columns))

print("\nNSL-KDD columns:")
print(list(nslkdd_common.columns))

print("\nCICIDS2017 missing values:")
print(
    int(cicids_common.isnull().sum().sum())
)

print("\nNSL-KDD missing values:")
print(
    int(nslkdd_common.isnull().sum().sum())
)


# ------------------------------------------------
# COMPLETION
# ------------------------------------------------

print("\n" + "=" * 70)
print("STEP 30 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCross-dataset common features:")

for i, feature in enumerate(
    common_features,
    start=1
):
    print(f"{i}. {feature}")

print("\nCICIDS2017 saved to:")
print(CICIDS_OUTPUT)

print("\nNSL-KDD saved to:")
print(NSLKDD_OUTPUT)

print("\nFeature mapping saved to:")
print(MAPPING_OUTPUT)

print("\nIMPORTANT:")
print(
    "Only semantically defensible common features "
    "were retained."
)

print(
    "No non-comparable NSL-KDD or CICIDS2017 "
    "features were artificially created."
)