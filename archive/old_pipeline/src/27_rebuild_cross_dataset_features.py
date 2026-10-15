import pandas as pd
import numpy as np
import os

print("=" * 70)
print("STEP 27 - REBUILD CROSS-DATASET SEMANTIC FEATURES")
print("=" * 70)

# ============================================================
# FILE PATHS
# ============================================================

CICIDS_FILE = "data/CICIDS2017_final_features.csv"
NSLKDD_FILE = "data/NSLKDD_final_features.csv"

OUTPUT_CICIDS = "data/CICIDS2017_common_semantic.csv"
OUTPUT_NSLKDD = "data/NSLKDD_common_semantic.csv"
OUTPUT_MAPPING = "data/final_cross_dataset_mapping.csv"


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading CICIDS2017...")
cicids = pd.read_csv(CICIDS_FILE)

print("Loading NSL-KDD...")
nslkdd = pd.read_csv(NSLKDD_FILE)

print("\nOriginal shapes:")
print("CICIDS2017:", cicids.shape)
print("NSL-KDD:", nslkdd.shape)


# ============================================================
# DISPLAY COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("CICIDS2017 COLUMNS")
print("=" * 70)

for i, col in enumerate(cicids.columns, 1):
    print(f"{i:2}. {col}")


print("\n" + "=" * 70)
print("NSL-KDD COLUMNS")
print("=" * 70)

for i, col in enumerate(nslkdd.columns, 1):
    print(f"{i:2}. {col}")


# ============================================================
# REQUIRED FEATURE CHECK
# ============================================================

cicids_required = [
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

nslkdd_required = [
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

print("\n" + "=" * 70)
print("CHECKING REQUIRED FEATURES")
print("=" * 70)

missing_cicids = [
    col for col in cicids_required
    if col not in cicids.columns
]

missing_nslkdd = [
    col for col in nslkdd_required
    if col not in nslkdd.columns
]

if missing_cicids:
    print("\nMissing CICIDS2017 features:")
    for col in missing_cicids:
        print("-", col)

if missing_nslkdd:
    print("\nMissing NSL-KDD features:")
    for col in missing_nslkdd:
        print("-", col)

if missing_cicids or missing_nslkdd:

    print("\nERROR:")
    print("Required source features are missing.")
    print("Step 27 cannot continue safely.")

    raise ValueError(
        "Required features missing. Check the input files."
    )

print("\n[OK] All required source features found.")


# ============================================================
# CREATE COMMON SEMANTIC FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING COMMON SEMANTIC FEATURES")
print("=" * 70)


# ------------------------------------------------------------
# CICIDS2017
# ------------------------------------------------------------

cic = pd.DataFrame()

cic["forward_bytes"] = cicids["Total Length of Fwd Packets"]
cic["backward_bytes"] = cicids["Total Length of Bwd Packets"]

cic["total_bytes"] = (
    cic["forward_bytes"] +
    cic["backward_bytes"]
)

cic["forward_packet_max"] = cicids["Fwd Packet Length Max"]

cic["backward_packet_mean"] = cicids["Bwd Packet Length Mean"]

cic["packet_length_variance"] = (
    cicids["Packet Length Variance"]
)

cic["packet_length_std"] = (
    cicids["Packet Length Std"]
)

cic["average_packet_size"] = (
    cicids["Average Packet Size"]
)

cic["destination_port"] = (
    cicids["Destination Port"]
)

cic["init_window_forward"] = (
    cicids["Init_Win_bytes_forward"]
)

cic["init_window_backward"] = (
    cicids["Init_Win_bytes_backward"]
)

cic["forward_iat_max"] = (
    cicids["Fwd IAT Max"]
)

cic["forward_header_length"] = (
    cicids["Fwd Header Length.1"]
)

cic["backward_header_length"] = (
    cicids["Bwd Header Length"]
)

# Avoid division by zero
cic["traffic_ratio"] = (
    cic["forward_bytes"] /
    (cic["backward_bytes"] + 1)
)

cic["Label"] = cicids["Label"]


# ------------------------------------------------------------
# NSL-KDD
# ------------------------------------------------------------

nsl = pd.DataFrame()

nsl["forward_bytes"] = nslkdd["src_bytes"]

nsl["backward_bytes"] = nslkdd["dst_bytes"]

nsl["total_bytes"] = (
    nsl["forward_bytes"] +
    nsl["backward_bytes"]
)

nsl["traffic_ratio"] = (
    nsl["forward_bytes"] /
    (nsl["backward_bytes"] + 1)
)

# NSL-KDD does not contain reliable equivalents
# for the following CICIDS-specific packet-level features.
# Therefore they are NOT artificially filled with zero.

nsl["forward_packet_max"] = np.nan
nsl["backward_packet_mean"] = np.nan
nsl["packet_length_variance"] = np.nan
nsl["packet_length_std"] = np.nan
nsl["average_packet_size"] = np.nan
nsl["destination_port"] = np.nan
nsl["init_window_forward"] = np.nan
nsl["init_window_backward"] = np.nan
nsl["forward_iat_max"] = np.nan
nsl["forward_header_length"] = np.nan
nsl["backward_header_length"] = np.nan

nsl["Label"] = nslkdd["Label"]


# ============================================================
# REMOVE NON-COMPARABLE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("SELECTING TRULY COMPARABLE FEATURES")
print("=" * 70)

common_features = [
    "forward_bytes",
    "backward_bytes",
    "total_bytes",
    "traffic_ratio"
]

cic_final = cic[common_features + ["Label"]].copy()
nsl_final = nsl[common_features + ["Label"]].copy()


# ============================================================
# VALIDATION
# ============================================================

print("\nCICIDS2017 common shape:")
print(cic_final.shape)

print("\nNSL-KDD common shape:")
print(nsl_final.shape)


print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print("\nCICIDS2017 missing values:")
print(cic_final.isnull().sum().sum())

print("\nNSL-KDD missing values:")
print(nsl_final.isnull().sum().sum())


print("\n" + "=" * 70)
print("INFINITY CHECK")
print("=" * 70)

cic_inf = np.isinf(
    cic_final.select_dtypes(include=np.number)
).sum().sum()

nsl_inf = np.isinf(
    nsl_final.select_dtypes(include=np.number)
).sum().sum()

print("CICIDS2017 infinity values:", cic_inf)
print("NSL-KDD infinity values:", nsl_inf)


# ============================================================
# LABEL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print("\nCICIDS2017:")
print(cic_final["Label"].value_counts())

print("\nNSL-KDD:")
print(nsl_final["Label"].value_counts())


# ============================================================
# SAVE DATASETS
# ============================================================

cic_final.to_csv(
    OUTPUT_CICIDS,
    index=False
)

nsl_final.to_csv(
    OUTPUT_NSLKDD,
    index=False
)


# ============================================================
# SAVE FINAL MAPPING
# ============================================================

mapping = pd.DataFrame({

    "Common_Feature": [
        "forward_bytes",
        "backward_bytes",
        "total_bytes",
        "traffic_ratio"
    ],

    "CICIDS2017_Source": [
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd + Bwd Packet Length",
        "Forward Bytes / (Backward Bytes + 1)"
    ],

    "NSLKDD_Source": [
        "src_bytes",
        "dst_bytes",
        "src_bytes + dst_bytes",
        "src_bytes / (dst_bytes + 1)"
    ],

    "Semantic_Validity": [
        "Direct",
        "Direct",
        "Derived",
        "Derived"
    ]
})

mapping.to_csv(
    OUTPUT_MAPPING,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 27 COMPLETED")
print("=" * 70)

print("\nFinal common features:")

for i, feature in enumerate(common_features, 1):
    print(f"{i}. {feature}")

print("\nCICIDS2017 saved to:")
print(OUTPUT_CICIDS)

print("\nNSL-KDD saved to:")
print(OUTPUT_NSLKDD)

print("\nFeature mapping saved to:")
print(OUTPUT_MAPPING)

print("\nFinal shapes:")
print("CICIDS2017:", cic_final.shape)
print("NSL-KDD:", nsl_final.shape)

print("\nIMPORTANT:")
print("Only semantically defensible common features were retained.")
print("Non-comparable features were NOT artificially filled.")