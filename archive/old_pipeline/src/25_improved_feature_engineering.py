import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 25 - IMPROVED CROSS-DATASET FEATURE ENGINEERING
# ============================================================

print("=" * 70)
print("STEP 25 - IMPROVED CROSS-DATASET FEATURE ENGINEERING")
print("=" * 70)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CIC_FILE = DATA_DIR / "CICIDS2017_final_features.csv"
NSL_FILE = DATA_DIR / "NSLKDD_final_features.csv"


# ------------------------------------------------------------
# 2. LOAD DATASETS
# ------------------------------------------------------------

print("\nLoading CICIDS2017...")
cic = pd.read_csv(CIC_FILE)

print("Loading NSL-KDD...")
nsl = pd.read_csv(NSL_FILE)

print("\nOriginal shapes:")
print("CICIDS2017:", cic.shape)
print("NSL-KDD:", nsl.shape)


# ------------------------------------------------------------
# 3. DISPLAY AVAILABLE FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CICIDS2017 FEATURES")
print("=" * 70)

for i, col in enumerate(cic.columns, 1):
    print(f"{i:2}. {col}")


print("\n" + "=" * 70)
print("NSL-KDD FEATURES")
print("=" * 70)

for i, col in enumerate(nsl.columns, 1):
    print(f"{i:2}. {col}")


# ------------------------------------------------------------
# 4. CREATE UNIFIED SEMANTIC FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CREATING UNIFIED SEMANTIC FEATURES")
print("=" * 70)


# ============================================================
# CICIDS2017
# ============================================================

print("\nProcessing CICIDS2017...")


cic_unified = pd.DataFrame()

# Forward traffic volume
cic_unified["forward_bytes"] = cic["Total Length of Fwd Packets"]

# Backward traffic volume
cic_unified["backward_bytes"] = cic["Total Length of Bwd Packets"]

# Total traffic volume
cic_unified["total_bytes"] = (
    cic_unified["forward_bytes"] +
    cic_unified["backward_bytes"]
)

# Traffic direction ratio
cic_unified["traffic_ratio"] = (
    cic_unified["forward_bytes"] /
    (cic_unified["backward_bytes"] + 1)
)

# Forward packet size
cic_unified["forward_packet_max"] = cic["Fwd Packet Length Max"]

# Backward packet size
cic_unified["backward_packet_mean"] = cic["Bwd Packet Length Mean"]

# Packet size variation
cic_unified["packet_length_variance"] = cic["Packet Length Variance"]

# Packet length standard deviation
cic_unified["packet_length_std"] = cic["Packet Length Std"]

# Average packet size
cic_unified["average_packet_size"] = cic["Average Packet Size"]

# Destination port
cic_unified["destination_port"] = cic["Destination Port"]

# Initial forward TCP window
cic_unified["init_window_forward"] = cic["Init_Win_bytes_forward"]

# Initial backward TCP window
cic_unified["init_window_backward"] = cic["Init_Win_bytes_backward"]

# Forward inter-arrival time
cic_unified["forward_iat_max"] = cic["Fwd IAT Max"]

# Forward header length
cic_unified["forward_header_length"] = cic["Fwd Header Length.1"]

# Backward header length
cic_unified["backward_header_length"] = cic["Bwd Header Length"]

# Label
cic_unified["Label"] = cic["Label"]


# ============================================================
# NSL-KDD
# ============================================================

print("Processing NSL-KDD...")


nsl_unified = pd.DataFrame()

# Forward traffic volume
nsl_unified["forward_bytes"] = nsl["src_bytes"]

# Backward traffic volume
nsl_unified["backward_bytes"] = nsl["dst_bytes"]

# Total traffic volume
nsl_unified["total_bytes"] = (
    nsl_unified["forward_bytes"] +
    nsl_unified["backward_bytes"]
)

# Traffic direction ratio
nsl_unified["traffic_ratio"] = (
    nsl_unified["forward_bytes"] /
    (nsl_unified["backward_bytes"] + 1)
)

# ------------------------------------------------------------
# Features that do not have direct NSL-KDD equivalents
# ------------------------------------------------------------

nsl_unified["forward_packet_max"] = np.nan

nsl_unified["backward_packet_mean"] = np.nan

nsl_unified["packet_length_variance"] = np.nan

nsl_unified["packet_length_std"] = np.nan

nsl_unified["average_packet_size"] = np.nan

nsl_unified["destination_port"] = np.nan

nsl_unified["init_window_forward"] = np.nan

nsl_unified["init_window_backward"] = np.nan

nsl_unified["forward_iat_max"] = np.nan

nsl_unified["forward_header_length"] = np.nan

nsl_unified["backward_header_length"] = np.nan

# Label
nsl_unified["Label"] = nsl["Label"]


# ------------------------------------------------------------
# 5. VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATING UNIFIED FEATURES")
print("=" * 70)

print("\nCICIDS2017 unified shape:")
print(cic_unified.shape)

print("\nNSL-KDD unified shape:")
print(nsl_unified.shape)


# ------------------------------------------------------------
# 6. MISSING VALUE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print("\nCICIDS2017 missing values:")
print(cic_unified.isnull().sum())

print("\nNSL-KDD missing values:")
print(nsl_unified.isnull().sum())


# ------------------------------------------------------------
# 7. INFINITY CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INFINITY CHECK")
print("=" * 70)

cic_inf = np.isinf(
    cic_unified.select_dtypes(include=np.number)
).sum().sum()

nsl_inf = np.isinf(
    nsl_unified.select_dtypes(include=np.number)
).sum().sum()

print("CICIDS2017 infinity values:", cic_inf)
print("NSL-KDD infinity values:", nsl_inf)


# ------------------------------------------------------------
# 8. REMOVE INVALID VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLEANING INVALID VALUES")
print("=" * 70)


cic_unified = cic_unified.replace([np.inf, -np.inf], np.nan)
nsl_unified = nsl_unified.replace([np.inf, -np.inf], np.nan)


# Only rows with valid core features are required
core_features = [
    "forward_bytes",
    "backward_bytes",
    "total_bytes",
    "traffic_ratio"
]

before_cic = len(cic_unified)
before_nsl = len(nsl_unified)

cic_unified = cic_unified.dropna(subset=core_features)
nsl_unified = nsl_unified.dropna(subset=core_features)

print(
    "CICIDS2017 rows removed:",
    before_cic - len(cic_unified)
)

print(
    "NSL-KDD rows removed:",
    before_nsl - len(nsl_unified)
)


# ------------------------------------------------------------
# 9. FEATURE RANGE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CORE FEATURE STATISTICS")
print("=" * 70)

print("\nCICIDS2017:")
print(cic_unified[core_features].describe())

print("\nNSL-KDD:")
print(nsl_unified[core_features].describe())


# ------------------------------------------------------------
# 10. LABEL CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print("\nCICIDS2017:")
print(cic_unified["Label"].value_counts())

print("\nNSL-KDD:")
print(nsl_unified["Label"].value_counts())


# ------------------------------------------------------------
# 11. SAVE OUTPUTS
# ------------------------------------------------------------

cic_output = DATA_DIR / "CICIDS2017_improved_features.csv"
nsl_output = DATA_DIR / "NSLKDD_improved_features.csv"

cic_unified.to_csv(cic_output, index=False)
nsl_unified.to_csv(nsl_output, index=False)


# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 25 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCICIDS2017 saved to:")
print(cic_output)

print("\nNSL-KDD saved to:")
print(nsl_output)

print("\nCICIDS2017 final shape:")
print(cic_unified.shape)

print("\nNSL-KDD final shape:")
print(nsl_unified.shape)

print("\nUnified feature list:")

for i, feature in enumerate(cic_unified.columns, 1):
    print(f"{i:2}. {feature}")