import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 20 - CREATING COMMON FEATURE REPRESENTATION")
print("=" * 70)


# =========================================================
# 1. Load datasets
# =========================================================

cic_file = "data/CICIDS2017_final_features.csv"
nsl_file = "data/NSLKDD_final_features.csv"

print("\nLoading CICIDS2017...")
cic = pd.read_csv(cic_file)

print("Loading NSL-KDD...")
nsl = pd.read_csv(nsl_file)


# =========================================================
# 2. Create standardized CICIDS2017 features
# =========================================================

print("\n" + "=" * 70)
print("PROCESSING CICIDS2017")
print("=" * 70)

cic_common = pd.DataFrame()

cic_common["forward_bytes"] = cic["Total Length of Fwd Packets"]
cic_common["backward_bytes"] = cic["Total Length of Bwd Packets"]

# Total traffic volume
cic_common["total_bytes"] = (
    cic_common["forward_bytes"] +
    cic_common["backward_bytes"]
)

# Forward/backward traffic ratio
cic_common["traffic_ratio"] = (
    cic_common["forward_bytes"] /
    (cic_common["backward_bytes"] + 1)
)

# Label
cic_common["Label"] = cic["Label"]


print("\nCICIDS2017 common dataset shape:")
print(cic_common.shape)


# =========================================================
# 3. Create standardized NSL-KDD features
# =========================================================

print("\n" + "=" * 70)
print("PROCESSING NSL-KDD")
print("=" * 70)

nsl_common = pd.DataFrame()

nsl_common["forward_bytes"] = nsl["src_bytes"]
nsl_common["backward_bytes"] = nsl["dst_bytes"]

# Total traffic volume
nsl_common["total_bytes"] = (
    nsl_common["forward_bytes"] +
    nsl_common["backward_bytes"]
)

# Forward/backward traffic ratio
nsl_common["traffic_ratio"] = (
    nsl_common["forward_bytes"] /
    (nsl_common["backward_bytes"] + 1)
)

# Label
nsl_common["Label"] = nsl["Label"]


print("\nNSL-KDD common dataset shape:")
print(nsl_common.shape)


# =========================================================
# 4. Check missing values
# =========================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print("\nCICIDS2017 missing values:")
print(cic_common.isnull().sum().sum())

print("\nNSL-KDD missing values:")
print(nsl_common.isnull().sum().sum())


# =========================================================
# 5. Check infinity
# =========================================================

print("\n" + "=" * 70)
print("INFINITY CHECK")
print("=" * 70)

cic_inf = np.isinf(
    cic_common.select_dtypes(include="number")
).sum().sum()

nsl_inf = np.isinf(
    nsl_common.select_dtypes(include="number")
).sum().sum()

print("\nCICIDS2017 infinity values:")
print(cic_inf)

print("\nNSL-KDD infinity values:")
print(nsl_inf)


# =========================================================
# 6. Replace infinity if any
# =========================================================

cic_common.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

nsl_common.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# =========================================================
# 7. Remove rows containing NaN
# =========================================================

cic_before = len(cic_common)
nsl_before = len(nsl_common)

cic_common.dropna(inplace=True)
nsl_common.dropna(inplace=True)

print("\nRows removed from CICIDS2017:")
print(cic_before - len(cic_common))

print("\nRows removed from NSL-KDD:")
print(nsl_before - len(nsl_common))


# =========================================================
# 8. Display feature statistics
# =========================================================

print("\n" + "=" * 70)
print("COMMON FEATURES")
print("=" * 70)

common_features = [
    "forward_bytes",
    "backward_bytes",
    "total_bytes",
    "traffic_ratio"
]

for i, feature in enumerate(common_features, 1):
    print(f"{i}. {feature}")


# =========================================================
# 9. Label distribution
# =========================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print("\nCICIDS2017:")
print(cic_common["Label"].value_counts())

print("\nNSL-KDD:")
print(nsl_common["Label"].value_counts())


# =========================================================
# 10. Save datasets
# =========================================================

cic_output = "data/CICIDS2017_common_features.csv"
nsl_output = "data/NSLKDD_common_features.csv"

cic_common.to_csv(
    cic_output,
    index=False
)

nsl_common.to_csv(
    nsl_output,
    index=False
)


# =========================================================
# 11. Final verification
# =========================================================

print("\n" + "=" * 70)
print("STEP 20 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCICIDS2017 saved to:")
print(cic_output)

print("\nNSL-KDD saved to:")
print(nsl_output)

print("\nCICIDS2017 final shape:")
print(cic_common.shape)

print("\nNSL-KDD final shape:")
print(nsl_common.shape)