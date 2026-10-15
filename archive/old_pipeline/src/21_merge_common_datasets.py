import pandas as pd
import os

print("=" * 70)
print("STEP 21 - MERGING CICIDS2017 AND NSL-KDD")
print("=" * 70)

# ============================================================
# 1. LOAD DATASETS
# ============================================================

print("\nLoading CICIDS2017...")
cicids = pd.read_csv("data/CICIDS2017_common_features.csv")

print("Loading NSL-KDD...")
nslkdd = pd.read_csv("data/NSLKDD_common_features.csv")


print("\nCICIDS2017 shape:")
print(cicids.shape)

print("\nNSL-KDD shape:")
print(nslkdd.shape)


# ============================================================
# 2. CHECK COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING COMMON COLUMNS")
print("=" * 70)

print("\nCICIDS2017 columns:")
print(list(cicids.columns))

print("\nNSL-KDD columns:")
print(list(nslkdd.columns))


# ============================================================
# 3. ADD DATASET SOURCE
# ============================================================

print("\n" + "=" * 70)
print("ADDING DATASET SOURCE")
print("=" * 70)

cicids["Dataset"] = "CICIDS2017"
nslkdd["Dataset"] = "NSL-KDD"

print("[OK] Dataset source added.")


# ============================================================
# 4. MERGE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("MERGING DATASETS")
print("=" * 70)

combined = pd.concat(
    [cicids, nslkdd],
    axis=0,
    ignore_index=True
)


print("\nCombined dataset shape:")
print(combined.shape)


# ============================================================
# 5. SHUFFLE DATA
# ============================================================

print("\n" + "=" * 70)
print("SHUFFLING DATA")
print("=" * 70)

combined = combined.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("[OK] Dataset shuffled.")


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing = combined.isnull().sum().sum()

print("Total missing values:", missing)


# ============================================================
# 7. CHECK INFINITY
# ============================================================

print("\n" + "=" * 70)
print("INFINITY CHECK")
print("=" * 70)

numeric_columns = combined.select_dtypes(include=["number"]).columns

infinity_count = (
    combined[numeric_columns]
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

print("Total infinity values:", infinity_count)


# ============================================================
# 8. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("COMBINED LABEL DISTRIBUTION")
print("=" * 70)

print(combined["Label"].value_counts())

print("\nLabel percentage:")

print(
    (combined["Label"].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 9. DATASET SOURCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DATASET SOURCE DISTRIBUTION")
print("=" * 70)

print(combined["Dataset"].value_counts())


# ============================================================
# 10. FINAL COLUMN ORDER
# ============================================================

final_columns = [
    "forward_bytes",
    "backward_bytes",
    "total_bytes",
    "traffic_ratio",
    "Dataset",
    "Label"
]

combined = combined[final_columns]


# ============================================================
# 11. SAVE FINAL COMBINED DATASET
# ============================================================

output_path = "data/combined_network_dataset.csv"

combined.to_csv(
    output_path,
    index=False
)


# ============================================================
# 12. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print("\nFinal dataset shape:")
print(combined.shape)

print("\nFinal columns:")

for i, column in enumerate(combined.columns, 1):
    print(f"{i}. {column}")

print("\nMissing values:")
print(combined.isnull().sum().sum())

print("\nInfinity values:")
print(infinity_count)


print("\n" + "=" * 70)
print("STEP 21 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal combined dataset saved to:")
print(output_path)