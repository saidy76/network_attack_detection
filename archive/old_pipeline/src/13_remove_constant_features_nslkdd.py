import pandas as pd
import os

print("=" * 70)
print("NSL-KDD CONSTANT FEATURE REMOVAL")
print("=" * 70)

# ============================================================
# 1. LOAD DATASET
# ============================================================

input_file = "data/NSLKDD_combined_clean.csv"

print("\nLoading dataset...")
df = pd.read_csv(input_file)

print(f"Original dataset shape: {df.shape}")

# ============================================================
# 2. SEPARATE FEATURES AND LABEL
# ============================================================

X = df.drop(columns=["Label"])
y = df["Label"]

# ============================================================
# 3. FIND CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING CONSTANT FEATURES")
print("=" * 70)

constant_features = []

for column in X.columns:

    if X[column].nunique() <= 1:
        constant_features.append(column)

print("\nConstant features found:")

if len(constant_features) == 0:

    print("None")

else:

    for feature in constant_features:
        print(f"- {feature}")

print(f"\nNumber of constant features: {len(constant_features)}")

# ============================================================
# 4. REMOVE CONSTANT FEATURES
# ============================================================

if len(constant_features) > 0:

    X = X.drop(columns=constant_features)

# ============================================================
# 5. ADD LABEL BACK
# ============================================================

df_final = X.copy()
df_final["Label"] = y

# ============================================================
# 6. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("AFTER CONSTANT FEATURE REMOVAL")
print("=" * 70)

print(f"\nFinal dataset shape: {df_final.shape}")

print("\nRemaining features:")

for i, column in enumerate(X.columns, start=1):
    print(f"{i:2}. {column}")

# ============================================================
# 7. SAVE DATASET
# ============================================================

output_file = "data/NSLKDD_no_constant.csv"

df_final.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("NSL-KDD CONSTANT FEATURE REMOVAL COMPLETED")
print("=" * 70)

print(f"\nDataset saved to:")
print(output_file)

print("\nFinal shape:")
print(df_final.shape)