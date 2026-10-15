import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

print("=" * 70)
print("NSL-KDD CATEGORICAL FEATURE ENCODING")
print("=" * 70)

# ============================================================
# 1. LOAD DATASET
# ============================================================

input_file = "data/NSLKDD_no_constant.csv"

print("\nLoading dataset...")

df = pd.read_csv(input_file)

print(f"Original dataset shape: {df.shape}")

# ============================================================
# 2. IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "protocol_type",
    "service",
    "flag"
]

print("\n" + "=" * 70)
print("CATEGORICAL FEATURES")
print("=" * 70)

for feature in categorical_features:

    print(
        f"{feature}: "
        f"{df[feature].nunique()} unique values"
    )

# ============================================================
# 3. DISPLAY ORIGINAL VALUES
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL CATEGORICAL VALUES")
print("=" * 70)

for feature in categorical_features:

    print(f"\n{feature}:")
    print(df[feature].unique()[:20])

# ============================================================
# 4. ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("ENCODING CATEGORICAL FEATURES")
print("=" * 70)

label_encoders = {}

for feature in categorical_features:

    encoder = LabelEncoder()

    df[feature] = encoder.fit_transform(
        df[feature].astype(str)
    )

    label_encoders[feature] = encoder

    print(
        f"\n{feature} encoded successfully."
    )

    print(
        f"Number of encoded values: "
        f"{len(encoder.classes_)}"
    )

# ============================================================
# 5. DISPLAY ENCODING MAPPING
# ============================================================

print("\n" + "=" * 70)
print("ENCODING MAPPINGS")
print("=" * 70)

for feature in categorical_features:

    encoder = label_encoders[feature]

    print(f"\n{feature}:")

    for number, category in enumerate(
        encoder.classes_
    ):

        print(
            f"  {category} -> {number}"
        )

# ============================================================
# 6. CHECK DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES AFTER ENCODING")
print("=" * 70)

print(df.dtypes)

# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES CHECK")
print("=" * 70)

total_missing = df.isnull().sum().sum()

print(
    f"Total missing values: {total_missing}"
)

# ============================================================
# 8. CHECK INFINITY VALUES
# ============================================================

print("\n" + "=" * 70)
print("INFINITY VALUES CHECK")
print("=" * 70)

numeric_df = df.select_dtypes(include=["number"])

positive_inf = (
    numeric_df == float("inf")
).sum().sum()

negative_inf = (
    numeric_df == float("-inf")
).sum().sum()

print(f"Positive infinity: {positive_inf}")
print(f"Negative infinity: {negative_inf}")
print(
    f"Total infinity: "
    f"{positive_inf + negative_inf}"
)

# ============================================================
# 9. FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    f"\nFinal dataset shape: {df.shape}"
)

print("\nFinal columns:")

for i, column in enumerate(
    df.columns,
    start=1
):

    print(
        f"{i:2}. {column}"
    )

# ============================================================
# 10. SAVE ENCODED DATASET
# ============================================================

output_file = (
    "data/NSLKDD_encoded.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("NSL-KDD CATEGORICAL ENCODING COMPLETED")
print("=" * 70)

print("\nDataset saved to:")
print(output_file)

print(
    f"\nFinal dataset shape: {df.shape}"
)