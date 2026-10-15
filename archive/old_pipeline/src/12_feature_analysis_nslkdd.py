import pandas as pd
import numpy as np

print("=" * 70)
print("NSL-KDD FEATURE ANALYSIS")
print("=" * 70)

# ============================================================
# 1. LOAD DATASET
# ============================================================

input_file = "data/NSLKDD_combined_clean.csv"

df = pd.read_csv(input_file)

print("\nDataset shape:")
print(df.shape)

# ============================================================
# 2. ALL COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("ALL COLUMNS")
print("=" * 70)

for i, column in enumerate(df.columns):
    print(i, ":", column)

# ============================================================
# 3. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing[missing > 0])

print("\nTotal missing values:")
print(df.isnull().sum().sum())

# ============================================================
# 5. DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

# ============================================================
# 6. CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL FEATURES")
print("=" * 70)

categorical_features = df.select_dtypes(
    include=["object"]
).columns.tolist()

print("Number of categorical features:")
print(len(categorical_features))

for feature in categorical_features:
    print(
        f"\n{feature}:",
        df[feature].nunique(),
        "unique values"
    )

    print(
        df[feature].value_counts().head(20)
    )

# ============================================================
# 7. NUMERICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL FEATURES")
print("=" * 70)

numerical_features = df.select_dtypes(
    include=[np.number]
).columns.tolist()

print(
    "Number of numerical columns:",
    len(numerical_features)
)

# ============================================================
# 8. CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CONSTANT FEATURES")
print("=" * 70)

constant_features = []

for column in numerical_features:

    if column == "Label":
        continue

    if df[column].nunique() <= 1:
        constant_features.append(column)

print(
    "Number of constant features:",
    len(constant_features)
)

for feature in constant_features:
    print("-", feature)

# ============================================================
# 9. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("UNIQUE VALUES PER NUMERICAL FEATURE")
print("=" * 70)

unique_counts = df[numerical_features].nunique()

print(
    unique_counts.sort_values()
)

# ============================================================
# 10. NEGATIVE VALUES
# ============================================================

print("\n" + "=" * 70)
print("NEGATIVE VALUES")
print("=" * 70)

for column in numerical_features:

    if column == "Label":
        continue

    count_negative = (
        df[column] < 0
    ).sum()

    if count_negative > 0:
        print(
            f"{column}: {count_negative}"
        )

# ============================================================
# 11. INFINITY VALUES
# ============================================================

print("\n" + "=" * 70)
print("INFINITY VALUES")
print("=" * 70)

numeric_df = df[numerical_features]

positive_inf = np.isposinf(
    numeric_df
).sum().sum()

negative_inf = np.isneginf(
    numeric_df
).sum().sum()

print(
    "Positive infinity:",
    positive_inf
)

print(
    "Negative infinity:",
    negative_inf
)

print(
    "Total infinity:",
    positive_inf + negative_inf
)

# ============================================================
# 12. NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL FEATURE SUMMARY")
print("=" * 70)

summary = df[numerical_features].describe().T

print(summary)

# ============================================================
# 13. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print(
    df["Label"].value_counts()
)

print("\nLabel percentage:")

print(
    df["Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ============================================================
# 14. SAVE REPORTS
# ============================================================

unique_counts.to_csv(
    "data/NSLKDD_unique_counts.csv"
)

summary.to_csv(
    "data/NSLKDD_numerical_summary.csv"
)

print("\n" + "=" * 70)
print("NSL-KDD FEATURE ANALYSIS COMPLETED")
print("=" * 70)

print("\nReports saved:")
print("data/NSLKDD_unique_counts.csv")
print("data/NSLKDD_numerical_summary.csv")