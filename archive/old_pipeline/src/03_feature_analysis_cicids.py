import pandas as pd
import numpy as np

# ==========================================
# 1. Load cleaned CICIDS2017 dataset
# ==========================================

input_file = "data/CICIDS2017_clean.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("CICIDS2017 FEATURE ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# ==========================================
# 2. Show all columns
# ==========================================

print("\n" + "=" * 60)
print("ALL COLUMNS")
print("=" * 60)

for i, column in enumerate(df.columns):
    print(i, ":", column)


# ==========================================
# 3. Data types
# ==========================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ==========================================
# 4. Missing values
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

print("\nTotal missing values:", df.isnull().sum().sum())


# ==========================================
# 5. Duplicate rows
# ==========================================

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print("Duplicate rows:", df.duplicated().sum())


# ==========================================
# 6. Infinity values
# ==========================================

print("\n" + "=" * 60)
print("INFINITY VALUES")
print("=" * 60)

numeric_df = df.select_dtypes(include=np.number)

infinity_count = np.isinf(numeric_df).sum().sum()

print("Total infinity values:", infinity_count)


# ==========================================
# 7. Constant features
# ==========================================

print("\n" + "=" * 60)
print("CONSTANT FEATURES")
print("=" * 60)

constant_features = []

for column in df.columns:

    if column == "Label":
        continue

    if df[column].nunique() <= 1:
        constant_features.append(column)

print("Number of constant features:", len(constant_features))

for column in constant_features:
    print("-", column)


# ==========================================
# 8. Unique values per feature
# ==========================================

print("\n" + "=" * 60)
print("UNIQUE VALUES PER FEATURE")
print("=" * 60)

unique_counts = df.nunique().sort_values()

print(unique_counts)


# ==========================================
# 9. Numerical feature summary
# ==========================================

print("\n" + "=" * 60)
print("NUMERICAL FEATURE SUMMARY")
print("=" * 60)

print(df.describe().T)


# ==========================================
# 10. Label distribution
# ==========================================

print("\n" + "=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)

print(df["Label"].value_counts())

print("\nLabel percentage:")

print(
    df["Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ==========================================
# 11. Correlation with Label
# ==========================================

print("\n" + "=" * 60)
print("FEATURE CORRELATION WITH LABEL")
print("=" * 60)

correlation = (
    df.select_dtypes(include=np.number)
    .corr()["Label"]
    .drop("Label")
    .abs()
    .sort_values(ascending=False)
)

print(correlation)


# ==========================================
# 12. Save feature analysis report
# ==========================================

correlation.to_csv(
    "data/CICIDS2017_feature_correlation.csv"
)

unique_counts.to_csv(
    "data/CICIDS2017_unique_counts.csv"
)

print("\n" + "=" * 60)
print("FEATURE ANALYSIS COMPLETED!")
print("=" * 60)

print("\nReports saved:")
print("data/CICIDS2017_feature_correlation.csv")
print("data/CICIDS2017_unique_counts.csv")