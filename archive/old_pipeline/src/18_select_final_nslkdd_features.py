import pandas as pd
import os

print("=" * 70)
print("NSL-KDD FINAL FEATURE SELECTION")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

input_file = "data/NSLKDD_encoded.csv"

df = pd.read_csv(input_file)

print("\nOriginal dataset shape:")
print(df.shape)

# ---------------------------------------------------------
# 2. Define final selected features
# ---------------------------------------------------------

selected_features = [
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

# Add target label
final_columns = selected_features + ["Label"]

# ---------------------------------------------------------
# 3. Check features
# ---------------------------------------------------------

missing_features = [
    feature for feature in final_columns
    if feature not in df.columns
]

if missing_features:
    print("\nERROR: Missing features:")
    for feature in missing_features:
        print("-", feature)

    raise ValueError("Some selected features are missing.")

print("\nAll selected features found successfully.")

# ---------------------------------------------------------
# 4. Create final dataset
# ---------------------------------------------------------

final_df = df[final_columns].copy()

print("\nFinal dataset shape:")
print(final_df.shape)

# ---------------------------------------------------------
# 5. Display final features
# ---------------------------------------------------------

print("\nFinal selected features:")

for i, feature in enumerate(selected_features, 1):
    print(f"{i:2}. {feature}")

print(f"{len(selected_features) + 1:2}. Label")

# ---------------------------------------------------------
# 6. Check missing values
# ---------------------------------------------------------

print("\nMissing values:")
print(final_df.isnull().sum().sum())

# ---------------------------------------------------------
# 7. Check infinity
# ---------------------------------------------------------

numeric_columns = final_df.select_dtypes(include="number").columns

infinity_count = final_df[numeric_columns].isin(
    [float("inf"), float("-inf")]
).sum().sum()

print("\nInfinity values:")
print(infinity_count)

# ---------------------------------------------------------
# 8. Label distribution
# ---------------------------------------------------------

print("\nLabel distribution:")
print(final_df["Label"].value_counts())

# ---------------------------------------------------------
# 9. Save dataset
# ---------------------------------------------------------

output_file = "data/NSLKDD_final_features.csv"

final_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("NSL-KDD FINAL FEATURE SELECTION COMPLETED")
print("=" * 70)

print("\nFinal dataset saved to:")
print(output_file)

print("\nFinal dataset shape:")
print(final_df.shape)