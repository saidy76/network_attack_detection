import pandas as pd
import numpy as np

print("=" * 70)
print("FINAL CICIDS2017 FEATURE SELECTION")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

input_file = "data/CICIDS2017_no_constant.csv"

df = pd.read_csv(input_file)

print("\nOriginal dataset shape:", df.shape)


# ---------------------------------------------------------
# 2. Load common Top-30 features
# ---------------------------------------------------------

common_file = "data/CICIDS2017_common_top_features.csv"

common_features_df = pd.read_csv(common_file)

common_features = common_features_df["Feature"].tolist()

print("\nNumber of common features:", len(common_features))

print("\nCommon features:")
for i, feature in enumerate(common_features, 1):
    print(f"{i:2d}. {feature}")


# ---------------------------------------------------------
# 3. Check whether all features exist
# ---------------------------------------------------------

missing_features = [
    feature for feature in common_features
    if feature not in df.columns
]

if missing_features:
    print("\nERROR! These features are missing:")
    
    for feature in missing_features:
        print("-", feature)
    
    raise ValueError("Some selected features are missing from dataset.")

print("\nAll common features found successfully.")


# ---------------------------------------------------------
# 4. Create temporary dataset
# ---------------------------------------------------------

temp_features = common_features + ["Label"]

temp_df = df[temp_features].copy()

print("\nTemporary dataset shape:", temp_df.shape)


# ---------------------------------------------------------
# 5. Calculate correlation
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING HIGH CORRELATION")
print("=" * 70)

X = temp_df[common_features]

correlation_matrix = X.corr().abs()

upper_triangle = correlation_matrix.where(
    np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)
)


# ---------------------------------------------------------
# 6. Find redundant features
# ---------------------------------------------------------

correlation_threshold = 0.95

features_to_remove = []

for column in upper_triangle.columns:

    highly_correlated = upper_triangle[column][
        upper_triangle[column] > correlation_threshold
    ]

    if len(highly_correlated) > 0:
        features_to_remove.append(column)


print("\nCorrelation threshold:", correlation_threshold)

print("\nFeatures selected for removal:")

if len(features_to_remove) == 0:

    print("None")

else:

    for feature in features_to_remove:
        print("-", feature)


print(
    "\nNumber of features to remove:",
    len(features_to_remove)
)


# ---------------------------------------------------------
# 7. Create final feature list
# ---------------------------------------------------------

final_features = [
    feature
    for feature in common_features
    if feature not in features_to_remove
]


print("\n" + "=" * 70)
print("FINAL FEATURE LIST")
print("=" * 70)

for i, feature in enumerate(final_features, 1):
    print(f"{i:2d}. {feature}")


print(
    "\nFinal number of features:",
    len(final_features)
)


# ---------------------------------------------------------
# 8. Create final dataset
# ---------------------------------------------------------

final_columns = final_features + ["Label"]

final_df = df[final_columns].copy()

print("\nFinal dataset shape:", final_df.shape)


# ---------------------------------------------------------
# 9. Save final dataset
# ---------------------------------------------------------

output_file = "data/CICIDS2017_final_features.csv"

final_df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# 10. Save feature list
# ---------------------------------------------------------

feature_list_file = "data/CICIDS2017_final_feature_list.csv"

feature_list_df = pd.DataFrame({
    "Feature": final_features
})

feature_list_df.to_csv(
    feature_list_file,
    index=False
)


# ---------------------------------------------------------
# 11. Final verification
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print("\nFinal dataset shape:")
print(final_df.shape)

print("\nFinal columns:")

for i, column in enumerate(final_df.columns, 1):
    print(f"{i:2d}. {column}")


print("\nLabel distribution:")

print(final_df["Label"].value_counts())


print("\n" + "=" * 70)
print("STEP 10 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal dataset saved to:")
print(output_file)

print("\nFinal feature list saved to:")
print(feature_list_file)