import pandas as pd

# ==========================================
# Load cleaned CICIDS2017 dataset
# ==========================================

input_file = "data/CICIDS2017_clean.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("REMOVING CONSTANT FEATURES")
print("=" * 60)

print("\nOriginal shape:", df.shape)


# ==========================================
# Find constant features
# ==========================================

constant_features = []

for column in df.columns:

    # Never remove Label
    if column == "Label":
        continue

    if df[column].nunique() <= 1:
        constant_features.append(column)


# ==========================================
# Show constant features
# ==========================================

print("\nConstant features found:")

for feature in constant_features:
    print("-", feature)

print("\nNumber of constant features:",
      len(constant_features))


# ==========================================
# Remove constant features
# ==========================================

df = df.drop(columns=constant_features)


# ==========================================
# Show new shape
# ==========================================

print("\nAfter removing constant features:",
      df.shape)


# ==========================================
# Save dataset
# ==========================================

output_file = "data/CICIDS2017_no_constant.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nDataset saved to:")
print(output_file)

print("\n" + "=" * 60)
print("CONSTANT FEATURE REMOVAL COMPLETED")
print("=" * 60)