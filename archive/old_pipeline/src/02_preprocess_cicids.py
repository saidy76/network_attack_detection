import pandas as pd
import numpy as np

# Input file
input_file = "data/CICIDS2017_merged.csv"

# Read dataset
df = pd.read_csv(input_file)

print("Original shape:", df.shape)

# Clean column names
df.columns = df.columns.str.strip()

# Remove duplicate rows
df = df.drop_duplicates()

print("After duplicate removal:", df.shape)

# Replace infinity values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Remove missing values
df = df.dropna()

print("After removing NaN:", df.shape)

# Check label distribution
print("\nLabel distribution:")
print(df["Label"].value_counts())

# Convert to binary classification
# BENIGN = 0
# Attack = 1
df["Label"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

print("\nBinary labels:")
print(df["Label"].value_counts())

# Save cleaned dataset
output_file = "data/CICIDS2017_clean.csv"

df.to_csv(output_file, index=False)

print("\nCICIDS2017 preprocessing completed!")
print("Cleaned dataset saved to:", output_file)