import pandas as pd
import numpy as np

# ==========================================
# Load dataset
# ==========================================

input_file = "data/CICIDS2017_no_constant.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)


# ==========================================
# Select numeric columns
# ==========================================

numeric_df = df.select_dtypes(include=np.number)

print("\nNumber of numeric columns:", len(numeric_df.columns))


# ==========================================
# Calculate correlation matrix
# ==========================================

print("\nCalculating correlation matrix...")

correlation_matrix = numeric_df.corr()


# ==========================================
# Find highly correlated feature pairs
# ==========================================

threshold = 0.90

high_corr_pairs = []

columns = correlation_matrix.columns

for i in range(len(columns)):

    for j in range(i + 1, len(columns)):

        correlation_value = correlation_matrix.iloc[i, j]

        if abs(correlation_value) >= threshold:

            high_corr_pairs.append(
                (
                    columns[i],
                    columns[j],
                    correlation_value
                )
            )


# ==========================================
# Convert to DataFrame
# ==========================================

high_corr_df = pd.DataFrame(
    high_corr_pairs,
    columns=[
        "Feature_1",
        "Feature_2",
        "Correlation"
    ]
)

# Sort by absolute correlation
if not high_corr_df.empty:

    high_corr_df["Absolute_Correlation"] = (
        high_corr_df["Correlation"].abs()
    )

    high_corr_df = high_corr_df.sort_values(
        "Absolute_Correlation",
        ascending=False
    )


# ==========================================
# Display results
# ==========================================

print("\n" + "=" * 60)
print("HIGHLY CORRELATED FEATURE PAIRS")
print("=" * 60)

if high_corr_df.empty:

    print("No feature pairs found with correlation >= 0.90")

else:

    print(
        high_corr_df.to_string(index=False)
    )

    print(
        "\nTotal highly correlated pairs:",
        len(high_corr_df)
    )


# ==========================================
# Save results
# ==========================================

output_file = "data/CICIDS2017_high_correlations.csv"

high_corr_df.to_csv(
    output_file,
    index=False
)

print("\nResults saved to:")
print(output_file)


# ==========================================
# Save full correlation matrix
# ==========================================

correlation_matrix.to_csv(
    "data/CICIDS2017_correlation_matrix.csv"
)

print("\nFull correlation matrix saved to:")
print("data/CICIDS2017_correlation_matrix.csv")


print("\n" + "=" * 60)
print("CORRELATION ANALYSIS COMPLETED")
print("=" * 60)