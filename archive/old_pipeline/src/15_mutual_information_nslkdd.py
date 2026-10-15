import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif

print("=" * 70)
print("NSL-KDD MUTUAL INFORMATION FEATURE SELECTION")
print("=" * 70)

# ============================================================
# 1. LOAD DATASET
# ============================================================

input_file = "data/NSLKDD_encoded.csv"

print("\nLoading dataset...")

df = pd.read_csv(input_file)

print(f"Original dataset shape: {df.shape}")

# ============================================================
# 2. SEPARATE FEATURES AND LABEL
# ============================================================

X = df.drop(columns=["Label"])
y = df["Label"]

print(f"\nNumber of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")

# ============================================================
# 3. CHECK DATA
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

print(
    f"\nMissing values: "
    f"{X.isnull().sum().sum()}"
)

print(
    f"Infinity values: "
    f"{np.isinf(X.select_dtypes(include=np.number)).sum().sum()}"
)

# ============================================================
# 4. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print(y.value_counts())

# ============================================================
# 5. CALCULATE MUTUAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING MUTUAL INFORMATION")
print("=" * 70)

print("\nThis may take some time...")

mi_scores = mutual_info_classif(
    X,
    y,
    random_state=42
)

# ============================================================
# 6. CREATE RESULTS DATAFRAME
# ============================================================

mi_df = pd.DataFrame({
    "Feature": X.columns,
    "Mutual_Information": mi_scores
})

# Sort from highest to lowest
mi_df = mi_df.sort_values(
    by="Mutual_Information",
    ascending=False
).reset_index(drop=True)

# Add rank
mi_df["MI_Rank"] = (
    mi_df.index + 1
)

# ============================================================
# 7. DISPLAY ALL MI SCORES
# ============================================================

print("\n" + "=" * 70)
print("MUTUAL INFORMATION SCORES")
print("=" * 70)

print(
    mi_df[
        [
            "Feature",
            "Mutual_Information",
            "MI_Rank"
        ]
    ].to_string(index=False)
)

# ============================================================
# 8. DISPLAY TOP 30 FEATURES
# ============================================================

top_n = min(30, len(mi_df))

print("\n" + "=" * 70)
print(f"TOP {top_n} NSL-KDD FEATURES")
print("=" * 70)

print(
    mi_df.head(top_n).to_string(index=False)
)

# ============================================================
# 9. SAVE RESULTS
# ============================================================

output_file = (
    "data/NSLKDD_mutual_information.csv"
)

mi_df.to_csv(
    output_file,
    index=False
)

# ============================================================
# 10. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MUTUAL INFORMATION ANALYSIS COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(output_file)

print(
    f"\nTotal features analyzed: "
    f"{len(mi_df)}"
)

print(
    f"Top features reported: "
    f"{top_n}"
)