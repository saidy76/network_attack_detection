import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif

# ==========================================
# 1. Load dataset
# ==========================================

input_file = "data/CICIDS2017_no_constant.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("MUTUAL INFORMATION FEATURE SELECTION")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)


# ==========================================
# 2. Stratified sampling
# ==========================================

sample_size_per_class = 50000

normal = df[df["Label"] == 0]

attack = df[df["Label"] == 1]

print("\nNormal samples available:", len(normal))
print("Attack samples available:", len(attack))


normal_sample = normal.sample(
    n=min(sample_size_per_class, len(normal)),
    random_state=42
)

attack_sample = attack.sample(
    n=min(sample_size_per_class, len(attack)),
    random_state=42
)


sample_df = pd.concat(
    [normal_sample, attack_sample],
    ignore_index=True
)

# Shuffle
sample_df = sample_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print("\nSample used for MI:", sample_df.shape)

print("\nSample label distribution:")
print(sample_df["Label"].value_counts())


# ==========================================
# 3. Separate X and y
# ==========================================

X = sample_df.drop(columns=["Label"])

y = sample_df["Label"]


# ==========================================
# 4. Handle infinite values
# ==========================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Fill missing values with median
X = X.fillna(
    X.median()
)


# ==========================================
# 5. Calculate Mutual Information
# ==========================================

print("\nCalculating Mutual Information...")
print("This may take some time...")


mi_scores = mutual_info_classif(
    X,
    y,
    random_state=42
)


# ==========================================
# 6. Create result DataFrame
# ==========================================

mi_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Mutual_Information": mi_scores
    }
)


# Sort descending
mi_df = mi_df.sort_values(
    "Mutual_Information",
    ascending=False
)


# ==========================================
# 7. Display results
# ==========================================

print("\n" + "=" * 60)
print("MUTUAL INFORMATION SCORES")
print("=" * 60)

print(
    mi_df.to_string(index=False)
)


# ==========================================
# 8. Top 30 features
# ==========================================

print("\n" + "=" * 60)
print("TOP 30 FEATURES")
print("=" * 60)

print(
    mi_df.head(30).to_string(index=False)
)


# ==========================================
# 9. Save results
# ==========================================

output_file = "data/CICIDS2017_mutual_information.csv"

mi_df.to_csv(
    output_file,
    index=False
)


print("\nResults saved to:")
print(output_file)


print("\n" + "=" * 60)
print("MUTUAL INFORMATION ANALYSIS COMPLETED")
print("=" * 60)
