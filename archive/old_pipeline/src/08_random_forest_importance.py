import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. Load dataset
# ==========================================

input_file = "data/CICIDS2017_no_constant.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)


# ==========================================
# 2. Balanced sampling
# ==========================================

sample_size_per_class = 50000

normal = df[df["Label"] == 0]
attack = df[df["Label"] == 1]

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

sample_df = sample_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print("\nSample shape:", sample_df.shape)

print("\nSample label distribution:")
print(sample_df["Label"].value_counts())


# ==========================================
# 3. Separate X and y
# ==========================================

X = sample_df.drop(columns=["Label"])

y = sample_df["Label"]


# ==========================================
# 4. Handle invalid values
# ==========================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)


# ==========================================
# 5. Train Random Forest
# ==========================================

print("\nTraining Random Forest...")
print("This may take some time...")


rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X, y)


# ==========================================
# 6. Feature importance
# ==========================================

importance_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }
)

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


# ==========================================
# 7. Display all features
# ==========================================

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance_df.to_string(index=False)
)


# ==========================================
# 8. Top 30
# ==========================================

print("\n" + "=" * 60)
print("TOP 30 FEATURES")
print("=" * 60)

print(
    importance_df.head(30).to_string(index=False)
)


# ==========================================
# 9. Save results
# ==========================================

output_file = "data/CICIDS2017_random_forest_importance.csv"

importance_df.to_csv(
    output_file,
    index=False
)

print("\nResults saved to:")
print(output_file)


print("\n" + "=" * 60)
print("RANDOM FOREST ANALYSIS COMPLETED")
print("=" * 60)