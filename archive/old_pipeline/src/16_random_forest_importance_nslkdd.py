import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("=" * 70)
print("NSL-KDD RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv("data/NSLKDD_encoded.csv")

print("Original dataset shape:", df.shape)

# ============================================================
# 2. SEPARATE FEATURES AND LABEL
# ============================================================

X = df.drop(columns=["Label"])
y = df["Label"]

print("\nNumber of features:", X.shape[1])
print("Number of samples:", X.shape[0])

# ============================================================
# 3. DATA VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

print("\nMissing values:", X.isnull().sum().sum())

print("Infinity values:", np.isinf(X.select_dtypes(include=np.number)).sum().sum())

# ============================================================
# 4. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print(y.value_counts())

# ============================================================
# 5. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

print("\nThis may take some time...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X, y)

# ============================================================
# 6. CALCULATE FEATURE IMPORTANCE
# ============================================================

importance = rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

# Add rank
importance_df["RF_Rank"] = range(1, len(importance_df) + 1)

# ============================================================
# 7. DISPLAY ALL FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df[
        ["Feature", "Importance", "RF_Rank"]
    ].to_string(index=False)
)

# ============================================================
# 8. TOP 30 FEATURES
# ============================================================

top_30 = importance_df.head(30)

print("\n" + "=" * 70)
print("TOP 30 NSL-KDD FEATURES")
print("=" * 70)

print(
    top_30[
        ["Feature", "Importance", "RF_Rank"]
    ].to_string(index=False)
)

# ============================================================
# 9. SAVE RESULTS
# ============================================================

output_file = "data/NSLKDD_random_forest_importance.csv"

importance_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("RANDOM FOREST ANALYSIS COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(output_file)

print("\nTotal features analyzed:", len(importance_df))
print("Top features reported:", len(top_30))