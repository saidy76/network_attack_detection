import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

print("=" * 70)
print("STEP 23 - SMOTE ON TRAINING DATA")
print("=" * 70)

# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

train = pd.read_csv("data/train_data.csv")

print("\nOriginal training shape:")
print(train.shape)


# ============================================================
# 2. SEPARATE FEATURES AND LABEL
# ============================================================

feature_columns = [
    "forward_bytes",
    "backward_bytes",
    "total_bytes",
    "traffic_ratio"
]

X_train = train[feature_columns].copy()
y_train = train["Label"].copy()


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print("Missing values:", X_train.isnull().sum().sum())


# ============================================================
# 4. CHECK INFINITY
# ============================================================

print("\n" + "=" * 70)
print("INFINITY CHECK")
print("=" * 70)

inf_count = np.isinf(X_train.to_numpy()).sum()

print("Infinity values:", inf_count)


# ============================================================
# 5. ORIGINAL CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL TRAINING CLASS DISTRIBUTION")
print("=" * 70)

print(y_train.value_counts())

print("\nPercentage:")

print(
    (y_train.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 6. APPLY SMOTE
# ============================================================

print("\n" + "=" * 70)
print("APPLYING SMOTE")
print("=" * 70)

print("\nThis may take some time because the dataset is large...")

smote = SMOTE(
    sampling_strategy="auto",
    random_state=42,
    k_neighbors=5
)

X_resampled, y_resampled = smote.fit_resample(
    X_train,
    y_train
)


# ============================================================
# 7. CHECK BALANCED DATA
# ============================================================

print("\n" + "=" * 70)
print("SMOTE COMPLETED")
print("=" * 70)

print("\nOriginal training samples:")
print(len(y_train))

print("\nSMOTE training samples:")
print(len(y_resampled))

print("\nAfter SMOTE class distribution:")
print(pd.Series(y_resampled).value_counts())

print("\nAfter SMOTE percentage:")

print(
    pd.Series(y_resampled)
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 8. CREATE BALANCED DATASET
# ============================================================

balanced_train = pd.DataFrame(
    X_resampled,
    columns=feature_columns
)

balanced_train["Label"] = y_resampled


# ============================================================
# 9. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print("\nBalanced training shape:")
print(balanced_train.shape)

print("\nMissing values:")
print(balanced_train.isnull().sum().sum())

print("\nInfinity values:")

print(
    np.isinf(
        balanced_train[feature_columns].to_numpy()
    ).sum()
)


# ============================================================
# 10. SAVE DATASET
# ============================================================

output_path = "data/train_data_smote.csv"

print("\nSaving balanced training dataset...")

balanced_train.to_csv(
    output_path,
    index=False
)


# ============================================================
# 11. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("STEP 23 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nBalanced training dataset saved to:")
print(output_path)