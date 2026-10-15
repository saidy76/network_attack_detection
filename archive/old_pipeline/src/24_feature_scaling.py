import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


print("=" * 70)
print("STEP 24 - FEATURE SCALING")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading training data...")
train_df = pd.read_csv("data/train_data_smote.csv")

print("Loading testing data...")
test_df = pd.read_csv("data/test_data.csv")


print("\nTraining dataset shape:")
print(train_df.shape)

print("\nTesting dataset shape:")
print(test_df.shape)


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

features = [
    "forward_bytes",
    "backward_bytes",
    "total_bytes",
    "traffic_ratio"
]

label_column = "Label"


print("\n" + "=" * 70)
print("FEATURES TO BE SCALED")
print("=" * 70)

for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")


# ============================================================
# 3. VALIDATE COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING REQUIRED COLUMNS")
print("=" * 70)

for feature in features:
    if feature not in train_df.columns:
        raise ValueError(f"Missing feature in training data: {feature}")

    if feature not in test_df.columns:
        raise ValueError(f"Missing feature in testing data: {feature}")

print("[OK] All required features found.")


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

train_missing = train_df[features].isnull().sum().sum()
test_missing = test_df[features].isnull().sum().sum()

print("Training missing values:", train_missing)
print("Testing missing values:", test_missing)


# ============================================================
# 5. CHECK INFINITY VALUES
# ============================================================

print("\n" + "=" * 70)
print("INFINITY CHECK")
print("=" * 70)

train_inf = np.isinf(train_df[features].to_numpy()).sum()
test_inf = np.isinf(test_df[features].to_numpy()).sum()

print("Training infinity values:", train_inf)
print("Testing infinity values:", test_inf)


# ============================================================
# 6. SEPARATE FEATURES
# ============================================================

X_train = train_df[features].copy()
y_train = train_df[label_column].copy()

X_test = test_df[features].copy()
y_test = test_df[label_column].copy()


# ============================================================
# 7. CREATE STANDARD SCALER
# ============================================================

print("\n" + "=" * 70)
print("CREATING STANDARD SCALER")
print("=" * 70)

scaler = StandardScaler()


# ============================================================
# 8. FIT ONLY ON TRAINING DATA
# ============================================================

print("\nFitting scaler on training data...")

X_train_scaled = scaler.fit_transform(X_train)

print("[OK] Scaler fitted using training data only.")


# ============================================================
# 9. TRANSFORM TEST DATA
# ============================================================

print("\nTransforming testing data...")

X_test_scaled = scaler.transform(X_test)

print("[OK] Testing data transformed using the training scaler.")


# ============================================================
# 10. CREATE SCALED DATAFRAMES
# ============================================================

train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=features
)

test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=features
)


# Add labels

train_scaled_df["Label"] = y_train.values
test_scaled_df["Label"] = y_test.values


# ============================================================
# 11. CHECK SCALED DATA
# ============================================================

print("\n" + "=" * 70)
print("SCALED FEATURE SUMMARY")
print("=" * 70)

print("\nTraining scaled data:")
print(train_scaled_df[features].describe())


# ============================================================
# 12. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print("\nTraining:")
print(train_scaled_df["Label"].value_counts())

print("\nTesting:")
print(test_scaled_df["Label"].value_counts())


# ============================================================
# 13. CHECK FINAL SHAPES
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SHAPES")
print("=" * 70)

print("\nScaled training shape:")
print(train_scaled_df.shape)

print("\nScaled testing shape:")
print(test_scaled_df.shape)


# ============================================================
# 14. SAVE SCALER
# ============================================================

print("\n" + "=" * 70)
print("SAVING SCALER")
print("=" * 70)

joblib.dump(
    scaler,
    "data/standard_scaler.pkl"
)

print("[OK] Scaler saved to:")
print("data/standard_scaler.pkl")


# ============================================================
# 15. SAVE SCALED DATASETS
# ============================================================

print("\n" + "=" * 70)
print("SAVING SCALED DATASETS")
print("=" * 70)

train_scaled_df.to_csv(
    "data/train_data_scaled.csv",
    index=False
)

test_scaled_df.to_csv(
    "data/test_data_scaled.csv",
    index=False
)

print("[OK] Scaled training data saved to:")
print("data/train_data_scaled.csv")

print("[OK] Scaled testing data saved to:")
print("data/test_data_scaled.csv")


# ============================================================
# 16. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print("\nTraining missing values:")
print(train_scaled_df.isnull().sum().sum())

print("\nTesting missing values:")
print(test_scaled_df.isnull().sum().sum())

print("\nTraining infinity values:")
print(np.isinf(train_scaled_df[features].to_numpy()).sum())

print("\nTesting infinity values:")
print(np.isinf(test_scaled_df[features].to_numpy()).sum())


# ============================================================
# 17. COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("STEP 24 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput files:")
print("1. data/train_data_scaled.csv")
print("2. data/test_data_scaled.csv")
print("3. data/standard_scaler.pkl")