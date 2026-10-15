import pandas as pd
import os

from sklearn.model_selection import train_test_split

print("=" * 70)
print("STEP 22 - TRAIN/TEST SPLIT")
print("=" * 70)


# ============================================================
# 1. LOAD COMBINED DATASET
# ============================================================

print("\nLoading combined dataset...")

df = pd.read_csv("data/combined_network_dataset.csv")

print("\nOriginal dataset shape:")
print(df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND LABEL
# ============================================================

print("\n" + "=" * 70)
print("SEPARATING FEATURES AND LABEL")
print("=" * 70)

# Dataset column is NOT used as an ML feature
feature_columns = [
    "forward_bytes",
    "backward_bytes",
    "total_bytes",
    "traffic_ratio"
]

X = df[feature_columns]
y = df["Label"]

print("\nFeature matrix shape:")
print(X.shape)

print("\nLabel shape:")
print(y.shape)


# ============================================================
# 3. CHECK ORIGINAL LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL LABEL DISTRIBUTION")
print("=" * 70)

print(y.value_counts())

print("\nOriginal label percentage:")
print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 4. TRAIN/TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("PERFORMING TRAIN/TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. TRAINING SET INFORMATION
# ============================================================

print("\nTraining set shape:")
print(X_train.shape)

print("\nTraining labels shape:")
print(y_train.shape)

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTraining label percentage:")
print(
    (y_train.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 6. TESTING SET INFORMATION
# ============================================================

print("\nTesting set shape:")
print(X_test.shape)

print("\nTesting labels shape:")
print(y_test.shape)

print("\nTesting label distribution:")
print(y_test.value_counts())

print("\nTesting label percentage:")
print(
    (y_test.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 7. SAVE TRAINING DATA
# ============================================================

train_data = X_train.copy()
train_data["Label"] = y_train.values

test_data = X_test.copy()
test_data["Label"] = y_test.values


train_path = "data/train_data.csv"
test_path = "data/test_data.csv"

train_data.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)


# ============================================================
# 8. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print("\nTraining dataset:")
print(train_data.shape)

print("\nTesting dataset:")
print(test_data.shape)

print("\nTraining missing values:")
print(train_data.isnull().sum().sum())

print("\nTesting missing values:")
print(test_data.isnull().sum().sum())


# ============================================================
# 9. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("STEP 22 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nTraining data saved to:")
print(train_path)

print("\nTesting data saved to:")
print(test_path)