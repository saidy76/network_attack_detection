"""
STEP 10 - FINAL CICIoT2023 TEST EVALUATION

Purpose:
    Evaluate the selected XGBoost baseline model on the
    completely unseen CICIoT2023 test dataset.

Dataset:
    CICIoT2023

Target:
    Label - 34 raw attack classes

Features:
    30 features selected in Step 7

Training:
    CICIoT2023_train_clean.csv

Validation:
    CICIoT2023_validation_clean.csv

Test:
    CICIoT2023_test_clean.csv

Model:
    XGBoost Baseline from Step 9

IMPORTANT:
    - Test data is NOT used for model selection.
    - Test data is NOT used for tuning.
    - No new feature selection is performed.
    - No SMOTE is applied.
    - No class weighting is applied.
    - The 30 features selected in Step 7 are fixed.
    - Model configuration is fixed according to Step 9.
"""

import os
import gc
import json
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\DELL\OneDrive\Desktop\network_attack_detection"

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "modeling",
    "CICIoT2023_train_clean.csv"
)

VALIDATION_PATH = os.path.join(
    BASE_DIR,
    "data",
    "modeling",
    "CICIoT2023_validation_clean.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "modeling",
    "CICIoT2023_test_clean.csv"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "experiments",
    "step7_selected_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "experiments"
)

RESULT_PATH = os.path.join(
    OUTPUT_DIR,
    "step10_ciciot2023_test_results.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "step10_ciciot2023_test_summary.json"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "step10_ciciot2023_classification_report.csv"
)

CONFUSION_PATH = os.path.join(
    OUTPUT_DIR,
    "step10_ciciot2023_confusion_matrix.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 10 - FINAL CICIoT2023 TEST EVALUATION")
print("=" * 70)

print("""
Dataset:
CICIoT2023

Target:
Label - 34 raw attack classes

Features:
30 features selected in Step 7

Model:
XGBoost Baseline selected in Step 9

Test:
CICIoT2023_test_clean.csv

IMPORTANT:
Test data is being used for FINAL evaluation only.
No tuning or feature selection will be performed using test data.
""")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

required_files = {
    "Training": TRAIN_PATH,
    "Validation": VALIDATION_PATH,
    "Test": TEST_PATH,
    "Selected features": FEATURE_PATH,
}

for name, path in required_files.items():

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] {name} file not found:\n{path}"
        )

    print(f"[OK] {path}")


# ============================================================
# LOAD SELECTED FEATURES
# ============================================================

print()
print("=" * 70)
print("LOADING STEP 7 SELECTED FEATURES")
print("=" * 70)

feature_df = pd.read_csv(FEATURE_PATH)

if "feature" not in feature_df.columns:
    raise ValueError(
        "Feature file must contain a 'feature' column."
    )

selected_features = (
    feature_df["feature"]
    .dropna()
    .tolist()
)

print(
    f"Selected feature count: "
    f"{len(selected_features)}"
)

if len(selected_features) != 30:
    raise ValueError(
        f"Expected exactly 30 selected features, "
        f"but found {len(selected_features)}."
    )

for i, feature in enumerate(
    selected_features,
    start=1
):
    print(f"{i:02d}. {feature}")

print(
    "[OK] Exactly 30 selected features loaded."
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)
print(
    f"Training shape: "
    f"{train_df.shape}"
)

print()
print("Loading validation data...")
validation_df = pd.read_csv(
    VALIDATION_PATH
)
print(
    f"Validation shape: "
    f"{validation_df.shape}"
)

print()
print("Loading test data...")
test_df = pd.read_csv(TEST_PATH)
print(
    f"Test shape: "
    f"{test_df.shape}"
)


# ============================================================
# VALIDATE TARGET
# ============================================================

print()
print("=" * 70)
print("TARGET VALIDATION")
print("=" * 70)

TARGET = "Label"

for name, df in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Test", test_df),
]:

    if TARGET not in df.columns:
        raise ValueError(
            f"{TARGET} column missing "
            f"from {name} dataset."
        )

    print(
        f"{name} classes: "
        f"{df[TARGET].nunique()}"
    )


train_classes = set(
    train_df[TARGET].unique()
)

validation_classes = set(
    validation_df[TARGET].unique()
)

test_classes = set(
    test_df[TARGET].unique()
)

print()

if train_classes != validation_classes:
    raise ValueError(
        "Training and validation classes "
        "do not match."
    )

if train_classes != test_classes:

    missing_in_test = (
        train_classes - test_classes
    )

    extra_in_test = (
        test_classes - train_classes
    )

    print(
        "[WARNING] Class sets are not identical."
    )

    if missing_in_test:
        print("Missing in test:")
        for c in sorted(missing_in_test):
            print("  ", c)

    if extra_in_test:
        print("Extra in test:")
        for c in sorted(extra_in_test):
            print("  ", c)

else:

    print(
        "[OK] Training, validation and test "
        "contain the same 34 classes."
    )


# ============================================================
# VALIDATE FEATURES
# ============================================================

print()
print("=" * 70)
print("VALIDATING FEATURE COLUMNS")
print("=" * 70)

for feature in selected_features:

    if feature not in train_df.columns:
        raise ValueError(
            f"Feature missing from training data: "
            f"{feature}"
        )

    if feature not in validation_df.columns:
        raise ValueError(
            f"Feature missing from validation data: "
            f"{feature}"
        )

    if feature not in test_df.columns:
        raise ValueError(
            f"Feature missing from test data: "
            f"{feature}"
        )

print(
    "[OK] All 30 selected features "
    "exist in all datasets."
)


# ============================================================
# LABEL ENCODING
# ============================================================

print()
print("=" * 70)
print("ENCODING LABELS")
print("=" * 70)

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(
    train_df[TARGET]
)

y_validation = label_encoder.transform(
    validation_df[TARGET]
)

y_test = label_encoder.transform(
    test_df[TARGET]
)

class_names = label_encoder.classes_

print(
    f"Number of classes: "
    f"{len(class_names)}"
)

print()
print("Class mapping:")

for i, class_name in enumerate(
    class_names
):

    print(
        f"{i:02d} -> {class_name}"
    )

print("[OK] Labels encoded.")


# ============================================================
# PREPARE FEATURE MATRICES
# ============================================================

print()
print("=" * 70)
print("PREPARING FEATURE MATRICES")
print("=" * 70)

print(
    "Converting features to float32..."
)

X_train = (
    train_df[selected_features]
    .astype(np.float32)
    .values
)

X_validation = (
    validation_df[selected_features]
    .astype(np.float32)
    .values
)

X_test = (
    test_df[selected_features]
    .astype(np.float32)
    .values
)

print(
    f"Training matrix: "
    f"{X_train.shape}"
)

print(
    f"Validation matrix: "
    f"{X_validation.shape}"
)

print(
    f"Test matrix: "
    f"{X_test.shape}"
)

print(
    f"Training matrix memory: "
    f"{X_train.nbytes / 1024**2:.2f} MB"
)

print(
    f"Validation matrix memory: "
    f"{X_validation.nbytes / 1024**2:.2f} MB"
)

print(
    f"Test matrix memory: "
    f"{X_test.nbytes / 1024**2:.2f} MB"
)


# ============================================================
# RELEASE DATAFRAMES
# ============================================================

del train_df
del validation_df
del test_df
del feature_df

gc.collect()

print()
print(
    "[OK] Original DataFrames released."
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print()
print("=" * 70)
print("TRAINING FINAL XGBOOST MODEL")
print("=" * 70)

print("""
Model:
XGBoost Baseline

Parameters:
n_estimators = 250
max_depth = 8
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
objective = multi:softprob
num_class = 34
tree_method = hist
n_jobs = 2
random_state = 42

Class weighting:
False
""")

model = XGBClassifier(
    n_estimators=250,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(class_names),
    tree_method="hist",
    n_jobs=2,
    random_state=42,
    eval_metric="mlogloss",
)

print("Training model...")

train_start = time.time()

model.fit(
    X_train,
    y_train
)

train_time = (
    time.time() - train_start
)

print(
    f"Training completed in "
    f"{train_time:.2f} seconds."
)


# ============================================================
# FINAL TEST PREDICTION
# ============================================================

print()
print("=" * 70)
print("FINAL TEST PREDICTION")
print("=" * 70)

print(
    "Predicting on the completely unseen "
    "CICIoT2023 test dataset..."
)

prediction_start = time.time()

y_pred = model.predict(
    X_test
)

prediction_time = (
    time.time() - prediction_start
)

print(
    f"Prediction completed in "
    f"{prediction_time:.2f} seconds."
)


# ============================================================
# FINAL METRICS
# ============================================================

print()
print("=" * 70)
print("FINAL CICIoT2023 TEST RESULTS")
print("=" * 70)

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = (
    balanced_accuracy_score(
        y_test,
        y_pred
    )
)

precision_macro = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("-" * 70)

print(
    f"Accuracy           : "
    f"{accuracy:.6f}"
)

print(
    f"Balanced Accuracy  : "
    f"{balanced_accuracy:.6f}"
)

print(
    f"Macro Precision    : "
    f"{precision_macro:.6f}"
)

print(
    f"Macro Recall       : "
    f"{recall_macro:.6f}"
)

print(
    f"Macro F1           : "
    f"{f1_macro:.6f}"
)

print(
    f"Weighted F1        : "
    f"{f1_weighted:.6f}"
)

print(
    f"Training Time      : "
    f"{train_time:.2f} sec"
)

print(
    f"Prediction Time    : "
    f"{prediction_time:.2f} sec"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASS-WISE PERFORMANCE")
print("=" * 70)

report_dict = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

print(
    report_df.to_string()
)

report_df.to_csv(
    REPORT_PATH
)

print()
print(
    "[OK] Classification report saved:\n"
    f"{REPORT_PATH}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(
    CONFUSION_PATH
)

print(
    "[OK] Confusion matrix saved:\n"
    f"{CONFUSION_PATH}"
)


# ============================================================
# SAVE MAIN RESULT
# ============================================================

results_df = pd.DataFrame([
    {
        "dataset": "CICIoT2023",
        "model": "XGBoost_Baseline",
        "feature_count": len(
            selected_features
        ),
        "test_samples": len(y_test),
        "num_classes": len(
            class_names
        ),
        "accuracy": accuracy,
        "balanced_accuracy":
            balanced_accuracy,
        "precision_macro":
            precision_macro,
        "recall_macro":
            recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted":
            f1_weighted,
        "training_time_seconds":
            train_time,
        "prediction_time_seconds":
            prediction_time,
    }
])

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print()
print(
    "[OK] Final test results saved:\n"
    f"{RESULT_PATH}"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "step": 10,
    "dataset": "CICIoT2023",
    "target": "Label",
    "num_classes": len(
        class_names
    ),
    "feature_count": len(
        selected_features
    ),
    "selected_features":
        selected_features,
    "model":
        "XGBoost_Baseline",

    "parameters": {
        "n_estimators": 250,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective":
            "multi:softprob",
        "num_class":
            len(class_names),
        "tree_method": "hist",
        "n_jobs": 2,
        "random_state": 42,
    },

    "class_weighting": False,
    "smote": False,
    "new_feature_selection": False,

    "test_used_for":
        "FINAL_EVALUATION_ONLY",

    "metrics": {
        "accuracy":
            float(accuracy),
        "balanced_accuracy":
            float(
                balanced_accuracy
            ),
        "precision_macro":
            float(
                precision_macro
            ),
        "recall_macro":
            float(
                recall_macro
            ),
        "f1_macro":
            float(f1_macro),
        "f1_weighted":
            float(f1_weighted),
    },

    "training_time_seconds":
        float(train_time),

    "prediction_time_seconds":
        float(prediction_time),

    "step8_xgboost_macro_f1":
        0.642746,

    "step9_best_validation_macro_f1":
        0.648867,

    "data_leakage_policy": {
        "test_used_for_model_selection":
            False,
        "test_used_for_feature_selection":
            False,
        "test_used_for_tuning":
            False,
    }
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

print(
    "[OK] Summary saved:\n"
    f"{SUMMARY_PATH}"
)


# ============================================================
# CLEANUP
# ============================================================

del model
del X_train
del X_validation
del X_test
del y_train
del y_validation
del y_test
del y_pred

gc.collect()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("STEP 10 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Dataset:
CICIoT2023

Model:
XGBoost Baseline

Features:
30 features selected in Step 7

Test:
CICIoT2023 TEST dataset

IMPORTANT:
This is the FINAL CICIoT2023 test evaluation.
The test result must NOT be used for further tuning.

Next stage:
CIC-IDS2018 independent dataset pipeline.
""")

print("=" * 70)