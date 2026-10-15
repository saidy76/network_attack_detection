# ================================================================
# STEP 9 - XGBOOST MODEL IMPROVEMENT / CLASS IMBALANCE HANDLING
# ================================================================

import os
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\DELL\OneDrive\Desktop\network_attack_detection"

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "modeling",
    "CICIoT2023_train_clean.csv"
)

VAL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "modeling",
    "CICIoT2023_validation_clean.csv"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "experiments",
    "step7_selected_features.csv"
)

RESULT_PATH = os.path.join(
    BASE_DIR,
    "experiments",
    "step9_xgboost_improvement_results.csv"
)

SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "experiments",
    "step9_xgboost_improvement_summary.json"
)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("STEP 9 - XGBOOST MODEL IMPROVEMENT")
print("=" * 70)

print()
print("Dataset:")
print("CICIoT2023")

print()
print("Target:")
print("Label - 34 raw attack classes")

print()
print("Features:")
print("30 features selected in Step 7")

print()
print("Training:")
print("CICIoT2023_train_clean.csv")

print()
print("Validation:")
print("CICIoT2023_validation_clean.csv")

print()
print("Test:")
print("NOT USED")

print()
print("Primary metric:")
print("Macro-F1")

print()
print("Improvement techniques:")
print("1. Class/sample weighting")
print("2. XGBoost parameter tuning")
print("3. Validation-based model selection")

print()
print("IMPORTANT:")
print("No test data is used.")
print("No new feature selection is performed.")
print("No SMOTE is applied.")
print("=" * 70)


# ================================================================
# CHECK FILES
# ================================================================

print()
print("=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

for path in [TRAIN_PATH, VAL_PATH, FEATURE_PATH]:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"[OK] {path}")


# ================================================================
# LOAD DATA
# ================================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

print()
print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)

print("Loading validation data...")
val_df = pd.read_csv(VAL_PATH)

print()
print("Training shape:", train_df.shape)
print("Validation shape:", val_df.shape)


# ================================================================
# LOAD SELECTED FEATURES
# ================================================================

print()
print("=" * 70)
print("LOADING STEP 7 SELECTED FEATURES")
print("=" * 70)

feature_df = pd.read_csv(FEATURE_PATH)

features = feature_df["feature"].tolist()

print()
print("Selected feature count:", len(features))

for i, feature in enumerate(features, 1):
    print(f"{i:02d}. {feature}")

if len(features) != 30:
    raise ValueError(
        f"Expected 30 selected features, found {len(features)}"
    )

print()
print("[OK] Exactly 30 selected features found.")


# ================================================================
# VALIDATE FEATURES
# ================================================================

print()
print("=" * 70)
print("VALIDATING FEATURE COLUMNS")
print("=" * 70)

missing_train = [
    f for f in features
    if f not in train_df.columns
]

missing_val = [
    f for f in features
    if f not in val_df.columns
]

if missing_train:
    raise ValueError(
        f"Missing features in training data: {missing_train}"
    )

if missing_val:
    raise ValueError(
        f"Missing features in validation data: {missing_val}"
    )

print("[OK] All selected features exist in training data.")
print("[OK] All selected features exist in validation data.")


# ================================================================
# TARGET VALIDATION
# ================================================================

print()
print("=" * 70)
print("TARGET VALIDATION")
print("=" * 70)

target = "Label"

train_classes = sorted(train_df[target].unique())
val_classes = sorted(val_df[target].unique())

print()
print("Training classes:", len(train_classes))
print("Validation classes:", len(val_classes))

if set(train_classes) != set(val_classes):
    raise ValueError(
        "Training and validation classes do not match."
    )

print("[OK] Training and validation contain the same 34 classes.")


# ================================================================
# LABEL ENCODING
# ================================================================

print()
print("=" * 70)
print("ENCODING MULTICLASS LABELS")
print("=" * 70)

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(
    train_df[target]
)

y_val = label_encoder.transform(
    val_df[target]
)

num_classes = len(label_encoder.classes_)

print()
print("Number of classes:", num_classes)

print()
print("Class mapping:")

for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i:02d} -> {class_name}")

print()
print("[OK] Labels encoded.")


# ================================================================
# PREPARE FEATURES
# ================================================================

print()
print("=" * 70)
print("PREPARING FEATURE MATRICES")
print("=" * 70)

print()
print("Converting features to float32...")

X_train = train_df[features].astype(
    np.float32
).values

X_val = val_df[features].astype(
    np.float32
).values

print()
print("Training matrix:", X_train.shape)
print("Validation matrix:", X_val.shape)

print(
    "Training matrix memory:",
    round(X_train.nbytes / (1024 ** 2), 2),
    "MB"
)

print(
    "Validation matrix memory:",
    round(X_val.nbytes / (1024 ** 2), 2),
    "MB"
)

# Release DataFrames
del train_df
del val_df
gc.collect()

print()
print("[OK] Original DataFrames released.")


# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print()
print("=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

class_counts = np.bincount(y_train)

for class_id, count in enumerate(class_counts):

    class_name = label_encoder.classes_[class_id]

    print(
        f"{class_name:<30} : {count}"
    )


# ================================================================
# CALCULATE CLASS WEIGHTS
# ================================================================

print()
print("=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

total_samples = len(y_train)
number_of_classes = num_classes

class_weights = (
    total_samples /
    (number_of_classes * class_counts)
)

print()
print("Class weights:")

for class_id, weight in enumerate(class_weights):

    class_name = label_encoder.classes_[class_id]

    print(
        f"{class_name:<30} : {weight:.4f}"
    )


# ================================================================
# CREATE SAMPLE WEIGHTS
# ================================================================

sample_weights = class_weights[y_train]

print()
print("[OK] Sample weights created.")


# ================================================================
# MODEL CONFIGURATIONS
# ================================================================

models_config = {

    "XGBoost_Baseline": {
        "use_weights": False,

        "params": {
            "n_estimators": 250,
            "max_depth": 8,
            "learning_rate": 0.10,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "objective": "multi:softprob",
            "num_class": num_classes,
            "tree_method": "hist",
            "n_jobs": 2,
            "random_state": 42
        }
    },

    "XGBoost_ClassWeighted": {
        "use_weights": True,

        "params": {
            "n_estimators": 250,
            "max_depth": 8,
            "learning_rate": 0.10,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "objective": "multi:softprob",
            "num_class": num_classes,
            "tree_method": "hist",
            "n_jobs": 2,
            "random_state": 42
        }
    },

    "XGBoost_Tuned": {
        "use_weights": True,

        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
            "objective": "multi:softprob",
            "num_class": num_classes,
            "tree_method": "hist",
            "n_jobs": 2,
            "random_state": 42
        }
    }
}


# ================================================================
# MODEL EVALUATION FUNCTION
# ================================================================

def evaluate_model(
    model_name,
    config
):

    print()
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    print()
    print("Class weighting:",
          config["use_weights"])

    print()
    print("Parameters:")

    for key, value in config["params"].items():
        print(f"{key}: {value}")

    model = XGBClassifier(
        **config["params"]
    )

    print()
    print("Training model...")

    start_time = time.time()

    if config["use_weights"]:

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )

    else:

        model.fit(
            X_train,
            y_train
        )

    training_time = time.time() - start_time

    print(
        f"Training completed in {training_time:.2f} seconds."
    )

    # ------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------

    print()
    print("Predicting validation set...")

    start_time = time.time()

    y_pred = model.predict(X_val)

    prediction_time = time.time() - start_time

    print(
        f"Prediction completed in {prediction_time:.2f} seconds."
    )

    # ------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_val,
        y_pred
    )

    precision_macro = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print()
    print("-" * 70)
    print("RESULT")
    print("-" * 70)

    print(f"Model              : {model_name}")
    print(f"Accuracy           : {accuracy:.6f}")
    print(f"Balanced Accuracy  : {balanced_accuracy:.6f}")
    print(f"Macro Precision    : {precision_macro:.6f}")
    print(f"Macro Recall       : {recall_macro:.6f}")
    print(f"Macro F1           : {f1_macro:.6f}")
    print(f"Weighted F1        : {f1_weighted:.6f}")
    print(f"Training Time      : {training_time:.2f} sec")
    print(f"Prediction Time    : {prediction_time:.2f} sec")

    result = {
        "model": model_name,
        "feature_count": len(features),
        "class_weighting": config["use_weights"],
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time
    }

    # Release model
    del model
    del y_pred

    gc.collect()

    print()
    print("[OK] Model memory released.")

    return result


# ================================================================
# RUN MODELS
# ================================================================

results = []

for model_name, config in models_config.items():

    result = evaluate_model(
        model_name,
        config
    )

    results.append(result)


# ================================================================
# RESULTS TABLE
# ================================================================

print()
print("=" * 70)
print("XGBOOST IMPROVEMENT COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_macro",
    ascending=False
).reset_index(drop=True)

print()
print(
    results_df.to_string(
        index=False
    )
)


# ================================================================
# BEST MODEL
# ================================================================

best_model = results_df.iloc[0]

print()
print("=" * 70)
print("BEST IMPROVED MODEL")
print("=" * 70)

print()
print("Model:", best_model["model"])
print(
    "Macro-F1:",
    f"{best_model['f1_macro']:.6f}"
)

print(
    "Accuracy:",
    f"{best_model['accuracy']:.6f}"
)

print(
    "Balanced Accuracy:",
    f"{best_model['balanced_accuracy']:.6f}"
)


# ================================================================
# COMPARE WITH STEP 8
# ================================================================

STEP8_BASELINE_F1 = 0.642746

improvement = (
    best_model["f1_macro"]
    - STEP8_BASELINE_F1
)

percentage_improvement = (
    improvement /
    STEP8_BASELINE_F1
) * 100


print()
print("=" * 70)
print("COMPARISON WITH STEP 8")
print("=" * 70)

print()
print(
    "Step 8 XGBoost Macro-F1 :",
    f"{STEP8_BASELINE_F1:.6f}"
)

print(
    "Step 9 Best Macro-F1    :",
    f"{best_model['f1_macro']:.6f}"
)

print(
    "Absolute improvement   :",
    f"{improvement:.6f}"
)

print(
    "Relative improvement   :",
    f"{percentage_improvement:.2f}%"
)


# ================================================================
# SAVE RESULTS
# ================================================================

print()
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print()
print("[OK] Results saved:")
print(RESULT_PATH)


summary = {
    "step": 9,
    "dataset": "CICIoT2023",
    "target": "Label",
    "number_of_classes": num_classes,
    "feature_count": len(features),
    "selected_features": features,

    "test_used": False,
    "smote_used": False,
    "scaling_used": False,
    "new_feature_selection": False,

    "step8_baseline_xgboost_macro_f1":
        STEP8_BASELINE_F1,

    "best_model":
        best_model["model"],

    "best_macro_f1":
        float(best_model["f1_macro"]),

    "best_accuracy":
        float(best_model["accuracy"]),

    "best_balanced_accuracy":
        float(best_model["balanced_accuracy"]),

    "absolute_improvement":
        float(improvement),

    "relative_improvement_percent":
        float(percentage_improvement)
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

print()
print("[OK] Summary saved:")
print(SUMMARY_PATH)


# ================================================================
# FINAL
# ================================================================

print()
print("=" * 70)
print("STEP 9 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("Dataset used:")
print("CICIoT2023")

print()
print("Feature subset:")
print("30 features selected in Step 7")

print()
print("Models evaluated:")
print("1. XGBoost Baseline")
print("2. XGBoost Class-Weighted")
print("3. XGBoost Tuned + Class-Weighted")

print()
print("Test data:")
print("NOT USED")

print()
print("Primary selection metric:")
print("Macro-F1")

print()
print("Next stage:")
print("Final CICIoT2023 evaluation on the TEST dataset.")

print("=" * 70)