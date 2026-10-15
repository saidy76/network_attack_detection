# ================================================================
# STEP 8 - MEMORY-EFFICIENT BASELINE MODEL COMPARISON
# Dataset: CICIoT2023
# Target: Label (34 raw attack classes)
# Features: 30 selected features from Step 7
# Models: Random Forest, Extra Trees, XGBoost
# ================================================================

import os
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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

EXPERIMENT_DIR = os.path.join(
    BASE_DIR,
    "experiments"
)

RESULT_PATH = os.path.join(
    EXPERIMENT_DIR,
    "step8_baseline_model_results.csv"
)

SUMMARY_PATH = os.path.join(
    EXPERIMENT_DIR,
    "step8_baseline_model_summary.json"
)

os.makedirs(EXPERIMENT_DIR, exist_ok=True)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("STEP 8 - MEMORY-EFFICIENT BASELINE MODEL COMPARISON")
print("=" * 70)

print()
print("Dataset:")
print("CICIoT2023")

print()
print("Target:")
print("Label - 34 raw attack classes")

print()
print("Feature selection:")
print("Step 7 selected 30 features")

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
print("Models:")
print("1. Random Forest")
print("2. Extra Trees")
print("3. XGBoost")

print()
print("MEMORY OPTIMIZATION:")
print("- Reduced number of trees")
print("- Limited parallel workers")
print("- Float32 feature matrices")
print("- Explicit garbage collection")
print("- Models deleted after evaluation")

print()
print("IMPORTANT:")
print("Training data is used for model training.")
print("Validation data is used for model comparison.")
print("Test data is NOT used.")
print("No SMOTE is applied.")
print("No scaling is applied.")
print("No new feature selection is performed.")


# ================================================================
# CHECK INPUT FILES
# ================================================================

print()
print("=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

for path in [TRAIN_PATH, VAL_PATH, FEATURE_PATH]:

    if os.path.exists(path):
        print("[OK]", path)
    else:
        print("[ERROR] File not found:", path)
        raise FileNotFoundError(path)


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

print("Training shape:", train_df.shape)

print()
print("Loading validation data...")

val_df = pd.read_csv(VAL_PATH)

print("Validation shape:", val_df.shape)


# ================================================================
# LOAD SELECTED FEATURES
# ================================================================

print()
print("=" * 70)
print("LOADING STEP 7 SELECTED FEATURES")
print("=" * 70)

feature_df = pd.read_csv(FEATURE_PATH)

if "feature" not in feature_df.columns:
    raise ValueError(
        "Step 7 feature file does not contain 'feature' column."
    )

selected_features = feature_df["feature"].tolist()

print()
print("Selected feature count:", len(selected_features))

print()
print("Selected features:")

for i, feature in enumerate(selected_features, start=1):
    print(f"{i:02d}. {feature}")

if len(selected_features) != 30:
    raise ValueError(
        f"Expected 30 selected features, found {len(selected_features)}"
    )

print()
print("[OK] Exactly 30 selected features found.")


# ================================================================
# VALIDATE FEATURE COLUMNS
# ================================================================

print()
print("=" * 70)
print("VALIDATING FEATURE COLUMNS")
print("=" * 70)

missing_train = [
    f for f in selected_features
    if f not in train_df.columns
]

missing_val = [
    f for f in selected_features
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

if "Label" not in train_df.columns:
    raise ValueError("Label column missing from training data.")

if "Label" not in val_df.columns:
    raise ValueError("Label column missing from validation data.")

train_classes = set(train_df["Label"].unique())
val_classes = set(val_df["Label"].unique())

print()
print("Training classes:", len(train_classes))
print("Validation classes:", len(val_classes))

if train_classes != val_classes:
    raise ValueError(
        "Training and validation classes are different."
    )

print()
print(
    "[OK] Training and validation contain the same "
    f"{len(train_classes)} classes."
)


# ================================================================
# ENCODE LABELS
# ================================================================

print()
print("=" * 70)
print("ENCODING MULTICLASS LABELS")
print("=" * 70)

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(
    train_df["Label"]
)

y_val = label_encoder.transform(
    val_df["Label"]
)

num_classes = len(label_encoder.classes_)

print()
print("Number of classes:", num_classes)

print("[OK] Labels encoded for multiclass models.")


# ================================================================
# PREPARE FEATURE MATRICES
# ================================================================

print()
print("=" * 70)
print("PREPARING FEATURE MATRICES")
print("=" * 70)

print()
print("Converting features to float32...")

X_train = train_df[selected_features].to_numpy(
    dtype=np.float32
)

X_val = val_df[selected_features].to_numpy(
    dtype=np.float32
)

print()
print("Training feature matrix:", X_train.shape)
print("Validation feature matrix:", X_val.shape)

print()
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


# ================================================================
# FREE DATAFRAME MEMORY
# ================================================================

del train_df
del val_df
gc.collect()

print()
print("[OK] Original DataFrames released from memory.")


# ================================================================
# MODEL EVALUATION FUNCTION
# ================================================================

results = []


def evaluate_model(model_name, model):

    print()
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    print()
    print("Training model...")

    start_train = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_train

    print(
        f"Training completed in {training_time:.2f} seconds."
    )

    print()
    print("Predicting validation set...")

    start_pred = time.time()

    y_pred = model.predict(X_val)

    prediction_time = time.time() - start_pred

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

    print(
        f"Training Time     : {training_time:.2f} sec"
    )

    print(
        f"Prediction Time   : {prediction_time:.2f} sec"
    )

    results.append({
        "model": model_name,
        "feature_count": len(selected_features),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time
    })


# ================================================================
# MODEL 1 - RANDOM FOREST
# ================================================================

rf_model = RandomForestClassifier(
    n_estimators=80,
    max_depth=25,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=None,
    random_state=42,
    n_jobs=2
)

evaluate_model(
    "Random Forest",
    rf_model
)

del rf_model
gc.collect()

print()
print("[OK] Random Forest memory released.")


# ================================================================
# MODEL 2 - EXTRA TREES
# ================================================================

et_model = ExtraTreesClassifier(
    n_estimators=80,
    max_depth=25,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=None,
    random_state=42,
    n_jobs=2
)

evaluate_model(
    "Extra Trees",
    et_model
)

del et_model
gc.collect()

print()
print("[OK] Extra Trees memory released.")


# ================================================================
# MODEL 3 - XGBOOST
# ================================================================

xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.10,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="hist",
    max_bin=128,
    random_state=42,
    n_jobs=2
)

evaluate_model(
    "XGBoost",
    xgb_model
)

del xgb_model
gc.collect()

print()
print("[OK] XGBoost memory released.")


# ================================================================
# MODEL COMPARISON
# ================================================================

print()
print("=" * 70)
print("BASELINE MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_macro",
    ascending=False
).reset_index(drop=True)

print()
print(results_df.to_string(index=False))


# ================================================================
# BEST MODEL
# ================================================================

best_model = results_df.iloc[0]

print()
print("=" * 70)
print("BEST BASELINE MODEL")
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


# ================================================================
# SAVE SUMMARY
# ================================================================

summary = {
    "step": 8,
    "dataset": "CICIoT2023",
    "target": "Label",
    "number_of_classes": int(num_classes),
    "selected_feature_count": len(selected_features),
    "selected_features": selected_features,
    "models_evaluated": [
        "Random Forest",
        "Extra Trees",
        "XGBoost"
    ],
    "primary_metric": "Macro-F1",
    "test_data_used": False,
    "smote_used": False,
    "scaling_used": False,
    "new_feature_selection": False,
    "best_model": str(best_model["model"]),
    "best_macro_f1": float(best_model["f1_macro"]),
    "best_accuracy": float(best_model["accuracy"]),
    "best_balanced_accuracy": float(
        best_model["balanced_accuracy"]
    )
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
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("Dataset used:")
print("CICIoT2023")

print()
print("Feature subset:")
print("30 features selected in Step 7")

print()
print("Models compared:")
print("1. Random Forest")
print("2. Extra Trees")
print("3. XGBoost")

print()
print("Test data:")
print("NOT USED")

print()
print("Primary selection metric:")
print("Macro-F1")

print()
print("Next stage:")
print("Select the strongest baseline model and proceed to")
print("model improvement / imbalance handling.")