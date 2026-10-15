"""
==============================================================================
STEP 18 - CIC-IDS2018 BASELINE MODEL EXPERIMENTS
==============================================================================

Purpose
-------
Establish reproducible baseline models for CIC-IDS2018 multiclass intrusion
detection unde different TRAINING-ONLY imbalance strategies.

Training strategies
--------------------
B0 : Original imbalanced training
B1 : Controlled majority undersampling
B2 : Hybrid controlled sampling

Models
------
1. Random Forest
2. XGBoost

Validation
----------
Used ONLY for model comparison.

Test
----
COMPLETELY UNTOUCHED.

Important
---------
- No test data is loaded.
- No feature selection is performed here.
- No SMOTE is used.
- No validation balancing is performed.
- Class weights are mapped correctly to encoded class IDs.
- Same validation set is used for every experiment.
- Fixed random seeds ensure reproducibility.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input directories
MODEL_DIR = PROJECT_ROOT / "data" / "modeling"
BALANCED_DIR = MODEL_DIR / "balanced"

EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
MODEL_OUTPUT_DIR = EXPERIMENT_DIR / "models"
REPORT_OUTPUT_DIR = EXPERIMENT_DIR / "step18_reports"

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# INPUT FILES
# =============================================================================

TRAINING_FILES = {
    "B0_original": BALANCED_DIR / "CICIDS2018_train_B0_original.csv",
    "B1_undersampled": BALANCED_DIR / "CICIDS2018_train_B1_undersampled.csv",
    "B2_hybrid": BALANCED_DIR / "CICIDS2018_train_B2_hybrid.csv",
}

VALIDATION_FILE = (
    MODEL_DIR / "CICIDS2018_validation_selected.csv"
)

CLASS_WEIGHT_FILE = (
    EXPERIMENT_DIR / "step17_cicids2018_class_weights.csv"
)


# =============================================================================
# OUTPUT FILES
# =============================================================================

RESULTS_FILE = (
    EXPERIMENT_DIR / "step18_cicids2018_baseline_results.csv"
)

PER_CLASS_FILE = (
    EXPERIMENT_DIR / "step18_cicids2018_per_class_metrics.csv"
)

CONFUSION_MATRIX_FILE = (
    EXPERIMENT_DIR / "step18_cicids2018_confusion_matrices.npz"
)

SUMMARY_FILE = (
    EXPERIMENT_DIR / "step18_cicids2018_summary.json"
)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Random Forest configuration
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


# XGBoost configuration
#
# tree_method="hist" is selected because CIC-IDS2018 contains millions of
# training samples and 30 selected numerical features.
#
# multi:softprob is used because this is a 14-class multiclass problem.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.10,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_subheader(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def check_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"[ERROR] {description} not found:\n{path}"
        )

    print(f"[OK] {description}: {path}")


def calculate_metrics(y_true, y_pred):
    """
    Calculate robust multiclass metrics.

    Macro metrics are particularly important for this dataset because
    CIC-IDS2018 is severely imbalanced.
    """

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Balanced_Accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),

        "Macro_Precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Macro_Recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Macro_F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Weighted_Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Weighted_Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Weighted_F1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
    }

    return metrics


def create_model_name(strategy, model_name):
    return f"{strategy}_{model_name}"


# =============================================================================
# START
# =============================================================================

overall_start = time.time()

print_header(
    "STEP 18 - CIC-IDS2018 BASELINE MODEL EXPERIMENTS"
)

print(
    """
Purpose:
Establish reproducible baseline models.

Training strategies:
- B0: Original imbalanced
- B1: Controlled majority undersampling
- B2: Hybrid controlled sampling

Models:
- Random Forest
- XGBoost

Validation:
USED for model comparison.

Test:
COMPLETELY UNTOUCHED.
"""
)


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

print_header("CHECKING INPUT FILES")

for strategy_name, path in TRAINING_FILES.items():
    check_file(path, strategy_name)

check_file(
    VALIDATION_FILE,
    "Validation dataset"
)

check_file(
    CLASS_WEIGHT_FILE,
    "Class-weight file"
)

print()
print("[IMPORTANT] Test dataset is NOT loaded.")


# =============================================================================
# LOAD VALIDATION DATA
# =============================================================================

print_header("LOADING VALIDATION DATA")

validation_start = time.time()

print("Loading CICIDS2018_validation_selected.csv...")

validation_df = pd.read_csv(
    VALIDATION_FILE,
    low_memory=False
)

print("[OK] Validation dataset loaded.")

print(
    f"Shape: {validation_df.shape}"
)

print(
    f"Loading time: {time.time() - validation_start:.2f} sec"
)


# =============================================================================
# VALIDATION CHECKS
# =============================================================================

print_header("VALIDATION DATA CHECK")

if "Label" not in validation_df.columns:
    raise ValueError(
        "Label column not found in validation dataset."
    )

feature_columns = [
    col for col in validation_df.columns
    if col != "Label"
]

if len(feature_columns) != 30:
    raise ValueError(
        f"Expected 30 selected features, "
        f"found {len(feature_columns)}."
    )

if validation_df[feature_columns].isnull().sum().sum() > 0:
    raise ValueError(
        "Missing values found in validation features."
    )

if np.isinf(
    validation_df[feature_columns].to_numpy(
        dtype=np.float64
    )
).sum() > 0:
    raise ValueError(
        "Infinity values found in validation features."
    )

print("[OK] Label column found.")
print("[OK] 30 selected features found.")
print("[OK] No missing values.")
print("[OK] No Infinity values.")


# =============================================================================
# LABEL ENCODING
# =============================================================================

print_header("PREPARING TARGET LABEL ENCODING")

# Important:
# We create ONE GLOBAL class mapping from the validation + known CICIDS labels.
#
# The class ordering is deterministic and is later used consistently for
# every training strategy and every model.

classes = sorted(
    validation_df["Label"].unique().tolist()
)

class_to_id = {
    label: idx
    for idx, label in enumerate(classes)
}

id_to_class = {
    idx: label
    for label, idx in class_to_id.items()
}

print(
    f"[OK] Number of classes detected: {len(classes)}"
)

print()
print("Class mapping:")

for idx, label in id_to_class.items():
    print(
        f"{idx:02d}. {label}"
    )


# Encode validation target
unknown_validation_labels = set(
    validation_df["Label"]
) - set(class_to_id)

if unknown_validation_labels:
    raise ValueError(
        f"Unknown validation labels found: "
        f"{unknown_validation_labels}"
    )

y_validation = (
    validation_df["Label"]
    .map(class_to_id)
    .astype(np.int32)
    .to_numpy()
)

X_validation = (
    validation_df[feature_columns]
    .astype(np.float32)
    .to_numpy()
)

print()
print(
    f"Validation feature matrix: "
    f"{X_validation.shape}"
)

print(
    f"Validation target vector: "
    f"{y_validation.shape}"
)


# Free validation dataframe after matrix creation
del validation_df


# =============================================================================
# LOAD CLASS WEIGHTS
# =============================================================================

print_header("LOADING CLASS WEIGHTS")

class_weight_df = pd.read_csv(
    CLASS_WEIGHT_FILE
)

required_weight_columns = {
    "Label",
    "Class_Weight"
}

missing_weight_columns = (
    required_weight_columns
    - set(class_weight_df.columns)
)

if missing_weight_columns:
    raise ValueError(
        "Class-weight file is missing columns: "
        f"{missing_weight_columns}"
    )

print("[OK] Class-weight file loaded.")

# -------------------------------------------------------------------------
# IMPORTANT CORRECTION
# -------------------------------------------------------------------------
#
# The previous code attempted to give RandomForest class_weight using
# string labels while y_train contained encoded integer labels.
#
# That causes:
#
# ValueError:
# The classes, [0, 1, ... 13], are not in class_weight
#
# Therefore, we explicitly convert:
#
# "BENIGN" -> 0
# "Bot"    -> 1
# ...
#
# into:
#
# {0: weight0, 1: weight1, ...}
#
# This is the critical correction.
# -------------------------------------------------------------------------

class_weights_by_label = dict(
    zip(
        class_weight_df["Label"],
        class_weight_df["Class_Weight"]
    )
)

missing_weights = (
    set(classes)
    - set(class_weights_by_label.keys())
)

if missing_weights:
    raise ValueError(
        "Class weights missing for classes: "
        f"{missing_weights}"
    )

encoded_class_weights = {
    class_to_id[label]: float(
        class_weights_by_label[label]
    )
    for label in classes
}

print("[OK] Class weights successfully mapped to encoded IDs.")

print()
print("Encoded class weights:")

for class_id in sorted(encoded_class_weights):
    print(
        f"{class_id:02d} "
        f"{id_to_class[class_id]:35s} "
        f"{encoded_class_weights[class_id]:.6f}"
    )


# =============================================================================
# RESULT STORAGE
# =============================================================================

all_results = []
all_per_class = {}
all_confusion_matrices = {}


# =============================================================================
# EXPERIMENT FUNCTION
# =============================================================================

def run_experiment(
    strategy_name,
    model_name,
    model,
    training_file,
    use_class_weight=False
):

    experiment_start = time.time()

    experiment_id = create_model_name(
        strategy_name,
        model_name
    )

    print_header(
        f"EXPERIMENT: {experiment_id}"
    )

    print(
        f"Training strategy : {strategy_name}"
    )

    print(
        f"Model              : {model_name}"
    )

    print(
        f"Training file      : {training_file}"
    )

    print(
        f"Class-weight mode  : "
        f"{'ENABLED' if use_class_weight else 'DISABLED'}"
    )

    # -------------------------------------------------------------------------
    # LOAD TRAINING DATA
    # -------------------------------------------------------------------------

    load_start = time.time()

    print()
    print("Loading training dataset...")

    train_df = pd.read_csv(
        training_file,
        low_memory=False
    )

    print(
        f"[OK] Training dataset loaded in "
        f"{time.time() - load_start:.2f} sec"
    )

    # -------------------------------------------------------------------------
    # VALIDATE TRAINING DATA
    # -------------------------------------------------------------------------

    if "Label" not in train_df.columns:
        raise ValueError(
            f"Label column missing in {training_file}"
        )

    train_features = [
        col for col in train_df.columns
        if col != "Label"
    ]

    if train_features != feature_columns:
        missing = set(feature_columns) - set(train_features)
        extra = set(train_features) - set(feature_columns)

        raise ValueError(
            f"Feature mismatch.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Training features: {len(train_features)}"
    )

    # -------------------------------------------------------------------------
    # CHECK LABELS
    # -------------------------------------------------------------------------

    unknown_labels = (
        set(train_df["Label"])
        - set(class_to_id)
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown labels found in {strategy_name}: "
            f"{unknown_labels}"
        )

    # -------------------------------------------------------------------------
    # CREATE MATRICES
    # -------------------------------------------------------------------------

    X_train = (
        train_df[feature_columns]
        .astype(np.float32)
        .to_numpy()
    )

    y_train = (
        train_df["Label"]
        .map(class_to_id)
        .astype(np.int32)
        .to_numpy()
    )

    print(
        f"Training matrix: {X_train.shape}"
    )

    # -------------------------------------------------------------------------
    # RELEASE DATAFRAME
    # -------------------------------------------------------------------------

    del train_df

    # -------------------------------------------------------------------------
    # CLASS-WEIGHT CORRECTION
    # -------------------------------------------------------------------------

    if use_class_weight:

        if model_name == "RandomForest":

            # RandomForest expects:
            # {encoded_class_id: weight}

            model.set_params(
                class_weight=encoded_class_weights
            )

            print(
                "[OK] Random Forest class_weight "
                "mapped to encoded class IDs."
            )

        elif model_name == "XGBoost":

            # XGBoost multiclass does not use sklearn's
            # class_weight parameter.
            #
            # Instead, sample_weight is passed to fit().
            #
            sample_weights = np.array(
                [
                    encoded_class_weights[int(label)]
                    for label in y_train
                ],
                dtype=np.float32
            )

            print(
                "[OK] XGBoost sample weights prepared."
            )

        else:
            sample_weights = None

    else:
        sample_weights = None

    # -------------------------------------------------------------------------
    # TRAIN MODEL
    # -------------------------------------------------------------------------

    print()
    print("Training model...")

    training_start = time.time()

    if model_name == "XGBoost":

        if use_class_weight:

            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                eval_set=[
                    (X_validation, y_validation)
                ],
                verbose=False
            )

        else:

            model.fit(
                X_train,
                y_train,
                eval_set=[
                    (X_validation, y_validation)
                ],
                verbose=False
            )

    else:

        model.fit(
            X_train,
            y_train
        )

    training_time = (
        time.time()
        - training_start
    )

    print(
        f"[OK] Training completed in "
        f"{training_time:.2f} sec"
    )

    # -------------------------------------------------------------------------
    # VALIDATION PREDICTION
    # -------------------------------------------------------------------------

    print()
    print("Generating validation predictions...")

    prediction_start = time.time()

    y_pred = model.predict(
        X_validation
    )

    prediction_time = (
        time.time()
        - prediction_start
    )

    print(
        f"[OK] Prediction completed in "
        f"{prediction_time:.2f} sec"
    )

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------

    print()
    print("Calculating validation metrics...")

    metrics = calculate_metrics(
        y_validation,
        y_pred
    )

    print()
    print(
        f"Accuracy          : "
        f"{metrics['Accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{metrics['Balanced_Accuracy']:.6f}"
    )

    print(
        f"Macro Precision   : "
        f"{metrics['Macro_Precision']:.6f}"
    )

    print(
        f"Macro Recall      : "
        f"{metrics['Macro_Recall']:.6f}"
    )

    print(
        f"Macro F1          : "
        f"{metrics['Macro_F1']:.6f}"
    )

    print(
        f"Weighted F1       : "
        f"{metrics['Weighted_F1']:.6f}"
    )

    # -------------------------------------------------------------------------
    # PER-CLASS REPORT
    # -------------------------------------------------------------------------

    report = classification_report(
        y_validation,
        y_pred,
        labels=list(range(len(classes))),
        target_names=[
            id_to_class[i]
            for i in range(len(classes))
        ],
        output_dict=True,
        zero_division=0
    )

    per_class_rows = []

    for class_id, class_name in id_to_class.items():

        row = {
            "Experiment": experiment_id,
            "Strategy": strategy_name,
            "Model": model_name,
            "Class_ID": class_id,
            "Label": class_name,
            "Precision": report[class_name]["precision"],
            "Recall": report[class_name]["recall"],
            "F1": report[class_name]["f1-score"],
            "Support": int(report[class_name]["support"]),
        }

        per_class_rows.append(row)

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    all_per_class[experiment_id] = per_class_df

    # -------------------------------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------------------------------

    cm = confusion_matrix(
        y_validation,
        y_pred,
        labels=list(range(len(classes)))
    )

    all_confusion_matrices[experiment_id] = cm

    # -------------------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------------------

    model_file = (
        MODEL_OUTPUT_DIR
        / f"{experiment_id}.joblib"
    )

    joblib.dump(
        model,
        model_file
    )

    print(
        f"[OK] Model saved:\n{model_file}"
    )

    # -------------------------------------------------------------------------
    # SAVE PER-CLASS REPORT
    # -------------------------------------------------------------------------

    per_class_file = (
        REPORT_OUTPUT_DIR
        / f"{experiment_id}_per_class.csv"
    )

    per_class_df.to_csv(
        per_class_file,
        index=False
    )

    print(
        f"[OK] Per-class metrics saved:\n"
        f"{per_class_file}"
    )

    # -------------------------------------------------------------------------
    # STORE SUMMARY
    # -------------------------------------------------------------------------

    result = {
        "Experiment": experiment_id,
        "Strategy": strategy_name,
        "Model": model_name,
        "Training_Rows": int(len(y_train)),
        "Validation_Rows": int(len(y_validation)),
        "Features": int(len(feature_columns)),
        "Classes": int(len(classes)),
        "Class_Weight_Used": bool(use_class_weight),
        "Training_Time_sec": round(
            training_time,
            4
        ),
        "Prediction_Time_sec": round(
            prediction_time,
            4
        ),
        "Accuracy": metrics["Accuracy"],
        "Balanced_Accuracy": metrics["Balanced_Accuracy"],
        "Macro_Precision": metrics["Macro_Precision"],
        "Macro_Recall": metrics["Macro_Recall"],
        "Macro_F1": metrics["Macro_F1"],
        "Weighted_Precision": metrics["Weighted_Precision"],
        "Weighted_Recall": metrics["Weighted_Recall"],
        "Weighted_F1": metrics["Weighted_F1"],
        "Model_File": str(model_file),
    }

    all_results.append(result)

    print()
    print(
        f"[OK] Experiment completed: "
        f"{experiment_id}"
    )

    print(
        f"Total experiment time: "
        f"{time.time() - experiment_start:.2f} sec"
    )

    # Release large matrices
    del X_train
    del y_train
    del y_pred

    return result


# =============================================================================
# EXPERIMENT 1
# B0 + RANDOM FOREST
# =============================================================================

model = RandomForestClassifier(
    **RF_PARAMS
)

run_experiment(
    strategy_name="B0_original",
    model_name="RandomForest",
    model=model,
    training_file=TRAINING_FILES["B0_original"],
    use_class_weight=True
)


# =============================================================================
# EXPERIMENT 2
# B1 + RANDOM FOREST
# =============================================================================

model = RandomForestClassifier(
    **RF_PARAMS
)

run_experiment(
    strategy_name="B1_undersampled",
    model_name="RandomForest",
    model=model,
    training_file=TRAINING_FILES["B1_undersampled"],
    use_class_weight=False
)


# =============================================================================
# EXPERIMENT 3
# B2 + RANDOM FOREST
# =============================================================================

model = RandomForestClassifier(
    **RF_PARAMS
)

run_experiment(
    strategy_name="B2_hybrid",
    model_name="RandomForest",
    model=model,
    training_file=TRAINING_FILES["B2_hybrid"],
    use_class_weight=False
)


# =============================================================================
# EXPERIMENT 4
# B0 + XGBOOST
# =============================================================================

model = XGBClassifier(
    **XGB_PARAMS,
    num_class=len(classes)
)

run_experiment(
    strategy_name="B0_original",
    model_name="XGBoost",
    model=model,
    training_file=TRAINING_FILES["B0_original"],
    use_class_weight=True
)


# =============================================================================
# EXPERIMENT 5
# B1 + XGBOOST
# =============================================================================

model = XGBClassifier(
    **XGB_PARAMS,
    num_class=len(classes)
)

run_experiment(
    strategy_name="B1_undersampled",
    model_name="XGBoost",
    model=model,
    training_file=TRAINING_FILES["B1_undersampled"],
    use_class_weight=False
)


# =============================================================================
# EXPERIMENT 6
# B2 + XGBOOST
# =============================================================================

model = XGBClassifier(
    **XGB_PARAMS,
    num_class=len(classes)
)

run_experiment(
    strategy_name="B2_hybrid",
    model_name="XGBoost",
    model=model,
    training_file=TRAINING_FILES["B2_hybrid"],
    use_class_weight=False
)


# =============================================================================
# SAVE COMBINED RESULTS
# =============================================================================

print_header(
    "SAVING BASELINE EXPERIMENT RESULTS"
)

results_df = pd.DataFrame(
    all_results
)

# Sort primarily by Macro-F1 because this is a highly imbalanced
# multiclass intrusion detection problem.
results_df = results_df.sort_values(
    by=[
        "Macro_F1",
        "Balanced_Accuracy",
        "Weighted_F1"
    ],
    ascending=False
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

print(
    f"[OK] Overall results saved:\n"
    f"{RESULTS_FILE}"
)


# =============================================================================
# SAVE COMBINED PER-CLASS RESULTS
# =============================================================================

combined_per_class = pd.concat(
    all_per_class.values(),
    ignore_index=True
)

combined_per_class.to_csv(
    PER_CLASS_FILE,
    index=False
)

print(
    f"[OK] Combined per-class metrics saved:\n"
    f"{PER_CLASS_FILE}"
)


# =============================================================================
# SAVE CONFUSION MATRICES
# =============================================================================

np.savez_compressed(
    CONFUSION_MATRIX_FILE,
    **all_confusion_matrices
)

print(
    f"[OK] Confusion matrices saved:\n"
    f"{CONFUSION_MATRIX_FILE}"
)


# =============================================================================
# FINAL RANKING
# =============================================================================

print_header(
    "BASELINE MODEL RANKING"
)

display_columns = [
    "Experiment",
    "Strategy",
    "Model",
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_F1",
]

print(
    results_df[display_columns].to_string(
        index=False
    )
)


# =============================================================================
# BEST MODEL
# =============================================================================

best_result = results_df.iloc[0].to_dict()

print()
print(
    "BEST BASELINE MODEL"
)

print(
    f"Experiment        : "
    f"{best_result['Experiment']}"
)

print(
    f"Strategy          : "
    f"{best_result['Strategy']}"
)

print(
    f"Model             : "
    f"{best_result['Model']}"
)

print(
    f"Accuracy          : "
    f"{best_result['Accuracy']:.6f}"
)

print(
    f"Balanced Accuracy : "
    f"{best_result['Balanced_Accuracy']:.6f}"
)

print(
    f"Macro F1          : "
    f"{best_result['Macro_F1']:.6f}"
)

print(
    f"Weighted F1       : "
    f"{best_result['Weighted_F1']:.6f}"
)


# =============================================================================
# SAVE SUMMARY JSON
# =============================================================================

summary = {
    "step": 18,
    "dataset": "CIC-IDS2018",

    "purpose": (
        "Reproducible multiclass baseline evaluation "
        "under training-only imbalance strategies."
    ),

    "training_strategies": [
        "B0_original",
        "B1_undersampled",
        "B2_hybrid"
    ],

    "models": [
        "RandomForest",
        "XGBoost"
    ],

    "features": 30,
    "classes": len(classes),

    "validation_used": True,
    "test_used": False,

    "smote_used": False,
    "feature_selection_performed": False,

    "random_state": RANDOM_STATE,

    "metrics": [
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "Weighted_Precision",
        "Weighted_Recall",
        "Weighted_F1"
    ],

    "class_mapping": class_to_id,

    "best_baseline": best_result,

    "result_file": str(RESULTS_FILE),
    "per_class_file": str(PER_CLASS_FILE),
    "confusion_matrix_file": str(CONFUSION_MATRIX_FILE),

    "model_directory": str(
        MODEL_OUTPUT_DIR
    ),

    "total_experiments": len(
        all_results
    ),

    "total_processing_time_sec": round(
        time.time() - overall_start,
        4
    )
}

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


# =============================================================================
# FINAL
# =============================================================================

print_header(
    "STEP 18 COMPLETED SUCCESSFULLY"
)

print(
    """
CIC-IDS2018 baseline model experiments completed.

Training strategies evaluated:
- B0 Original
- B1 Controlled Undersampling
- B2 Hybrid Controlled Sampling

Models evaluated:
- Random Forest
- XGBoost

Validation:
USED

Test:
NOT USED

SMOTE:
NOT USED

Feature selection:
NOT PERFORMED

Evaluation emphasis:
- Macro-F1
- Balanced Accuracy
- Macro Recall
- Macro Precision
- Weighted-F1
- Accuracy

The test dataset remains completely untouched.

Next:
STEP 19 - BASELINE MODEL ANALYSIS AND
STATISTICAL COMPARISON
"""
)

print()
print(
    f"Total processing time: "
    f"{time.time() - overall_start:.2f} sec"
)

print()
print("[OK] Step 18 finished.")