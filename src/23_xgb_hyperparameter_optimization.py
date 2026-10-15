from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "balanced"
    / "CICIDS2018_train_B1_undersampled.csv"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "CICIDS2018_validation_selected.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step23_reports"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step23_xgb_tuning"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "BENIGN",
    "Bot",
    "Brute Force -Web",
    "Brute Force -XSS",
    "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP",
    "DoS attacks-GoldenEye",
    "DoS attacks-Hulk",
    "DoS attacks-SlowHTTPTest",
    "DoS attacks-Slowloris",
    "FTP-BruteForce",
    "Infiltration",
    "SQL Injection",
    "SSH-Bruteforce",
]

CLASS_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(CLASS_NAMES)
}


ID_TO_CLASS = {
    class_id: class_name
    for class_id, class_name in enumerate(CLASS_NAMES)
}


# ============================================================
# START
# ============================================================

print("=" * 78)
print("STEP 23 - CONTROLLED XGBOOST HYPERPARAMETER OPTIMIZATION")
print("=" * 78)

print(f"Project root : {PROJECT_ROOT}")
print(f"Training file: {TRAIN_FILE}")
print(f"Validation   : {VALIDATION_FILE}")
print(f"Output dir   : {OUTPUT_DIR}")


# ============================================================
# CHECK INPUTS
# ============================================================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 78)
print("LOADING B1 TRAINING DATA")
print("=" * 78)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VALIDATION_FILE)

print(f"Training shape   : {train_df.shape}")
print(f"Validation shape : {val_df.shape}")


if "Label" not in train_df.columns:
    raise ValueError(
        "Training dataset has no Label column."
    )

if "Label" not in val_df.columns:
    raise ValueError(
        "Validation dataset has no Label column."
    )


# ============================================================
# FEATURE VERIFICATION
# ============================================================

FEATURE_COLUMNS = [
    column
    for column in train_df.columns
    if column != "Label"
]

if len(FEATURE_COLUMNS) != 30:
    raise ValueError(
        f"Expected 30 features, found {len(FEATURE_COLUMNS)}"
    )


missing_validation_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in val_df.columns
]

if missing_validation_features:
    raise ValueError(
        "Validation dataset is missing features:\n"
        + "\n".join(missing_validation_features)
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
X_val = val_df[FEATURE_COLUMNS]


# ============================================================
# LABEL ENCODING
# ============================================================

print("\n" + "=" * 78)
print("ENCODING CLASS LABELS")
print("=" * 78)

unknown_train_labels = sorted(
    set(train_df["Label"].astype(str))
    - set(CLASS_TO_ID.keys())
)

unknown_val_labels = sorted(
    set(val_df["Label"].astype(str))
    - set(CLASS_TO_ID.keys())
)


if unknown_train_labels:
    raise ValueError(
        "Unknown labels found in training dataset:\n"
        + "\n".join(unknown_train_labels)
    )


if unknown_val_labels:
    raise ValueError(
        "Unknown labels found in validation dataset:\n"
        + "\n".join(unknown_val_labels)
    )


y_train = (
    train_df["Label"]
    .astype(str)
    .map(CLASS_TO_ID)
    .astype(np.int32)
)

y_val = (
    val_df["Label"]
    .astype(str)
    .map(CLASS_TO_ID)
    .astype(np.int32)
)


if y_train.isna().any():
    raise ValueError(
        "NaN values detected after training label encoding."
    )


if y_val.isna().any():
    raise ValueError(
        "NaN values detected after validation label encoding."
    )


print(f"Feature count     : {len(FEATURE_COLUMNS)}")
print(f"Training samples  : {len(X_train)}")
print(f"Validation rows   : {len(X_val)}")
print(
    f"Training classes  : {sorted(y_train.unique().tolist())}"
)
print(
    f"Validation classes: {sorted(y_val.unique().tolist())}"
)


# ============================================================
# CONTROLLED SEARCH SPACE
# ============================================================

PARAM_DISTRIBUTIONS = {
    "n_estimators": [200, 300, 400],
    "max_depth": [6, 8, 10],
    "learning_rate": [0.05, 0.10, 0.15],
    "subsample": [0.70, 0.80, 0.90],
    "colsample_bytree": [0.70, 0.80, 0.90],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
    "reg_lambda": [1, 2, 5],
}

N_ITER = 12
CV_FOLDS = 3


print("\n" + "=" * 78)
print("CONTROLLED SEARCH DESIGN")
print("=" * 78)

print(f"Randomized configurations : {N_ITER}")
print(f"Cross-validation folds    : {CV_FOLDS}")
print(f"Total model fits          : {N_ITER * CV_FOLDS}")
print("Primary metric            : Macro F1")
print(f"Random state              : {RANDOM_STATE}")
print("Training representation   : B1 undersampled")
print("Validation used in tuning : NO")
print("Test set used             : NO")


# ============================================================
# BASE XGBOOST
# ============================================================

BASE_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "num_class": len(CLASS_NAMES),
}

model = XGBClassifier(
    **BASE_PARAMS
)


# ============================================================
# STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# RANDOMIZED SEARCH
# ============================================================

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=PARAM_DISTRIBUTIONS,
    n_iter=N_ITER,
    scoring="f1_macro",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=1,
    verbose=2,
    refit=True,
)


# ============================================================
# RUN SEARCH
# ============================================================

print("\n" + "=" * 78)
print("STARTING CONTROLLED XGBOOST SEARCH")
print("=" * 78)

start_time = time.time()

search.fit(
    X_train,
    y_train
)

search_time = time.time() - start_time


print("\n" + "=" * 78)
print("SEARCH COMPLETED")
print("=" * 78)

print(
    f"Search time : {search_time / 3600:.4f} hours"
)

print(
    f"Best CV Macro F1 : {search.best_score_:.6f}"
)

print("\nBest parameters:")

for key, value in search.best_params_.items():
    print(f"  {key}: {value}")


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 78)
print("VALIDATION EVALUATION")
print("=" * 78)

best_model = search.best_estimator_

y_pred = best_model.predict(X_val)

y_pred = np.asarray(y_pred).astype(np.int32)


accuracy = accuracy_score(
    y_val,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_val,
    y_pred
)

macro_precision = precision_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_val,
    y_pred,
    average="weighted",
    zero_division=0
)


print(f"Accuracy           : {accuracy:.6f}")
print(f"Balanced Accuracy  : {balanced_accuracy:.6f}")
print(f"Macro Precision    : {macro_precision:.6f}")
print(f"Macro Recall       : {macro_recall:.6f}")
print(f"Macro F1           : {macro_f1:.6f}")
print(f"Weighted F1        : {weighted_f1:.6f}")


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print("\n" + "=" * 78)
print("PER-CLASS VALIDATION RESULTS")
print("=" * 78)

report = classification_report(
    y_val,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)


per_class = []

for class_id, class_name in enumerate(CLASS_NAMES):

    row = report[class_name]

    per_class.append(
        {
            "Class_ID": class_id,
            "Label": class_name,
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1-score"],
            "Support": row["support"],
        }
    )


per_class_df = pd.DataFrame(per_class)

print(
    per_class_df.to_string(index=False)
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 78)
print("CONFUSION MATRIX")
print("=" * 78)

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
)


cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

print(cm_df)


# ============================================================
# SAVE BEST PARAMETERS + SUMMARY
# ============================================================

metrics = {
    "experiment": "Step23_Tuned_XGBoost",
    "dataset": "CIC-IDS2018",
    "training_strategy": "B1_undersampled",
    "optimization_method": "RandomizedSearchCV",
    "primary_metric": "f1_macro",
    "best_cv_macro_f1": float(search.best_score_),
    "best_parameters": search.best_params_,
    "random_state": RANDOM_STATE,
    "n_iter": N_ITER,
    "cv_folds": CV_FOLDS,
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "feature_count": len(FEATURE_COLUMNS),
    "class_count": len(CLASS_NAMES),
    "label_encoding": CLASS_TO_ID,
    "search_time_hours": float(
        search_time / 3600
    ),
    "validation_metrics": {
        "Accuracy": float(accuracy),
        "Balanced_Accuracy": float(balanced_accuracy),
        "Macro_Precision": float(macro_precision),
        "Macro_Recall": float(macro_recall),
        "Macro_F1": float(macro_f1),
        "Weighted_F1": float(weighted_f1),
    },
}


with open(
    OUTPUT_DIR / "step23_best_xgb_parameters.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE RANDOMIZED SEARCH RESULTS
# ============================================================

pd.DataFrame(
    search.cv_results_
).to_csv(
    OUTPUT_DIR
    / "step23_xgb_randomized_search_results.csv",
    index=False,
)


# ============================================================
# SAVE VALIDATION METRICS
# ============================================================

pd.DataFrame(
    [
        {
            "Experiment": "Step23_Tuned_XGBoost",
            "Dataset": "CIC-IDS2018",
            "Training_Strategy": "B1_undersampled",
            "Accuracy": accuracy,
            "Balanced_Accuracy": balanced_accuracy,
            "Macro_Precision": macro_precision,
            "Macro_Recall": macro_recall,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1,
            "Best_CV_Macro_F1": search.best_score_,
            "N_Iterations": N_ITER,
            "CV_Folds": CV_FOLDS,
            "Random_State": RANDOM_STATE,
            "Training_Samples": len(X_train),
            "Validation_Samples": len(X_val),
            "Feature_Count": len(FEATURE_COLUMNS),
            "Search_Time_Hours": search_time / 3600,
        }
    ]
).to_csv(
    OUTPUT_DIR
    / "step23_xgb_validation_metrics.csv",
    index=False,
)


# ============================================================
# SAVE PER-CLASS METRICS
# ============================================================

per_class_df.to_csv(
    OUTPUT_DIR
    / "step23_xgb_validation_per_class_metrics.csv",
    index=False,
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df.to_csv(
    OUTPUT_DIR
    / "step23_xgb_validation_confusion_matrix.csv"
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_DIR / "step23_best_xgboost.joblib",
)


# ============================================================
# OPTIMIZATION PROTOCOL
# ============================================================

protocol = f"""
STEP 23 - CONTROLLED XGBOOST HYPERPARAMETER OPTIMIZATION

Dataset:
CIC-IDS2018

Training representation:
B1 undersampled

Features:
30

Classes:
14

Label encoding:
Fixed 14-class mapping.
String labels converted to integer IDs 0-13.

Optimization:
RandomizedSearchCV

Primary metric:
Macro F1

Cross-validation:
Stratified {CV_FOLDS}-fold

Randomized configurations:
{N_ITER}

Total model fits:
{N_ITER * CV_FOLDS}

Random state:
{RANDOM_STATE}

Validation dataset:
Used only after hyperparameter selection.

Test dataset:
NOT USED.

Feature selection:
NOT PERFORMED.

Additional balancing:
NOT PERFORMED.

SMOTE:
NOT USED.

Search time hours:
{search_time / 3600:.4f}

Best CV Macro F1:
{search.best_score_:.6f}

Validation Accuracy:
{accuracy:.6f}

Validation Balanced Accuracy:
{balanced_accuracy:.6f}

Validation Macro Precision:
{macro_precision:.6f}

Validation Macro Recall:
{macro_recall:.6f}

Validation Macro F1:
{macro_f1:.6f}

Validation Weighted F1:
{weighted_f1:.6f}
"""


with open(
    OUTPUT_DIR / "step23_xgb_optimization_protocol.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(protocol)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 78)
print("STEP 23 COMPLETED")
print("=" * 78)

print("Generated files:")

print(
    "  - step23_best_xgb_parameters.json"
)

print(
    "  - step23_xgb_randomized_search_results.csv"
)

print(
    "  - step23_xgb_validation_metrics.csv"
)

print(
    "  - step23_xgb_validation_per_class_metrics.csv"
)

print(
    "  - step23_xgb_validation_confusion_matrix.csv"
)

print(
    "  - step23_xgb_optimization_protocol.txt"
)

print(
    "  - step23_best_xgboost.joblib"
)

print("\nNEXT:")
print("STEP 24 - RF vs XGBoost comparative analysis")