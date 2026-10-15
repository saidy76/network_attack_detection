# =============================================================================
# STEP 21 - CONTROLLED RANDOM FOREST HYPERPARAMETER OPTIMIZATION
# =============================================================================
#
# Research Project:
# Behaviour Based Network Attack Detection System using Machine Learning
#
# Dataset:
# CIC-IDS2018
#
# Purpose:
# Evidence-based Random Forest hyperparameter optimization.
#
# IMPORTANT RESEARCH POLICY:
# - Uses the EXACT B1 undersampled training representation from Step 18.
# - Uses the EXACT selected-feature validation representation from Step 18.
# - No preprocessing is rebuilt.
# - No new train/validation split is created.
# - Validation data is NOT used during hyperparameter optimization.
# - Test data is STRICTLY NOT USED.
#
# Primary optimization metric:
# Macro F1
#
# Optimization:
# RandomizedSearchCV + StratifiedKFold
#
# Random state:
# 42
# =============================================================================

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(r"D:\network_attack_detection")

MODEL_DIR = PROJECT_ROOT / "data" / "modeling"
BALANCED_DIR = MODEL_DIR / "balanced"

TRAINING_FILE = (
    BALANCED_DIR /
    "CICIDS2018_train_B1_undersampled.csv"
)

VALIDATION_FILE = (
    MODEL_DIR /
    "CICIDS2018_validation_selected.csv"
)

EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

STEP21_DIR = (
    EXPERIMENT_DIR /
    "step21_rf_tuning"
)

STEP21_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. EXPERIMENT CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

# Reduced from 30 to 12 configurations because the dataset
# contains more than 1 million training samples.
N_ITER = 12

# Keep 3-fold stratified CV.
CV_FOLDS = 3

PRIMARY_SCORING = "f1_macro"

# Search itself runs sequentially.
N_JOBS_SEARCH = 1

# Each Random Forest uses all available CPU threads.
N_JOBS_RF = -1


# =============================================================================
# 3. CONTROLLED SEARCH SPACE
# =============================================================================
#
# The search space remains evidence-based but is intentionally smaller
# to make the experiment computationally feasible on the available system.
#
# =============================================================================

PARAM_DISTRIBUTIONS = {

    "n_estimators": [
        100,
        150,
        200,
    ],

    "max_depth": [
        None,
        20,
        30,
    ],

    "min_samples_split": [
        2,
        5,
        10,
    ],

    "min_samples_leaf": [
        1,
        2,
        5,
    ],

    "max_features": [
        "sqrt",
        "log2",
        0.5,
    ],

    "criterion": [
        "gini",
        "entropy",
    ],

    "class_weight": [
        None,
        "balanced",
        "balanced_subsample",
    ],

    "bootstrap": [
        True,
    ],

    "min_impurity_decrease": [
        0.0,
        1e-7,
        1e-6,
    ],
}


# =============================================================================
# 4. OUTPUT FILES
# =============================================================================

CV_RESULTS_FILE = (
    STEP21_DIR /
    "step21_rf_randomized_search_results.csv"
)

BEST_MODEL_FILE = (
    STEP21_DIR /
    "step21_best_random_forest.joblib"
)

VALIDATION_METRICS_FILE = (
    STEP21_DIR /
    "step21_rf_validation_metrics.csv"
)

PER_CLASS_FILE = (
    STEP21_DIR /
    "step21_rf_validation_per_class_metrics.csv"
)

CONFUSION_FILE = (
    STEP21_DIR /
    "step21_rf_validation_confusion_matrix.csv"
)

DIFFICULT_CLASSES_FILE = (
    STEP21_DIR /
    "step21_rf_difficult_classes.csv"
)

INFILTRATION_CONFUSION_FILE = (
    STEP21_DIR /
    "step21_infiltration_benign_confusion.csv"
)

BEST_PARAMS_FILE = (
    STEP21_DIR /
    "step21_best_rf_parameters.json"
)

PROTOCOL_FILE = (
    STEP21_DIR /
    "step21_rf_optimization_protocol.txt"
)


# =============================================================================
# 5. HEADER
# =============================================================================

print("=" * 80)
print("STEP 21 - CONTROLLED RANDOM FOREST HYPERPARAMETER OPTIMIZATION")
print("=" * 80)

print(f"Project root : {PROJECT_ROOT}")
print(f"Training file: {TRAINING_FILE}")
print(f"Validation   : {VALIDATION_FILE}")
print(f"Output dir   : {STEP21_DIR}")


# =============================================================================
# 6. FILE VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("CHECKING REQUIRED DATASETS")
print("=" * 80)

if not TRAINING_FILE.exists():
    raise FileNotFoundError(
        f"\nB1 training dataset not found:\n{TRAINING_FILE}"
    )

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"\nValidation dataset not found:\n{VALIDATION_FILE}"
    )

print("[OK] B1 undersampled training dataset found.")
print("[OK] Selected validation dataset found.")
print("[IMPORTANT] Test dataset will NOT be loaded.")


# =============================================================================
# 7. LOAD B1 TRAINING DATA
# =============================================================================

print("\n" + "=" * 80)
print("LOADING B1 UNDERSAMPLED TRAINING DATA")
print("=" * 80)

start = time.time()

train_df = pd.read_csv(
    TRAINING_FILE,
    low_memory=False
)

print(
    f"[OK] Training data loaded in "
    f"{time.time() - start:.2f} sec"
)

print(
    f"Training shape: {train_df.shape}"
)


# =============================================================================
# 8. LOAD VALIDATION DATA
# =============================================================================

print("\n" + "=" * 80)
print("LOADING VALIDATION DATA")
print("=" * 80)

start = time.time()

validation_df = pd.read_csv(
    VALIDATION_FILE,
    low_memory=False
)

print(
    f"[OK] Validation data loaded in "
    f"{time.time() - start:.2f} sec"
)

print(
    f"Validation shape: {validation_df.shape}"
)


# =============================================================================
# 9. BASIC DATA VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("DATASET INTEGRITY CHECK")
print("=" * 80)

if "Label" not in train_df.columns:
    raise ValueError(
        "Label column missing from B1 training dataset."
    )

if "Label" not in validation_df.columns:
    raise ValueError(
        "Label column missing from validation dataset."
    )


train_features = [
    c for c in train_df.columns
    if c != "Label"
]

validation_features = [
    c for c in validation_df.columns
    if c != "Label"
]


print(
    f"Training features   : {len(train_features)}"
)

print(
    f"Validation features : {len(validation_features)}"
)


if train_features != validation_features:

    missing_from_training = (
        set(validation_features)
        - set(train_features)
    )

    extra_in_training = (
        set(train_features)
        - set(validation_features)
    )

    raise ValueError(
        "\nFeature mismatch detected.\n"
        f"Missing from training: {missing_from_training}\n"
        f"Extra in training: {extra_in_training}"
    )


if len(train_features) != 30:

    raise ValueError(
        f"Expected exactly 30 selected features, "
        f"found {len(train_features)}."
    )


print(
    "[OK] Training and validation use the same 30 features."
)


# =============================================================================
# 10. MISSING / INFINITY CHECK
# =============================================================================

if (
    train_df[train_features]
    .isnull()
    .sum()
    .sum()
    > 0
):

    raise ValueError(
        "Missing values detected in training features."
    )


if (
    validation_df[validation_features]
    .isnull()
    .sum()
    .sum()
    > 0
):

    raise ValueError(
        "Missing values detected in validation features."
    )


if np.isinf(
    train_df[train_features]
    .to_numpy(dtype=np.float64)
).sum() > 0:

    raise ValueError(
        "Infinity values detected in training features."
    )


if np.isinf(
    validation_df[validation_features]
    .to_numpy(dtype=np.float64)
).sum() > 0:

    raise ValueError(
        "Infinity values detected in validation features."
    )


print("[OK] No missing values.")
print("[OK] No infinity values.")


# =============================================================================
# 11. CLASS LABEL CONSISTENCY
# =============================================================================

print("\n" + "=" * 80)
print("CLASS LABEL CONSISTENCY CHECK")
print("=" * 80)

train_classes = set(
    train_df["Label"].unique()
)

validation_classes = set(
    validation_df["Label"].unique()
)

print(
    f"Training classes   : {len(train_classes)}"
)

print(
    f"Validation classes : {len(validation_classes)}"
)


missing_validation_classes = (
    validation_classes
    - train_classes
)


if missing_validation_classes:

    raise ValueError(
        "Validation contains classes absent from training:\n"
        f"{missing_validation_classes}"
    )


print(
    "[OK] All validation classes are represented in training."
)


# =============================================================================
# 12. FEATURE / TARGET MATRICES
# =============================================================================

print("\n" + "=" * 80)
print("CREATING TRAINING / VALIDATION MATRICES")
print("=" * 80)


X_train = (
    train_df[train_features]
    .astype(np.float32)
    .to_numpy()
)

y_train = (
    train_df["Label"]
    .astype(str)
    .to_numpy()
)


X_validation = (
    validation_df[validation_features]
    .astype(np.float32)
    .to_numpy()
)

y_validation = (
    validation_df["Label"]
    .astype(str)
    .to_numpy()
)


print(
    f"X_train       : {X_train.shape}"
)

print(
    f"y_train       : {y_train.shape}"
)

print(
    f"X_validation  : {X_validation.shape}"
)

print(
    f"y_validation  : {y_validation.shape}"
)


# Release DataFrames

del train_df
del validation_df


# =============================================================================
# 13. CLASS DISTRIBUTION
# =============================================================================

print("\n" + "=" * 80)
print("B1 TRAINING CLASS DISTRIBUTION")
print("=" * 80)

train_distribution = (
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

print(
    train_distribution.to_string()
)


print("\n" + "=" * 80)
print("VALIDATION CLASS DISTRIBUTION")
print("=" * 80)

validation_distribution = (
    pd.Series(y_validation)
    .value_counts()
    .sort_index()
)

print(
    validation_distribution.to_string()
)


# =============================================================================
# 14. CROSS-VALIDATION DESIGN
# =============================================================================

print("\n" + "=" * 80)
print("STRATIFIED CROSS-VALIDATION DESIGN")
print("=" * 80)

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

print(f"CV folds      : {CV_FOLDS}")
print("Shuffle        : True")
print(f"Random state   : {RANDOM_STATE}")
print(f"Primary metric : {PRIMARY_SCORING}")


# =============================================================================
# 15. RANDOM FOREST BASE ESTIMATOR
# =============================================================================

rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS_RF
)


# =============================================================================
# 16. SEARCH SPACE REPORT
# =============================================================================

print("\n" + "=" * 80)
print("CONTROLLED RANDOM FOREST SEARCH SPACE")
print("=" * 80)

for parameter, values in PARAM_DISTRIBUTIONS.items():

    print(
        f"{parameter:28s}: {values}"
    )

print(
    f"\nRandom configurations : {N_ITER}"
)

print(
    f"CV folds              : {CV_FOLDS}"
)

print(
    f"Maximum model fits    : "
    f"{N_ITER * CV_FOLDS}"
)


# =============================================================================
# 17. RANDOMIZED SEARCH
# =============================================================================

print("\n" + "=" * 80)
print("STARTING RANDOMIZED SEARCH")
print("=" * 80)

print(
    "\nIMPORTANT:"
)

print(
    "Validation data is NOT used during hyperparameter optimization."
)

print(
    "Only B1 training data participates in cross-validation."
)

print(
    "\nCONTROLLED COMPUTATION:"
)

print(
    f"{N_ITER} configurations × {CV_FOLDS} folds "
    f"= {N_ITER * CV_FOLDS} model fits"
)

print(
    "Please allow the experiment to complete."
)


search_start = time.time()


search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=PARAM_DISTRIBUTIONS,
    n_iter=N_ITER,
    scoring=PRIMARY_SCORING,
    cv=cv,
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS_SEARCH,
    refit=True,
    return_train_score=True,
    error_score="raise"
)


search.fit(
    X_train,
    y_train
)


search_elapsed = (
    time.time()
    - search_start
)


# =============================================================================
# 18. SEARCH RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("RANDOMIZED SEARCH COMPLETED")
print("=" * 80)

print(
    f"Elapsed time: "
    f"{search_elapsed / 3600:.2f} hours"
)

print(
    f"\nBest CV Macro F1: "
    f"{search.best_score_:.6f}"
)

print("\nBest parameters:")

for key, value in search.best_params_.items():

    print(
        f"  {key}: {value}"
    )


# =============================================================================
# 19. SAVE CV RESULTS
# =============================================================================

cv_results = pd.DataFrame(
    search.cv_results_
)

cv_results = cv_results.sort_values(
    by="rank_test_score"
)

cv_results.to_csv(
    CV_RESULTS_FILE,
    index=False
)

print(
    f"\n[OK] CV results saved:\n"
    f"{CV_RESULTS_FILE}"
)


# =============================================================================
# 20. SAVE BEST MODEL
# =============================================================================

joblib.dump(
    search.best_estimator_,
    BEST_MODEL_FILE
)

print(
    f"[OK] Best RF model saved:\n"
    f"{BEST_MODEL_FILE}"
)


# =============================================================================
# 21. FINAL VALIDATION EVALUATION
# =============================================================================
#
# Validation is used ONLY now, after hyperparameter selection.
#
# =============================================================================

print("\n" + "=" * 80)
print("FINAL VALIDATION EVALUATION")
print("=" * 80)

prediction_start = time.time()

y_pred = (
    search.best_estimator_
    .predict(X_validation)
)

prediction_time = (
    time.time()
    - prediction_start
)

print(
    f"Prediction time: "
    f"{prediction_time:.2f} sec"
)


# =============================================================================
# 22. OVERALL METRICS
# =============================================================================

accuracy = accuracy_score(
    y_validation,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_validation,
    y_pred
)

macro_precision = precision_score(
    y_validation,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_validation,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_validation,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_validation,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\nVALIDATION RESULTS")
print("-" * 80)

print(
    f"Accuracy          : {accuracy:.6f}"
)

print(
    f"Balanced Accuracy : {balanced_accuracy:.6f}"
)

print(
    f"Macro Precision   : {macro_precision:.6f}"
)

print(
    f"Macro Recall      : {macro_recall:.6f}"
)

print(
    f"Macro F1          : {macro_f1:.6f}"
)

print(
    f"Weighted F1       : {weighted_f1:.6f}"
)


# =============================================================================
# 23. SAVE OVERALL METRICS
# =============================================================================

overall_metrics = pd.DataFrame(
    [
        {
            "Experiment":
                "Step21_Tuned_RandomForest",

            "Dataset":
                "CIC-IDS2018",

            "Training_Strategy":
                "B1_undersampled",

            "Accuracy":
                accuracy,

            "Balanced_Accuracy":
                balanced_accuracy,

            "Macro_Precision":
                macro_precision,

            "Macro_Recall":
                macro_recall,

            "Macro_F1":
                macro_f1,

            "Weighted_F1":
                weighted_f1,

            "Best_CV_Macro_F1":
                search.best_score_,

            "N_Iterations":
                N_ITER,

            "CV_Folds":
                CV_FOLDS,

            "Random_State":
                RANDOM_STATE,

            "Training_Samples":
                len(y_train),

            "Validation_Samples":
                len(y_validation),

            "Feature_Count":
                X_train.shape[1],

            "Search_Time_Hours":
                search_elapsed / 3600,
        }
    ]
)


overall_metrics.to_csv(
    VALIDATION_METRICS_FILE,
    index=False
)

print(
    f"\n[OK] Overall validation metrics saved:\n"
    f"{VALIDATION_METRICS_FILE}"
)


# =============================================================================
# 24. PER-CLASS METRICS
# =============================================================================

report_dict = classification_report(
    y_validation,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).T

report_df.to_csv(
    PER_CLASS_FILE
)

print(
    f"[OK] Per-class metrics saved:\n"
    f"{PER_CLASS_FILE}"
)


# =============================================================================
# 25. DIFFICULT CLASS ANALYSIS
# =============================================================================

class_analysis = (
    report_df
    .copy()
)

class_analysis = class_analysis[
    ~class_analysis.index.isin(
        [
            "accuracy",
            "macro avg",
            "weighted avg"
        ]
    )
].copy()


class_analysis = class_analysis.rename(
    columns={
        "precision": "Precision",
        "recall": "Recall",
        "f1-score": "F1",
        "support": "Support"
    }
)


class_analysis["Error_Rate"] = (
    1.0 -
    class_analysis["Recall"]
)


class_analysis = class_analysis.sort_values(
    by="F1",
    ascending=True
)


class_analysis.to_csv(
    DIFFICULT_CLASSES_FILE,
    index_label="Label"
)

print(
    f"[OK] Difficult-class analysis saved:\n"
    f"{DIFFICULT_CLASSES_FILE}"
)


# =============================================================================
# 26. CONFUSION MATRIX
# =============================================================================

class_labels = sorted(
    set(y_validation)
    |
    set(y_pred)
)

cm = confusion_matrix(
    y_validation,
    y_pred,
    labels=class_labels
)


cm_df = pd.DataFrame(
    cm,
    index=class_labels,
    columns=class_labels
)

cm_df.index.name = "True_Label"

cm_df.to_csv(
    CONFUSION_FILE
)

print(
    f"[OK] Confusion matrix saved:\n"
    f"{CONFUSION_FILE}"
)


# =============================================================================
# 27. INFILTRATION <-> BENIGN ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("PRIMARY DIFFICULT-CLASS ANALYSIS")
print("=" * 80)


if (
    "BENIGN" in class_labels
    and
    "Infiltration" in class_labels
):

    benign_index = (
        class_labels.index("BENIGN")
    )

    infiltration_index = (
        class_labels.index("Infiltration")
    )


    benign_to_infiltration = int(
        cm[
            benign_index,
            infiltration_index
        ]
    )


    infiltration_to_benign = int(
        cm[
            infiltration_index,
            benign_index
        ]
    )


    benign_support = int(
        cm[
            benign_index,
            :
        ].sum()
    )


    infiltration_support = int(
        cm[
            infiltration_index,
            :
        ].sum()
    )


    print(
        "BENIGN -> Infiltration:",
        benign_to_infiltration
    )

    print(
        "Infiltration -> BENIGN:",
        infiltration_to_benign
    )

    print(
        "BENIGN support:",
        benign_support
    )

    print(
        "Infiltration support:",
        infiltration_support
    )


    focus_df = pd.DataFrame(
        [
            {
                "True_Label":
                    "BENIGN",

                "Predicted_Label":
                    "Infiltration",

                "Misclassified_Count":
                    benign_to_infiltration,

                "True_Class_Support":
                    benign_support,

                "Error_Rate":
                    (
                        benign_to_infiltration /
                        benign_support
                    )
            },

            {
                "True_Label":
                    "Infiltration",

                "Predicted_Label":
                    "BENIGN",

                "Misclassified_Count":
                    infiltration_to_benign,

                "True_Class_Support":
                    infiltration_support,

                "Error_Rate":
                    (
                        infiltration_to_benign /
                        infiltration_support
                    )
            }
        ]
    )


    focus_df.to_csv(
        INFILTRATION_CONFUSION_FILE,
        index=False
    )


    print(
        f"[OK] Infiltration/BENIGN analysis saved:\n"
        f"{INFILTRATION_CONFUSION_FILE}"
    )


else:

    print(
        "[INFO] Both Infiltration and BENIGN "
        "were not found."
    )


# =============================================================================
# 28. SAVE BEST PARAMETERS
# =============================================================================

best_parameters = {

    "experiment":
        "Step21_Tuned_RandomForest",

    "dataset":
        "CIC-IDS2018",

    "training_strategy":
        "B1_undersampled",

    "optimization_method":
        "RandomizedSearchCV",

    "primary_metric":
        PRIMARY_SCORING,

    "best_cv_macro_f1":
        float(search.best_score_),

    "best_parameters":
        search.best_params_,

    "random_state":
        RANDOM_STATE,

    "n_iter":
        N_ITER,

    "cv_folds":
        CV_FOLDS,

    "training_samples":
        int(X_train.shape[0]),

    "validation_samples":
        int(X_validation.shape[0]),

    "feature_count":
        int(X_train.shape[1]),

    "search_time_hours":
        float(search_elapsed / 3600),

    "validation_metrics":
        {
            "Accuracy":
                float(accuracy),

            "Balanced_Accuracy":
                float(balanced_accuracy),

            "Macro_Precision":
                float(macro_precision),

            "Macro_Recall":
                float(macro_recall),

            "Macro_F1":
                float(macro_f1),

            "Weighted_F1":
                float(weighted_f1)
        }
}


with open(
    BEST_PARAMS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        best_parameters,
        f,
        indent=4,
        default=str
    )


print(
    f"[OK] Best parameters saved:\n"
    f"{BEST_PARAMS_FILE}"
)


# =============================================================================
# 29. RESEARCH PROTOCOL
# =============================================================================

protocol = f"""
STEP 21 - CONTROLLED RANDOM FOREST HYPERPARAMETER OPTIMIZATION

================================================================

Research Project:
Behaviour Based Network Attack Detection System using Machine Learning

Dataset:
CIC-IDS2018

Training representation:
B1 undersampled training configuration

Training file:
{TRAINING_FILE}

Validation file:
{VALIDATION_FILE}

Selected feature count:
{X_train.shape[1]}

Optimization method:
RandomizedSearchCV

Primary optimization metric:
Macro F1

Cross-validation:
Stratified {CV_FOLDS}-fold cross-validation

Randomized configurations:
{N_ITER}

Random state:
{RANDOM_STATE}

Search parallelism:
RandomizedSearchCV n_jobs={N_JOBS_SEARCH}

Random Forest parallelism:
RandomForestClassifier n_jobs={N_JOBS_RF}


IMPORTANT VALIDATION POLICY
---------------------------

The validation dataset was NOT used during hyperparameter optimization.

The validation dataset was evaluated only after the best configuration
was selected using cross-validation on the B1 training dataset.


TEST-SET POLICY
---------------

The held-out CIC-IDS2018 test set was NOT loaded or evaluated.


SEARCH SPACE
------------

{json.dumps(PARAM_DISTRIBUTIONS, indent=4, default=str)}


STEP 20A MOTIVATION
-------------------

Step 20A identified substantial class-level weaknesses, especially:

- Infiltration
- FTP-BruteForce
- DoS attacks-SlowHTTPTest
- SQL Injection
- Brute Force -Web

The dominant error pattern was:

BENIGN <-> Infiltration

Therefore Macro F1 was selected as the primary optimization objective
rather than Accuracy.


COMPUTATIONAL CONTROL
---------------------

The previous configuration required:

30 configurations x 3 folds = 90 model fits.

Because the B1 training dataset contains more than one million samples,
that configuration was computationally excessive for the available system.

This controlled experiment therefore uses:

12 configurations x 3 folds = 36 model fits.

The research protocol, dataset representation, feature representation,
validation isolation, and primary metric remain unchanged.


VALIDATION RESULTS
------------------

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


RESEARCH INTEGRITY
------------------

No additional preprocessing was introduced.

No new train/validation split was created.

No test-set information influenced hyperparameter selection.

No manual class-specific threshold manipulation was performed.

The optimized Random Forest is a validation-stage candidate only.

It must NOT yet be reported as the final held-out test result.

UNSW-NB15 remains reserved for external validation.

CICIoT2023 remains the primary IoT dataset in the overall
research architecture.


NEXT EXPERIMENT
---------------

Step 22 should compare the tuned Random Forest against the
Step 18/20 baseline and determine whether the observed improvement
is meaningful, especially for difficult classes.
"""


with open(
    PROTOCOL_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        protocol.strip()
    )


print(
    f"[OK] Research protocol saved:\n"
    f"{PROTOCOL_FILE}"
)


# =============================================================================
# 30. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("STEP 21 COMPLETED")
print("=" * 80)


print("\nBEST RANDOM FOREST PARAMETERS")
print("-" * 80)

for key, value in search.best_params_.items():

    print(
        f"{key:28s}: {value}"
    )


print("\nVALIDATION PERFORMANCE")
print("-" * 80)

print(
    f"Accuracy          : {accuracy:.6f}"
)

print(
    f"Balanced Accuracy : {balanced_accuracy:.6f}"
)

print(
    f"Macro Precision   : {macro_precision:.6f}"
)

print(
    f"Macro Recall      : {macro_recall:.6f}"
)

print(
    f"Macro F1          : {macro_f1:.6f}"
)

print(
    f"Weighted F1       : {weighted_f1:.6f}"
)


print("\nRESEARCH INTEGRITY")
print("-" * 80)

print(
    "[OK] Exact B1 training representation used."
)

print(
    "[OK] Exact Step 18 validation representation used."
)

print(
    "[OK] Stratified CV performed only on training data."
)

print(
    "[OK] Validation used only after model selection."
)

print(
    "[OK] Test set was NOT accessed."
)

print(
    "[OK] No preprocessing was rebuilt."
)


print("\nOUTPUT FILES")
print("-" * 80)

for file in sorted(
    STEP21_DIR.iterdir()
):

    if file.is_file():

        print(
            f" - {file}"
        )


print("\nNEXT:")
print(
    "Step 22 - Compare tuned Random Forest against "
    "the Step 18/20 baseline."
)

print("=" * 80)