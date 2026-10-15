from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib

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
# STEP 31 - PROBABILITY FUSION EVALUATION
# =============================================================================
#
# Research Project:
# Behaviour Based Network Attack Detection System using Machine Learning
#
# Purpose:
# Evaluate a heterogeneous probability-fusion ensemble consisting of:
#
#   1. Tuned Random Forest  - Step 21
#   2. Tuned XGBoost        - Step 23
#
# Fusion rule:
#
#   P_fused = 0.5 * P_RF + 0.5 * P_XGB
#
# Final prediction:
#
#   y_hat = argmax(P_fused)
#
# IMPORTANT RESEARCH POLICY:
# - Same CIC-IDS2018 B1 training representation
# - Same 30 selected features
# - Same 14 classes
# - Validation set is used ONLY for evaluation
# - No fusion-weight optimization is performed here
# - No test-set data is used
# - No additional class balancing
# - No threshold manipulation
# - No retraining
#
# This makes Step 31 a validation-stage ensemble evaluation.
# =============================================================================


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "CICIDS2018_validation_selected.csv"
)

RF_MODEL_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "step21_rf_tuning"
    / "step21_best_random_forest.joblib"
)

XGB_MODEL_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "step23_xgb_tuning"
    / "step23_best_xgboost.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step31_probability_fusion"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# CLASS DEFINITIONS
# =============================================================================

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


# =============================================================================
# FUSION CONFIGURATION
# =============================================================================

RF_WEIGHT = 0.5
XGB_WEIGHT = 0.5

if not np.isclose(
    RF_WEIGHT + XGB_WEIGHT,
    1.0
):
    raise ValueError(
        "Fusion weights must sum to 1.0."
    )


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

EXPECTED_FEATURE_COUNT = 30


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def fail(message):
    print()
    print("=" * 80)
    print("ERROR")
    print("=" * 80)
    print(message)
    raise SystemExit(1)


def normalize_columns(df):
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]
    return df


def check_probability_matrix(
    probabilities,
    model_name
):
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    if probabilities.ndim != 2:
        raise ValueError(
            f"{model_name} probability output must be 2-dimensional. "
            f"Found shape: {probabilities.shape}"
        )

    if probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"{model_name} probability matrix has "
            f"{probabilities.shape[1]} classes, expected "
            f"{len(CLASS_NAMES)}."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"{model_name} probability matrix contains "
            f"NaN or infinite values."
        )

    row_sums = probabilities.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-5
    ):
        max_error = np.max(
            np.abs(row_sums - 1.0)
        )

        raise ValueError(
            f"{model_name} probability rows do not sum to 1. "
            f"Maximum error: {max_error}"
        )

    if (probabilities < -1e-8).any():
        raise ValueError(
            f"{model_name} probability matrix contains "
            f"negative probabilities."
        )

    return probabilities


def align_rf_probabilities(
    rf_model,
    rf_probabilities
):
    """
    RF stores original string labels in model.classes_.

    Reorder RF probability columns into the fixed
    research class order defined by CLASS_NAMES.
    """

    rf_classes = [
        str(value)
        for value in rf_model.classes_
    ]

    if set(rf_classes) != set(CLASS_NAMES):
        raise ValueError(
            "RF model classes do not match the expected "
            "14-class label set."
        )

    column_indices = [
        rf_classes.index(class_name)
        for class_name in CLASS_NAMES
    ]

    return rf_probabilities[:, column_indices]


def align_xgb_probabilities(
    xgb_model,
    xgb_probabilities
):
    """
    XGBoost stores integer class IDs.

    Reorder probability columns into the fixed
    research class order 0-13.
    """

    xgb_classes = [
        int(value)
        for value in xgb_model.classes_
    ]

    expected_ids = list(
        range(len(CLASS_NAMES))
    )

    if sorted(xgb_classes) != expected_ids:
        raise ValueError(
            "XGBoost model classes do not match "
            "expected class IDs 0-13."
        )

    column_indices = [
        xgb_classes.index(class_id)
        for class_id in expected_ids
    ]

    return xgb_probabilities[:, column_indices]


def calculate_metrics(
    y_true,
    y_pred
):
    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),
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
        "Weighted_F1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
    }


def save_per_class_results(
    y_true,
    y_pred,
    output_path,
    model_name
):
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for class_id, class_name in enumerate(CLASS_NAMES):

        result = report[class_name]

        rows.append(
            {
                "Class_ID": class_id,
                "Label": class_name,
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1-score"],
                "Support": result["support"],
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print(f"{model_name} per-class results:")
    print(
        df.to_string(index=False)
    )

    return df


def save_confusion_matrix(
    y_true,
    y_pred,
    output_path
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES)))
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    cm_df.to_csv(
        output_path
    )

    return cm_df


# =============================================================================
# START
# =============================================================================

start_time = time.time()

print("=" * 80)
print("STEP 31 - PROBABILITY FUSION EVALUATION")
print("=" * 80)

print(
    "Research Project:\n"
    "Behaviour Based Network Attack Detection System using Machine Learning"
)

print()
print(f"Project root : {PROJECT_ROOT}")
print(f"Validation   : {VALIDATION_FILE}")
print(f"RF model     : {RF_MODEL_FILE}")
print(f"XGB model    : {XGB_MODEL_FILE}")
print(f"Output       : {OUTPUT_DIR}")


# =============================================================================
# INPUT CHECK
# =============================================================================

section("INPUT FILE CHECK")

required_files = [
    VALIDATION_FILE,
    RF_MODEL_FILE,
    XGB_MODEL_FILE,
]

for path in required_files:

    if not path.exists():
        fail(
            f"Required file not found:\n{path}"
        )

    print(f"FOUND: {path}")


# =============================================================================
# LOAD VALIDATION DATA
# =============================================================================

section("LOADING VALIDATION DATA")

val_df = pd.read_csv(
    VALIDATION_FILE,
    low_memory=False
)

val_df = normalize_columns(
    val_df
)

print(
    f"Validation shape: {val_df.shape}"
)

if "Label" not in val_df.columns:
    fail(
        "Validation dataset does not contain "
        "'Label' column."
    )


# =============================================================================
# FEATURE VERIFICATION
# =============================================================================

section("FEATURE VERIFICATION")

FEATURE_COLUMNS = [
    column
    for column in val_df.columns
    if column != "Label"
]

print(
    f"Feature count: {len(FEATURE_COLUMNS)}"
)

if len(FEATURE_COLUMNS) != EXPECTED_FEATURE_COUNT:

    fail(
        f"Expected {EXPECTED_FEATURE_COUNT} features, "
        f"found {len(FEATURE_COLUMNS)}."
    )


# =============================================================================
# LABEL ENCODING
# =============================================================================

section("LABEL ENCODING")

unknown_labels = sorted(
    set(
        val_df["Label"]
        .astype(str)
        .str.strip()
    )
    - set(CLASS_TO_ID.keys())
)

if unknown_labels:

    fail(
        "Unknown labels found in validation dataset:\n"
        + "\n".join(
            f"  - {label}"
            for label in unknown_labels
        )
    )


y_true = (
    val_df["Label"]
    .astype(str)
    .str.strip()
    .map(CLASS_TO_ID)
    .astype(np.int32)
    .to_numpy()
)

X_val = val_df[
    FEATURE_COLUMNS
]


print(
    f"Validation samples : {len(X_val):,}"
)

print(
    f"Classes present    : "
    f"{sorted(np.unique(y_true).tolist())}"
)


# =============================================================================
# LOAD MODELS
# =============================================================================

section("LOADING TRAINED MODELS")

print("Loading Random Forest...")

rf_model = joblib.load(
    RF_MODEL_FILE
)

print(
    f"RF model type      : {type(rf_model)}"
)

print(
    f"RF classes         : {rf_model.classes_}"
)

print(
    f"RF n_classes       : {rf_model.n_classes_}"
)

print(
    f"RF n_features      : {rf_model.n_features_in_}"
)


print()
print("Loading XGBoost...")

xgb_model = joblib.load(
    XGB_MODEL_FILE
)

print(
    f"XGB model type     : {type(xgb_model)}"
)

print(
    f"XGB classes        : {xgb_model.classes_}"
)

print(
    f"XGB n_classes      : {xgb_model.n_classes_}"
)

print(
    f"XGB n_features     : {xgb_model.n_features_in_}"
)


# =============================================================================
# MODEL STRUCTURAL CHECK
# =============================================================================

section("MODEL STRUCTURAL VALIDATION")

if rf_model.n_classes_ != len(CLASS_NAMES):

    fail(
        "Random Forest does not contain "
        "the expected 14 classes."
    )


if xgb_model.n_classes_ != len(CLASS_NAMES):

    fail(
        "XGBoost does not contain "
        "the expected 14 classes."
    )


if rf_model.n_features_in_ != EXPECTED_FEATURE_COUNT:

    fail(
        "Random Forest feature count mismatch."
    )


if xgb_model.n_features_in_ != EXPECTED_FEATURE_COUNT:

    fail(
        "XGBoost feature count mismatch."
    )


print(
    "RF class count       : PASS"
)

print(
    "XGB class count      : PASS"
)

print(
    "RF feature count     : PASS"
)

print(
    "XGB feature count    : PASS"
)


# =============================================================================
# MODEL FEATURE ORDER CHECK
# =============================================================================

section("MODEL FEATURE ORDER CHECK")

# Both models were trained using the 30-column
# training representation. The validation file must
# therefore use the same feature names/order.

if hasattr(rf_model, "feature_names_in_"):

    rf_features = [
        str(x)
        for x in rf_model.feature_names_in_
    ]

    if rf_features != FEATURE_COLUMNS:

        fail(
            "Validation feature order does not match "
            "Random Forest feature order."
        )

    print(
        "Random Forest feature order: PASS"
    )

else:

    print(
        "Random Forest feature_names_in_ unavailable; "
        "feature count validated."
    )


if hasattr(xgb_model, "feature_names_in_"):

    xgb_features = [
        str(x)
        for x in xgb_model.feature_names_in_
    ]

    if xgb_features != FEATURE_COLUMNS:

        fail(
            "Validation feature order does not match "
            "XGBoost feature order."
        )

    print(
        "XGBoost feature order: PASS"
    )

else:

    print(
        "XGBoost feature_names_in_ unavailable; "
        "feature count validated."
    )


# =============================================================================
# RANDOM FOREST PROBABILITIES
# =============================================================================

section("GENERATING RANDOM FOREST PROBABILITIES")

rf_probability_raw = rf_model.predict_proba(
    X_val
)

print(
    f"Raw RF probability shape: "
    f"{rf_probability_raw.shape}"
)

rf_probability_raw = check_probability_matrix(
    rf_probability_raw,
    "Random Forest"
)

rf_probability = align_rf_probabilities(
    rf_model,
    rf_probability_raw
)

print(
    f"Aligned RF probability shape: "
    f"{rf_probability.shape}"
)


# =============================================================================
# XGBOOST PROBABILITIES
# =============================================================================

section("GENERATING XGBOOST PROBABILITIES")

xgb_probability_raw = xgb_model.predict_proba(
    X_val
)

print(
    f"Raw XGB probability shape: "
    f"{xgb_probability_raw.shape}"
)

xgb_probability_raw = check_probability_matrix(
    xgb_probability_raw,
    "XGBoost"
)

xgb_probability = align_xgb_probabilities(
    xgb_model,
    xgb_probability_raw
)

print(
    f"Aligned XGB probability shape: "
    f"{xgb_probability.shape}"
)


# =============================================================================
# PROBABILITY ALIGNMENT CHECK
# =============================================================================

section("PROBABILITY ALIGNMENT CHECK")

if rf_probability.shape != xgb_probability.shape:

    fail(
        "RF and XGBoost probability matrices "
        "have different shapes."
    )


if rf_probability.shape[0] != len(y_true):

    fail(
        "RF probability row count does not match "
        "validation sample count."
    )


if xgb_probability.shape[0] != len(y_true):

    fail(
        "XGBoost probability row count does not match "
        "validation sample count."
    )


print(
    "Probability row alignment: PASS"
)

print(
    f"Samples: {rf_probability.shape[0]:,}"
)

print(
    f"Classes: {rf_probability.shape[1]}"
)


# =============================================================================
# INDIVIDUAL MODEL PREDICTIONS
# =============================================================================

section("GENERATING INDIVIDUAL MODEL PREDICTIONS")

rf_pred = np.argmax(
    rf_probability,
    axis=1
).astype(np.int32)

xgb_pred = np.argmax(
    xgb_probability,
    axis=1
).astype(np.int32)


# =============================================================================
# PROBABILITY FUSION
# =============================================================================

section("EQUAL-WEIGHT PROBABILITY FUSION")

print(
    "Fusion method:"
)

print(
    "P_fused = "
    f"{RF_WEIGHT:.1f} * P_RF + "
    f"{XGB_WEIGHT:.1f} * P_XGB"
)

print()
print(
    f"Random Forest weight : {RF_WEIGHT:.2f}"
)

print(
    f"XGBoost weight       : {XGB_WEIGHT:.2f}"
)

print(
    "Fusion weight tuning : NOT PERFORMED"
)

print(
    "Validation used to tune weights: NO"
)


fused_probability = (
    RF_WEIGHT * rf_probability
    +
    XGB_WEIGHT * xgb_probability
)


# =============================================================================
# FUSED PROBABILITY VALIDATION
# =============================================================================

fused_probability = check_probability_matrix(
    fused_probability,
    "Fused model"
)


# =============================================================================
# FINAL FUSION PREDICTION
# =============================================================================

fused_pred = np.argmax(
    fused_probability,
    axis=1
).astype(np.int32)


print()
print(
    f"Fused probability shape: "
    f"{fused_probability.shape}"
)

print(
    "Fused prediction generated successfully."
)


# =============================================================================
# CALCULATE MODEL METRICS
# =============================================================================

section("OVERALL VALIDATION METRICS")

rf_metrics = calculate_metrics(
    y_true,
    rf_pred
)

xgb_metrics = calculate_metrics(
    y_true,
    xgb_pred
)

fusion_metrics = calculate_metrics(
    y_true,
    fused_pred
)


comparison_df = pd.DataFrame(
    [
        {
            "Model": "Tuned Random Forest",
            **rf_metrics,
        },
        {
            "Model": "Tuned XGBoost",
            **xgb_metrics,
        },
        {
            "Model": "Equal-Weight Probability Fusion",
            **fusion_metrics,
        },
    ]
)


print(
    comparison_df.to_string(
        index=False
    )
)


# =============================================================================
# PRIMARY METRIC IMPROVEMENT
# =============================================================================

section("FUSION IMPROVEMENT ANALYSIS")

rf_macro_f1 = rf_metrics["Macro_F1"]
xgb_macro_f1 = xgb_metrics["Macro_F1"]
fusion_macro_f1 = fusion_metrics["Macro_F1"]

fusion_vs_rf = (
    fusion_macro_f1
    - rf_macro_f1
)

fusion_vs_xgb = (
    fusion_macro_f1
    - xgb_macro_f1
)


print(
    f"RF Macro F1      : {rf_macro_f1:.6f}"
)

print(
    f"XGB Macro F1     : {xgb_macro_f1:.6f}"
)

print(
    f"Fusion Macro F1  : {fusion_macro_f1:.6f}"
)

print()
print(
    f"Fusion - RF      : {fusion_vs_rf:+.6f}"
)

print(
    f"Fusion - XGB     : {fusion_vs_xgb:+.6f}"
)


if fusion_macro_f1 > rf_macro_f1:

    primary_result = (
        "Fusion improves over Random Forest"
    )

elif fusion_macro_f1 < rf_macro_f1:

    primary_result = (
        "Fusion does not improve over Random Forest"
    )

else:

    primary_result = (
        "Fusion ties Random Forest"
    )


print()
print(
    f"Primary result: {primary_result}"
)


# =============================================================================
# PER-CLASS RESULTS
# =============================================================================

section("PER-CLASS VALIDATION ANALYSIS")

rf_per_class = save_per_class_results(
    y_true,
    rf_pred,
    OUTPUT_DIR
    / "step31_rf_per_class.csv",
    "Random Forest"
)

xgb_per_class = save_per_class_results(
    y_true,
    xgb_pred,
    OUTPUT_DIR
    / "step31_xgb_per_class.csv",
    "XGBoost"
)

fusion_per_class = save_per_class_results(
    y_true,
    fused_pred,
    OUTPUT_DIR
    / "step31_fusion_per_class.csv",
    "Probability Fusion"
)


# =============================================================================
# PER-CLASS COMPARISON
# =============================================================================

section("PER-CLASS F1 COMPARISON")

per_class_comparison = pd.DataFrame(
    {
        "Class_ID": range(len(CLASS_NAMES)),
        "Label": CLASS_NAMES,
        "RF_F1": rf_per_class["F1"].to_numpy(),
        "XGB_F1": xgb_per_class["F1"].to_numpy(),
        "Fusion_F1": fusion_per_class["F1"].to_numpy(),
        "Support": fusion_per_class["Support"].to_numpy(),
    }
)


per_class_comparison["Fusion_vs_RF_F1"] = (
    per_class_comparison["Fusion_F1"]
    - per_class_comparison["RF_F1"]
)

per_class_comparison["Fusion_vs_XGB_F1"] = (
    per_class_comparison["Fusion_F1"]
    - per_class_comparison["XGB_F1"]
)


def determine_winner(row):

    values = {
        "Random Forest": row["RF_F1"],
        "XGBoost": row["XGB_F1"],
        "Fusion": row["Fusion_F1"],
    }

    max_value = max(
        values.values()
    )

    winners = [
        name
        for name, value in values.items()
        if np.isclose(
            value,
            max_value,
            atol=1e-12
        )
    ]

    if len(winners) == 1:
        return winners[0]

    return "Tie"


per_class_comparison["Per_Class_Winner"] = (
    per_class_comparison.apply(
        determine_winner,
        axis=1
    )
)


print(
    per_class_comparison.to_string(
        index=False
    )
)


# =============================================================================
# FUSION CLASS WIN COUNT
# =============================================================================

section("PER-CLASS WIN COUNT")

winner_counts = (
    per_class_comparison[
        "Per_Class_Winner"
    ]
    .value_counts()
    .to_dict()
)

rf_class_wins = int(
    winner_counts.get(
        "Random Forest",
        0
    )
)

xgb_class_wins = int(
    winner_counts.get(
        "XGBoost",
        0
    )
)

fusion_class_wins = int(
    winner_counts.get(
        "Fusion",
        0
    )
)

tie_count = int(
    winner_counts.get(
        "Tie",
        0
    )
)


print(
    f"Random Forest class wins : {rf_class_wins}"
)

print(
    f"XGBoost class wins       : {xgb_class_wins}"
)

print(
    f"Fusion class wins        : {fusion_class_wins}"
)

print(
    f"Ties                     : {tie_count}"
)


# =============================================================================
# CONFUSION MATRICES
# =============================================================================

section("CONFUSION MATRICES")

rf_cm = save_confusion_matrix(
    y_true,
    rf_pred,
    OUTPUT_DIR
    / "step31_rf_confusion_matrix.csv"
)

xgb_cm = save_confusion_matrix(
    y_true,
    xgb_pred,
    OUTPUT_DIR
    / "step31_xgb_confusion_matrix.csv"
)

fusion_cm = save_confusion_matrix(
    y_true,
    fused_pred,
    OUTPUT_DIR
    / "step31_fusion_confusion_matrix.csv"
)


print(
    "\nFusion confusion matrix:"
)

print(
    fusion_cm
)


# =============================================================================
# FUSION PROBABILITY STATISTICS
# =============================================================================

section("FUSION PROBABILITY STATISTICS")

fusion_max_probability = (
    np.max(
        fused_probability,
        axis=1
    )
)

fusion_entropy = -np.sum(
    np.where(
        fused_probability > 0,
        fused_probability
        * np.log(
            fused_probability
        ),
        0.0
    ),
    axis=1
)


probability_summary = {
    "mean_max_probability": float(
        np.mean(
            fusion_max_probability
        )
    ),
    "median_max_probability": float(
        np.median(
            fusion_max_probability
        )
    ),
    "minimum_max_probability": float(
        np.min(
            fusion_max_probability
        )
    ),
    "maximum_max_probability": float(
        np.max(
            fusion_max_probability
        )
    ),
    "mean_entropy": float(
        np.mean(
            fusion_entropy
        )
    ),
}


print(
    f"Mean max probability   : "
    f"{probability_summary['mean_max_probability']:.6f}"
)

print(
    f"Median max probability : "
    f"{probability_summary['median_max_probability']:.6f}"
)

print(
    f"Minimum max probability: "
    f"{probability_summary['minimum_max_probability']:.6f}"
)

print(
    f"Maximum max probability: "
    f"{probability_summary['maximum_max_probability']:.6f}"
)

print(
    f"Mean entropy           : "
    f"{probability_summary['mean_entropy']:.6f}"
)


# =============================================================================
# SAVE OVERALL COMPARISON
# =============================================================================

section("SAVING RESULTS")

comparison_df.to_csv(
    OUTPUT_DIR
    / "step31_model_comparison_metrics.csv",
    index=False
)


per_class_comparison.to_csv(
    OUTPUT_DIR
    / "step31_per_class_fusion_comparison.csv",
    index=False
)


# =============================================================================
# SAVE FUSED PROBABILITIES
# =============================================================================
#
# This file allows later analysis without rerunning
# the models. It contains the fused probability for
# each of the 14 classes.
#
# =============================================================================

fusion_probability_df = pd.DataFrame(
    fused_probability,
    columns=[
        f"Prob_{class_name}"
        for class_name in CLASS_NAMES
    ]
)

fusion_probability_df.insert(
    0,
    "True_Label_ID",
    y_true
)

fusion_probability_df.insert(
    1,
    "True_Label",
    [
        ID_TO_CLASS[int(value)]
        for value in y_true
    ]
)

fusion_probability_df.insert(
    2,
    "Predicted_Label_ID",
    fused_pred
)

fusion_probability_df.insert(
    3,
    "Predicted_Label",
    [
        ID_TO_CLASS[int(value)]
        for value in fused_pred
    ]
)

fusion_probability_df.to_csv(
    OUTPUT_DIR
    / "step31_fused_probabilities.csv",
    index=False
)


# =============================================================================
# SAVE SUMMARY JSON
# =============================================================================

runtime_seconds = (
    time.time()
    - start_time
)


summary = {
    "step": "31",
    "experiment": "Equal_Weight_Probability_Fusion",
    "dataset": "CIC-IDS2018",
    "training_representation": "B1_undersampled",
    "evaluation_dataset": (
        "CICIDS2018_validation_selected.csv"
    ),
    "feature_count": EXPECTED_FEATURE_COUNT,
    "class_count": len(CLASS_NAMES),
    "random_state": RANDOM_STATE,

    "models": {
        "random_forest": (
            "Step21 tuned Random Forest"
        ),
        "xgboost": (
            "Step23 tuned XGBoost"
        ),
    },

    "fusion_method": (
        "Weighted arithmetic mean of class "
        "probabilities"
    ),

    "fusion_weights": {
        "Random_Forest": RF_WEIGHT,
        "XGBoost": XGB_WEIGHT,
    },

    "weight_optimization": False,

    "validation_used_for_weight_tuning": False,

    "test_set_used": False,

    "additional_training": False,

    "additional_balancing": False,

    "smote_used": False,

    "threshold_manipulation": False,

    "validation_samples": int(
        len(y_true)
    ),

    "metrics": {
        "Random_Forest": {
            key: float(value)
            for key, value in rf_metrics.items()
        },
        "XGBoost": {
            key: float(value)
            for key, value in xgb_metrics.items()
        },
        "Probability_Fusion": {
            key: float(value)
            for key, value in fusion_metrics.items()
        },
    },

    "macro_f1_comparison": {
        "Random_Forest": float(
            rf_macro_f1
        ),
        "XGBoost": float(
            xgb_macro_f1
        ),
        "Probability_Fusion": float(
            fusion_macro_f1
        ),
        "Fusion_minus_RF": float(
            fusion_vs_rf
        ),
        "Fusion_minus_XGB": float(
            fusion_vs_xgb
        ),
    },

    "per_class_wins": {
        "Random_Forest": rf_class_wins,
        "XGBoost": xgb_class_wins,
        "Probability_Fusion": fusion_class_wins,
        "Tie": tie_count,
    },

    "probability_statistics": probability_summary,

    "primary_result": primary_result,

    "runtime_seconds": float(
        runtime_seconds
    ),
}


with open(
    OUTPUT_DIR
    / "step31_probability_fusion_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# =============================================================================
# SAVE RESEARCH PROTOCOL
# =============================================================================

protocol = f"""
STEP 31 - PROBABILITY FUSION EVALUATION
========================================

Research Project:
Behaviour Based Network Attack Detection System using Machine Learning

Dataset:
CIC-IDS2018

Training representation:
B1 undersampled

Evaluation dataset:
CICIDS2018_validation_selected.csv

Feature count:
{EXPECTED_FEATURE_COUNT}

Class count:
{len(CLASS_NAMES)}

Models:
1. Step 21 Tuned Random Forest
2. Step 23 Tuned XGBoost

Fusion:
Equal-weight probability fusion

Formula:
P_fused = {RF_WEIGHT:.2f} * P_RF + {XGB_WEIGHT:.2f} * P_XGB

Prediction:
argmax(P_fused)

Fusion weight optimization:
NOT PERFORMED

Reason:
To avoid using the validation set for ensemble-weight
selection before establishing an unbiased validation-stage
ensemble result.

Validation data:
Used only for final evaluation of the already-defined
equal-weight fusion.

Test dataset:
NOT USED

Additional training:
NOT PERFORMED

Additional class balancing:
NOT PERFORMED

SMOTE:
NOT USED

Threshold manipulation:
NOT PERFORMED

Primary metric:
Macro F1

VALIDATION RESULTS
------------------

Random Forest Macro F1:
{rf_macro_f1:.6f}

XGBoost Macro F1:
{xgb_macro_f1:.6f}

Probability Fusion Macro F1:
{fusion_macro_f1:.6f}

Fusion minus Random Forest:
{fusion_vs_rf:+.6f}

Fusion minus XGBoost:
{fusion_vs_xgb:+.6f}

Random Forest class wins:
{rf_class_wins}

XGBoost class wins:
{xgb_class_wins}

Probability Fusion class wins:
{fusion_class_wins}

Ties:
{tie_count}

PRIMARY RESULT
--------------

{primary_result}

RESEARCH INTEGRITY
------------------

The Step 21 and Step 23 trained models were reused.

No model was retrained.

No validation information was used to optimize
fusion weights.

No test-set information was accessed.

No additional preprocessing was introduced.

No class-specific threshold manipulation was performed.

This experiment is a validation-stage ensemble evaluation.

The held-out CIC-IDS2018 test set remains reserved
for the final evaluation stage.

UNSW-NB15 remains reserved for external validation.

CICIoT2023 remains the primary IoT dataset in the
overall research architecture.

Runtime seconds:
{runtime_seconds:.3f}
"""


with open(
    OUTPUT_DIR
    / "step31_probability_fusion_protocol.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        protocol.strip()
        + "\n"
    )


# =============================================================================
# FINAL OUTPUT
# =============================================================================

section("STEP 31 COMPLETED")

print(
    f"Output directory:\n{OUTPUT_DIR}"
)

print()
print(
    "Generated files:"
)

for path in sorted(
    OUTPUT_DIR.iterdir()
):

    if path.is_file():

        print(
            f"  - {path.name}"
        )


print()
print(
    "=" * 80
)

print(
    "FINAL VALIDATION SUMMARY"
)

print(
    "=" * 80
)

print(
    f"RF Macro F1      : {rf_macro_f1:.6f}"
)

print(
    f"XGB Macro F1     : {xgb_macro_f1:.6f}"
)

print(
    f"Fusion Macro F1  : {fusion_macro_f1:.6f}"
)

print(
    f"Fusion vs RF     : {fusion_vs_rf:+.6f}"
)

print(
    f"Fusion vs XGB    : {fusion_vs_xgb:+.6f}"
)

print(
    f"Primary result   : {primary_result}"
)

print()
print(
    f"Runtime          : "
    f"{runtime_seconds / 60:.2f} minutes"
)

print()
print(
    "STEP 31 FINISHED SUCCESSFULLY."
)

print(
    "No test-set data was used."
)

print(
    "No fusion-weight optimization was performed."
)

print(
    "The next decision should be based on the observed "
    "validation Macro F1 and difficult-class behaviour."
)

print(
    "=" * 80
)