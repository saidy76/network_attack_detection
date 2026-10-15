# =============================================================================
# STEP 33 - XGBOOST DIFFICULT-CLASS & ERROR ANALYSIS
# =============================================================================
#
# Research Project:
# Behaviour Based Network Attack Detection System using Machine Learning
#
# Purpose:
#   Perform detailed validation-error analysis of the already-trained
#   Step 23 XGBoost model.
#
# Important methodological rules:
#   - No additional model training
#   - No hyperparameter optimization
#   - Validation set used only for evaluation
#   - Test set is NOT used
#   - No leakage
#
# =============================================================================

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "CICIDS2018_validation_selected.csv"
)

XGB_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "step23_xgb_tuning"
    / "step23_best_xgboost.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step33_xgb_error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

EXPECTED_N_FEATURES = 30
EXPECTED_N_CLASSES = 14


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
    label: idx
    for idx, label in enumerate(CLASS_NAMES)
}

ID_TO_CLASS = {
    idx: label
    for idx, label in enumerate(CLASS_NAMES)
}


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("STEP 33 - XGBOOST DIFFICULT-CLASS & ERROR ANALYSIS")
print("=" * 80)
print("Research Project:")
print("Behaviour Based Network Attack Detection System using Machine Learning")
print()
print(f"Project root : {PROJECT_ROOT}")
print(f"Validation   : {VALIDATION_PATH}")
print(f"XGB model    : {XGB_MODEL_PATH}")
print(f"Output       : {OUTPUT_DIR}")
print()


# =============================================================================
# INPUT FILE CHECK
# =============================================================================

print("=" * 80)
print("INPUT FILE CHECK")
print("=" * 80)

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_PATH}"
    )

if not XGB_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"XGBoost model not found:\n{XGB_MODEL_PATH}"
    )

print(f"FOUND: {VALIDATION_PATH}")
print(f"FOUND: {XGB_MODEL_PATH}")
print()


# =============================================================================
# LOAD VALIDATION DATA
# =============================================================================

print("=" * 80)
print("LOADING VALIDATION DATA")
print("=" * 80)

df = pd.read_csv(VALIDATION_PATH)

print(f"Validation shape: {df.shape}")
print()


# =============================================================================
# LABEL COLUMN DETECTION
# =============================================================================

possible_label_columns = [
    "Label",
    "label",
    "Attack",
    "attack",
    "Class",
    "class",
]

label_column = None

for candidate in possible_label_columns:
    if candidate in df.columns:
        label_column = candidate
        break

if label_column is None:
    raise ValueError(
        "Could not identify the validation label column."
    )

print("=" * 80)
print("LABEL COLUMN")
print("=" * 80)
print(f"Label column: {label_column}")
print()


# =============================================================================
# LOAD XGBOOST MODEL
# =============================================================================

print("=" * 80)
print("LOADING XGBOOST MODEL")
print("=" * 80)

model = joblib.load(XGB_MODEL_PATH)

print(f"Model type       : {type(model)}")
print(f"Model classes    : {model.classes_}")
print(f"Model n_classes  : {model.n_classes_}")
print(f"Model n_features : {model.n_features_in_}")
print()


# =============================================================================
# MODEL STRUCTURAL VALIDATION
# =============================================================================

print("=" * 80)
print("MODEL STRUCTURAL VALIDATION")
print("=" * 80)

if model.n_classes_ != EXPECTED_N_CLASSES:
    raise ValueError(
        f"Expected {EXPECTED_N_CLASSES} classes, "
        f"but model contains {model.n_classes_}."
    )

if model.n_features_in_ != EXPECTED_N_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_N_FEATURES} features, "
        f"but model expects {model.n_features_in_}."
    )

print("XGB class count  : PASS")
print("XGB feature count: PASS")
print()


# =============================================================================
# FEATURE SELECTION
# =============================================================================
#
# IMPORTANT:
# Step 23 XGBoost was trained with the 30 selected features.
#
# In some saved XGBoost models feature_names_in_ is available; in others,
# the model was fitted from a NumPy array and therefore does not retain
# feature names.
#
# Therefore:
#   1. Prefer model.feature_names_in_ when available.
#   2. Otherwise reconstruct the feature list from the validation CSV by
#      excluding the label column and verifying exactly 30 columns.
#
# This avoids incorrectly assuming arbitrary feature names.
# =============================================================================

print("=" * 80)
print("FEATURE VERIFICATION")
print("=" * 80)

if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)

    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Model feature names are unavailable in validation data:\n"
            + "\n".join(missing_features)
        )

    feature_columns = model_features

    print("Feature source    : XGBoost feature_names_in_")
    print("Feature order     : VERIFIED")

else:
    # The model was fitted without pandas feature names.
    # Use all validation columns except Label.
    feature_columns = [
        col
        for col in df.columns
        if col != label_column
    ]

    if len(feature_columns) != EXPECTED_N_FEATURES:
        raise ValueError(
            f"Validation dataset contains {len(feature_columns)} "
            f"non-label columns, but XGBoost expects "
            f"{EXPECTED_N_FEATURES} features."
        )

    print(
        "Feature source    : validation CSV "
        "(model feature_names_in_ unavailable)"
    )

    print(
        "Feature count     : PASS"
    )

print(f"Feature count     : {len(feature_columns)}")
print()


# =============================================================================
# PREPARE X AND Y
# =============================================================================

X = df[feature_columns]
y_raw = df[label_column]


# =============================================================================
# LABEL ENCODING
# =============================================================================
#
# Validation labels in this project are already encoded as integer IDs
# 0..13. However, the code also safely handles string labels.
# =============================================================================

print("=" * 80)
print("LABEL ENCODING")
print("=" * 80)

if pd.api.types.is_numeric_dtype(y_raw):

    y = y_raw.astype(int).to_numpy()

else:

    unknown_labels = sorted(
        set(y_raw.astype(str).unique())
        - set(CLASS_TO_ID.keys())
    )

    if unknown_labels:
        raise ValueError(
            "Unknown validation labels found:\n"
            + "\n".join(map(str, unknown_labels))
        )

    y = (
        y_raw.astype(str)
        .map(CLASS_TO_ID)
        .astype(int)
        .to_numpy()
    )

print(f"Validation samples : {len(y):,}")
print(
    f"Classes present    : "
    f"{sorted(np.unique(y).tolist())}"
)
print()


# =============================================================================
# FEATURE NUMERIC VALIDATION
# =============================================================================

if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):

    non_numeric = [
        col
        for col in X.columns
        if not pd.api.types.is_numeric_dtype(X[col])
    ]

    raise ValueError(
        "Non-numeric features detected:\n"
        + "\n".join(non_numeric)
    )


# =============================================================================
# GENERATE XGBOOST VALIDATION PREDICTIONS
# =============================================================================

print("=" * 80)
print("GENERATING XGBOOST VALIDATION PREDICTIONS")
print("=" * 80)

prediction_start = time.time()

X_values = X.to_numpy()

y_pred = model.predict(X_values)

prediction_runtime = time.time() - prediction_start

# XGBoost sometimes returns float class IDs depending on configuration.
y_pred = np.asarray(y_pred).astype(int)

print(f"Prediction shape: {y_pred.shape}")
print(
    f"Prediction runtime: "
    f"{prediction_runtime:.2f} seconds"
)
print()


# =============================================================================
# PREDICTION STRUCTURAL VALIDATION
# =============================================================================

if len(y_pred) != len(y):
    raise ValueError(
        "Prediction count does not match validation sample count."
    )

if not np.all(np.isin(y_pred, np.arange(EXPECTED_N_CLASSES))):
    raise ValueError(
        "Predictions contain class IDs outside expected range 0-13."
    )


# =============================================================================
# OVERALL METRICS
# =============================================================================

print("=" * 80)
print("OVERALL XGBOOST VALIDATION METRICS")
print("=" * 80)

accuracy = accuracy_score(y, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y,
    y_pred
)

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        y,
        y_pred,
        labels=np.arange(EXPECTED_N_CLASSES),
        average="macro",
        zero_division=0,
    )
)

_, _, f1_weighted, _ = (
    precision_recall_fscore_support(
        y,
        y_pred,
        labels=np.arange(EXPECTED_N_CLASSES),
        average="weighted",
        zero_division=0,
    )
)

print(f"Accuracy          : {accuracy:.6f}")
print(f"Balanced Accuracy : {balanced_accuracy:.6f}")
print(f"Macro Precision   : {precision_macro:.6f}")
print(f"Macro Recall      : {recall_macro:.6f}")
print(f"Macro F1          : {f1_macro:.6f}")
print(f"Weighted F1       : {f1_weighted:.6f}")
print()


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

print("=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

cm = confusion_matrix(
    y,
    y_pred,
    labels=np.arange(EXPECTED_N_CLASSES)
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

print(cm_df.to_string())
print()


# =============================================================================
# PER-CLASS METRICS
# =============================================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y,
        y_pred,
        labels=np.arange(EXPECTED_N_CLASSES),
        average=None,
        zero_division=0,
    )
)

predicted_counts = np.bincount(
    y_pred,
    minlength=EXPECTED_N_CLASSES
)

true_positive = np.diag(cm)

false_positive = (
    predicted_counts
    - true_positive
)

false_negative = (
    np.bincount(
        y,
        minlength=EXPECTED_N_CLASSES
    )
    - true_positive
)

error_count = false_negative

error_rate = np.divide(
    false_negative,
    support,
    out=np.zeros_like(
        false_negative,
        dtype=float
    ),
    where=support != 0
)

per_class_df = pd.DataFrame({
    "Class_ID": np.arange(EXPECTED_N_CLASSES),
    "Label": CLASS_NAMES,
    "Support": support.astype(int),
    "Predicted_Count": predicted_counts.astype(int),
    "True_Positive": true_positive.astype(int),
    "False_Positive": false_positive.astype(int),
    "False_Negative": false_negative.astype(int),
    "Error_Count": error_count.astype(int),
    "Error_Rate": error_rate,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
})

print("=" * 80)
print("PER-CLASS ERROR ANALYSIS")
print("=" * 80)

print(
    per_class_df.to_string(
        index=False
    )
)

print()


# =============================================================================
# CONFUSION PAIR ANALYSIS
# =============================================================================

print("=" * 80)
print("CONFUSION PAIR ANALYSIS")
print("=" * 80)

confusion_pairs = []

for actual_id in range(EXPECTED_N_CLASSES):

    actual_support = int(
        support[actual_id]
    )

    for predicted_id in range(EXPECTED_N_CLASSES):

        if actual_id == predicted_id:
            continue

        count = int(
            cm[actual_id, predicted_id]
        )

        if count == 0:
            continue

        percentage = (
            count / actual_support
            if actual_support > 0
            else 0.0
        )

        confusion_pairs.append({
            "Actual_Class_ID": actual_id,
            "Actual_Label": ID_TO_CLASS[actual_id],
            "Predicted_Class_ID": predicted_id,
            "Predicted_Label": ID_TO_CLASS[predicted_id],
            "Confusion_Count": count,
            "Percentage_of_Actual_Class": percentage,
        })

confusion_pairs_df = pd.DataFrame(
    confusion_pairs
)

if not confusion_pairs_df.empty:

    confusion_pairs_df = (
        confusion_pairs_df
        .sort_values(
            by="Confusion_Count",
            ascending=False
        )
        .reset_index(drop=True)
    )

print(
    confusion_pairs_df.to_string(
        index=False
    )
)

print()


# =============================================================================
# TOP 20 CONFUSION PAIRS
# =============================================================================

print("=" * 80)
print("TOP 20 CONFUSION PAIRS")
print("=" * 80)

top_confusions_df = (
    confusion_pairs_df
    .head(20)
    .copy()
)

print(
    top_confusions_df.to_string(
        index=False
    )
)

print()


# =============================================================================
# DIFFICULT CLASS RANKING
# =============================================================================

print("=" * 80)
print("DIFFICULT CLASS RANKING")
print("=" * 80)

difficult_df = per_class_df.copy()

difficult_df = difficult_df.sort_values(
    by=["F1", "Error_Rate"],
    ascending=[True, False]
).reset_index(drop=True)

difficult_df.insert(
    0,
    "Difficulty_Rank",
    np.arange(1, len(difficult_df) + 1)
)

difficult_output_columns = [
    "Difficulty_Rank",
    "Class_ID",
    "Label",
    "Support",
    "F1",
    "Recall",
    "Precision",
    "Error_Rate",
    "Error_Count",
]

print(
    difficult_df[
        difficult_output_columns
    ].to_string(index=False)
)

print()


# =============================================================================
# INFILTRATION-SPECIFIC ANALYSIS
# =============================================================================

INFILTRATION_ID = CLASS_TO_ID["Infiltration"]

print("=" * 80)
print("INFILTRATION-SPECIFIC ANALYSIS")
print("=" * 80)

infiltration_support = int(
    support[INFILTRATION_ID]
)

infiltration_tp = int(
    true_positive[INFILTRATION_ID]
)

infiltration_fn = int(
    false_negative[INFILTRATION_ID]
)

infiltration_fp = int(
    false_positive[INFILTRATION_ID]
)

print(
    f"Infiltration support       : "
    f"{infiltration_support:,}"
)

print(
    f"Infiltration TP            : "
    f"{infiltration_tp:,}"
)

print(
    f"Infiltration FN            : "
    f"{infiltration_fn:,}"
)

print(
    f"Infiltration FP            : "
    f"{infiltration_fp:,}"
)

print(
    f"Infiltration Recall        : "
    f"{recall[INFILTRATION_ID]:.6f}"
)

print(
    f"Infiltration Precision     : "
    f"{precision[INFILTRATION_ID]:.6f}"
)

print()

infiltration_rows = []

for predicted_id in range(EXPECTED_N_CLASSES):

    count = int(
        cm[INFILTRATION_ID, predicted_id]
    )

    if count == 0:
        continue

    percentage = (
        count / infiltration_support
        if infiltration_support > 0
        else 0.0
    )

    infiltration_rows.append({
        "Actual_Class": "Infiltration",
        "Actual_Class_ID": INFILTRATION_ID,
        "Predicted_Class_ID": predicted_id,
        "Predicted_Label": ID_TO_CLASS[predicted_id],
        "Count": count,
        "Percentage_of_Infiltration": percentage,
    })

infiltration_df = pd.DataFrame(
    infiltration_rows
)

if not infiltration_df.empty:

    infiltration_df = (
        infiltration_df
        .sort_values(
            by="Count",
            ascending=False
        )
        .reset_index(drop=True)
    )

print("Infiltration prediction distribution:")

print(
    infiltration_df.to_string(
        index=False
    )
)

print()


# =============================================================================
# RARE CLASS ANALYSIS
# =============================================================================

RARE_CLASS_THRESHOLD = 100

rare_df = per_class_df[
    per_class_df["Support"]
    <= RARE_CLASS_THRESHOLD
].copy()

rare_df = rare_df.sort_values(
    by="F1",
    ascending=True
)

print("=" * 80)
print("RARE CLASS ANALYSIS")
print("=" * 80)

print(
    rare_df[
        [
            "Class_ID",
            "Label",
            "Support",
            "Precision",
            "Recall",
            "F1",
            "Error_Count",
            "Error_Rate",
        ]
    ].to_string(index=False)
)

print()


# =============================================================================
# ERROR CONCENTRATION ANALYSIS
# =============================================================================

total_samples = len(y)

total_errors = int(
    np.sum(y != y_pred)
)

correct_predictions = (
    total_samples - total_errors
)

top_20_error_count = (
    int(
        top_confusions_df[
            "Confusion_Count"
        ].sum()
    )
    if not top_confusions_df.empty
    else 0
)

error_concentration = (
    top_20_error_count / total_errors
    if total_errors > 0
    else 0.0
)

print("=" * 80)
print("ERROR CONCENTRATION ANALYSIS")
print("=" * 80)

print(
    f"Total validation samples : "
    f"{total_samples:,}"
)

print(
    f"Correct predictions      : "
    f"{correct_predictions:,}"
)

print(
    f"Total classification errors: "
    f"{total_errors:,}"
)

print(
    f"Top 20 confusion errors  : "
    f"{top_20_error_count:,}"
)

print(
    f"Top 20 error concentration: "
    f"{error_concentration:.6f}"
)

print()


# =============================================================================
# RESEARCH IMPROVEMENT PRIORITY
# =============================================================================

print("=" * 80)
print("RESEARCH IMPROVEMENT PRIORITY")
print("=" * 80)

improvement_df = difficult_df[
    difficult_df["Error_Count"] > 0
].copy()

improvement_df = improvement_df[
    [
        "Difficulty_Rank",
        "Class_ID",
        "Label",
        "Support",
        "F1",
        "Recall",
        "Precision",
        "Error_Count",
        "Error_Rate",
    ]
]

print(
    improvement_df.to_string(
        index=False
    )
)

print()


# =============================================================================
# SAVE RESULTS
# =============================================================================

print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)


# Confusion matrix
cm_df.to_csv(
    OUTPUT_DIR
    / "step33_xgb_confusion_matrix.csv"
)


# Per-class error analysis
per_class_df.to_csv(
    OUTPUT_DIR
    / "step33_xgb_per_class_error_analysis.csv",
    index=False
)


# Confusion pairs
confusion_pairs_df.to_csv(
    OUTPUT_DIR
    / "step33_xgb_confusion_pairs.csv",
    index=False
)


# Top confusions
top_confusions_df.to_csv(
    OUTPUT_DIR
    / "step33_xgb_top_confusions.csv",
    index=False
)


# Difficult classes
difficult_df[
    difficult_output_columns
].to_csv(
    OUTPUT_DIR
    / "step33_xgb_difficult_classes.csv",
    index=False
)


# Infiltration analysis
infiltration_df.to_csv(
    OUTPUT_DIR
    / "step33_xgb_infiltration_analysis.csv",
    index=False
)


# Rare classes
rare_df[
    [
        "Class_ID",
        "Label",
        "Support",
        "Precision",
        "Recall",
        "F1",
        "Error_Count",
        "Error_Rate",
    ]
].to_csv(
    OUTPUT_DIR
    / "step33_xgb_rare_class_analysis.csv",
    index=False
)


# =============================================================================
# SUMMARY JSON
# =============================================================================

summary = {
    "step": "33",
    "experiment": "XGBoost_Difficult_Class_and_Error_Analysis",
    "dataset": "CIC-IDS2018",
    "evaluation_dataset": str(
        VALIDATION_PATH
    ),
    "model": str(
        XGB_MODEL_PATH
    ),
    "training_performed": False,
    "hyperparameter_optimization_performed": False,
    "test_set_used": False,
    "validation_samples": int(total_samples),
    "feature_count": int(len(feature_columns)),
    "class_count": int(EXPECTED_N_CLASSES),
    "metrics": {
        "Accuracy": float(accuracy),
        "Balanced_Accuracy": float(
            balanced_accuracy
        ),
        "Macro_Precision": float(
            precision_macro
        ),
        "Macro_Recall": float(
            recall_macro
        ),
        "Macro_F1": float(
            f1_macro
        ),
        "Weighted_F1": float(
            f1_weighted
        ),
    },
    "errors": {
        "total_errors": int(total_errors),
        "correct_predictions": int(
            correct_predictions
        ),
        "top_20_confusion_errors": int(
            top_20_error_count
        ),
        "top_20_error_concentration": float(
            error_concentration
        ),
    },
    "infiltration": {
        "support": int(
            infiltration_support
        ),
        "true_positive": int(
            infiltration_tp
        ),
        "false_negative": int(
            infiltration_fn
        ),
        "false_positive": int(
            infiltration_fp
        ),
        "recall": float(
            recall[INFILTRATION_ID]
        ),
        "precision": float(
            precision[INFILTRATION_ID]
        ),
        "f1": float(
            f1[INFILTRATION_ID]
        ),
    },
    "rare_class_threshold": int(
        RARE_CLASS_THRESHOLD
    ),
    "rare_class_count": int(
        len(rare_df)
    ),
    "difficult_classes": [
        {
            "rank": int(row["Difficulty_Rank"]),
            "class_id": int(row["Class_ID"]),
            "label": str(row["Label"]),
            "support": int(row["Support"]),
            "f1": float(row["F1"]),
            "recall": float(row["Recall"]),
            "precision": float(row["Precision"]),
            "error_count": int(
                row["Error_Count"]
            ),
            "error_rate": float(
                row["Error_Rate"]
            ),
        }
        for _, row in difficult_df.iterrows()
    ],
    "top_confusion_pairs": [
        {
            "actual_class_id": int(
                row["Actual_Class_ID"]
            ),
            "actual_label": str(
                row["Actual_Label"]
            ),
            "predicted_class_id": int(
                row["Predicted_Class_ID"]
            ),
            "predicted_label": str(
                row["Predicted_Label"]
            ),
            "count": int(
                row["Confusion_Count"]
            ),
            "percentage_of_actual_class": float(
                row["Percentage_of_Actual_Class"]
            ),
        }
        for _, row in top_confusions_df.iterrows()
    ],
}


with open(
    OUTPUT_DIR
    / "step33_xgb_error_analysis_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# =============================================================================
# PROTOCOL FILE
# =============================================================================

runtime_seconds = (
    time.time() - START_TIME
)

protocol = f"""
STEP 33 - XGBOOST DIFFICULT-CLASS & ERROR ANALYSIS

Research Project:
Behaviour Based Network Attack Detection System using Machine Learning

Dataset:
CIC-IDS2018

Model:
Previously trained Step 23 XGBoost classifier

Evaluation dataset:
CICIDS2018_validation_selected.csv

Training representation:
B1 undersampled

Feature count:
30

Class count:
14

Validation samples:
{total_samples:,}

Primary model:
XGBoost

Overall validation metrics:
Accuracy          = {accuracy:.6f}
Balanced Accuracy = {balanced_accuracy:.6f}
Macro Precision   = {precision_macro:.6f}
Macro Recall      = {recall_macro:.6f}
Macro F1          = {f1_macro:.6f}
Weighted F1       = {f1_weighted:.6f}

Total classification errors:
{total_errors:,}

Infiltration:
Support    = {infiltration_support:,}
Precision  = {precision[INFILTRATION_ID]:.6f}
Recall     = {recall[INFILTRATION_ID]:.6f}
F1         = {f1[INFILTRATION_ID]:.6f}

Rare class threshold:
Support <= {RARE_CLASS_THRESHOLD}

Rare class count:
{len(rare_df)}

Top error concentration:
Top 20 confusion errors = {top_20_error_count:,}
Concentration = {error_concentration:.6f}

Methodological controls:
- No additional training performed.
- No hyperparameter optimization performed.
- No validation-based model tuning performed.
- No test-set data used.
- Analysis performed on the completed Step 23 XGBoost model.
- The validation dataset was used only for post-training evaluation.

Runtime:
{runtime_seconds:.2f} seconds
"""

with open(
    OUTPUT_DIR
    / "step33_xgb_error_analysis_protocol.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        protocol.strip()
        + "\n"
    )


# =============================================================================
# FINAL REPORT
# =============================================================================

print()
print("=" * 80)
print("STEP 33 COMPLETED")
print("=" * 80)

print(
    f"XGB Macro F1          : "
    f"{f1_macro:.6f}"
)

print(
    f"Total errors          : "
    f"{total_errors:,}"
)

print(
    f"Infiltration F1       : "
    f"{f1[INFILTRATION_ID]:.6f}"
)

print(
    f"Infiltration Recall   : "
    f"{recall[INFILTRATION_ID]:.6f}"
)

print(
    f"Infiltration Precision: "
    f"{precision[INFILTRATION_ID]:.6f}"
)

print(
    f"Rare classes          : "
    f"{len(rare_df)}"
)

print()

print("Top difficult classes:")

for _, row in difficult_df.head(5).iterrows():

    print(
        f"  {int(row['Difficulty_Rank'])}. "
        f"{row['Label']} | "
        f"F1={row['F1']:.6f} | "
        f"Support={int(row['Support']):,}"
    )

print()

print("Top confusion pairs:")

for _, row in top_confusions_df.head(5).iterrows():

    print(
        f"  {row['Actual_Label']} -> "
        f"{row['Predicted_Label']} : "
        f"{int(row['Confusion_Count']):,}"
    )

print()

print("Output directory:")
print(OUTPUT_DIR)

print()

print("Generated files:")
print("  - step33_xgb_confusion_pairs.csv")
print("  - step33_xgb_difficult_classes.csv")
print("  - step33_xgb_error_analysis_protocol.txt")
print("  - step33_xgb_error_analysis_summary.json")
print("  - step33_xgb_infiltration_analysis.csv")
print("  - step33_xgb_per_class_error_analysis.csv")
print("  - step33_xgb_rare_class_analysis.csv")
print("  - step33_xgb_top_confusions.csv")
print("  - step33_xgb_confusion_matrix.csv")

print()
print("No additional training was performed.")
print("No hyperparameter optimization was performed.")
print("No test-set data was used.")

print()
print("=" * 80)
print("STEP 33 FINISHED SUCCESSFULLY.")
print("=" * 80)