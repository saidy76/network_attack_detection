from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =============================================================================
# STEP 32 - RANDOM FOREST DIFFICULT-CLASS & ERROR ANALYSIS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "CICIDS2018_validation_selected.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "step21_rf_tuning"
    / "step21_best_random_forest.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step32_rf_error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# FIXED CLASS DEFINITIONS
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
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}


# =============================================================================
# PRINT HEADER
# =============================================================================

print("=" * 80)
print("STEP 32 - RANDOM FOREST DIFFICULT-CLASS & ERROR ANALYSIS")
print("=" * 80)

print(
    "Research Project:"
    "\nBehaviour Based Network Attack Detection System using Machine Learning"
)

print()
print(f"Project root : {PROJECT_ROOT}")
print(f"Validation   : {VALIDATION_PATH}")
print(f"RF model     : {MODEL_PATH}")
print(f"Output       : {OUTPUT_DIR}")


# =============================================================================
# INPUT CHECK
# =============================================================================

print("\n" + "=" * 80)
print("INPUT FILE CHECK")
print("=" * 80)

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Random Forest model not found:\n{MODEL_PATH}"
    )

print(f"FOUND: {VALIDATION_PATH}")
print(f"FOUND: {MODEL_PATH}")


# =============================================================================
# LOAD VALIDATION DATA
# =============================================================================

print("\n" + "=" * 80)
print("LOADING VALIDATION DATA")
print("=" * 80)

start_time = time.time()

df = pd.read_csv(
    VALIDATION_PATH
)

print(f"Validation shape: {df.shape}")

if "Label" not in df.columns:
    raise ValueError(
        "Validation dataset must contain a 'Label' column."
    )


# =============================================================================
# FEATURE VERIFICATION
# =============================================================================

print("\n" + "=" * 80)
print("FEATURE VERIFICATION")
print("=" * 80)

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column != "Label"
]

print(f"Feature count: {len(FEATURE_COLUMNS)}")

if len(FEATURE_COLUMNS) != 30:
    raise ValueError(
        f"Expected 30 features, found {len(FEATURE_COLUMNS)}."
    )


X_val = df[FEATURE_COLUMNS]
y_val_raw = df["Label"]


# =============================================================================
# LABEL ENCODING
# =============================================================================

print("\n" + "=" * 80)
print("LABEL ENCODING")
print("=" * 80)

unknown_labels = sorted(
    set(y_val_raw.unique()) - set(CLASS_NAMES)
)

if unknown_labels:
    raise ValueError(
        f"Unknown validation labels detected: {unknown_labels}"
    )

y_val = y_val_raw.map(CLASS_TO_ID).astype(np.int32)

print(f"Validation samples : {len(y_val):,}")
print(f"Classes present    : {sorted(y_val.unique().tolist())}")


# =============================================================================
# LOAD RANDOM FOREST
# =============================================================================

print("\n" + "=" * 80)
print("LOADING RANDOM FOREST MODEL")
print("=" * 80)

model = joblib.load(
    MODEL_PATH
)

print(f"Model type       : {type(model)}")
print(f"Model classes    : {model.classes_}")
print(f"Model n_classes  : {model.n_classes_}")
print(f"Model n_features : {model.n_features_in_}")


# =============================================================================
# MODEL STRUCTURAL VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("MODEL STRUCTURAL VALIDATION")
print("=" * 80)

if model.n_classes_ != len(CLASS_NAMES):
    raise ValueError(
        "RF class count does not match the fixed 14-class definition."
    )

if model.n_features_in_ != len(FEATURE_COLUMNS):
    raise ValueError(
        "RF feature count does not match validation feature count."
    )

print("RF class count  : PASS")
print("RF feature count: PASS")


# =============================================================================
# GENERATE PREDICTIONS
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING RANDOM FOREST VALIDATION PREDICTIONS")
print("=" * 80)

prediction_start = time.time()

y_pred_raw = model.predict(
    X_val
)

prediction_time = time.time() - prediction_start

# RF stores original string class labels.
# Convert them back into the fixed integer class IDs.
if y_pred_raw.dtype.kind in {"U", "S", "O"}:

    unknown_predictions = sorted(
        set(y_pred_raw) - set(CLASS_NAMES)
    )

    if unknown_predictions:
        raise ValueError(
            f"Unknown RF predictions detected: {unknown_predictions}"
        )

    y_pred = pd.Series(
        y_pred_raw
    ).map(
        CLASS_TO_ID
    ).astype(
        np.int32
    ).to_numpy()

else:
    y_pred = np.asarray(
        y_pred_raw
    ).astype(
        np.int32
    )

print(f"Prediction shape: {y_pred.shape}")
print(f"Prediction runtime: {prediction_time:.2f} seconds")


# =============================================================================
# OVERALL METRICS
# =============================================================================

print("\n" + "=" * 80)
print("OVERALL RF VALIDATION METRICS")
print("=" * 80)

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

print(f"Accuracy          : {accuracy:.6f}")
print(f"Balanced Accuracy : {balanced_accuracy:.6f}")
print(f"Macro Precision   : {macro_precision:.6f}")
print(f"Macro Recall      : {macro_recall:.6f}")
print(f"Macro F1          : {macro_f1:.6f}")
print(f"Weighted F1       : {weighted_f1:.6f}")


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

labels = list(range(len(CLASS_NAMES)))

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

print(cm_df.to_string())


# =============================================================================
# PER-CLASS METRICS
# =============================================================================

print("\n" + "=" * 80)
print("PER-CLASS ERROR ANALYSIS")
print("=" * 80)

report = classification_report(
    y_val,
    y_pred,
    labels=labels,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

per_class_rows = []

for class_id, class_name in enumerate(CLASS_NAMES):

    tp = int(cm[class_id, class_id])

    actual_support = int(
        cm[class_id, :].sum()
    )

    predicted_support = int(
        cm[:, class_id].sum()
    )

    false_negative = int(
        actual_support - tp
    )

    false_positive = int(
        predicted_support - tp
    )

    error_count = false_negative

    error_rate = (
        error_count / actual_support
        if actual_support > 0
        else 0.0
    )

    row = report[class_name]

    per_class_rows.append(
        {
            "Class_ID": class_id,
            "Label": class_name,
            "Support": actual_support,
            "Predicted_Count": predicted_support,
            "True_Positive": tp,
            "False_Positive": false_positive,
            "False_Negative": false_negative,
            "Error_Count": error_count,
            "Error_Rate": error_rate,
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1-score"],
        }
    )


per_class_df = pd.DataFrame(
    per_class_rows
)

print(
    per_class_df.to_string(
        index=False
    )
)


# =============================================================================
# CONFUSION PAIR ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("CONFUSION PAIR ANALYSIS")
print("=" * 80)

confusion_rows = []

for actual_id in labels:

    for predicted_id in labels:

        if actual_id == predicted_id:
            continue

        count = int(
            cm[actual_id, predicted_id]
        )

        if count == 0:
            continue

        actual_support = int(
            cm[actual_id, :].sum()
        )

        percentage_of_actual = (
            count / actual_support
            if actual_support > 0
            else 0.0
        )

        confusion_rows.append(
            {
                "Actual_Class_ID": actual_id,
                "Actual_Label": CLASS_NAMES[actual_id],
                "Predicted_Class_ID": predicted_id,
                "Predicted_Label": CLASS_NAMES[predicted_id],
                "Confusion_Count": count,
                "Percentage_of_Actual_Class": percentage_of_actual,
            }
        )


confusion_pairs_df = pd.DataFrame(
    confusion_rows
)

if not confusion_pairs_df.empty:

    confusion_pairs_df = confusion_pairs_df.sort_values(
        by=[
            "Confusion_Count",
            "Percentage_of_Actual_Class",
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

print(
    confusion_pairs_df.head(30).to_string(
        index=False
    )
)


# =============================================================================
# TOP CONFUSIONS
# =============================================================================

top_confusions_df = confusion_pairs_df.head(
    20
).copy()

print("\n" + "=" * 80)
print("TOP 20 CONFUSION PAIRS")
print("=" * 80)

print(
    top_confusions_df.to_string(
        index=False
    )
)


# =============================================================================
# DIFFICULT CLASS RANKING
# =============================================================================

print("\n" + "=" * 80)
print("DIFFICULT CLASS RANKING")
print("=" * 80)

difficult_classes_df = per_class_df.copy()

# Difficulty score is based primarily on low F1.
# Error rate is retained as a secondary diagnostic.
difficult_classes_df["Difficulty_Score"] = (
    1.0 - difficult_classes_df["F1"]
)

difficult_classes_df = difficult_classes_df.sort_values(
    by=[
        "Difficulty_Score",
        "Error_Rate",
    ],
    ascending=False
).reset_index(
    drop=True
)

difficult_classes_df["Difficulty_Rank"] = (
    np.arange(
        1,
        len(difficult_classes_df) + 1
    )
)

print(
    difficult_classes_df[
        [
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
    ].to_string(
        index=False
    )
)


# =============================================================================
# INFILTRATION-SPECIFIC ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("INFILTRATION-SPECIFIC ANALYSIS")
print("=" * 80)

INFILTRATION_ID = CLASS_TO_ID["Infiltration"]

infiltration_support = int(
    cm[INFILTRATION_ID, :].sum()
)

infiltration_tp = int(
    cm[INFILTRATION_ID, INFILTRATION_ID]
)

infiltration_fn = (
    infiltration_support
    - infiltration_tp
)

infiltration_recall = (
    infiltration_tp / infiltration_support
    if infiltration_support > 0
    else 0.0
)

infiltration_predictions = cm[
    :,
    INFILTRATION_ID
].copy()

infiltration_fp = (
    int(infiltration_predictions.sum())
    - infiltration_tp
)

infiltration_precision = (
    infiltration_tp
    / int(infiltration_predictions.sum())
    if int(infiltration_predictions.sum()) > 0
    else 0.0
)

infiltration_rows = []

for predicted_id in labels:

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

    infiltration_rows.append(
        {
            "Actual_Class": "Infiltration",
            "Actual_Class_ID": INFILTRATION_ID,
            "Predicted_Class_ID": predicted_id,
            "Predicted_Label": CLASS_NAMES[predicted_id],
            "Count": count,
            "Percentage_of_Infiltration": percentage,
        }
    )


infiltration_df = pd.DataFrame(
    infiltration_rows
).sort_values(
    by="Count",
    ascending=False
).reset_index(
    drop=True
)

print(
    f"Infiltration support       : {infiltration_support:,}"
)

print(
    f"Infiltration TP            : {infiltration_tp:,}"
)

print(
    f"Infiltration FN            : {infiltration_fn:,}"
)

print(
    f"Infiltration FP            : {infiltration_fp:,}"
)

print(
    f"Infiltration Recall        : {infiltration_recall:.6f}"
)

print(
    f"Infiltration Precision     : {infiltration_precision:.6f}"
)

print("\nInfiltration prediction distribution:")

print(
    infiltration_df.to_string(
        index=False
    )
)


# =============================================================================
# RARE CLASS ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("RARE CLASS ANALYSIS")
print("=" * 80)

RARE_SUPPORT_THRESHOLD = 100

rare_class_df = per_class_df[
    per_class_df["Support"]
    <= RARE_SUPPORT_THRESHOLD
].copy()

rare_class_df = rare_class_df.sort_values(
    by="Support",
    ascending=True
).reset_index(
    drop=True
)

rare_class_df["Rare_Class"] = True

print(
    rare_class_df[
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
    ].to_string(
        index=False
    )
)


# =============================================================================
# MAJOR ERROR CONCENTRATION
# =============================================================================

print("\n" + "=" * 80)
print("ERROR CONCENTRATION ANALYSIS")
print("=" * 80)

total_samples = len(y_val)

correct_predictions = int(
    np.trace(cm)
)

total_errors = (
    total_samples
    - correct_predictions
)

print(
    f"Total validation samples : {total_samples:,}"
)

print(
    f"Correct predictions      : {correct_predictions:,}"
)

print(
    f"Total classification errors: {total_errors:,}"
)

if total_errors > 0:

    top_error_count = int(
        top_confusions_df["Confusion_Count"].sum()
    )

    top_error_concentration = (
        top_error_count / total_errors
    )

else:

    top_error_count = 0
    top_error_concentration = 0.0


print(
    f"Top 20 confusion errors  : {top_error_count:,}"
)

print(
    f"Top 20 error concentration: "
    f"{top_error_concentration:.6f}"
)


# =============================================================================
# IDENTIFY PRIORITY CLASSES
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH IMPROVEMENT PRIORITY")
print("=" * 80)

priority_candidates = difficult_classes_df[
    (
        difficult_classes_df["Support"] >= 10
    )
    &
    (
        difficult_classes_df["F1"] < 0.80
    )
].copy()

priority_candidates = priority_candidates[
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
    priority_candidates.to_string(
        index=False
    )
)


# =============================================================================
# SAVE PER-CLASS ERROR ANALYSIS
# =============================================================================

per_class_df.to_csv(
    OUTPUT_DIR
    / "step32_rf_per_class_error_analysis.csv",
    index=False
)


# =============================================================================
# SAVE CONFUSION PAIRS
# =============================================================================

confusion_pairs_df.to_csv(
    OUTPUT_DIR
    / "step32_rf_confusion_pairs.csv",
    index=False
)


# =============================================================================
# SAVE TOP CONFUSIONS
# =============================================================================

top_confusions_df.to_csv(
    OUTPUT_DIR
    / "step32_rf_top_confusions.csv",
    index=False
)


# =============================================================================
# SAVE DIFFICULT CLASSES
# =============================================================================

difficult_classes_df.to_csv(
    OUTPUT_DIR
    / "step32_rf_difficult_classes.csv",
    index=False
)


# =============================================================================
# SAVE INFILTRATION ANALYSIS
# =============================================================================

infiltration_df.to_csv(
    OUTPUT_DIR
    / "step32_rf_infiltration_analysis.csv",
    index=False
)


# =============================================================================
# SAVE RARE CLASS ANALYSIS
# =============================================================================

rare_class_df.to_csv(
    OUTPUT_DIR
    / "step32_rf_rare_class_analysis.csv",
    index=False
)


# =============================================================================
# SUMMARY JSON
# =============================================================================

top_difficult_records = []

for _, row in difficult_classes_df.head(10).iterrows():

    top_difficult_records.append(
        {
            "rank": int(row["Difficulty_Rank"]),
            "class_id": int(row["Class_ID"]),
            "label": row["Label"],
            "support": int(row["Support"]),
            "f1": float(row["F1"]),
            "recall": float(row["Recall"]),
            "precision": float(row["Precision"]),
            "error_count": int(row["Error_Count"]),
            "error_rate": float(row["Error_Rate"]),
        }
    )


top_confusion_records = []

for _, row in top_confusions_df.head(20).iterrows():

    top_confusion_records.append(
        {
            "actual_class_id": int(
                row["Actual_Class_ID"]
            ),
            "actual_label": row["Actual_Label"],
            "predicted_class_id": int(
                row["Predicted_Class_ID"]
            ),
            "predicted_label": row["Predicted_Label"],
            "count": int(
                row["Confusion_Count"]
            ),
            "percentage_of_actual_class": float(
                row["Percentage_of_Actual_Class"]
            ),
        }
    )


summary = {
    "step": 32,
    "experiment": "Random Forest Difficult-Class and Error Analysis",
    "dataset": "CIC-IDS2018",
    "evaluation_dataset": "CICIDS2018_validation_selected.csv",
    "model": "step21_best_random_forest.joblib",
    "training_representation": "B1_undersampled",
    "feature_count": len(FEATURE_COLUMNS),
    "class_count": len(CLASS_NAMES),
    "validation_samples": int(total_samples),
    "test_set_used": False,
    "additional_training": False,
    "hyperparameter_optimization": False,
    "overall_metrics": {
        "Accuracy": float(accuracy),
        "Balanced_Accuracy": float(balanced_accuracy),
        "Macro_Precision": float(macro_precision),
        "Macro_Recall": float(macro_recall),
        "Macro_F1": float(macro_f1),
        "Weighted_F1": float(weighted_f1),
    },
    "error_statistics": {
        "correct_predictions": int(correct_predictions),
        "total_errors": int(total_errors),
        "top_20_confusion_errors": int(top_error_count),
        "top_20_error_concentration": float(
            top_error_concentration
        ),
    },
    "infiltration_analysis": {
        "support": infiltration_support,
        "true_positive": infiltration_tp,
        "false_negative": infiltration_fn,
        "false_positive": infiltration_fp,
        "recall": float(infiltration_recall),
        "precision": float(infiltration_precision),
    },
    "rare_class_threshold": RARE_SUPPORT_THRESHOLD,
    "rare_class_count": int(
        len(rare_class_df)
    ),
    "top_difficult_classes": top_difficult_records,
    "top_confusion_pairs": top_confusion_records,
    "priority_classes": [
        {
            "class_id": int(row["Class_ID"]),
            "label": row["Label"],
            "support": int(row["Support"]),
            "f1": float(row["F1"]),
            "recall": float(row["Recall"]),
            "precision": float(row["Precision"]),
        }
        for _, row in priority_candidates.iterrows()
    ],
    "methodological_notes": [
        "Existing tuned Random Forest from Step 21 was evaluated.",
        "No additional model training was performed.",
        "No hyperparameter optimization was performed.",
        "No validation data was used for model tuning.",
        "No test-set data was used.",
        "Analysis focuses on class-level errors and confusion structure.",
    ],
    "runtime_seconds": float(
        time.time() - start_time
    ),
}


with open(
    OUTPUT_DIR
    / "step32_rf_error_analysis_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# =============================================================================
# RESEARCH PROTOCOL
# =============================================================================

protocol = f"""
STEP 32 - RANDOM FOREST DIFFICULT-CLASS & ERROR ANALYSIS

Project:
Behaviour Based Network Attack Detection System using Machine Learning

Dataset:
CIC-IDS2018

Evaluation dataset:
CICIDS2018_validation_selected.csv

Model:
Step 21 tuned Random Forest

Training representation:
B1 undersampled

Feature count:
{len(FEATURE_COLUMNS)}

Class count:
{len(CLASS_NAMES)}

Validation samples:
{total_samples:,}

Overall validation metrics:

Accuracy:
{accuracy:.6f}

Balanced Accuracy:
{balanced_accuracy:.6f}

Macro Precision:
{macro_precision:.6f}

Macro Recall:
{macro_recall:.6f}

Macro F1:
{macro_f1:.6f}

Weighted F1:
{weighted_f1:.6f}

Error analysis:

Correct predictions:
{correct_predictions:,}

Total classification errors:
{total_errors:,}

Top 20 confusion error count:
{top_error_count:,}

Top 20 error concentration:
{top_error_concentration:.6f}

Infiltration:

Support:
{infiltration_support:,}

True Positive:
{infiltration_tp:,}

False Negative:
{infiltration_fn:,}

False Positive:
{infiltration_fp:,}

Recall:
{infiltration_recall:.6f}

Precision:
{infiltration_precision:.6f}

Rare class definition:
Support <= {RARE_SUPPORT_THRESHOLD}

Methodological controls:

1. Existing Step 21 Random Forest was reused.
2. No additional model training was performed.
3. No hyperparameter optimization was performed.
4. Validation data was used only for post-training error analysis.
5. Test data was not used.
6. No additional balancing was performed.
7. No feature selection was performed.
8. Analysis focused on difficult classes and confusion patterns.
9. Results are intended to determine the evidence-based priorities for Step 33.

Primary purpose:

Identify the dominant failure modes of the current best Random Forest
before attempting any model improvement.

This step does not claim novelty by itself.
It provides empirical evidence for subsequent research decisions.
"""

with open(
    OUTPUT_DIR
    / "step32_rf_error_analysis_protocol.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        protocol.strip()
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("STEP 32 COMPLETED")
print("=" * 80)

print(
    f"RF Macro F1          : {macro_f1:.6f}"
)

print(
    f"Total errors         : {total_errors:,}"
)

print(
    f"Infiltration F1      : "
    f"{per_class_df.loc[INFILTRATION_ID, 'F1']:.6f}"
)

print(
    f"Infiltration Recall  : "
    f"{infiltration_recall:.6f}"
)

print(
    f"Infiltration Precision: "
    f"{infiltration_precision:.6f}"
)

print(
    f"Rare classes         : "
    f"{len(rare_class_df)}"
)

print()
print("Top difficult classes:")

for _, row in difficult_classes_df.head(5).iterrows():

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

for file_path in sorted(
    OUTPUT_DIR.iterdir()
):

    if file_path.is_file():
        print(
            f"  - {file_path.name}"
        )

print()
print("No additional training was performed.")
print("No hyperparameter optimization was performed.")
print("No test-set data was used.")

print()
print("=" * 80)
print("STEP 32 FINISHED SUCCESSFULLY.")
print("=" * 80)