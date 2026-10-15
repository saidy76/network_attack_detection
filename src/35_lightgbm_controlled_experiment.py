from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# =============================================================================
# STEP 35 - CONTROLLED LIGHTGBM EXPERIMENT
# =============================================================================

PROJECT_ROOT = Path(r"D:\network_attack_detection")

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "balanced"
    / "CICIDS2018_train_B1_undersampled.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "CICIDS2018_validation_selected.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step35_lightgbm_experiment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "step35_best_lightgbm.joblib"
METRICS_PATH = OUTPUT_DIR / "step35_lightgbm_validation_metrics.csv"
PER_CLASS_PATH = OUTPUT_DIR / "step35_lightgbm_per_class_metrics.csv"
CONFUSION_PATH = OUTPUT_DIR / "step35_lightgbm_confusion_matrix.csv"
SUMMARY_PATH = OUTPUT_DIR / "step35_lightgbm_experiment_summary.json"
PROTOCOL_PATH = OUTPUT_DIR / "step35_lightgbm_experiment_protocol.txt"

# Fixed class mapping used throughout the project.
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

LABEL_TO_ID = {label: idx for idx, label in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(CLASS_NAMES)}

LABEL_COLUMN = "Label"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def encode_labels(series, dataset_name):
    """
    Convert the project's string class labels into the fixed numeric IDs 0-13.
    This fixes the previous Step 35 failure where string labels were incorrectly
    checked against numeric class IDs.
    """
    unknown = sorted(set(series.astype(str).unique()) - set(LABEL_TO_ID.keys()))

    if unknown:
        raise ValueError(
            f"Unknown {dataset_name} labels detected:\n"
            + "\n".join(str(x) for x in unknown)
        )

    return series.astype(str).map(LABEL_TO_ID).astype(np.int64)


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("STEP 35 - CONTROLLED LIGHTGBM EXPERIMENT")
print("=" * 80)
print("Research Project:")
print("Behaviour Based Network Attack Detection System using Machine Learning")
print()
print(f"Training     : {TRAIN_PATH}")
print(f"Validation   : {VALIDATION_PATH}")
print(f"Output       : {OUTPUT_DIR}")
print("=" * 80)


# =============================================================================
# INPUT FILE CHECK
# =============================================================================

print_section("INPUT FILE CHECK")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(f"Validation file not found: {VALIDATION_PATH}")

print(f"FOUND: {TRAIN_PATH}")
print(f"FOUND: {VALIDATION_PATH}")


# =============================================================================
# LOAD DATA
# =============================================================================

print_section("LOADING B1 TRAINING DATA")

train_df = pd.read_csv(TRAIN_PATH)
print(f"Training shape: {train_df.shape}")

print_section("LOADING VALIDATION DATA")

val_df = pd.read_csv(VALIDATION_PATH)
print(f"Validation shape: {val_df.shape}")


# =============================================================================
# LABEL COLUMN
# =============================================================================

print_section("LABEL VERIFICATION")

if LABEL_COLUMN not in train_df.columns:
    raise ValueError(f"Training data missing '{LABEL_COLUMN}' column.")

if LABEL_COLUMN not in val_df.columns:
    raise ValueError(f"Validation data missing '{LABEL_COLUMN}' column.")

print(f"Label column: {LABEL_COLUMN}")

train_labels_raw = train_df[LABEL_COLUMN].astype(str)
val_labels_raw = val_df[LABEL_COLUMN].astype(str)

print(f"Training samples   : {len(train_df):,}")
print(f"Validation samples : {len(val_df):,}")

print(f"Training classes   : {sorted(train_labels_raw.unique().tolist())}")
print(f"Validation classes : {sorted(val_labels_raw.unique().tolist())}")


# =============================================================================
# FEATURE VERIFICATION
# =============================================================================

print_section("FEATURE VERIFICATION")

train_features = [c for c in train_df.columns if c != LABEL_COLUMN]
val_features = [c for c in val_df.columns if c != LABEL_COLUMN]

if len(train_features) != 30:
    raise ValueError(
        f"Expected 30 training features, found {len(train_features)}."
    )

if len(val_features) != 30:
    raise ValueError(
        f"Expected 30 validation features, found {len(val_features)}."
    )

if train_features != val_features:
    missing_in_val = [c for c in train_features if c not in val_features]
    extra_in_val = [c for c in val_features if c not in train_features]

    raise ValueError(
        "Training/validation feature mismatch.\n"
        f"Missing in validation: {missing_in_val}\n"
        f"Extra in validation: {extra_in_val}"
    )

FEATURES = train_features

print(f"Training feature count   : {len(train_features)}")
print(f"Validation feature count : {len(val_features)}")
print(f"Feature count : {len(FEATURES)}")
print("Feature order : VERIFIED")


# =============================================================================
# LABEL ENCODING
# =============================================================================

print_section("LABEL ENCODING")

print("Fixed project class mapping:")
for class_id, label in ID_TO_LABEL.items():
    print(f"  {class_id:2d} -> {label}")

y_train = encode_labels(train_labels_raw, "training")
y_val = encode_labels(val_labels_raw, "validation")

expected_ids = np.arange(len(CLASS_NAMES))

train_ids = np.sort(y_train.unique())
val_ids = np.sort(y_val.unique())

print()
print(f"Training encoded classes   : {train_ids.tolist()}")
print(f"Validation encoded classes : {val_ids.tolist()}")

if not np.array_equal(train_ids, expected_ids):
    raise ValueError(
        f"Unexpected encoded training class IDs.\n"
        f"Expected: {expected_ids.tolist()}\n"
        f"Found   : {train_ids.tolist()}"
    )

if not np.array_equal(val_ids, expected_ids):
    raise ValueError(
        f"Unexpected encoded validation class IDs.\n"
        f"Expected: {expected_ids.tolist()}\n"
        f"Found   : {val_ids.tolist()}"
    )

print("Training class IDs   : PASS")
print("Validation class IDs : PASS")


# =============================================================================
# PREPARE FEATURES
# =============================================================================

print_section("PREPARING FEATURE MATRICES")

X_train = train_df[FEATURES]
X_val = val_df[FEATURES]

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape  : {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape  : {y_val.shape}")


# =============================================================================
# CONTROLLED LIGHTGBM CONFIGURATION
# =============================================================================

print_section("CONTROLLED LIGHTGBM CONFIGURATION")

# A single controlled configuration is used.
# No validation data is used for hyperparameter optimization.
# No test-set data is used.

params = {
    "objective": "multiclass",
    "num_class": len(CLASS_NAMES),
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

for key, value in params.items():
    print(f"{key:22s}: {value}")


# =============================================================================
# TRAIN LIGHTGBM
# =============================================================================

print_section("TRAINING CONTROLLED LIGHTGBM")

start_time = time.time()

model = LGBMClassifier(**params)

model.fit(
    X_train,
    y_train,
)

training_runtime = time.time() - start_time

print(f"Training runtime: {training_runtime:.2f} seconds")
print("Training completed successfully.")


# =============================================================================
# MODEL STRUCTURAL VALIDATION
# =============================================================================

print_section("MODEL STRUCTURAL VALIDATION")

if not hasattr(model, "classes_"):
    raise ValueError("LightGBM model does not expose classes_.")

model_classes = np.asarray(model.classes_)

print(f"Model classes    : {model_classes}")
print(f"Model n_classes  : {len(model_classes)}")
print(f"Model n_features : {model.n_features_in_}")

if not np.array_equal(model_classes, expected_ids):
    raise ValueError(
        f"Unexpected LightGBM model classes.\n"
        f"Expected: {expected_ids.tolist()}\n"
        f"Found   : {model_classes.tolist()}"
    )

if model.n_features_in_ != len(FEATURES):
    raise ValueError(
        f"Unexpected model feature count. "
        f"Expected {len(FEATURES)}, found {model.n_features_in_}."
    )

print("LightGBM class count  : PASS")
print("LightGBM feature count: PASS")


# =============================================================================
# VALIDATION PREDICTIONS
# =============================================================================

print_section("GENERATING LIGHTGBM VALIDATION PREDICTIONS")

start_time = time.time()

y_pred = model.predict(X_val)
y_pred = np.asarray(y_pred).astype(np.int64).reshape(-1)

prediction_runtime = time.time() - start_time

print(f"Prediction shape: {y_pred.shape}")
print(f"Prediction runtime: {prediction_runtime:.2f} seconds")

if len(y_pred) != len(y_val):
    raise ValueError(
        f"Prediction length mismatch: {len(y_pred)} vs {len(y_val)}"
    )


# =============================================================================
# OVERALL METRICS
# =============================================================================

print_section("OVERALL LIGHTGBM VALIDATION METRICS")

accuracy = accuracy_score(y_val, y_pred)
balanced_accuracy = balanced_accuracy_score(y_val, y_pred)

macro_precision = precision_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_val,
    y_pred,
    average="weighted",
    zero_division=0,
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

print_section("CONFUSION MATRIX")

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=expected_ids,
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

print(cm_df.to_string())

cm_df.to_csv(CONFUSION_PATH)


# =============================================================================
# PER-CLASS METRICS
# =============================================================================

print_section("PER-CLASS METRICS")

report = classification_report(
    y_val,
    y_pred,
    labels=expected_ids,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

rows = []

for class_id, label in enumerate(CLASS_NAMES):
    row = report[label]

    support = int(row["support"])
    precision = float(row["precision"])
    recall = float(row["recall"])
    f1 = float(row["f1-score"])

    error_count = support - int(cm[class_id, class_id])
    error_rate = error_count / support if support > 0 else 0.0

    rows.append(
        {
            "Class_ID": class_id,
            "Label": label,
            "Support": support,
            "True_Positive": int(cm[class_id, class_id]),
            "False_Positive": int(cm[:, class_id].sum() - cm[class_id, class_id]),
            "False_Negative": int(cm[class_id, :].sum() - cm[class_id, class_id]),
            "Error_Count": error_count,
            "Error_Rate": error_rate,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    )

per_class_df = pd.DataFrame(rows)

per_class_df = per_class_df.sort_values(
    by=["F1", "Support"],
    ascending=[True, True],
).reset_index(drop=True)

per_class_df["Difficulty_Rank"] = np.arange(1, len(per_class_df) + 1)

print(per_class_df.to_string(index=False))

per_class_df.to_csv(PER_CLASS_PATH, index=False)


# =============================================================================
# DIFFICULT CLASS ANALYSIS
# =============================================================================

print_section("DIFFICULT CLASS ANALYSIS")

difficult_df = per_class_df.head(5).copy()

print(difficult_df[
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
].to_string(index=False))


# =============================================================================
# INFILTRATION ANALYSIS
# =============================================================================

print_section("INFILTRATION ANALYSIS")

infiltration_id = LABEL_TO_ID["Infiltration"]
infiltration_row = per_class_df[
    per_class_df["Class_ID"] == infiltration_id
].iloc[0]

infiltration_support = int(infiltration_row["Support"])
infiltration_tp = int(infiltration_row["True_Positive"])
infiltration_fn = int(infiltration_row["False_Negative"])
infiltration_fp = int(infiltration_row["False_Positive"])

infiltration_recall = float(infiltration_row["Recall"])
infiltration_precision = float(infiltration_row["Precision"])
infiltration_f1 = float(infiltration_row["F1"])

print(f"Infiltration support   : {infiltration_support:,}")
print(f"Infiltration TP        : {infiltration_tp:,}")
print(f"Infiltration FN        : {infiltration_fn:,}")
print(f"Infiltration FP        : {infiltration_fp:,}")
print(f"Infiltration Recall    : {infiltration_recall:.6f}")
print(f"Infiltration Precision : {infiltration_precision:.6f}")
print(f"Infiltration F1        : {infiltration_f1:.6f}")


# =============================================================================
# INFILTRATION PREDICTION DISTRIBUTION
# =============================================================================

infiltration_mask = y_val == infiltration_id

infiltration_predictions = y_pred[infiltration_mask]

infiltration_distribution = (
    pd.Series(infiltration_predictions)
    .value_counts()
    .sort_index()
)

infiltration_rows = []

for predicted_id, count in infiltration_distribution.items():
    infiltration_rows.append(
        {
            "Actual_Class": "Infiltration",
            "Actual_Class_ID": infiltration_id,
            "Predicted_Class_ID": int(predicted_id),
            "Predicted_Label": ID_TO_LABEL[int(predicted_id)],
            "Count": int(count),
            "Percentage_of_Infiltration": (
                int(count) / infiltration_support
            ),
        }
    )

infiltration_df = pd.DataFrame(infiltration_rows)

if not infiltration_df.empty:
    print()
    print(infiltration_df.to_string(index=False))

infiltration_output_path = (
    OUTPUT_DIR / "step35_lightgbm_infiltration_analysis.csv"
)

infiltration_df.to_csv(infiltration_output_path, index=False)


# =============================================================================
# CONFUSION PAIR ANALYSIS
# =============================================================================

print_section("TOP CONFUSION PAIRS")

confusion_rows = []

for actual_id in expected_ids:
    for predicted_id in expected_ids:
        count = int(cm[actual_id, predicted_id])

        if actual_id == predicted_id or count == 0:
            continue

        support = int(cm[actual_id, :].sum())

        confusion_rows.append(
            {
                "Actual_Class_ID": int(actual_id),
                "Actual_Label": ID_TO_LABEL[int(actual_id)],
                "Predicted_Class_ID": int(predicted_id),
                "Predicted_Label": ID_TO_LABEL[int(predicted_id)],
                "Confusion_Count": count,
                "Percentage_of_Actual_Class": (
                    count / support if support > 0 else 0.0
                ),
            }
        )

confusion_pairs_df = pd.DataFrame(confusion_rows)

if not confusion_pairs_df.empty:
    confusion_pairs_df = confusion_pairs_df.sort_values(
        by="Confusion_Count",
        ascending=False,
    ).reset_index(drop=True)

    print(confusion_pairs_df.head(20).to_string(index=False))
else:
    print("No classification confusion pairs found.")

confusion_pairs_path = (
    OUTPUT_DIR / "step35_lightgbm_confusion_pairs.csv"
)

confusion_pairs_df.to_csv(confusion_pairs_path, index=False)


# =============================================================================
# ERROR CONCENTRATION
# =============================================================================

print_section("ERROR CONCENTRATION ANALYSIS")

total_samples = len(y_val)
correct_predictions = int(np.sum(y_val == y_pred))
total_errors = total_samples - correct_predictions

top_20_error_count = (
    int(confusion_pairs_df.head(20)["Confusion_Count"].sum())
    if not confusion_pairs_df.empty
    else 0
)

error_concentration = (
    top_20_error_count / total_errors
    if total_errors > 0
    else 0.0
)

print(f"Total validation samples : {total_samples:,}")
print(f"Correct predictions      : {correct_predictions:,}")
print(f"Total classification errors: {total_errors:,}")
print(f"Top 20 confusion errors  : {top_20_error_count:,}")
print(f"Top 20 error concentration: {error_concentration:.6f}")


# =============================================================================
# METRICS TABLE
# =============================================================================

metrics_df = pd.DataFrame(
    [
        {
            "Model": "Controlled LightGBM",
            "Accuracy": accuracy,
            "Balanced_Accuracy": balanced_accuracy,
            "Macro_Precision": macro_precision,
            "Macro_Recall": macro_recall,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1,
        }
    ]
)

metrics_df.to_csv(METRICS_PATH, index=False)


# =============================================================================
# MODEL SAVE
# =============================================================================

print_section("SAVING LIGHTGBM MODEL")

joblib.dump(model, MODEL_PATH)

print(f"Model saved: {MODEL_PATH}")


# =============================================================================
# SUMMARY
# =============================================================================

summary = {
    "step": 35,
    "experiment": "Controlled LightGBM Experiment",
    "project": "Behaviour Based Network Attack Detection System using Machine Learning",
    "training_file": str(TRAIN_PATH),
    "validation_file": str(VALIDATION_PATH),
    "test_set_used": False,
    "hyperparameter_optimization": False,
    "training_samples": int(len(train_df)),
    "validation_samples": int(len(val_df)),
    "feature_count": int(len(FEATURES)),
    "class_count": int(len(CLASS_NAMES)),
    "class_mapping": {
        str(class_id): label
        for class_id, label in ID_TO_LABEL.items()
    },
    "parameters": params,
    "training_runtime_seconds": training_runtime,
    "prediction_runtime_seconds": prediction_runtime,
    "metrics": {
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
    },
    "infiltration": {
        "support": infiltration_support,
        "true_positive": infiltration_tp,
        "false_negative": infiltration_fn,
        "false_positive": infiltration_fp,
        "precision": infiltration_precision,
        "recall": infiltration_recall,
        "f1": infiltration_f1,
    },
    "total_errors": int(total_errors),
    "top_20_error_concentration": error_concentration,
    "difficult_classes": difficult_df[
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
    ].to_dict(orient="records"),
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)


# =============================================================================
# PROTOCOL
# =============================================================================

protocol = f"""
STEP 35 - CONTROLLED LIGHTGBM EXPERIMENT
=========================================

Research Project:
Behaviour Based Network Attack Detection System using Machine Learning

Purpose:
Evaluate a third heterogeneous tree-based learner (LightGBM) under the
same validation protocol used for the existing Random Forest and XGBoost
models.

Data:
Training:
{TRAIN_PATH}

Validation:
{VALIDATION_PATH}

Test data:
NOT USED.

Training samples:
{len(train_df):,}

Validation samples:
{len(val_df):,}

Features:
30 selected features.

Classes:
14 CIC-IDS2018 multiclass labels.

Label encoding:
The source CSV files contain string class labels.
They are deterministically mapped to the project's fixed IDs:

{json.dumps({str(k): v for k, v in ID_TO_LABEL.items()}, indent=2)}

Balancing:
The already prepared B1 undersampled training dataset was used.
No additional balancing was performed in Step 35.

Hyperparameter optimization:
NOT PERFORMED.

Validation leakage:
Validation data was not used for model fitting or hyperparameter search.

Test leakage:
No test-set data was used.

Controlled LightGBM parameters:
{json.dumps(params, indent=2)}

Validation metrics:
Accuracy          : {accuracy:.6f}
Balanced Accuracy : {balanced_accuracy:.6f}
Macro Precision   : {macro_precision:.6f}
Macro Recall      : {macro_recall:.6f}
Macro F1          : {macro_f1:.6f}
Weighted F1       : {weighted_f1:.6f}

Infiltration:
Support           : {infiltration_support:,}
Precision         : {infiltration_precision:.6f}
Recall            : {infiltration_recall:.6f}
F1                : {infiltration_f1:.6f}

Total errors:
{total_errors:,}

Top 20 error concentration:
{error_concentration:.6f}

Research interpretation:
LightGBM is evaluated as a third heterogeneous tabular learner.
Its Macro F1 and difficult-class behaviour will be compared against the
tuned Random Forest, tuned XGBoost, and previously evaluated probability
fusion before the final ensemble decision.

Important methodological note:
The class-label issue encountered in the initial implementation was corrected.
The CSV labels are strings, while the model internally operates on fixed
numeric class IDs 0-13. Both training and validation labels are therefore
explicitly encoded using the same project-wide mapping before fitting and
evaluation.

No additional training beyond this controlled LightGBM experiment was used.
No test-set data was used.
"""

with open(PROTOCOL_PATH, "w", encoding="utf-8") as f:
    f.write(protocol.strip() + "\n")


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print_section("STEP 35 COMPLETED")

print(f"LightGBM Macro F1          : {macro_f1:.6f}")
print(f"LightGBM Balanced Accuracy : {balanced_accuracy:.6f}")
print(f"LightGBM Macro Recall      : {macro_recall:.6f}")
print(f"LightGBM Macro Precision   : {macro_precision:.6f}")
print(f"LightGBM Weighted F1       : {weighted_f1:.6f}")

print()
print("Infiltration:")
print(f"  F1        = {infiltration_f1:.6f}")
print(f"  Recall    = {infiltration_recall:.6f}")
print(f"  Precision = {infiltration_precision:.6f}")

print()
print("Top difficult classes:")

for _, row in difficult_df.iterrows():
    print(
        f"  {int(row['Difficulty_Rank'])}. "
        f"{row['Label']} | "
        f"F1={row['F1']:.6f} | "
        f"Support={int(row['Support']):,}"
    )

print()
print("Top confusion pairs:")

if not confusion_pairs_df.empty:
    for _, row in confusion_pairs_df.head(5).iterrows():
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
print(f"  - {MODEL_PATH.name}")
print(f"  - {METRICS_PATH.name}")
print(f"  - {PER_CLASS_PATH.name}")
print(f"  - {CONFUSION_PATH.name}")
print(f"  - {confusion_pairs_path.name}")
print(f"  - {infiltration_output_path.name}")
print(f"  - {SUMMARY_PATH.name}")
print(f"  - {PROTOCOL_PATH.name}")

print()
print("No test-set data was used.")
print("No hyperparameter optimization was performed.")
print("Training used only the B1 undersampled training dataset.")
print("Validation was used only for final Step 35 evaluation.")

print("=" * 80)
print("STEP 35 FINISHED SUCCESSFULLY.")
print("=" * 80)