from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LSTM_DATA_DIR = PROJECT_ROOT / "experiments" / "step25_lstm"
MODEL_DIR = PROJECT_ROOT / "experiments" / "step26_lstm"

REPORT_DIR = PROJECT_ROOT / "experiments" / "step27_reports"
ANALYSIS_DIR = PROJECT_ROOT / "experiments" / "step27_lstm_analysis"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

X_VALIDATION_PATH = LSTM_DATA_DIR / "step25_X_validation_sequences.npy"
Y_VALIDATION_PATH = LSTM_DATA_DIR / "step25_y_validation_sequences.npy"
LABEL_ENCODER_PATH = LSTM_DATA_DIR / "step25_lstm_label_encoder.joblib"
FEATURES_PATH = LSTM_DATA_DIR / "step25_lstm_features.json"
BEST_MODEL_PATH = MODEL_DIR / "step26_best_lstm.keras"


def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def save_protocol(lines):
    path = REPORT_DIR / "step27_lstm_validation_protocol.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


print("=" * 78)
print("STEP 27 - LSTM VALIDATION & ERROR ANALYSIS")
print("=" * 78)

print(f"Project root : {PROJECT_ROOT}")
print(f"LSTM data dir: {LSTM_DATA_DIR}")
print(f"Model dir    : {MODEL_DIR}")
print(f"Report dir   : {REPORT_DIR}")
print(f"Analysis dir : {ANALYSIS_DIR}")


# =============================================================================
# STEP 1 - CHECK INPUT FILES
# =============================================================================

print_section("CHECKING INPUT FILES")

required_files = [
    X_VALIDATION_PATH,
    Y_VALIDATION_PATH,
    LABEL_ENCODER_PATH,
    FEATURES_PATH,
    BEST_MODEL_PATH,
]

for path in required_files:
    print(f"Checking: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

print()
print("All required input files found.")


# =============================================================================
# STEP 2 - LOAD VALIDATION DATA
# =============================================================================

print_section("LOADING VALIDATION DATA")

X_val = np.load(X_VALIDATION_PATH)
y_val = np.load(Y_VALIDATION_PATH)

print(f"Validation sequence shape : {X_val.shape}")
print(f"Validation label shape    : {y_val.shape}")


# =============================================================================
# STEP 3 - VALIDATE DATA
# =============================================================================

print_section("VALIDATING VALIDATION DATA")

if len(X_val) != len(y_val):
    raise ValueError("Number of validation sequences and labels do not match.")

if X_val.ndim != 3:
    raise ValueError(f"Expected 3D sequence data, got {X_val.shape}")

if np.isnan(X_val).any():
    raise ValueError("Validation sequences contain NaN values.")

if np.isinf(X_val).any():
    raise ValueError("Validation sequences contain infinite values.")

print(f"Samples     : {X_val.shape[0]}")
print(f"Sequence len: {X_val.shape[1]}")
print(f"Features    : {X_val.shape[2]}")
print("Validation data quality verified.")


# =============================================================================
# STEP 4 - LOAD LABEL ENCODER
# =============================================================================

print_section("LOADING LABEL ENCODER")

label_encoder = joblib.load(LABEL_ENCODER_PATH)

class_names = list(label_encoder.classes_)
n_classes = len(class_names)

print(f"Number of classes: {n_classes}")

for idx, label in enumerate(class_names):
    print(f"{idx:02d} -> {label}")


# =============================================================================
# STEP 5 - LOAD FEATURE INFORMATION
# =============================================================================

print_section("LOADING FEATURE INFORMATION")

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_info = json.load(f)

print("Feature information loaded.")


# =============================================================================
# STEP 6 - LOAD BEST MODEL
# =============================================================================

print_section("LOADING BEST LSTM MODEL")

model = load_model(BEST_MODEL_PATH)

print(f"Loaded model: {BEST_MODEL_PATH}")
print()
model.summary()


# =============================================================================
# STEP 7 - GENERATE PREDICTIONS
# =============================================================================

print_section("GENERATING VALIDATION PREDICTIONS")

probabilities = model.predict(
    X_val,
    batch_size=512,
    verbose=1
)

if probabilities.shape[1] != n_classes:
    raise ValueError(
        f"Model output classes ({probabilities.shape[1]}) "
        f"do not match label encoder classes ({n_classes})."
    )

y_pred = np.argmax(probabilities, axis=1)
confidence = np.max(probabilities, axis=1)

print()
print(f"Prediction matrix shape : {probabilities.shape}")
print(f"Predicted labels shape  : {y_pred.shape}")


# =============================================================================
# STEP 8 - VALIDATION METRICS
# =============================================================================

print_section("CALCULATING VALIDATION METRICS")

accuracy = accuracy_score(y_val, y_pred)

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

weighted_precision = precision_score(
    y_val,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_val,
    y_pred,
    average="weighted",
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
print(f"Weighted Precision : {weighted_precision:.6f}")
print(f"Weighted Recall    : {weighted_recall:.6f}")
print(f"Weighted F1        : {weighted_f1:.6f}")


# =============================================================================
# STEP 9 - PER-CLASS CLASSIFICATION REPORT
# =============================================================================

print_section("PER-CLASS CLASSIFICATION REPORT")

report_dict = classification_report(
    y_val,
    y_pred,
    labels=np.arange(n_classes),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

per_class_rows = []

for class_id, class_name in enumerate(class_names):
    metrics = report_dict[class_name]

    per_class_rows.append({
        "Class_ID": class_id,
        "Label": class_name,
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1-score"],
        "Support": int(metrics["support"]),
    })

    print(
        f"{class_id:02d} | "
        f"{class_name:<30} | "
        f"Precision={metrics['precision']:.6f} | "
        f"Recall={metrics['recall']:.6f} | "
        f"F1={metrics['f1-score']:.6f} | "
        f"Support={int(metrics['support'])}"
    )

per_class_df = pd.DataFrame(per_class_rows)

per_class_path = REPORT_DIR / "step27_lstm_per_class_metrics.csv"
per_class_df.to_csv(per_class_path, index=False)


# =============================================================================
# STEP 10 - CONFUSION MATRIX
# =============================================================================

print_section("GENERATING CONFUSION MATRIX")

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=np.arange(n_classes)
)

cm_path = REPORT_DIR / "step27_lstm_confusion_matrix.csv"

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(cm_path)


# =============================================================================
# STEP 11 - NORMALIZED CONFUSION MATRIX
# =============================================================================

row_totals = cm.sum(axis=1, keepdims=True)

cm_normalized = np.divide(
    cm,
    row_totals,
    out=np.zeros_like(cm, dtype=float),
    where=row_totals != 0
)

cm_normalized_path = (
    REPORT_DIR / "step27_lstm_confusion_matrix_normalized.csv"
)

cm_normalized_df = pd.DataFrame(
    cm_normalized,
    index=class_names,
    columns=class_names
)

cm_normalized_df.to_csv(cm_normalized_path)


# =============================================================================
# STEP 12 - OVERALL ERROR ANALYSIS
# =============================================================================

print_section("PERFORMING ERROR ANALYSIS")

correct_mask = y_val == y_pred
error_mask = ~correct_mask

total_samples = len(y_val)
correct_predictions = int(correct_mask.sum())
incorrect_predictions = int(error_mask.sum())

error_rate = incorrect_predictions / total_samples

print(f"Total validation samples : {total_samples}")
print(f"Correct predictions      : {correct_predictions}")
print(f"Incorrect predictions    : {incorrect_predictions}")
print(f"Overall error rate       : {error_rate:.6f}")


# =============================================================================
# STEP 13 - PER-CLASS ERROR COUNTS
# =============================================================================

error_rows = []

for class_id, class_name in enumerate(class_names):

    actual_mask = y_val == class_id
    actual_count = int(actual_mask.sum())

    correct_count = int(
        ((y_val == class_id) & (y_pred == class_id)).sum()
    )

    error_count = actual_count - correct_count

    class_error_rate = (
        error_count / actual_count
        if actual_count > 0
        else 0.0
    )

    error_rows.append({
        "Class_ID": class_id,
        "Label": class_name,
        "Support": actual_count,
        "Correct": correct_count,
        "Errors": error_count,
        "Error_Rate": class_error_rate,
    })

error_df = pd.DataFrame(error_rows)

error_df = error_df.sort_values(
    by="Error_Rate",
    ascending=False
)

error_path = REPORT_DIR / "step27_lstm_error_summary.csv"
error_df.to_csv(error_path, index=False)

print()
print("Highest class-wise error rates:")

for _, row in error_df.head(10).iterrows():
    print(
        f"{int(row['Class_ID']):02d} | "
        f"{row['Label']:<30} | "
        f"Error Rate={row['Error_Rate']:.6f} | "
        f"Errors={int(row['Errors'])}"
    )


# =============================================================================
# STEP 14 - CONFIDENCE ANALYSIS
# =============================================================================

print_section("PREDICTION CONFIDENCE ANALYSIS")

mean_confidence = float(np.mean(confidence))

correct_confidence = confidence[correct_mask]
incorrect_confidence = confidence[error_mask]

mean_correct_confidence = (
    float(np.mean(correct_confidence))
    if len(correct_confidence) > 0
    else 0.0
)

mean_incorrect_confidence = (
    float(np.mean(incorrect_confidence))
    if len(incorrect_confidence) > 0
    else 0.0
)

confidence_rows = [
    {
        "Metric": "Mean_All_Confidence",
        "Value": mean_confidence
    },
    {
        "Metric": "Mean_Correct_Confidence",
        "Value": mean_correct_confidence
    },
    {
        "Metric": "Mean_Incorrect_Confidence",
        "Value": mean_incorrect_confidence
    },
    {
        "Metric": "Minimum_Confidence",
        "Value": float(np.min(confidence))
    },
    {
        "Metric": "Maximum_Confidence",
        "Value": float(np.max(confidence))
    },
    {
        "Metric": "Median_Confidence",
        "Value": float(np.median(confidence))
    },
]

confidence_df = pd.DataFrame(confidence_rows)

confidence_path = (
    REPORT_DIR / "step27_lstm_confidence_summary.csv"
)

confidence_df.to_csv(confidence_path, index=False)

print(f"Mean confidence          : {mean_confidence:.6f}")
print(f"Correct prediction conf. : {mean_correct_confidence:.6f}")
print(f"Incorrect prediction conf: {mean_incorrect_confidence:.6f}")


# =============================================================================
# STEP 15 - VALIDATION METRICS CSV
# =============================================================================

metrics_df = pd.DataFrame([
    {"Metric": "Accuracy", "Value": accuracy},
    {"Metric": "Balanced Accuracy", "Value": balanced_accuracy},
    {"Metric": "Macro Precision", "Value": macro_precision},
    {"Metric": "Macro Recall", "Value": macro_recall},
    {"Metric": "Macro F1", "Value": macro_f1},
    {"Metric": "Weighted Precision", "Value": weighted_precision},
    {"Metric": "Weighted Recall", "Value": weighted_recall},
    {"Metric": "Weighted F1", "Value": weighted_f1},
    {"Metric": "Total Samples", "Value": total_samples},
    {"Metric": "Correct Predictions", "Value": correct_predictions},
    {"Metric": "Incorrect Predictions", "Value": incorrect_predictions},
    {"Metric": "Error Rate", "Value": error_rate},
])

metrics_path = REPORT_DIR / "step27_lstm_validation_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)


# =============================================================================
# STEP 16 - CONFUSION MATRIX FIGURE
# =============================================================================

print_section("GENERATING CONFUSION MATRIX FIGURES")

plt.figure(figsize=(14, 12))

plt.imshow(cm, interpolation="nearest")

plt.title(
    "LSTM Validation Confusion Matrix",
    fontsize=16
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    np.arange(n_classes),
    class_names,
    rotation=90
)

plt.yticks(
    np.arange(n_classes),
    class_names
)

plt.colorbar()
plt.tight_layout()

cm_figure_path = (
    ANALYSIS_DIR / "step27_lstm_confusion_matrix.png"
)

plt.savefig(
    cm_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# STEP 17 - NORMALIZED CONFUSION MATRIX FIGURE
# =============================================================================

plt.figure(figsize=(14, 12))

plt.imshow(
    cm_normalized,
    interpolation="nearest",
    vmin=0,
    vmax=1
)

plt.title(
    "LSTM Validation Normalized Confusion Matrix",
    fontsize=16
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    np.arange(n_classes),
    class_names,
    rotation=90
)

plt.yticks(
    np.arange(n_classes),
    class_names
)

plt.colorbar()
plt.tight_layout()

normalized_cm_figure_path = (
    ANALYSIS_DIR /
    "step27_lstm_confusion_matrix_normalized.png"
)

plt.savefig(
    normalized_cm_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# STEP 18 - CONFIDENCE DISTRIBUTION
# =============================================================================

plt.figure(figsize=(10, 6))

plt.hist(
    correct_confidence,
    bins=30,
    alpha=0.7,
    label="Correct Predictions"
)

if len(incorrect_confidence) > 0:
    plt.hist(
        incorrect_confidence,
        bins=30,
        alpha=0.7,
        label="Incorrect Predictions"
    )

plt.xlabel("Prediction Confidence")
plt.ylabel("Frequency")
plt.title("LSTM Prediction Confidence Distribution")
plt.legend()
plt.tight_layout()

confidence_figure_path = (
    ANALYSIS_DIR /
    "step27_lstm_confidence_distribution.png"
)

plt.savefig(
    confidence_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# STEP 19 - CLASS ERROR DISTRIBUTION
# =============================================================================

plot_error_df = error_df.sort_values(
    "Error_Rate",
    ascending=True
)

plt.figure(figsize=(12, 8))

plt.barh(
    plot_error_df["Label"],
    plot_error_df["Error_Rate"]
)

plt.xlabel("Error Rate")
plt.ylabel("Attack Class")
plt.title("LSTM Per-Class Validation Error Rate")
plt.tight_layout()

error_figure_path = (
    ANALYSIS_DIR /
    "step27_lstm_error_distribution.png"
)

plt.savefig(
    error_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# STEP 20 - JSON SUMMARY
# =============================================================================

print_section("SAVING VALIDATION SUMMARY")

summary = {
    "step": 27,
    "model": "LSTM",
    "validation_samples": total_samples,
    "sequence_length": int(X_val.shape[1]),
    "features_per_step": int(X_val.shape[2]),
    "number_of_classes": n_classes,

    "metrics": {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "error_rate": float(error_rate),
    },

    "confidence": {
        "mean": mean_confidence,
        "mean_correct": mean_correct_confidence,
        "mean_incorrect": mean_incorrect_confidence,
        "minimum": float(np.min(confidence)),
        "maximum": float(np.max(confidence)),
        "median": float(np.median(confidence)),
    },

    "best_epoch_from_step26": 14,

    "model_path": str(BEST_MODEL_PATH),

    "methodological_note": (
        "Validation results must be interpreted using macro-level "
        "metrics and per-class performance because the validation "
        "distribution is highly imbalanced."
    )
}

summary_path = (
    REPORT_DIR /
    "step27_lstm_validation_summary.json"
)

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)


# =============================================================================
# STEP 21 - PROTOCOL
# =============================================================================

protocol_lines = [
    "STEP 27 - LSTM VALIDATION & ERROR ANALYSIS",
    "",
    "Objective:",
    "Rigorous validation and error analysis of the best LSTM model",
    "selected during Step 26 training.",
    "",
    "Validation data:",
    str(X_VALIDATION_PATH),
    "",
    "Model:",
    str(BEST_MODEL_PATH),
    "",
    "Sequence length:",
    str(X_val.shape[1]),
    "",
    "Features per sequence step:",
    str(X_val.shape[2]),
    "",
    "Number of classes:",
    str(n_classes),
    "",
    "Validation samples:",
    str(total_samples),
    "",
    "Performance:",
    f"Accuracy = {accuracy:.6f}",
    f"Balanced Accuracy = {balanced_accuracy:.6f}",
    f"Macro Precision = {macro_precision:.6f}",
    f"Macro Recall = {macro_recall:.6f}",
    f"Macro F1 = {macro_f1:.6f}",
    f"Weighted F1 = {weighted_f1:.6f}",
    "",
    "Error analysis:",
    f"Correct predictions = {correct_predictions}",
    f"Incorrect predictions = {incorrect_predictions}",
    f"Overall error rate = {error_rate:.6f}",
    "",
    "Confidence analysis:",
    f"Mean confidence = {mean_confidence:.6f}",
    f"Mean correct confidence = {mean_correct_confidence:.6f}",
    f"Mean incorrect confidence = {mean_incorrect_confidence:.6f}",
    "",
    "Important interpretation:",
    "Accuracy and weighted F1 should not be interpreted in isolation",
    "because the validation set is highly class-imbalanced.",
    "Macro F1, balanced accuracy, per-class recall, per-class F1,",
    "confusion matrices, and error concentration are therefore",
    "reported for rigorous model assessment.",
    "",
    "Sequence-design consideration:",
    "The temporal validity of the sequence construction should be",
    "considered during research interpretation. Consecutive rows in",
    "a tabular network-flow dataset do not automatically guarantee",
    "that all observations belong to the same temporal/session stream.",
]

protocol_path = save_protocol(protocol_lines)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print_section("STEP 27 COMPLETED")

print("Generated reports:")
print(f"  - {metrics_path.name}")
print(f"  - {per_class_path.name}")
print(f"  - {cm_path.name}")
print(f"  - {cm_normalized_path.name}")
print(f"  - {error_path.name}")
print(f"  - {confidence_path.name}")
print(f"  - {summary_path.name}")
print(f"  - {protocol_path.name}")

print()
print("Generated analysis figures:")
print(f"  - {cm_figure_path.name}")
print(f"  - {normalized_cm_figure_path.name}")
print(f"  - {confidence_figure_path.name}")
print(f"  - {error_figure_path.name}")

print()
print("Primary validation result:")
print(f"  Accuracy          : {accuracy:.6f}")
print(f"  Balanced Accuracy : {balanced_accuracy:.6f}")
print(f"  Macro F1          : {macro_f1:.6f}")
print(f"  Weighted F1       : {weighted_f1:.6f}")

print()
print("NEXT:")
print("STEP 28 - LSTM ERROR / SEQUENCE VALIDITY ANALYSIS")