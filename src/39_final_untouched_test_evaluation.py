# =============================================================================
# STEP 39 — FINAL UNTOUCHED TEST EVALUATION
# =============================================================================
#
# Project:
# Behaviour Based Network Attack Detection System using Machine Learning
#
# Purpose:
# Final unbiased evaluation of the frozen primary model on the completely
# untouched CIC-IDS2018 test set.
#
# IMPORTANT RESEARCH-INTEGRITY RULES:
# - Test set is used ONLY in this step and later reporting steps.
# - No training is performed.
# - No hyperparameter optimization is performed.
# - No model selection is performed using test performance.
# - The final candidate was frozen after Step 38.
# - The selected model is the Tuned Random Forest.
# - No XGBoost fusion is used because Step 37/38 showed no improvement.
# - Feature order must remain unchanged.
# - Class mapping must remain unchanged.
#
# Primary metrics:
#   Macro F1
#   Balanced Accuracy
#   Macro Recall
#   Macro Precision
#
# Additional metrics:
#   Weighted F1
#   Accuracy
#   ROC-AUC OvR Macro
#   ROC-AUC OvR Weighted
#   PR-AUC / Average Precision Macro
#   PR-AUC / Average Precision Weighted
#   Per-class Precision / Recall / F1
#   Per-class FPR / FNR
#   Confusion Matrix
#
# Output:
# D:\network_attack_detection\experiments\
#     step39_final_untouched_test_evaluation
#
# =============================================================================

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"D:\network_attack_detection")

# -------------------------------------------------------------------------
# Frozen final model selected after Step 38
# -------------------------------------------------------------------------
RF_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "step21_rf_tuning"
    / "step21_best_random_forest.joblib"
)

# -------------------------------------------------------------------------
# Untouched test dataset
# -------------------------------------------------------------------------
TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "CICIDS2018_test_selected.csv"
)

# -------------------------------------------------------------------------
# Step 39 output directory
# -------------------------------------------------------------------------
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step39_final_untouched_test_evaluation"
)

# -------------------------------------------------------------------------
# Expected project structure
# -------------------------------------------------------------------------
LABEL_COLUMN = "Label"
EXPECTED_FEATURE_COUNT = 30
EXPECTED_CLASS_COUNT = 14
RANDOM_STATE = 42


# =============================================================================
# 2. FIXED PROJECT CLASS MAPPING
# =============================================================================
#
# This mapping MUST NOT change.
#
# CSV labels are strings.
# Models may internally expose either string or numeric class labels.
#
# Project convention:
#   0 -> BENIGN
#   1 -> Bot
#   ...
#   13 -> SSH-Bruteforce
#
# =============================================================================

CLASS_NAMES: List[str] = [
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

CLASS_TO_ID: Dict[str, int] = {
    class_name: idx for idx, class_name in enumerate(CLASS_NAMES)
}

ID_TO_CLASS: Dict[int, str] = {
    idx: class_name for idx, class_name in enumerate(CLASS_NAMES)
}

EXPECTED_CLASS_IDS = np.arange(EXPECTED_CLASS_COUNT)


# =============================================================================
# 3. UTILITY FUNCTIONS
# =============================================================================

def print_section(title: str) -> None:
    print()
    print("-" * 88)
    print(title)
    print("-" * 88)


def fatal(message: str) -> None:
    raise RuntimeError(
        "\n[STEP 39 FATAL ERROR]\n"
        f"{message}\n"
    )


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        fatal(
            f"{description} does not exist:\n"
            f"{path}"
        )


def json_safe(value):
    """
    Convert NumPy/Pandas values into JSON-safe Python values.
    """
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=json_safe,
        )


def calculate_binary_class_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_id: int,
) -> Tuple[int, int, int, int, float, float]:
    """
    One-vs-rest confusion statistics for one class.

    Returns:
        TP, FN, FP, TN, FPR, FNR
    """

    true_positive = int(
        np.sum((y_true == class_id) & (y_pred == class_id))
    )

    false_negative = int(
        np.sum((y_true == class_id) & (y_pred != class_id))
    )

    false_positive = int(
        np.sum((y_true != class_id) & (y_pred == class_id))
    )

    true_negative = int(
        np.sum((y_true != class_id) & (y_pred != class_id))
    )

    negative_count = true_negative + false_positive
    positive_count = true_positive + false_negative

    if negative_count > 0:
        fpr = false_positive / negative_count
    else:
        fpr = np.nan

    if positive_count > 0:
        fnr = false_negative / positive_count
    else:
        fnr = np.nan

    return (
        true_positive,
        false_negative,
        false_positive,
        true_negative,
        float(fpr),
        float(fnr),
    )


def verify_feature_columns(
    dataframe: pd.DataFrame,
) -> List[str]:
    """
    Verify and return the exact feature order from the test dataframe.

    Feature selection was already completed in earlier steps.
    Therefore this function does NOT perform feature selection.
    """

    if LABEL_COLUMN not in dataframe.columns:
        fatal(
            f"Required label column '{LABEL_COLUMN}' was not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    feature_columns = [
        col for col in dataframe.columns
        if col != LABEL_COLUMN
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        fatal(
            "Unexpected test feature count.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found: {len(feature_columns)}"
        )

    return feature_columns


def encode_labels(
    labels: pd.Series,
) -> np.ndarray:
    """
    Encode string labels using the fixed project mapping.
    """

    observed_labels = sorted(
        labels.astype(str).unique().tolist()
    )

    unknown_labels = [
        label
        for label in observed_labels
        if label not in CLASS_TO_ID
    ]

    if unknown_labels:
        fatal(
            "Unknown test-set labels detected.\n"
            f"Unknown labels: {unknown_labels}\n"
            f"Expected labels: {CLASS_NAMES}"
        )

    encoded = labels.astype(str).map(CLASS_TO_ID)

    if encoded.isna().any():
        fatal(
            "Label encoding produced NaN values."
        )

    y = encoded.to_numpy(dtype=np.int64)

    observed_ids = np.unique(y)

    print(f"Observed class IDs: {observed_ids.tolist()}")

    missing_ids = [
        int(class_id)
        for class_id in EXPECTED_CLASS_IDS
        if class_id not in observed_ids
    ]

    if missing_ids:
        print(
            "WARNING: The untouched test set does not contain "
            f"the following expected class IDs: {missing_ids}"
        )
    else:
        print("All 14 project classes present: VERIFIED")

    return y


def align_model_probability_matrix(
    model,
    probabilities: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """
    Align probability columns to the fixed project class ordering.

    RF from Step 21 exposes string class labels.
    The project mapping converts those labels to numeric IDs.

    This prevents silent class-column misalignment.
    """

    if probabilities.ndim != 2:
        fatal(
            f"{model_name} probability output must be 2-dimensional.\n"
            f"Found shape: {probabilities.shape}"
        )

    if probabilities.shape[1] != EXPECTED_CLASS_COUNT:
        fatal(
            f"{model_name} probability matrix has unexpected "
            "number of classes.\n"
            f"Expected: {EXPECTED_CLASS_COUNT}\n"
            f"Found: {probabilities.shape[1]}"
        )

    model_classes = getattr(model, "classes_", None)

    if model_classes is None:
        fatal(
            f"{model_name} does not expose classes_. "
            "Cannot safely verify probability-column ordering."
        )

    model_classes_list = list(model_classes)

    # ---------------------------------------------------------------------
    # Case 1: Random Forest exposes original string labels.
    # ---------------------------------------------------------------------
    if all(isinstance(value, str) for value in model_classes_list):

        mapped_ids = []

        for label in model_classes_list:
            if label not in CLASS_TO_ID:
                fatal(
                    f"{model_name} contains an unknown class label:\n"
                    f"{label}"
                )

            mapped_ids.append(CLASS_TO_ID[label])

        mapped_ids = np.asarray(mapped_ids, dtype=np.int64)

        if not np.array_equal(
            np.sort(mapped_ids),
            EXPECTED_CLASS_IDS,
        ):
            fatal(
                f"{model_name} class set does not match "
                "the fixed project class mapping.\n"
                f"Mapped IDs: {mapped_ids.tolist()}\n"
                f"Expected: {EXPECTED_CLASS_IDS.tolist()}"
            )

        # Reorder columns from model order to fixed 0...13 order.
        aligned = np.zeros_like(probabilities, dtype=np.float64)

        for source_column, class_id in enumerate(mapped_ids):
            aligned[:, class_id] = probabilities[:, source_column]

        return aligned

    # ---------------------------------------------------------------------
    # Case 2: Model exposes numeric IDs.
    # ---------------------------------------------------------------------
    try:
        numeric_classes = np.asarray(
            model_classes,
            dtype=np.int64,
        )
    except Exception:
        fatal(
            f"{model_name} class labels could not be interpreted safely."
        )

    if not np.array_equal(
        np.sort(numeric_classes),
        EXPECTED_CLASS_IDS,
    ):
        fatal(
            f"{model_name} numeric class ordering/set does not match "
            "the fixed project mapping.\n"
            f"Found: {numeric_classes.tolist()}\n"
            f"Expected: {EXPECTED_CLASS_IDS.tolist()}"
        )

    aligned = np.zeros_like(probabilities, dtype=np.float64)

    for source_column, class_id in enumerate(numeric_classes):
        aligned[:, class_id] = probabilities[:, source_column]

    return aligned


# =============================================================================
# 4. MAIN
# =============================================================================

def main() -> None:

    start_time = time.perf_counter()

    print("=" * 88)
    print("STEP 39 — FINAL UNTOUCHED TEST EVALUATION")
    print("=" * 88)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Untouched test dataset:")
    print(TEST_DATA_PATH)

    print()
    print("Frozen final candidate:")
    print("Tuned Random Forest")

    print()
    print("Random Forest model:")
    print(RF_MODEL_PATH)

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    print()
    print("-" * 88)
    print("RESEARCH INTEGRITY")
    print("-" * 88)
    print("Test set used: YES — FIRST FINAL TEST EVALUATION")
    print("Training performed: NO")
    print("Hyperparameter optimization: NO")
    print("Model selection using test set: NO")
    print("Validation tuning after test evaluation: NO")
    print("Final candidate frozen before test evaluation: YES")
    print("Primary metric: Macro F1")
    print("Secondary primary metric: Balanced Accuracy")
    print("=" * 88)

    # ---------------------------------------------------------------------
    # STEP 1 — Input validation
    # ---------------------------------------------------------------------

    print_section(
        "[1/12] Checking required input files..."
    )

    ensure_exists(
        TEST_DATA_PATH,
        "Untouched CIC-IDS2018 test dataset",
    )

    ensure_exists(
        RF_MODEL_PATH,
        "Frozen Random Forest model",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Required inputs: VERIFIED")
    print(f"Output directory: {OUTPUT_DIR}")

    # ---------------------------------------------------------------------
    # STEP 2 — Load untouched test data
    # ---------------------------------------------------------------------

    print_section(
        "[2/12] Loading untouched CIC-IDS2018 test dataset..."
    )

    test_df = pd.read_csv(TEST_DATA_PATH)

    print(f"Test shape: {test_df.shape}")

    if LABEL_COLUMN not in test_df.columns:
        fatal(
            f"'{LABEL_COLUMN}' column is missing from test dataset."
        )

    # ---------------------------------------------------------------------
    # STEP 3 — Verify test feature structure
    # ---------------------------------------------------------------------

    print_section(
        "[3/12] Verifying test feature structure..."
    )

    feature_columns = verify_feature_columns(
        test_df
    )

    print(
        f"Feature count: {len(feature_columns)} VERIFIED"
    )

    print("Label column: VERIFIED")

    # ---------------------------------------------------------------------
    # STEP 4 — Encode test labels
    # ---------------------------------------------------------------------

    print_section(
        "[4/12] Encoding untouched test labels..."
    )

    y_test = encode_labels(
        test_df[LABEL_COLUMN]
    )

    print(
        f"Encoded test label vector shape: {y_test.shape}"
    )

    # ---------------------------------------------------------------------
    # STEP 5 — Prepare feature matrix
    # ---------------------------------------------------------------------

    print_section(
        "[5/12] Preparing untouched test feature matrix..."
    )

    X_test = test_df[
        feature_columns
    ]

    # ---------------------------------------------------------------------
    # Explicit numerical validation
    # ---------------------------------------------------------------------

    if X_test.shape[1] != EXPECTED_FEATURE_COUNT:
        fatal(
            f"Unexpected feature matrix width.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found: {X_test.shape[1]}"
        )

    if X_test.isnull().any().any():
        null_count = int(
            X_test.isnull().sum().sum()
        )

        fatal(
            f"Test feature matrix contains {null_count} null values."
        )

    print(
        f"Test feature matrix: {X_test.shape} VERIFIED"
    )

    # ---------------------------------------------------------------------
    # STEP 6 — Load frozen RF model
    # ---------------------------------------------------------------------

    print_section(
        "[6/12] Loading frozen final Random Forest..."
    )

    rf_model = joblib.load(
        RF_MODEL_PATH
    )

    print(
        f"Random Forest type: {type(rf_model).__name__}"
    )

    if not hasattr(
        rf_model,
        "predict_proba",
    ):
        fatal(
            "Frozen Random Forest does not expose predict_proba()."
        )

    if not hasattr(
        rf_model,
        "predict",
    ):
        fatal(
            "Frozen Random Forest does not expose predict()."
        )

    # ---------------------------------------------------------------------
    # STEP 7 — Verify frozen model compatibility
    # ---------------------------------------------------------------------

    print_section(
        "[7/12] Verifying frozen model compatibility..."
    )

    # Feature count
    model_n_features = getattr(
        rf_model,
        "n_features_in_",
        None,
    )

    if model_n_features is not None:

        if int(model_n_features) != EXPECTED_FEATURE_COUNT:
            fatal(
                "Frozen Random Forest feature count mismatch.\n"
                f"Expected: {EXPECTED_FEATURE_COUNT}\n"
                f"Model: {model_n_features}"
            )

        print(
            f"Random Forest feature count: "
            f"{model_n_features} VERIFIED"
        )

    else:
        print(
            "Random Forest does not expose n_features_in_; "
            "feature count will be validated through prediction."
        )

    # Model classes
    model_classes = getattr(
        rf_model,
        "classes_",
        None,
    )

    if model_classes is None:
        fatal(
            "Frozen Random Forest does not expose classes_."
        )

    print(
        "Random Forest model classes:"
    )
    print(
        np.asarray(model_classes)
    )

    # ---------------------------------------------------------------------
    # STEP 8 — Generate final test predictions/probabilities
    # ---------------------------------------------------------------------

    print_section(
        "[8/12] Generating final untouched-test predictions..."
    )

    prediction_start = time.perf_counter()

    print("Generating Random Forest probabilities...")

    rf_prob_raw = rf_model.predict_proba(
        X_test
    )

    print(
        f"Raw RF probability shape: "
        f"{rf_prob_raw.shape}"
    )

    rf_prob = align_model_probability_matrix(
        rf_model,
        rf_prob_raw,
        "Random Forest",
    )

    if rf_prob.shape != (
        len(test_df),
        EXPECTED_CLASS_COUNT,
    ):
        fatal(
            "Aligned RF probability matrix has unexpected shape.\n"
            f"Expected: "
            f"({len(test_df)}, {EXPECTED_CLASS_COUNT})\n"
            f"Found: {rf_prob.shape}"
        )

    print(
        f"Aligned RF probability shape: "
        f"{rf_prob.shape} VERIFIED"
    )

    # Prediction from aligned probability matrix.
    y_pred = np.argmax(
        rf_prob,
        axis=1,
    ).astype(np.int64)

    # Check probability validity.
    probability_row_sums = rf_prob.sum(
        axis=1
    )

    if not np.allclose(
        probability_row_sums,
        1.0,
        atol=1e-5,
    ):
        fatal(
            "RF probability rows do not sum to approximately 1."
        )

    print(
        "Probability normalization: VERIFIED"
    )

    prediction_runtime = (
        time.perf_counter()
        - prediction_start
    )

    print(
        f"Prediction runtime: "
        f"{prediction_runtime:.2f} seconds"
    )

    # ---------------------------------------------------------------------
    # STEP 9 — Final metrics
    # ---------------------------------------------------------------------

    print_section(
        "[9/12] Computing final untouched-test metrics..."
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred,
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        labels=EXPECTED_CLASS_IDS,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        labels=EXPECTED_CLASS_IDS,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=EXPECTED_CLASS_IDS,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        labels=EXPECTED_CLASS_IDS,
        average="weighted",
        zero_division=0,
    )

    # ---------------------------------------------------------------------
    # Multiclass ROC-AUC
    # ---------------------------------------------------------------------

    try:

        roc_auc_macro = roc_auc_score(
            y_test,
            rf_prob,
            multi_class="ovr",
            average="macro",
            labels=EXPECTED_CLASS_IDS,
        )

        roc_auc_weighted = roc_auc_score(
            y_test,
            rf_prob,
            multi_class="ovr",
            average="weighted",
            labels=EXPECTED_CLASS_IDS,
        )

    except ValueError as exc:

        print(
            "WARNING: Multiclass ROC-AUC could not be computed "
            f"for all classes: {exc}"
        )

        roc_auc_macro = np.nan
        roc_auc_weighted = np.nan

    # ---------------------------------------------------------------------
    # Multiclass PR-AUC / Average Precision
    # ---------------------------------------------------------------------
    #
    # One-vs-rest one-hot representation.
    #
    # This is particularly useful for imbalanced multiclass IDS evaluation.
    # ---------------------------------------------------------------------

    y_test_one_hot = np.zeros(
        (
            len(y_test),
            EXPECTED_CLASS_COUNT,
        ),
        dtype=np.int8,
    )

    y_test_one_hot[
        np.arange(len(y_test)),
        y_test,
    ] = 1

    try:

        ap_per_class = average_precision_score(
            y_test_one_hot,
            rf_prob,
            average=None,
        )

        pr_auc_macro = average_precision_score(
            y_test_one_hot,
            rf_prob,
            average="macro",
        )

        pr_auc_weighted = average_precision_score(
            y_test_one_hot,
            rf_prob,
            average="weighted",
        )

    except ValueError as exc:

        print(
            "WARNING: PR-AUC/AP could not be computed: "
            f"{exc}"
        )

        ap_per_class = np.full(
            EXPECTED_CLASS_COUNT,
            np.nan,
        )

        pr_auc_macro = np.nan
        pr_auc_weighted = np.nan

    print()
    print("FINAL TUNED RANDOM FOREST — UNTOUCHED TEST")

    print(
        f"  Accuracy                        : "
        f"{accuracy:.6f}"
    )

    print(
        f"  Balanced Accuracy               : "
        f"{balanced_accuracy:.6f}"
    )

    print(
        f"  Macro Precision                 : "
        f"{macro_precision:.6f}"
    )

    print(
        f"  Macro Recall                    : "
        f"{macro_recall:.6f}"
    )

    print(
        f"  Macro F1                        : "
        f"{macro_f1:.6f}"
    )

    print(
        f"  Weighted F1                     : "
        f"{weighted_f1:.6f}"
    )

    print(
        f"  ROC-AUC OvR Macro               : "
        f"{roc_auc_macro:.6f}"
    )

    print(
        f"  ROC-AUC OvR Weighted            : "
        f"{roc_auc_weighted:.6f}"
    )

    print(
        f"  PR-AUC / AP Macro               : "
        f"{pr_auc_macro:.6f}"
    )

    print(
        f"  PR-AUC / AP Weighted            : "
        f"{pr_auc_weighted:.6f}"
    )

    # ---------------------------------------------------------------------
    # STEP 10 — Per-class evaluation
    # ---------------------------------------------------------------------

    print_section(
        "[10/12] Computing per-class metrics, FPR, FNR and confusion matrix..."
    )

    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=EXPECTED_CLASS_IDS,
            average=None,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=EXPECTED_CLASS_IDS,
    )

    per_class_rows = []

    for class_id in EXPECTED_CLASS_IDS:

        (
            tp,
            fn,
            fp,
            tn,
            fpr,
            fnr,
        ) = calculate_binary_class_rates(
            y_test,
            y_pred,
            int(class_id),
        )

        row = {
            "class_id": int(class_id),
            "class_name": ID_TO_CLASS[int(class_id)],
            "support": int(support_per_class[class_id]),
            "precision": float(
                precision_per_class[class_id]
            ),
            "recall": float(
                recall_per_class[class_id]
            ),
            "f1": float(
                f1_per_class[class_id]
            ),
            "average_precision": float(
                ap_per_class[class_id]
            ),
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "FPR": fpr,
            "FNR": fnr,
        }

        per_class_rows.append(row)

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    print()
    print(
        per_class_df[
            [
                "class_name",
                "support",
                "precision",
                "recall",
                "f1",
                "FPR",
                "FNR",
            ]
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------------
    # Confusion matrix dataframe
    # ---------------------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    cm_df.index.name = "true_class"
    cm_df.columns.name = "predicted_class"

    # ---------------------------------------------------------------------
    # Identify difficult classes
    # ---------------------------------------------------------------------

    difficult_df = (
        per_class_df
        .sort_values(
            by=["f1", "support"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------------
    # Low-support classes
    # ---------------------------------------------------------------------

    low_support_threshold = 100

    low_support_df = (
        per_class_df[
            per_class_df["support"]
            <= low_support_threshold
        ]
        .sort_values(
            by="support",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------------
    # Substantial-support classes
    # ---------------------------------------------------------------------

    substantial_support_threshold = 1000

    substantial_support_df = (
        per_class_df[
            per_class_df["support"]
            >= substantial_support_threshold
        ]
        .sort_values(
            by="f1",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------------
    # Infiltration-specific analysis
    # ---------------------------------------------------------------------

    infiltration_id = CLASS_TO_ID["Infiltration"]

    infiltration_row = per_class_df[
        per_class_df["class_id"]
        == infiltration_id
    ].iloc[0]

    infiltration_mask_true = (
        y_test == infiltration_id
    )

    infiltration_mask_pred = (
        y_pred == infiltration_id
    )

    infiltration_true_count = int(
        infiltration_mask_true.sum()
    )

    infiltration_pred_count = int(
        infiltration_mask_pred.sum()
    )

    infiltration_to_benign = int(
        np.sum(
            (y_test == infiltration_id)
            & (y_pred == CLASS_TO_ID["BENIGN"])
        )
    )

    benign_to_infiltration = int(
        np.sum(
            (y_test == CLASS_TO_ID["BENIGN"])
            & (y_pred == infiltration_id)
        )
    )

    # ---------------------------------------------------------------------
    # Top confusion pairs
    # ---------------------------------------------------------------------

    confusion_rows = []

    for true_id in EXPECTED_CLASS_IDS:

        for pred_id in EXPECTED_CLASS_IDS:

            count = int(
                cm[
                    true_id,
                    pred_id,
                ]
            )

            if true_id == pred_id:
                continue

            if count <= 0:
                continue

            confusion_rows.append(
                {
                    "true_class_id": int(true_id),
                    "true_class": ID_TO_CLASS[int(true_id)],
                    "predicted_class_id": int(pred_id),
                    "predicted_class": ID_TO_CLASS[int(pred_id)],
                    "error_count": count,
                }
            )

    confusion_pairs_df = pd.DataFrame(
        confusion_rows
    )

    if not confusion_pairs_df.empty:

        confusion_pairs_df = (
            confusion_pairs_df
            .sort_values(
                by="error_count",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        confusion_pairs_df[
            "error_rank"
        ] = np.arange(
            1,
            len(confusion_pairs_df) + 1,
        )

    # ---------------------------------------------------------------------
    # Print key difficult-class findings
    # ---------------------------------------------------------------------

    print()
    print("KEY SUBSTANTIAL-SUPPORT DIFFICULT CLASSES")
    print(
        substantial_support_df[
            [
                "class_name",
                "support",
                "precision",
                "recall",
                "f1",
                "FPR",
                "FNR",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("LOW-SUPPORT CLASSES — INTERPRET CAUTIOUSLY")

    if low_support_df.empty:
        print("No classes below the low-support threshold.")
    else:
        print(
            low_support_df[
                [
                    "class_name",
                    "support",
                    "precision",
                    "recall",
                    "f1",
                    "FPR",
                    "FNR",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print("INFILTRATION")
    print(
        f"  Support                       : "
        f"{int(infiltration_row['support'])}"
    )

    print(
        f"  Predicted as Infiltration     : "
        f"{infiltration_pred_count}"
    )

    print(
        f"  Actual Infiltration           : "
        f"{infiltration_true_count}"
    )

    print(
        f"  TP                            : "
        f"{int(infiltration_row['TP'])}"
    )

    print(
        f"  FN                            : "
        f"{int(infiltration_row['FN'])}"
    )

    print(
        f"  FP                            : "
        f"{int(infiltration_row['FP'])}"
    )

    print(
        f"  Recall                        : "
        f"{infiltration_row['recall']:.6f}"
    )

    print(
        f"  Precision                     : "
        f"{infiltration_row['precision']:.6f}"
    )

    print(
        f"  F1                            : "
        f"{infiltration_row['f1']:.6f}"
    )

    print(
        f"  FPR                           : "
        f"{infiltration_row['FPR']:.6f}"
    )

    print(
        f"  FNR                           : "
        f"{infiltration_row['FNR']:.6f}"
    )

    print(
        f"  BENIGN -> Infiltration        : "
        f"{benign_to_infiltration}"
    )

    print(
        f"  Infiltration -> BENIGN        : "
        f"{infiltration_to_benign}"
    )

    # ---------------------------------------------------------------------
    # Top confusion pairs
    # ---------------------------------------------------------------------

    print()
    print("TOP TEST-SET CONFUSION PAIRS")

    if confusion_pairs_df.empty:
        print("No misclassification pairs found.")
    else:
        print(
            confusion_pairs_df.head(20).to_string(
                index=False
            )
        )

    # ---------------------------------------------------------------------
    # STEP 11 — Validation vs final test comparison
    # ---------------------------------------------------------------------

    print_section(
        "[11/12] Comparing final untouched-test performance with Step 38 validation..."
    )

    # Step 38 frozen validation metrics.
    validation_metrics = {
        "accuracy": 0.952812,
        "balanced_accuracy": 0.790996,
        "macro_precision": 0.799326,
        "macro_recall": 0.790996,
        "macro_f1": 0.790499,
        "weighted_f1": 0.959964,
        "roc_auc_ovr_macro": 0.993759,
        "roc_auc_ovr_weighted": 0.985419,
        "pr_auc_ap_macro": 0.831159,
        "pr_auc_ap_weighted": 0.980744,
    }

    test_metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "roc_auc_ovr_macro": float(
            roc_auc_macro
        ),
        "roc_auc_ovr_weighted": float(
            roc_auc_weighted
        ),
        "pr_auc_ap_macro": float(
            pr_auc_macro
        ),
        "pr_auc_ap_weighted": float(
            pr_auc_weighted
        ),
    }

    comparison_rows = []

    for metric_name in validation_metrics:

        validation_value = validation_metrics[
            metric_name
        ]

        test_value = test_metrics[
            metric_name
        ]

        delta = (
            test_value
            - validation_value
        )

        comparison_rows.append(
            {
                "metric": metric_name,
                "validation": validation_value,
                "untouched_test": test_value,
                "test_minus_validation": delta,
                "absolute_gap": abs(delta),
            }
        )

    validation_test_comparison_df = pd.DataFrame(
        comparison_rows
    )

    print(
        validation_test_comparison_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------------
    # STEP 12 — Save all research artifacts
    # ---------------------------------------------------------------------

    print_section(
        "[12/12] Saving Step 39 final research artifacts..."
    )

    # ---------------------------------------------------------------------
    # Main metrics
    # ---------------------------------------------------------------------

    metrics_dict = {
        "step": 39,
        "step_title": (
            "FINAL UNTOUCHED TEST EVALUATION"
        ),
        "model": "Tuned Random Forest",
        "model_path": str(RF_MODEL_PATH),
        "test_dataset": str(TEST_DATA_PATH),
        "test_set_used": True,
        "training_performed": False,
        "hyperparameter_optimization_performed": False,
        "model_selection_using_test": False,
        "validation_tuning_after_test": False,
        "final_candidate_frozen_before_test": True,
        "primary_metric": "Macro F1",
        "secondary_primary_metric": (
            "Balanced Accuracy"
        ),
        "n_test_samples": int(len(test_df)),
        "n_features": int(len(feature_columns)),
        "n_classes": int(EXPECTED_CLASS_COUNT),
        "metrics": test_metrics,
        "prediction_runtime_seconds": (
            float(prediction_runtime)
        ),
    }

    save_json(
        metrics_dict,
        OUTPUT_DIR
        / "step39_final_test_metrics.json",
    )

    # ---------------------------------------------------------------------
    # Per-class metrics
    # ---------------------------------------------------------------------

    per_class_df.to_csv(
        OUTPUT_DIR
        / "step39_test_per_class_metrics.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------------

    cm_df.to_csv(
        OUTPUT_DIR
        / "step39_test_confusion_matrix.csv"
    )

    # ---------------------------------------------------------------------
    # Confusion pairs
    # ---------------------------------------------------------------------

    confusion_pairs_df.to_csv(
        OUTPUT_DIR
        / "step39_test_confusion_pairs.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Difficult classes
    # ---------------------------------------------------------------------

    difficult_df.to_csv(
        OUTPUT_DIR
        / "step39_test_difficult_classes.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Low-support classes
    # ---------------------------------------------------------------------

    low_support_df.to_csv(
        OUTPUT_DIR
        / "step39_test_low_support_classes.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Substantial-support difficult classes
    # ---------------------------------------------------------------------

    substantial_support_df.to_csv(
        OUTPUT_DIR
        / "step39_test_substantial_support_classes.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Validation vs test comparison
    # ---------------------------------------------------------------------

    validation_test_comparison_df.to_csv(
        OUTPUT_DIR
        / "step39_validation_vs_test_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Infiltration analysis
    # ---------------------------------------------------------------------

    infiltration_analysis = {
        "class_name": "Infiltration",
        "class_id": infiltration_id,
        "support": int(
            infiltration_row["support"]
        ),
        "precision": float(
            infiltration_row["precision"]
        ),
        "recall": float(
            infiltration_row["recall"]
        ),
        "f1": float(
            infiltration_row["f1"]
        ),
        "FPR": float(
            infiltration_row["FPR"]
        ),
        "FNR": float(
            infiltration_row["FNR"]
        ),
        "TP": int(
            infiltration_row["TP"]
        ),
        "FN": int(
            infiltration_row["FN"]
        ),
        "FP": int(
            infiltration_row["FP"]
        ),
        "TN": int(
            infiltration_row["TN"]
        ),
        "benign_to_infiltration": (
            benign_to_infiltration
        ),
        "infiltration_to_benign": (
            infiltration_to_benign
        ),
    }

    save_json(
        infiltration_analysis,
        OUTPUT_DIR
        / "step39_infiltration_analysis.json",
    )

    # ---------------------------------------------------------------------
    # Test prediction distribution
    # ---------------------------------------------------------------------

    prediction_distribution = []

    for class_id in EXPECTED_CLASS_IDS:

        actual_count = int(
            np.sum(y_test == class_id)
        )

        predicted_count = int(
            np.sum(y_pred == class_id)
        )

        prediction_distribution.append(
            {
                "class_id": int(class_id),
                "class_name": ID_TO_CLASS[int(class_id)],
                "actual_count": actual_count,
                "predicted_count": predicted_count,
            }
        )

    prediction_distribution_df = pd.DataFrame(
        prediction_distribution
    )

    prediction_distribution_df.to_csv(
        OUTPUT_DIR
        / "step39_test_class_distribution_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Model information
    # ---------------------------------------------------------------------

    model_info = {
        "model_type": type(rf_model).__name__,
        "model_path": str(RF_MODEL_PATH),
        "n_features_in": (
            int(rf_model.n_features_in_)
            if hasattr(rf_model, "n_features_in_")
            else None
        ),
        "n_classes": (
            int(len(rf_model.classes_))
            if hasattr(rf_model, "classes_")
            else None
        ),
        "classes_exposed_by_model": [
            json_safe(value)
            for value in rf_model.classes_
        ],
        "fixed_project_class_mapping": {
            str(k): v
            for k, v in ID_TO_CLASS.items()
        },
    }

    save_json(
        model_info,
        OUTPUT_DIR
        / "step39_frozen_model_verification.json",
    )

    # ---------------------------------------------------------------------
    # Research protocol text
    # ---------------------------------------------------------------------

    runtime = (
        time.perf_counter()
        - start_time
    )

    protocol_text = f"""
STEP 39 — FINAL UNTOUCHED TEST EVALUATION
============================================================

Project:
Behaviour Based Network Attack Detection System using Machine Learning

Primary dataset:
CIC-IDS2018

Final frozen candidate:
Tuned Random Forest

Model:
{RF_MODEL_PATH}

Test dataset:
{TEST_DATA_PATH}

Output directory:
{OUTPUT_DIR}

RESEARCH-INTEGRITY STATUS
------------------------------------------------------------
Test set used:
YES — this is the first final untouched-test evaluation.

Training performed:
NO

Hyperparameter optimization:
NO

Model selection using test data:
NO

Validation tuning after test evaluation:
NO

Final candidate frozen before test:
YES

Test data was not used in:
- RF hyperparameter optimization
- XGBoost hyperparameter optimization
- Step 37 ensemble weight evaluation
- Step 38 ensemble validation/error analysis

The final candidate was frozen after Step 38:
Tuned Random Forest

Step 37 conclusion:
RF + XGBoost probability fusion did not improve RF.

Therefore the frozen final candidate entering Step 39 was:
Tuned Random Forest

EVALUATION
------------------------------------------------------------
Test samples:
{len(test_df)}

Features:
{len(feature_columns)}

Classes:
{EXPECTED_CLASS_COUNT}

Primary metric:
Macro F1

Secondary primary metric:
Balanced Accuracy

Additional metrics:
- Macro Precision
- Macro Recall
- Weighted F1
- Accuracy
- ROC-AUC OvR Macro
- ROC-AUC OvR Weighted
- PR-AUC / Average Precision Macro
- PR-AUC / Average Precision Weighted
- Per-class Precision
- Per-class Recall
- Per-class F1
- Per-class FPR
- Per-class FNR
- Confusion Matrix

FINAL TEST RESULTS
------------------------------------------------------------
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

ROC-AUC OvR Macro:
{roc_auc_macro:.6f}

ROC-AUC OvR Weighted:
{roc_auc_weighted:.6f}

PR-AUC / AP Macro:
{pr_auc_macro:.6f}

PR-AUC / AP Weighted:
{pr_auc_weighted:.6f}

INFILTRATION
------------------------------------------------------------
Support:
{int(infiltration_row["support"])}

Precision:
{infiltration_row["precision"]:.6f}

Recall:
{infiltration_row["recall"]:.6f}

F1:
{infiltration_row["f1"]:.6f}

FPR:
{infiltration_row["FPR"]:.6f}

FNR:
{infiltration_row["FNR"]:.6f}

BENIGN -> Infiltration:
{benign_to_infiltration}

Infiltration -> BENIGN:
{infiltration_to_benign}

VALIDATION VS TEST
------------------------------------------------------------
The validation-vs-test comparison is descriptive only.

The untouched test result must not be used to retune or reselect
the final model.

Any observed validation-test performance gap should be interpreted
as an estimate of generalization behaviour on the primary dataset.

SCIENTIFIC INTERPRETATION RULE
------------------------------------------------------------
Low-support classes must be interpreted cautiously.

Substantial-support difficult classes, especially Infiltration,
should receive greater scientific attention.

The final test result is an unbiased primary-dataset estimate only
under the stated train/validation/test separation.

No claim of universal generalization or zero-day detection is made
from this test evaluation alone.

Runtime:
{runtime:.2f} seconds

TEST SET STATUS:
USED — FINAL EVALUATION ONLY

TRAINING PERFORMED:
NO

HYPERPARAMETER OPTIMIZATION:
NO
"""

    with open(
        OUTPUT_DIR
        / "step39_final_test_evaluation_protocol.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            protocol_text.strip()
            + "\n"
        )

    # ---------------------------------------------------------------------
    # Final summary JSON
    # ---------------------------------------------------------------------

    summary = {
        "step": 39,
        "status": "COMPLETED",
        "final_candidate": "Tuned Random Forest",
        "test_set_used": True,
        "training_performed": False,
        "hyperparameter_optimization": False,
        "model_selection_using_test": False,
        "metrics": test_metrics,
        "infiltration": infiltration_analysis,
        "validation_comparison": comparison_rows,
        "output_directory": str(OUTPUT_DIR),
        "runtime_seconds": float(runtime),
    }

    save_json(
        summary,
        OUTPUT_DIR
        / "step39_final_test_summary.json",
    )

    # ---------------------------------------------------------------------
    # FINAL CONSOLE OUTPUT
    # ---------------------------------------------------------------------

    print()
    print("=" * 88)
    print("STEP 39 COMPLETE")
    print("=" * 88)

    print()
    print("FINAL UNTOUCHED TEST RESULT")
    print("Tuned Random Forest")

    print()
    print(
        f"Macro F1:             {macro_f1:.6f}"
    )

    print(
        f"Balanced Accuracy:    {balanced_accuracy:.6f}"
    )

    print(
        f"Macro Precision:      {macro_precision:.6f}"
    )

    print(
        f"Macro Recall:         {macro_recall:.6f}"
    )

    print(
        f"Weighted F1:          {weighted_f1:.6f}"
    )

    print(
        f"Accuracy:             {accuracy:.6f}"
    )

    print(
        f"ROC-AUC OvR Macro:    {roc_auc_macro:.6f}"
    )

    print(
        f"PR-AUC / AP Macro:    {pr_auc_macro:.6f}"
    )

    print()
    print("INFILTRATION")
    print(
        f"Support:              "
        f"{int(infiltration_row['support'])}"
    )

    print(
        f"Precision:            "
        f"{infiltration_row['precision']:.6f}"
    )

    print(
        f"Recall:               "
        f"{infiltration_row['recall']:.6f}"
    )

    print(
        f"F1:                   "
        f"{infiltration_row['f1']:.6f}"
    )

    print(
        f"FNR:                  "
        f"{infiltration_row['FNR']:.6f}"
    )

    print()
    print("MODEL STATUS")
    print(
        "Final candidate entering Step 39: "
        "Tuned Random Forest"
    )

    print(
        "RF + XGBoost fusion: "
        "NOT selected — no validation improvement"
    )

    print()
    print("RESEARCH-INTEGRITY STATUS")
    print(
        "Test set: USED FOR FINAL EVALUATION ONLY"
    )

    print(
        "Training: NOT PERFORMED"
    )

    print(
        "Hyperparameter optimization: NOT PERFORMED"
    )

    print(
        "Post-test model selection: NOT PERFORMED"
    )

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    print()
    print(
        f"Runtime: {runtime:.2f} seconds"
    )

    print("=" * 88)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()