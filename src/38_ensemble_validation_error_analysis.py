# ==============================================================================
# STEP 38 — ENSEMBLE VALIDATION + ERROR ANALYSIS
# ==============================================================================
# Project:
# Behaviour Based Network Attack Detection System using Machine Learning
#
# Purpose:
#   Comprehensive validation and error analysis of the final model candidate
#   selected in Step 37.
#
# Step 37 decision:
#   Tuned Random Forest remains the optimal candidate.
#   No RF + XGBoost probability-fusion weight improved RF.
#
# This step:
#   1. Loads the existing validation dataset.
#   2. Loads the already-trained RF and XGBoost models.
#   3. Reconstructs aligned RF/XGB probabilities.
#   4. Reconstructs the best rejected fusion configuration.
#   5. Performs comprehensive validation evaluation.
#   6. Calculates:
#        - Accuracy
#        - Balanced Accuracy
#        - Macro Precision
#        - Macro Recall
#        - Macro F1
#        - Weighted F1
#        - ROC-AUC (OvR, macro/weighted/per-class)
#        - PR-AUC / Average Precision (OvR, macro/weighted/per-class)
#        - Per-class Precision / Recall / F1 / Support
#        - FPR
#        - FNR
#        - Confusion Matrix
#   7. Performs Infiltration-focused analysis.
#   8. Performs major confusion-pair analysis.
#   9. Compares RF against rejected RF+XGB fusion.
#  10. Produces research-ready CSV, JSON and TXT outputs.
#
# Research integrity:
#   - Test set: NOT USED
#   - Training: NOT PERFORMED
#   - Hyperparameter optimization: NOT PERFORMED
#   - Validation data only
#
# Environment:
#   Windows / Git Bash / MINGW64
#   Python executable:
#       ./venv/Scripts/python.exe
# ==============================================================================

from __future__ import annotations

import json
import math
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


# ==============================================================================
# 1. PROJECT CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(r"D:\network_attack_detection")

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "CICIDS2018_validation_selected.csv"
)

RF_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "step21_rf_tuning"
    / "step21_best_random_forest.joblib"
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
    / "step38_ensemble_validation_error_analysis"
)

RANDOM_STATE = 42

PRIMARY_METRIC = "macro_f1"

N_CLASSES = 14

# Fixed project class mapping.
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

LABEL_TO_ID: Dict[str, int] = {
    label: idx for idx, label in enumerate(CLASS_NAMES)
}

ID_TO_LABEL: Dict[int, str] = {
    idx: label for idx, label in enumerate(CLASS_NAMES)
}

# Best Step 37 configuration.
# Step 37 found RF=1.0 / XGB=0.0 as optimal.
BEST_RF_WEIGHT = 1.0
BEST_XGB_WEIGHT = 0.0


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def fail(message: str) -> None:
    """Fail loudly when a research assumption is violated."""
    raise RuntimeError(f"\n[STEP 38 FATAL ERROR]\n{message}\n")


def print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def print_section(title: str) -> None:
    print("\n" + "-" * 88)
    print(title)
    print("-" * 88)


def json_safe(value):
    """Convert NumPy/Pandas values into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]

    if pd.isna(value):
        return None

    return value


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=4, ensure_ascii=False)


def ensure_probability_matrix(
    probabilities: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Validate probability matrix shape and numerical integrity."""

    probabilities = np.asarray(probabilities, dtype=np.float64)

    expected_shape = (len(Y_VALIDATION), N_CLASSES)

    if probabilities.shape != expected_shape:
        fail(
            f"{model_name} probability matrix has invalid shape.\n"
            f"Expected: {expected_shape}\n"
            f"Found: {probabilities.shape}"
        )

    if not np.all(np.isfinite(probabilities)):
        fail(f"{model_name} probability matrix contains NaN or infinite values.")

    if np.any(probabilities < -1e-8):
        fail(f"{model_name} probability matrix contains negative probabilities.")

    row_sums = probabilities.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-5):
        fail(
            f"{model_name} probability rows do not sum to 1.\n"
            f"Minimum row sum: {row_sums.min()}\n"
            f"Maximum row sum: {row_sums.max()}"
        )

    return probabilities


def validate_feature_schema(
    X: pd.DataFrame,
    rf_model,
    xgb_model,
) -> None:
    """Validate feature count and feature ordering where available."""

    expected_features = X.shape[1]

    rf_count = getattr(rf_model, "n_features_in_", None)
    xgb_count = getattr(xgb_model, "n_features_in_", None)

    if rf_count is not None and int(rf_count) != expected_features:
        fail(
            f"Random Forest feature count mismatch.\n"
            f"Validation: {expected_features}\n"
            f"RF: {rf_count}"
        )

    if xgb_count is not None and int(xgb_count) != expected_features:
        fail(
            f"XGBoost feature count mismatch.\n"
            f"Validation: {expected_features}\n"
            f"XGB: {xgb_count}"
        )

    # RF feature names, if available.
    rf_feature_names = getattr(rf_model, "feature_names_in_", None)

    if rf_feature_names is not None:
        rf_feature_names = list(rf_feature_names)

        if rf_feature_names != list(X.columns):
            fail(
                "Random Forest feature order does not match validation feature order."
            )

    # XGB feature names, if available.
    xgb_feature_names = getattr(xgb_model, "feature_names_in_", None)

    if xgb_feature_names is not None:
        xgb_feature_names = list(xgb_feature_names)

        if xgb_feature_names != list(X.columns):
            fail(
                "XGBoost feature order does not match validation feature order."
            )


def encode_labels(series: pd.Series) -> np.ndarray:
    """Encode project labels using the fixed class mapping."""

    labels = series.astype(str)

    unknown = sorted(set(labels.unique()) - set(LABEL_TO_ID.keys()))

    if unknown:
        fail(
            "Validation dataset contains labels outside the fixed project mapping.\n"
            f"Unknown labels: {unknown}"
        )

    y = labels.map(LABEL_TO_ID).to_numpy(dtype=np.int64)

    observed = sorted(np.unique(y).tolist())

    expected = list(range(N_CLASSES))

    if observed != expected:
        fail(
            "Validation set does not contain exactly the expected 14 classes.\n"
            f"Expected: {expected}\n"
            f"Observed: {observed}"
        )

    return y


def extract_model_class_order(model, model_name: str) -> None:
    """
    Validate model class ordering.

    Important:
    RF was trained with string labels.
    XGBoost was trained with numeric IDs.
    """

    classes = getattr(model, "classes_", None)

    if classes is None:
        print(
            f"{model_name} does not expose classes_; "
            f"probability output shape will be validated."
        )
        return

    classes_array = np.asarray(classes)

    if model_name == "Random Forest":

        expected_strings = np.asarray(CLASS_NAMES, dtype=object)

        if classes_array.dtype.kind in {"U", "S", "O"}:

            if list(classes_array.astype(str)) != CLASS_NAMES:
                fail(
                    "Random Forest string class ordering mismatch.\n"
                    f"Expected: {CLASS_NAMES}\n"
                    f"Found: {classes_array.tolist()}"
                )

            print("Random Forest class ordering: VERIFIED (string labels)")

        elif np.issubdtype(classes_array.dtype, np.integer):

            if list(classes_array.astype(int)) != list(range(N_CLASSES)):
                fail(
                    "Random Forest numeric class ordering mismatch.\n"
                    f"Expected: {list(range(N_CLASSES))}\n"
                    f"Found: {classes_array.tolist()}"
                )

            print("Random Forest class ordering: VERIFIED (numeric IDs)")

        else:
            fail(
                "Unexpected Random Forest class representation.\n"
                f"Found dtype: {classes_array.dtype}\n"
                f"Classes: {classes_array.tolist()}"
            )

    elif model_name == "XGBoost":

        # XGB in the existing project was trained with numeric class IDs.
        if np.issubdtype(classes_array.dtype, np.number):

            numeric_classes = classes_array.astype(int).tolist()

            if numeric_classes != list(range(N_CLASSES)):
                fail(
                    "XGBoost numeric class ordering mismatch.\n"
                    f"Expected: {list(range(N_CLASSES))}\n"
                    f"Found: {numeric_classes}"
                )

            print("XGBoost class ordering: VERIFIED (numeric IDs)")

        else:

            string_classes = classes_array.astype(str).tolist()

            if string_classes != CLASS_NAMES:
                fail(
                    "XGBoost string class ordering mismatch.\n"
                    f"Expected: {CLASS_NAMES}\n"
                    f"Found: {string_classes}"
                )

            print("XGBoost class ordering: VERIFIED (string labels)")


def get_predictions(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Convert probability matrix into class predictions."""

    return np.argmax(probabilities, axis=1).astype(np.int64)


def calculate_basic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=list(range(N_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=list(range(N_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(N_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(N_CLASSES)),
                average="weighted",
                zero_division=0,
            )
        ),
    }


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
        zero_division=0,
    )

    rows = []

    for class_id in range(N_CLASSES):

        rows.append(
            {
                "class_id": class_id,
                "class_name": ID_TO_LABEL[class_id],
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
        )

    return pd.DataFrame(rows)


def calculate_fpr_fnr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
    )

    rows = []

    total_samples = cm.sum()

    for class_id in range(N_CLASSES):

        tp = int(cm[class_id, class_id])

        fn = int(cm[class_id, :].sum() - tp)

        fp = int(cm[:, class_id].sum() - tp)

        tn = int(total_samples - tp - fn - fp)

        fpr_denominator = fp + tn
        fnr_denominator = fn + tp

        fpr = (
            fp / fpr_denominator
            if fpr_denominator > 0
            else 0.0
        )

        fnr = (
            fn / fnr_denominator
            if fnr_denominator > 0
            else 0.0
        )

        rows.append(
            {
                "class_id": class_id,
                "class_name": ID_TO_LABEL[class_id],
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "FPR": float(fpr),
                "FNR": float(fnr),
            }
        )

    return pd.DataFrame(rows)


def calculate_auc_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame]:

    y_one_hot = np.zeros(
        (len(y_true), N_CLASSES),
        dtype=np.int8,
    )

    y_one_hot[
        np.arange(len(y_true)),
        y_true,
    ] = 1

    per_class_rows = []

    roc_values = []
    pr_values = []

    class_support = np.bincount(
        y_true,
        minlength=N_CLASSES,
    )

    for class_id in range(N_CLASSES):

        binary_true = y_one_hot[:, class_id]
        class_probability = probabilities[:, class_id]

        # ROC-AUC requires both positive and negative samples.
        if (
            np.unique(binary_true).size == 2
        ):
            roc_auc = roc_auc_score(
                binary_true,
                class_probability,
            )
        else:
            roc_auc = np.nan

        # Average Precision / PR-AUC.
        if binary_true.sum() > 0:
            pr_auc = average_precision_score(
                binary_true,
                class_probability,
            )
        else:
            pr_auc = np.nan

        if np.isfinite(roc_auc):
            roc_values.append(roc_auc)

        if np.isfinite(pr_auc):
            pr_values.append(pr_auc)

        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": ID_TO_LABEL[class_id],
                "support": int(class_support[class_id]),
                "roc_auc_ovr": (
                    float(roc_auc)
                    if np.isfinite(roc_auc)
                    else np.nan
                ),
                "pr_auc_average_precision": (
                    float(pr_auc)
                    if np.isfinite(pr_auc)
                    else np.nan
                ),
            }
        )

    # Global multiclass ROC-AUC.
    try:
        roc_macro = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=list(range(N_CLASSES)),
        )

        roc_weighted = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="weighted",
            labels=list(range(N_CLASSES)),
        )

    except ValueError:
        roc_macro = np.nan
        roc_weighted = np.nan

    # Macro PR-AUC.
    # Average per-class OvR AP gives a transparent multiclass macro PR-AUC.
    pr_macro = (
        float(np.mean(pr_values))
        if pr_values
        else np.nan
    )

    if pr_values:
        weights = np.asarray(
            [
                class_support[i]
                for i in range(N_CLASSES)
                if np.isfinite(
                    per_class_rows[i]["pr_auc_average_precision"]
                )
            ],
            dtype=np.float64,
        )

        values = np.asarray(pr_values, dtype=np.float64)

        if weights.sum() > 0:
            pr_weighted = float(
                np.average(values, weights=weights)
            )
        else:
            pr_weighted = np.nan
    else:
        pr_weighted = np.nan

    summary = {
        "roc_auc_ovr_macro": (
            float(roc_macro)
            if np.isfinite(roc_macro)
            else None
        ),
        "roc_auc_ovr_weighted": (
            float(roc_weighted)
            if np.isfinite(roc_weighted)
            else None
        ),
        "roc_auc_ovr_mean_per_class": (
            float(np.mean(roc_values))
            if roc_values
            else None
        ),
        "pr_auc_macro_average_precision": (
            float(pr_macro)
            if np.isfinite(pr_macro)
            else None
        ),
        "pr_auc_weighted_average_precision": (
            float(pr_weighted)
            if np.isfinite(pr_weighted)
            else None
        ),
    }

    return summary, pd.DataFrame(per_class_rows)


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    probabilities = ensure_probability_matrix(
        probabilities,
        model_name,
    )

    y_pred = get_predictions(probabilities)

    basic = calculate_basic_metrics(
        y_true,
        y_pred,
    )

    auc_summary, auc_per_class = calculate_auc_metrics(
        y_true,
        probabilities,
    )

    metrics = {
        **basic,
        **auc_summary,
    }

    per_class = calculate_per_class_metrics(
        y_true,
        y_pred,
    )

    per_class = per_class.merge(
        auc_per_class[
            [
                "class_id",
                "roc_auc_ovr",
                "pr_auc_average_precision",
            ]
        ],
        on="class_id",
        how="left",
    )

    error_df = calculate_fpr_fnr(
        y_true,
        y_pred,
    )

    per_class = per_class.merge(
        error_df[
            [
                "class_id",
                "TP",
                "TN",
                "FP",
                "FN",
                "FPR",
                "FNR",
            ]
        ],
        on="class_id",
        how="left",
    )

    per_class["model"] = model_name

    return (
        metrics,
        per_class,
        error_df,
        pd.DataFrame(
            confusion_matrix(
                y_true,
                y_pred,
                labels=list(range(N_CLASSES)),
            ),
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ),
    )


def print_metrics(
    model_name: str,
    metrics: Dict[str, float],
) -> None:

    print(f"\n{model_name}")

    ordered = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
        ("roc_auc_ovr_macro", "ROC-AUC OvR Macro"),
        ("roc_auc_ovr_weighted", "ROC-AUC OvR Weighted"),
        (
            "pr_auc_macro_average_precision",
            "PR-AUC / AP Macro",
        ),
        (
            "pr_auc_weighted_average_precision",
            "PR-AUC / AP Weighted",
        ),
    ]

    for key, label in ordered:

        value = metrics.get(key)

        if value is None or not np.isfinite(value):
            print(f"  {label:<32}: N/A")
        else:
            print(f"  {label:<32}: {value:.6f}")


def make_confusion_dataframe(
    cm: np.ndarray,
) -> pd.DataFrame:

    df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    df.index.name = "True / Predicted"

    return df


def calculate_confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_n: int = 30,
) -> pd.DataFrame:

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
    )

    rows = []

    for true_id in range(N_CLASSES):

        for pred_id in range(N_CLASSES):

            if true_id == pred_id:
                continue

            count = int(cm[true_id, pred_id])

            if count == 0:
                continue

            rows.append(
                {
                    "true_class_id": true_id,
                    "true_class": ID_TO_LABEL[true_id],
                    "predicted_class_id": pred_id,
                    "predicted_class": ID_TO_LABEL[pred_id],
                    "error_count": count,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "true_class_id",
                "true_class",
                "predicted_class_id",
                "predicted_class",
                "error_count",
            ]
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "error_count",
        ascending=False,
    ).reset_index(drop=True)

    df["error_rank"] = np.arange(1, len(df) + 1)

    return df.head(top_n)


def analyze_infiltration(
    y_true: np.ndarray,
    rf_pred: np.ndarray,
    fusion_pred: np.ndarray,
    rf_per_class: pd.DataFrame,
    fusion_per_class: pd.DataFrame,
) -> Dict:

    infiltration_id = LABEL_TO_ID["Infiltration"]

    true_infiltration = y_true == infiltration_id

    rf_pred_infiltration = (
        rf_pred == infiltration_id
    )

    fusion_pred_infiltration = (
        fusion_pred == infiltration_id
    )

    rf_tp = int(
        np.sum(
            true_infiltration
            & rf_pred_infiltration
        )
    )

    rf_fn = int(
        np.sum(
            true_infiltration
            & ~rf_pred_infiltration
        )
    )

    rf_fp = int(
        np.sum(
            ~true_infiltration
            & rf_pred_infiltration
        )
    )

    fusion_tp = int(
        np.sum(
            true_infiltration
            & fusion_pred_infiltration
        )
    )

    fusion_fn = int(
        np.sum(
            true_infiltration
            & ~fusion_pred_infiltration
        )
    )

    fusion_fp = int(
        np.sum(
            ~true_infiltration
            & fusion_pred_infiltration
        )
    )

    return {
        "class": "Infiltration",
        "class_id": infiltration_id,
        "support": int(true_infiltration.sum()),
        "rf": {
            "TP": rf_tp,
            "FN": rf_fn,
            "FP": rf_fp,
            "recall": (
                rf_tp / (rf_tp + rf_fn)
                if (rf_tp + rf_fn) > 0
                else 0.0
            ),
            "precision": (
                rf_tp / (rf_tp + rf_fp)
                if (rf_tp + rf_fp) > 0
                else 0.0
            ),
        },
        "best_step37_fusion": {
            "rf_weight": BEST_RF_WEIGHT,
            "xgb_weight": BEST_XGB_WEIGHT,
            "TP": fusion_tp,
            "FN": fusion_fn,
            "FP": fusion_fp,
            "recall": (
                fusion_tp / (fusion_tp + fusion_fn)
                if (fusion_tp + fusion_fn) > 0
                else 0.0
            ),
            "precision": (
                fusion_tp / (fusion_tp + fusion_fp)
                if (fusion_tp + fusion_fp) > 0
                else 0.0
            ),
        },
    }


def calculate_prediction_disagreement(
    y_true: np.ndarray,
    rf_pred: np.ndarray,
    xgb_pred: np.ndarray,
    fusion_pred: np.ndarray,
) -> Dict:

    rf_correct = rf_pred == y_true
    xgb_correct = xgb_pred == y_true
    fusion_correct = fusion_pred == y_true

    rf_xgb_disagreement = rf_pred != xgb_pred

    return {
        "total_samples": int(len(y_true)),
        "rf_correct": int(rf_correct.sum()),
        "xgb_correct": int(xgb_correct.sum()),
        "fusion_correct": int(fusion_correct.sum()),
        "rf_xgb_disagreement_count": int(
            rf_xgb_disagreement.sum()
        ),
        "rf_xgb_disagreement_rate": float(
            rf_xgb_disagreement.mean()
        ),
        "both_correct": int(
            (rf_correct & xgb_correct).sum()
        ),
        "rf_correct_xgb_wrong": int(
            (rf_correct & ~xgb_correct).sum()
        ),
        "xgb_correct_rf_wrong": int(
            (xgb_correct & ~rf_correct).sum()
        ),
        "both_wrong": int(
            (~rf_correct & ~xgb_correct).sum()
        ),
        "fusion_correct_rf_wrong": int(
            (fusion_correct & ~rf_correct).sum()
        ),
        "fusion_wrong_rf_correct": int(
            (~fusion_correct & rf_correct).sum()
        ),
    }


def calculate_fusion_error_changes(
    y_true: np.ndarray,
    rf_pred: np.ndarray,
    fusion_pred: np.ndarray,
) -> Dict:

    rf_correct = rf_pred == y_true
    fusion_correct = fusion_pred == y_true

    return {
        "total_samples": int(len(y_true)),
        "rf_correct_fusion_correct": int(
            (rf_correct & fusion_correct).sum()
        ),
        "rf_correct_fusion_wrong": int(
            (rf_correct & ~fusion_correct).sum()
        ),
        "rf_wrong_fusion_correct": int(
            (~rf_correct & fusion_correct).sum()
        ),
        "rf_wrong_fusion_wrong": int(
            (~rf_correct & ~fusion_correct).sum()
        ),
        "net_correct_change": int(
            fusion_correct.sum()
            - rf_correct.sum()
        ),
    }


def calculate_class_error_comparison(
    y_true: np.ndarray,
    rf_pred: np.ndarray,
    fusion_pred: np.ndarray,
) -> pd.DataFrame:

    rows = []

    for class_id in range(N_CLASSES):

        true_mask = y_true == class_id

        support = int(true_mask.sum())

        rf_recall = (
            np.sum(
                rf_pred[true_mask] == class_id
            ) / support
            if support > 0
            else 0.0
        )

        fusion_recall = (
            np.sum(
                fusion_pred[true_mask] == class_id
            ) / support
            if support > 0
            else 0.0
        )

        rows.append(
            {
                "class_id": class_id,
                "class_name": ID_TO_LABEL[class_id],
                "support": support,
                "rf_recall": float(rf_recall),
                "fusion_recall": float(fusion_recall),
                "recall_delta_fusion_minus_rf": float(
                    fusion_recall - rf_recall
                ),
            }
        )

    return pd.DataFrame(rows)


def write_protocol(
    metrics_rf: Dict,
    metrics_fusion: Dict,
    infiltration_analysis: Dict,
    prediction_analysis: Dict,
    fusion_error_analysis: Dict,
    runtime_seconds: float,
) -> None:

    path = (
        OUTPUT_DIR
        / "step38_ensemble_validation_error_analysis_protocol.txt"
    )

    with open(path, "w", encoding="utf-8") as f:

        f.write(
            "STEP 38 — ENSEMBLE VALIDATION + ERROR ANALYSIS\n"
        )
        f.write("=" * 88 + "\n\n")

        f.write(
            "PROJECT: Behaviour Based Network Attack Detection System using Machine Learning\n\n"
        )

        f.write("RESEARCH INTEGRITY\n")
        f.write("-" * 88 + "\n")
        f.write("Test set used: NO\n")
        f.write("Training performed: NO\n")
        f.write("Hyperparameter optimization performed: NO\n")
        f.write("Data used: CIC-IDS2018 validation set only\n")
        f.write("Primary model-selection metric: Macro F1\n")
        f.write(
            "Secondary primary metric: Balanced Accuracy\n"
        )
        f.write("\n")

        f.write("VALIDATION DATASET\n")
        f.write("-" * 88 + "\n")
        f.write(f"Path: {VALIDATION_PATH}\n")
        f.write(f"Rows: {len(Y_VALIDATION):,}\n")
        f.write(f"Features: {X_VALIDATION.shape[1]}\n")
        f.write(f"Classes: {N_CLASSES}\n")
        f.write("\n")

        f.write("MODEL DECISION FROM STEP 37\n")
        f.write("-" * 88 + "\n")
        f.write(
            "RF + XGBoost probability fusion was evaluated using controlled weights.\n"
        )
        f.write(
            "No fusion configuration improved the tuned Random Forest.\n"
        )
        f.write(
            "Therefore, Random Forest remains the final primary candidate.\n"
        )
        f.write("\n")

        f.write("STEP 37 BEST CONFIGURATION\n")
        f.write("-" * 88 + "\n")
        f.write(f"RF weight: {BEST_RF_WEIGHT:.2f}\n")
        f.write(f"XGB weight: {BEST_XGB_WEIGHT:.2f}\n")
        f.write(
            "Interpretation: the selected configuration is equivalent to RF-only.\n"
        )
        f.write("\n")

        f.write("RANDOM FOREST VALIDATION METRICS\n")
        f.write("-" * 88 + "\n")

        for key, value in metrics_rf.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")

        f.write("STEP 37 BEST FUSION VALIDATION METRICS\n")
        f.write("-" * 88 + "\n")

        for key, value in metrics_fusion.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")

        f.write("PREDICTION-LEVEL COMPLEMENTARITY\n")
        f.write("-" * 88 + "\n")

        for key, value in prediction_analysis.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")

        f.write("FUSION ERROR CHANGE ANALYSIS\n")
        f.write("-" * 88 + "\n")

        for key, value in fusion_error_analysis.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")

        f.write("INFILTRATION ANALYSIS\n")
        f.write("-" * 88 + "\n")

        f.write(
            json.dumps(
                json_safe(infiltration_analysis),
                indent=4,
                ensure_ascii=False,
            )
        )

        f.write("\n\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 88 + "\n")
        f.write(
            "The purpose of Step 38 is not to introduce another model or perform\n"
        )
        f.write(
            "additional training. It provides comprehensive validation evidence\n"
        )
        f.write(
            "for the model-selection decision made using the validation set.\n"
        )
        f.write(
            "Rare classes with very small support must be interpreted cautiously.\n"
        )
        f.write(
            "Substantial-support difficult classes, particularly Infiltration,\n"
        )
        f.write(
            "receive greater analytical emphasis.\n"
        )
        f.write(
            "The untouched CIC-IDS2018 test set remains reserved for Step 39.\n"
        )

        f.write("\n")
        f.write(
            f"Runtime: {runtime_seconds:.2f} seconds\n"
        )


# ==============================================================================
# 3. MAIN
# ==============================================================================

def main():

    global Y_VALIDATION
    global X_VALIDATION

    start_time = time.time()

    print_header(
        "STEP 38 — ENSEMBLE VALIDATION + ERROR ANALYSIS"
    )

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print("\nValidation dataset:")
    print(VALIDATION_PATH)

    print("\nRandom Forest model:")
    print(RF_MODEL_PATH)

    print("\nXGBoost model:")
    print(XGB_MODEL_PATH)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print_section("RESEARCH INTEGRITY")

    print("Test set used: NO")
    print("Training performed: NO")
    print("Hyperparameter optimization: NO")
    print("Validation-only analysis: YES")
    print("Primary metric: Macro F1")
    print("Secondary primary metric: Balanced Accuracy")

    # ==========================================================================
    # [1/12] CHECK FILES
    # ==========================================================================

    print_section(
        "[1/12] Checking required input files..."
    )

    required_files = [
        VALIDATION_PATH,
        RF_MODEL_PATH,
        XGB_MODEL_PATH,
    ]

    for path in required_files:

        if not path.exists():
            fail(
                f"Required input file does not exist:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Required inputs: VERIFIED")
    print(f"Output directory: {OUTPUT_DIR}")

    # ==========================================================================
    # [2/12] LOAD VALIDATION DATASET
    # ==========================================================================

    print_section(
        "[2/12] Loading validation dataset..."
    )

    df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation shape: {df.shape}"
    )

    if "Label" not in df.columns:
        fail(
            "Validation dataset does not contain the required 'Label' column."
        )

    feature_columns = [
        column
        for column in df.columns
        if column != "Label"
    ]

    if len(feature_columns) != 30:
        fail(
            "Validation dataset does not contain exactly 30 selected features.\n"
            f"Found: {len(feature_columns)}"
        )

    print("Feature count: 30 VERIFIED")

    # ==========================================================================
    # [3/12] ENCODE LABELS
    # ==========================================================================

    print_section(
        "[3/12] Encoding validation labels..."
    )

    Y_VALIDATION = encode_labels(
        df["Label"]
    )

    observed_ids = sorted(
        np.unique(Y_VALIDATION).tolist()
    )

    print(
        f"Observed class IDs: {observed_ids}"
    )

    print(
        "All 14 classes present: VERIFIED"
    )

    # ==========================================================================
    # [4/12] PREPARE FEATURES
    # ==========================================================================

    print_section(
        "[4/12] Preparing validation feature matrix..."
    )

    X_VALIDATION = df[
        feature_columns
    ].copy()

    if X_VALIDATION.isnull().any().any():
        null_count = int(
            X_VALIDATION.isnull().sum().sum()
        )

        fail(
            f"Validation feature matrix contains {null_count:,} null values."
        )

    non_numeric_columns = X_VALIDATION.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric_columns:
        fail(
            "Validation feature matrix contains non-numeric columns.\n"
            f"Columns: {non_numeric_columns}"
        )

    print(
        f"Validation feature matrix: {X_VALIDATION.shape}"
    )

    # ==========================================================================
    # [5/12] LOAD MODELS
    # ==========================================================================

    print_section(
        "[5/12] Loading existing trained models..."
    )

    rf_model = joblib.load(
        RF_MODEL_PATH
    )

    xgb_model = joblib.load(
        XGB_MODEL_PATH
    )

    print(
        f"Random Forest type: {type(rf_model).__name__}"
    )

    print(
        f"XGBoost type: {type(xgb_model).__name__}"
    )

    # ==========================================================================
    # [6/12] MODEL COMPATIBILITY
    # ==========================================================================

    print_section(
        "[6/12] Verifying model compatibility..."
    )

    validate_feature_schema(
        X_VALIDATION,
        rf_model,
        xgb_model,
    )

    print(
        "Random Forest feature count: "
        f"{getattr(rf_model, 'n_features_in_', 'N/A')} VERIFIED"
    )

    print(
        "XGBoost feature count: "
        f"{getattr(xgb_model, 'n_features_in_', 'N/A')} VERIFIED"
    )

    extract_model_class_order(
        rf_model,
        "Random Forest",
    )

    extract_model_class_order(
        xgb_model,
        "XGBoost",
    )

    # ==========================================================================
    # [7/12] GENERATE PROBABILITIES
    # ==========================================================================

    print_section(
        "[7/12] Generating aligned validation probability matrices..."
    )

    print(
        "Generating Random Forest probabilities..."
    )

    rf_probabilities = np.asarray(
        rf_model.predict_proba(
            X_VALIDATION
        ),
        dtype=np.float64,
    )

    print(
        "Generating XGBoost probabilities..."
    )

    xgb_probabilities = np.asarray(
        xgb_model.predict_proba(
            X_VALIDATION
        ),
        dtype=np.float64,
    )

    rf_probabilities = ensure_probability_matrix(
        rf_probabilities,
        "Random Forest",
    )

    xgb_probabilities = ensure_probability_matrix(
        xgb_probabilities,
        "XGBoost",
    )

    print(
        f"RF probability shape: {rf_probabilities.shape} VERIFIED"
    )

    print(
        f"XGB probability shape: {xgb_probabilities.shape} VERIFIED"
    )

    # ==========================================================================
    # [8/12] BUILD STEP 37 BEST FUSION
    # ==========================================================================

    print_section(
        "[8/12] Reconstructing Step 37 best fusion configuration..."
    )

    fusion_probabilities = (
        BEST_RF_WEIGHT * rf_probabilities
        + BEST_XGB_WEIGHT * xgb_probabilities
    )

    fusion_probabilities = ensure_probability_matrix(
        fusion_probabilities,
        "Step 37 best RF/XGB fusion",
    )

    rf_predictions = get_predictions(
        rf_probabilities
    )

    xgb_predictions = get_predictions(
        xgb_probabilities
    )

    fusion_predictions = get_predictions(
        fusion_probabilities
    )

    print(
        f"Step 37 selected configuration: "
        f"RF={BEST_RF_WEIGHT:.1f}, XGB={BEST_XGB_WEIGHT:.1f}"
    )

    print(
        "Selected configuration is equivalent to RF-only: VERIFIED"
    )

    # ==========================================================================
    # [9/12] COMPREHENSIVE MODEL EVALUATION
    # ==========================================================================

    print_section(
        "[9/12] Comprehensive validation evaluation..."
    )

    (
        rf_metrics,
        rf_per_class,
        rf_error_df,
        rf_cm,
    ) = evaluate_model(
        "Tuned Random Forest",
        Y_VALIDATION,
        rf_probabilities,
    )

    (
        fusion_metrics,
        fusion_per_class,
        fusion_error_df,
        fusion_cm,
    ) = evaluate_model(
        "Step 37 Best RF/XGB Fusion",
        Y_VALIDATION,
        fusion_probabilities,
    )

    print_metrics(
        "TUNED RANDOM FOREST",
        rf_metrics,
    )

    print_metrics(
        "STEP 37 BEST RF/XGB FUSION",
        fusion_metrics,
    )

    # ==========================================================================
    # [10/12] ERROR / COMPLEMENTARITY ANALYSIS
    # ==========================================================================

    print_section(
        "[10/12] Prediction-level complementarity and error analysis..."
    )

    prediction_analysis = calculate_prediction_disagreement(
        Y_VALIDATION,
        rf_predictions,
        xgb_predictions,
        fusion_predictions,
    )

    print(
        "\nRF vs XGB prediction-level analysis:"
    )

    print(
        f"  Total samples                  : "
        f"{prediction_analysis['total_samples']:,}"
    )

    print(
        f"  Both correct                   : "
        f"{prediction_analysis['both_correct']:,}"
    )

    print(
        f"  RF correct, XGB wrong          : "
        f"{prediction_analysis['rf_correct_xgb_wrong']:,}"
    )

    print(
        f"  XGB correct, RF wrong          : "
        f"{prediction_analysis['xgb_correct_rf_wrong']:,}"
    )

    print(
        f"  Both wrong                     : "
        f"{prediction_analysis['both_wrong']:,}"
    )

    print(
        f"  RF/XGB disagreement            : "
        f"{prediction_analysis['rf_xgb_disagreement_count']:,}"
        f" "
        f"({prediction_analysis['rf_xgb_disagreement_rate']:.4%})"
    )

    fusion_error_analysis = calculate_fusion_error_changes(
        Y_VALIDATION,
        rf_predictions,
        fusion_predictions,
    )

    print(
        "\nRF vs selected fusion error changes:"
    )

    print(
        f"  RF correct → Fusion correct    : "
        f"{fusion_error_analysis['rf_correct_fusion_correct']:,}"
    )

    print(
        f"  RF correct → Fusion wrong      : "
        f"{fusion_error_analysis['rf_correct_fusion_wrong']:,}"
    )

    print(
        f"  RF wrong → Fusion correct      : "
        f"{fusion_error_analysis['rf_wrong_fusion_correct']:,}"
    )

    print(
        f"  RF wrong → Fusion wrong        : "
        f"{fusion_error_analysis['rf_wrong_fusion_wrong']:,}"
    )

    print(
        f"  Net correct prediction change  : "
        f"{fusion_error_analysis['net_correct_change']:+,}"
    )

    # ==========================================================================
    # [11/12] INFILTRATION + DIFFICULT CLASSES
    # ==========================================================================

    print_section(
        "[11/12] Infiltration and difficult-class analysis..."
    )

    infiltration_analysis = analyze_infiltration(
        Y_VALIDATION,
        rf_predictions,
        fusion_predictions,
        rf_per_class,
        fusion_per_class,
    )

    print(
        "\nINFILTRATION"
    )

    print(
        f"  Support                       : "
        f"{infiltration_analysis['support']:,}"
    )

    print(
        f"  RF TP                         : "
        f"{infiltration_analysis['rf']['TP']:,}"
    )

    print(
        f"  RF FN                         : "
        f"{infiltration_analysis['rf']['FN']:,}"
    )

    print(
        f"  RF FP                         : "
        f"{infiltration_analysis['rf']['FP']:,}"
    )

    print(
        f"  RF Recall                     : "
        f"{infiltration_analysis['rf']['recall']:.6f}"
    )

    print(
        f"  RF Precision                  : "
        f"{infiltration_analysis['rf']['precision']:.6f}"
    )

    print(
        "\nSubstantial-support difficult classes:"
    )

    substantial_support = rf_per_class[
        rf_per_class["support"] >= 100
    ].copy()

    substantial_support = substantial_support.sort_values(
        "f1",
        ascending=True,
    )

    print(
        substantial_support[
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

    print(
        "\nLow-support classes requiring cautious interpretation:"
    )

    low_support = rf_per_class[
        rf_per_class["support"] < 100
    ].copy()

    print(
        low_support[
            [
                "class_name",
                "support",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )

    # Major confusion pairs.
    rf_confusion_pairs = calculate_confusion_pairs(
        Y_VALIDATION,
        rf_predictions,
        top_n=30,
    )

    fusion_confusion_pairs = calculate_confusion_pairs(
        Y_VALIDATION,
        fusion_predictions,
        top_n=30,
    )

    print(
        "\nTop RF confusion pairs:"
    )

    print(
        rf_confusion_pairs.head(15).to_string(
            index=False
        )
    )

    # Class recall comparison.
    class_error_comparison = calculate_class_error_comparison(
        Y_VALIDATION,
        rf_predictions,
        fusion_predictions,
    )

    class_error_comparison = class_error_comparison.sort_values(
        "recall_delta_fusion_minus_rf",
        ascending=False,
    )

    # ==========================================================================
    # [12/12] SAVE ALL RESEARCH ARTIFACTS
    # ==========================================================================

    print_section(
        "[12/12] Saving Step 38 research artifacts..."
    )

    # --------------------------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------------------------

    metrics_rows = []

    for model_name, metrics in [
        ("Tuned Random Forest", rf_metrics),
        (
            "Step 37 Best RF/XGB Fusion",
            fusion_metrics,
        ),
    ]:

        row = {
            "model": model_name,
            **metrics,
        }

        metrics_rows.append(row)

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_df.to_csv(
        OUTPUT_DIR
        / "step38_comprehensive_validation_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Per-class RF
    # --------------------------------------------------------------------------

    rf_per_class.to_csv(
        OUTPUT_DIR
        / "step38_rf_per_class_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Per-class fusion
    # --------------------------------------------------------------------------

    fusion_per_class.to_csv(
        OUTPUT_DIR
        / "step38_fusion_per_class_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # RF confusion matrix
    # --------------------------------------------------------------------------

    rf_cm.to_csv(
        OUTPUT_DIR
        / "step38_rf_confusion_matrix.csv"
    )

    # --------------------------------------------------------------------------
    # Fusion confusion matrix
    # --------------------------------------------------------------------------

    fusion_cm.to_csv(
        OUTPUT_DIR
        / "step38_fusion_confusion_matrix.csv"
    )

    # --------------------------------------------------------------------------
    # RF error rates
    # --------------------------------------------------------------------------

    rf_error_df.to_csv(
        OUTPUT_DIR
        / "step38_rf_fpr_fnr_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Fusion error rates
    # --------------------------------------------------------------------------

    fusion_error_df.to_csv(
        OUTPUT_DIR
        / "step38_fusion_fpr_fnr_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # RF confusion pairs
    # --------------------------------------------------------------------------

    rf_confusion_pairs.to_csv(
        OUTPUT_DIR
        / "step38_rf_top_confusion_pairs.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Fusion confusion pairs
    # --------------------------------------------------------------------------

    fusion_confusion_pairs.to_csv(
        OUTPUT_DIR
        / "step38_fusion_top_confusion_pairs.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Class-level recall comparison
    # --------------------------------------------------------------------------

    class_error_comparison.to_csv(
        OUTPUT_DIR
        / "step38_rf_vs_fusion_class_recall_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Difficult classes
    # --------------------------------------------------------------------------

    difficult_classes = rf_per_class.sort_values(
        ["f1", "support"],
        ascending=[True, False],
    )

    difficult_classes.to_csv(
        OUTPUT_DIR
        / "step38_rf_difficult_classes.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Substantial-support difficult classes
    # --------------------------------------------------------------------------

    substantial_support.sort_values(
        "f1",
        ascending=True,
    ).to_csv(
        OUTPUT_DIR
        / "step38_substantial_support_difficult_classes.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Low-support classes
    # --------------------------------------------------------------------------

    low_support.to_csv(
        OUTPUT_DIR
        / "step38_low_support_class_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Infiltration
    # --------------------------------------------------------------------------

    pd.DataFrame(
        [
            {
                "class": "Infiltration",
                "support": infiltration_analysis["support"],
                "rf_TP": infiltration_analysis["rf"]["TP"],
                "rf_FN": infiltration_analysis["rf"]["FN"],
                "rf_FP": infiltration_analysis["rf"]["FP"],
                "rf_recall": infiltration_analysis["rf"]["recall"],
                "rf_precision": infiltration_analysis["rf"]["precision"],
                "fusion_TP": infiltration_analysis[
                    "best_step37_fusion"
                ]["TP"],
                "fusion_FN": infiltration_analysis[
                    "best_step37_fusion"
                ]["FN"],
                "fusion_FP": infiltration_analysis[
                    "best_step37_fusion"
                ]["FP"],
                "fusion_recall": infiltration_analysis[
                    "best_step37_fusion"
                ]["recall"],
                "fusion_precision": infiltration_analysis[
                    "best_step37_fusion"
                ]["precision"],
            }
        ]
    ).to_csv(
        OUTPUT_DIR
        / "step38_infiltration_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Prediction complementarity
    # --------------------------------------------------------------------------

    pd.DataFrame(
        [prediction_analysis]
    ).to_csv(
        OUTPUT_DIR
        / "step38_prediction_level_complementarity.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Fusion error changes
    # --------------------------------------------------------------------------

    pd.DataFrame(
        [fusion_error_analysis]
    ).to_csv(
        OUTPUT_DIR
        / "step38_rf_vs_fusion_error_changes.csv",
        index=False,
    )

    # --------------------------------------------------------------------------
    # Probability metadata
    # --------------------------------------------------------------------------

    probability_metadata = {
        "validation_rows": int(len(Y_VALIDATION)),
        "number_of_classes": N_CLASSES,
        "rf_probability_shape": list(
            rf_probabilities.shape
        ),
        "xgb_probability_shape": list(
            xgb_probabilities.shape
        ),
        "fusion_probability_shape": list(
            fusion_probabilities.shape
        ),
        "rf_weight": BEST_RF_WEIGHT,
        "xgb_weight": BEST_XGB_WEIGHT,
        "test_set_used": False,
        "training_performed": False,
        "hyperparameter_optimization_performed": False,
    }

    save_json(
        probability_metadata,
        OUTPUT_DIR
        / "step38_probability_validation_metadata.json",
    )

    # --------------------------------------------------------------------------
    # Final summary JSON
    # --------------------------------------------------------------------------

    rf_f1 = rf_metrics["macro_f1"]
    fusion_f1 = fusion_metrics["macro_f1"]

    rf_balanced = rf_metrics[
        "balanced_accuracy"
    ]

    fusion_balanced = fusion_metrics[
        "balanced_accuracy"
    ]

    summary = {
        "step": 38,
        "title": (
            "Ensemble Validation + Error Analysis"
        ),
        "status": "COMPLETED",
        "project_root": str(PROJECT_ROOT),
        "validation_dataset": str(VALIDATION_PATH),
        "test_set_used": False,
        "training_performed": False,
        "hyperparameter_optimization_performed": False,
        "primary_metric": PRIMARY_METRIC,
        "number_of_classes": N_CLASSES,
        "number_of_features": int(
            X_VALIDATION.shape[1]
        ),
        "validation_rows": int(
            len(Y_VALIDATION)
        ),
        "selected_candidate": "Tuned Random Forest",
        "step37_best_rf_weight": BEST_RF_WEIGHT,
        "step37_best_xgb_weight": BEST_XGB_WEIGHT,
        "rf_metrics": rf_metrics,
        "step37_best_fusion_metrics": fusion_metrics,
        "rf_vs_fusion_macro_f1_delta": float(
            fusion_f1 - rf_f1
        ),
        "rf_vs_fusion_balanced_accuracy_delta": float(
            fusion_balanced - rf_balanced
        ),
        "prediction_level_complementarity": prediction_analysis,
        "fusion_error_changes": fusion_error_analysis,
        "infiltration_analysis": infiltration_analysis,
        "key_substantial_support_difficult_classes": (
            substantial_support[
                [
                    "class_id",
                    "class_name",
                    "support",
                    "precision",
                    "recall",
                    "f1",
                    "FPR",
                    "FNR",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "rare_low_support_classes": (
            low_support[
                [
                    "class_id",
                    "class_name",
                    "support",
                    "precision",
                    "recall",
                    "f1",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "output_directory": str(
            OUTPUT_DIR
        ),
    }

    save_json(
        summary,
        OUTPUT_DIR
        / "step38_ensemble_validation_error_analysis_summary.json",
    )

    # --------------------------------------------------------------------------
    # Protocol
    # --------------------------------------------------------------------------

    runtime_seconds = time.time() - start_time

    write_protocol(
        rf_metrics,
        fusion_metrics,
        infiltration_analysis,
        prediction_analysis,
        fusion_error_analysis,
        runtime_seconds,
    )

    # ==========================================================================
    # FINAL CONSOLE SUMMARY
    # ==========================================================================

    print_header(
        "STEP 38 COMPLETE"
    )

    print("\nFINAL VALIDATION CANDIDATE")
    print(
        "Tuned Random Forest"
    )

    print(
        f"\nRF Macro F1: "
        f"{rf_metrics['macro_f1']:.6f}"
    )

    print(
        f"RF Balanced Accuracy: "
        f"{rf_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"RF Macro Precision: "
        f"{rf_metrics['macro_precision']:.6f}"
    )

    print(
        f"RF Macro Recall: "
        f"{rf_metrics['macro_recall']:.6f}"
    )

    print(
        f"RF Weighted F1: "
        f"{rf_metrics['weighted_f1']:.6f}"
    )

    print(
        f"RF Accuracy: "
        f"{rf_metrics['accuracy']:.6f}"
    )

    print(
        f"RF ROC-AUC OvR Macro: "
        f"{rf_metrics['roc_auc_ovr_macro']:.6f}"
        if rf_metrics["roc_auc_ovr_macro"] is not None
        else "RF ROC-AUC OvR Macro: N/A"
    )

    print(
        f"RF PR-AUC / AP Macro: "
        f"{rf_metrics['pr_auc_macro_average_precision']:.6f}"
        if rf_metrics[
            "pr_auc_macro_average_precision"
        ] is not None
        else "RF PR-AUC / AP Macro: N/A"
    )

    print("\nENSEMBLE DECISION")

    if (
        fusion_f1 > rf_f1
        and fusion_balanced >= rf_balanced
    ):
        print(
            "RF/XGB fusion improved RF on the primary validation criteria."
        )
    else:
        print(
            "RF/XGB fusion did NOT improve RF."
        )
        print(
            "Tuned Random Forest remains the final primary candidate."
        )

    print(
        f"\nMacro F1 delta "
        f"(Fusion - RF): "
        f"{fusion_f1 - rf_f1:+.6f}"
    )

    print(
        f"Balanced Accuracy delta "
        f"(Fusion - RF): "
        f"{fusion_balanced - rf_balanced:+.6f}"
    )

    print("\nINFILTRATION")

    print(
        f"Support: "
        f"{infiltration_analysis['support']:,}"
    )

    print(
        f"RF Recall: "
        f"{infiltration_analysis['rf']['recall']:.6f}"
    )

    print(
        f"RF Precision: "
        f"{infiltration_analysis['rf']['precision']:.6f}"
    )

    print("\nOUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    print("\nTEST SET USED: NO")
    print("TRAINING PERFORMED: NO")
    print("HYPERPARAMETER OPTIMIZATION: NO")

    print(
        f"\nRuntime: {runtime_seconds:.2f} seconds"
    )

    print("=" * 88)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()