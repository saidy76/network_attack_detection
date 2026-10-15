"""
Behaviour Based Network Attack Detection System using Machine Learning

STEP 37 — FINAL RF + XGBoost HETEROGENEOUS ENSEMBLE CONSTRUCTION

Purpose:
    Empirically determine whether controlled weighted probability
    fusion of the tuned Random Forest and tuned XGBoost improves
    over the strongest standalone Random Forest on the
    CIC-IDS2018 validation set.

Research rules:
    - Validation set only.
    - Test set is NOT used.
    - No model retraining.
    - No hyperparameter optimization.
    - Fixed 14-class mapping.
    - Fixed 30-feature order.
    - Explicit feature and class-order validation.
    - Probability matrices must be aligned.
    - Macro F1 is the primary selection metric.
    - Prediction-level complementarity is explicitly analysed.
    - Infiltration receives dedicated analysis.

Project:
    Behaviour Based Network Attack Detection System using Machine Learning

Step:
    37 — Final RF + XGBoost Heterogeneous Ensemble Construction
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\network_attack_detection")

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step37_final_ensemble"
)

# ------------------------------------------------------------
# Validation dataset ONLY
# ------------------------------------------------------------

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "CICIDS2018_validation_selected.csv"
)

# ------------------------------------------------------------
# Existing trained Random Forest
# ------------------------------------------------------------

RF_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "step21_rf_tuning"
    / "step21_best_random_forest.joblib"
)

# ------------------------------------------------------------
# Existing trained XGBoost
# ------------------------------------------------------------

XGB_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "step23_xgb_tuning"
    / "step23_best_xgboost.joblib"
)

LABEL_COLUMN = "Label"

RANDOM_STATE = 42

# ============================================================
# 2. FIXED PROJECT CLASS MAPPING
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
    class_name: idx
    for idx, class_name in enumerate(CLASS_NAMES)
}

ID_TO_CLASS = {
    idx: class_name
    for idx, class_name in enumerate(CLASS_NAMES)
}

EXPECTED_N_CLASSES = len(CLASS_NAMES)

EXPECTED_CLASS_IDS = np.arange(
    EXPECTED_N_CLASSES,
    dtype=np.int64,
)

EXPECTED_CLASS_NAMES = np.asarray(
    CLASS_NAMES,
    dtype=object,
)


# ============================================================
# 3. CONTROLLED FUSION WEIGHTS
# ============================================================

FUSION_WEIGHTS = [
    (1.0, 0.0),
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
    (0.0, 1.0),
]


# ============================================================
# 4. UTILITY FUNCTIONS
# ============================================================

def fail(message: str) -> None:
    """Fail loudly when a research assumption is violated."""
    raise RuntimeError(
        f"\n[STEP 37 FATAL ERROR]\n{message}\n"
    )


def check_file_exists(
    path: Path,
    description: str,
) -> None:
    """Check that an expected input file exists."""

    if not path.exists():
        fail(
            f"{description} not found:\n"
            f"{path}"
        )


def normalize_label(value) -> str:
    """Normalize a dataset label without changing its meaning."""
    return str(value).strip()


def encode_labels(
    series: pd.Series,
) -> np.ndarray:
    """
    Convert validation string labels into the fixed
    project numeric class IDs.
    """

    labels = series.map(normalize_label)

    unknown = sorted(
        set(labels.unique())
        -
        set(CLASS_TO_ID.keys())
    )

    if unknown:
        fail(
            "Unknown validation labels detected:\n"
            f"{unknown}\n\n"
            "Expected labels:\n"
            f"{CLASS_NAMES}"
        )

    return labels.map(
        CLASS_TO_ID
    ).to_numpy(
        dtype=np.int64
    )


def get_model_feature_names(model):
    """
    Return model feature names when available.
    """

    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_names is None:
        return None

    return list(feature_names)


def validate_probability_matrix(
    probabilities: np.ndarray,
    model_name: str,
    expected_rows: int,
) -> None:
    """
    Validate probability matrix shape,
    numerical validity, and row normalization.
    """

    probabilities = np.asarray(
        probabilities
    )

    expected_shape = (
        expected_rows,
        EXPECTED_N_CLASSES,
    )

    if probabilities.ndim != 2:
        fail(
            f"{model_name} probability output "
            "must be 2-dimensional.\n"
            f"Observed shape: {probabilities.shape}"
        )

    if probabilities.shape != expected_shape:
        fail(
            f"{model_name} probability matrix has "
            "incorrect shape.\n"
            f"Expected: {expected_shape}\n"
            f"Observed: {probabilities.shape}"
        )

    if not np.isfinite(
        probabilities
    ).all():
        fail(
            f"{model_name} probability matrix "
            "contains NaN or infinite values."
        )

    if np.any(
        probabilities < -1e-8
    ):
        fail(
            f"{model_name} probability matrix "
            "contains negative probabilities."
        )

    row_sums = probabilities.sum(
        axis=1
    )

    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-5,
    ):
        fail(
            f"{model_name} probability rows "
            "do not sum to 1 within tolerance."
        )


def verify_model_class_order(
    model,
    model_name: str,
) -> str:
    """
    Verify model class ordering.

    Models may expose classes_ as either:
        ['BENIGN', 'Bot', ...]
    or:
        [0, 1, ..., 13]

    Both representations are accepted only when they
    correspond exactly to the fixed project class mapping.

    Returns:
        'string labels'
        'numeric IDs'
        'unavailable'
    """

    model_classes = getattr(
        model,
        "classes_",
        None,
    )

    if model_classes is None:

        print(
            f"{model_name} does not expose classes_. "
            "Class-order verification unavailable."
        )

        return "unavailable"

    model_classes = np.asarray(
        model_classes
    )

    # --------------------------------------------------------
    # Number of classes
    # --------------------------------------------------------

    if model_classes.shape[0] != EXPECTED_N_CLASSES:
        fail(
            f"{model_name} exposes "
            f"{model_classes.shape[0]} classes; "
            f"expected {EXPECTED_N_CLASSES}."
        )

    # --------------------------------------------------------
    # String class representation
    # --------------------------------------------------------

    if all(
        isinstance(value, str)
        for value in model_classes
    ):

        normalized_classes = np.asarray(
            [
                normalize_label(value)
                for value in model_classes
            ],
            dtype=object,
        )

        if not np.array_equal(
            normalized_classes,
            EXPECTED_CLASS_NAMES,
        ):
            fail(
                f"{model_name} string class ordering "
                "does not match the fixed project mapping.\n\n"
                f"Expected:\n{EXPECTED_CLASS_NAMES}\n\n"
                f"Found:\n{normalized_classes}"
            )

        print(
            f"{model_name} class ordering: "
            "VERIFIED (string labels)"
        )

        return "string labels"

    # --------------------------------------------------------
    # Numeric class representation
    # --------------------------------------------------------

    try:

        numeric_classes = model_classes.astype(
            np.int64
        )

    except (
        ValueError,
        TypeError,
    ):

        fail(
            f"{model_name} exposes unsupported class "
            f"labels:\n{model_classes}"
        )

    if not np.array_equal(
        numeric_classes,
        EXPECTED_CLASS_IDS,
    ):
        fail(
            f"{model_name} numeric class ordering "
            "does not match the fixed project mapping.\n"
            f"Expected:\n{EXPECTED_CLASS_IDS}\n"
            f"Found:\n{numeric_classes}"
        )

    print(
        f"{model_name} class ordering: "
        "VERIFIED (numeric IDs)"
    )

    return "numeric IDs"


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Calculate the project-standard classification metrics.
    """

    return {
        "Accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "Balanced Accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "Macro Precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "Macro Recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "Macro F1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "Weighted F1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """
    Generate class-wise precision, recall, F1 and support.
    """

    labels = list(
        range(EXPECTED_N_CLASSES)
    )

    precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    support = np.bincount(
        y_true,
        minlength=EXPECTED_N_CLASSES,
    )

    rows = []

    for class_id in labels:

        rows.append(
            {
                "Model": model_name,
                "Class_ID": class_id,
                "Class": ID_TO_CLASS[class_id],
                "Precision": float(
                    precision[class_id]
                ),
                "Recall": float(
                    recall[class_id]
                ),
                "F1": float(
                    f1[class_id]
                ),
                "Support": int(
                    support[class_id]
                ),
            }
        )

    return pd.DataFrame(rows)


def confusion_matrix_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """
    Generate a labelled confusion matrix.
    """

    labels = list(
        range(EXPECTED_N_CLASSES)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    df.index.name = "True_Class"

    df.columns = [
        f"Predicted_{name}"
        for name in df.columns
    ]

    df.insert(
        0,
        "Model",
        model_name,
    )

    return df.reset_index()


def prediction_complementarity(
    y_true: np.ndarray,
    rf_pred: np.ndarray,
    xgb_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Quantify prediction-level agreement and disagreement.
    """

    rf_correct = (
        rf_pred == y_true
    )

    xgb_correct = (
        xgb_pred == y_true
    )

    both_correct = (
        rf_correct
        &
        xgb_correct
    )

    rf_only_correct = (
        rf_correct
        &
        ~xgb_correct
    )

    xgb_only_correct = (
        ~rf_correct
        &
        xgb_correct
    )

    both_wrong = (
        ~rf_correct
        &
        ~xgb_correct
    )

    disagreement = (
        rf_pred != xgb_pred
    )

    total = len(
        y_true
    )

    rows = [
        {
            "Category": "Both correct",
            "Count": int(
                both_correct.sum()
            ),
            "Rate": float(
                both_correct.mean()
            ),
        },
        {
            "Category": "RF correct, XGB wrong",
            "Count": int(
                rf_only_correct.sum()
            ),
            "Rate": float(
                rf_only_correct.mean()
            ),
        },
        {
            "Category": "XGB correct, RF wrong",
            "Count": int(
                xgb_only_correct.sum()
            ),
            "Rate": float(
                xgb_only_correct.mean()
            ),
        },
        {
            "Category": "Both wrong",
            "Count": int(
                both_wrong.sum()
            ),
            "Rate": float(
                both_wrong.mean()
            ),
        },
        {
            "Category": "Prediction disagreement",
            "Count": int(
                disagreement.sum()
            ),
            "Rate": float(
                disagreement.mean()
            ),
        },
        {
            "Category": "Prediction agreement",
            "Count": int(
                (~disagreement).sum()
            ),
            "Rate": float(
                (~disagreement).mean()
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


def class_disagreement_analysis(
    y_true: np.ndarray,
    rf_pred: np.ndarray,
    xgb_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Analyze RF/XGB prediction disagreement by true class.
    """

    rows = []

    for class_id in range(
        EXPECTED_N_CLASSES
    ):

        mask = (
            y_true == class_id
        )

        if not mask.any():
            continue

        true_class_rf = (
            rf_pred[mask]
        )

        true_class_xgb = (
            xgb_pred[mask]
        )

        disagreement = (
            true_class_rf
            !=
            true_class_xgb
        )

        rf_correct = (
            true_class_rf
            ==
            class_id
        )

        xgb_correct = (
            true_class_xgb
            ==
            class_id
        )

        rows.append(
            {
                "Class_ID": class_id,
                "Class": ID_TO_CLASS[class_id],
                "Support": int(
                    mask.sum()
                ),
                "Disagreement_Count": int(
                    disagreement.sum()
                ),
                "Disagreement_Rate": float(
                    disagreement.mean()
                ),
                "RF_Correct_Count": int(
                    rf_correct.sum()
                ),
                "XGB_Correct_Count": int(
                    xgb_correct.sum()
                ),
                "XGB_Only_Correct": int(
                    (
                        ~rf_correct
                        &
                        xgb_correct
                    ).sum()
                ),
                "RF_Only_Correct": int(
                    (
                        rf_correct
                        &
                        ~xgb_correct
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def infiltration_analysis(
    y_true: np.ndarray,
    predictions: dict,
) -> pd.DataFrame:
    """
    Dedicated Infiltration analysis.

    Includes:
        TP
        FN
        FP
        precision
        recall
        F1
        BENIGN -> Infiltration
        Infiltration -> BENIGN
    """

    infiltration_id = CLASS_TO_ID[
        "Infiltration"
    ]

    benign_id = CLASS_TO_ID[
        "BENIGN"
    ]

    rows = []

    for model_name, y_pred in predictions.items():

        true_infiltration = (
            y_true
            ==
            infiltration_id
        )

        predicted_infiltration = (
            y_pred
            ==
            infiltration_id
        )

        tp = int(
            np.sum(
                true_infiltration
                &
                predicted_infiltration
            )
        )

        fn = int(
            np.sum(
                true_infiltration
                &
                ~predicted_infiltration
            )
        )

        fp = int(
            np.sum(
                ~true_infiltration
                &
                predicted_infiltration
            )
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            /
            (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        benign_to_infiltration = int(
            np.sum(
                (
                    y_true
                    ==
                    benign_id
                )
                &
                (
                    y_pred
                    ==
                    infiltration_id
                )
            )
        )

        infiltration_to_benign = int(
            np.sum(
                (
                    y_true
                    ==
                    infiltration_id
                )
                &
                (
                    y_pred
                    ==
                    benign_id
                )
            )
        )

        rows.append(
            {
                "Model": model_name,
                "Infiltration_TP": tp,
                "Infiltration_FN": fn,
                "Infiltration_FP": fp,
                "Infiltration_Precision": precision,
                "Infiltration_Recall": recall,
                "Infiltration_F1": f1,
                "BENIGN_to_Infiltration": (
                    benign_to_infiltration
                ),
                "Infiltration_to_BENIGN": (
                    infiltration_to_benign
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def save_json(
    path: Path,
    data: dict,
) -> None:
    """
    Save JSON using UTF-8.
    """

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# 5. MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 80)
    print(
        "STEP 37 — FINAL RF + XGBoost "
        "HETEROGENEOUS ENSEMBLE"
    )
    print("=" * 80)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nValidation dataset:")
    print(VALIDATION_PATH)

    print("\nRandom Forest model:")
    print(RF_MODEL_PATH)

    print("\nXGBoost model:")
    print(XGB_MODEL_PATH)

    print("\n" + "-" * 80)
    print("RESEARCH INTEGRITY")
    print("-" * 80)

    print("Test set used: NO")
    print("Training performed: NO")
    print("Hyperparameter optimization: NO")
    print("Validation split: YES")
    print("Primary metric: Macro F1")

    print("=" * 80)

    # ========================================================
    # 5.1 Output directory
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 5.2 Input file validation
    # ========================================================

    print(
        "\n[1/10] Checking required input files..."
    )

    check_file_exists(
        VALIDATION_PATH,
        "Validation dataset",
    )

    check_file_exists(
        RF_MODEL_PATH,
        "Random Forest model",
    )

    check_file_exists(
        XGB_MODEL_PATH,
        "XGBoost model",
    )

    # --------------------------------------------------------
    # Explicit test-set protection
    # --------------------------------------------------------

    forbidden_path_terms = [
        "test",
        "cicids2018_test",
    ]

    for path in [
        VALIDATION_PATH,
        RF_MODEL_PATH,
        XGB_MODEL_PATH,
    ]:

        path_lower = str(
            path
        ).lower()

        for term in forbidden_path_terms:

            if term in path_lower:

                fail(
                    "Potential test-set path detected "
                    f"in Step 37:\n{path}"
                )

    print(
        "Required inputs: VERIFIED"
    )

    # ========================================================
    # 5.3 Load validation dataset
    # ========================================================

    print(
        "\n[2/10] Loading validation dataset..."
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation shape: "
        f"{validation_df.shape}"
    )

    if LABEL_COLUMN not in validation_df.columns:

        fail(
            f"Validation dataset does not contain "
            f"'{LABEL_COLUMN}' column."
        )

    feature_columns = [
        column
        for column in validation_df.columns
        if column != LABEL_COLUMN
    ]

    if len(feature_columns) != 30:

        fail(
            "Expected exactly 30 selected features.\n"
            f"Found: {len(feature_columns)}"
        )

    print(
        f"Feature count: "
        f"{len(feature_columns)} VERIFIED"
    )

    # ========================================================
    # 5.4 Encode labels
    # ========================================================

    print(
        "\n[3/10] Encoding validation labels..."
    )

    y_true = encode_labels(
        validation_df[
            LABEL_COLUMN
        ]
    )

    observed_classes = sorted(
        np.unique(
            y_true
        ).tolist()
    )

    print(
        f"Observed class IDs: "
        f"{observed_classes}"
    )

    missing_classes = sorted(
        set(
            range(
                EXPECTED_N_CLASSES
            )
        )
        -
        set(observed_classes)
    )

    if missing_classes:

        print(
            "\nWARNING: Validation set is missing "
            "the following classes:"
        )

        for class_id in missing_classes:

            print(
                f"  {class_id}: "
                f"{ID_TO_CLASS[class_id]}"
            )

    else:

        print(
            "All 14 classes present: VERIFIED"
        )

    # ========================================================
    # 5.5 Prepare validation feature matrix
    # ========================================================

    print(
        "\n[4/10] Preparing validation feature matrix..."
    )

    X_validation_df = validation_df[
        feature_columns
    ]

    non_numeric = (
        X_validation_df
        .select_dtypes(
            exclude=[np.number]
        )
        .columns
        .tolist()
    )

    if non_numeric:

        fail(
            "Non-numeric feature columns detected:\n"
            f"{non_numeric}"
        )

    X_validation = (
        X_validation_df
        .to_numpy(
            dtype=np.float32
        )
    )

    print(
        f"Validation feature matrix: "
        f"{X_validation.shape}"
    )

    # ========================================================
    # 5.6 Load existing models
    # ========================================================

    print(
        "\n[5/10] Loading existing trained models..."
    )

    rf_model = joblib.load(
        RF_MODEL_PATH
    )

    xgb_model = joblib.load(
        XGB_MODEL_PATH
    )

    print(
        "Random Forest type: "
        f"{type(rf_model).__name__}"
    )

    print(
        "XGBoost type: "
        f"{type(xgb_model).__name__}"
    )

    # ========================================================
    # 5.7 Verify model features and class ordering
    # ========================================================

    print(
        "\n[6/10] Verifying model compatibility..."
    )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    for model_name, model in [
        ("Random Forest", rf_model),
        ("XGBoost", xgb_model),
    ]:

        n_features = getattr(
            model,
            "n_features_in_",
            None,
        )

        if n_features is not None:

            if n_features != len(
                feature_columns
            ):

                fail(
                    f"{model_name} expects "
                    f"{n_features} features, while "
                    f"validation contains "
                    f"{len(feature_columns)}."
                )

            print(
                f"{model_name} feature count: "
                f"{n_features} VERIFIED"
            )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    rf_features = get_model_feature_names(
        rf_model
    )

    xgb_features = get_model_feature_names(
        xgb_model
    )

    if rf_features is not None:

        if rf_features != feature_columns:

            fail(
                "Random Forest feature order does not "
                "match validation feature order."
            )

        print(
            "Random Forest feature order: VERIFIED"
        )

    else:

        print(
            "Random Forest does not expose "
            "feature_names_in_; feature count verified."
        )

    if xgb_features is not None:

        if xgb_features != feature_columns:

            fail(
                "XGBoost feature order does not "
                "match validation feature order."
            )

        print(
            "XGBoost feature order: VERIFIED"
        )

    else:

        print(
            "XGBoost does not expose "
            "feature_names_in_; feature count verified."
        )

    # --------------------------------------------------------
    # Class ordering
    # --------------------------------------------------------

    rf_class_format = verify_model_class_order(
        rf_model,
        "Random Forest",
    )

    xgb_class_format = verify_model_class_order(
        xgb_model,
        "XGBoost",
    )

    # ========================================================
    # 5.8 Generate probability matrices
    # ========================================================

    print(
        "\n[7/10] Generating aligned "
        "validation probability matrices..."
    )

    print(
        "Generating Random Forest probabilities..."
    )

    rf_prob = np.asarray(
        rf_model.predict_proba(
            X_validation
        )
    )

    print(
        "Generating XGBoost probabilities..."
    )

    xgb_prob = np.asarray(
        xgb_model.predict_proba(
            X_validation
        )
    )

    # --------------------------------------------------------
    # Probability validation
    # --------------------------------------------------------

    validate_probability_matrix(
        rf_prob,
        "Random Forest",
        len(y_true),
    )

    validate_probability_matrix(
        xgb_prob,
        "XGBoost",
        len(y_true),
    )

    print(
        f"RF probability shape: "
        f"{rf_prob.shape} VERIFIED"
    )

    print(
        f"XGB probability shape: "
        f"{xgb_prob.shape} VERIFIED"
    )

    # ========================================================
    # 5.9 Standalone predictions and complementarity
    # ========================================================

    print(
        "\n[8/10] Evaluating standalone models "
        "and prediction-level complementarity..."
    )

    rf_pred = np.argmax(
        rf_prob,
        axis=1,
    )

    xgb_pred = np.argmax(
        xgb_prob,
        axis=1,
    )

    rf_metrics = calculate_metrics(
        y_true,
        rf_pred,
    )

    xgb_metrics = calculate_metrics(
        y_true,
        xgb_pred,
    )

    baseline_rows = [
        {
            "Model": "Tuned Random Forest",
            **rf_metrics,
        },
        {
            "Model": "Tuned XGBoost",
            **xgb_metrics,
        },
    ]

    baseline_df = pd.DataFrame(
        baseline_rows
    )

    baseline_df.to_csv(
        OUTPUT_DIR
        / "step37_baseline_metrics.csv",
        index=False,
    )

    print(
        "\nTUNED RANDOM FOREST"
    )

    for metric, value in rf_metrics.items():

        print(
            f"  {metric:<20}: "
            f"{value:.6f}"
        )

    print(
        "\nTUNED XGBOOST"
    )

    for metric, value in xgb_metrics.items():

        print(
            f"  {metric:<20}: "
            f"{value:.6f}"
        )

    # --------------------------------------------------------
    # Prediction-level complementarity
    # --------------------------------------------------------

    complementarity_df = prediction_complementarity(
        y_true,
        rf_pred,
        xgb_pred,
    )

    complementarity_df.to_csv(
        OUTPUT_DIR
        / "step37_prediction_complementarity.csv",
        index=False,
    )

    class_disagreement_df = (
        class_disagreement_analysis(
            y_true,
            rf_pred,
            xgb_pred,
        )
    )

    class_disagreement_df.to_csv(
        OUTPUT_DIR
        / "step37_class_disagreement_analysis.csv",
        index=False,
    )

    print(
        "\nPrediction-level complementarity:"
    )

    for _, row in complementarity_df.iterrows():

        print(
            f"  {row['Category']:<32}"
            f": {row['Count']:>9,} "
            f"({row['Rate']:.4%})"
        )

    # ========================================================
    # 5.10 Controlled probability fusion
    # ========================================================

    print(
        "\n[9/10] Evaluating controlled "
        "RF/XGBoost probability fusion..."
    )

    weight_rows = []

    per_class_results = []

    confusion_results = []

    infiltration_results = []

    error_results = []

    # --------------------------------------------------------
    # Baseline error count
    # --------------------------------------------------------

    rf_total_errors = int(
        np.sum(
            rf_pred != y_true
        )
    )

    xgb_total_errors = int(
        np.sum(
            xgb_pred != y_true
        )
    )

    # --------------------------------------------------------
    # Weight sweep
    # --------------------------------------------------------

    for rf_weight, xgb_weight in FUSION_WEIGHTS:

        print(
            f"\nEvaluating "
            f"RF={rf_weight:.1f}, "
            f"XGB={xgb_weight:.1f}"
        )

        # ----------------------------------------------------
        # Weighted probability fusion
        # ----------------------------------------------------

        fusion_prob = (
            rf_weight * rf_prob
            +
            xgb_weight * xgb_prob
        )

        # ----------------------------------------------------
        # Probability validation
        # ----------------------------------------------------

        if not np.isfinite(
            fusion_prob
        ).all():

            fail(
                f"Fusion probability contains "
                f"invalid values for "
                f"RF={rf_weight}, "
                f"XGB={xgb_weight}."
            )

        if not np.allclose(
            fusion_prob.sum(
                axis=1
            ),
            1.0,
            atol=1e-5,
        ):

            fail(
                f"Fusion probabilities do not sum "
                f"to 1 for RF={rf_weight}, "
                f"XGB={xgb_weight}."
            )

        # ----------------------------------------------------
        # Final prediction
        # ----------------------------------------------------

        fusion_pred = np.argmax(
            fusion_prob,
            axis=1,
        )

        model_name = (
            f"RF_{rf_weight:.1f}_XGB_{xgb_weight:.1f}"
        )

        # ----------------------------------------------------
        # Overall metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            y_true,
            fusion_pred,
        )

        weight_rows.append(
            {
                "RF_Weight": rf_weight,
                "XGB_Weight": xgb_weight,
                "Model": model_name,
                **metrics,
            }
        )

        # ----------------------------------------------------
        # Per-class metrics
        # ----------------------------------------------------

        class_df = calculate_per_class_metrics(
            y_true,
            fusion_pred,
            model_name,
        )

        class_df.insert(
            0,
            "RF_Weight",
            rf_weight,
        )

        class_df.insert(
            1,
            "XGB_Weight",
            xgb_weight,
        )

        per_class_results.append(
            class_df
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm_df = confusion_matrix_dataframe(
            y_true,
            fusion_pred,
            model_name,
        )

        cm_df.insert(
            1,
            "RF_Weight",
            rf_weight,
        )

        cm_df.insert(
            2,
            "XGB_Weight",
            xgb_weight,
        )

        confusion_results.append(
            cm_df
        )

        # ----------------------------------------------------
        # Infiltration analysis
        # ----------------------------------------------------

        infil_df = infiltration_analysis(
            y_true,
            {
                model_name: fusion_pred,
            },
        )

        infil_df.insert(
            1,
            "RF_Weight",
            rf_weight,
        )

        infil_df.insert(
            2,
            "XGB_Weight",
            xgb_weight,
        )

        infiltration_results.append(
            infil_df
        )

        # ----------------------------------------------------
        # Error comparison against RF
        # ----------------------------------------------------

        fusion_total_errors = int(
            np.sum(
                fusion_pred != y_true
            )
        )

        corrected_rf_errors = int(
            np.sum(
                (
                    rf_pred != y_true
                )
                &
                (
                    fusion_pred == y_true
                )
            )
        )

        introduced_errors = int(
            np.sum(
                (
                    rf_pred == y_true
                )
                &
                (
                    fusion_pred != y_true
                )
            )
        )

        xgb_correct_rf_wrong = int(
            np.sum(
                (
                    rf_pred != y_true
                )
                &
                (
                    xgb_pred == y_true
                )
            )
        )

        error_results.append(
            {
                "RF_Weight": rf_weight,
                "XGB_Weight": xgb_weight,
                "Model": model_name,
                "Total_Errors": fusion_total_errors,
                "RF_Total_Errors": rf_total_errors,
                "XGB_Total_Errors": xgb_total_errors,
                "Fusion_Corrected_RF_Errors": (
                    corrected_rf_errors
                ),
                "Fusion_Introduced_Errors": (
                    introduced_errors
                ),
                "Net_Error_Change_vs_RF": (
                    fusion_total_errors
                    -
                    rf_total_errors
                ),
                "Potential_XGB_Only_Corrections": (
                    xgb_correct_rf_wrong
                ),
            }
        )

        print(
            f"  Macro F1: "
            f"{metrics['Macro F1']:.6f}"
        )

        print(
            f"  Balanced Accuracy: "
            f"{metrics['Balanced Accuracy']:.6f}"
        )

    # ========================================================
    # 5.11 Combine result tables
    # ========================================================

    weight_results_df = pd.DataFrame(
        weight_rows
    )

    per_class_results_df = pd.concat(
        per_class_results,
        ignore_index=True,
    )

    confusion_results_df = pd.concat(
        confusion_results,
        ignore_index=True,
    )

    infiltration_results_df = pd.concat(
        infiltration_results,
        ignore_index=True,
    )

    error_results_df = pd.DataFrame(
        error_results
    )

    # --------------------------------------------------------
    # Save weight comparison
    # --------------------------------------------------------

    weight_results_df.to_csv(
        OUTPUT_DIR
        / "step37_weight_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save per-class metrics
    # --------------------------------------------------------

    per_class_results_df.to_csv(
        OUTPUT_DIR
        / "step37_per_class_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save confusion matrices
    # --------------------------------------------------------

    confusion_results_df.to_csv(
        OUTPUT_DIR
        / "step37_confusion_matrices.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save Infiltration analysis
    # --------------------------------------------------------

    infiltration_results_df.to_csv(
        OUTPUT_DIR
        / "step37_infiltration_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save error comparison
    # --------------------------------------------------------

    error_results_df.to_csv(
        OUTPUT_DIR
        / "step37_error_comparison.csv",
        index=False,
    )

    # ========================================================
    # 5.12 Rank all configurations
    # ========================================================

    print(
        "\n[10/10] Ranking validation configurations..."
    )

    ranked_results_df = (
        weight_results_df
        .sort_values(
            by=[
                "Macro F1",
                "Balanced Accuracy",
                "Macro Recall",
                "Macro Precision",
                "RF_Weight",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    ranked_results_df.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(ranked_results_df) + 1,
        ),
    )

    ranked_results_df.to_csv(
        OUTPUT_DIR
        / "step37_model_ranking.csv",
        index=False,
    )

    # ========================================================
    # 5.13 Best configuration
    # ========================================================

    best_row = ranked_results_df.iloc[
        0
    ]

    best_rf_weight = float(
        best_row["RF_Weight"]
    )

    best_xgb_weight = float(
        best_row["XGB_Weight"]
    )

    best_model_name = str(
        best_row["Model"]
    )

    best_macro_f1 = float(
        best_row["Macro F1"]
    )

    best_balanced_accuracy = float(
        best_row["Balanced Accuracy"]
    )

    best_macro_precision = float(
        best_row["Macro Precision"]
    )

    best_macro_recall = float(
        best_row["Macro Recall"]
    )

    best_weighted_f1 = float(
        best_row["Weighted F1"]
    )

    best_accuracy = float(
        best_row["Accuracy"]
    )

    rf_macro_f1 = float(
        rf_metrics["Macro F1"]
    )

    rf_balanced_accuracy = float(
        rf_metrics["Balanced Accuracy"]
    )

    # --------------------------------------------------------
    # Improvement over RF
    # --------------------------------------------------------

    macro_f1_delta = (
        best_macro_f1
        -
        rf_macro_f1
    )

    balanced_accuracy_delta = (
        best_balanced_accuracy
        -
        rf_balanced_accuracy
    )

    # --------------------------------------------------------
    # Important methodological decision
    # --------------------------------------------------------
    #
    # RF-only is already one of the evaluated endpoints.
    #
    # Therefore, if the best configuration is:
    #
    #     RF = 1.0
    #     XGB = 0.0
    #
    # then the controlled search confirms that RF itself
    # is the best validation configuration.
    #
    # We do NOT call that an ensemble improvement.
    # --------------------------------------------------------

    if (
        best_rf_weight == 1.0
        and
        best_xgb_weight == 0.0
    ):

        ensemble_improves_rf = False

        ensemble_decision = (
            "RF remains optimal; "
            "no XGBoost fusion weight improved RF."
        )

        final_candidate = (
            "Tuned Random Forest"
        )

    elif best_macro_f1 > rf_macro_f1:

        ensemble_improves_rf = True

        ensemble_decision = (
            "A weighted RF + XGBoost fusion "
            "improves Macro F1 over RF on validation."
        )

        final_candidate = (
            "RF + XGBoost weighted probability ensemble"
        )

    else:

        ensemble_improves_rf = False

        ensemble_decision = (
            "No RF + XGBoost fusion configuration "
            "improved RF on validation."
        )

        final_candidate = (
            "Tuned Random Forest"
        )

    # ========================================================
    # 5.14 Generate best configuration predictions
    # ========================================================

    best_fusion_prob = (
        best_rf_weight * rf_prob
        +
        best_xgb_weight * xgb_prob
    )

    best_fusion_pred = np.argmax(
        best_fusion_prob,
        axis=1,
    )

    best_per_class_df = (
        calculate_per_class_metrics(
            y_true,
            best_fusion_pred,
            best_model_name,
        )
    )

    best_per_class_df.to_csv(
        OUTPUT_DIR
        / "step37_best_configuration_per_class.csv",
        index=False,
    )

    best_confusion_df = (
        confusion_matrix_dataframe(
            y_true,
            best_fusion_pred,
            best_model_name,
        )
    )

    best_confusion_df.to_csv(
        OUTPUT_DIR
        / "step37_best_configuration_confusion_matrix.csv",
        index=False,
    )

    # ========================================================
    # 5.15 Infiltration comparison
    # ========================================================

    best_infiltration_df = infiltration_analysis(
        y_true,
        {
            "Tuned Random Forest": rf_pred,
            "Tuned XGBoost": xgb_pred,
            best_model_name: best_fusion_pred,
        },
    )

    best_infiltration_df.to_csv(
        OUTPUT_DIR
        / "step37_best_configuration_infiltration_comparison.csv",
        index=False,
    )

    # ========================================================
    # 5.16 Difficult-class analysis
    # ========================================================

    difficult_classes = (
        best_per_class_df[
            best_per_class_df[
                "Support"
            ]
            <= 100
        ]
        .sort_values(
            by="F1"
        )
    )

    substantial_support_difficult = (
        best_per_class_df[
            (
                best_per_class_df[
                    "Support"
                ]
                >= 1000
            )
            &
            (
                best_per_class_df[
                    "F1"
                ]
                < 0.70
            )
        ]
        .sort_values(
            by="F1"
        )
    )

    difficult_classes.to_csv(
        OUTPUT_DIR
        / "step37_rare_class_analysis.csv",
        index=False,
    )

    substantial_support_difficult.to_csv(
        OUTPUT_DIR
        / "step37_substantial_support_difficult_classes.csv",
        index=False,
    )

    # ========================================================
    # 5.17 Save selected configuration
    # ========================================================

    selected_configuration = {
        "step": 37,
        "selection_metric": "Macro F1",
        "selected_model": final_candidate,
        "validation_best_configuration": {
            "configuration_name": best_model_name,
            "rf_weight": best_rf_weight,
            "xgb_weight": best_xgb_weight,
            "macro_f1": best_macro_f1,
            "balanced_accuracy": (
                best_balanced_accuracy
            ),
            "macro_precision": (
                best_macro_precision
            ),
            "macro_recall": (
                best_macro_recall
            ),
            "weighted_f1": (
                best_weighted_f1
            ),
            "accuracy": (
                best_accuracy
            ),
        },
        "rf_baseline_macro_f1": (
            rf_macro_f1
        ),
        "rf_baseline_balanced_accuracy": (
            rf_balanced_accuracy
        ),
        "macro_f1_delta_vs_rf": (
            macro_f1_delta
        ),
        "balanced_accuracy_delta_vs_rf": (
            balanced_accuracy_delta
        ),
        "ensemble_improves_rf": (
            ensemble_improves_rf
        ),
        "ensemble_decision": (
            ensemble_decision
        ),
        "test_set_used": False,
        "training_performed": False,
        "hyperparameter_optimization_performed": False,
    }

    save_json(
        OUTPUT_DIR
        / "step37_best_ensemble_config.json",
        selected_configuration,
    )

    # ========================================================
    # 5.18 Full summary JSON
    # ========================================================

    elapsed_seconds = (
        time.time()
        -
        start_time
    )

    summary = {
        "project": (
            "Behaviour Based Network Attack "
            "Detection System using Machine Learning"
        ),
        "step": 37,
        "step_title": (
            "Final RF + XGBoost Heterogeneous "
            "Ensemble Construction"
        ),
        "dataset": "CIC-IDS2018",
        "evaluation_split": "validation",
        "test_set_used": False,
        "training_performed": False,
        "hyperparameter_optimization_performed": False,
        "n_validation_samples": int(
            len(y_true)
        ),
        "n_features": int(
            len(feature_columns)
        ),
        "n_classes": int(
            EXPECTED_N_CLASSES
        ),
        "class_mapping": CLASS_TO_ID,
        "rf_class_representation": (
            rf_class_format
        ),
        "xgb_class_representation": (
            xgb_class_format
        ),
        "rf_model_path": str(
            RF_MODEL_PATH
        ),
        "xgb_model_path": str(
            XGB_MODEL_PATH
        ),
        "validation_path": str(
            VALIDATION_PATH
        ),
        "rf_baseline": rf_metrics,
        "xgb_baseline": xgb_metrics,
        "best_validation_configuration": {
            "model": best_model_name,
            "rf_weight": best_rf_weight,
            "xgb_weight": best_xgb_weight,
            "metrics": {
                "Accuracy": best_accuracy,
                "Balanced Accuracy": (
                    best_balanced_accuracy
                ),
                "Macro Precision": (
                    best_macro_precision
                ),
                "Macro Recall": (
                    best_macro_recall
                ),
                "Macro F1": (
                    best_macro_f1
                ),
                "Weighted F1": (
                    best_weighted_f1
                ),
            },
        },
        "rf_macro_f1": rf_macro_f1,
        "best_macro_f1": best_macro_f1,
        "macro_f1_delta_vs_rf": (
            macro_f1_delta
        ),
        "rf_balanced_accuracy": (
            rf_balanced_accuracy
        ),
        "best_balanced_accuracy": (
            best_balanced_accuracy
        ),
        "balanced_accuracy_delta_vs_rf": (
            balanced_accuracy_delta
        ),
        "ensemble_improves_rf": (
            ensemble_improves_rf
        ),
        "ensemble_decision": (
            ensemble_decision
        ),
        "final_candidate_after_step37": (
            final_candidate
        ),
        "runtime_seconds": (
            elapsed_seconds
        ),
    }

    save_json(
        OUTPUT_DIR
        / "step37_summary.json",
        summary,
    )

    # ========================================================
    # 5.19 Protocol TXT
    # ========================================================

    protocol = f"""
STEP 37 — FINAL RF + XGBoost HETEROGENEOUS ENSEMBLE CONSTRUCTION

Project:
Behaviour Based Network Attack Detection System using Machine Learning

Dataset:
CIC-IDS2018

Evaluation split:
Validation only

Validation samples:
{len(y_true):,}

Features:
{len(feature_columns)}

Classes:
{EXPECTED_N_CLASSES}

Primary model:
Tuned Random Forest

Complementary candidate:
Tuned XGBoost

Fusion method:
Weighted probability averaging

Formula:
P_fusion = w_RF * P_RF + w_XGB * P_XGB

Controlled weights:
{FUSION_WEIGHTS}

Random Forest baseline:
Macro F1 = {rf_macro_f1:.6f}
Balanced Accuracy = {rf_balanced_accuracy:.6f}

XGBoost baseline:
Macro F1 = {xgb_metrics['Macro F1']:.6f}
Balanced Accuracy = {xgb_metrics['Balanced Accuracy']:.6f}

Best validation configuration:
{best_model_name}

Best RF weight:
{best_rf_weight:.1f}

Best XGBoost weight:
{best_xgb_weight:.1f}

Best configuration Macro F1:
{best_macro_f1:.6f}

Best configuration Balanced Accuracy:
{best_balanced_accuracy:.6f}

Macro F1 delta versus RF:
{macro_f1_delta:+.6f}

Balanced Accuracy delta versus RF:
{balanced_accuracy_delta:+.6f}

Ensemble improves RF:
{ensemble_improves_rf}

Decision:
{ensemble_decision}

Final candidate after Step 37:
{final_candidate}

Prediction-level complementarity:
Explicitly evaluated.

Analyses performed:
- RF/XGB agreement
- RF/XGB disagreement
- RF-only correct predictions
- XGB-only correct predictions
- both-correct predictions
- both-wrong predictions
- class-specific disagreement
- weighted fusion
- error correction versus RF
- fusion-introduced errors
- Infiltration-specific analysis
- rare-class analysis
- substantial-support difficult-class analysis

Research integrity:
- Test set NOT used
- No training performed
- No hyperparameter optimization performed
- Fixed 14-class mapping
- Fixed 30-feature structure
- Feature order verified
- Probability dimensions verified
- Class ordering verified
- Validation-only ensemble selection

Test set status:
UNTOUCHED

Runtime:
{elapsed_seconds:.2f} seconds
""".strip()

    with open(
        OUTPUT_DIR
        / "step37_protocol.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            protocol
        )

    # ========================================================
    # 5.20 Final console report
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 37 COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nVALIDATION BASELINES"
    )

    print(
        f"RF Macro F1:       "
        f"{rf_macro_f1:.6f}"
    )

    print(
        f"XGB Macro F1:      "
        f"{xgb_metrics['Macro F1']:.6f}"
    )

    print(
        "\nBEST CONFIGURATION"
    )

    print(
        f"Configuration:     "
        f"{best_model_name}"
    )

    print(
        f"RF weight:         "
        f"{best_rf_weight:.1f}"
    )

    print(
        f"XGB weight:        "
        f"{best_xgb_weight:.1f}"
    )

    print(
        f"Macro F1:          "
        f"{best_macro_f1:.6f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{best_balanced_accuracy:.6f}"
    )

    print(
        f"Macro Precision:   "
        f"{best_macro_precision:.6f}"
    )

    print(
        f"Macro Recall:      "
        f"{best_macro_recall:.6f}"
    )

    print(
        f"Weighted F1:       "
        f"{best_weighted_f1:.6f}"
    )

    print(
        f"Accuracy:          "
        f"{best_accuracy:.6f}"
    )

    print(
        "\nCHANGE VS RANDOM FOREST"
    )

    print(
        f"Macro F1 delta: "
        f"{macro_f1_delta:+.6f}"
    )

    print(
        f"Balanced Accuracy delta: "
        f"{balanced_accuracy_delta:+.6f}"
    )

    print(
        "\nENSEMBLE DECISION"
    )

    print(
        ensemble_decision
    )

    print(
        f"\nFinal candidate: "
        f"{final_candidate}"
    )

    print(
        "\nOUTPUT DIRECTORY:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nTEST SET USED: NO"
    )

    print(
        "\nTRAINING PERFORMED: NO"
    )

    print(
        "\nHYPERPARAMETER OPTIMIZATION: NO"
    )

    print(
        f"\nRuntime: "
        f"{elapsed_seconds:.2f} seconds"
    )

    print(
        "=" * 80
    )


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()