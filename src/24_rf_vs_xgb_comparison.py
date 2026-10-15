from pathlib import Path
import json
import pandas as pd


# ============================================================
# STEP 24 - RF VS XGBOOST COMPARATIVE ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STEP21_DIR = PROJECT_ROOT / "experiments" / "step21_rf_tuning"
STEP23_DIR = PROJECT_ROOT / "experiments" / "step23_reports"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "step24_reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 78)
print("STEP 24 - RANDOM FOREST VS XGBOOST COMPARATIVE ANALYSIS")
print("=" * 78)

print(f"Project root : {PROJECT_ROOT}")
print(f"Step 21 dir  : {STEP21_DIR}")
print(f"Step 23 dir  : {STEP23_DIR}")
print(f"Output dir   : {OUTPUT_DIR}")


# ============================================================
# INPUT FILES
# ============================================================

RF_METRICS_FILE = (
    STEP21_DIR / "step21_rf_validation_metrics.csv"
)

XGB_METRICS_FILE = (
    STEP23_DIR / "step23_xgb_validation_metrics.csv"
)

RF_PER_CLASS_FILE = (
    STEP21_DIR / "step21_rf_validation_per_class_metrics.csv"
)

XGB_PER_CLASS_FILE = (
    STEP23_DIR / "step23_xgb_validation_per_class_metrics.csv"
)

RF_PARAMS_FILE = (
    STEP21_DIR / "step21_best_rf_parameters.json"
)

XGB_PARAMS_FILE = (
    STEP23_DIR / "step23_best_xgb_parameters.json"
)


# ============================================================
# CHECK INPUTS
# ============================================================

print("\n" + "=" * 78)
print("CHECKING STEP 21 AND STEP 23 OUTPUTS")
print("=" * 78)

required_files = [
    RF_METRICS_FILE,
    XGB_METRICS_FILE,
    RF_PER_CLASS_FILE,
    XGB_PER_CLASS_FILE,
    RF_PARAMS_FILE,
    XGB_PARAMS_FILE,
]

for file_path in required_files:
    print(f"Checking: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required Step 24 input not found:\n{file_path}"
        )

print("\nAll required input files found.")


# ============================================================
# LOAD OVERALL METRICS
# ============================================================

print("\n" + "=" * 78)
print("LOADING OVERALL VALIDATION METRICS")
print("=" * 78)

rf_metrics = pd.read_csv(RF_METRICS_FILE)
xgb_metrics = pd.read_csv(XGB_METRICS_FILE)

if len(rf_metrics) != 1:
    raise ValueError(
        "Step 21 RF metrics file must contain exactly one result row."
    )

if len(xgb_metrics) != 1:
    raise ValueError(
        "Step 23 XGBoost metrics file must contain exactly one result row."
    )

rf_row = rf_metrics.iloc[0]
xgb_row = xgb_metrics.iloc[0]


# ============================================================
# METRIC DEFINITIONS
# ============================================================

METRICS = [
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_F1",
]


# ============================================================
# BUILD OVERALL COMPARISON
# ============================================================

comparison_rows = []

for metric in METRICS:

    if metric not in rf_metrics.columns:
        raise ValueError(
            f"RF metrics file missing column: {metric}"
        )

    if metric not in xgb_metrics.columns:
        raise ValueError(
            f"XGBoost metrics file missing column: {metric}"
        )

    rf_value = float(rf_row[metric])
    xgb_value = float(xgb_row[metric])

    difference = xgb_value - rf_value

    if difference > 0:
        winner = "XGBoost"
    elif difference < 0:
        winner = "Random Forest"
    else:
        winner = "Tie"

    comparison_rows.append(
        {
            "Metric": metric,
            "Random_Forest": rf_value,
            "XGBoost": xgb_value,
            "XGBoost_minus_RF": difference,
            "Winner": winner,
        }
    )

comparison_df = pd.DataFrame(comparison_rows)


# ============================================================
# PRIMARY METRIC DECISION
# ============================================================

rf_macro_f1 = float(rf_row["Macro_F1"])
xgb_macro_f1 = float(xgb_row["Macro_F1"])

macro_f1_difference = xgb_macro_f1 - rf_macro_f1

if xgb_macro_f1 > rf_macro_f1:
    primary_winner = "XGBoost"
elif rf_macro_f1 > xgb_macro_f1:
    primary_winner = "Random Forest"
else:
    primary_winner = "Tie"


# ============================================================
# LOAD PER-CLASS RESULTS
# ============================================================

print("\n" + "=" * 78)
print("LOADING PER-CLASS VALIDATION RESULTS")
print("=" * 78)

rf_per_class = pd.read_csv(RF_PER_CLASS_FILE)
xgb_per_class = pd.read_csv(XGB_PER_CLASS_FILE)

print("RF columns :", rf_per_class.columns.tolist())
print("XGB columns:", xgb_per_class.columns.tolist())


# ============================================================
# NORMALIZE RF FILE
# ============================================================

# Step 21 RF classification report format:
#
# Unnamed: 0 | precision | recall | f1-score | support
#
# Example:
# BENIGN
# Bot
# Brute Force -Web
# ...
# macro avg
# weighted avg


if "Unnamed: 0" in rf_per_class.columns:

    rf_per_class = rf_per_class.rename(
        columns={
            "Unnamed: 0": "Label",
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1",
            "support": "Support",
        }
    )

elif "Label" not in rf_per_class.columns:

    raise ValueError(
        "Unable to identify class label column in RF per-class file."
    )


# ============================================================
# NORMALIZE XGB FILE
# ============================================================

xgb_per_class = xgb_per_class.rename(
    columns={
        "precision": "Precision",
        "recall": "Recall",
        "f1-score": "F1",
        "support": "Support",
    }
)


# ============================================================
# CLEAN LABELS
# ============================================================

rf_per_class["Label"] = (
    rf_per_class["Label"]
    .astype(str)
    .str.strip()
)

xgb_per_class["Label"] = (
    xgb_per_class["Label"]
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE AGGREGATE ROWS
# ============================================================

aggregate_labels = {
    "accuracy",
    "macro avg",
    "weighted avg",
    "micro avg",
    "samples avg",
}


rf_per_class = rf_per_class[
    ~rf_per_class["Label"]
    .str.lower()
    .isin(aggregate_labels)
].copy()


xgb_per_class = xgb_per_class[
    ~xgb_per_class["Label"]
    .str.lower()
    .isin(aggregate_labels)
].copy()


# ============================================================
# RESET INDEX
# ============================================================

rf_per_class = rf_per_class.reset_index(drop=True)
xgb_per_class = xgb_per_class.reset_index(drop=True)


# ============================================================
# CREATE RF CLASS IDs
# ============================================================

# RF file does not contain Class_ID.
# Assign IDs according to class order.

rf_per_class["Class_ID"] = range(
    len(rf_per_class)
)


# ============================================================
# CREATE XGB CLASS IDs IF NEEDED
# ============================================================

if "Class_ID" not in xgb_per_class.columns:

    xgb_per_class["Class_ID"] = range(
        len(xgb_per_class)
    )


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Class_ID",
    "Label",
    "Precision",
    "Recall",
    "F1",
    "Support",
]


for column in required_columns:

    if column not in rf_per_class.columns:

        raise ValueError(
            f"RF per-class file missing column after normalization: {column}"
        )

    if column not in xgb_per_class.columns:

        raise ValueError(
            f"XGBoost per-class file missing column: {column}"
        )


# ============================================================
# CLEAN CLASS IDS
# ============================================================

rf_per_class["Class_ID"] = pd.to_numeric(
    rf_per_class["Class_ID"],
    errors="coerce"
)

xgb_per_class["Class_ID"] = pd.to_numeric(
    xgb_per_class["Class_ID"],
    errors="coerce"
)


if rf_per_class["Class_ID"].isna().any():

    raise ValueError(
        "RF Class_ID contains invalid values."
    )


if xgb_per_class["Class_ID"].isna().any():

    raise ValueError(
        "XGBoost Class_ID contains invalid values."
    )


rf_per_class["Class_ID"] = (
    rf_per_class["Class_ID"]
    .astype(int)
)

xgb_per_class["Class_ID"] = (
    xgb_per_class["Class_ID"]
    .astype(int)
)


# ============================================================
# SORT BY CLASS ID
# ============================================================

rf_per_class = (
    rf_per_class
    .sort_values("Class_ID")
    .reset_index(drop=True)
)

xgb_per_class = (
    xgb_per_class
    .sort_values("Class_ID")
    .reset_index(drop=True)
)


# ============================================================
# PRINT ACTUAL CLASS COUNTS
# ============================================================

print(
    f"\nRF actual classes  : {len(rf_per_class)}"
)

print(
    f"XGB actual classes : {len(xgb_per_class)}"
)


# ============================================================
# VERIFY CLASS COUNT
# ============================================================

if len(rf_per_class) != len(xgb_per_class):

    print("\nRF labels:")
    print(
        rf_per_class[
            ["Class_ID", "Label"]
        ].to_string(index=False)
    )

    print("\nXGB labels:")
    print(
        xgb_per_class[
            ["Class_ID", "Label"]
        ].to_string(index=False)
    )

    raise ValueError(
        "RF and XGBoost have different numbers of actual classes."
    )


# ============================================================
# VERIFY CLASS LABEL ALIGNMENT
# ============================================================

rf_classes = rf_per_class[
    ["Class_ID", "Label"]
].copy()

xgb_classes = xgb_per_class[
    ["Class_ID", "Label"]
].copy()


rf_classes["Label"] = (
    rf_classes["Label"]
    .astype(str)
    .str.strip()
)

xgb_classes["Label"] = (
    xgb_classes["Label"]
    .astype(str)
    .str.strip()
)


if not rf_classes.equals(xgb_classes):

    print("\nRF classes:")
    print(
        rf_classes.to_string(index=False)
    )

    print("\nXGBoost classes:")
    print(
        xgb_classes.to_string(index=False)
    )

    raise ValueError(
        "RF and XGBoost class definitions do not match."
    )


# ============================================================
# MERGE PER-CLASS RESULTS
# ============================================================

rf_subset = rf_per_class[
    [
        "Class_ID",
        "Label",
        "Precision",
        "Recall",
        "F1",
        "Support",
    ]
].copy()


rf_subset = rf_subset.rename(
    columns={
        "Precision": "RF_Precision",
        "Recall": "RF_Recall",
        "F1": "RF_F1",
        "Support": "RF_Support",
    }
)


xgb_subset = xgb_per_class[
    [
        "Class_ID",
        "Label",
        "Precision",
        "Recall",
        "F1",
        "Support",
    ]
].copy()


xgb_subset = xgb_subset.rename(
    columns={
        "Precision": "XGB_Precision",
        "Recall": "XGB_Recall",
        "F1": "XGB_F1",
        "Support": "XGB_Support",
    }
)


per_class_comparison = rf_subset.merge(
    xgb_subset,
    on=["Class_ID", "Label"],
    how="inner",
)


# ============================================================
# VERIFY MERGE
# ============================================================

if len(per_class_comparison) != len(rf_per_class):

    raise ValueError(
        "Per-class merge failed. "
        "RF and XGBoost class labels may not align."
    )


# ============================================================
# F1 DIFFERENCE
# ============================================================

per_class_comparison["F1_Difference"] = (
    per_class_comparison["XGB_F1"]
    - per_class_comparison["RF_F1"]
)


# ============================================================
# PER-CLASS WINNER
# ============================================================

def determine_winner(value):

    if value > 0:
        return "XGBoost"

    elif value < 0:
        return "Random Forest"

    else:
        return "Tie"


per_class_comparison["Per_Class_Winner"] = (
    per_class_comparison["F1_Difference"]
    .apply(determine_winner)
)


# ============================================================
# COUNT CLASS WINS
# ============================================================

xgb_class_wins = int(
    (
        per_class_comparison["Per_Class_Winner"]
        == "XGBoost"
    ).sum()
)


rf_class_wins = int(
    (
        per_class_comparison["Per_Class_Winner"]
        == "Random Forest"
    ).sum()
)


tie_class_count = int(
    (
        per_class_comparison["Per_Class_Winner"]
        == "Tie"
    ).sum()
)


# ============================================================
# LOAD BEST PARAMETERS
# ============================================================

print("\n" + "=" * 78)
print("LOADING BEST MODEL PARAMETERS")
print("=" * 78)


with open(
    RF_PARAMS_FILE,
    "r",
    encoding="utf-8",
) as f:

    rf_params_data = json.load(f)


with open(
    XGB_PARAMS_FILE,
    "r",
    encoding="utf-8",
) as f:

    xgb_params_data = json.load(f)


# ============================================================
# PRINT OVERALL COMPARISON
# ============================================================

print("\n" + "=" * 78)
print("OVERALL MODEL COMPARISON")
print("=" * 78)

print(
    comparison_df.to_string(index=False)
)


# ============================================================
# PRIMARY METRIC DECISION
# ============================================================

print("\n" + "=" * 78)
print("PRIMARY METRIC DECISION")
print("=" * 78)

print(
    f"RF Macro F1    : {rf_macro_f1:.6f}"
)

print(
    f"XGB Macro F1   : {xgb_macro_f1:.6f}"
)

print(
    f"Difference     : {macro_f1_difference:+.6f}"
)

print(
    f"Primary winner : {primary_winner}"
)


# ============================================================
# PER-CLASS F1 COMPARISON
# ============================================================

print("\n" + "=" * 78)
print("PER-CLASS F1 COMPARISON")
print("=" * 78)

print(
    per_class_comparison[
        [
            "Class_ID",
            "Label",
            "RF_F1",
            "XGB_F1",
            "F1_Difference",
            "Per_Class_Winner",
        ]
    ].to_string(index=False)
)


# ============================================================
# PER-CLASS WIN COUNT
# ============================================================

print("\n" + "=" * 78)
print("PER-CLASS WIN COUNT")
print("=" * 78)

print(
    f"Random Forest wins : {rf_class_wins}"
)

print(
    f"XGBoost wins       : {xgb_class_wins}"
)

print(
    f"Ties               : {tie_class_count}"
)


# ============================================================
# SAVE OVERALL COMPARISON
# ============================================================

comparison_df.to_csv(
    OUTPUT_DIR / "step24_rf_vs_xgb_comparison.csv",
    index=False,
)


# ============================================================
# SAVE PER-CLASS COMPARISON
# ============================================================

per_class_comparison.to_csv(
    OUTPUT_DIR / "step24_rf_vs_xgb_per_class_comparison.csv",
    index=False,
)


# ============================================================
# SAVE DECISION SUMMARY
# ============================================================

decision_summary = {

    "experiment":
        "Step24_RF_vs_XGBoost_Comparative_Analysis",

    "dataset":
        "CIC-IDS2018",

    "training_strategy":
        "B1_undersampled",

    "feature_count":
        30,

    "class_count":
        14,

    "comparison_basis":
        "Same validation dataset and same evaluation metrics",

    "primary_metric":
        "Macro_F1",

    "random_forest_macro_f1":
        rf_macro_f1,

    "xgboost_macro_f1":
        xgb_macro_f1,

    "xgboost_minus_rf_macro_f1":
        macro_f1_difference,

    "primary_winner":
        primary_winner,

    "random_forest_class_wins":
        rf_class_wins,

    "xgboost_class_wins":
        xgb_class_wins,

    "tie_class_count":
        tie_class_count,

    "random_forest_parameters":
        rf_params_data,

    "xgboost_parameters":
        xgb_params_data,
}


with open(
    OUTPUT_DIR / "step24_model_selection_decision.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        decision_summary,
        f,
        indent=4,
    )


# ============================================================
# SAVE PROTOCOL
# ============================================================

protocol = f"""
STEP 24 - RANDOM FOREST VS XGBOOST COMPARATIVE ANALYSIS

Dataset:
CIC-IDS2018

Training representation:
B1 undersampled

Feature count:
30

Class count:
14

Comparison:
Tuned Random Forest vs Tuned XGBoost

Evaluation dataset:
CICIDS2018_validation_selected.csv

Primary metric:
Macro F1

Random Forest Macro F1:
{rf_macro_f1:.6f}

XGBoost Macro F1:
{xgb_macro_f1:.6f}

XGBoost minus Random Forest Macro F1:
{macro_f1_difference:+.6f}

Primary model winner:
{primary_winner}

Random Forest per-class F1 wins:
{rf_class_wins}

XGBoost per-class F1 wins:
{xgb_class_wins}

Per-class F1 ties:
{tie_class_count}

Important:

No additional training was performed in Step 24.

No hyperparameter optimization was performed in Step 24.

No validation data was used to tune either model.

No test dataset was used.

The comparison uses the completed outputs from Step 21 and Step 23.
"""


with open(
    OUTPUT_DIR / "step24_comparative_analysis_protocol.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(protocol)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 78)
print("STEP 24 COMPLETED")
print("=" * 78)

print("Generated files:")

print(
    "  - step24_rf_vs_xgb_comparison.csv"
)

print(
    "  - step24_rf_vs_xgb_per_class_comparison.csv"
)

print(
    "  - step24_model_selection_decision.json"
)

print(
    "  - step24_comparative_analysis_protocol.txt"
)

print("\nPRIMARY MODEL DECISION:")

print(
    f"  {primary_winner} selected based on Macro F1."
)

print("\nNEXT:")

print(
    "STEP 25 - LSTM SEQUENCE MODEL PREPARATION"
)