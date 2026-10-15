"""
STEP 19 - CIC-IDS2018 BASELINE MODEL ANALYSIS AND COMPARISON

Purpose:
    Publication-oriented analysis of Step 18 baseline experiments.

Input:
    1. step18_cicids2018_baseline_results.csv
    2. step18_cicids2018_per_class_metrics.csv
    3. step18_cicids2018_confusion_matrices.npz
    4. step16_cicids2018_train_class_distribution.csv

Important:
    - Validation data only.
    - Test data is NOT used.
    - No model training.
    - No feature selection.
    - No SMOTE.
    - No hyperparameter tuning.

Primary evaluation criterion:
    Macro-F1

Secondary criteria:
    Balanced Accuracy
    Macro Recall
    Macro Precision
    Weighted-F1
    Accuracy

Statistical caution:
    This step performs descriptive and comparative analysis.
    It does NOT claim statistical significance from a single
    validation split because paired prediction-level outputs
    are not available from Step 18.
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = r"C:\Users\DELL\OneDrive\Desktop\network_attack_detection"

EXPERIMENT_DIR = os.path.join(
    BASE_DIR,
    "experiments"
)

REPORT_DIR = os.path.join(
    EXPERIMENT_DIR,
    "step19_reports"
)

os.makedirs(REPORT_DIR, exist_ok=True)


# Step 18 outputs
OVERALL_RESULTS_FILE = os.path.join(
    EXPERIMENT_DIR,
    "step18_cicids2018_baseline_results.csv"
)

PER_CLASS_FILE = os.path.join(
    EXPERIMENT_DIR,
    "step18_cicids2018_per_class_metrics.csv"
)

CONFUSION_FILE = os.path.join(
    EXPERIMENT_DIR,
    "step18_cicids2018_confusion_matrices.npz"
)

# Step 16 output
TRAIN_DISTRIBUTION_FILE = os.path.join(
    EXPERIMENT_DIR,
    "step16_cicids2018_train_class_distribution.csv"
)

# Step 19 outputs
SUMMARY_JSON = os.path.join(
    EXPERIMENT_DIR,
    "step19_cicids2018_baseline_analysis_summary.json"
)

OVERALL_ANALYSIS_FILE = os.path.join(
    REPORT_DIR,
    "step19_overall_metric_analysis.csv"
)

MODEL_COMPARISON_FILE = os.path.join(
    REPORT_DIR,
    "step19_model_comparison.csv"
)

STRATEGY_COMPARISON_FILE = os.path.join(
    REPORT_DIR,
    "step19_strategy_comparison.csv"
)

RELATIVE_IMPROVEMENT_FILE = os.path.join(
    REPORT_DIR,
    "step19_relative_improvements.csv"
)

PER_CLASS_ANALYSIS_FILE = os.path.join(
    REPORT_DIR,
    "step19_per_class_analysis.csv"
)

MINORITY_ANALYSIS_FILE = os.path.join(
    REPORT_DIR,
    "step19_minority_class_analysis.csv"
)

CONFUSION_ANALYSIS_FILE = os.path.join(
    REPORT_DIR,
    "step19_confusion_matrix_analysis.csv"
)

RANKING_FILE = os.path.join(
    REPORT_DIR,
    "step19_final_baseline_ranking.csv"
)

PARETO_FILE = os.path.join(
    REPORT_DIR,
    "step19_pareto_analysis.csv"
)

TEXT_REPORT_FILE = os.path.join(
    REPORT_DIR,
    "step19_baseline_analysis_report.txt"
)


# ============================================================================
# CONSTANTS
# ============================================================================

PRIMARY_METRIC = "Macro_F1"

METRICS = [
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_F1",
]

REQUIRED_OVERALL_COLUMNS = [
    "Experiment",
    "Strategy",
    "Model",
] + METRICS


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(title):
    """Print a formatted section header."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check_file(path, description):
    """Check whether an input file exists."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[ERROR] {description} not found:\n{path}"
        )

    print(f"[OK] {description}:")
    print(path)


def safe_percentage_change(new_value, old_value):
    """
    Calculate percentage change safely.
    """
    if old_value == 0:
        return np.nan

    return ((new_value - old_value) / abs(old_value)) * 100.0


def normalize_column_names(df):
    """
    Standardize common column naming differences.
    """
    df = df.copy()

    rename_map = {}

    for col in df.columns:
        clean = str(col).strip()

        if clean == "Balanced Accuracy":
            rename_map[col] = "Balanced_Accuracy"

        elif clean == "Macro Precision":
            rename_map[col] = "Macro_Precision"

        elif clean == "Macro Recall":
            rename_map[col] = "Macro_Recall"

        elif clean == "Macro F1":
            rename_map[col] = "Macro_F1"

        elif clean == "Weighted F1":
            rename_map[col] = "Weighted_F1"

    return df.rename(columns=rename_map)


def classify_class_frequency(count):
    """
    Categorize class frequency using the same conceptual categories
    used in Step 16.
    """
    if count >= 100000:
        return "Common"

    if count >= 10000:
        return "Minority"

    if count >= 1000:
        return "Rare"

    if count >= 100:
        return "Very_Rare"

    return "Extremely_Rare"


# ============================================================================
# START
# ============================================================================

print_header(
    "STEP 19 - CIC-IDS2018 BASELINE MODEL ANALYSIS AND COMPARISON"
)

print()
print("Purpose:")
print("Publication-oriented analysis of Step 18 baseline experiments.")
print()
print("Primary metric:")
print("Macro-F1")
print()
print("Validation:")
print("USED")
print()
print("Test:")
print("NOT USED")
print()
print("Model training:")
print("NOT PERFORMED")
print()
print("Feature selection:")
print("NOT PERFORMED")
print()
print("SMOTE:")
print("NOT USED")


# ============================================================================
# 1. CHECK INPUT FILES
# ============================================================================

print_header("CHECKING INPUT FILES")

check_file(
    OVERALL_RESULTS_FILE,
    "Step 18 overall results"
)

check_file(
    PER_CLASS_FILE,
    "Step 18 per-class metrics"
)

check_file(
    CONFUSION_FILE,
    "Step 18 confusion matrices"
)

check_file(
    TRAIN_DISTRIBUTION_FILE,
    "Step 16 training class distribution"
)


# ============================================================================
# 2. LOAD OVERALL RESULTS
# ============================================================================

print_header("LOADING STEP 18 OVERALL RESULTS")

overall = pd.read_csv(OVERALL_RESULTS_FILE)

overall = normalize_column_names(overall)

print("[OK] Overall results loaded.")
print(f"Rows    : {len(overall):,}")
print(f"Columns : {len(overall.columns)}")

missing_columns = [
    col for col in REQUIRED_OVERALL_COLUMNS
    if col not in overall.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns in Step 18 results:\n"
        + "\n".join(missing_columns)
    )

print("[OK] Required columns found.")


# ============================================================================
# 3. BASIC VALIDATION
# ============================================================================

print_header("VALIDATING BASELINE RESULTS")

if overall["Experiment"].duplicated().any():
    duplicates = overall.loc[
        overall["Experiment"].duplicated(),
        "Experiment"
    ].tolist()

    raise ValueError(
        "Duplicate experiment names detected:\n"
        + "\n".join(duplicates)
    )

if overall[METRICS].isnull().any().any():
    raise ValueError(
        "Missing metric values detected in Step 18 results."
    )

if not overall[METRICS].apply(
    lambda col: col.between(0, 1).all()
).all():
    raise ValueError(
        "One or more metric values are outside [0, 1]."
    )

print("[OK] No duplicate experiments.")
print("[OK] No missing metric values.")
print("[OK] All metrics are within [0, 1].")

print()
print(f"Experiments : {len(overall)}")
print(
    f"Models      : "
    f"{overall['Model'].nunique()}"
)
print(
    f"Strategies  : "
    f"{overall['Strategy'].nunique()}"
)


# ============================================================================
# 4. CONVERT METRICS TO PERCENTAGE
# ============================================================================

for metric in METRICS:
    overall[f"{metric}_Percent"] = (
        overall[metric] * 100.0
    )


# ============================================================================
# 5. PRIMARY METRIC RANKING
# ============================================================================

print_header("PRIMARY BASELINE RANKING")

ranking = overall.copy()

# Primary ranking:
# 1. Macro-F1
# 2. Balanced Accuracy
# 3. Macro Recall
# 4. Macro Precision
# 5. Weighted-F1
# 6. Accuracy

ranking = ranking.sort_values(
    by=[
        "Macro_F1",
        "Balanced_Accuracy",
        "Macro_Recall",
        "Macro_Precision",
        "Weighted_F1",
        "Accuracy",
    ],
    ascending=False,
).reset_index(drop=True)

ranking.insert(
    0,
    "Primary_Rank",
    np.arange(1, len(ranking) + 1)
)

ranking["Primary_Criterion"] = "Macro_F1"

print(
    ranking[
        [
            "Primary_Rank",
            "Experiment",
            "Strategy",
            "Model",
            "Macro_F1",
            "Balanced_Accuracy",
            "Macro_Precision",
            "Macro_Recall",
            "Weighted_F1",
            "Accuracy",
        ]
    ].to_string(index=False)
)

ranking.to_csv(
    RANKING_FILE,
    index=False
)


# ============================================================================
# 6. OVERALL METRIC ANALYSIS
# ============================================================================

print_header("OVERALL METRIC ANALYSIS")

overall_analysis = overall[
    [
        "Experiment",
        "Strategy",
        "Model",
    ] + METRICS
].copy()

for metric in METRICS:
    overall_analysis[f"{metric}_Percent"] = (
        overall_analysis[metric] * 100.0
    )

overall_analysis.to_csv(
    OVERALL_ANALYSIS_FILE,
    index=False
)

print(
    "[OK] Overall metric analysis saved:"
)
print(OVERALL_ANALYSIS_FILE)


# ============================================================================
# 7. BEST MODEL FOR EACH METRIC
# ============================================================================

print_header("BEST EXPERIMENT BY EACH METRIC")

best_by_metric = {}

for metric in METRICS:
    idx = overall[metric].idxmax()
    row = overall.loc[idx]

    best_by_metric[metric] = {
        "Experiment": str(row["Experiment"]),
        "Model": str(row["Model"]),
        "Strategy": str(row["Strategy"]),
        "Value": float(row[metric]),
    }

    print(
        f"{metric:20s}: "
        f"{row['Experiment']} "
        f"({row[metric]:.6f})"
    )


# ============================================================================
# 8. MODEL-LEVEL COMPARISON
# ============================================================================

print_header("RANDOM FOREST VS XGBOOST")

model_comparison_rows = []

for model_name in sorted(
    overall["Model"].unique()
):
    subset = overall[
        overall["Model"] == model_name
    ]

    row = {
        "Model": model_name,
        "Experiments": len(subset),
    }

    for metric in METRICS:
        row[f"Mean_{metric}"] = subset[metric].mean()
        row[f"Std_{metric}"] = subset[metric].std(
            ddof=0
        )
        row[f"Best_{metric}"] = subset[metric].max()
        row[f"Worst_{metric}"] = subset[metric].min()

    model_comparison_rows.append(row)


model_comparison = pd.DataFrame(
    model_comparison_rows
)

model_comparison.to_csv(
    MODEL_COMPARISON_FILE,
    index=False
)

print(
    model_comparison[
        [
            "Model",
            "Mean_Macro_F1",
            "Mean_Balanced_Accuracy",
            "Mean_Macro_Precision",
            "Mean_Macro_Recall",
            "Mean_Weighted_F1",
            "Mean_Accuracy",
        ]
    ].to_string(index=False)
)


# ============================================================================
# 9. STRATEGY-LEVEL COMPARISON
# ============================================================================

print_header("TRAINING STRATEGY COMPARISON")

strategy_rows = []

for strategy_name in sorted(
    overall["Strategy"].unique()
):
    subset = overall[
        overall["Strategy"] == strategy_name
    ]

    row = {
        "Strategy": strategy_name,
        "Experiments": len(subset),
    }

    for metric in METRICS:
        row[f"Mean_{metric}"] = subset[metric].mean()
        row[f"Std_{metric}"] = subset[metric].std(
            ddof=0
        )
        row[f"Best_{metric}"] = subset[metric].max()
        row[f"Worst_{metric}"] = subset[metric].min()

    strategy_rows.append(row)


strategy_comparison = pd.DataFrame(
    strategy_rows
)

strategy_comparison.to_csv(
    STRATEGY_COMPARISON_FILE,
    index=False
)

print(
    strategy_comparison[
        [
            "Strategy",
            "Mean_Macro_F1",
            "Mean_Balanced_Accuracy",
            "Mean_Macro_Precision",
            "Mean_Macro_Recall",
            "Mean_Weighted_F1",
            "Mean_Accuracy",
        ]
    ].to_string(index=False)
)


# ============================================================================
# 10. RELATIVE IMPROVEMENT ANALYSIS
# ============================================================================

print_header("RELATIVE IMPROVEMENT ANALYSIS")

improvement_rows = []

for model_name in sorted(
    overall["Model"].unique()
):
    model_subset = overall[
        overall["Model"] == model_name
    ]

    baseline_rows = model_subset[
        model_subset["Strategy"] == "B0_original"
    ]

    if baseline_rows.empty:
        continue

    baseline = baseline_rows.iloc[0]

    for _, row in model_subset.iterrows():

        if row["Strategy"] == "B0_original":
            continue

        result = {
            "Model": model_name,
            "Baseline_Strategy": "B0_original",
            "Compared_Strategy": row["Strategy"],
            "Baseline_Experiment": baseline["Experiment"],
            "Compared_Experiment": row["Experiment"],
        }

        for metric in METRICS:
            result[
                f"{metric}_Absolute_Difference"
            ] = (
                row[metric] - baseline[metric]
            )

            result[
                f"{metric}_Relative_Improvement_Percent"
            ] = safe_percentage_change(
                row[metric],
                baseline[metric]
            )

        improvement_rows.append(result)


relative_improvement = pd.DataFrame(
    improvement_rows
)

relative_improvement.to_csv(
    RELATIVE_IMPROVEMENT_FILE,
    index=False
)

print(
    "[OK] Relative improvement analysis saved:"
)
print(RELATIVE_IMPROVEMENT_FILE)


# ============================================================================
# 11. LOAD PER-CLASS METRICS
# ============================================================================

print_header("LOADING PER-CLASS METRICS")

per_class = pd.read_csv(PER_CLASS_FILE)

per_class = normalize_column_names(per_class)

print("[OK] Per-class metrics loaded.")
print(f"Rows    : {len(per_class):,}")
print(f"Columns : {len(per_class.columns)}")

print()
print("Available columns:")
print(list(per_class.columns))


# ============================================================================
# 12. IDENTIFY PER-CLASS METRIC COLUMNS
# ============================================================================

possible_label_columns = [
    "Label",
    "Class",
    "Class_Name",
    "Attack",
]

label_column = None

for col in possible_label_columns:
    if col in per_class.columns:
        label_column = col
        break

if label_column is None:
    raise ValueError(
        "Could not identify class-label column in "
        "Step 18 per-class metrics."
    )

possible_f1_columns = [
    "F1",
    "F1_Score",
    "F1-Score",
    "f1-score",
    "Macro_F1",
]

f1_column = None

for col in possible_f1_columns:
    if col in per_class.columns:
        f1_column = col
        break

if f1_column is None:
    raise ValueError(
        "Could not identify F1 column in "
        "Step 18 per-class metrics."
    )

possible_recall_columns = [
    "Recall",
    "Recall_Score",
    "recall",
]

recall_column = None

for col in possible_recall_columns:
    if col in per_class.columns:
        recall_column = col
        break

possible_precision_columns = [
    "Precision",
    "Precision_Score",
    "precision",
]

precision_column = None

for col in possible_precision_columns:
    if col in per_class.columns:
        precision_column = col
        break

print(
    f"[OK] Class label column: {label_column}"
)

print(
    f"[OK] F1 column: {f1_column}"
)

if recall_column:
    print(
        f"[OK] Recall column: {recall_column}"
    )

if precision_column:
    print(
        f"[OK] Precision column: {precision_column}"
    )


# ============================================================================
# 13. LOAD TRAINING CLASS DISTRIBUTION
# ============================================================================

print_header("LOADING TRAINING CLASS DISTRIBUTION")

train_distribution = pd.read_csv(
    TRAIN_DISTRIBUTION_FILE
)

train_distribution = normalize_column_names(
    train_distribution
)

print("[OK] Training class distribution loaded.")

print(
    "Columns:"
)

print(
    list(train_distribution.columns)
)


# ============================================================================
# 14. IDENTIFY CLASS COUNT COLUMNS
# ============================================================================

distribution_label_column = None

for col in possible_label_columns:
    if col in train_distribution.columns:
        distribution_label_column = col
        break

if distribution_label_column is None:
    raise ValueError(
        "Could not identify class column in "
        "Step 16 training distribution."
    )

possible_count_columns = [
    "Count",
    "Training_Count",
    "Original_Count",
]

count_column = None

for col in possible_count_columns:
    if col in train_distribution.columns:
        count_column = col
        break

if count_column is None:
    raise ValueError(
        "Could not identify count column in "
        "Step 16 training distribution."
    )

class_counts = train_distribution[
    [
        distribution_label_column,
        count_column,
    ]
].copy()

class_counts.columns = [
    "Label",
    "Training_Count",
]

class_counts["Frequency_Category"] = (
    class_counts["Training_Count"]
    .apply(classify_class_frequency)
)


# ============================================================================
# 15. PER-CLASS ANALYSIS
# ============================================================================

print_header("PER-CLASS PERFORMANCE ANALYSIS")

per_class_analysis = per_class.copy()

per_class_analysis = per_class_analysis.merge(
    class_counts,
    left_on=label_column,
    right_on="Label",
    how="left",
)

per_class_analysis["F1_Percent"] = (
    per_class_analysis[f1_column] * 100.0
)

if recall_column:
    per_class_analysis["Recall_Percent"] = (
        per_class_analysis[recall_column] * 100.0
    )

if precision_column:
    per_class_analysis["Precision_Percent"] = (
        per_class_analysis[precision_column] * 100.0
    )

per_class_analysis.to_csv(
    PER_CLASS_ANALYSIS_FILE,
    index=False
)

print(
    "[OK] Per-class analysis saved:"
)
print(PER_CLASS_ANALYSIS_FILE)


# ============================================================================
# 16. MINORITY-CLASS ANALYSIS
# ============================================================================

print_header("MINORITY-CLASS PERFORMANCE ANALYSIS")

minority_categories = [
    "Minority",
    "Rare",
    "Very_Rare",
    "Extremely_Rare",
]

minority_subset = per_class_analysis[
    per_class_analysis["Frequency_Category"].isin(
        minority_categories
    )
].copy()

if not minority_subset.empty:

    minority_summary = (
        minority_subset
        .groupby(
            ["Experiment", "Model", "Strategy"],
            as_index=False
        )
        .agg(
            Minority_Mean_F1=(
                f1_column,
                "mean"
            ),
            Minority_Median_F1=(
                f1_column,
                "median"
            ),
            Minority_Min_F1=(
                f1_column,
                "min"
            ),
            Minority_Max_F1=(
                f1_column,
                "max"
            ),
            Minority_Class_Count=(
                label_column,
                "nunique"
            ),
        )
    )

else:

    minority_summary = pd.DataFrame()

minority_summary.to_csv(
    MINORITY_ANALYSIS_FILE,
    index=False
)

if not minority_summary.empty:

    print(
        minority_summary.to_string(
            index=False
        )
    )

else:

    print(
        "[WARNING] No minority classes found "
        "in per-class metrics."
    )


# ============================================================================
# 17. CONFUSION MATRIX ANALYSIS
# ============================================================================

print_header("CONFUSION MATRIX ANALYSIS")

confusion_data = np.load(
    CONFUSION_FILE,
    allow_pickle=True
)

print("[OK] Confusion matrix archive loaded.")

print(
    "Stored arrays:"
)

for key in confusion_data.files:
    arr = confusion_data[key]

    print(
        f" - {key}: shape={arr.shape}"
    )


# Try to identify matrices automatically.
confusion_matrices = {}

for key in confusion_data.files:

    arr = confusion_data[key]

    if isinstance(arr, np.ndarray):

        if arr.ndim == 3:

            # Expected shape:
            # experiments x classes x classes
            confusion_matrices[key] = arr

        elif arr.ndim == 2:

            # A single matrix may be stored.
            confusion_matrices[key] = arr


print()
print(
    f"Detected matrix arrays: "
    f"{len(confusion_matrices)}"
)


confusion_rows = []


for key, matrix_array in confusion_matrices.items():

    if matrix_array.ndim == 2:

        matrix = matrix_array

        total = matrix.sum()

        correct = np.trace(matrix)

        accuracy = (
            correct / total
            if total > 0
            else np.nan
        )

        confusion_rows.append(
            {
                "Source": key,
                "Matrix_Index": 0,
                "Accuracy_From_Matrix": accuracy,
                "Total_Samples": int(total),
            }
        )

    elif matrix_array.ndim == 3:

        for idx in range(
            matrix_array.shape[0]
        ):

            matrix = matrix_array[idx]

            total = matrix.sum()

            correct = np.trace(matrix)

            accuracy = (
                correct / total
                if total > 0
                else np.nan
            )

            confusion_rows.append(
                {
                    "Source": key,
                    "Matrix_Index": idx,
                    "Accuracy_From_Matrix": accuracy,
                    "Total_Samples": int(total),
                }
            )


confusion_analysis = pd.DataFrame(
    confusion_rows
)

confusion_analysis.to_csv(
    CONFUSION_ANALYSIS_FILE,
    index=False
)

print(
    "[OK] Confusion matrix analysis saved:"
)

print(
    CONFUSION_ANALYSIS_FILE
)


# ============================================================================
# 18. PARETO ANALYSIS
# ============================================================================

print_header("PARETO-STYLE BASELINE ANALYSIS")

pareto = overall[
    [
        "Experiment",
        "Strategy",
        "Model",
    ] + METRICS
].copy()


def is_dominated(row, dataframe, objective_columns):
    """
    Determine whether a row is dominated by another experiment.

    An experiment A dominates B when A is at least as good as B
    on every objective and strictly better on at least one.
    """

    for _, other in dataframe.iterrows():

        if other["Experiment"] == row["Experiment"]:
            continue

        at_least_equal = all(
            other[col] >= row[col]
            for col in objective_columns
        )

        strictly_better = any(
            other[col] > row[col]
            for col in objective_columns
        )

        if at_least_equal and strictly_better:
            return True

    return False


pareto["Dominated"] = pareto.apply(
    lambda row: is_dominated(
        row,
        pareto,
        [
            "Macro_F1",
            "Balanced_Accuracy",
            "Macro_Precision",
            "Macro_Recall",
        ],
    ),
    axis=1,
)

pareto["Pareto_Efficient"] = (
    ~pareto["Dominated"]
)

pareto.to_csv(
    PARETO_FILE,
    index=False
)

print(
    pareto[
        [
            "Experiment",
            "Model",
            "Strategy",
            "Macro_F1",
            "Balanced_Accuracy",
            "Macro_Precision",
            "Macro_Recall",
            "Pareto_Efficient",
        ]
    ].to_string(index=False)
)


# ============================================================================
# 19. BASELINE SELECTION
# ============================================================================

print_header("SELECTING CANDIDATE BASELINE")

best_row = ranking.iloc[0]

best_experiment = str(
    best_row["Experiment"]
)

best_model = str(
    best_row["Model"]
)

best_strategy = str(
    best_row["Strategy"]
)

best_macro_f1 = float(
    best_row["Macro_F1"]
)

best_balanced_accuracy = float(
    best_row["Balanced_Accuracy"]
)

print()
print(
    "PRIMARY SELECTION CRITERION:"
)
print(
    "Macro-F1, because the dataset contains severe "
    "multiclass imbalance."
)

print()
print(
    f"Selected candidate baseline : "
    f"{best_experiment}"
)

print(
    f"Model                       : "
    f"{best_model}"
)

print(
    f"Training strategy           : "
    f"{best_strategy}"
)

print(
    f"Macro-F1                    : "
    f"{best_macro_f1:.6f}"
)

print(
    f"Balanced Accuracy           : "
    f"{best_balanced_accuracy:.6f}"
)


# ============================================================================
# 20. IMPORTANT STATISTICAL LIMITATION
# ============================================================================

print_header("STATISTICAL VALIDITY NOTE")

statistical_note = (
    "The baseline experiments were evaluated on the same fixed "
    "validation dataset. Step 18 provides aggregate metrics, "
    "per-class metrics, and confusion matrices, but does not "
    "retain sample-level prediction vectors required for paired "
    "prediction-level significance tests such as McNemar's test. "
    "Therefore, Step 19 reports descriptive comparisons, absolute "
    "differences, relative improvements, class-wise analysis, and "
    "Pareto-style comparisons rather than claiming statistical "
    "significance. A future repeated or cross-validation experiment "
    "can be used for formal statistical significance testing."
)

print(statistical_note)


# ============================================================================
# 21. GENERATE TEXT REPORT
# ============================================================================

print_header("GENERATING PUBLICATION-ORIENTED REPORT")

report_lines = []

report_lines.append(
    "CIC-IDS2018 BASELINE MODEL ANALYSIS"
)

report_lines.append(
    "=" * 60
)

report_lines.append("")

report_lines.append(
    "Primary evaluation criterion: Macro-F1"
)

report_lines.append(
    "Secondary criteria: Balanced Accuracy, Macro Recall, "
    "Macro Precision, Weighted-F1, Accuracy"
)

report_lines.append("")

report_lines.append(
    "BASELINE RANKING"
)

report_lines.append(
    "-" * 60
)

for _, row in ranking.iterrows():

    report_lines.append(
        f"{int(row['Primary_Rank'])}. "
        f"{row['Experiment']} | "
        f"Model={row['Model']} | "
        f"Strategy={row['Strategy']} | "
        f"Macro-F1={row['Macro_F1']:.6f} | "
        f"Balanced Accuracy={row['Balanced_Accuracy']:.6f} | "
        f"Macro Precision={row['Macro_Precision']:.6f} | "
        f"Macro Recall={row['Macro_Recall']:.6f} | "
        f"Weighted F1={row['Weighted_F1']:.6f} | "
        f"Accuracy={row['Accuracy']:.6f}"
    )


report_lines.append("")
report_lines.append(
    "BEST RESULT BY METRIC"
)

report_lines.append(
    "-" * 60
)

for metric, result in best_by_metric.items():

    report_lines.append(
        f"{metric}: "
        f"{result['Experiment']} "
        f"({result['Value']:.6f})"
    )


report_lines.append("")
report_lines.append(
    "CANDIDATE BASELINE FOR NEXT STAGE"
)

report_lines.append(
    "-" * 60
)

report_lines.append(
    f"Experiment: {best_experiment}"
)

report_lines.append(
    f"Model: {best_model}"
)

report_lines.append(
    f"Strategy: {best_strategy}"
)

report_lines.append(
    f"Macro-F1: {best_macro_f1:.6f}"
)

report_lines.append(
    f"Balanced Accuracy: {best_balanced_accuracy:.6f}"
)


report_lines.append("")
report_lines.append(
    "STATISTICAL LIMITATION"
)

report_lines.append(
    "-" * 60
)

report_lines.append(
    statistical_note
)

report_lines.append("")
report_lines.append(
    "TEST DATA STATUS"
)

report_lines.append(
    "-" * 60
)

report_lines.append(
    "The CIC-IDS2018 test dataset was not loaded or evaluated "
    "during Step 19."
)

report_lines.append("")
report_lines.append(
    "NEXT STAGE"
)

report_lines.append(
    "-" * 60
)

report_lines.append(
    "Hyperparameter optimization and controlled model "
    "improvement should be performed using the training data "
    "and validation data only. The test dataset must remain "
    "untouched until the final locked model is selected."
)

with open(
    TEXT_REPORT_FILE,
    "w",
    encoding="utf-8"
) as report_file:

    report_file.write(
        "\n".join(report_lines)
    )

print(
    "[OK] Text report saved:"
)

print(
    TEXT_REPORT_FILE
)


# ============================================================================
# 22. SAVE JSON SUMMARY
# ============================================================================

print_header("SAVING STEP 19 SUMMARY")

summary = {
    "step": 19,
    "dataset": "CIC-IDS2018",
    "purpose": (
        "Baseline model analysis and comparative evaluation"
    ),
    "validation_used": True,
    "test_used": False,
    "model_training_performed": False,
    "feature_selection_performed": False,
    "smote_used": False,
    "primary_metric": PRIMARY_METRIC,
    "secondary_metrics": [
        "Balanced_Accuracy",
        "Macro_Recall",
        "Macro_Precision",
        "Weighted_F1",
        "Accuracy",
    ],
    "number_of_experiments": int(
        len(overall)
    ),
    "models_evaluated": sorted(
        overall["Model"].unique().tolist()
    ),
    "strategies_evaluated": sorted(
        overall["Strategy"].unique().tolist()
    ),
    "best_experiment": best_experiment,
    "best_model": best_model,
    "best_strategy": best_strategy,
    "best_macro_f1": best_macro_f1,
    "best_balanced_accuracy": best_balanced_accuracy,
    "best_by_metric": best_by_metric,
    "statistical_significance_claimed": False,
    "statistical_limitation": statistical_note,
    "test_data_status": (
        "Untouched and not loaded."
    ),
    "generated_at": datetime.now().isoformat(),
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as summary_file:

    json.dump(
        summary,
        summary_file,
        indent=4,
        ensure_ascii=False
    )

print(
    "[OK] JSON summary saved:"
)

print(
    SUMMARY_JSON
)


# ============================================================================
# 23. FINAL SUMMARY
# ============================================================================

print_header(
    "STEP 19 COMPLETED SUCCESSFULLY"
)

print()
print(
    "Baseline analysis completed."
)

print()
print(
    f"Experiments analyzed : {len(overall)}"
)

print(
    f"Models analyzed      : "
    f"{', '.join(sorted(overall['Model'].unique()))}"
)

print(
    f"Strategies analyzed  : "
    f"{', '.join(sorted(overall['Strategy'].unique()))}"
)

print()
print(
    "PRIMARY BASELINE CANDIDATE"
)

print(
    f"Experiment            : "
    f"{best_experiment}"
)

print(
    f"Model                 : "
    f"{best_model}"
)

print(
    f"Strategy              : "
    f"{best_strategy}"
)

print(
    f"Macro-F1              : "
    f"{best_macro_f1:.6f}"
)

print(
    f"Balanced Accuracy     : "
    f"{best_balanced_accuracy:.6f}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "- Validation data was used for comparison."
)

print(
    "- Test data was NOT loaded."
)

print(
    "- No model was retrained."
)

print(
    "- No feature selection was performed."
)

print(
    "- No SMOTE was applied."
)

print(
    "- No statistical significance claim was made."
)

print()
print(
    "Reports generated in:"
)

print(
    REPORT_DIR
)

print()
print(
    "Next:"
)

print(
    "STEP 20 - BASELINE ERROR ANALYSIS / "
    "HYPERPARAMETER OPTIMIZATION DESIGN"
)

print()