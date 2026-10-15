"""
Step 20A - Baseline Error Analysis
Project:
Behaviour Based Network Attack Detection System using Machine Learning

Purpose:
    Publication-oriented error analysis of the six CIC-IDS2018 baseline
    experiments completed in Step 18.

Important methodological rules:
    - No model is retrained.
    - No test data is loaded.
    - No feature selection is performed.
    - No balancing is performed.
    - No hyperparameter tuning is performed here.
    - Existing Step 18/19 artifacts are analyzed only.
    - Validation results remain the basis of baseline comparison.

Author:
    Capstone Research Pipeline

Status:
    STEP 20A - EXECUTED ANALYSIS STAGE
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
STEP19_DIR = EXPERIMENTS_DIR / "step19_reports"

OUTPUT_DIR = EXPERIMENTS_DIR / "step20_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. EXPECTED INPUT FILES
# ============================================================

BASELINE_RESULTS = (
    EXPERIMENTS_DIR / "step18_cicids2018_baseline_results.csv"
)

CONFUSION_MATRIX_FILE = (
    EXPERIMENTS_DIR / "step18_cicids2018_confusion_matrices.npz"
)

PER_CLASS_CANDIDATES = [
    EXPERIMENTS_DIR / "step18_cicids2018_per_class_metrics.csv",
    EXPERIMENTS_DIR / "step18_cicids2018_per_class_results.csv",
    STEP19_DIR / "step19_per_class_analysis.csv",
]


# ============================================================
# 3. CLASS DEFINITIONS
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

CLASS_ID_TO_NAME = {
    idx: name for idx, name in enumerate(CLASS_NAMES)
}


# ============================================================
# 4. RESEARCH CONFIGURATION
# ============================================================

PRIMARY_METRIC = "Macro_F1"

BASELINE_CANDIDATE = "B1_undersampled_RandomForest"

EXPECTED_EXPERIMENTS = [
    "B0_original_RandomForest",
    "B1_undersampled_RandomForest",
    "B2_hybrid_RandomForest",
    "B0_original_XGBoost",
    "B1_undersampled_XGBoost",
    "B2_hybrid_XGBoost",
]


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def require_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing required file:\n"
            f"  {path}\n"
            f"Description: {description}\n"
        )


def find_per_class_file():
    for path in PER_CLASS_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "\nNo per-class metrics file was found.\n"
        "Checked:\n"
        + "\n".join(str(p) for p in PER_CLASS_CANDIDATES)
    )


def normalize_columns(df):
    """
    Normalize common naming differences between Step 18/19 artifacts.
    """

    rename_map = {}

    for col in df.columns:
        normalized = col.strip().lower()

        if normalized == "macro f1":
            rename_map[col] = "Macro_F1"

        elif normalized == "macro_f1":
            rename_map[col] = "Macro_F1"

        elif normalized == "balanced accuracy":
            rename_map[col] = "Balanced_Accuracy"

        elif normalized == "balanced_accuracy":
            rename_map[col] = "Balanced_Accuracy"

        elif normalized == "macro precision":
            rename_map[col] = "Macro_Precision"

        elif normalized == "macro_precision":
            rename_map[col] = "Macro_Precision"

        elif normalized == "macro recall":
            rename_map[col] = "Macro_Recall"

        elif normalized == "macro_recall":
            rename_map[col] = "Macro_Recall"

        elif normalized == "weighted f1":
            rename_map[col] = "Weighted_F1"

        elif normalized == "weighted_f1":
            rename_map[col] = "Weighted_F1"

        elif normalized == "class":
            rename_map[col] = "Label"

        elif normalized == "class_name":
            rename_map[col] = "Label"

        elif normalized == "f1":
            rename_map[col] = "F1"

        elif normalized == "precision":
            rename_map[col] = "Precision"

        elif normalized == "recall":
            rename_map[col] = "Recall"

        elif normalized == "support":
            rename_map[col] = "Support"

        elif normalized == "class_id":
            rename_map[col] = "Class_ID"

    return df.rename(columns=rename_map)


def validate_baseline_results(df):
    required = {
        "Experiment",
        "Strategy",
        "Model",
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "Weighted_F1",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Baseline results are missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    if df["Experiment"].duplicated().any():
        raise ValueError(
            "Duplicate experiment names detected in baseline results."
        )

    if len(df) != 6:
        print(
            f"[WARNING] Expected 6 baseline experiments, "
            f"but found {len(df)}."
        )

    invalid_metric_rows = []

    metric_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "Weighted_F1",
    ]

    for col in metric_columns:
        invalid = ~df[col].between(0.0, 1.0)

        if invalid.any():
            invalid_metric_rows.append(col)

    if invalid_metric_rows:
        raise ValueError(
            "Metrics outside [0,1] detected: "
            + ", ".join(invalid_metric_rows)
        )


def validate_per_class_results(df):
    required = {
        "Experiment",
        "Label",
        "Precision",
        "Recall",
        "F1",
        "Support",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Per-class results are missing required columns:\n"
            + "\n".join(sorted(missing))
        )


def safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# 6. LOAD BASELINE RESULTS
# ============================================================

print_section("STEP 20A - BASELINE ERROR ANALYSIS")

print(f"Project root : {PROJECT_ROOT}")
print(f"Output dir   : {OUTPUT_DIR}")

require_file(
    BASELINE_RESULTS,
    "Step 18 aggregate baseline results"
)

require_file(
    CONFUSION_MATRIX_FILE,
    "Step 18 confusion matrix archive"
)

PER_CLASS_FILE = find_per_class_file()

print("\n[INPUT FILES]")
print(f"Baseline results : {BASELINE_RESULTS}")
print(f"Per-class file   : {PER_CLASS_FILE}")
print(f"Confusion matrix : {CONFUSION_MATRIX_FILE}")


baseline_df = pd.read_csv(BASELINE_RESULTS)
baseline_df = normalize_columns(baseline_df)
validate_baseline_results(baseline_df)

baseline_df = baseline_df.sort_values(
    "Macro_F1",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 7. LOAD PER-CLASS RESULTS
# ============================================================

per_class_df = pd.read_csv(PER_CLASS_FILE)
per_class_df = normalize_columns(per_class_df)

per_class_df = safe_numeric(
    per_class_df,
    [
        "Class_ID",
        "Precision",
        "Recall",
        "F1",
        "Support",
    ],
)

validate_per_class_results(per_class_df)


# Convert class IDs to labels if necessary
if "Label" in per_class_df.columns:
    per_class_df["Label"] = per_class_df["Label"].astype(str)

if "Class_ID" in per_class_df.columns:
    mask = per_class_df["Label"].isna() | (
        per_class_df["Label"].str.lower().isin(["nan", "none"])
    )

    per_class_df.loc[mask, "Label"] = (
        per_class_df.loc[mask, "Class_ID"]
        .map(CLASS_ID_TO_NAME)
    )


# ============================================================
# 8. VALIDATE EXPERIMENT COVERAGE
# ============================================================

print_section("EXPERIMENT COVERAGE")

found_experiments = sorted(
    baseline_df["Experiment"].astype(str).unique()
)

print("Experiments found:")

for exp in found_experiments:
    print(f"  - {exp}")

missing_experiments = sorted(
    set(EXPECTED_EXPERIMENTS) - set(found_experiments)
)

unexpected_experiments = sorted(
    set(found_experiments) - set(EXPECTED_EXPERIMENTS)
)

if missing_experiments:
    print("\n[WARNING] Missing expected experiments:")
    for exp in missing_experiments:
        print(f"  - {exp}")

if unexpected_experiments:
    print("\n[WARNING] Unexpected experiments:")
    for exp in unexpected_experiments:
        print(f"  - {exp}")


# ============================================================
# 9. B1 RANDOM FOREST CLASS-LEVEL ERROR ANALYSIS
# ============================================================

print_section("B1 RANDOM FOREST - CLASS LEVEL ANALYSIS")

b1_rf = per_class_df[
    per_class_df["Experiment"] == BASELINE_CANDIDATE
].copy()

if b1_rf.empty:
    raise ValueError(
        f"No per-class records found for {BASELINE_CANDIDATE}"
    )

b1_rf = b1_rf.sort_values(
    "F1",
    ascending=True
).reset_index(drop=True)


# Add explicit difficulty indicators
b1_rf["F1_Gap_From_Perfect"] = 1.0 - b1_rf["F1"]
b1_rf["Recall_Gap_From_Perfect"] = 1.0 - b1_rf["Recall"]
b1_rf["Precision_Gap_From_Perfect"] = 1.0 - b1_rf["Precision"]

# Class difficulty score:
# Equal-weighted combination of F1 and recall deficit.
b1_rf["Difficulty_Score"] = (
    0.6 * b1_rf["F1_Gap_From_Perfect"]
    + 0.4 * b1_rf["Recall_Gap_From_Perfect"]
)

b1_output = OUTPUT_DIR / "step20_b1_rf_error_analysis.csv"

b1_rf.to_csv(
    b1_output,
    index=False
)

print(f"[OK] Saved: {b1_output}")


# ============================================================
# 10. WORST CLASSES
# ============================================================

print("\nWorst B1 Random Forest classes by F1:")

worst_f1 = b1_rf.sort_values(
    ["F1", "Support"],
    ascending=[True, False]
)

print(
    worst_f1[
        ["Label", "Precision", "Recall", "F1", "Support"]
    ].to_string(index=False)
)


print("\nWorst B1 Random Forest classes by Recall:")

worst_recall = b1_rf.sort_values(
    ["Recall", "Support"],
    ascending=[True, False]
)

print(
    worst_recall[
        ["Label", "Precision", "Recall", "F1", "Support"]
    ].to_string(index=False)
)


# ============================================================
# 11. LOAD CONFUSION MATRICES
# ============================================================

print_section("CONFUSION MATRIX ANALYSIS")

cm_archive = np.load(
    CONFUSION_MATRIX_FILE,
    allow_pickle=False
)

available_matrices = sorted(cm_archive.files)

print("Matrices found:")

for key in available_matrices:
    print(f"  - {key}: shape={cm_archive[key].shape}")


# ============================================================
# 12. CONFUSION PAIR EXTRACTION
# ============================================================

confusion_records = []

for experiment in available_matrices:

    matrix = np.asarray(
        cm_archive[experiment]
    )

    if matrix.shape != (14, 14):
        print(
            f"[WARNING] Skipping {experiment}: "
            f"expected (14,14), got {matrix.shape}"
        )
        continue

    matrix = matrix.astype(np.int64)

    total_errors = int(
        matrix.sum() - np.trace(matrix)
    )

    for true_id in range(14):

        for pred_id in range(14):

            if true_id == pred_id:
                continue

            count = int(
                matrix[true_id, pred_id]
            )

            if count <= 0:
                continue

            true_support = int(
                matrix[true_id, :].sum()
            )

            error_rate = (
                count / true_support
                if true_support > 0
                else 0.0
            )

            confusion_records.append(
                {
                    "Experiment": experiment,
                    "True_Class_ID": true_id,
                    "True_Label": CLASS_ID_TO_NAME[true_id],
                    "Predicted_Class_ID": pred_id,
                    "Predicted_Label": CLASS_ID_TO_NAME[pred_id],
                    "Misclassified_Count": count,
                    "True_Class_Support": true_support,
                    "Class_Error_Rate": error_rate,
                    "Total_Validation_Errors": total_errors,
                }
            )


confusion_df = pd.DataFrame(
    confusion_records
)

if confusion_df.empty:
    raise ValueError(
        "No off-diagonal confusion records were extracted."
    )

confusion_df = confusion_df.sort_values(
    [
        "Experiment",
        "Misclassified_Count",
    ],
    ascending=[
        True,
        False,
    ],
)


confusion_output = (
    OUTPUT_DIR /
    "step20_b1_rf_confusion_analysis.csv"
)

# Save all experiment confusion pairs.
confusion_df.to_csv(
    confusion_output,
    index=False
)

print(
    f"[OK] Saved all confusion-pair analysis: "
    f"{confusion_output}"
)


# ============================================================
# 13. B1 RF ONLY - TOP CONFUSIONS
# ============================================================

b1_confusions = confusion_df[
    confusion_df["Experiment"] == BASELINE_CANDIDATE
].copy()

b1_confusions = b1_confusions.sort_values(
    [
        "Misclassified_Count",
        "Class_Error_Rate",
    ],
    ascending=[
        False,
        False,
    ]
).reset_index(drop=True)

print("\nTop B1 Random Forest confusion pairs:")

print(
    b1_confusions[
        [
            "True_Label",
            "Predicted_Label",
            "Misclassified_Count",
            "True_Class_Support",
            "Class_Error_Rate",
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# 14. B1 RF CLASS-WISE COMPARISON AGAINST B0/B2 RF
# ============================================================

rf_experiments = [
    "B0_original_RandomForest",
    "B1_undersampled_RandomForest",
    "B2_hybrid_RandomForest",
]

rf_class_df = per_class_df[
    per_class_df["Experiment"].isin(rf_experiments)
].copy()

rf_pivot = rf_class_df.pivot_table(
    index=["Class_ID", "Label"],
    columns="Experiment",
    values=["Precision", "Recall", "F1"],
    aggfunc="first",
)

rf_pivot.columns = [
    "_".join(col)
    for col in rf_pivot.columns
]

rf_pivot = rf_pivot.reset_index()


# Add B1 improvements against B0 and B2
for metric in ["Precision", "Recall", "F1"]:

    b0_col = f"{metric}_B0_original_RandomForest"
    b1_col = f"{metric}_B1_undersampled_RandomForest"
    b2_col = f"{metric}_B2_hybrid_RandomForest"

    if all(
        col in rf_pivot.columns
        for col in [b0_col, b1_col, b2_col]
    ):
        rf_pivot[
            f"B1_minus_B0_{metric}"
        ] = (
            rf_pivot[b1_col]
            - rf_pivot[b0_col]
        )

        rf_pivot[
            f"B1_minus_B2_{metric}"
        ] = (
            rf_pivot[b1_col]
            - rf_pivot[b2_col]
        )


rf_comparison_output = (
    OUTPUT_DIR /
    "step20_rf_classwise_comparison.csv"
)

rf_pivot.to_csv(
    rf_comparison_output,
    index=False
)

print(
    f"[OK] Saved RF class-wise comparison: "
    f"{rf_comparison_output}"
)


# ============================================================
# 15. BASELINE-LEVEL COMPARISON
# ============================================================

comparison_metrics = [
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_F1",
]

comparison_df = baseline_df[
    [
        "Experiment",
        "Strategy",
        "Model",
        *comparison_metrics,
    ]
].copy()

comparison_df["Macro_F1_Rank"] = (
    comparison_df["Macro_F1"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

comparison_df["Balanced_Accuracy_Rank"] = (
    comparison_df["Balanced_Accuracy"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

comparison_df["Accuracy_Rank"] = (
    comparison_df["Accuracy"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

comparison_df["Macro_F1_Gap_vs_B1_RF"] = (
    comparison_df["Macro_F1"]
    - float(
        comparison_df.loc[
            comparison_df["Experiment"]
            == BASELINE_CANDIDATE,
            "Macro_F1"
        ].iloc[0]
    )
)


comparison_output = (
    OUTPUT_DIR /
    "step20_baseline_comparison.csv"
)

comparison_df.to_csv(
    comparison_output,
    index=False
)

print(
    f"[OK] Saved baseline comparison: "
    f"{comparison_output}"
)


# ============================================================
# 16. MINORITY CLASS ANALYSIS
# ============================================================

# Define minority classes using the known training-support threshold
# from the established Step 16 distribution.
#
# We deliberately do NOT load the test set.
# The threshold is based on the established training distribution.

MINORITY_SUPPORT_THRESHOLD = 10000

b1_minority = b1_rf[
    b1_rf["Support"] < MINORITY_SUPPORT_THRESHOLD
].copy()

minority_summary = {
    "minority_class_count": int(
        len(b1_minority)
    ),
    "minority_mean_f1": float(
        b1_minority["F1"].mean()
    ),
    "minority_median_f1": float(
        b1_minority["F1"].median()
    ),
    "minority_min_f1": float(
        b1_minority["F1"].min()
    ),
    "minority_max_f1": float(
        b1_minority["F1"].max()
    ),
}


# ============================================================
# 17. RESEARCH-ORIENTED DIFFICULTY IDENTIFICATION
# ============================================================

# Classes are flagged for further optimization if:
#
#   F1 < 0.70
# OR
#   Recall < 0.70
#
# This is an analysis flag, NOT a model-selection rule.

difficult_classes = b1_rf[
    (b1_rf["F1"] < 0.70)
    | (b1_rf["Recall"] < 0.70)
].copy()

difficult_classes = difficult_classes.sort_values(
    ["Difficulty_Score", "Support"],
    ascending=[False, True]
)


difficult_output = (
    OUTPUT_DIR /
    "step20_difficult_classes.csv"
)

difficult_classes.to_csv(
    difficult_output,
    index=False
)

print(
    f"[OK] Saved difficult-class analysis: "
    f"{difficult_output}"
)


# ============================================================
# 18. CREATE MACHINE-READABLE SUMMARY
# ============================================================

best_row = baseline_df.iloc[0]

summary = {
    "step": "20A",
    "status": "completed",
    "purpose": "baseline_error_analysis",
    "test_data_used": False,
    "test_data_modified": False,
    "model_retraining": False,
    "feature_selection": False,
    "balancing": False,
    "hyperparameter_tuning": False,

    "primary_metric": PRIMARY_METRIC,

    "current_baseline_candidate": BASELINE_CANDIDATE,

    "current_baseline_metrics": {
        "accuracy": float(
            baseline_df.loc[
                baseline_df["Experiment"]
                == BASELINE_CANDIDATE,
                "Accuracy"
            ].iloc[0]
        ),
        "balanced_accuracy": float(
            baseline_df.loc[
                baseline_df["Experiment"]
                == BASELINE_CANDIDATE,
                "Balanced_Accuracy"
            ].iloc[0]
        ),
        "macro_precision": float(
            baseline_df.loc[
                baseline_df["Experiment"]
                == BASELINE_CANDIDATE,
                "Macro_Precision"
            ].iloc[0]
        ),
        "macro_recall": float(
            baseline_df.loc[
                baseline_df["Experiment"]
                == BASELINE_CANDIDATE,
                "Macro_Recall"
            ].iloc[0]
        ),
        "macro_f1": float(
            baseline_df.loc[
                baseline_df["Experiment"]
                == BASELINE_CANDIDATE,
                "Macro_F1"
            ].iloc[0]
        ),
        "weighted_f1": float(
            baseline_df.loc[
                baseline_df["Experiment"]
                == BASELINE_CANDIDATE,
                "Weighted_F1"
            ].iloc[0]
        ),
    },

    "minority_summary": minority_summary,

    "difficult_class_count": int(
        len(difficult_classes)
    ),

    "top_confusion_pairs": (
        b1_confusions[
            [
                "True_Label",
                "Predicted_Label",
                "Misclassified_Count",
                "Class_Error_Rate",
            ]
        ]
        .head(10)
        .to_dict(orient="records")
    ),

    "next_stage": (
        "STEP 20B - evidence-based Random Forest "
        "hyperparameter optimization design"
    ),
}


summary_output = (
    OUTPUT_DIR /
    "step20_baseline_error_analysis_summary.json"
)

with open(
    summary_output,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )

print(
    f"[OK] Saved summary: {summary_output}"
)


# ============================================================
# 19. HUMAN-READABLE RESEARCH REPORT
# ============================================================

report_output = (
    OUTPUT_DIR /
    "step20_baseline_error_analysis_report.txt"
)

candidate_metrics = baseline_df[
    baseline_df["Experiment"] == BASELINE_CANDIDATE
].iloc[0]

with open(
    report_output,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 20A - BASELINE ERROR ANALYSIS\n"
    )
    f.write(
        "=" * 78 + "\n\n"
    )

    f.write(
        "Project: Behaviour Based Network Attack Detection System "
        "using Machine Learning\n\n"
    )

    f.write(
        "Dataset: CIC-IDS2018\n"
    )
    f.write(
        "Task: 14-class multiclass classification\n"
    )
    f.write(
        "Evaluation protocol: fixed validation set\n"
    )
    f.write(
        "Test data: untouched\n\n"
    )

    f.write(
        "CURRENT BASELINE\n"
    )
    f.write(
        "-" * 78 + "\n"
    )
    f.write(
        f"Experiment       : {BASELINE_CANDIDATE}\n"
    )
    f.write(
        f"Accuracy         : {candidate_metrics['Accuracy']:.6f}\n"
    )
    f.write(
        f"Balanced Accuracy: "
        f"{candidate_metrics['Balanced_Accuracy']:.6f}\n"
    )
    f.write(
        f"Macro Precision  : "
        f"{candidate_metrics['Macro_Precision']:.6f}\n"
    )
    f.write(
        f"Macro Recall     : "
        f"{candidate_metrics['Macro_Recall']:.6f}\n"
    )
    f.write(
        f"Macro F1         : "
        f"{candidate_metrics['Macro_F1']:.6f}\n"
    )
    f.write(
        f"Weighted F1      : "
        f"{candidate_metrics['Weighted_F1']:.6f}\n\n"
    )

    f.write(
        "MINORITY CLASS SUMMARY\n"
    )
    f.write(
        "-" * 78 + "\n"
    )

    for key, value in minority_summary.items():
        f.write(
            f"{key}: {value}\n"
        )

    f.write("\n")

    f.write(
        "DIFFICULT CLASSES\n"
    )
    f.write(
        "-" * 78 + "\n"
    )

    if difficult_classes.empty:
        f.write(
            "No classes crossed the predefined difficulty flags.\n"
        )
    else:
        for _, row in difficult_classes.iterrows():
            f.write(
                f"{row['Label']} | "
                f"Precision={row['Precision']:.6f} | "
                f"Recall={row['Recall']:.6f} | "
                f"F1={row['F1']:.6f} | "
                f"Support={int(row['Support'])}\n"
            )

    f.write("\n")

    f.write(
        "TOP CONFUSION PAIRS\n"
    )
    f.write(
        "-" * 78 + "\n"
    )

    for _, row in b1_confusions.head(15).iterrows():
        f.write(
            f"{row['True_Label']} -> "
            f"{row['Predicted_Label']} | "
            f"count={int(row['Misclassified_Count'])} | "
            f"error_rate={row['Class_Error_Rate']:.6f}\n"
        )

    f.write("\n")

    f.write(
        "METHODOLOGICAL STATUS\n"
    )
    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        "Step 20A performs analysis only.\n"
    )
    f.write(
        "No model was retrained.\n"
    )
    f.write(
        "No hyperparameter tuning was performed.\n"
    )
    f.write(
        "No test data was loaded.\n"
    )
    f.write(
        "No validation balancing was performed.\n"
    )
    f.write(
        "No feature selection was performed.\n"
    )

    f.write("\n")

    f.write(
        "NEXT STEP\n"
    )
    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        "STEP 20B - Evidence-based Random Forest "
        "hyperparameter optimization design.\n"
    )


print(
    f"[OK] Saved research report: {report_output}"
)


# ============================================================
# 20. FINAL STATUS
# ============================================================

print_section("STEP 20A COMPLETED")

print("No model was retrained.")
print("No test data was used.")
print("No hyperparameter tuning was performed.")
print("Existing Step 18/19 artifacts were analyzed only.")

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.glob("step20_*")):
    print(f"  - {path}")

print("\nNEXT:")
print(
    "STEP 20B - Evidence-based Random Forest "
    "hyperparameter optimization design"
)