"""
Step 22 - Tuned Random Forest Comparative Evaluation

Project:
    Behaviour Based Network Attack Detection System using Machine Learning

Purpose:
    Evidence-based comparison of the Step 21 tuned Random Forest
    against the Step 18/20 baseline Random Forest experiments.

Methodological rules:
    - No model retraining.
    - No test data loading.
    - No feature selection.
    - No balancing.
    - No hyperparameter tuning.
    - Existing Step 18/20/21 artifacts only.
    - Validation results remain the basis of comparison.

Author:
    Capstone Research Pipeline

Status:
    STEP 22 - COMPARATIVE EVALUATION
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

STEP20_DIR = EXPERIMENTS_DIR / "step20_reports"
STEP21_DIR = EXPERIMENTS_DIR / "step21_rf_tuning"

OUTPUT_DIR = EXPERIMENTS_DIR / "step22_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. INPUT FILES
# ============================================================

STEP20_BASELINE_COMPARISON = (
    STEP20_DIR / "step20_baseline_comparison.csv"
)

STEP20_RF_CLASSWISE = (
    STEP20_DIR / "step20_rf_classwise_comparison.csv"
)

STEP21_VALIDATION_METRICS = (
    STEP21_DIR / "step21_rf_validation_metrics.csv"
)

STEP21_PER_CLASS = (
    STEP21_DIR / "step21_rf_validation_per_class_metrics.csv"
)

STEP21_CONFUSION = (
    STEP21_DIR / "step21_infiltration_benign_confusion.csv"
)

STEP21_PARAMETERS = (
    STEP21_DIR / "step21_best_rf_parameters.json"
)


# ============================================================
# 3. RESEARCH CONFIGURATION
# ============================================================

BASELINE_CANDIDATE = "B1_undersampled_RandomForest"
TUNED_EXPERIMENT = "Step21_Tuned_RandomForest"

PRIMARY_METRIC = "Macro_F1"

RF_BASELINES = [
    "B0_original_RandomForest",
    "B1_undersampled_RandomForest",
    "B2_hybrid_RandomForest",
]

DIFFICULT_CLASSES = [
    "Brute Force -Web",
    "DoS attacks-SlowHTTPTest",
    "FTP-BruteForce",
    "Infiltration",
    "SQL Injection",
]

COMPARISON_METRICS = [
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_F1",
]


# ============================================================
# 4. CLASS DEFINITIONS
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


def safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )
    return df


def normalize_metric_columns(df):
    """
    Normalize common metric naming variants.
    """

    rename_map = {}

    for col in df.columns:

        normalized = str(col).strip().lower()

        mapping = {
            "accuracy": "Accuracy",

            "balanced accuracy": "Balanced_Accuracy",
            "balanced_accuracy": "Balanced_Accuracy",

            "macro precision": "Macro_Precision",
            "macro_precision": "Macro_Precision",

            "macro recall": "Macro_Recall",
            "macro_recall": "Macro_Recall",

            "macro f1": "Macro_F1",
            "macro_f1": "Macro_F1",

            "weighted f1": "Weighted_F1",
            "weighted_f1": "Weighted_F1",
        }

        if normalized in mapping:
            rename_map[col] = mapping[normalized]

    return df.rename(columns=rename_map)


def normalize_per_class_columns(df):

    rename_map = {}

    for col in df.columns:

        normalized = str(col).strip().lower()

        mapping = {
            "class": "Label",
            "class_name": "Label",

            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "f1-score": "F1",

            "support": "Support",

            "class_id": "Class_ID",
        }

        if normalized in mapping:
            rename_map[col] = mapping[normalized]

    return df.rename(columns=rename_map)


def validate_metric_columns(df):

    required = set(COMPARISON_METRICS)

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required metric columns:\n"
            + "\n".join(sorted(missing))
        )


def percentage_points(value):
    return value * 100.0


# ============================================================
# 6. START
# ============================================================

print_section(
    "STEP 22 - TUNED RANDOM FOREST COMPARATIVE EVALUATION"
)

print(f"Project root : {PROJECT_ROOT}")
print(f"Output dir   : {OUTPUT_DIR}")


# ============================================================
# 7. VERIFY INPUT ARTIFACTS
# ============================================================

print_section("INPUT ARTIFACT VERIFICATION")

require_file(
    STEP20_BASELINE_COMPARISON,
    "Step 20A baseline comparison"
)

require_file(
    STEP20_RF_CLASSWISE,
    "Step 20A RF class-wise comparison"
)

require_file(
    STEP21_VALIDATION_METRICS,
    "Step 21 tuned RF validation metrics"
)

require_file(
    STEP21_PER_CLASS,
    "Step 21 tuned RF validation per-class metrics"
)

require_file(
    STEP21_CONFUSION,
    "Step 21 BENIGN/Infiltration confusion analysis"
)

require_file(
    STEP21_PARAMETERS,
    "Step 21 best RF parameters"
)

print("[OK] All required Step 20/21 artifacts found.")


# ============================================================
# 8. LOAD STEP 20 BASELINE RESULTS
# ============================================================

print_section("LOADING STEP 20 BASELINE RESULTS")

baseline_df = pd.read_csv(
    STEP20_BASELINE_COMPARISON
)

baseline_df = normalize_metric_columns(
    baseline_df
)

validate_metric_columns(
    baseline_df
)

print(
    f"[OK] Loaded {len(baseline_df)} baseline experiments."
)


# ============================================================
# 9. LOAD STEP 21 VALIDATION RESULTS
# ============================================================

print_section("LOADING STEP 21 TUNED RF RESULTS")

tuned_df = pd.read_csv(
    STEP21_VALIDATION_METRICS
)

tuned_df = normalize_metric_columns(
    tuned_df
)

validate_metric_columns(
    tuned_df
)

if tuned_df.empty:
    raise ValueError(
        "Step 21 validation metrics file is empty."
    )

if len(tuned_df) != 1:
    print(
        "[WARNING] Expected one Step 21 tuned RF record, "
        f"found {len(tuned_df)}."
    )

tuned_row = tuned_df.iloc[0].copy()

print(
    "[OK] Loaded Step 21 tuned Random Forest validation result."
)


# ============================================================
# 10. BASELINE CANDIDATE EXTRACTION
# ============================================================

print_section("CURRENT BASELINE IDENTIFICATION")

baseline_rows = baseline_df[
    baseline_df["Experiment"].astype(str)
    == BASELINE_CANDIDATE
].copy()

if baseline_rows.empty:
    raise ValueError(
        f"Could not find baseline candidate: "
        f"{BASELINE_CANDIDATE}"
    )

if len(baseline_rows) > 1:
    raise ValueError(
        f"Multiple records found for {BASELINE_CANDIDATE}"
    )

baseline_row = baseline_rows.iloc[0]

print(
    f"Baseline candidate : {BASELINE_CANDIDATE}"
)

print(
    f"Baseline Macro F1  : "
    f"{baseline_row['Macro_F1']:.6f}"
)

print(
    f"Tuned Macro F1     : "
    f"{tuned_row['Macro_F1']:.6f}"
)


# ============================================================
# 11. OVERALL METRIC COMPARISON
# ============================================================

print_section("OVERALL VALIDATION METRIC COMPARISON")

overall_records = []

for metric in COMPARISON_METRICS:

    baseline_value = float(
        baseline_row[metric]
    )

    tuned_value = float(
        tuned_row[metric]
    )

    absolute_change = (
        tuned_value - baseline_value
    )

    relative_change = (
        absolute_change / baseline_value * 100.0
        if baseline_value != 0
        else np.nan
    )

    overall_records.append(
        {
            "Metric": metric,
            "B1_Baseline": baseline_value,
            "Step21_Tuned_RF": tuned_value,
            "Absolute_Change": absolute_change,
            "Change_Percentage": relative_change,
            "Change_Percentage_Points": (
                absolute_change * 100.0
            ),
        }
    )

overall_comparison_df = pd.DataFrame(
    overall_records
)

print(
    overall_comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE OVERALL COMPARISON
# ============================================================

overall_output = (
    OUTPUT_DIR /
    "step22_overall_metric_comparison.csv"
)

overall_comparison_df.to_csv(
    overall_output,
    index=False
)

print(
    f"\n[OK] Saved: {overall_output}"
)


# ============================================================
# 13. COMPARE AGAINST ALL RF BASELINES
# ============================================================

print_section(
    "TUNED RF VS ALL RANDOM FOREST BASELINES"
)

rf_baselines = baseline_df[
    baseline_df["Experiment"].isin(
        RF_BASELINES
    )
].copy()

if len(rf_baselines) != 3:
    print(
        "[WARNING] Expected 3 RF baselines, "
        f"found {len(rf_baselines)}."
    )

rf_comparison_records = []

for _, row in rf_baselines.iterrows():

    for metric in COMPARISON_METRICS:

        baseline_value = float(
            row[metric]
        )

        tuned_value = float(
            tuned_row[metric]
        )

        rf_comparison_records.append(
            {
                "Baseline_Experiment": row[
                    "Experiment"
                ],
                "Metric": metric,
                "Baseline_Value": baseline_value,
                "Tuned_RF_Value": tuned_value,
                "Absolute_Change": (
                    tuned_value - baseline_value
                ),
                "Change_Percentage_Points": (
                    tuned_value - baseline_value
                ) * 100.0,
            }
        )

rf_comparison_df = pd.DataFrame(
    rf_comparison_records
)

rf_comparison_output = (
    OUTPUT_DIR /
    "step22_tuned_rf_vs_all_rf_baselines.csv"
)

rf_comparison_df.to_csv(
    rf_comparison_output,
    index=False
)

print(
    f"[OK] Saved: {rf_comparison_output}"
)


# ============================================================
# 14. LOAD STEP 20 CLASS-WISE RF RESULTS
# ============================================================

print_section(
    "CLASS-WISE BASELINE DATA"
)

classwise_baseline_df = pd.read_csv(
    STEP20_RF_CLASSWISE
)

classwise_baseline_df = normalize_per_class_columns(
    classwise_baseline_df
)

if "Label" not in classwise_baseline_df.columns:
    raise ValueError(
        "Step 20 class-wise comparison has no Label column."
    )

print(
    f"[OK] Loaded {len(classwise_baseline_df)} "
    "class-wise baseline records."
)


# ============================================================
# 15. LOAD STEP 21 PER-CLASS RESULTS
# ============================================================

print_section(
    "CLASS-WISE TUNED RF RESULTS"
)

tuned_class_df = pd.read_csv(
    STEP21_PER_CLASS,
    index_col=0
)
tuned_class_df = tuned_class_df.reset_index().rename(columns={"index":"Label"})

tuned_class_df = normalize_per_class_columns(
    tuned_class_df
)

tuned_class_df = safe_numeric(tuned_class_df, ["Class_ID", "Precision", "Recall", "F1", "Support"])

if "Label" not in tuned_class_df.columns:
    raise ValueError(
        "Step 21 per-class metrics has no Label column."
    )

required_class_metrics = {
    "Label",
    "Precision",
    "Recall",
    "F1",
    "Support",
}

missing = (
    required_class_metrics
    - set(tuned_class_df.columns)
)

if missing:
    raise ValueError(
        "Step 21 per-class results are missing:\n"
        + "\n".join(sorted(missing))
    )

# Remove aggregate rows such as accuracy/macro avg/weighted avg.

aggregate_labels = {
    "accuracy",
    "macro avg",
    "weighted avg",
}

tuned_class_df = tuned_class_df[
    ~tuned_class_df["Label"]
    .astype(str)
    .str.lower()
    .isin(aggregate_labels)
].copy()

print(
    f"[OK] Loaded {len(tuned_class_df)} "
    "actual class-level tuned RF records."
)


# ============================================================
# 16. BUILD CLASS-WISE COMPARISON
# ============================================================

print_section(
    "BUILDING CLASS-WISE TUNED RF COMPARISON"
)

classwise_records = []

for _, tuned_class in tuned_class_df.iterrows():

    label = str(
        tuned_class["Label"]
    )

    baseline_match = classwise_baseline_df[
        classwise_baseline_df["Label"].astype(str)
        == label
    ]

    if baseline_match.empty:
        print(
            f"[WARNING] No Step 20 baseline record "
            f"found for class: {label}"
        )
        continue

    baseline_class = baseline_match.iloc[0]

    for metric in [
        "Precision",
        "Recall",
        "F1",
    ]:

        baseline_col = (
            f"{metric}_"
            f"{BASELINE_CANDIDATE}"
        )

        if baseline_col not in baseline_class:
            print(
                f"[WARNING] Missing column: "
                f"{baseline_col}"
            )
            continue

        baseline_value = float(
            baseline_class[baseline_col]
        )

        tuned_value = float(
            tuned_class[metric]
        )

        classwise_records.append(
            {
                "Label": label,
                "Metric": metric,
                "B1_Baseline": baseline_value,
                "Step21_Tuned_RF": tuned_value,
                "Absolute_Change": (
                    tuned_value - baseline_value
                ),
                "Change_Percentage_Points": (
                    tuned_value - baseline_value
                ) * 100.0,
                "Tuned_RF_Support": int(
                    tuned_class["Support"]
                ),
            }
        )


classwise_comparison_df = pd.DataFrame(
    classwise_records
)

classwise_output = (
    OUTPUT_DIR /
    "step22_classwise_tuned_rf_comparison.csv"
)

classwise_comparison_df.to_csv(
    classwise_output,
    index=False
)

print(
    f"[OK] Saved: {classwise_output}"
)


# ============================================================
# 17. F1-BASED CLASS-WISE IMPROVEMENT TABLE
# ============================================================

print_section(
    "CLASS-WISE F1 IMPROVEMENT"
)

f1_improvement_df = (
    classwise_comparison_df[
        classwise_comparison_df["Metric"] == "F1"
    ]
    .copy()
    .sort_values(
        "Absolute_Change",
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    f1_improvement_df.to_string(
        index=False
    )
)

f1_output = (
    OUTPUT_DIR /
    "step22_classwise_f1_improvement.csv"
)

f1_improvement_df.to_csv(
    f1_output,
    index=False
)

print(
    f"[OK] Saved: {f1_output}"
)


# ============================================================
# 18. DIFFICULT CLASS ANALYSIS
# ============================================================

print_section(
    "DIFFICULT CLASS ANALYSIS"
)

difficult_df = f1_improvement_df[
    f1_improvement_df["Label"].isin(
        DIFFICULT_CLASSES
    )
].copy()

# Add tuned precision and recall.

for metric in [
    "Precision",
    "Recall",
]:

    subset = classwise_comparison_df[
        classwise_comparison_df["Metric"] == metric
    ][
        [
            "Label",
            "Step21_Tuned_RF",
            "Absolute_Change",
        ]
    ].copy()

    subset = subset.rename(
        columns={
            "Step21_Tuned_RF":
                f"Tuned_{metric}",
            "Absolute_Change":
                f"{metric}_Change",
        }
    )

    difficult_df = difficult_df.merge(
        subset,
        on="Label",
        how="left"
    )

difficult_output = (
    OUTPUT_DIR /
    "step22_difficult_class_comparison.csv"
)

difficult_df.to_csv(
    difficult_output,
    index=False
)

print(
    difficult_df.to_string(
        index=False
    )
)

print(
    f"[OK] Saved: {difficult_output}"
)


# ============================================================
# 19. BENIGN / INFILTRATION ERROR ANALYSIS
# ============================================================

print_section(
    "BENIGN / INFILTRATION ERROR ANALYSIS"
)

confusion_df = pd.read_csv(
    STEP21_CONFUSION
)

required_confusion_columns = {
    "True_Label",
    "Predicted_Label",
    "Misclassified_Count",
    "True_Class_Support",
    "Error_Rate",
}

missing = (
    required_confusion_columns
    - set(confusion_df.columns)
)

if missing:
    raise ValueError(
        "Step 21 confusion artifact missing:\n"
        + "\n".join(sorted(missing))
    )

confusion_df["Misclassified_Count"] = pd.to_numeric(
    confusion_df["Misclassified_Count"],
    errors="coerce"
)

confusion_df["Error_Rate"] = pd.to_numeric(
    confusion_df["Error_Rate"],
    errors="coerce"
)

print(
    confusion_df.to_string(
        index=False
    )
)

confusion_output = (
    OUTPUT_DIR /
    "step22_benign_infiltration_error_analysis.csv"
)

confusion_df.to_csv(
    confusion_output,
    index=False
)

print(
    f"[OK] Saved: {confusion_output}"
)


# ============================================================
# 20. TUNED RF PARAMETERS
# ============================================================

print_section(
    "STEP 21 SELECTED PARAMETERS"
)

with open(
    STEP21_PARAMETERS,
    "r",
    encoding="utf-8"
) as f:
    parameters = json.load(f)

selected_parameters = parameters.get(
    "best_parameters",
    {}
)

print(
    json.dumps(
        selected_parameters,
        indent=4,
        ensure_ascii=False
    )
)


parameters_output = (
    OUTPUT_DIR /
    "step22_tuned_rf_parameters_reference.json"
)

with open(
    parameters_output,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "source": str(
                STEP21_PARAMETERS
            ),
            "experiment": parameters.get(
                "experiment"
            ),
            "optimization_method": parameters.get(
                "optimization_method"
            ),
            "primary_metric": parameters.get(
                "primary_metric"
            ),
            "best_cv_macro_f1": parameters.get(
                "best_cv_macro_f1"
            ),
            "best_parameters": selected_parameters,
        },
        f,
        indent=4,
        ensure_ascii=False
    )

print(
    f"[OK] Saved: {parameters_output}"
)


# ============================================================
# 21. OVERALL DECISION EVIDENCE
# ============================================================

print_section(
    "EVIDENCE-BASED TUNED RF ASSESSMENT"
)

baseline_macro_f1 = float(
    baseline_row["Macro_F1"]
)

tuned_macro_f1 = float(
    tuned_row["Macro_F1"]
)

macro_f1_change = (
    tuned_macro_f1
    - baseline_macro_f1
)

baseline_balanced_accuracy = float(
    baseline_row["Balanced_Accuracy"]
)

tuned_balanced_accuracy = float(
    tuned_row["Balanced_Accuracy"]
)

balanced_accuracy_change = (
    tuned_balanced_accuracy
    - baseline_balanced_accuracy
)

macro_f1_improved = (
    macro_f1_change > 0
)

balanced_accuracy_improved = (
    balanced_accuracy_change > 0
)

print(
    f"Baseline Macro F1       : "
    f"{baseline_macro_f1:.6f}"
)

print(
    f"Tuned RF Macro F1       : "
    f"{tuned_macro_f1:.6f}"
)

print(
    f"Macro F1 change         : "
    f"{macro_f1_change:+.6f} "
    f"({macro_f1_change * 100:+.3f} pp)"
)

print(
    f"Baseline Balanced Acc.  : "
    f"{baseline_balanced_accuracy:.6f}"
)

print(
    f"Tuned RF Balanced Acc.  : "
    f"{tuned_balanced_accuracy:.6f}"
)

print(
    f"Balanced Acc. change    : "
    f"{balanced_accuracy_change:+.6f} "
    f"({balanced_accuracy_change * 100:+.3f} pp)"
)

print(
    f"\nMacro F1 improved       : "
    f"{macro_f1_improved}"
)

print(
    f"Balanced Accuracy improved: "
    f"{balanced_accuracy_improved}"
)


# ============================================================
# 22. MACHINE-READABLE SUMMARY
# ============================================================

summary = {
    "step": "22",
    "status": "completed",
    "purpose": (
        "evidence_based_tuned_random_forest_comparison"
    ),

    "methodological_integrity": {
        "model_retraining": False,
        "test_data_used": False,
        "feature_selection": False,
        "balancing": False,
        "hyperparameter_tuning": False,
        "new_train_validation_split": False,
        "existing_artifacts_only": True,
    },

    "baseline_experiment": BASELINE_CANDIDATE,
    "tuned_experiment": TUNED_EXPERIMENT,
    "primary_metric": PRIMARY_METRIC,

    "baseline_macro_f1": baseline_macro_f1,
    "tuned_macro_f1": tuned_macro_f1,
    "macro_f1_absolute_change": macro_f1_change,
    "macro_f1_change_percentage_points":
        macro_f1_change * 100.0,

    "baseline_balanced_accuracy":
        baseline_balanced_accuracy,

    "tuned_balanced_accuracy":
        tuned_balanced_accuracy,

    "balanced_accuracy_absolute_change":
        balanced_accuracy_change,

    "balanced_accuracy_change_percentage_points":
        balanced_accuracy_change * 100.0,

    "macro_f1_improved":
        macro_f1_improved,

    "balanced_accuracy_improved":
        balanced_accuracy_improved,

    "difficult_classes_analyzed":
        DIFFICULT_CLASSES,

    "next_stage":
        "STEP 23 - controlled XGBoost improvement and evaluation",
}

summary_output = (
    OUTPUT_DIR /
    "step22_tuned_rf_comparative_summary.json"
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
    f"[OK] Saved: {summary_output}"
)


# ============================================================
# 23. HUMAN-READABLE RESEARCH REPORT
# ============================================================

report_output = (
    OUTPUT_DIR /
    "step22_tuned_rf_comparative_report.txt"
)

with open(
    report_output,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 22 - TUNED RANDOM FOREST COMPARATIVE EVALUATION\n"
    )
    f.write("=" * 78 + "\n\n")

    f.write(
        "Project: Behaviour Based Network Attack Detection System "
        "using Machine Learning\n"
    )
    f.write(
        "Dataset: CIC-IDS2018\n"
    )
    f.write(
        "Task: 14-class multiclass classification\n"
    )
    f.write(
        "Evaluation basis: fixed validation set\n"
    )
    f.write(
        "Test data: untouched\n\n"
    )

    f.write(
        "OVERALL COMPARISON\n"
    )
    f.write("-" * 78 + "\n")

    for _, row in overall_comparison_df.iterrows():

        f.write(
            f"{row['Metric']} | "
            f"B1={row['B1_Baseline']:.6f} | "
            f"TunedRF={row['Step21_Tuned_RF']:.6f} | "
            f"Change={row['Absolute_Change']:+.6f} | "
            f"{row['Change_Percentage_Points']:+.3f} pp\n"
        )

    f.write("\n")

    f.write(
        "PRIMARY METRIC ASSESSMENT\n"
    )
    f.write("-" * 78 + "\n")

    f.write(
        f"B1 baseline Macro F1: "
        f"{baseline_macro_f1:.6f}\n"
    )

    f.write(
        f"Step 21 tuned RF Macro F1: "
        f"{tuned_macro_f1:.6f}\n"
    )

    f.write(
        f"Absolute improvement: "
        f"{macro_f1_change:+.6f}\n"
    )

    f.write(
        f"Percentage-point improvement: "
        f"{macro_f1_change * 100:+.3f} pp\n\n"
    )

    f.write(
        "DIFFICULT CLASS FINDINGS\n"
    )
    f.write("-" * 78 + "\n")

    if difficult_df.empty:

        f.write(
            "No difficult-class records were available.\n"
        )

    else:

        for _, row in difficult_df.iterrows():

            f.write(
                f"{row['Label']} | "
                f"F1={row['Step21_Tuned_RF']:.6f} | "
                f"F1 change="
                f"{row['Absolute_Change']:+.6f}\n"
            )

    f.write("\n")

    f.write(
        "BENIGN / INFILTRATION CONFUSION\n"
    )
    f.write("-" * 78 + "\n")

    for _, row in confusion_df.iterrows():

        f.write(
            f"{row['True_Label']} -> "
            f"{row['Predicted_Label']} | "
            f"count={int(row['Misclassified_Count'])} | "
            f"error_rate={row['Error_Rate']:.6f}\n"
        )

    f.write("\n")

    f.write(
        "METHODOLOGICAL STATUS\n"
    )
    f.write("-" * 78 + "\n")

    f.write(
        "No model was retrained in Step 22.\n"
    )

    f.write(
        "No test data was loaded or evaluated.\n"
    )

    f.write(
        "No feature selection was performed.\n"
    )

    f.write(
        "No balancing was performed.\n"
    )

    f.write(
        "No hyperparameter tuning was performed.\n"
    )

    f.write(
        "Only existing Step 20/21 artifacts were analyzed.\n"
    )

    f.write("\n")

    f.write(
        "NEXT STAGE\n"
    )
    f.write("-" * 78 + "\n")

    f.write(
        "STEP 23 - Controlled XGBoost improvement and evaluation.\n"
    )


print(
    f"[OK] Saved: {report_output}"
)


# ============================================================
# 24. FINAL OUTPUT INVENTORY
# ============================================================

print_section(
    "STEP 22 COMPLETED"
)

print(
    "No model was retrained."
)

print(
    "No test data was used."
)

print(
    "No feature selection was performed."
)

print(
    "No balancing was performed."
)

print(
    "No hyperparameter tuning was performed."
)

print(
    "\nGenerated files:"
)

for path in sorted(
    OUTPUT_DIR.glob("step22_*")
):

    print(
        f"  - {path.name}"
    )

print(
    "\nNEXT:"
)

print(
    "STEP 23 - Controlled XGBoost improvement and evaluation"
)