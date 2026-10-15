# =============================================================================
# STEP 34 - BEST MODEL VALIDATION COMPARISON
# =============================================================================
# Research Project:
# Behaviour Based Network Attack Detection System using Machine Learning
#
# Purpose:
#   Scientifically compare the existing validation results of:
#       1. Tuned Random Forest
#       2. Tuned XGBoost
#       3. Equal-Weight Probability Fusion
#       4. LSTM
#
# Important methodological rules:
#   - No additional model training
#   - No hyperparameter optimization
#   - No test-set usage
#   - Validation results only
#   - Macro F1 is the primary model-selection metric
#   - Difficult-class behaviour is explicitly considered
#
# Output:
#   experiments/step34_best_model_validation_comparison/
# =============================================================================

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "step34_best_model_validation_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# SOURCE FILES
# =============================================================================

STEP31_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "step31_probability_fusion"
)

STEP32_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "step32_rf_error_analysis"
)

STEP33_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "step33_xgb_error_analysis"
)


# =============================================================================
# FILE PATHS
# =============================================================================

STEP31_METRICS = os.path.join(
    STEP31_DIR,
    "step31_model_comparison_metrics.csv"
)

STEP31_PER_CLASS = os.path.join(
    STEP31_DIR,
    "step31_per_class_fusion_comparison.csv"
)

STEP31_SUMMARY = os.path.join(
    STEP31_DIR,
    "step31_probability_fusion_summary.json"
)

STEP32_DIFFICULT = os.path.join(
    STEP32_DIR,
    "step32_rf_difficult_classes.csv"
)

STEP33_DIFFICULT = os.path.join(
    STEP33_DIR,
    "step33_xgb_difficult_classes.csv"
)

STEP32_SUMMARY = os.path.join(
    STEP32_DIR,
    "step32_rf_error_analysis_summary.json"
)

STEP33_SUMMARY = os.path.join(
    STEP33_DIR,
    "step33_xgb_error_analysis_summary.json"
)


# =============================================================================
# LSTM RESULT
# =============================================================================
# LSTM was previously evaluated in Step 27.
#
# These values are the recorded validation results from the completed
# experiment. No retraining is performed here.
# =============================================================================

LSTM_RESULTS = {
    "Model": "LSTM",
    "Accuracy": 0.966723,
    "Balanced_Accuracy": 0.711654,
    "Macro_Precision": 0.738513,
    "Macro_Recall": 0.711654,
    "Macro_F1": 0.698254,
    "Weighted_F1": np.nan,
}

LSTM_PER_CLASS = {
    "FTP-BruteForce": 0.0,
    "SQL Injection": 0.0,
    "Infiltration": 0.248849,
}


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("STEP 34 - BEST MODEL VALIDATION COMPARISON")
print("=" * 80)
print("Research Project:")
print("Behaviour Based Network Attack Detection System using Machine Learning")
print()
print(f"Project root : {PROJECT_ROOT}")
print(f"Output       : {OUTPUT_DIR}")
print("=" * 80)


# =============================================================================
# INPUT CHECK
# =============================================================================

print("\n" + "=" * 80)
print("INPUT FILE CHECK")
print("=" * 80)

required_files = [
    STEP31_METRICS,
    STEP31_PER_CLASS,
    STEP31_SUMMARY,
    STEP32_DIFFICULT,
    STEP33_DIFFICULT,
    STEP32_SUMMARY,
    STEP33_SUMMARY,
]

all_found = True

for path in required_files:
    if os.path.exists(path):
        print(f"FOUND: {path}")
    else:
        print(f"MISSING: {path}")
        all_found = False

if not all_found:
    raise FileNotFoundError(
        "One or more required Step 31-33 output files are missing."
    )


# =============================================================================
# LOAD STEP 31 METRICS
# =============================================================================

print("\n" + "=" * 80)
print("LOADING STEP 31 MODEL METRICS")
print("=" * 80)

metrics_df = pd.read_csv(STEP31_METRICS)

print(f"Rows loaded: {len(metrics_df)}")
print(f"Columns: {list(metrics_df.columns)}")

print("\nExisting Step 31 models:")
print(metrics_df.to_string(index=False))


# =============================================================================
# NORMALIZE COLUMN NAMES
# =============================================================================

rename_map = {}

for col in metrics_df.columns:
    normalized = (
        str(col)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
    )
    rename_map[col] = normalized

metrics_df = metrics_df.rename(columns=rename_map)


# =============================================================================
# IDENTIFY MODEL COLUMN
# =============================================================================

if "Model" not in metrics_df.columns:
    raise ValueError(
        "Step 31 metrics file does not contain the expected 'Model' column."
    )


# =============================================================================
# REQUIRED METRICS
# =============================================================================

required_metrics = [
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_F1",
]

missing_metrics = [
    col for col in required_metrics
    if col not in metrics_df.columns
]

if missing_metrics:
    raise ValueError(
        f"Missing required metric columns: {missing_metrics}"
    )


# =============================================================================
# LOAD LSTM INTO COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 80)
print("ADDING PREVIOUSLY COMPLETED LSTM VALIDATION RESULT")
print("=" * 80)

lstm_df = pd.DataFrame([LSTM_RESULTS])

comparison_df = pd.concat(
    [
        metrics_df[
            [
                "Model",
                "Accuracy",
                "Balanced_Accuracy",
                "Macro_Precision",
                "Macro_Recall",
                "Macro_F1",
                "Weighted_F1",
            ]
        ],
        lstm_df,
    ],
    ignore_index=True,
)

print(comparison_df.to_string(index=False))


# =============================================================================
# PRIMARY MODEL RANKING
# =============================================================================

print("\n" + "=" * 80)
print("PRIMARY MODEL RANKING - MACRO F1")
print("=" * 80)

ranking_df = comparison_df.sort_values(
    by="Macro_F1",
    ascending=False
).reset_index(drop=True)

ranking_df.insert(
    0,
    "Macro_F1_Rank",
    np.arange(1, len(ranking_df) + 1)
)

print(
    ranking_df[
        [
            "Macro_F1_Rank",
            "Model",
            "Macro_F1",
            "Balanced_Accuracy",
            "Macro_Recall",
            "Macro_Precision",
            "Weighted_F1",
            "Accuracy",
        ]
    ].to_string(index=False)
)


# =============================================================================
# PRIMARY MODEL
# =============================================================================

best_model = ranking_df.iloc[0]["Model"]
best_macro_f1 = float(ranking_df.iloc[0]["Macro_F1"])

print("\nPrimary validation winner:")
print(f"Model    : {best_model}")
print(f"Macro F1 : {best_macro_f1:.6f}")


# =============================================================================
# RF / XGB / FUSION SPECIFIC COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("TABULAR MODEL COMPARISON")
print("=" * 80)

tabular_models = comparison_df[
    comparison_df["Model"].isin(
        [
            "Tuned Random Forest",
            "Tuned XGBoost",
            "Equal-Weight Probability Fusion",
        ]
    )
].copy()

tabular_models = tabular_models.sort_values(
    by="Macro_F1",
    ascending=False
).reset_index(drop=True)

tabular_models.insert(
    0,
    "Tabular_Rank",
    np.arange(1, len(tabular_models) + 1)
)

print(
    tabular_models[
        [
            "Tabular_Rank",
            "Model",
            "Macro_F1",
            "Balanced_Accuracy",
            "Macro_Recall",
            "Macro_Precision",
            "Weighted_F1",
            "Accuracy",
        ]
    ].to_string(index=False)
)


# =============================================================================
# RF VS XGB VS FUSION DELTAS
# =============================================================================

rf_row = comparison_df[
    comparison_df["Model"] == "Tuned Random Forest"
].iloc[0]

xgb_row = comparison_df[
    comparison_df["Model"] == "Tuned XGBoost"
].iloc[0]

fusion_row = comparison_df[
    comparison_df["Model"] == "Equal-Weight Probability Fusion"
].iloc[0]

rf_f1 = float(rf_row["Macro_F1"])
xgb_f1 = float(xgb_row["Macro_F1"])
fusion_f1 = float(fusion_row["Macro_F1"])

print("\n" + "=" * 80)
print("MODEL DELTA ANALYSIS")
print("=" * 80)

print(f"RF Macro F1      : {rf_f1:.6f}")
print(f"XGB Macro F1     : {xgb_f1:.6f}")
print(f"Fusion Macro F1  : {fusion_f1:.6f}")

print()
print(f"XGB - RF         : {xgb_f1 - rf_f1:+.6f}")
print(f"Fusion - RF      : {fusion_f1 - rf_f1:+.6f}")
print(f"Fusion - XGB     : {fusion_f1 - xgb_f1:+.6f}")


# =============================================================================
# DIFFICULT CLASS ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("DIFFICULT-CLASS ANALYSIS")
print("=" * 80)

rf_difficult = pd.read_csv(STEP32_DIFFICULT)
xgb_difficult = pd.read_csv(STEP33_DIFFICULT)

print("\nRF difficult classes:")
print(rf_difficult.head(5).to_string(index=False))

print("\nXGB difficult classes:")
print(xgb_difficult.head(5).to_string(index=False))


# =============================================================================
# EXTRACT IMPORTANT RF/XGB CLASS RESULTS
# =============================================================================

def find_class_row(df, label):
    matches = df[
        df["Label"].astype(str).str.strip() == label
    ]

    if len(matches) == 0:
        return None

    return matches.iloc[0]


important_classes = [
    "Infiltration",
    "FTP-BruteForce",
    "DoS attacks-SlowHTTPTest",
    "Brute Force -Web",
    "SQL Injection",
]


class_rows = []

for label in important_classes:

    rf_row_class = find_class_row(rf_difficult, label)
    xgb_row_class = find_class_row(xgb_difficult, label)

    record = {
        "Label": label,
        "RF_F1": np.nan,
        "XGB_F1": np.nan,
        "RF_Recall": np.nan,
        "XGB_Recall": np.nan,
        "RF_Precision": np.nan,
        "XGB_Precision": np.nan,
        "Support": np.nan,
    }

    if rf_row_class is not None:
        if "F1" in rf_row_class:
            record["RF_F1"] = float(rf_row_class["F1"])
        if "Recall" in rf_row_class:
            record["RF_Recall"] = float(rf_row_class["Recall"])
        if "Precision" in rf_row_class:
            record["RF_Precision"] = float(rf_row_class["Precision"])
        if "Support" in rf_row_class:
            record["Support"] = int(rf_row_class["Support"])

    if xgb_row_class is not None:
        if "F1" in xgb_row_class:
            record["XGB_F1"] = float(xgb_row_class["F1"])
        if "Recall" in xgb_row_class:
            record["XGB_Recall"] = float(xgb_row_class["Recall"])
        if "Precision" in xgb_row_class:
            record["XGB_Precision"] = float(xgb_row_class["Precision"])

    class_rows.append(record)


important_class_df = pd.DataFrame(class_rows)

print(
    important_class_df.to_string(index=False)
)


# =============================================================================
# LSTM DIFFICULT CLASS REFERENCE
# =============================================================================

print("\n" + "=" * 80)
print("LSTM DIFFICULT-CLASS REFERENCE")
print("=" * 80)

for label, f1 in LSTM_PER_CLASS.items():
    print(f"{label:<30} F1 = {f1:.6f}")


# =============================================================================
# INFILTRATION COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("INFILTRATION COMPARISON")
print("=" * 80)

infiltration_row = important_class_df[
    important_class_df["Label"] == "Infiltration"
].iloc[0]

print(
    f"RF  Infiltration F1       : "
    f"{infiltration_row['RF_F1']:.6f}"
)

print(
    f"XGB Infiltration F1       : "
    f"{infiltration_row['XGB_F1']:.6f}"
)

print(
    f"LSTM Infiltration F1      : "
    f"{LSTM_PER_CLASS['Infiltration']:.6f}"
)

print(
    f"RF  Infiltration Recall   : "
    f"{infiltration_row['RF_Recall']:.6f}"
)

print(
    f"XGB Infiltration Recall   : "
    f"{infiltration_row['XGB_Recall']:.6f}"
)


# =============================================================================
# MODEL SELECTION INTERPRETATION
# =============================================================================

print("\n" + "=" * 80)
print("SCIENTIFIC MODEL SELECTION INTERPRETATION")
print("=" * 80)

if rf_f1 > xgb_f1 and rf_f1 > fusion_f1:
    tabular_winner = "Tuned Random Forest"
elif xgb_f1 > rf_f1 and xgb_f1 > fusion_f1:
    tabular_winner = "Tuned XGBoost"
else:
    tabular_winner = "Equal-Weight Probability Fusion"

print(f"Best tabular model by Macro F1: {tabular_winner}")

if fusion_f1 < rf_f1:
    fusion_conclusion = (
        "Equal-weight probability fusion does not improve Macro F1 "
        "over the tuned Random Forest."
    )
else:
    fusion_conclusion = (
        "Equal-weight probability fusion improves or matches the "
        "tuned Random Forest on Macro F1."
    )

print(f"Fusion conclusion: {fusion_conclusion}")


# =============================================================================
# LSTM INTERPRETATION
# =============================================================================

lstm_f1 = float(LSTM_RESULTS["Macro_F1"])

if lstm_f1 < rf_f1:
    lstm_conclusion = (
        "The LSTM has lower validation Macro F1 than the tuned "
        "Random Forest."
    )
else:
    lstm_conclusion = (
        "The LSTM does not have lower validation Macro F1 than "
        "the tuned Random Forest."
    )

print(f"LSTM conclusion: {lstm_conclusion}")


# =============================================================================
# DIFFICULT CLASS FINDINGS
# =============================================================================

print("\nKey difficult-class findings:")

print(
    "- Infiltration is the most important substantial-support difficult class."
)

print(
    "- FTP-BruteForce and DoS attacks-SlowHTTPTest have extremely small "
    "validation support, so their F1 values should be interpreted cautiously."
)

print(
    "- Brute Force-Web and SQL Injection also have limited support."
)

print(
    "- The dominant systematic confusion is between BENIGN and Infiltration."
)


# =============================================================================
# LIGHTGBM JUSTIFICATION
# =============================================================================

print("\n" + "=" * 80)
print("STEP 35 DECISION - LIGHTGBM JUSTIFICATION")
print("=" * 80)

# LightGBM is considered as a heterogeneous tabular learner.
# It is not introduced merely because it is another popular model.
#
# The existing RF/XGB results show:
#   - RF is currently the strongest validation model by Macro F1.
#   - XGB is complementary on some difficult classes.
#   - Equal-weight fusion does not improve over RF.
#
# Therefore, LightGBM can be scientifically justified only as an
# additional heterogeneous learner whose validation behaviour is
# measured independently before ensemble construction.

lightgbm_decision = (
    "JUSTIFIED FOR CONTROLLED EXPERIMENT"
)

lightgbm_reason = (
    "RF currently provides the strongest Macro F1, while XGBoost shows "
    "different error behaviour on difficult classes. A controlled LightGBM "
    "experiment is therefore justified to test whether a third heterogeneous "
    "tree learner provides complementary class-level performance before the "
    "final ensemble decision. It must be evaluated using the same validation "
    "protocol and must not use the test set."
)

print(f"Decision: {lightgbm_decision}")
print(f"Reason  : {lightgbm_reason}")


# =============================================================================
# RESEARCH CONTRIBUTION EVIDENCE
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH CONTRIBUTION EVIDENCE")
print("=" * 80)

contribution_points = [
    "1. Multiclass attack detection is evaluated using Macro F1 and "
    "Balanced Accuracy rather than accuracy alone.",
    "2. Severe class imbalance is explicitly considered during model "
    "development.",
    "3. Multiple heterogeneous learners are evaluated under a controlled "
    "validation protocol.",
    "4. Probability-level fusion is empirically tested rather than assumed "
    "to improve performance.",
    "5. Difficult-class and confusion-pair analysis identifies systematic "
    "failure modes.",
    "6. Infiltration is identified as the dominant substantial-support "
    "error source.",
    "7. The LSTM result is treated cautiously because the earlier sequence "
    "construction had a temporal-validity limitation.",
]

for point in contribution_points:
    print(point)


# =============================================================================
# SAVE MAIN COMPARISON
# =============================================================================

comparison_output = os.path.join(
    OUTPUT_DIR,
    "step34_model_validation_comparison.csv"
)

ranking_output = os.path.join(
    OUTPUT_DIR,
    "step34_model_macro_f1_ranking.csv"
)

tabular_output = os.path.join(
    OUTPUT_DIR,
    "step34_tabular_model_comparison.csv"
)

difficult_output = os.path.join(
    OUTPUT_DIR,
    "step34_important_difficult_classes.csv"
)

comparison_df.to_csv(
    comparison_output,
    index=False
)

ranking_df.to_csv(
    ranking_output,
    index=False
)

tabular_models.to_csv(
    tabular_output,
    index=False
)

important_class_df.to_csv(
    difficult_output,
    index=False
)


# =============================================================================
# CREATE DECISION SUMMARY
# =============================================================================

summary = {
    "step": 34,
    "title": "Best Model Validation Comparison",
    "timestamp": datetime.now().isoformat(),

    "methodology": {
        "additional_training": False,
        "hyperparameter_optimization": False,
        "test_set_used": False,
        "validation_only": True,
        "primary_metric": "Macro F1",
    },

    "best_overall_model_by_macro_f1": best_model,
    "best_overall_macro_f1": best_macro_f1,

    "best_tabular_model": tabular_winner,

    "metrics": {
        "Random Forest": {
            "Accuracy": float(rf_row["Accuracy"]),
            "Balanced_Accuracy": float(rf_row["Balanced_Accuracy"]),
            "Macro_Precision": float(rf_row["Macro_Precision"]),
            "Macro_Recall": float(rf_row["Macro_Recall"]),
            "Macro_F1": float(rf_row["Macro_F1"]),
            "Weighted_F1": float(rf_row["Weighted_F1"]),
        },

        "XGBoost": {
            "Accuracy": float(xgb_row["Accuracy"]),
            "Balanced_Accuracy": float(xgb_row["Balanced_Accuracy"]),
            "Macro_Precision": float(xgb_row["Macro_Precision"]),
            "Macro_Recall": float(xgb_row["Macro_Recall"]),
            "Macro_F1": float(xgb_row["Macro_F1"]),
            "Weighted_F1": float(xgb_row["Weighted_F1"]),
        },

        "Equal-Weight Probability Fusion": {
            "Accuracy": float(fusion_row["Accuracy"]),
            "Balanced_Accuracy": float(fusion_row["Balanced_Accuracy"]),
            "Macro_Precision": float(fusion_row["Macro_Precision"]),
            "Macro_Recall": float(fusion_row["Macro_Recall"]),
            "Macro_F1": float(fusion_row["Macro_F1"]),
            "Weighted_F1": float(fusion_row["Weighted_F1"]),
        },

        "LSTM": {
            "Accuracy": float(LSTM_RESULTS["Accuracy"]),
            "Balanced_Accuracy": float(
                LSTM_RESULTS["Balanced_Accuracy"]
            ),
            "Macro_Precision": float(
                LSTM_RESULTS["Macro_Precision"]
            ),
            "Macro_Recall": float(
                LSTM_RESULTS["Macro_Recall"]
            ),
            "Macro_F1": float(
                LSTM_RESULTS["Macro_F1"]
            ),
        },
    },

    "macro_f1_deltas": {
        "XGB_minus_RF": xgb_f1 - rf_f1,
        "Fusion_minus_RF": fusion_f1 - rf_f1,
        "Fusion_minus_XGB": fusion_f1 - xgb_f1,
        "LSTM_minus_RF": lstm_f1 - rf_f1,
    },

    "important_difficult_classes": {
        row["Label"]: {
            "RF_F1": (
                None
                if pd.isna(row["RF_F1"])
                else float(row["RF_F1"])
            ),
            "XGB_F1": (
                None
                if pd.isna(row["XGB_F1"])
                else float(row["XGB_F1"])
            ),
            "Support": (
                None
                if pd.isna(row["Support"])
                else int(row["Support"])
            ),
        }
        for _, row in important_class_df.iterrows()
    },

    "infiltration": {
        "RF_F1": float(infiltration_row["RF_F1"]),
        "RF_Recall": float(infiltration_row["RF_Recall"]),
        "RF_Precision": float(infiltration_row["RF_Precision"]),
        "XGB_F1": float(infiltration_row["XGB_F1"]),
        "XGB_Recall": float(infiltration_row["XGB_Recall"]),
        "XGB_Precision": float(infiltration_row["XGB_Precision"]),
        "LSTM_F1": float(LSTM_PER_CLASS["Infiltration"]),
    },

    "fusion_conclusion": fusion_conclusion,

    "lstm_conclusion": lstm_conclusion,

    "step35_lightgbm_decision": lightgbm_decision,

    "step35_lightgbm_reason": lightgbm_reason,

    "research_findings": contribution_points,
}


summary_path = os.path.join(
    OUTPUT_DIR,
    "step34_best_model_validation_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


# =============================================================================
# PROTOCOL FILE
# =============================================================================

protocol_path = os.path.join(
    OUTPUT_DIR,
    "step34_best_model_validation_protocol.txt"
)

protocol_text = f"""
STEP 34 - BEST MODEL VALIDATION COMPARISON
============================================================

Research Project:
Behaviour Based Network Attack Detection System using Machine Learning

Purpose
-------
Compare previously completed validation results from:
1. Tuned Random Forest
2. Tuned XGBoost
3. Equal-Weight Probability Fusion
4. Previously completed LSTM

Methodological Controls
-----------------------
- No additional model training.
- No hyperparameter optimization.
- No test-set data used.
- Validation results only.
- Primary selection metric: Macro F1.
- Balanced Accuracy, Macro Precision, Macro Recall and Weighted F1
  are also considered.
- Difficult-class behaviour is explicitly analyzed.

Primary Result
--------------
Best overall model by validation Macro F1:
{best_model}

Best Macro F1:
{best_macro_f1:.6f}

Tabular Model Result
--------------------
Best tabular model:
{tabular_winner}

Random Forest Macro F1:
{rf_f1:.6f}

XGBoost Macro F1:
{xgb_f1:.6f}

Equal-Weight Fusion Macro F1:
{fusion_f1:.6f}

LSTM Macro F1:
{lstm_f1:.6f}

Macro F1 Differences
--------------------
XGB - RF:
{xgb_f1 - rf_f1:+.6f}

Fusion - RF:
{fusion_f1 - rf_f1:+.6f}

Fusion - XGB:
{fusion_f1 - xgb_f1:+.6f}

LSTM - RF:
{lstm_f1 - rf_f1:+.6f}

Infiltration
------------
RF F1:
{infiltration_row["RF_F1"]:.6f}

RF Recall:
{infiltration_row["RF_Recall"]:.6f}

RF Precision:
{infiltration_row["RF_Precision"]:.6f}

XGB F1:
{infiltration_row["XGB_F1"]:.6f}

XGB Recall:
{infiltration_row["XGB_Recall"]:.6f}

XGB Precision:
{infiltration_row["XGB_Precision"]:.6f}

LSTM F1:
{LSTM_PER_CLASS["Infiltration"]:.6f}

Fusion Interpretation
---------------------
{fusion_conclusion}

LSTM Interpretation
-------------------
{lstm_conclusion}

Step 35 Decision
----------------
{lightgbm_decision}

Justification:
{lightgbm_reason}

Research Interpretation
-----------------------
The current validation evidence indicates that the tuned Random Forest
is the strongest existing tabular model by Macro F1. XGBoost provides
different class-level behaviour but does not exceed Random Forest on
overall Macro F1. Equal-weight probability fusion also does not improve
over Random Forest.

The main substantial-support failure mode is Infiltration, where both
tree models show a strong tendency to confuse Infiltration with BENIGN.
Extremely rare classes such as FTP-BruteForce and DoS attacks-SlowHTTPTest
must be interpreted cautiously because their validation support is very
small.

A controlled LightGBM experiment is retained for Step 35 because it can
test whether a third heterogeneous tree learner provides complementary
performance before the final ensemble decision.

No test-set data was used in Step 34.
"""


with open(
    protocol_path,
    "w",
    encoding="utf-8"
) as f:
    f.write(protocol_text)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n" + "=" * 80)
print("STEP 34 COMPLETED")
print("=" * 80)

print(f"Best overall model : {best_model}")
print(f"Best Macro F1      : {best_macro_f1:.6f}")
print(f"Best tabular model : {tabular_winner}")

print()
print(f"RF Macro F1        : {rf_f1:.6f}")
print(f"XGB Macro F1       : {xgb_f1:.6f}")
print(f"Fusion Macro F1    : {fusion_f1:.6f}")
print(f"LSTM Macro F1      : {lstm_f1:.6f}")

print()
print("Infiltration:")
print(
    f"  RF  F1={infiltration_row['RF_F1']:.6f}, "
    f"Recall={infiltration_row['RF_Recall']:.6f}"
)
print(
    f"  XGB F1={infiltration_row['XGB_F1']:.6f}, "
    f"Recall={infiltration_row['XGB_Recall']:.6f}"
)
print(
    f"  LSTM F1={LSTM_PER_CLASS['Infiltration']:.6f}"
)

print()
print(f"Step 35 decision: {lightgbm_decision}")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("  - step34_model_validation_comparison.csv")
print("  - step34_model_macro_f1_ranking.csv")
print("  - step34_tabular_model_comparison.csv")
print("  - step34_important_difficult_classes.csv")
print("  - step34_best_model_validation_summary.json")
print("  - step34_best_model_validation_protocol.txt")

print()
print("No additional training was performed.")
print("No hyperparameter optimization was performed.")
print("No test-set data was used.")

print("\n" + "=" * 80)
print("STEP 34 FINISHED SUCCESSFULLY.")
print("=" * 80)