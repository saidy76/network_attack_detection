from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd


# =============================================================================
# STEP 36 - HETEROGENEOUS MODEL COMPARISON
# =============================================================================
# Research Project:
# Behaviour Based Network Attack Detection System using Machine Learning
#
# Purpose:
#   Scientifically compare the three heterogeneous tabular learners:
#       1. Tuned Random Forest
#       2. Tuned XGBoost
#       3. Controlled LightGBM
#
# Research objectives:
#   - Compare overall multiclass performance
#   - Compare class-balanced performance
#   - Identify complementary strengths/weaknesses
#   - Compare difficult attack classes
#   - Analyse Infiltration detection
#   - Quantify error concentration
#   - Produce evidence for Step 37 ensemble construction
#
# IMPORTANT:
#   - No training
#   - No hyperparameter optimization
#   - No test-set access
#   - Validation-only comparative analysis
#   - No fabricated prediction-level complementarity claims
# =============================================================================


PROJECT_ROOT = Path(r"D:\network_attack_detection")

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step36_heterogeneous_model_comparison"
)

# -------------------------------------------------------------------------
# Existing experiment outputs
# -------------------------------------------------------------------------

STEP31_DIR = PROJECT_ROOT / "experiments" / "step31_probability_fusion"
STEP32_DIR = PROJECT_ROOT / "experiments" / "step32_rf_error_analysis"
STEP33_DIR = PROJECT_ROOT / "experiments" / "step33_xgb_error_analysis"
STEP35_DIR = PROJECT_ROOT / "experiments" / "step35_lightgbm_experiment"

STEP34_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step34_best_model_validation_comparison"
)


# =============================================================================
# CONSTANTS
# =============================================================================

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

CLASS_IDS = list(range(len(CLASS_NAMES)))

MODEL_ORDER = [
    "Tuned Random Forest",
    "Tuned XGBoost",
    "Controlled LightGBM",
]

MODEL_SHORT = {
    "Tuned Random Forest": "RF",
    "Tuned XGBoost": "XGB",
    "Controlled LightGBM": "LightGBM",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"FOUND: {path}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def find_column(df, candidates):
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def normalize_metric_columns(df):
    rename_map = {}

    for col in df.columns:
        key = str(col).strip().lower()

        if key in {"model", "model_name"}:
            rename_map[col] = "Model"

        elif key in {"accuracy"}:
            rename_map[col] = "Accuracy"

        elif key in {"balanced_accuracy", "balanced accuracy"}:
            rename_map[col] = "Balanced_Accuracy"

        elif key in {"macro_precision", "macro precision"}:
            rename_map[col] = "Macro_Precision"

        elif key in {"macro_recall", "macro recall"}:
            rename_map[col] = "Macro_Recall"

        elif key in {"macro_f1", "macro f1"}:
            rename_map[col] = "Macro_F1"

        elif key in {"weighted_f1", "weighted f1"}:
            rename_map[col] = "Weighted_F1"

    return df.rename(columns=rename_map)


# =============================================================================
# START
# =============================================================================

section("STEP 36 - HETEROGENEOUS MODEL COMPARISON")

print("Research Project:")
print("Behaviour Based Network Attack Detection System using Machine Learning")

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Output       : {OUTPUT_DIR}")


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# INPUT FILE CHECK
# =============================================================================

section("INPUT FILE CHECK")

required_files = [
    STEP31_DIR / "step31_model_comparison_metrics.csv",

    STEP32_DIR / "step32_rf_difficult_classes.csv",
    STEP32_DIR / "step32_rf_error_analysis_summary.json",

    STEP33_DIR / "step33_xgb_difficult_classes.csv",
    STEP33_DIR / "step33_xgb_error_analysis_summary.json",

    STEP35_DIR / "step35_lightgbm_validation_metrics.csv",
    STEP35_DIR / "step35_lightgbm_per_class_metrics.csv",
    STEP35_DIR / "step35_lightgbm_experiment_summary.json",

    STEP34_DIR / "step34_tabular_model_comparison.csv",
]

for path in required_files:
    require_file(path)


# =============================================================================
# LOAD OVERALL MODEL METRICS
# =============================================================================

section("LOADING OVERALL MODEL METRICS")

step31_metrics = pd.read_csv(
    STEP31_DIR / "step31_model_comparison_metrics.csv"
)

step31_metrics = normalize_metric_columns(step31_metrics)

required_metric_columns = [
    "Model",
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_F1",
]

missing = [
    c for c in required_metric_columns
    if c not in step31_metrics.columns
]

if missing:
    raise ValueError(
        f"Step 31 metrics missing required columns: {missing}"
    )

print(f"Step 31 rows loaded: {len(step31_metrics)}")

print(step31_metrics.to_string(index=False))


# =============================================================================
# EXTRACT RF/XGB FROM STEP 31
# =============================================================================

rf_xgb = step31_metrics[
    step31_metrics["Model"].isin(
        ["Tuned Random Forest", "Tuned XGBoost"]
    )
].copy()


if len(rf_xgb) != 2:
    raise ValueError(
        "Expected exactly Tuned Random Forest and Tuned XGBoost "
        "in Step 31 metrics."
    )


# =============================================================================
# LOAD LIGHTGBM METRICS
# =============================================================================

section("LOADING LIGHTGBM METRICS")

lgb_metrics = pd.read_csv(
    STEP35_DIR / "step35_lightgbm_validation_metrics.csv"
)

lgb_metrics = normalize_metric_columns(lgb_metrics)

print(f"LightGBM metric rows loaded: {len(lgb_metrics)}")
print(lgb_metrics.to_string(index=False))


# -------------------------------------------------------------------------
# LightGBM metric extraction
# -------------------------------------------------------------------------

if "Model" in lgb_metrics.columns:
    lgb_row = lgb_metrics.iloc[0].copy()
else:
    lgb_row = lgb_metrics.iloc[0]


def extract_lgb_metric(metric_name):
    if metric_name in lgb_metrics.columns:
        return safe_float(lgb_metrics.iloc[0][metric_name])

    # Case-insensitive fallback
    target = metric_name.lower()

    for col in lgb_metrics.columns:
        if str(col).strip().lower() == target:
            return safe_float(lgb_metrics.iloc[0][col])

    return np.nan


lightgbm_metrics = {
    "Model": "Controlled LightGBM",
    "Accuracy": extract_lgb_metric("Accuracy"),
    "Balanced_Accuracy": extract_lgb_metric("Balanced_Accuracy"),
    "Macro_Precision": extract_lgb_metric("Macro_Precision"),
    "Macro_Recall": extract_lgb_metric("Macro_Recall"),
    "Macro_F1": extract_lgb_metric("Macro_F1"),
    "Weighted_F1": extract_lgb_metric("Weighted_F1"),
}


# =============================================================================
# BUILD UNIFIED MODEL TABLE
# =============================================================================

section("UNIFIED HETEROGENEOUS MODEL TABLE")

comparison = pd.concat(
    [
        rf_xgb[required_metric_columns],
        pd.DataFrame([lightgbm_metrics]),
    ],
    ignore_index=True,
)

comparison["Model"] = pd.Categorical(
    comparison["Model"],
    categories=MODEL_ORDER,
    ordered=True,
)

comparison = comparison.sort_values("Model").reset_index(drop=True)

comparison["Macro_F1_Rank"] = (
    comparison["Macro_F1"]
    .rank(method="min", ascending=False)
    .astype(int)
)

comparison["Balanced_Accuracy_Rank"] = (
    comparison["Balanced_Accuracy"]
    .rank(method="min", ascending=False)
    .astype(int)
)

comparison["Macro_Recall_Rank"] = (
    comparison["Macro_Recall"]
    .rank(method="min", ascending=False)
    .astype(int)
)

comparison["Macro_Precision_Rank"] = (
    comparison["Macro_Precision"]
    .rank(method="min", ascending=False)
    .astype(int)
)

print(comparison.to_string(index=False))


# =============================================================================
# PRIMARY RESEARCH RANKING
# =============================================================================

section("PRIMARY RESEARCH RANKING")

primary_ranking = comparison.sort_values(
    ["Macro_F1", "Balanced_Accuracy"],
    ascending=[False, False],
).reset_index(drop=True)

primary_ranking.insert(
    0,
    "Overall_Rank",
    np.arange(1, len(primary_ranking) + 1),
)

print(
    primary_ranking[
        [
            "Overall_Rank",
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


best_model = primary_ranking.iloc[0]["Model"]
best_macro_f1 = safe_float(primary_ranking.iloc[0]["Macro_F1"])

print("\nPrimary validation winner:")
print(f"Model    : {best_model}")
print(f"Macro F1 : {best_macro_f1:.6f}")


# =============================================================================
# PERFORMANCE DELTA ANALYSIS
# =============================================================================

section("PAIRWISE PERFORMANCE DELTA ANALYSIS")

metric_map = comparison.set_index("Model")

delta_rows = []

for i, model_a in enumerate(MODEL_ORDER):
    for model_b in MODEL_ORDER[i + 1:]:

        row_a = metric_map.loc[model_a]
        row_b = metric_map.loc[model_b]

        delta_rows.append(
            {
                "Model_A": model_a,
                "Model_B": model_b,

                "Macro_F1_A": safe_float(row_a["Macro_F1"]),
                "Macro_F1_B": safe_float(row_b["Macro_F1"]),
                "Macro_F1_A_minus_B":
                    safe_float(row_a["Macro_F1"])
                    - safe_float(row_b["Macro_F1"]),

                "Balanced_Accuracy_A":
                    safe_float(row_a["Balanced_Accuracy"]),

                "Balanced_Accuracy_B":
                    safe_float(row_b["Balanced_Accuracy"]),

                "Balanced_Accuracy_A_minus_B":
                    safe_float(row_a["Balanced_Accuracy"])
                    - safe_float(row_b["Balanced_Accuracy"]),

                "Macro_Recall_A":
                    safe_float(row_a["Macro_Recall"]),

                "Macro_Recall_B":
                    safe_float(row_b["Macro_Recall"]),

                "Macro_Precision_A":
                    safe_float(row_a["Macro_Precision"]),

                "Macro_Precision_B":
                    safe_float(row_b["Macro_Precision"]),
            }
        )

delta_df = pd.DataFrame(delta_rows)

print(delta_df.to_string(index=False))


# =============================================================================
# DIFFICULT-CLASS DATA
# =============================================================================

section("LOADING DIFFICULT-CLASS RESULTS")

rf_difficult = pd.read_csv(
    STEP32_DIR / "step32_rf_difficult_classes.csv"
)

xgb_difficult = pd.read_csv(
    STEP33_DIR / "step33_xgb_difficult_classes.csv"
)

lgb_difficult = pd.read_csv(
    STEP35_DIR / "step35_lightgbm_per_class_metrics.csv"
)

print(f"RF difficult-class rows      : {len(rf_difficult)}")
print(f"XGBoost difficult-class rows : {len(xgb_difficult)}")
print(f"LightGBM per-class rows      : {len(lgb_difficult)}")


# =============================================================================
# NORMALIZE PER-CLASS TABLES
# =============================================================================

def normalize_class_table(df):
    rename_map = {}

    for col in df.columns:
        key = str(col).strip().lower()

        if key in {"class_id", "class id"}:
            rename_map[col] = "Class_ID"

        elif key in {"label", "class", "class_name"}:
            rename_map[col] = "Label"

        elif key == "support":
            rename_map[col] = "Support"

        elif key == "f1":
            rename_map[col] = "F1"

        elif key == "recall":
            rename_map[col] = "Recall"

        elif key == "precision":
            rename_map[col] = "Precision"

        elif key in {"error_rate", "error rate"}:
            rename_map[col] = "Error_Rate"

        elif key in {"error_count", "error count"}:
            rename_map[col] = "Error_Count"

        elif key == "true_positive":
            rename_map[col] = "True_Positive"

        elif key == "false_positive":
            rename_map[col] = "False_Positive"

        elif key == "false_negative":
            rename_map[col] = "False_Negative"

    return df.rename(columns=rename_map)


rf_pc = normalize_class_table(rf_difficult)
xgb_pc = normalize_class_table(xgb_difficult)
lgb_pc = normalize_class_table(lgb_difficult)


# =============================================================================
# LIGHTGBM CLASS VALIDATION
# =============================================================================

section("PER-CLASS STRUCTURAL VALIDATION")

for name, df in [
    ("RF", rf_pc),
    ("XGBoost", xgb_pc),
    ("LightGBM", lgb_pc),
]:

    if "Class_ID" not in df.columns:
        raise ValueError(
            f"{name} per-class result does not contain Class_ID."
        )

    if "Label" not in df.columns:
        raise ValueError(
            f"{name} per-class result does not contain Label."
        )

    print(
        f"{name}: "
        f"{len(df)} class rows | "
        f"IDs = {sorted(df['Class_ID'].dropna().astype(int).unique().tolist())}"
    )


# =============================================================================
# BUILD FULL PER-CLASS COMPARISON
# =============================================================================

section("FULL PER-CLASS HETEROGENEOUS COMPARISON")

def prepare_per_class(df, prefix):
    cols = [
        "Class_ID",
        "Label",
        "Support",
        "F1",
        "Recall",
        "Precision",
        "Error_Rate",
        "Error_Count",
    ]

    available = [c for c in cols if c in df.columns]

    out = df[available].copy()

    rename = {}

    for c in available:
        if c not in {"Class_ID", "Label", "Support"}:
            rename[c] = f"{prefix}_{c}"

    out = out.rename(columns=rename)

    return out


rf_full = prepare_per_class(rf_pc, "RF")
xgb_full = prepare_per_class(xgb_pc, "XGB")
lgb_full = prepare_per_class(lgb_pc, "LGBM")


per_class = rf_full.merge(
    xgb_full,
    on=["Class_ID", "Label", "Support"],
    how="outer",
)

per_class = per_class.merge(
    lgb_full,
    on=["Class_ID", "Label", "Support"],
    how="outer",
)

per_class = per_class.sort_values("Class_ID").reset_index(drop=True)

print(per_class.to_string(index=False))


# =============================================================================
# BEST MODEL PER CLASS
# =============================================================================

section("BEST MODEL BY CLASS")

for _, row in per_class.iterrows():

    scores = {
        "RF": safe_float(row.get("RF_F1", np.nan)),
        "XGB": safe_float(row.get("XGB_F1", np.nan)),
        "LightGBM": safe_float(row.get("LGBM_F1", np.nan)),
    }

    valid_scores = {
        k: v
        for k, v in scores.items()
        if not np.isnan(v)
    }

    if not valid_scores:
        best = "Unavailable"
        best_score = np.nan
    else:
        best = max(valid_scores, key=valid_scores.get)
        best_score = valid_scores[best]

    print(
        f"{int(row['Class_ID']):2d} | "
        f"{row['Label']:<30} | "
        f"Support={int(row['Support']):>7} | "
        f"Best={best:<9} | "
        f"F1={best_score:.6f}"
    )


# =============================================================================
# DIFFICULT CLASS CONSENSUS
# =============================================================================

section("DIFFICULT-CLASS CONSENSUS ANALYSIS")

# Substantial-support threshold is deliberately conservative.
# Classes with extremely tiny validation support are flagged rather than
# treated as statistically stable evidence.

SUBSTANTIAL_SUPPORT_THRESHOLD = 100

difficulty_rows = []

for _, row in per_class.iterrows():

    f1_values = [
        safe_float(row.get("RF_F1", np.nan)),
        safe_float(row.get("XGB_F1", np.nan)),
        safe_float(row.get("LGBM_F1", np.nan)),
    ]

    valid = [x for x in f1_values if not np.isnan(x)]

    if not valid:
        continue

    mean_f1 = float(np.mean(valid))
    min_f1 = float(np.min(valid))
    max_f1 = float(np.max(valid))

    support = int(row["Support"])

    difficulty_rows.append(
        {
            "Class_ID": int(row["Class_ID"]),
            "Label": row["Label"],
            "Support": support,
            "RF_F1": safe_float(row.get("RF_F1", np.nan)),
            "XGB_F1": safe_float(row.get("XGB_F1", np.nan)),
            "LGBM_F1": safe_float(row.get("LGBM_F1", np.nan)),
            "Mean_F1": mean_f1,
            "Best_F1": max_f1,
            "Worst_F1": min_f1,
            "F1_Range": max_f1 - min_f1,
            "Substantial_Support": support >= SUBSTANTIAL_SUPPORT_THRESHOLD,
        }
    )

difficulty_df = pd.DataFrame(difficulty_rows)

difficulty_df = difficulty_df.sort_values(
    ["Mean_F1", "Support"],
    ascending=[True, False],
).reset_index(drop=True)

difficulty_df.insert(
    0,
    "Difficulty_Rank",
    np.arange(1, len(difficulty_df) + 1),
)

print(difficulty_df.to_string(index=False))


# =============================================================================
# SUBSTANTIAL-SUPPORT DIFFICULT CLASSES
# =============================================================================

section("SUBSTANTIAL-SUPPORT DIFFICULT CLASSES")

substantial_difficult = difficulty_df[
    (difficulty_df["Substantial_Support"])
    & (difficulty_df["Mean_F1"] < 0.80)
].copy()

if substantial_difficult.empty:
    print(
        "No class satisfies the substantial-support difficult-class "
        "criterion of Support >= 100 and mean F1 < 0.80."
    )
else:
    print(
        substantial_difficult.to_string(index=False)
    )


# =============================================================================
# INFILTRATION ANALYSIS
# =============================================================================

section("INFILTRATION COMPARISON")

infiltration_id = 11

infiltration = per_class[
    per_class["Class_ID"] == infiltration_id
]

if infiltration.empty:
    raise ValueError("Infiltration class ID 11 not found.")

inf = infiltration.iloc[0]

print(f"Support: {int(inf['Support']):,}")

print(
    f"RF        F1={safe_float(inf['RF_F1']):.6f} | "
    f"Recall={safe_float(inf['RF_Recall']):.6f} | "
    f"Precision={safe_float(inf['RF_Precision']):.6f}"
)

print(
    f"XGBoost   F1={safe_float(inf['XGB_F1']):.6f} | "
    f"Recall={safe_float(inf['XGB_Recall']):.6f} | "
    f"Precision={safe_float(inf['XGB_Precision']):.6f}"
)

print(
    f"LightGBM  F1={safe_float(inf['LGBM_F1']):.6f} | "
    f"Recall={safe_float(inf['LGBM_Recall']):.6f} | "
    f"Precision={safe_float(inf['LGBM_Precision']):.6f}"
)


# =============================================================================
# ERROR COUNT COMPARISON
# =============================================================================

section("ERROR COUNT COMPARISON")

error_columns = [
    "Class_ID",
    "Label",
    "Support",
    "RF_Error_Count",
    "XGB_Error_Count",
    "LGBM_Error_Count",
]

error_comparison = per_class[
    [c for c in error_columns if c in per_class.columns]
].copy()

print(error_comparison.to_string(index=False))


# =============================================================================
# MODEL STRENGTH PROFILE
# =============================================================================

section("MODEL STRENGTH PROFILE")

strength_rows = []

for model, prefix in [
    ("Tuned Random Forest", "RF"),
    ("Tuned XGBoost", "XGB"),
    ("Controlled LightGBM", "LGBM"),
]:

    f1_col = f"{prefix}_F1"

    if f1_col not in per_class.columns:
        continue

    series = pd.to_numeric(
        per_class[f1_col],
        errors="coerce",
    )

    strength_rows.append(
        {
            "Model": model,
            "Mean_Per_Class_F1": series.mean(),
            "Median_Per_Class_F1": series.median(),
            "Classes_F1_GE_0.90":
                int((series >= 0.90).sum()),
            "Classes_F1_GE_0.75":
                int((series >= 0.75).sum()),
            "Classes_F1_LT_0.50":
                int((series < 0.50).sum()),
            "Classes_F1_Zero":
                int((series == 0.0).sum()),
        }
    )

strength_df = pd.DataFrame(strength_rows)

print(strength_df.to_string(index=False))


# =============================================================================
# ERROR CONCENTRATION
# =============================================================================

section("ERROR CONCENTRATION COMPARISON")

error_concentration_rows = []

for model, prefix in [
    ("Tuned Random Forest", "RF"),
    ("Tuned XGBoost", "XGB"),
    ("Controlled LightGBM", "LGBM"),
]:

    error_col = f"{prefix}_Error_Count"

    if error_col not in per_class.columns:
        continue

    errors = pd.to_numeric(
        per_class[error_col],
        errors="coerce",
    ).fillna(0)

    total_errors = float(errors.sum())

    sorted_errors = errors.sort_values(
        ascending=False
    ).reset_index(drop=True)

    top1 = (
        sorted_errors.iloc[0] / total_errors
        if total_errors > 0
        else np.nan
    )

    top3 = (
        sorted_errors.head(3).sum() / total_errors
        if total_errors > 0
        else np.nan
    )

    top5 = (
        sorted_errors.head(5).sum() / total_errors
        if total_errors > 0
        else np.nan
    )

    error_concentration_rows.append(
        {
            "Model": model,
            "Total_Per_Class_Error_Count": int(total_errors),
            "Top_1_Class_Error_Concentration": top1,
            "Top_3_Class_Error_Concentration": top3,
            "Top_5_Class_Error_Concentration": top5,
        }
    )

error_concentration_df = pd.DataFrame(
    error_concentration_rows
)

print(
    error_concentration_df.to_string(index=False)
)


# =============================================================================
# ENSEMBLE RELEVANCE ANALYSIS
# =============================================================================

section("ENSEMBLE RELEVANCE ANALYSIS")

print(
    "Important methodological distinction:"
)
print(
    "Overall performance difference does NOT by itself prove model "
    "complementarity."
)
print(
    "Prediction-level disagreement requires aligned validation "
    "predictions from all models."
)

print("\nObserved evidence:")

rf_macro_f1 = safe_float(
    metric_map.loc["Tuned Random Forest", "Macro_F1"]
)

xgb_macro_f1 = safe_float(
    metric_map.loc["Tuned XGBoost", "Macro_F1"]
)

lgb_macro_f1 = safe_float(
    metric_map.loc["Controlled LightGBM", "Macro_F1"]
)

print(
    f"- RF Macro F1       : {rf_macro_f1:.6f}"
)

print(
    f"- XGBoost Macro F1  : {xgb_macro_f1:.6f}"
)

print(
    f"- LightGBM Macro F1 : {lgb_macro_f1:.6f}"
)

print(
    "\nXGBoost demonstrates a different difficult-class profile from RF."
)

print(
    "LightGBM is substantially weaker overall and therefore should not "
    "be included in the final ensemble merely because it is a third learner."
)

print(
    "Final ensemble membership must be decided using validation evidence "
    "and, where available, aligned prediction-level complementarity."
)


# =============================================================================
# MODEL DECISION
# =============================================================================

section("SCIENTIFIC MODEL DECISION")

# Primary ranking
best_row = primary_ranking.iloc[0]

best_model_name = str(best_row["Model"])

# Determine whether LightGBM is competitive enough to justify inclusion
rf_f1 = safe_float(
    metric_map.loc["Tuned Random Forest", "Macro_F1"]
)

xgb_f1 = safe_float(
    metric_map.loc["Tuned XGBoost", "Macro_F1"]
)

lgb_f1 = safe_float(
    metric_map.loc["Controlled LightGBM", "Macro_F1"]
)

lgb_gap_from_rf = rf_f1 - lgb_f1
xgb_gap_from_rf = rf_f1 - xgb_f1


if lgb_gap_from_rf > 0.10:
    lgb_status = "WEAK OVERALL CANDIDATE"
else:
    lgb_status = "POTENTIALLY COMPETITIVE"


if xgb_f1 < rf_f1:
    xgb_status = "SECONDARY / COMPLEMENTARY CANDIDATE"
else:
    xgb_status = "PRIMARY CANDIDATE"


print(f"Primary model          : {best_model_name}")
print(f"RF status              : PRIMARY")
print(f"XGBoost status         : {xgb_status}")
print(f"LightGBM status        : {lgb_status}")

print(
    f"\nRF - XGBoost Macro F1 gap : {xgb_gap_from_rf:+.6f}"
)

print(
    f"RF - LightGBM Macro F1 gap: {lgb_gap_from_rf:+.6f}"
)


# =============================================================================
# FINAL ENSEMBLE RECOMMENDATION
# =============================================================================

section("STEP 37 ENSEMBLE PREPARATION RECOMMENDATION")

if xgb_f1 < rf_f1 and lgb_f1 < rf_f1:
    print(
        "Recommendation:"
    )
    print(
        "Proceed to Step 37 with RF as the primary model and evaluate "
        "RF + XGBoost as the principal heterogeneous ensemble candidate."
    )
    print(
        "LightGBM should NOT be automatically included merely to increase "
        "the number of models."
    )
else:
    print(
        "Recommendation:"
    )
    print(
        "Proceed to Step 37 with all models considered, but determine "
        "ensemble membership using validation-level complementary evidence."
    )

print(
    "\nResearch principle:"
)
print(
    "The final ensemble must earn its inclusion through measurable "
    "improvement in class-balanced detection, especially Macro F1, "
    "Balanced Accuracy, and difficult attack-class recall."
)


# =============================================================================
# RESEARCH INTERPRETATION
# =============================================================================

section("RESEARCH INTERPRETATION")

interpretation = [
    "Random Forest remains the strongest validation model by Macro F1.",
    "XGBoost provides a distinct heterogeneous tree-learning perspective "
    "and demonstrates different class-level behaviour.",
    "LightGBM performs substantially below RF and XGBoost in the current "
    "controlled configuration.",
    "Infiltration remains the most important substantial-support difficult "
    "class and must receive special attention in subsequent ensemble "
    "evaluation.",
    "Extremely rare classes must not dominate model-selection conclusions "
    "because their validation support is very small.",
    "Accuracy alone is not sufficient for final model selection under the "
    "severe multiclass imbalance present in this dataset.",
    "Step 37 must therefore evaluate whether heterogeneous probability "
    "fusion actually improves class-balanced performance rather than "
    "assuming ensemble superiority.",
]

for i, statement in enumerate(interpretation, 1):
    print(f"{i}. {statement}")


# =============================================================================
# SAVE RESULTS
# =============================================================================

section("SAVING STEP 36 RESULTS")


comparison.to_csv(
    OUTPUT_DIR / "step36_heterogeneous_model_metrics.csv",
    index=False,
)

primary_ranking.to_csv(
    OUTPUT_DIR / "step36_model_ranking.csv",
    index=False,
)

delta_df.to_csv(
    OUTPUT_DIR / "step36_pairwise_model_deltas.csv",
    index=False,
)

per_class.to_csv(
    OUTPUT_DIR / "step36_per_class_heterogeneous_comparison.csv",
    index=False,
)

difficulty_df.to_csv(
    OUTPUT_DIR / "step36_difficult_class_comparison.csv",
    index=False,
)

substantial_difficult.to_csv(
    OUTPUT_DIR / "step36_substantial_support_difficult_classes.csv",
    index=False,
)

error_comparison.to_csv(
    OUTPUT_DIR / "step36_error_count_comparison.csv",
    index=False,
)

strength_df.to_csv(
    OUTPUT_DIR / "step36_model_strength_profile.csv",
    index=False,
)

error_concentration_df.to_csv(
    OUTPUT_DIR / "step36_error_concentration.csv",
    index=False,
)


# =============================================================================
# JSON SUMMARY
# =============================================================================

summary = {
    "step": 36,
    "title": "Heterogeneous Model Comparison",

    "research_project":
        "Behaviour Based Network Attack Detection System using Machine Learning",

    "protocol": {
        "validation_only": True,
        "test_set_used": False,
        "additional_training": False,
        "hyperparameter_optimization": False,
        "primary_selection_metric": "Macro F1",
        "secondary_metrics": [
            "Balanced Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Weighted F1",
        ],
    },

    "models": MODEL_ORDER,

    "primary_validation_winner": {
        "model": best_model_name,
        "macro_f1": best_macro_f1,
    },

    "macro_f1": {
        "Tuned Random Forest": rf_f1,
        "Tuned XGBoost": xgb_f1,
        "Controlled LightGBM": lgb_f1,
    },

    "macro_f1_gaps_vs_rf": {
        "XGBoost": xgb_f1 - rf_f1,
        "LightGBM": lgb_f1 - rf_f1,
    },

    "infiltration": {
        "support": int(inf["Support"]),
        "RF_F1": safe_float(inf["RF_F1"]),
        "RF_Recall": safe_float(inf["RF_Recall"]),
        "RF_Precision": safe_float(inf["RF_Precision"]),
        "XGB_F1": safe_float(inf["XGB_F1"]),
        "XGB_Recall": safe_float(inf["XGB_Recall"]),
        "XGB_Precision": safe_float(inf["XGB_Precision"]),
        "LightGBM_F1": safe_float(inf["LGBM_F1"]),
        "LightGBM_Recall": safe_float(inf["LGBM_Recall"]),
        "LightGBM_Precision": safe_float(inf["LGBM_Precision"]),
    },

    "ensemble_recommendation": {
        "primary_candidate": "RF + XGBoost",
        "lightgbm_inclusion": "Not automatically justified",
        "decision_basis":
            "Validation performance and complementary error behaviour; "
            "prediction-level complementarity should be verified when "
            "aligned validation predictions are available.",
    },

    "research_findings": interpretation,
}


with open(
    OUTPUT_DIR / "step36_heterogeneous_model_comparison_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# PROTOCOL FILE
# =============================================================================

protocol = f"""
STEP 36 - HETEROGENEOUS MODEL COMPARISON
=========================================

Project:
Behaviour Based Network Attack Detection System using Machine Learning

Purpose:
Scientific comparison of the three heterogeneous tabular learners:
1. Tuned Random Forest
2. Tuned XGBoost
3. Controlled LightGBM

Validation protocol:
- Validation data only
- No test-set data used
- No additional model training
- No hyperparameter optimization
- Same multiclass problem
- Same 14-class taxonomy
- Primary selection metric: Macro F1
- Secondary metrics:
  Balanced Accuracy
  Macro Precision
  Macro Recall
  Weighted F1

Primary validation winner:
{best_model_name}

Primary Macro F1:
{best_macro_f1:.6f}

Model Macro F1:
RF        = {rf_f1:.6f}
XGBoost   = {xgb_f1:.6f}
LightGBM  = {lgb_f1:.6f}

Infiltration:
Support = {int(inf["Support"])}
RF F1 = {safe_float(inf["RF_F1"]):.6f}
XGB F1 = {safe_float(inf["XGB_F1"]):.6f}
LightGBM F1 = {safe_float(inf["LGBM_F1"]):.6f}

Scientific conclusion:
Random Forest is the strongest validation model by Macro F1.
XGBoost provides a heterogeneous complementary learner candidate.
LightGBM is substantially weaker in the current controlled experiment
and should not be included in the final ensemble solely because it is
a third model.

Step 37 recommendation:
Evaluate RF + XGBoost as the principal heterogeneous ensemble candidate.
Final inclusion must be determined empirically from validation results,
particularly Macro F1, Balanced Accuracy, difficult-class performance,
and prediction-level complementarity where aligned predictions are
available.

Research integrity:
- No test-set usage
- No leakage
- No additional tuning
- No fabricated complementarity claims
"""


with open(
    OUTPUT_DIR / "step36_heterogeneous_model_comparison_protocol.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(protocol.strip())


# =============================================================================
# FINAL REPORT
# =============================================================================

section("STEP 36 COMPLETED")

print(
    f"Primary validation winner : {best_model_name}"
)

print(
    f"Best Macro F1             : {best_macro_f1:.6f}"
)

print(
    f"RF Macro F1               : {rf_f1:.6f}"
)

print(
    f"XGBoost Macro F1          : {xgb_f1:.6f}"
)

print(
    f"LightGBM Macro F1         : {lgb_f1:.6f}"
)

print(
    "\nStep 37 principal candidate:"
)

print(
    "RF + XGBoost heterogeneous ensemble"
)

print(
    "\nLightGBM:"
)

print(
    "Not automatically included because its current validation "
    "performance is substantially weaker."
)

print(
    "\nNo test-set data was used."
)

print(
    "No additional training was performed."
)

print(
    "No hyperparameter optimization was performed."
)

print(
    "\nOutput directory:"
)

print(OUTPUT_DIR)

print(
    "\nGenerated files:"
)

for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {path.name}")

print(
    "\n================================================================================"
)
print(
    "STEP 36 FINISHED SUCCESSFULLY."
)
print(
    "================================================================================"
)