"""
STEP 20B - Evidence-Based Random Forest Hyperparameter Optimization Design

Purpose
-------
Design a research-grade Random Forest hyperparameter search space based on
the observed Step 20A error analysis.

IMPORTANT
---------
This step DOES NOT train a model.
This step DOES NOT use the test set.
This step DOES NOT perform hyperparameter tuning.

It only:
1. Reads Step 20A evidence.
2. Identifies difficult classes.
3. Builds a justified RF search space.
4. Saves the tuning protocol for the next experiment.

Research objective
------------------
Improve minority/difficult-class detection, especially:
    - Infiltration
    - FTP-BruteForce
    - DoS attacks-SlowHTTPTest
    - SQL Injection
    - Brute Force -Web

while preserving overall performance and avoiding unnecessary overfitting.
"""

from pathlib import Path
import json
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step20_reports"
)

OUTPUT_DIR = REPORT_DIR


# ============================================================
# INPUT FILES
# ============================================================

ERROR_ANALYSIS_FILE = (
    REPORT_DIR
    / "step20_b1_rf_error_analysis.csv"
)

CONFUSION_FILE = (
    REPORT_DIR
    / "step20_b1_rf_confusion_analysis.csv"
)

DIFFICULT_CLASS_FILE = (
    REPORT_DIR
    / "step20_difficult_classes.csv"
)

BASELINE_COMPARISON_FILE = (
    REPORT_DIR
    / "step20_baseline_comparison.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

TUNING_SPACE_FILE = (
    OUTPUT_DIR
    / "step20b_rf_tuning_search_space.json"
)

TUNING_RATIONALE_FILE = (
    OUTPUT_DIR
    / "step20b_rf_tuning_rationale.csv"
)

TUNING_PROTOCOL_FILE = (
    OUTPUT_DIR
    / "step20b_rf_tuning_protocol.txt"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "step20b_rf_tuning_design_summary.json"
)


# ============================================================
# HELPER
# ============================================================

def check_file(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


# ============================================================
# START
# ============================================================

print("=" * 78)
print("STEP 20B - EVIDENCE-BASED RANDOM FOREST TUNING DESIGN")
print("=" * 78)

print(f"Project root : {PROJECT_ROOT}")
print(f"Report dir   : {REPORT_DIR}")


# ============================================================
# CHECK INPUTS
# ============================================================

print("\n[INPUT FILES]")

check_file(
    ERROR_ANALYSIS_FILE,
    "Step 20A RF error analysis"
)

check_file(
    CONFUSION_FILE,
    "Step 20A confusion analysis"
)

check_file(
    DIFFICULT_CLASS_FILE,
    "Step 20A difficult-class analysis"
)

check_file(
    BASELINE_COMPARISON_FILE,
    "Step 20A baseline comparison"
)

print(f"[OK] {ERROR_ANALYSIS_FILE.name}")
print(f"[OK] {CONFUSION_FILE.name}")
print(f"[OK] {DIFFICULT_CLASS_FILE.name}")
print(f"[OK] {BASELINE_COMPARISON_FILE.name}")


# ============================================================
# LOAD EVIDENCE
# ============================================================

error_df = pd.read_csv(ERROR_ANALYSIS_FILE)
confusion_df = pd.read_csv(CONFUSION_FILE)
difficult_df = pd.read_csv(DIFFICULT_CLASS_FILE)
baseline_df = pd.read_csv(BASELINE_COMPARISON_FILE)


# ============================================================
# DISPLAY BASIC EVIDENCE
# ============================================================

print("\n" + "=" * 78)
print("STEP 20A EVIDENCE")
print("=" * 78)

print("\nDifficult classes detected:")

if "Label" in error_df.columns and "F1" in error_df.columns:
    difficult_view = (
        error_df[
            [
                c
                for c in
                ["Label", "Precision", "Recall", "F1", "Support"]
                if c in error_df.columns
            ]
        ]
        .sort_values("F1", ascending=True)
    )

    print(difficult_view.to_string(index=False))


# ============================================================
# IDENTIFY MAIN FAILURE MODE
# ============================================================

main_confusion = None

if not confusion_df.empty:

    required_columns = {
        "True_Label",
        "Predicted_Label",
        "Misclassified_Count",
        "True_Class_Support",
        "Class_Error_Rate",
    }

    if required_columns.issubset(confusion_df.columns):

        main_confusion = confusion_df.iloc[0]

        print("\n" + "=" * 78)
        print("DOMINANT CONFUSION")
        print("=" * 78)

        print(
            f"True class        : {main_confusion['True_Label']}"
        )

        print(
            f"Predicted as      : {main_confusion['Predicted_Label']}"
        )

        print(
            f"Misclassified     : "
            f"{int(main_confusion['Misclassified_Count'])}"
        )

        print(
            f"Class support     : "
            f"{int(main_confusion['True_Class_Support'])}"
        )

        print(
            f"Error rate        : "
            f"{main_confusion['Class_Error_Rate']:.6f}"
        )


# ============================================================
# RESEARCH-DRIVEN SEARCH SPACE
# ============================================================

"""
Rationale:

n_estimators
------------
More trees generally stabilize RF predictions but increase computation.
We therefore test moderate-to-large forests.

max_depth
---------
Depth controls model complexity.
None can produce highly complex trees and possible overfitting.
Several finite depths are included to test controlled complexity.

min_samples_split
-----------------
Increasing this value prevents extremely small internal partitions.
Useful when trying to improve generalization.

min_samples_leaf
-----------------
One of the most important parameters for this problem.
Larger leaves can reduce overfitting and may improve minority-class
generalization, but excessive values can erase rare attack patterns.

max_features
------------
Controls feature diversity at each split.
sqrt is the standard RF baseline.
log2 and fractional values provide controlled alternatives.

class_weight
------------
The observed error analysis shows severe class imbalance and weak
performance on rare classes.
Balanced weighting is therefore scientifically justified.

bootstrap
---------
Bootstrap sampling is retained because it is fundamental to standard
Random Forest behavior.

criterion
---------
We test gini and entropy/log-loss alternatives where supported.
For broad compatibility, the search space uses gini and entropy.

min_impurity_decrease
---------------------
A very small range is used to investigate whether suppressing weak
splits improves generalization without making the model too restrictive.
"""


search_space = {

    "n_estimators": [
        300,
        500,
        800
    ],

    "max_depth": [
        None,
        20,
        30,
        40
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        5
    ],

    "max_features": [
        "sqrt",
        "log2",
        0.5
    ],

    "criterion": [
        "gini",
        "entropy"
    ],

    "class_weight": [
        None,
        "balanced",
        "balanced_subsample"
    ],

    "bootstrap": [
        True
    ],

    "min_impurity_decrease": [
        0.0,
        1e-7,
        1e-6
    ]
}


# ============================================================
# IMPORTANT TUNING POLICY
# ============================================================

tuning_policy = {

    "training_data": "B1 undersampled training configuration",
    "validation_data": "Validation split only",
    "test_data": "STRICTLY FORBIDDEN during tuning",

    "optimization_method": (
        "RandomizedSearchCV or equivalent controlled randomized search"
    ),

    "cross_validation": (
        "StratifiedKFold with shuffle=True and fixed random_state"
    ),

    "primary_metric": (
        "Macro F1"
    ),

    "secondary_metrics": [
        "Balanced Accuracy",
        "Macro Recall",
        "Macro Precision",
        "Weighted F1"
    ],

    "critical_class_metrics": [
        "Infiltration F1",
        "Infiltration Recall",
        "FTP-BruteForce F1",
        "DoS attacks-SlowHTTPTest F1",
        "SQL Injection F1",
        "Brute Force -Web F1"
    ],

    "reproducibility": {
        "random_state": 42,
        "n_jobs": -1
    },

    "model_selection_rule": (
        "Select the model primarily by validation Macro F1, "
        "while inspecting balanced accuracy and difficult-class "
        "performance before final selection."
    ),

    "test_policy": (
        "The untouched test set must not influence hyperparameter "
        "selection."
    )
}


# ============================================================
# EVIDENCE-BASED RATIONALE TABLE
# ============================================================

rationale_rows = [

    {
        "Parameter": "n_estimators",
        "Evidence": (
            "Baseline RF shows strong majority-class performance but "
            "unstable rare-class behavior."
        ),
        "Research_Rationale": (
            "Increasing tree count can reduce variance and stabilize "
            "ensemble predictions."
        ),
        "Search_Values": "300, 500, 800"
    },

    {
        "Parameter": "max_depth",
        "Evidence": (
            "BENIGN and several large attack classes have near-perfect "
            "scores, while difficult classes remain weak."
        ),
        "Research_Rationale": (
            "Controlled tree depth can test whether reduced complexity "
            "improves generalization."
        ),
        "Search_Values": "None, 20, 30, 40"
    },

    {
        "Parameter": "min_samples_split",
        "Evidence": (
            "Rare attack classes have extremely small supports."
        ),
        "Research_Rationale": (
            "Larger split thresholds can suppress highly specific "
            "partitions that may generalize poorly."
        ),
        "Search_Values": "2, 5, 10"
    },

    {
        "Parameter": "min_samples_leaf",
        "Evidence": (
            "Rare classes such as FTP-BruteForce and SlowHTTPTest "
            "show extremely poor F1."
        ),
        "Research_Rationale": (
            "Testing larger leaf sizes evaluates whether smoother "
            "decision boundaries improve generalization."
        ),
        "Search_Values": "1, 2, 5"
    },

    {
        "Parameter": "max_features",
        "Evidence": (
            "The dataset contains many correlated flow features."
        ),
        "Research_Rationale": (
            "Feature subsampling changes tree diversity and can reduce "
            "correlated-tree behavior."
        ),
        "Search_Values": "sqrt, log2, 0.5"
    },

    {
        "Parameter": "criterion",
        "Evidence": (
            "Current baseline performance is highly asymmetric across "
            "classes."
        ),
        "Research_Rationale": (
            "Different split criteria may produce different class "
            "separability."
        ),
        "Search_Values": "gini, entropy"
    },

    {
        "Parameter": "class_weight",
        "Evidence": (
            "Strong class imbalance and poor minority-class recall."
        ),
        "Research_Rationale": (
            "Balanced weighting explicitly increases the learning "
            "importance of underrepresented classes."
        ),
        "Search_Values": "None, balanced, balanced_subsample"
    },

    {
        "Parameter": "min_impurity_decrease",
        "Evidence": (
            "The model separates several dominant classes extremely "
            "well but struggles with minority classes."
        ),
        "Research_Rationale": (
            "A small regularization threshold can test whether weak "
            "splits are contributing to overfitting."
        ),
        "Search_Values": "0, 1e-7, 1e-6"
    }
]

rationale_df = pd.DataFrame(rationale_rows)


# ============================================================
# SAVE SEARCH SPACE
# ============================================================

with open(TUNING_SPACE_FILE, "w", encoding="utf-8") as f:

    json.dump(
        {
            "experiment": "Step 20B",
            "model": "RandomForestClassifier",
            "search_space": search_space,
            "tuning_policy": tuning_policy
        },
        f,
        indent=4
    )


# ============================================================
# SAVE RATIONALE
# ============================================================

rationale_df.to_csv(
    TUNING_RATIONALE_FILE,
    index=False
)


# ============================================================
# SAVE RESEARCH PROTOCOL
# ============================================================

protocol = f"""
STEP 20B
EVIDENCE-BASED RANDOM FOREST HYPERPARAMETER OPTIMIZATION DESIGN
================================================================

PROJECT
-------
Behaviour Based Network Attack Detection System using Machine Learning

DATASET STAGE
-------------
CIC-IDS2018

CURRENT BASELINE
----------------
B1 undersampled Random Forest

STEP 20A FINDINGS
-----------------
The baseline error analysis identified severe class-level weaknesses.

Most important difficult classes include:

1. Infiltration
2. FTP-BruteForce
3. DoS attacks-SlowHTTPTest
4. SQL Injection
5. Brute Force -Web

The dominant confusion pattern is:

BENIGN -> Infiltration
and
Infiltration -> BENIGN

This indicates that the principal problem is not simply overall
classification accuracy. The key research problem is minority/difficult
class discrimination.

TUNING OBJECTIVE
----------------
Improve multiclass generalization and minority-class recognition while
avoiding degradation of the strong majority-class performance.

PRIMARY OPTIMIZATION METRIC
---------------------------
Macro F1

WHY MACRO F1?
-------------
Accuracy and weighted F1 are dominated by large classes such as BENIGN.
Macro F1 gives every class equal importance and therefore better reflects
the actual multiclass IDS objective.

SECONDARY METRICS
-----------------
- Balanced Accuracy
- Macro Recall
- Macro Precision
- Weighted F1

CRITICAL CLASS METRICS
----------------------
- Infiltration F1
- Infiltration Recall
- FTP-BruteForce F1
- DoS attacks-SlowHTTPTest F1
- SQL Injection F1
- Brute Force -Web F1

SEARCH SPACE
------------
n_estimators:
    300, 500, 800

max_depth:
    None, 20, 30, 40

min_samples_split:
    2, 5, 10

min_samples_leaf:
    1, 2, 5

max_features:
    sqrt, log2, 0.5

criterion:
    gini, entropy

class_weight:
    None, balanced, balanced_subsample

bootstrap:
    True

min_impurity_decrease:
    0, 1e-7, 1e-6

VALIDATION POLICY
-----------------
Hyperparameter optimization must use training/validation data only.

The untouched test set must NOT be used to select hyperparameters.

REPRODUCIBILITY
---------------
random_state = 42
n_jobs = -1

NEXT EXPERIMENT
---------------
The next step should execute controlled Random Forest optimization using
this predefined search space.

The final model must not be selected using test-set performance.

EXTERNAL VALIDATION
-------------------
UNSW-NB15 remains reserved for the external validation stage after the
final model has been locked.

CICIoT2023 remains the primary IoT dataset in the overall research
architecture.

RESEARCH INTERPRETATION
-----------------------
The purpose of tuning is not to maximize accuracy.

The purpose is to determine whether controlled Random Forest complexity
and class-aware learning can improve difficult attack-class recognition,
particularly Infiltration and rare web/brute-force/slow-DoS classes,
without causing unacceptable degradation of the overall multiclass
classifier.

NO MODEL WAS TRAINED IN STEP 20B.
NO TEST DATA WAS USED IN STEP 20B.
NO HYPERPARAMETER OPTIMIZATION WAS EXECUTED IN STEP 20B.
"""


with open(
    TUNING_PROTOCOL_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(protocol.strip())


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "step": "20B",

    "status": "completed",

    "model": "RandomForestClassifier",

    "dataset": "CIC-IDS2018",

    "baseline": "B1_undersampled_RandomForest",

    "main_problem": (
        "Severe difficult-class performance, especially "
        "Infiltration and rare attack classes."
    ),

    "dominant_confusion": (
        "BENIGN <-> Infiltration"
    ),

    "primary_metric": "Macro F1",

    "secondary_metrics": [
        "Balanced Accuracy",
        "Macro Recall",
        "Macro Precision",
        "Weighted F1"
    ],

    "test_set_used": False,

    "model_trained": False,

    "tuning_executed": False,

    "random_state": 42,

    "next_step": (
        "Execute controlled Random Forest hyperparameter optimization "
        "using the predefined search space."
    )
}

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 78)
print("STEP 20B COMPLETED")
print("=" * 78)

print("\nResearch decision:")
print(
    "Random Forest tuning will prioritize Macro F1 rather than Accuracy."
)

print("\nPrimary difficult-class issue:")
print(
    "Infiltration ↔ BENIGN confusion"
)

print("\nSearch parameters:")
for parameter, values in search_space.items():
    print(f"  {parameter}: {values}")

print("\nGenerated files:")

print(
    f"  - {TUNING_SPACE_FILE}"
)

print(
    f"  - {TUNING_RATIONALE_FILE}"
)

print(
    f"  - {TUNING_PROTOCOL_FILE}"
)

print(
    f"  - {SUMMARY_FILE}"
)

print("\nIMPORTANT:")
print("  - No model was trained.")
print("  - No test data was used.")
print("  - No hyperparameter tuning was executed.")
print("  - Search space was designed from Step 20A evidence.")

print("\nNEXT:")
print(
    "STEP 21 - Controlled Random Forest hyperparameter optimization"
)

print("=" * 78)