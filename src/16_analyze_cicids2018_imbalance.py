import os
import json
import pandas as pd
import numpy as np


# ==============================================================
# STEP 16 - CIC-IDS2018 CLASS IMBALANCE ANALYSIS
# ==============================================================

print("=" * 70)
print("STEP 16 - CIC-IDS2018 CLASS IMBALANCE ANALYSIS")
print("=" * 70)

# --------------------------------------------------------------
# PROJECT PATH
# --------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "modeling"
)

EXPERIMENTS_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments"
)

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


# --------------------------------------------------------------
# INPUT FILES
# --------------------------------------------------------------

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "CICIDS2018_train_selected.csv"
)

VALIDATION_FILE = os.path.join(
    DATA_DIR,
    "CICIDS2018_validation_selected.csv"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "CICIDS2018_test_selected.csv"
)


# --------------------------------------------------------------
# CHECK INPUT FILES
# --------------------------------------------------------------

print()
print("=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

for file_path in [TRAIN_FILE, VALIDATION_FILE, TEST_FILE]:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Input file not found:\n{file_path}"
        )

    print(f"[OK] {os.path.basename(file_path)}")


# --------------------------------------------------------------
# FUNCTION: ANALYZE CLASS DISTRIBUTION
# --------------------------------------------------------------

def analyze_dataset(file_path, dataset_name):

    print()
    print("-" * 70)
    print(f"ANALYZING: {dataset_name}")
    print("-" * 70)

    print("Loading dataset...")

    df = pd.read_csv(file_path)

    print(f"[OK] Dataset loaded.")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    if "Label" not in df.columns:
        raise ValueError(
            f"Label column not found in {dataset_name}"
        )

    # ----------------------------------------------------------
    # CLASS COUNTS
    # ----------------------------------------------------------

    counts = df["Label"].value_counts()

    percentages = (
        df["Label"]
        .value_counts(normalize=True)
        * 100
    )

    result = pd.DataFrame({
        "Label": counts.index,
        "Count": counts.values,
        "Percentage": percentages.values
    })

    result["Percentage"] = result["Percentage"].round(6)

    # ----------------------------------------------------------
    # IMBALANCE RATIO
    # ----------------------------------------------------------

    majority_count = counts.max()
    minority_count = counts.min()

    imbalance_ratio = (
        majority_count / minority_count
    )

    result["Relative_to_Minority"] = (
        result["Count"] / minority_count
    ).round(2)

    result["Relative_to_Majority"] = (
        result["Count"] / majority_count
    ).round(6)

    # ----------------------------------------------------------
    # CLASS CATEGORY
    # ----------------------------------------------------------

    result["Category"] = "Majority"

    # Define thresholds based on percentage
    result.loc[
        result["Percentage"] < 1,
        "Category"
    ] = "Severe Minority"

    result.loc[
        (result["Percentage"] >= 1) &
        (result["Percentage"] < 5),
        "Category"
    ] = "Minority"

    result.loc[
        result["Percentage"] >= 5,
        "Category"
    ] = "Majority"

    # ----------------------------------------------------------
    # PRINT DISTRIBUTION
    # ----------------------------------------------------------

    print()
    print("CLASS DISTRIBUTION")
    print("-" * 70)

    for _, row in result.iterrows():

        print(
            f"{row['Label']:<35} "
            f"{int(row['Count']):>10,} "
            f"({row['Percentage']:>8.4f}%) "
            f"[{row['Category']}]"
        )

    # ----------------------------------------------------------
    # PRINT IMBALANCE INFORMATION
    # ----------------------------------------------------------

    print()
    print("IMBALANCE INFORMATION")
    print("-" * 70)

    print(
        f"Majority class : {counts.idxmax()}"
    )

    print(
        f"Majority count : {majority_count:,}"
    )

    print(
        f"Minority class : {counts.idxmin()}"
    )

    print(
        f"Minority count : {minority_count:,}"
    )

    print(
        f"Imbalance ratio: {imbalance_ratio:,.2f}:1"
    )

    # ----------------------------------------------------------
    # MINORITY CLASS COUNT
    # ----------------------------------------------------------

    severe_minority = result[
        result["Category"] == "Severe Minority"
    ]

    minority = result[
        result["Category"] == "Minority"
    ]

    majority = result[
        result["Category"] == "Majority"
    ]

    print()
    print("CLASS CATEGORY SUMMARY")
    print("-" * 70)

    print(
        f"Majority classes       : {len(majority)}"
    )

    print(
        f"Minority classes       : {len(minority)}"
    )

    print(
        f"Severe minority classes: {len(severe_minority)}"
    )

    return result


# ==============================================================
# ANALYZE TRAINING DATA
# ==============================================================

train_result = analyze_dataset(
    TRAIN_FILE,
    "CIC-IDS2018 TRAINING"
)


# ==============================================================
# ANALYZE VALIDATION DATA
# ==============================================================

validation_result = analyze_dataset(
    VALIDATION_FILE,
    "CIC-IDS2018 VALIDATION"
)


# ==============================================================
# ANALYZE TEST DATA
# ==============================================================

test_result = analyze_dataset(
    TEST_FILE,
    "CIC-IDS2018 TEST"
)


# ==============================================================
# COMPARE ALL SPLITS
# ==============================================================

print()
print("=" * 70)
print("TRAIN / VALIDATION / TEST COMPARISON")
print("=" * 70)

comparison = pd.DataFrame()

comparison["Train"] = (
    train_result
    .set_index("Label")["Count"]
)

comparison["Validation"] = (
    validation_result
    .set_index("Label")["Count"]
)

comparison["Test"] = (
    test_result
    .set_index("Label")["Count"]
)

comparison = comparison.fillna(0).astype(int)

print()
print(comparison.to_string())


# ==============================================================
# TRAINING-ONLY BALANCING ANALYSIS
# ==============================================================

print()
print("=" * 70)
print("TRAINING-ONLY BALANCING ANALYSIS")
print("=" * 70)

print()
print("IMPORTANT:")
print("Validation and Test datasets will NOT be modified.")
print("Only the Training dataset may be balanced later.")


# --------------------------------------------------------------
# TRAINING CLASS STATISTICS
# --------------------------------------------------------------

train_counts = (
    train_result
    .set_index("Label")["Count"]
)

print()
print("TRAINING CLASS STATISTICS")
print("-" * 70)

for label, count in train_counts.items():

    percentage = (
        count / train_counts.sum()
    ) * 100

    if percentage < 0.01:
        status = "EXTREMELY RARE"
    elif percentage < 0.1:
        status = "VERY RARE"
    elif percentage < 1:
        status = "RARE"
    elif percentage < 5:
        status = "MINORITY"
    else:
        status = "MAJORITY"

    print(
        f"{label:<35} "
        f"{count:>10,} "
        f"{percentage:>8.4f}% "
        f"{status}"
    )


# ==============================================================
# RECOMMENDATION
# ==============================================================

print()
print("=" * 70)
print("INITIAL BALANCING RECOMMENDATION")
print("=" * 70)

print()
print("Based on the class distribution:")

print()
print("1. Do NOT balance Validation data.")
print("2. Do NOT balance Test data.")
print("3. Only Training data may be balanced.")
print("4. Extremely rare classes require special handling.")
print("5. SMOTE should NOT blindly be applied to every class.")
print("6. Class weights and controlled sampling will be evaluated.")
print("7. Final balancing strategy will be selected after analysis.")


# ==============================================================
# SAVE CSV REPORT
# ==============================================================

train_report_file = os.path.join(
    EXPERIMENTS_DIR,
    "step16_cicids2018_train_class_distribution.csv"
)

validation_report_file = os.path.join(
    EXPERIMENTS_DIR,
    "step16_cicids2018_validation_class_distribution.csv"
)

test_report_file = os.path.join(
    EXPERIMENTS_DIR,
    "step16_cicids2018_test_class_distribution.csv"
)

comparison_file = os.path.join(
    EXPERIMENTS_DIR,
    "step16_cicids2018_split_comparison.csv"
)


train_result.to_csv(
    train_report_file,
    index=False
)

validation_result.to_csv(
    validation_report_file,
    index=False
)

test_result.to_csv(
    test_report_file,
    index=False
)

comparison.to_csv(
    comparison_file
)

print()
print("=" * 70)
print("SAVING REPORTS")
print("=" * 70)

print(
    f"[OK] Training report:\n{train_report_file}"
)

print(
    f"[OK] Validation report:\n{validation_report_file}"
)

print(
    f"[OK] Test report:\n{test_report_file}"
)

print(
    f"[OK] Split comparison:\n{comparison_file}"
)


# ==============================================================
# SAVE JSON SUMMARY
# ==============================================================

summary = {
    "step": 16,
    "dataset": "CIC-IDS2018",

    "input_files": {
        "train": os.path.basename(TRAIN_FILE),
        "validation": os.path.basename(VALIDATION_FILE),
        "test": os.path.basename(TEST_FILE)
    },

    "train_rows": int(len(train_result)),
    "validation_rows": int(len(validation_result)),
    "test_rows": int(len(test_result)),

    "train_classes": int(len(train_result)),
    "validation_classes": int(len(validation_result)),
    "test_classes": int(len(test_result)),

    "training_majority_class": str(
        train_result.iloc[
            train_result["Count"].idxmax()
        ]["Label"]
    ),

    "training_minority_class": str(
        train_result.iloc[
            train_result["Count"].idxmin()
        ]["Label"]
    ),

    "training_majority_count": int(
        train_result["Count"].max()
    ),

    "training_minority_count": int(
        train_result["Count"].min()
    ),

    "training_imbalance_ratio": float(
        train_result["Count"].max()
        /
        train_result["Count"].min()
    ),

    "severe_minority_classes": (
        train_result[
            train_result["Category"] == "Severe Minority"
        ]["Label"]
        .tolist()
    ),

    "minority_classes": (
        train_result[
            train_result["Category"] == "Minority"
        ]["Label"]
        .tolist()
    ),

    "validation_modified": False,
    "test_modified": False,

    "smote_used": False,
    "model_training": False
}


summary_file = os.path.join(
    EXPERIMENTS_DIR,
    "step16_cicids2018_imbalance_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ==============================================================
# FINAL
# ==============================================================

print()
print("=" * 70)
print("STEP 16 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("CIC-IDS2018 class imbalance analysis completed.")

print()
print("Important:")
print("- Training analyzed")
print("- Validation analyzed")
print("- Test analyzed")
print("- Validation unchanged")
print("- Test unchanged")
print("- No SMOTE")
print("- No model training")

print()
print("Next:")
print("STEP 17 - TRAINING-ONLY CLASS BALANCING STRATEGY")

print()
print("=" * 70)