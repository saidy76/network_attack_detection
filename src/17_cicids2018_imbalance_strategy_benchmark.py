"""
STEP 17 - CIC-IDS2018 IMBALANCE STRATEGY BENCHMARK

Purpose:
    Evaluate candidate training-only imbalance strategies
    without modifying validation or test data.

Dataset:
    CIC-IDS2018

Input:
    data/modeling/CICIDS2018_train_selected.csv

Strategies:
    B0 - Original imbalanced training set
    B1 - Controlled majority undersampling
    B2 - Hybrid controlled sampling
    B3 - Class-weight baseline

IMPORTANT:
    - Validation is NOT used.
    - Test is NOT used.
    - No model training is performed here.
    - No SMOTE/ADASYN is applied here.
    - Original training data is never overwritten.

Outputs:
    data/modeling/balanced/
    experiments/step17_*.csv
    experiments/step17_*.json
"""

import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/modeling/CICIDS2018_train_selected.csv"
)

OUTPUT_DIR = "data/modeling/balanced"
EXPERIMENT_DIR = "experiments"

RANDOM_STATE = 42

# ------------------------------------------------------------
# Controlled sampling targets
# ------------------------------------------------------------
#
# We deliberately DO NOT force every class to the same size.
#
# Majority class:
#   BENIGN -> 500,000
#
# Medium classes:
#   Keep their original counts if below the cap.
#
# Rare classes:
#   Increase them only to controlled minimum levels.
#
# Extremely rare classes (<100 original samples):
#   DO NOT synthetically generate samples here.
#   They are preserved as-is.
# ------------------------------------------------------------

BENIGN_TARGET = 500_000

# Controlled minimum counts for classes having enough
# real observations.
RARE_TARGET = 5_000

VERY_RARE_THRESHOLD = 100


# ============================================================
# DIRECTORY SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("STEP 17 - CIC-IDS2018 IMBALANCE STRATEGY BENCHMARK")
print("=" * 78)

print()
print("Purpose:")
print("Evaluate training-only imbalance strategies.")
print()
print("Validation and Test:")
print("NOT USED")
print()
print("SMOTE:")
print("NOT USED in this stage")
print()


# ============================================================
# CHECK INPUT
# ============================================================

print("=" * 78)
print("CHECKING INPUT FILE")
print("=" * 78)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Training file not found:\n{INPUT_FILE}"
    )

print("[OK] Training file found:")
print(os.path.abspath(INPUT_FILE))


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print()
print("=" * 78)
print("LOADING TRAINING DATA")
print("=" * 78)

start = time.time()

df = pd.read_csv(INPUT_FILE)

print("[OK] Training dataset loaded.")
print(f"Shape: {df.shape}")
print(f"Loading time: {time.time() - start:.2f} sec")


# ============================================================
# BASIC VALIDATION
# ============================================================

print()
print("=" * 78)
print("BASIC VALIDATION")
print("=" * 78)

if "Label" not in df.columns:
    raise ValueError("Label column not found.")

if df["Label"].isna().any():
    raise ValueError("Missing labels detected.")

if df["Label"].astype(str).str.strip().eq("").any():
    raise ValueError("Empty labels detected.")

print("[OK] Label column found.")
print("[OK] No missing labels.")
print("[OK] No empty labels.")

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")
print(f"Features: {len(df.columns) - 1}")
print(f"Classes : {df['Label'].nunique()}")


# ============================================================
# ORIGINAL DISTRIBUTION
# ============================================================

print()
print("=" * 78)
print("ORIGINAL TRAINING CLASS DISTRIBUTION")
print("=" * 78)

original_counts = df["Label"].value_counts()

distribution_rows = []

for label, count in original_counts.items():

    percentage = (count / len(df)) * 100

    if count == original_counts.max():
        category = "MAJORITY"
    elif count < VERY_RARE_THRESHOLD:
        category = "EXTREMELY_RARE"
    elif count < 1000:
        category = "VERY_RARE"
    elif count < 10000:
        category = "RARE"
    else:
        category = "MINORITY"

    distribution_rows.append(
        {
            "Label": label,
            "Count": int(count),
            "Percentage": percentage,
            "Category": category,
        }
    )

original_distribution = pd.DataFrame(distribution_rows)

print(
    original_distribution.to_string(
        index=False,
        formatters={
            "Percentage": "{:.4f}".format
        }
    )
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def controlled_sample(
    data,
    target_counts,
    random_state=42
):
    """
    Create a controlled training sample.

    If target < original:
        random undersampling.

    If target == original:
        keep all samples.

    If target > original:
        random oversampling WITH replacement.

    This function is intentionally used only for
    controlled experimentation.
    """

    rng = np.random.RandomState(random_state)

    parts = []

    for label, target in target_counts.items():

        class_df = data[data["Label"] == label]

        original = len(class_df)

        if original == 0:
            continue

        if target <= original:

            sampled = class_df.sample(
                n=int(target),
                replace=False,
                random_state=random_state
            )

        else:

            indices = rng.choice(
                class_df.index.to_numpy(),
                size=int(target),
                replace=True
            )

            sampled = class_df.loc[indices]

        parts.append(sampled)

    result = pd.concat(
        parts,
        axis=0,
        ignore_index=True
    )

    # Shuffle final dataset
    result = result.sample(
        frac=1.0,
        random_state=random_state
    ).reset_index(drop=True)

    return result


# ============================================================
# STRATEGY B0
# ============================================================

print()
print("=" * 78)
print("B0 - ORIGINAL IMBALANCED TRAINING")
print("=" * 78)

b0 = df.copy()

b0_path = os.path.join(
    OUTPUT_DIR,
    "CICIDS2018_train_B0_original.csv"
)

b0.to_csv(
    b0_path,
    index=False
)

print("[OK] B0 saved:")
print(os.path.abspath(b0_path))


# ============================================================
# STRATEGY B1
# ============================================================

print()
print("=" * 78)
print("B1 - CONTROLLED MAJORITY UNDERSAMPLING")
print("=" * 78)

b1_targets = {}

for label, count in original_counts.items():

    if label.upper() == "BENIGN":
        b1_targets[label] = min(
            BENIGN_TARGET,
            int(count)
        )
    else:
        b1_targets[label] = int(count)

print("Target counts:")

for label, target in b1_targets.items():
    print(f"{label:<35} {target:,}")

start = time.time()

b1 = controlled_sample(
    df,
    b1_targets,
    RANDOM_STATE
)

print()
print(f"B1 rows: {len(b1):,}")
print(f"Processing time: {time.time() - start:.2f} sec")

b1_path = os.path.join(
    OUTPUT_DIR,
    "CICIDS2018_train_B1_undersampled.csv"
)

b1.to_csv(
    b1_path,
    index=False
)

print("[OK] B1 saved:")
print(os.path.abspath(b1_path))


# ============================================================
# STRATEGY B2
# ============================================================

print()
print("=" * 78)
print("B2 - HYBRID CONTROLLED SAMPLING")
print("=" * 78)

b2_targets = {}

for label, count in original_counts.items():

    # Majority class
    if label.upper() == "BENIGN":

        target = BENIGN_TARGET

    # Extremely rare classes
    elif count < VERY_RARE_THRESHOLD:

        # Preserve real observations.
        target = int(count)

    # Other rare classes
    elif count < RARE_TARGET:

        target = RARE_TARGET

    # Larger minority classes
    else:

        target = int(count)

    b2_targets[label] = target

print("Target counts:")

for label, target in b2_targets.items():

    original = int(original_counts[label])

    if target > original:
        action = "OVERSAMPLE"
    elif target < original:
        action = "UNDERSAMPLE"
    else:
        action = "KEEP"

    print(
        f"{label:<35} "
        f"{original:>10,} -> "
        f"{target:>10,} "
        f"[{action}]"
    )

start = time.time()

b2 = controlled_sample(
    df,
    b2_targets,
    RANDOM_STATE
)

print()
print(f"B2 rows: {len(b2):,}")
print(f"Processing time: {time.time() - start:.2f} sec")

b2_path = os.path.join(
    OUTPUT_DIR,
    "CICIDS2018_train_B2_hybrid.csv"
)

b2.to_csv(
    b2_path,
    index=False
)

print("[OK] B2 saved:")
print(os.path.abspath(b2_path))


# ============================================================
# STRATEGY B3 - CLASS WEIGHTS
# ============================================================

print()
print("=" * 78)
print("B3 - CLASS-WEIGHT BASELINE")
print("=" * 78)

classes = np.array(
    sorted(df["Label"].unique())
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["Label"]
)

class_weight_dict = {
    str(label): float(weight)
    for label, weight in zip(classes, weights)
}

b3_rows = []

for label in classes:

    b3_rows.append(
        {
            "Label": label,
            "Original_Count": int(
                original_counts[label]
            ),
            "Class_Weight": class_weight_dict[label]
        }
    )

b3_df = pd.DataFrame(b3_rows)

b3_path = os.path.join(
    EXPERIMENT_DIR,
    "step17_cicids2018_class_weights.csv"
)

b3_df.to_csv(
    b3_path,
    index=False
)

print("[OK] Class weights saved:")
print(os.path.abspath(b3_path))

print()
print(b3_df.to_string(index=False))


# ============================================================
# GENERATE COMPARISON REPORT
# ============================================================

print()
print("=" * 78)
print("GENERATING STRATEGY COMPARISON")
print("=" * 78)

comparison_rows = []

strategies = {
    "B0_Original": b0,
    "B1_Undersampled": b1,
    "B2_Hybrid": b2,
}

for strategy_name, strategy_df in strategies.items():

    counts = strategy_df["Label"].value_counts()

    for label in classes:

        comparison_rows.append(
            {
                "Strategy": strategy_name,
                "Label": label,
                "Count": int(
                    counts.get(label, 0)
                ),
                "Percentage": (
                    counts.get(label, 0)
                    / len(strategy_df)
                    * 100
                )
            }
        )

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_path = os.path.join(
    EXPERIMENT_DIR,
    "step17_cicids2018_strategy_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print("[OK] Strategy comparison saved:")
print(os.path.abspath(comparison_path))


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "step": 17,
    "dataset": "CIC-IDS2018",
    "input": INPUT_FILE,
    "training_only": True,
    "validation_used": False,
    "test_used": False,
    "smote_used": False,
    "random_state": RANDOM_STATE,

    "original_rows": int(len(df)),
    "original_features": int(len(df.columns) - 1),
    "number_of_classes": int(df["Label"].nunique()),

    "strategies": {
        "B0": {
            "name": "Original imbalanced training",
            "sampling": "none"
        },
        "B1": {
            "name": "Controlled majority undersampling",
            "benign_target": BENIGN_TARGET
        },
        "B2": {
            "name": "Hybrid controlled sampling",
            "benign_target": BENIGN_TARGET,
            "rare_target": RARE_TARGET,
            "extremely_rare_policy":
                "preserve original observations"
        },
        "B3": {
            "name": "Class-weight baseline",
            "sampling": "none"
        }
    },

    "outputs": {
        "B0": b0_path,
        "B1": b1_path,
        "B2": b2_path,
        "class_weights": b3_path,
        "comparison": comparison_path
    }
}

summary_path = os.path.join(
    EXPERIMENT_DIR,
    "step17_cicids2018_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

print("[OK] Summary saved:")
print(os.path.abspath(summary_path))


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 78)
print("STEP 17 COMPLETED")
print("=" * 78)

print()
print("Candidate training strategies created:")
print()
print("B0 - Original imbalanced training")
print("B1 - Controlled majority undersampling")
print("B2 - Hybrid controlled sampling")
print("B3 - Class-weight baseline")
print()
print("IMPORTANT:")
print("- Validation was NOT modified.")
print("- Test was NOT modified.")
print("- No SMOTE was applied.")
print("- Original training dataset was NOT overwritten.")
print()
print("Next:")
print("STEP 18 - BASELINE MODEL EXPERIMENTS")
print("=" * 78)