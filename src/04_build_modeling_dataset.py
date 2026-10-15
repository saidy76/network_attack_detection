"""
STEP 4 - MEMORY-EFFICIENT MODELING DATASET CONSTRUCTION

Project:
Behaviour-Based Network Attack Detection using Machine Learning

Dataset:
CICIoT2023

Purpose:
1. Read the 63 CICIoT2023 CSV files in chunks.
2. Preserve the original 34-class attack labels.
3. Create the research-oriented attack-family label.
4. Perform deterministic class-aware sampling.
5. Preserve rare attack classes.
6. Remove exact duplicate rows from the sampled dataset.
7. Create leakage-safe stratified train/validation/test splits.
8. Keep validation and test data untouched.
9. Do NOT apply SMOTE, scaling, feature selection, or modeling.

Important:
The original raw dataset is NEVER modified.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import hashlib


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "CICIoT2023"

DATA_DIR = PROJECT_ROOT / "data"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

MODELING_DIR = DATA_DIR / "modeling"

MODELING_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

LABEL_COLUMN = "Label"

CHUNK_SIZE = 200_000

RANDOM_STATE = 42

# Maximum number of samples retained for each raw class.
# Rare classes are kept completely.
MAX_SAMPLES_PER_CLASS = 60_000

# Split ratios
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# ATTACK FAMILY MAPPING
# ============================================================

def map_attack_family(label):

    if pd.isna(label):
        return "UNKNOWN_LABEL"

    label = str(label).strip().upper()

    if label == "BENIGN":
        return "BENIGN"

    if label.startswith("DDOS-"):
        return "DDOS"

    if label.startswith("DOS-"):
        return "DOS"

    if label.startswith("MIRAI-"):
        return "BOTNET"

    if label.startswith("RECON-"):
        return "RECONNAISSANCE"

    if label in {
        "MITM-ARPSPOOFING",
        "DNS_SPOOFING",
    }:
        return "SPOOFING_MITM"

    if label in {
        "XSS",
        "SQLINJECTION",
        "COMMANDINJECTION",
        "UPLOADING_ATTACK",
        "BROWSERHIJACKING",
    }:
        return "WEB_APPLICATION"

    if label in {
        "DICTIONARYBRUTEFORCE",
    }:
        return "BRUTE_FORCE"

    if label in {
        "BACKDOOR_MALWARE",
    }:
        return "MALWARE_BACKDOOR"

    if label in {
        "VULNERABILITYSCAN",
    }:
        return "VULNERABILITY_SCAN"

    return "OTHER"


# ============================================================
# EXACT ROW HASH
# ============================================================

def create_row_hash(df):

    """
    Create deterministic hashes for duplicate detection.

    Hash includes all columns so that two completely
    identical traffic records receive the same hash.
    """

    hash_values = pd.util.hash_pandas_object(
        df,
        index=False
    )

    return hash_values.values


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 4 - MEMORY-EFFICIENT MODELING DATASET CONSTRUCTION")
    print("=" * 75)

    print("\nDataset:")
    print(RAW_DIR)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"CICIoT2023 directory not found:\n{RAW_DIR}"
        )

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No CSV files found."
        )

    print(f"\nCSV files found: {len(csv_files)}")

    # ========================================================
    # PHASE 1
    # CLASS-AWARE RESERVOIR SAMPLING
    # ========================================================

    print("\n" + "=" * 75)
    print("PHASE 1 - CLASS-AWARE SAMPLING")
    print("=" * 75)

    print(
        f"\nMaximum samples per raw class: "
        f"{MAX_SAMPLES_PER_CLASS:,}"
    )

    print(
        "\nRare classes will be preserved as much as possible."
    )

    rng = np.random.default_rng(RANDOM_STATE)

    # Reservoir storage
    reservoirs = {}

    class_seen = {}

    total_rows_seen = 0

    # --------------------------------------------------------
    # Read every CSV in chunks
    # --------------------------------------------------------

    for file_index, csv_file in enumerate(
        csv_files,
        start=1
    ):

        print(
            f"\n[{file_index}/{len(csv_files)}] "
            f"{csv_file.name}"
        )

        for chunk in pd.read_csv(
            csv_file,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):

            total_rows_seen += len(chunk)

            # Remove rows with missing labels
            chunk = chunk[
                chunk[LABEL_COLUMN].notna()
            ].copy()

            if chunk.empty:
                continue

            # Normalize label strings
            chunk[LABEL_COLUMN] = (
                chunk[LABEL_COLUMN]
                .astype(str)
                .str.strip()
            )

            # ------------------------------------------------
            # Process each label appearing in this chunk
            # ------------------------------------------------

            for label in chunk[LABEL_COLUMN].unique():

                label_rows = chunk[
                    chunk[LABEL_COLUMN] == label
                ].copy()

                if label not in reservoirs:

                    reservoirs[label] = label_rows.copy()

                    class_seen[label] = len(label_rows)

                else:

                    current = reservoirs[label]

                    seen = class_seen[label]

                    incoming = len(label_rows)

                    class_seen[label] = seen + incoming

                    # ----------------------------------------
                    # Reservoir sampling
                    # ----------------------------------------

                    if len(current) < MAX_SAMPLES_PER_CLASS:

                        available_space = (
                            MAX_SAMPLES_PER_CLASS
                            - len(current)
                        )

                        if incoming <= available_space:

                            reservoirs[label] = pd.concat(
                                [
                                    current,
                                    label_rows
                                ],
                                ignore_index=True
                            )

                        else:

                            selected = label_rows.sample(
                                n=available_space,
                                random_state=RANDOM_STATE
                            )

                            reservoirs[label] = pd.concat(
                                [
                                    current,
                                    selected
                                ],
                                ignore_index=True
                            )

                    else:

                        # Reservoir replacement
                        sample_size = min(
                            incoming,
                            MAX_SAMPLES_PER_CLASS
                        )

                        combined = pd.concat(
                            [
                                current,
                                label_rows
                            ],
                            ignore_index=True
                        )

                        # Deterministic downsampling
                        reservoirs[label] = (
                            combined
                            .sample(
                                n=MAX_SAMPLES_PER_CLASS,
                                random_state=RANDOM_STATE
                            )
                            .reset_index(drop=True)
                        )

            del chunk

    # ========================================================
    # SAMPLING SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("SAMPLING SUMMARY")
    print("=" * 75)

    summary_rows = []

    for label in sorted(reservoirs.keys()):

        available = class_seen[label]

        retained = len(reservoirs[label])

        summary_rows.append(
            {
                "raw_label": label,
                "original_count": available,
                "retained_count": retained,
                "sampling_rate_percent":
                    (retained / available) * 100
            }
        )

    sampling_summary = pd.DataFrame(
        summary_rows
    )

    sampling_summary = sampling_summary.sort_values(
        by="retained_count",
        ascending=False
    )

    print(
        sampling_summary.to_string(
            index=False,
            formatters={
                "sampling_rate_percent":
                    "{:.4f}".format
            }
        )
    )

    sampling_summary.to_csv(
        EXPERIMENTS_DIR /
        "step4_sampling_summary.csv",
        index=False
    )

    # ========================================================
    # PHASE 2
    # COMBINE SAMPLED DATA
    # ========================================================

    print("\n" + "=" * 75)
    print("PHASE 2 - COMBINING SAMPLED DATA")
    print("=" * 75)

    sampled_parts = []

    for label in sorted(reservoirs.keys()):

        df = reservoirs[label].copy()

        sampled_parts.append(df)

    combined = pd.concat(
        sampled_parts,
        ignore_index=True
    )

    print(
        f"\nSampled dataset shape: "
        f"{combined.shape}"
    )

    # ========================================================
    # PHASE 3
    # DUPLICATE REMOVAL
    # ========================================================

    print("\n" + "=" * 75)
    print("PHASE 3 - EXACT DUPLICATE REMOVAL")
    print("=" * 75)

    rows_before = len(combined)

    # Exact duplicate across all original columns
    combined = combined.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)

    rows_after = len(combined)

    duplicates_removed = (
        rows_before - rows_after
    )

    print(
        f"\nRows before duplicate removal: "
        f"{rows_before:,}"
    )

    print(
        f"Rows after duplicate removal : "
        f"{rows_after:,}"
    )

    print(
        f"Duplicate rows removed        : "
        f"{duplicates_removed:,}"
    )

    # ========================================================
    # PHASE 4
    # ATTACK FAMILY
    # ========================================================

    print("\n" + "=" * 75)
    print("PHASE 4 - CREATING ATTACK FAMILY")
    print("=" * 75)

    combined["Attack_Family"] = (
        combined[LABEL_COLUMN]
        .apply(map_attack_family)
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("LABEL VALIDATION")
    print("=" * 75)

    unknown = combined[
        combined["Attack_Family"] == "UNKNOWN_LABEL"
    ]

    if len(unknown) > 0:

        print(
            f"[ERROR] Unknown labels found: "
            f"{len(unknown)}"
        )

        raise ValueError(
            "Unknown labels detected."
        )

    print(
        "[OK] All retained labels mapped "
        "to an attack family."
    )

    # ========================================================
    # SHUFFLE
    # ========================================================

    print("\n" + "=" * 75)
    print("PHASE 5 - RANDOMIZATION")
    print("=" * 75)

    combined = combined.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # ========================================================
    # SAVE MASTER MODELING DATASET
    # ========================================================

    master_file = (
        MODELING_DIR /
        "CICIoT2023_modeling_master.csv"
    )

    print("\nSaving master modeling dataset...")

    combined.to_csv(
        master_file,
        index=False
    )

    print(
        f"Saved:\n{master_file}"
    )

    # ========================================================
    # PHASE 6
    # STRATIFIED SPLIT
    # ========================================================

    print("\n" + "=" * 75)
    print("PHASE 6 - STRATIFIED TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 75)

    # --------------------------------------------------------
    # We use the raw attack label for stratification.
    # This ensures every fine-grained class is represented
    # proportionally in all splits.
    # --------------------------------------------------------

    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        combined,
        test_size=(
            VALIDATION_RATIO + TEST_RATIO
        ),
        stratify=combined[LABEL_COLUMN],
        random_state=RANDOM_STATE
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[LABEL_COLUMN],
        random_state=RANDOM_STATE
    )

    # ========================================================
    # SPLIT SHAPES
    # ========================================================

    print("\nTrain shape:")
    print(train_df.shape)

    print("\nValidation shape:")
    print(validation_df.shape)

    print("\nTest shape:")
    print(test_df.shape)

    # ========================================================
    # SPLIT CLASS DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 75)
    print("RAW CLASS DISTRIBUTION")
    print("=" * 75)

    print("\nTRAIN:")

    print(
        train_df[LABEL_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nVALIDATION:")

    print(
        validation_df[LABEL_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nTEST:")

    print(
        test_df[LABEL_COLUMN]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # FAMILY DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 75)
    print("ATTACK FAMILY DISTRIBUTION")
    print("=" * 75)

    print("\nTRAIN FAMILY DISTRIBUTION:")

    print(
        train_df["Attack_Family"]
        .value_counts()
    )

    # ========================================================
    # SAVE SPLITS
    # ========================================================

    print("\n" + "=" * 75)
    print("SAVING MODELING SPLITS")
    print("=" * 75)

    train_file = (
        MODELING_DIR /
        "CICIoT2023_train.csv"
    )

    validation_file = (
        MODELING_DIR /
        "CICIoT2023_validation.csv"
    )

    test_file = (
        MODELING_DIR /
        "CICIoT2023_test.csv"
    )

    train_df.to_csv(
        train_file,
        index=False
    )

    validation_df.to_csv(
        validation_file,
        index=False
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("FINAL VALIDATION")
    print("=" * 75)

    for name, df in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df)
    ]:

        print(f"\n{name}")

        print(
            f"Shape: {df.shape}"
        )

        print(
            f"Missing values: "
            f"{df.isna().sum().sum()}"
        )

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        infinity_count = np.isinf(
            df[numeric_columns]
            .to_numpy()
        ).sum()

        print(
            f"Infinity values: "
            f"{infinity_count}"
        )

        print(
            f"Unique raw classes: "
            f"{df[LABEL_COLUMN].nunique()}"
        )

    # ========================================================
    # CHECK CLASS OVERLAP
    # ========================================================

    train_classes = set(
        train_df[LABEL_COLUMN].unique()
    )

    validation_classes = set(
        validation_df[LABEL_COLUMN].unique()
    )

    test_classes = set(
        test_df[LABEL_COLUMN].unique()
    )

    print("\n" + "=" * 75)
    print("CLASS COVERAGE CHECK")
    print("=" * 75)

    print(
        f"\nTrain classes: "
        f"{len(train_classes)}"
    )

    print(
        f"Validation classes: "
        f"{len(validation_classes)}"
    )

    print(
        f"Test classes: "
        f"{len(test_classes)}"
    )

    if (
        train_classes ==
        validation_classes ==
        test_classes
    ):

        print(
            "\n[OK] All classes are represented "
            "in train, validation and test."
        )

    else:

        print(
            "\n[WARNING] Class coverage differs "
            "between splits."
        )

    # ========================================================
    # COMPLETION
    # ========================================================

    print("\n" + "=" * 75)
    print("STEP 4 COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print("\nCreated files:")

    print(
        f"Master:\n{master_file}"
    )

    print(
        f"\nTrain:\n{train_file}"
    )

    print(
        f"\nValidation:\n{validation_file}"
    )

    print(
        f"\nTest:\n{test_file}"
    )

    print(
        f"\nSampling report:\n"
        f"{EXPERIMENTS_DIR / 'step4_sampling_summary.csv'}"
    )

    print("\nIMPORTANT:")
    print("1. Original CICIoT2023 files were not modified.")
    print("2. Raw 34-class labels were preserved.")
    print("3. Attack-family labels were added.")
    print("4. Class-aware sampling was used.")
    print("5. Exact duplicates were removed from the sampled data.")
    print("6. Stratified 70/15/15 splitting was used.")
    print("7. No SMOTE was applied.")
    print("8. No scaling was applied.")
    print("9. No feature selection was applied.")
    print("10. No model was trained.")
    print("11. Validation and test data remain untouched.")


if __name__ == "__main__":
    main()