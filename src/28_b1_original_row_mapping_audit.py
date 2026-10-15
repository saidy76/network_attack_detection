# =============================================================================
# STEP 28 - B1 TO ORIGINAL TRAIN ROW MAPPING & TEMPORAL RECOVERABILITY AUDIT
# =============================================================================
#
# Purpose:
#   Determine whether rows in the B1 undersampled training dataset can be
#   reliably mapped back to the original CICIDS2018_train.csv rows using the
#   30 selected modeling features + Label.
#
# This is a DIAGNOSTIC AUDIT ONLY.
# It does NOT modify any existing dataset.
# It does NOT train any ML model.
#
# Main questions:
#   1. How many B1 rows have an exact match in original train?
#   2. How many matches are unique?
#   3. How many are ambiguous because multiple original rows have identical
#      selected-feature values + Label?
#   4. How many B1 rows cannot be matched?
#   5. If a row is uniquely matched, can its original Timestamp be recovered?
#   6. Is reliable row identity recovery possible for temporal reconstruction?
#
# =============================================================================

from pathlib import Path
import sys
import time
import json
import hashlib

import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "CICIDS2018_train.csv"
)

B1_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "balanced"
    / "CICIDS2018_train_B1_undersampled.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step28_mapping_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EXPECTED 30 SELECTED FEATURES
# =============================================================================

SELECTED_FEATURES = [
    "Init Fwd Win Byts",
    "Flow Duration",
    "Fwd Header Len",
    "Dst Port",
    "Bwd Pkts/s",
    "Fwd IAT Tot",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Bwd Pkt Len Mean",
    "Fwd Seg Size Min",
    "Bwd Pkt Len Std",
    "Fwd Pkt Len Max",
    "Fwd IAT Min",
    "Init Bwd Win Byts",
    "Fwd Pkt Len Std",
    "Pkt Len Mean",
    "Tot Fwd Pkts",
    "Flow IAT Std",
    "Flow Byts/s",
    "Bwd IAT Tot",
    "Flow IAT Min",
    "Tot Bwd Pkts",
    "Bwd IAT Min",
    "Fwd IAT Std",
    "Bwd IAT Std",
    "Fwd Act Data Pkts",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "Idle Mean",
    "Down/Up Ratio",
]

LABEL_COL = "Label"
TIMESTAMP_COL = "Timestamp"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_column_names(df):
    """
    Strip leading/trailing whitespace from column names.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def make_canonical_string_series(df, columns):
    """
    Create a deterministic row signature from selected columns.

    Values are converted to strings with enough precision to preserve exact
    pandas CSV-parsed values.

    This is used for efficient matching without brute-force row comparison.
    """

    temp = df[columns].copy()

    # Normalize string columns such as Label.
    for col in columns:
        if temp[col].dtype == object:
            temp[col] = temp[col].astype(str).str.strip()

    # Use pandas' stable representation for numeric values.
    # This preserves the values loaded by pandas rather than introducing
    # arbitrary rounding.
    return temp.astype(str).agg("\x1f".join, axis=1)


def calculate_file_hash(path, block_size=1024 * 1024):
    """
    Calculate SHA-256 hash of a file for audit reproducibility.
    """
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            sha256.update(block)

    return sha256.hexdigest()


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# START
# =============================================================================

start_time = time.time()

print("=" * 80)
print("STEP 28 - B1 TO ORIGINAL TRAIN ROW MAPPING & TEMPORAL RECOVERABILITY AUDIT")
print("=" * 80)

print(f"Project root   : {PROJECT_ROOT}")
print(f"Original train : {ORIGINAL_TRAIN}")
print(f"B1 train       : {B1_TRAIN}")
print(f"Output folder  : {OUTPUT_DIR}")
print()


# =============================================================================
# FILE EXISTENCE CHECK
# =============================================================================

if not ORIGINAL_TRAIN.exists():
    print("ERROR: Original training file was not found.")
    print(f"Expected: {ORIGINAL_TRAIN}")
    sys.exit(1)

if not B1_TRAIN.exists():
    print("ERROR: B1 training file was not found.")
    print(f"Expected: {B1_TRAIN}")
    sys.exit(1)


# =============================================================================
# FILE HASHES
# =============================================================================

print_section("FILE INTEGRITY INFORMATION")

print("Calculating SHA-256 hashes...")
print("This may take some time for large CSV files.")

original_hash = calculate_file_hash(ORIGINAL_TRAIN)
b1_hash = calculate_file_hash(B1_TRAIN)

print(f"Original train SHA-256 : {original_hash}")
print(f"B1 train SHA-256       : {b1_hash}")


# =============================================================================
# LOAD DATA
# =============================================================================

print_section("LOADING DATA")

load_start = time.time()

print("Loading original CICIDS2018_train.csv...")
original_df = pd.read_csv(ORIGINAL_TRAIN, low_memory=False)

print(
    f"Original dataset loaded: "
    f"{original_df.shape[0]:,} rows × {original_df.shape[1]} columns"
)

print("Loading B1 undersampled dataset...")
b1_df = pd.read_csv(B1_TRAIN, low_memory=False)

print(
    f"B1 dataset loaded: "
    f"{b1_df.shape[0]:,} rows × {b1_df.shape[1]} columns"
)

load_time = time.time() - load_start

print(f"Loading time: {load_time:.2f} seconds")


# =============================================================================
# NORMALIZE COLUMN NAMES
# =============================================================================

original_df = normalize_column_names(original_df)
b1_df = normalize_column_names(b1_df)


# =============================================================================
# COLUMN VALIDATION
# =============================================================================

print_section("COLUMN VALIDATION")

required_original = set(SELECTED_FEATURES + [LABEL_COL, TIMESTAMP_COL])
required_b1 = set(SELECTED_FEATURES + [LABEL_COL])

missing_original = [
    col for col in required_original
    if col not in original_df.columns
]

missing_b1 = [
    col for col in required_b1
    if col not in b1_df.columns
]

if missing_original:
    print("ERROR: Missing columns in original training dataset:")
    for col in missing_original:
        print(f"  - {col}")
    sys.exit(1)

if missing_b1:
    print("ERROR: Missing columns in B1 dataset:")
    for col in missing_b1:
        print(f"  - {col}")
    sys.exit(1)

print("All required columns are present.")

print()
print(f"Selected feature count : {len(SELECTED_FEATURES)}")
print(f"Label column            : {LABEL_COL}")
print(f"Timestamp available     : YES in original / NO expected in B1")


# =============================================================================
# BASIC STRUCTURE CHECK
# =============================================================================

print_section("DATASET STRUCTURE CHECK")

print("Original columns:")
print(list(original_df.columns))

print()
print("B1 columns:")
print(list(b1_df.columns))

print()
print(f"Original rows : {len(original_df):,}")
print(f"B1 rows      : {len(b1_df):,}")


# =============================================================================
# CHECK DUPLICATE ROWS IN ORIGINAL USING MATCHING FEATURES + LABEL
# =============================================================================

print_section("ORIGINAL DATA DUPLICATE ANALYSIS")

MATCH_COLUMNS = SELECTED_FEATURES + [LABEL_COL]

print(
    "Building exact signatures from 30 selected features + Label..."
)

original_signatures = make_canonical_string_series(
    original_df,
    MATCH_COLUMNS
)

b1_signatures = make_canonical_string_series(
    b1_df,
    MATCH_COLUMNS
)

print("Signature construction complete.")


# Count how many original rows exist for each signature.
signature_counts = original_signatures.value_counts(sort=False)

unique_signature_count = int((signature_counts == 1).sum())
duplicate_signature_count = int((signature_counts > 1).sum())
maximum_duplicate_count = int(signature_counts.max())

duplicate_original_rows = int(
    signature_counts[signature_counts > 1].sum()
)

print()
print(f"Unique signatures in original      : {unique_signature_count:,}")
print(f"Duplicate signatures in original  : {duplicate_signature_count:,}")
print(f"Rows belonging to duplicate groups : {duplicate_original_rows:,}")
print(f"Maximum identical-row multiplicity : {maximum_duplicate_count:,}")


# =============================================================================
# CREATE ORIGINAL SIGNATURE → ROW INDICES MAPPING
# =============================================================================

print_section("BUILDING ORIGINAL ROW INDEX")

# We store lists of original row positions for every signature.
#
# Example:
#
# signature A -> [15]
# signature B -> [20, 21, 22]
#
# A is uniquely identifiable.
# B is ambiguous.

original_index_map = {}

for idx, signature in enumerate(original_signatures):
    original_index_map.setdefault(signature, []).append(idx)

print(f"Unique signature keys in index: {len(original_index_map):,}")


# =============================================================================
# MAP B1 ROWS
# =============================================================================

print_section("MAPPING B1 ROWS TO ORIGINAL TRAIN")

mapping_start = time.time()

b1_total = len(b1_df)

unique_matches = 0
ambiguous_matches = 0
unmatched = 0

match_counts = np.zeros(b1_total, dtype=np.int32)

matched_original_indices = np.full(
    b1_total,
    -1,
    dtype=np.int64
)

match_type = np.empty(
    b1_total,
    dtype=object
)

for i, signature in enumerate(b1_signatures):

    candidates = original_index_map.get(signature)

    if candidates is None:
        unmatched += 1
        match_type[i] = "UNMATCHED"
        match_counts[i] = 0

    elif len(candidates) == 1:
        unique_matches += 1
        match_type[i] = "UNIQUE"
        match_counts[i] = 1
        matched_original_indices[i] = candidates[0]

    else:
        ambiguous_matches += 1
        match_type[i] = "AMBIGUOUS"
        match_counts[i] = len(candidates)

mapping_time = time.time() - mapping_start

print(f"Mapping time: {mapping_time:.2f} seconds")

print()
print(f"Total B1 rows      : {b1_total:,}")
print(f"Unique matches     : {unique_matches:,}")
print(f"Ambiguous matches  : {ambiguous_matches:,}")
print(f"Unmatched rows     : {unmatched:,}")


# =============================================================================
# COVERAGE CALCULATION
# =============================================================================

exactly_matched = unique_matches + ambiguous_matches

match_coverage = (
    exactly_matched / b1_total
    if b1_total > 0
    else 0.0
)

unique_mapping_rate = (
    unique_matches / b1_total
    if b1_total > 0
    else 0.0
)

ambiguous_rate = (
    ambiguous_matches / b1_total
    if b1_total > 0
    else 0.0
)

unmatched_rate = (
    unmatched / b1_total
    if b1_total > 0
    else 0.0
)

print_section("MAPPING COVERAGE")

print(f"Exact signature coverage : {match_coverage * 100:.6f}%")
print(f"Unique mapping rate      : {unique_mapping_rate * 100:.6f}%")
print(f"Ambiguous mapping rate   : {ambiguous_rate * 100:.6f}%")
print(f"Unmatched rate           : {unmatched_rate * 100:.6f}%")


# =============================================================================
# TIMESTAMP RECOVERABILITY FOR UNIQUE MATCHES
# =============================================================================

print_section("TIMESTAMP RECOVERABILITY")

unique_mask = match_type == "UNIQUE"

unique_b1_positions = np.where(unique_mask)[0]

print(f"B1 rows with unique mapping: {len(unique_b1_positions):,}")

if len(unique_b1_positions) > 0:

    recovered_timestamps = original_df.iloc[
        matched_original_indices[unique_b1_positions]
    ][TIMESTAMP_COL].reset_index(drop=True)

    valid_timestamp_mask = (
        recovered_timestamps.notna()
        & (recovered_timestamps.astype(str).str.strip() != "")
    )

    valid_recovered_timestamps = int(valid_timestamp_mask.sum())

    invalid_recovered_timestamps = (
        len(recovered_timestamps)
        - valid_recovered_timestamps
    )

    print(
        f"Valid recovered timestamps   : "
        f"{valid_recovered_timestamps:,}"
    )

    print(
        f"Invalid/missing timestamps    : "
        f"{invalid_recovered_timestamps:,}"
    )

    timestamp_recovery_rate = (
        valid_recovered_timestamps / len(unique_b1_positions)
    )

else:

    recovered_timestamps = pd.Series(dtype=object)
    valid_recovered_timestamps = 0
    invalid_recovered_timestamps = 0
    timestamp_recovery_rate = 0.0


print(
    f"Timestamp recovery rate       : "
    f"{timestamp_recovery_rate * 100:.6f}%"
)


# =============================================================================
# SAVE DETAILED UNIQUE MAPPING SAMPLE
# =============================================================================

print_section("SAVING AUDIT OUTPUTS")

# ---------------------------------------------------------------------------
# 1. Overall summary JSON
# ---------------------------------------------------------------------------

elapsed_time = time.time() - start_time

summary = {
    "step": 28,
    "purpose": (
        "Audit whether B1 rows can be mapped to original "
        "CICIDS2018_train.csv rows using selected features + Label."
    ),
    "original_train_path": str(ORIGINAL_TRAIN),
    "b1_train_path": str(B1_TRAIN),
    "original_train_sha256": original_hash,
    "b1_train_sha256": b1_hash,
    "original_rows": int(len(original_df)),
    "original_columns": int(len(original_df.columns)),
    "b1_rows": int(len(b1_df)),
    "b1_columns": int(len(b1_df.columns)),
    "selected_feature_count": len(SELECTED_FEATURES),
    "matching_columns": MATCH_COLUMNS,
    "unique_original_signatures": unique_signature_count,
    "duplicate_original_signatures": duplicate_signature_count,
    "duplicate_original_rows": duplicate_original_rows,
    "maximum_duplicate_multiplicity": maximum_duplicate_count,
    "b1_unique_matches": unique_matches,
    "b1_ambiguous_matches": ambiguous_matches,
    "b1_unmatched": unmatched,
    "exact_signature_coverage": match_coverage,
    "unique_mapping_rate": unique_mapping_rate,
    "ambiguous_mapping_rate": ambiguous_rate,
    "unmatched_rate": unmatched_rate,
    "valid_recovered_timestamps": valid_recovered_timestamps,
    "invalid_recovered_timestamps": invalid_recovered_timestamps,
    "timestamp_recovery_rate": timestamp_recovery_rate,
    "load_time_seconds": load_time,
    "mapping_time_seconds": mapping_time,
    "total_runtime_seconds": elapsed_time,
}

summary_path = OUTPUT_DIR / "step28_mapping_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)


# ---------------------------------------------------------------------------
# 2. B1 mapping status table
# ---------------------------------------------------------------------------

mapping_df = pd.DataFrame({
    "B1_Row_Index": np.arange(b1_total),
    "Match_Type": match_type,
    "Original_Match_Count": match_counts,
    "Original_Row_Index": matched_original_indices,
})

mapping_csv_path = OUTPUT_DIR / "step28_b1_mapping_status.csv"

mapping_df.to_csv(
    mapping_csv_path,
    index=False
)


# ---------------------------------------------------------------------------
# 3. Unique recovered rows with Timestamp
# ---------------------------------------------------------------------------

if len(unique_b1_positions) > 0:

    unique_output = b1_df.iloc[
        unique_b1_positions
    ].copy()

    unique_output.insert(
        0,
        "B1_Row_Index",
        unique_b1_positions
    )

    unique_output.insert(
        1,
        "Original_Row_Index",
        matched_original_indices[unique_b1_positions]
    )

    unique_output.insert(
        2,
        "Recovered_Timestamp",
        original_df.iloc[
            matched_original_indices[unique_b1_positions]
        ][TIMESTAMP_COL].values
    )

    unique_output_path = (
        OUTPUT_DIR
        / "step28_unique_mapped_rows_with_timestamp.csv"
    )

    unique_output.to_csv(
        unique_output_path,
        index=False
    )

else:

    unique_output_path = None


# ---------------------------------------------------------------------------
# 4. Ambiguous match summary
# ---------------------------------------------------------------------------

ambiguous_positions = np.where(match_type == "AMBIGUOUS")[0]

ambiguous_output = pd.DataFrame({
    "B1_Row_Index": ambiguous_positions,
    "Original_Match_Count": match_counts[ambiguous_positions],
})

ambiguous_csv_path = (
    OUTPUT_DIR
    / "step28_ambiguous_matches.csv"
)

ambiguous_output.to_csv(
    ambiguous_csv_path,
    index=False
)


# ---------------------------------------------------------------------------
# 5. Unmatched rows
# ---------------------------------------------------------------------------

unmatched_positions = np.where(match_type == "UNMATCHED")[0]

unmatched_output = pd.DataFrame({
    "B1_Row_Index": unmatched_positions,
})

unmatched_csv_path = (
    OUTPUT_DIR
    / "step28_unmatched_rows.csv"
)

unmatched_output.to_csv(
    unmatched_csv_path,
    index=False
)


# =============================================================================
# FINAL AUTOMATIC INTERPRETATION
# =============================================================================

print_section("PRELIMINARY MAPPING INTERPRETATION")

# IMPORTANT:
# This is NOT the final scientific decision.
# It is only an automatic diagnostic interpretation.
#
# We deliberately use strict thresholds.
#
# For temporal reconstruction, very high unique mapping is required.
# Ambiguous rows cannot safely identify a single original Timestamp.

if (
    unique_mapping_rate >= 0.999
    and ambiguous_rate <= 0.001
    and unmatched_rate == 0
):
    preliminary_status = (
        "VERY_HIGH_RECOVERABILITY"
    )

elif (
    unique_mapping_rate >= 0.99
    and unmatched_rate <= 0.01
):
    preliminary_status = (
        "HIGH_RECOVERABILITY_BUT_REQUIRES_REVIEW"
    )

elif (
    match_coverage >= 0.99
):
    preliminary_status = (
        "HIGH_MATCHING_BUT_AMBIGUITY_REQUIRES_REVIEW"
    )

else:
    preliminary_status = (
        "LOW_OR_NON_UNIQUE_RECOVERABILITY"
    )

print(f"Preliminary status: {preliminary_status}")

print()
print(
    "IMPORTANT: This status is only a diagnostic indication."
)
print(
    "The final LSTM decision will be made after examining the "
    "actual mapping statistics and duplicate structure."
)


# =============================================================================
# FINAL OUTPUT SUMMARY
# =============================================================================

print_section("OUTPUT FILES")

print(f"1. Summary JSON:")
print(f"   {summary_path}")

print()
print(f"2. Full B1 mapping status:")
print(f"   {mapping_csv_path}")

print()
print(f"3. Unique mapped rows + recovered Timestamp:")
if unique_output_path is not None:
    print(f"   {unique_output_path}")
else:
    print("   No unique mappings available.")

print()
print(f"4. Ambiguous matches:")
print(f"   {ambiguous_csv_path}")

print()
print(f"5. Unmatched rows:")
print(f"   {unmatched_csv_path}")


# =============================================================================
# FINAL TERMINAL SUMMARY
# =============================================================================

print_section("STEP 28 COMPLETE")

print(f"B1 rows                 : {b1_total:,}")
print(f"Unique matches          : {unique_matches:,}")
print(f"Ambiguous matches       : {ambiguous_matches:,}")
print(f"Unmatched               : {unmatched:,}")
print(f"Unique mapping rate     : {unique_mapping_rate * 100:.6f}%")
print(f"Ambiguous mapping rate  : {ambiguous_rate * 100:.6f}%")
print(f"Unmatched rate          : {unmatched_rate * 100:.6f}%")
print(f"Timestamp recovery rate : {timestamp_recovery_rate * 100:.6f}%")
print(f"Total runtime           : {elapsed_time:.2f} seconds")

print()
print("DO NOT start any LSTM retraining yet.")
print("Send the complete terminal output for scientific interpretation.")
print("=" * 80)