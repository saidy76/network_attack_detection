# =============================================================================
# STEP 29 - TEMPORAL SEQUENCE FEASIBILITY AUDIT
# =============================================================================
#
# Purpose:
#   Determine whether the B1 undersampled CICIDS2018 training rows can be
#   reconstructed into scientifically meaningful chronological sequences
#   using the Timestamps recovered in Step 28.
#
# IMPORTANT:
#   - No ML model is trained.
#   - No existing dataset is modified.
#   - No LSTM is retrained.
#   - This is a diagnostic audit only.
#
# Main questions:
#   1. Can all B1 rows be assigned a valid original Timestamp?
#   2. How many B1 rows share the same Timestamp?
#   3. Is B1 original-row order chronological?
#   4. What happens after sorting B1 rows by recovered Timestamp?
#   5. Are there timestamp ties?
#   6. Are timestamps monotonic after chronological reconstruction?
#   7. Can consecutive chronological rows reasonably form temporal sequences?
#   8. Are there large temporal gaps that make naive sequences questionable?
#
# =============================================================================

from pathlib import Path
import sys
import json
import time

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

STEP28_UNIQUE = (
    PROJECT_ROOT
    / "experiments"
    / "step28_mapping_audit"
    / "step28_unique_mapped_rows_with_timestamp.csv"
)

STEP28_SUMMARY = (
    PROJECT_ROOT
    / "experiments"
    / "step28_mapping_audit"
    / "step28_mapping_summary.json"
)

STEP28B_SUMMARY = (
    PROJECT_ROOT
    / "experiments"
    / "step28b_ambiguous_temporal_audit"
    / "step28b_temporal_audit_summary.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step29_temporal_sequence_feasibility_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

TIMESTAMP_COL = "Recovered_Timestamp"
B1_INDEX_COL = "B1_Row_Index"
ORIGINAL_INDEX_COL = "Original_Row_Index"

SEQ_LENGTH = 10

# Large gap threshold is diagnostic only.
# It does NOT automatically invalidate sequences.
LARGE_GAP_SECONDS = 60.0


# =============================================================================
# HELPERS
# =============================================================================

def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


# =============================================================================
# START
# =============================================================================

start_time = time.time()

print("=" * 80)
print("STEP 29 - TEMPORAL SEQUENCE FEASIBILITY AUDIT")
print("=" * 80)

print(f"Project root       : {PROJECT_ROOT}")
print(f"B1 training        : {B1_TRAIN}")
print(f"Step 28 unique map : {STEP28_UNIQUE}")
print(f"Step 28 summary    : {STEP28_SUMMARY}")
print(f"Step 28B summary   : {STEP28B_SUMMARY}")
print(f"Output directory   : {OUTPUT_DIR}")
print(f"Diagnostic sequence length : {SEQ_LENGTH}")
print(f"Large-gap threshold        : {LARGE_GAP_SECONDS} seconds")


# =============================================================================
# FILE CHECK
# =============================================================================

print_section("INPUT FILE CHECK")

required_files = [
    B1_TRAIN,
    STEP28_UNIQUE,
    STEP28_SUMMARY,
    STEP28B_SUMMARY,
]

for path in required_files:

    if not path.exists():
        print("ERROR: Required file not found:")
        print(path)
        sys.exit(1)

    print(f"FOUND: {path}")


# =============================================================================
# READ STEP 28 SUMMARY
# =============================================================================

print_section("READING STEP 28 SUMMARY")

with open(STEP28_SUMMARY, "r", encoding="utf-8") as f:
    step28_summary = json.load(f)

print(
    f"B1 rows                     : "
    f"{step28_summary['b1_rows']:,}"
)

print(
    f"Unique mappings              : "
    f"{step28_summary['b1_unique_matches']:,}"
)

print(
    f"Ambiguous mappings           : "
    f"{step28_summary['b1_ambiguous_matches']:,}"
)

print(
    f"Unmatched rows               : "
    f"{step28_summary['b1_unmatched']:,}"
)

print(
    f"Timestamp recovery rate     : "
    f"{step28_summary['timestamp_recovery_rate'] * 100:.6f}%"
)


# =============================================================================
# READ STEP 28B SUMMARY
# =============================================================================

print_section("READING STEP 28B SUMMARY")

with open(STEP28B_SUMMARY, "r", encoding="utf-8") as f:
    step28b_summary = json.load(f)

print(
    f"Ambiguous rows               : "
    f"{step28b_summary['ambiguous_b1_rows']:,}"
)

print(
    f"Same timestamp rows          : "
    f"{step28b_summary['same_timestamp_rows']:,}"
)

print(
    f"Different timestamp rows     : "
    f"{step28b_summary['different_timestamp_rows']:,}"
)

print(
    f"Missing timestamp rows       : "
    f"{step28b_summary['missing_timestamp_rows']:,}"
)

print(
    f"Temporal ambiguity rate      : "
    f"{step28b_summary['different_timestamp_rate'] * 100:.6f}%"
)


# =============================================================================
# LOAD B1 DATASET
# =============================================================================

print_section("LOADING B1 DATASET")

load_start = time.time()

b1_df = pd.read_csv(
    B1_TRAIN,
    low_memory=False
)

b1_df.columns = [
    str(c).strip()
    for c in b1_df.columns
]

load_time = time.time() - load_start

print(
    f"B1 dataset loaded: "
    f"{len(b1_df):,} rows × {len(b1_df.columns):,} columns"
)

print(f"Loading time: {load_time:.2f} seconds")


# =============================================================================
# VALIDATE B1 ROW COUNT
# =============================================================================

print_section("B1 STRUCTURE VALIDATION")

expected_b1_rows = int(step28_summary["b1_rows"])

if len(b1_df) != expected_b1_rows:

    print("ERROR: B1 row count does not match Step 28 summary.")

    print(
        f"Expected: {expected_b1_rows:,}"
    )

    print(
        f"Actual  : {len(b1_df):,}"
    )

    sys.exit(1)

print("B1 row count matches Step 28 summary.")


# =============================================================================
# LOAD UNIQUE MAPPING WITH RECOVERED TIMESTAMP
# =============================================================================

print_section("LOADING RECOVERED TIMESTAMPS")

unique_df = pd.read_csv(
    STEP28_UNIQUE,
    low_memory=False
)

unique_df.columns = [
    str(c).strip()
    for c in unique_df.columns
]

required_unique_columns = [
    B1_INDEX_COL,
    ORIGINAL_INDEX_COL,
    TIMESTAMP_COL,
]

missing_columns = [
    col
    for col in required_unique_columns
    if col not in unique_df.columns
]

if missing_columns:

    print("ERROR: Missing columns in Step 28 unique mapping:")

    for col in missing_columns:
        print(f"  - {col}")

    sys.exit(1)


print(
    f"Unique mapped rows loaded: "
    f"{len(unique_df):,}"
)


# =============================================================================
# PARSE TIMESTAMPS
# =============================================================================

print_section("PARSING RECOVERED TIMESTAMPS")

unique_df[TIMESTAMP_COL] = pd.to_datetime(
    unique_df[TIMESTAMP_COL],
    errors="coerce"
)

valid_timestamp_mask = unique_df[TIMESTAMP_COL].notna()

valid_timestamp_count = int(
    valid_timestamp_mask.sum()
)

invalid_timestamp_count = (
    len(unique_df)
    - valid_timestamp_count
)

print(
    f"Valid recovered timestamps   : "
    f"{valid_timestamp_count:,}"
)

print(
    f"Invalid recovered timestamps : "
    f"{invalid_timestamp_count:,}"
)


# =============================================================================
# BUILD FULL B1 TIMESTAMP TABLE
# =============================================================================

print_section("CONSTRUCTING FULL B1 TEMPORAL TABLE")

# Start from B1 row indices so every B1 row must be accounted for.
temporal_df = pd.DataFrame({
    B1_INDEX_COL: np.arange(
        len(b1_df),
        dtype=np.int64
    )
})

temporal_df = temporal_df.merge(
    unique_df[
        [
            B1_INDEX_COL,
            ORIGINAL_INDEX_COL,
            TIMESTAMP_COL,
        ]
    ],
    on=B1_INDEX_COL,
    how="left",
    validate="one_to_one",
)

print(
    f"Full B1 temporal table rows: "
    f"{len(temporal_df):,}"
)

full_valid_timestamp_count = int(
    temporal_df[TIMESTAMP_COL].notna().sum()
)

full_missing_timestamp_count = (
    len(temporal_df)
    - full_valid_timestamp_count
)

print(
    f"Rows with recovered Timestamp : "
    f"{full_valid_timestamp_count:,}"
)

print(
    f"Rows without recovered Timestamp: "
    f"{full_missing_timestamp_count:,}"
)


# =============================================================================
# TEMPORAL COVERAGE CHECK
# =============================================================================

print_section("TEMPORAL COVERAGE")

temporal_coverage_rate = (
    full_valid_timestamp_count
    / len(temporal_df)
    if len(temporal_df) > 0
    else 0.0
)

print(
    f"Temporal coverage rate: "
    f"{temporal_coverage_rate * 100:.6f}%"
)


# =============================================================================
# ORIGINAL B1 ORDER TEMPORAL ANALYSIS
# =============================================================================

print_section("ORIGINAL B1 ORDER TEMPORAL ANALYSIS")

ordered_valid = temporal_df[
    temporal_df[TIMESTAMP_COL].notna()
].copy()

ordered_timestamps = ordered_valid[
    TIMESTAMP_COL
].reset_index(drop=True)

if len(ordered_timestamps) >= 2:

    timestamp_diffs_original_order = (
        ordered_timestamps.diff()
        .dt.total_seconds()
        .dropna()
    )

    negative_original = int(
        (timestamp_diffs_original_order < 0).sum()
    )

    zero_original = int(
        (timestamp_diffs_original_order == 0).sum()
    )

    positive_original = int(
        (timestamp_diffs_original_order > 0).sum()
    )

    total_comparisons_original = (
        len(timestamp_diffs_original_order)
    )

    nondecreasing_original = (
        negative_original == 0
    )

    print(
        f"Timestamp comparisons         : "
        f"{total_comparisons_original:,}"
    )

    print(
        f"Negative time differences     : "
        f"{negative_original:,}"
    )

    print(
        f"Zero time differences         : "
        f"{zero_original:,}"
    )

    print(
        f"Positive time differences     : "
        f"{positive_original:,}"
    )

    print(
        f"Original B1 order nondecreasing: "
        f"{nondecreasing_original}"
    )

    if len(timestamp_diffs_original_order) > 0:

        print(
            f"Minimum difference (sec)     : "
            f"{timestamp_diffs_original_order.min():.6f}"
        )

        print(
            f"Maximum difference (sec)     : "
            f"{timestamp_diffs_original_order.max():.6f}"
        )

        print(
            f"Median difference (sec)      : "
            f"{timestamp_diffs_original_order.median():.6f}"
        )

else:

    negative_original = 0
    zero_original = 0
    positive_original = 0
    total_comparisons_original = 0
    nondecreasing_original = False

    timestamp_diffs_original_order = pd.Series(
        dtype=float
    )


# =============================================================================
# CHRONOLOGICAL SORT
# =============================================================================

print_section("CHRONOLOGICAL RECONSTRUCTION")

chronological_df = temporal_df[
    temporal_df[TIMESTAMP_COL].notna()
].copy()

chronological_df = chronological_df.sort_values(
    by=[
        TIMESTAMP_COL,
        ORIGINAL_INDEX_COL,
        B1_INDEX_COL,
    ],
    kind="mergesort"
).reset_index(drop=True)

print(
    f"Chronologically ordered rows: "
    f"{len(chronological_df):,}"
)

if len(chronological_df) > 0:

    print(
        f"Earliest Timestamp: "
        f"{chronological_df[TIMESTAMP_COL].iloc[0]}"
    )

    print(
        f"Latest Timestamp  : "
        f"{chronological_df[TIMESTAMP_COL].iloc[-1]}"
    )


# =============================================================================
# CHRONOLOGICAL ORDER VALIDATION
# =============================================================================

print_section("CHRONOLOGICAL ORDER VALIDATION")

chronological_timestamps = chronological_df[
    TIMESTAMP_COL
].reset_index(drop=True)

if len(chronological_timestamps) >= 2:

    timestamp_diffs_sorted = (
        chronological_timestamps.diff()
        .dt.total_seconds()
        .dropna()
    )

    negative_sorted = int(
        (timestamp_diffs_sorted < 0).sum()
    )

    zero_sorted = int(
        (timestamp_diffs_sorted == 0).sum()
    )

    positive_sorted = int(
        (timestamp_diffs_sorted > 0).sum()
    )

    nondecreasing_sorted = (
        negative_sorted == 0
    )

    print(
        f"Negative differences : "
        f"{negative_sorted:,}"
    )

    print(
        f"Zero differences     : "
        f"{zero_sorted:,}"
    )

    print(
        f"Positive differences : "
        f"{positive_sorted:,}"
    )

    print(
        f"Chronological order valid: "
        f"{nondecreasing_sorted}"
    )

else:

    negative_sorted = 0
    zero_sorted = 0
    positive_sorted = 0
    nondecreasing_sorted = False

    timestamp_diffs_sorted = pd.Series(
        dtype=float
    )


# =============================================================================
# TIMESTAMP UNIQUENESS / TIES
# =============================================================================

print_section("TIMESTAMP TIE ANALYSIS")

timestamp_counts = (
    chronological_df[TIMESTAMP_COL]
    .value_counts(sort=False)
)

unique_timestamp_count = int(
    len(timestamp_counts)
)

duplicate_timestamp_groups = int(
    (timestamp_counts > 1).sum()
)

rows_in_duplicate_timestamp_groups = int(
    timestamp_counts[
        timestamp_counts > 1
    ].sum()
)

maximum_rows_same_timestamp = int(
    timestamp_counts.max()
    if len(timestamp_counts) > 0
    else 0
)

print(
    f"Unique timestamps                 : "
    f"{unique_timestamp_count:,}"
)

print(
    f"Timestamp groups with duplicates  : "
    f"{duplicate_timestamp_groups:,}"
)

print(
    f"Rows in duplicated timestamp groups: "
    f"{rows_in_duplicate_timestamp_groups:,}"
)

print(
    f"Maximum rows sharing one timestamp : "
    f"{maximum_rows_same_timestamp:,}"
)


# =============================================================================
# TIME GAP ANALYSIS
# =============================================================================

print_section("CHRONOLOGICAL TIME GAP ANALYSIS")

positive_gaps = timestamp_diffs_sorted[
    timestamp_diffs_sorted > 0
]

if len(positive_gaps) > 0:

    large_gap_count = int(
        (positive_gaps > LARGE_GAP_SECONDS).sum()
    )

    print(
        f"Positive temporal gaps           : "
        f"{len(positive_gaps):,}"
    )

    print(
        f"Minimum positive gap (sec)       : "
        f"{positive_gaps.min():.6f}"
    )

    print(
        f"Maximum positive gap (sec)       : "
        f"{positive_gaps.max():.6f}"
    )

    print(
        f"Median positive gap (sec)        : "
        f"{positive_gaps.median():.6f}"
    )

    print(
        f"95th percentile positive gap    : "
        f"{positive_gaps.quantile(0.95):.6f}"
    )

    print(
        f"Gaps > {LARGE_GAP_SECONDS:.1f} sec             : "
        f"{large_gap_count:,}"
    )

else:

    large_gap_count = 0

    print(
        "No positive temporal gaps were available."
    )


# =============================================================================
# ORIGINAL ROW INDEX ORDER VS CHRONOLOGICAL ORDER
# =============================================================================

print_section("ORIGINAL ROW ORDER VS CHRONOLOGICAL ORDER")

original_indices = chronological_df[
    ORIGINAL_INDEX_COL
].to_numpy()

if len(original_indices) >= 2:

    original_index_diffs = np.diff(
        original_indices
    )

    backward_original_index_count = int(
        (original_index_diffs < 0).sum()
    )

    print(
        f"Chronological rows with backward "
        f"original-index movement: "
        f"{backward_original_index_count:,}"
    )

    original_index_monotonic = (
        backward_original_index_count == 0
    )

    print(
        f"Original row index remains monotonic "
        f"after chronological sort: "
        f"{original_index_monotonic}"
    )

else:

    backward_original_index_count = 0
    original_index_monotonic = False


# =============================================================================
# LOCAL SEQUENCE FEASIBILITY
# =============================================================================

print_section("LOCAL SEQUENCE FEASIBILITY")

chronological_row_count = len(
    chronological_df
)

if chronological_row_count >= SEQ_LENGTH:

    possible_sequences = (
        chronological_row_count
        - SEQ_LENGTH
        + 1
    )

    # A sequence is considered temporally continuous for this diagnostic
    # only when timestamps are nondecreasing. Equal timestamps are allowed.
    #
    # This does not claim that every sequence represents one network session.
    # It only measures whether timestamp order itself prevents construction.

    seq_timestamp_matrix = chronological_timestamps.to_numpy()

    local_window_valid = 0
    local_window_large_gap = 0

    # Use vectorized timestamp difference checks.
    if len(seq_timestamp_matrix) > SEQ_LENGTH:

        gaps = (
            chronological_timestamps
            .diff()
            .dt.total_seconds()
            .fillna(0)
            .to_numpy()
        )

        # A sequence of length N has N-1 internal gaps.
        internal_gap_count = (
            len(gaps) - 1
        )

        if internal_gap_count >= SEQ_LENGTH - 1:

            window_max_gaps = pd.Series(
                gaps[1:]
            ).rolling(
                window=SEQ_LENGTH - 1
            ).max()

            valid_window_mask = (
                window_max_gaps.notna()
            )

            local_window_valid = int(
                valid_window_mask.sum()
            )

            local_window_large_gap = int(
                (
                    window_max_gaps[
                        valid_window_mask
                    ] > LARGE_GAP_SECONDS
                ).sum()
            )

    print(
        f"Sequence length                    : "
        f"{SEQ_LENGTH}"
    )

    print(
        f"Possible overlapping sequences      : "
        f"{possible_sequences:,}"
    )

    print(
        f"Windows evaluated for gap continuity : "
        f"{local_window_valid:,}"
    )

    print(
        f"Windows containing gap > "
        f"{LARGE_GAP_SECONDS:.1f}s               : "
        f"{local_window_large_gap:,}"
    )

else:

    possible_sequences = 0
    local_window_valid = 0
    local_window_large_gap = 0

    print(
        "Not enough chronological rows to "
        "construct diagnostic sequences."
    )


# =============================================================================
# SAVE CHRONOLOGICAL MAPPING
# =============================================================================

print_section("SAVING CHRONOLOGICAL AUDIT TABLE")

chronological_output = (
    OUTPUT_DIR
    / "step29_b1_chronological_mapping.csv"
)

chronological_df.to_csv(
    chronological_output,
    index=False
)

print(
    f"Saved: {chronological_output}"
)


# =============================================================================
# SAVE TIME GAP SAMPLE
# =============================================================================

print_section("SAVING TIME GAP DIAGNOSTIC")

gap_output = (
    OUTPUT_DIR
    / "step29_chronological_time_gaps.csv"
)

gap_df = pd.DataFrame({
    "Chronological_Position": np.arange(
        len(timestamp_diffs_sorted),
        dtype=np.int64
    ) + 1,
    "Timestamp_Gap_Seconds": (
        timestamp_diffs_sorted.to_numpy()
    )
})

gap_df.to_csv(
    gap_output,
    index=False
)

print(
    f"Saved: {gap_output}"
)


# =============================================================================
# SCIENTIFIC INTERPRETATION
# =============================================================================

print_section("SCIENTIFIC INTERPRETATION")

if full_missing_timestamp_count > 0:

    temporal_status = (
        "INCOMPLETE_TIMESTAMP_RECOVERY"
    )

elif not nondecreasing_sorted:

    temporal_status = (
        "CHRONOLOGICAL_RECONSTRUCTION_FAILED"
    )

elif chronological_row_count < SEQ_LENGTH:

    temporal_status = (
        "INSUFFICIENT_ROWS_FOR_SEQUENCE"
    )

elif positive_gaps.empty:

    temporal_status = (
        "TEMPORAL_ORDER_VALID_BUT_NO_POSITIVE_GAPS"
    )

elif local_window_large_gap == 0:

    temporal_status = (
        "TEMPORAL_SEQUENCE_CONSTRUCTION_FEASIBLE"
    )

else:

    temporal_status = (
        "TEMPORAL_ORDER_VALID_BUT_GAPS_REQUIRE_REVIEW"
    )


print(
    f"Temporal feasibility status: "
    f"{temporal_status}"
)

print()

print(
    "IMPORTANT:"
)

print(
    "A valid chronological order does NOT automatically prove that"
)

print(
    "consecutive flows belong to the same network session or connection."
)

print(
    "This audit only establishes whether timestamp-based chronological"
)

print(
    "sequence construction is technically and temporally possible."
)


# =============================================================================
# SAVE SUMMARY JSON
# =============================================================================

elapsed_time = time.time() - start_time

summary = {

    "step": 29,

    "purpose": (
        "Audit feasibility of reconstructing chronological B1 "
        "training sequences using recovered original timestamps."
    ),

    "b1_rows": int(len(b1_df)),

    "unique_mapped_rows": int(
        len(unique_df)
    ),

    "valid_unique_timestamps": int(
        valid_timestamp_count
    ),

    "invalid_unique_timestamps": int(
        invalid_timestamp_count
    ),

    "full_b1_valid_timestamps": int(
        full_valid_timestamp_count
    ),

    "full_b1_missing_timestamps": int(
        full_missing_timestamp_count
    ),

    "temporal_coverage_rate": float(
        temporal_coverage_rate
    ),

    "original_b1_order": {
        "negative_differences": int(
            negative_original
        ),
        "zero_differences": int(
            zero_original
        ),
        "positive_differences": int(
            positive_original
        ),
        "nondecreasing": bool(
            nondecreasing_original
        ),
    },

    "chronological_order": {
        "rows": int(
            chronological_row_count
        ),
        "negative_differences": int(
            negative_sorted
        ),
        "zero_differences": int(
            zero_sorted
        ),
        "positive_differences": int(
            positive_sorted
        ),
        "nondecreasing": bool(
            nondecreasing_sorted
        ),
    },

    "timestamp_ties": {
        "unique_timestamp_count": int(
            unique_timestamp_count
        ),
        "duplicate_timestamp_groups": int(
            duplicate_timestamp_groups
        ),
        "rows_in_duplicate_timestamp_groups": int(
            rows_in_duplicate_timestamp_groups
        ),
        "maximum_rows_same_timestamp": int(
            maximum_rows_same_timestamp
        ),
    },

    "time_gaps_seconds": {

        "minimum_positive_gap": (
            safe_float(
                positive_gaps.min()
            )
            if len(positive_gaps) > 0
            else None
        ),

        "maximum_positive_gap": (
            safe_float(
                positive_gaps.max()
            )
            if len(positive_gaps) > 0
            else None
        ),

        "median_positive_gap": (
            safe_float(
                positive_gaps.median()
            )
            if len(positive_gaps) > 0
            else None
        ),

        "p95_positive_gap": (
            safe_float(
                positive_gaps.quantile(0.95)
            )
            if len(positive_gaps) > 0
            else None
        ),

        "large_gap_threshold_seconds": float(
            LARGE_GAP_SECONDS
        ),

        "large_gap_count": int(
            large_gap_count
        ),
    },

    "sequence_feasibility": {

        "sequence_length": int(
            SEQ_LENGTH
        ),

        "possible_overlapping_sequences": int(
            possible_sequences
        ),

        "windows_evaluated": int(
            local_window_valid
        ),

        "windows_with_large_gap": int(
            local_window_large_gap
        ),
    },

    "temporal_feasibility_status": (
        temporal_status
    ),

    "step28_timestamp_recovery_rate": float(
        step28_summary[
            "timestamp_recovery_rate"
        ]
    ),

    "step28b_same_timestamp_rate": float(
        step28b_summary[
            "same_timestamp_rate"
        ]
    ),

    "step28b_different_timestamp_rate": float(
        step28b_summary[
            "different_timestamp_rate"
        ]
    ),

    "load_time_seconds": float(
        load_time
    ),

    "runtime_seconds": float(
        elapsed_time
    ),
}

summary_path = (
    OUTPUT_DIR
    / "step29_temporal_sequence_feasibility_summary.json"
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

print(
    f"Summary saved: {summary_path}"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print_section("STEP 29 COMPLETE")

print(
    f"B1 rows                         : "
    f"{len(b1_df):,}"
)

print(
    f"Valid recovered timestamps      : "
    f"{full_valid_timestamp_count:,}"
)

print(
    f"Missing timestamps              : "
    f"{full_missing_timestamp_count:,}"
)

print(
    f"Temporal coverage               : "
    f"{temporal_coverage_rate * 100:.6f}%"
)

print(
    f"Original B1 order nondecreasing : "
    f"{nondecreasing_original}"
)

print(
    f"Chronological order valid       : "
    f"{nondecreasing_sorted}"
)

print(
    f"Duplicate timestamp groups      : "
    f"{duplicate_timestamp_groups:,}"
)

print(
    f"Maximum same-timestamp rows     : "
    f"{maximum_rows_same_timestamp:,}"
)

print(
    f"Positive temporal gaps          : "
    f"{len(positive_gaps):,}"
)

print(
    f"Large gaps > {LARGE_GAP_SECONDS:.1f}s          : "
    f"{large_gap_count:,}"
)

print(
    f"Possible sequences (L={SEQ_LENGTH}): "
    f"{possible_sequences:,}"
)

print(
    f"Windows with large gaps         : "
    f"{local_window_large_gap:,}"
)

print(
    f"FINAL TEMPORAL STATUS           : "
    f"{temporal_status}"
)

print()
print(
    "DO NOT RETRAIN THE LSTM YET."
)

print(
    "Use the complete terminal output and the Step 29 summary"
)

print(
    "for the final methodological decision."
)

print("=" * 80)