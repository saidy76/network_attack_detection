# =============================================================================
# STEP 28B - AMBIGUOUS MAPPING & TEMPORAL CONSISTENCY AUDIT
# =============================================================================
#
# Purpose:
#   Deeply audit the 1,570 ambiguous B1 rows identified in Step 28.
#
# Main questions:
#   1. Does each ambiguous B1 row have exactly two original candidates?
#   2. Do the candidate rows have the same Timestamp?
#   3. If timestamps differ, how large is the difference?
#   4. Are the ambiguous rows actually problematic for temporal recovery?
#   5. Can we safely recover a Timestamp for ambiguous rows?
#
# IMPORTANT:
#   - No ML model is trained.
#   - No existing dataset is modified.
#   - No LSTM is retrained.
#   - The original training CSV is loaded only because candidate timestamps
#     must be verified against the original source.
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

STEP28_MAPPING = (
    PROJECT_ROOT
    / "experiments"
    / "step28_mapping_audit"
    / "step28_b1_mapping_status.csv"
)

STEP28_SUMMARY = (
    PROJECT_ROOT
    / "experiments"
    / "step28_mapping_audit"
    / "step28_mapping_summary.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step28b_ambiguous_temporal_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SELECTED FEATURES
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
# HELPERS
# =============================================================================

def normalize_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


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
print("STEP 28B - AMBIGUOUS MAPPING & TEMPORAL CONSISTENCY AUDIT")
print("=" * 80)

print(f"Project root       : {PROJECT_ROOT}")
print(f"Original train     : {ORIGINAL_TRAIN}")
print(f"B1 train           : {B1_TRAIN}")
print(f"Step 28 mapping    : {STEP28_MAPPING}")
print(f"Output directory   : {OUTPUT_DIR}")


# =============================================================================
# FILE CHECK
# =============================================================================

print_section("INPUT FILE CHECK")

required_files = [
    ORIGINAL_TRAIN,
    B1_TRAIN,
    STEP28_MAPPING,
    STEP28_SUMMARY,
]

for path in required_files:

    if not path.exists():
        print(f"ERROR: Required file not found:")
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
    f"Step 28 unique mapping rate : "
    f"{step28_summary['unique_mapping_rate'] * 100:.6f}%"
)

print(
    f"Step 28 ambiguous rate       : "
    f"{step28_summary['ambiguous_mapping_rate'] * 100:.6f}%"
)

print(
    f"Step 28 unmatched rate       : "
    f"{step28_summary['unmatched_rate'] * 100:.6f}%"
)


# =============================================================================
# LOAD STEP 28 MAPPING STATUS
# =============================================================================

print_section("LOADING STEP 28 MAPPING STATUS")

mapping_df = pd.read_csv(STEP28_MAPPING)

print(
    f"Mapping table rows: "
    f"{len(mapping_df):,}"
)

required_mapping_columns = [
    "B1_Row_Index",
    "Match_Type",
    "Original_Match_Count",
    "Original_Row_Index",
]

missing_mapping = [
    col
    for col in required_mapping_columns
    if col not in mapping_df.columns
]

if missing_mapping:
    print("ERROR: Missing mapping columns:")
    for col in missing_mapping:
        print(f"  - {col}")
    sys.exit(1)


# =============================================================================
# EXTRACT AMBIGUOUS ROWS
# =============================================================================

print_section("IDENTIFYING AMBIGUOUS ROWS")

ambiguous_df = mapping_df[
    mapping_df["Match_Type"] == "AMBIGUOUS"
].copy()

ambiguous_df = ambiguous_df.reset_index(drop=True)

print(
    f"Ambiguous B1 rows identified: "
    f"{len(ambiguous_df):,}"
)

if len(ambiguous_df) == 0:

    print()
    print("No ambiguous rows were found.")
    print("Step 28B has nothing further to analyse.")
    sys.exit(0)


# =============================================================================
# VERIFY AMBIGUITY MULTIPLICITY
# =============================================================================

print_section("AMBIGUITY MULTIPLICITY CHECK")

multiplicity_counts = (
    ambiguous_df["Original_Match_Count"]
    .value_counts()
    .sort_index()
)

for multiplicity, count in multiplicity_counts.items():

    print(
        f"Original candidate count = {multiplicity}: "
        f"{count:,} B1 rows"
    )

max_multiplicity = int(
    ambiguous_df["Original_Match_Count"].max()
)

print()
print(
    f"Maximum candidate multiplicity: "
    f"{max_multiplicity}"
)


# =============================================================================
# LOAD ORIGINAL DATASET
# =============================================================================

print_section("LOADING ORIGINAL DATASET FOR TIMESTAMP VERIFICATION")

load_start = time.time()

print("Loading original CICIDS2018_train.csv...")

original_df = pd.read_csv(
    ORIGINAL_TRAIN,
    low_memory=False
)

original_df = normalize_column_names(original_df)

load_time = time.time() - load_start

print(
    f"Original dataset loaded: "
    f"{len(original_df):,} rows × "
    f"{len(original_df.columns):,} columns"
)

print(f"Loading time: {load_time:.2f} seconds")


# =============================================================================
# TIMESTAMP PARSING
# =============================================================================

print_section("PARSING ORIGINAL TIMESTAMPS")

original_timestamps = pd.to_datetime(
    original_df[TIMESTAMP_COL],
    errors="coerce"
)

valid_timestamp_count = int(
    original_timestamps.notna().sum()
)

invalid_timestamp_count = (
    len(original_timestamps)
    - valid_timestamp_count
)

print(
    f"Valid timestamps   : "
    f"{valid_timestamp_count:,}"
)

print(
    f"Invalid timestamps : "
    f"{invalid_timestamp_count:,}"
)


# =============================================================================
# ANALYSE AMBIGUOUS CANDIDATES
# =============================================================================

print_section("ANALYSING AMBIGUOUS TIMESTAMP CANDIDATES")

results = []

same_timestamp_count = 0
different_timestamp_count = 0
missing_timestamp_count = 0

time_differences_seconds = []


for _, row in ambiguous_df.iterrows():

    b1_index = int(row["B1_Row_Index"])
    candidate_count = int(row["Original_Match_Count"])

    #
    # Step 28 stored only the first original index in Original_Row_Index
    # for ambiguous rows. Because the original duplicate multiplicity is
    # known to be at most 2, we identify the second candidate by matching
    # the same selected-feature signature directly within the original data.
    #
    # To avoid a full 4.1M-row scan for every ambiguous row, we construct
    # candidate signatures from the original dataset once below.
    #

    results.append({
        "B1_Row_Index": b1_index,
        "Original_Match_Count": candidate_count,
    })


# =============================================================================
# BUILD SIGNATURE FOR ORIGINAL + AMBIGUOUS B1 ROWS
# =============================================================================

print_section("BUILDING SIGNATURE INDEX FOR AMBIGUOUS ROWS")

# Load only the B1 rows corresponding to ambiguous mappings.
#
# This avoids constructing signatures for all 1M B1 rows.

ambiguous_b1_indices = (
    ambiguous_df["B1_Row_Index"]
    .astype(int)
    .tolist()
)

print(
    f"Number of B1 rows requiring detailed analysis: "
    f"{len(ambiguous_b1_indices):,}"
)

print("Loading B1 dataset...")

b1_df = pd.read_csv(
    B1_TRAIN,
    low_memory=False
)

b1_df = normalize_column_names(b1_df)

ambiguous_b1_df = b1_df.iloc[
    ambiguous_b1_indices
].copy()

ambiguous_b1_df["B1_Row_Index"] = ambiguous_b1_indices


def make_signature_from_row(row):
    values = []

    for col in SELECTED_FEATURES + [LABEL_COL]:

        value = row[col]

        if pd.isna(value):
            values.append("<NA>")
        else:
            values.append(str(value))

    return "\x1f".join(values)


print("Creating signatures for ambiguous B1 rows...")

ambiguous_b1_df["Signature"] = ambiguous_b1_df.apply(
    make_signature_from_row,
    axis=1
)

print("Creating signatures for original rows...")

original_signature = (
    original_df[
        SELECTED_FEATURES + [LABEL_COL]
    ]
    .astype(str)
    .agg("\x1f".join, axis=1)
)

print("Building candidate lookup...")

candidate_map = {}

for idx, signature in enumerate(original_signature):

    if signature in set(
        ambiguous_b1_df["Signature"].values
    ):
        candidate_map.setdefault(
            signature,
            []
        ).append(idx)

print(
    f"Candidate signatures found: "
    f"{len(candidate_map):,}"
)


# =============================================================================
# DETAILED TEMPORAL ANALYSIS
# =============================================================================

print_section("DETAILED TEMPORAL ANALYSIS")

detailed_results = []

for _, b1_row in ambiguous_b1_df.iterrows():

    b1_index = int(b1_row["B1_Row_Index"])
    signature = b1_row["Signature"]

    candidates = candidate_map.get(
        signature,
        []
    )

    candidate_timestamps = []

    for original_index in candidates:

        timestamp = original_timestamps.iloc[
            original_index
        ]

        candidate_timestamps.append(
            timestamp
        )

    # -------------------------------------------------------------------------
    # Candidate timestamp analysis
    # -------------------------------------------------------------------------

    valid_candidates = [
        ts
        for ts in candidate_timestamps
        if pd.notna(ts)
    ]

    if len(valid_candidates) == 0:

        timestamp_status = "NO_VALID_TIMESTAMP"
        same_timestamp = False
        difference_seconds = np.nan

        missing_timestamp_count += 1

    elif len(valid_candidates) == 1:

        timestamp_status = "ONE_VALID_TIMESTAMP"
        same_timestamp = True
        difference_seconds = 0.0

        same_timestamp_count += 1

    else:

        # Two or more valid timestamps
        min_ts = min(valid_candidates)
        max_ts = max(valid_candidates)

        difference_seconds = (
            max_ts - min_ts
        ).total_seconds()

        if difference_seconds == 0:

            timestamp_status = "SAME_TIMESTAMP"
            same_timestamp = True
            same_timestamp_count += 1

        else:

            timestamp_status = "DIFFERENT_TIMESTAMP"
            same_timestamp = False
            different_timestamp_count += 1

            time_differences_seconds.append(
                difference_seconds
            )

    # -------------------------------------------------------------------------
    # Create readable candidate representation
    # -------------------------------------------------------------------------

    candidate_records = []

    for original_index in candidates:

        ts = original_timestamps.iloc[
            original_index
        ]

        candidate_records.append({
            "original_index": int(original_index),
            "timestamp": (
                ts.isoformat()
                if pd.notna(ts)
                else None
            ),
        })

    detailed_results.append({

        "B1_Row_Index": b1_index,

        "Candidate_Count": len(candidates),

        "Timestamp_Status": timestamp_status,

        "Same_Timestamp": same_timestamp,

        "Timestamp_Difference_Seconds": difference_seconds,

        "Candidate_1_Original_Index": (
            candidate_records[0]["original_index"]
            if len(candidate_records) >= 1
            else np.nan
        ),

        "Candidate_1_Timestamp": (
            candidate_records[0]["timestamp"]
            if len(candidate_records) >= 1
            else None
        ),

        "Candidate_2_Original_Index": (
            candidate_records[1]["original_index"]
            if len(candidate_records) >= 2
            else np.nan
        ),

        "Candidate_2_Timestamp": (
            candidate_records[1]["timestamp"]
            if len(candidate_records) >= 2
            else None
        ),

    })


detailed_df = pd.DataFrame(
    detailed_results
)


# =============================================================================
# TEMPORAL DIFFERENCE STATISTICS
# =============================================================================

print_section("TEMPORAL DIFFERENCE STATISTICS")

print(
    f"Ambiguous rows with same timestamp      : "
    f"{same_timestamp_count:,}"
)

print(
    f"Ambiguous rows with different timestamps: "
    f"{different_timestamp_count:,}"
)

print(
    f"Ambiguous rows with no valid timestamp   : "
    f"{missing_timestamp_count:,}"
)

if len(time_differences_seconds) > 0:

    differences = np.array(
        time_differences_seconds,
        dtype=float
    )

    print()
    print(
        f"Minimum timestamp difference : "
        f"{differences.min():.6f} seconds"
    )

    print(
        f"Maximum timestamp difference : "
        f"{differences.max():.6f} seconds"
    )

    print(
        f"Mean timestamp difference    : "
        f"{differences.mean():.6f} seconds"
    )

    print(
        f"Median timestamp difference  : "
        f"{np.median(differences):.6f} seconds"
    )

    print(
        f"95th percentile difference   : "
        f"{np.percentile(differences, 95):.6f} seconds"
    )

else:

    differences = np.array([], dtype=float)

    print()
    print(
        "No ambiguous rows had different valid timestamps."
    )


# =============================================================================
# SAVE DETAILED RESULTS
# =============================================================================

print_section("SAVING RESULTS")

detailed_csv = (
    OUTPUT_DIR
    / "step28b_ambiguous_timestamp_details.csv"
)

detailed_df.to_csv(
    detailed_csv,
    index=False
)

print(
    f"Detailed timestamp audit saved to:"
)
print(detailed_csv)


# =============================================================================
# SUMMARY
# =============================================================================

ambiguous_total = len(ambiguous_df)

same_timestamp_rate = (
    same_timestamp_count / ambiguous_total
    if ambiguous_total > 0
    else 0.0
)

different_timestamp_rate = (
    different_timestamp_count / ambiguous_total
    if ambiguous_total > 0
    else 0.0
)

missing_timestamp_rate = (
    missing_timestamp_count / ambiguous_total
    if ambiguous_total > 0
    else 0.0
)


# =============================================================================
# PRELIMINARY INTERPRETATION
# =============================================================================

print_section("PRELIMINARY INTERPRETATION")

if ambiguous_total == 0:

    interpretation = (
        "NO_AMBIGUOUS_ROWS"
    )

elif different_timestamp_count == 0:

    interpretation = (
        "AMBIGUOUS_FEATURE_MAPPING_BUT_TIMESTAMP_IS_IDENTICAL"
    )

elif different_timestamp_rate <= 0.01:

    interpretation = (
        "VERY_LOW_TEMPORAL_AMBIGUITY"
    )

elif different_timestamp_rate <= 0.10:

    interpretation = (
        "MODERATE_TEMPORAL_AMBIGUITY_REQUIRES_REVIEW"
    )

else:

    interpretation = (
        "MATERIAL_TEMPORAL_AMBIGUITY"
    )


print(
    f"Preliminary interpretation: "
    f"{interpretation}"
)

print()
print(
    "This is NOT the final LSTM decision."
)

print(
    "The final decision must also consider whether chronological ordering "
    "and sequence construction are scientifically meaningful."
)


# =============================================================================
# SAVE SUMMARY JSON
# =============================================================================

summary = {

    "step": "28B",

    "purpose": (
        "Detailed temporal audit of ambiguous B1-to-original row mappings."
    ),

    "ambiguous_b1_rows": int(ambiguous_total),

    "same_timestamp_rows": int(
        same_timestamp_count
    ),

    "different_timestamp_rows": int(
        different_timestamp_count
    ),

    "missing_timestamp_rows": int(
        missing_timestamp_count
    ),

    "same_timestamp_rate": float(
        same_timestamp_rate
    ),

    "different_timestamp_rate": float(
        different_timestamp_rate
    ),

    "missing_timestamp_rate": float(
        missing_timestamp_rate
    ),

    "min_timestamp_difference_seconds": (
        float(differences.min())
        if len(differences) > 0
        else None
    ),

    "max_timestamp_difference_seconds": (
        float(differences.max())
        if len(differences) > 0
        else None
    ),

    "mean_timestamp_difference_seconds": (
        float(differences.mean())
        if len(differences) > 0
        else None
    ),

    "median_timestamp_difference_seconds": (
        float(np.median(differences))
        if len(differences) > 0
        else None
    ),

    "p95_timestamp_difference_seconds": (
        float(np.percentile(differences, 95))
        if len(differences) > 0
        else None
    ),

    "preliminary_interpretation": interpretation,

    "original_rows": int(
        len(original_df)
    ),

    "b1_rows": int(
        len(b1_df)
    ),

    "original_valid_timestamps": int(
        valid_timestamp_count
    ),

    "original_invalid_timestamps": int(
        invalid_timestamp_count
    ),

    "runtime_seconds": float(
        time.time() - start_time
    ),
}

summary_json = (
    OUTPUT_DIR
    / "step28b_temporal_audit_summary.json"
)

with open(
    summary_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

print()
print(
    f"Summary JSON saved to:"
)
print(summary_json)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

total_runtime = time.time() - start_time

print_section("STEP 28B COMPLETE")

print(
    f"Ambiguous B1 rows          : "
    f"{ambiguous_total:,}"
)

print(
    f"Same timestamp             : "
    f"{same_timestamp_count:,}"
)

print(
    f"Different timestamp        : "
    f"{different_timestamp_count:,}"
)

print(
    f"Missing timestamp          : "
    f"{missing_timestamp_count:,}"
)

print(
    f"Same timestamp rate        : "
    f"{same_timestamp_rate * 100:.6f}%"
)

print(
    f"Different timestamp rate   : "
    f"{different_timestamp_rate * 100:.6f}%"
)

print(
    f"Missing timestamp rate     : "
    f"{missing_timestamp_rate * 100:.6f}%"
)

print(
    f"Total runtime              : "
    f"{total_runtime:.2f} seconds"
)

print()
print(
    "DO NOT retrain the LSTM yet."
)

print(
    "Send the COMPLETE terminal output for the final methodological decision."
)
