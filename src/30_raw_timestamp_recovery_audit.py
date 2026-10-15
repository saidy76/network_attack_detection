from pathlib import Path
import sys
import json
import time
import pandas as pd
import numpy as np


# =============================================================================
# STEP 30 - RAW TIMESTAMP RECOVERY AUDIT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "cicids2018"

ORIGINAL_TRAIN = (
    PROJECT_ROOT / "data" / "modeling" / "CICIDS2018_train.csv"
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

STEP28B_SUMMARY = (
    PROJECT_ROOT
    / "experiments"
    / "step28b_ambiguous_temporal_audit"
    / "step28b_temporal_audit_summary.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step30_raw_timestamp_recovery_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Label"

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

MATCH_COLUMNS = SELECTED_FEATURES + [LABEL_COL]


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def fail(message):
    print()
    print("ERROR:")
    print(message)
    sys.exit(1)


def normalize_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_for_hash(df):
    out = pd.DataFrame(index=df.index)

    for col in MATCH_COLUMNS:
        if col == LABEL_COL:
            out[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )
        else:
            out[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return out


def row_hashes(df):
    normalized = normalize_for_hash(df)

    return pd.util.hash_pandas_object(
        normalized,
        index=False
    ).to_numpy(dtype=np.uint64)


# =============================================================================
# START
# =============================================================================

start_time = time.time()

print("=" * 80)
print("STEP 30 - RAW TIMESTAMP RECOVERY AUDIT")
print("=" * 80)
print(f"Project root : {PROJECT_ROOT}")
print(f"Raw directory: {RAW_DIR}")
print(f"Original     : {ORIGINAL_TRAIN}")
print(f"B1           : {B1_TRAIN}")
print(f"Output       : {OUTPUT_DIR}")
print(f"Chunk size   : {CHUNK_SIZE:,}")


# =============================================================================
# INPUT CHECK
# =============================================================================

section("INPUT FILE CHECK")

required = [
    ORIGINAL_TRAIN,
    B1_TRAIN,
    STEP28_MAPPING,
    STEP28_SUMMARY,
    STEP28B_SUMMARY,
]

for path in required:
    if not path.exists():
        fail(f"Required file not found:\n{path}")
    print(f"FOUND: {path}")


# =============================================================================
# RAW FILE DISCOVERY
# =============================================================================

section("RAW CICIDS2018 FILE DISCOVERY")

raw_files = sorted(RAW_DIR.glob("*.csv"))

if not raw_files:
    fail(f"No CSV files found in:\n{RAW_DIR}")

print(f"Raw files: {len(raw_files):,}")

for f in raw_files:
    print(f"  - {f.name}")


# =============================================================================
# PREVIOUS AUDIT SUMMARIES
# =============================================================================

section("PREVIOUS AUDIT SUMMARY")

with open(STEP28_SUMMARY, "r", encoding="utf-8") as f:
    step28 = json.load(f)

with open(STEP28B_SUMMARY, "r", encoding="utf-8") as f:
    step28b = json.load(f)

print("Step 28 summary loaded.")
print("Step 28B summary loaded.")

for key in [
    "b1_rows",
    "b1_unique_matches",
    "b1_ambiguous_matches",
    "b1_unmatched",
]:
    if key in step28:
        print(f"{key}: {step28[key]}")

for key in [
    "same_timestamp_rows",
    "different_timestamp_rows",
]:
    if key in step28b:
        print(f"{key}: {step28b[key]}")


# =============================================================================
# LOAD ORIGINAL TRAIN
# =============================================================================

section("LOADING ORIGINAL TRAIN")

original_df = pd.read_csv(
    ORIGINAL_TRAIN,
    low_memory=False
)

original_df = normalize_columns(original_df)

print(f"Rows    : {len(original_df):,}")
print(f"Columns : {len(original_df.columns):,}")

if TIMESTAMP_COL not in original_df.columns:
    fail("Timestamp column not found in original train.")

missing = [
    c for c in MATCH_COLUMNS
    if c not in original_df.columns
]

if missing:
    fail(
        "Missing columns in original train:\n"
        + "\n".join(f"  - {c}" for c in missing)
    )

print(
    f"Timestamp unique values: "
    f"{original_df[TIMESTAMP_COL].nunique(dropna=False):,}"
)


# =============================================================================
# BUILD ORIGINAL HASH INDEX
# =============================================================================

section("BUILDING ORIGINAL TRAIN HASH INDEX")

original_hash = row_hashes(original_df)

original_hash_map = {}

for idx, h in enumerate(original_hash):
    key = int(h)

    if key not in original_hash_map:
        original_hash_map[key] = [idx]
    else:
        original_hash_map[key].append(idx)

duplicate_hash_groups = sum(
    1
    for values in original_hash_map.values()
    if len(values) > 1
)

max_multiplicity = max(
    len(values)
    for values in original_hash_map.values()
)

print(
    f"Total original rows       : {len(original_hash):,}"
)

print(
    f"Unique hash groups        : {len(original_hash_map):,}"
)

print(
    f"Duplicate hash groups     : {duplicate_hash_groups:,}"
)

print(
    f"Maximum hash multiplicity : {max_multiplicity:,}"
)


# =============================================================================
# LOAD B1
# =============================================================================

section("LOADING B1")

b1_df = pd.read_csv(
    B1_TRAIN,
    low_memory=False
)

b1_df = normalize_columns(b1_df)

print(f"B1 rows    : {len(b1_df):,}")
print(f"B1 columns : {len(b1_df.columns):,}")

missing = [
    c for c in MATCH_COLUMNS
    if c not in b1_df.columns
]

if missing:
    fail(
        "Missing columns in B1:\n"
        + "\n".join(f"  - {c}" for c in missing)
    )


# =============================================================================
# B1 HASH MAPPING
# =============================================================================

section("VALIDATING B1 -> ORIGINAL TRAIN MAPPING")

b1_hash = row_hashes(b1_df)

b1_unique = 0
b1_ambiguous = 0
b1_unmatched = 0

b1_original_index = np.full(
    len(b1_df),
    -1,
    dtype=np.int64
)

for i, h in enumerate(b1_hash):

    candidates = original_hash_map.get(
        int(h),
        []
    )

    if len(candidates) == 1:
        b1_original_index[i] = candidates[0]
        b1_unique += 1

    elif len(candidates) > 1:
        b1_ambiguous += 1

    else:
        b1_unmatched += 1

print(f"B1 unique mappings    : {b1_unique:,}")
print(f"B1 ambiguous mappings : {b1_ambiguous:,}")
print(f"B1 unmatched rows     : {b1_unmatched:,}")


# =============================================================================
# RAW FILE COLUMN VALIDATION
# =============================================================================

section("VALIDATING RAW FILE COLUMNS")

for raw_file in raw_files:

    sample = pd.read_csv(
        raw_file,
        nrows=5,
        low_memory=False
    )

    sample = normalize_columns(sample)

    missing = [
        c
        for c in MATCH_COLUMNS + [TIMESTAMP_COL]
        if c not in sample.columns
    ]

    if missing:
        fail(
            f"Missing columns in {raw_file.name}:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    print(f"OK: {raw_file.name}")

print("All raw files validated.")


# =============================================================================
# RECOVER ORIGINAL TRAIN TIMESTAMPS
# =============================================================================

section("SCANNING RAW FILES FOR TIMESTAMP RECOVERY")

recovered_timestamp = np.empty(
    len(original_df),
    dtype=object
)
recovered_timestamp[:] = None

recovered_file = np.empty(
    len(original_df),
    dtype=object
)
recovered_file[:] = None

recovered_raw_index = np.full(
    len(original_df),
    -1,
    dtype=np.int64
)

raw_match_count = np.zeros(
    len(original_df),
    dtype=np.int32
)

total_raw_rows = 0
total_matching_raw_rows = 0
duplicate_raw_matches = 0

file_stats = []

for raw_file in raw_files:

    print()
    print("-" * 80)
    print(f"FILE: {raw_file.name}")
    print("-" * 80)

    file_start = time.time()
    file_rows = 0
    file_matches = 0

    for chunk in pd.read_csv(
        raw_file,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk = normalize_columns(chunk)
        hashes = row_hashes(chunk)

        for local_i, h in enumerate(hashes):

            candidates = original_hash_map.get(
                int(h),
                []
            )

            if not candidates:
                continue

            total_matching_raw_rows += 1
            file_matches += 1

            raw_index = file_rows + local_i
            timestamp = chunk.iloc[local_i][TIMESTAMP_COL]

            for original_i in candidates:

                raw_match_count[original_i] += 1

                if recovered_file[original_i] is None:

                    recovered_timestamp[original_i] = timestamp
                    recovered_file[original_i] = raw_file.name
                    recovered_raw_index[original_i] = raw_index

                else:

                    duplicate_raw_matches += 1

        file_rows += len(chunk)
        total_raw_rows += len(chunk)

        print(
            f"\rRows: {file_rows:,} | "
            f"Matches: {file_matches:,}",
            end="",
            flush=True
        )

    print()

    elapsed = time.time() - file_start

    file_stats.append(
        {
            "file": raw_file.name,
            "rows_scanned": int(file_rows),
            "matching_rows": int(file_matches),
            "runtime_seconds": float(elapsed),
        }
    )

    print(f"Rows scanned : {file_rows:,}")
    print(f"Matches      : {file_matches:,}")
    print(f"Runtime      : {elapsed:.2f} sec")


# =============================================================================
# ORIGINAL TRAIN RECOVERY STATISTICS
# =============================================================================

section("RAW RECOVERY STATISTICS")

original_recovered = int(
    np.sum(raw_match_count > 0)
)

original_unmatched = int(
    np.sum(raw_match_count == 0)
)

original_multiple = int(
    np.sum(raw_match_count > 1)
)

print(
    f"Original train rows           : "
    f"{len(original_df):,}"
)

print(
    f"Raw rows scanned              : "
    f"{total_raw_rows:,}"
)

print(
    f"Original rows recovered       : "
    f"{original_recovered:,}"
)

print(
    f"Original rows unmatched       : "
    f"{original_unmatched:,}"
)

print(
    f"Original rows with >1 matches : "
    f"{original_multiple:,}"
)

print(
    f"Total raw matching rows       : "
    f"{total_matching_raw_rows:,}"
)

print(
    f"Duplicate raw matches         : "
    f"{duplicate_raw_matches:,}"
)


# =============================================================================
# PARSE RECOVERED TIMESTAMPS
# =============================================================================

section("TIMESTAMP VALIDATION")

original_ts = pd.to_datetime(
    pd.Series(recovered_timestamp),
    errors="coerce",
    dayfirst=True
)

valid_original_ts = int(
    original_ts.notna().sum()
)

invalid_original_ts = int(
    original_ts.isna().sum()
)

print(
    f"Valid timestamps   : "
    f"{valid_original_ts:,}"
)

print(
    f"Invalid/missing    : "
    f"{invalid_original_ts:,}"
)

if valid_original_ts > 0:

    print(
        f"Earliest          : "
        f"{original_ts.min()}"
    )

    print(
        f"Latest            : "
        f"{original_ts.max()}"
    )


# =============================================================================
# SAVE ORIGINAL RECOVERY TABLE
# =============================================================================

section("SAVING ORIGINAL RECOVERY TABLE")

original_recovery = pd.DataFrame(
    {
        "Original_Row_Index": np.arange(
            len(original_df)
        ),
        "Raw_Match_Count": raw_match_count,
        "Recovered_Timestamp": original_ts,
        "Raw_File": recovered_file,
        "Raw_Row_Index": recovered_raw_index,
    }
)

original_recovery_path = (
    OUTPUT_DIR
    / "step30_original_train_raw_timestamp_recovery.csv"
)

original_recovery.to_csv(
    original_recovery_path,
    index=False
)

print(f"Saved: {original_recovery_path}")


# =============================================================================
# MAP TIMESTAMPS TO B1
# =============================================================================

section("MAPPING RECOVERED TIMESTAMPS TO B1")

b1_timestamp = np.empty(
    len(b1_df),
    dtype=object
)
b1_timestamp[:] = None

b1_file = np.empty(
    len(b1_df),
    dtype=object
)
b1_file[:] = None

b1_raw_index = np.full(
    len(b1_df),
    -1,
    dtype=np.int64
)

b1_source = np.empty(
    len(b1_df),
    dtype=object
)
b1_source[:] = "UNRECOVERED"

for i in range(len(b1_df)):

    candidates = original_hash_map.get(
        int(b1_hash[i]),
        []
    )

    if len(candidates) == 1:

        original_i = candidates[0]

        if recovered_timestamp[original_i] is not None:

            b1_timestamp[i] = recovered_timestamp[original_i]
            b1_file[i] = recovered_file[original_i]
            b1_raw_index[i] = recovered_raw_index[original_i]

            b1_source[i] = (
                "RAW_VIA_UNIQUE_ORIGINAL_MAPPING"
            )

    elif len(candidates) > 1:

        candidate_ts = []

        for original_i in candidates:

            ts = recovered_timestamp[original_i]

            if ts is not None:
                candidate_ts.append(str(ts))

        if candidate_ts:

            if len(set(candidate_ts)) == 1:

                original_i = candidates[0]

                b1_timestamp[i] = recovered_timestamp[original_i]
                b1_file[i] = recovered_file[original_i]
                b1_raw_index[i] = recovered_raw_index[original_i]

                b1_source[i] = (
                    "RAW_VIA_AMBIGUOUS_IDENTICAL_TIMESTAMP"
                )


# =============================================================================
# B1 TIMESTAMP SUMMARY
# =============================================================================

section("B1 TIMESTAMP RECOVERY SUMMARY")

b1_ts = pd.to_datetime(
    pd.Series(b1_timestamp),
    errors="coerce",
    dayfirst=True
)

b1_valid = int(
    b1_ts.notna().sum()
)

b1_missing = int(
    b1_ts.isna().sum()
)

b1_unique_mapping = int(
    np.sum(
        b1_source
        == "RAW_VIA_UNIQUE_ORIGINAL_MAPPING"
    )
)

b1_ambiguous_identical = int(
    np.sum(
        b1_source
        == "RAW_VIA_AMBIGUOUS_IDENTICAL_TIMESTAMP"
    )
)

b1_recovery_rate = (
    b1_valid / len(b1_df)
    if len(b1_df)
    else 0.0
)

print(f"B1 rows                         : {len(b1_df):,}")
print(f"B1 valid timestamps             : {b1_valid:,}")
print(f"B1 missing timestamps           : {b1_missing:,}")
print(f"Unique mapping recovery         : {b1_unique_mapping:,}")
print(f"Ambiguous identical recovery    : {b1_ambiguous_identical:,}")
print(f"B1 recovery rate                : {b1_recovery_rate * 100:.6f}%")


# =============================================================================
# SAVE B1 RECOVERY TABLE
# =============================================================================

section("SAVING B1 RECOVERY TABLE")

b1_recovery = pd.DataFrame(
    {
        "B1_Row_Index": np.arange(
            len(b1_df)
        ),
        "Timestamp_Recovery_Status": np.where(
            b1_ts.notna(),
            "RECOVERED",
            "UNRECOVERED"
        ),
        "Recovered_Timestamp": b1_ts,
        "Raw_File": b1_file,
        "Raw_Row_Index": b1_raw_index,
        "Timestamp_Source": b1_source,
    }
)

b1_recovery_path = (
    OUTPUT_DIR
    / "step30_b1_raw_timestamp_recovery.csv"
)

b1_recovery.to_csv(
    b1_recovery_path,
    index=False
)

print(f"Saved: {b1_recovery_path}")


# =============================================================================
# CHRONOLOGICAL ORDER AUDIT
# =============================================================================

section("B1 CHRONOLOGICAL ORDER AUDIT")

valid_ts = (
    b1_ts[b1_ts.notna()]
    .reset_index(drop=True)
)

print(
    f"Valid timestamps available: "
    f"{len(valid_ts):,}"
)

if len(valid_ts) >= 2:

    diffs = (
        valid_ts
        .diff()
        .dt.total_seconds()
        .dropna()
        .to_numpy()
    )

    negative = int(
        np.sum(diffs < 0)
    )

    zero = int(
        np.sum(diffs == 0)
    )

    positive = int(
        np.sum(diffs > 0)
    )

else:

    diffs = np.array([], dtype=float)

    negative = 0
    zero = 0
    positive = 0

nondecreasing = (
    negative == 0
)

print(
    f"Negative differences : "
    f"{negative:,}"
)

print(
    f"Zero differences     : "
    f"{zero:,}"
)

print(
    f"Positive differences : "
    f"{positive:,}"
)

print(
    f"Nondecreasing order  : "
    f"{nondecreasing}"
)


# =============================================================================
# TIMESTAMP UNIQUENESS
# =============================================================================

section("B1 TIMESTAMP UNIQUENESS")

if len(valid_ts) > 0:

    unique_timestamps = int(
        valid_ts.nunique()
    )

    duplicated_rows = int(
        len(valid_ts) - unique_timestamps
    )

else:

    unique_timestamps = 0
    duplicated_rows = 0

print(
    f"Unique timestamps       : "
    f"{unique_timestamps:,}"
)

print(
    f"Rows sharing timestamps : "
    f"{duplicated_rows:,}"
)


# =============================================================================
# TIME GAP ANALYSIS
# =============================================================================

section("B1 TIME GAP ANALYSIS")

positive_gaps = (
    diffs[diffs > 0]
    if len(diffs)
    else np.array([], dtype=float)
)

if len(positive_gaps) > 0:

    min_gap = float(
        np.min(positive_gaps)
    )

    max_gap = float(
        np.max(positive_gaps)
    )

    median_gap = float(
        np.median(positive_gaps)
    )

    p95_gap = float(
        np.percentile(
            positive_gaps,
            95
        )
    )

    print(
        f"Positive gaps       : "
        f"{len(positive_gaps):,}"
    )

    print(
        f"Minimum gap         : "
        f"{min_gap:.6f} sec"
    )

    print(
        f"Maximum gap         : "
        f"{max_gap:.6f} sec"
    )

    print(
        f"Median gap          : "
        f"{median_gap:.6f} sec"
    )

    print(
        f"95th percentile     : "
        f"{p95_gap:.6f} sec"
    )

else:

    min_gap = None
    max_gap = None
    median_gap = None
    p95_gap = None

    print("No positive gaps available.")


# =============================================================================
# SAVE GAP TABLE
# =============================================================================

gap_path = (
    OUTPUT_DIR
    / "step30_b1_time_gaps.csv"
)

if len(valid_ts) >= 2:

    gap_df = pd.DataFrame(
        {
            "B1_Valid_Sequence_Index": np.arange(
                len(valid_ts)
            ),
            "Timestamp": valid_ts,
            "Previous_Timestamp": valid_ts.shift(1),
            "Gap_Seconds": (
                valid_ts
                .diff()
                .dt.total_seconds()
            ),
        }
    )

else:

    gap_df = pd.DataFrame(
        columns=[
            "B1_Valid_Sequence_Index",
            "Timestamp",
            "Previous_Timestamp",
            "Gap_Seconds",
        ]
    )

gap_df.to_csv(
    gap_path,
    index=False
)

print(f"Saved: {gap_path}")


# =============================================================================
# FINAL SCIENTIFIC STATUS
# =============================================================================

section("FINAL SCIENTIFIC INTERPRETATION")

if (
    b1_recovery_rate >= 0.999
    and nondecreasing
):

    final_status = (
        "RAW_TIMESTAMP_RECOVERY_SUCCESSFUL"
    )

elif b1_recovery_rate >= 0.99:

    final_status = (
        "HIGH_RAW_TIMESTAMP_RECOVERY_REQUIRES_REVIEW"
    )

elif b1_recovery_rate >= 0.90:

    final_status = (
        "PARTIAL_RAW_TIMESTAMP_RECOVERY"
    )

else:

    final_status = (
        "INSUFFICIENT_RAW_TIMESTAMP_RECOVERY"
    )

print(
    f"FINAL STATUS: {final_status}"
)

print()
print(
    "Timestamp recovery establishes whether the original "
    "timestamp information can be recovered."
)

print(
    "It does NOT by itself prove that consecutive flows form "
    "the same network session or connection."
)


# =============================================================================
# SAVE SUMMARY
# =============================================================================

section("SAVING STEP 30 SUMMARY")

summary = {
    "step": "30",
    "purpose": (
        "Recover and validate original CICIDS2018 timestamps "
        "from raw TrafficForML_CICFlowMeter CSV files."
    ),
    "raw_directory": str(RAW_DIR),
    "raw_file_count": len(raw_files),
    "original_train_path": str(ORIGINAL_TRAIN),
    "b1_train_path": str(B1_TRAIN),
    "original_train_rows": int(len(original_df)),
    "b1_rows": int(len(b1_df)),
    "raw_rows_scanned": int(total_raw_rows),
    "total_raw_matching_rows": int(total_matching_raw_rows),
    "original_unique_hash_groups": int(
        len(original_hash_map)
    ),
    "original_duplicate_hash_groups": int(
        duplicate_hash_groups
    ),
    "maximum_original_hash_multiplicity": int(
        max_multiplicity
    ),
    "b1_unique_hash_matches": int(b1_unique),
    "b1_ambiguous_hash_matches": int(b1_ambiguous),
    "b1_unmatched_hashes": int(b1_unmatched),
    "original_rows_recovered_from_raw": int(
        original_recovered
    ),
    "original_rows_unmatched_from_raw": int(
        original_unmatched
    ),
    "original_rows_with_multiple_raw_matches": int(
        original_multiple
    ),
    "duplicate_raw_matches": int(
        duplicate_raw_matches
    ),
    "valid_original_recovered_timestamps": int(
        valid_original_ts
    ),
    "invalid_original_recovered_timestamps": int(
        invalid_original_ts
    ),
    "b1_valid_recovered_timestamps": int(
        b1_valid
    ),
    "b1_missing_timestamps": int(
        b1_missing
    ),
    "b1_recovery_rate": float(
        b1_recovery_rate
    ),
    "b1_unique_mapping_timestamp_recovery": int(
        b1_unique_mapping
    ),
    "b1_ambiguous_identical_timestamp_recovery": int(
        b1_ambiguous_identical
    ),
    "b1_negative_time_differences": int(
        negative
    ),
    "b1_zero_time_differences": int(
        zero
    ),
    "b1_positive_time_differences": int(
        positive
    ),
    "b1_unique_timestamps": int(
        unique_timestamps
    ),
    "b1_duplicated_timestamp_rows": int(
        duplicated_rows
    ),
    "positive_gap_count": int(
        len(positive_gaps)
    ),
    "minimum_positive_gap_seconds": min_gap,
    "maximum_positive_gap_seconds": max_gap,
    "median_positive_gap_seconds": median_gap,
    "p95_positive_gap_seconds": p95_gap,
    "chronological_order_non_decreasing": bool(
        nondecreasing
    ),
    "final_status": final_status,
    "runtime_seconds": float(
        time.time() - start_time
    ),
    "raw_file_statistics": file_stats,
}

summary_path = (
    OUTPUT_DIR
    / "step30_raw_timestamp_recovery_summary.json"
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
    f"Saved: {summary_path}"
)


# =============================================================================
# FINAL
# =============================================================================

section("STEP 30 COMPLETE")

print(
    f"Raw files scanned          : {len(raw_files):,}"
)

print(
    f"Raw rows scanned           : {total_raw_rows:,}"
)

print(
    f"Original rows recovered    : {original_recovered:,}"
)

print(
    f"Original rows unmatched    : {original_unmatched:,}"
)

print(
    f"B1 rows                    : {len(b1_df):,}"
)

print(
    f"B1 timestamps recovered    : {b1_valid:,}"
)

print(
    f"B1 timestamps missing      : {b1_missing:,}"
)

print(
    f"B1 recovery rate           : "
    f"{b1_recovery_rate * 100:.6f}%"
)

print(
    f"Chronological nondecreasing: {nondecreasing}"
)

print(
    f"FINAL STATUS               : {final_status}"
)

print(
    f"Runtime                    : "
    f"{time.time() - start_time:.2f} sec"
)

print()
print(
    "DO NOT retrain the LSTM yet."
)

print(
    "Send the COMPLETE terminal output for interpretation."
)

print("=" * 80)