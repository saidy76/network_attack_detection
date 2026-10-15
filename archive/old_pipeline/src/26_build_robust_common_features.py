import pandas as pd
import os

print("=" * 70)
print("STEP 26 - ROBUST CROSS-DATASET FEATURE MAPPING")
print("=" * 70)

# ============================================================
# FILE PATHS
# ============================================================

CICIDS_FILE = "data/CICIDS2017_improved_features.csv"
NSLKDD_FILE = "data/NSLKDD_improved_features.csv"

# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading CICIDS2017...")
cicids = pd.read_csv(CICIDS_FILE)

print("Loading NSL-KDD...")
nslkdd = pd.read_csv(NSLKDD_FILE)

print("\nOriginal shapes:")
print("CICIDS2017:", cicids.shape)
print("NSL-KDD:", nslkdd.shape)

# ============================================================
# DISPLAY AVAILABLE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CICIDS2017 AVAILABLE FEATURES")
print("=" * 70)

for i, col in enumerate(cicids.columns, 1):
    print(f"{i:2}. {col}")

print("\n" + "=" * 70)
print("NSL-KDD AVAILABLE FEATURES")
print("=" * 70)

for i, col in enumerate(nslkdd.columns, 1):
    print(f"{i:2}. {col}")

# ============================================================
# PROPOSED SEMANTIC MAPPING
# ============================================================

mapping = [
    {
        "Concept": "Forward Traffic Volume",
        "CICIDS2017": "forward_bytes",
        "NSLKDD": "src_bytes",
        "Comparable": "YES",
        "Reason": "Both represent bytes sent in the forward/source direction"
    },
    {
        "Concept": "Backward Traffic Volume",
        "CICIDS2017": "backward_bytes",
        "NSLKDD": "dst_bytes",
        "Comparable": "YES",
        "Reason": "Both represent bytes sent in the backward/destination direction"
    },
    {
        "Concept": "Total Traffic Volume",
        "CICIDS2017": "total_bytes",
        "NSLKDD": "src_bytes + dst_bytes",
        "Comparable": "YES",
        "Reason": "Total bytes transferred in both directions"
    },
    {
        "Concept": "Traffic Direction Ratio",
        "CICIDS2017": "traffic_ratio",
        "NSLKDD": "src_bytes / (dst_bytes + 1)",
        "Comparable": "YES",
        "Reason": "Represents relative traffic volume between directions"
    },
    {
        "Concept": "Service Similarity",
        "CICIDS2017": "NOT AVAILABLE IN CURRENT SELECTED FEATURES",
        "NSLKDD": "same_srv_rate",
        "Comparable": "NO",
        "Reason": "No sufficiently equivalent CICIDS feature in current selected feature set"
    },
    {
        "Concept": "Service Diversity",
        "CICIDS2017": "NOT AVAILABLE IN CURRENT SELECTED FEATURES",
        "NSLKDD": "diff_srv_rate",
        "Comparable": "NO",
        "Reason": "No sufficiently equivalent CICIDS feature in current selected feature set"
    },
    {
        "Concept": "Connection Count",
        "CICIDS2017": "NOT AVAILABLE IN CURRENT SELECTED FEATURES",
        "NSLKDD": "count",
        "Comparable": "NO",
        "Reason": "Exact temporal/window definition differs between datasets"
    },
    {
        "Concept": "Server Error Rate",
        "CICIDS2017": "NOT AVAILABLE IN CURRENT SELECTED FEATURES",
        "NSLKDD": "srv_serror_rate",
        "Comparable": "NO",
        "Reason": "No sufficiently equivalent CICIDS feature in current selected feature set"
    },
    {
        "Concept": "Error Rate",
        "CICIDS2017": "NOT AVAILABLE IN CURRENT SELECTED FEATURES",
        "NSLKDD": "serror_rate",
        "Comparable": "NO",
        "Reason": "No sufficiently equivalent CICIDS feature in current selected feature set"
    },
    {
        "Concept": "Login Status",
        "CICIDS2017": "NOT AVAILABLE IN CURRENT SELECTED FEATURES",
        "NSLKDD": "logged_in",
        "Comparable": "NO",
        "Reason": "No sufficiently equivalent CICIDS feature in current selected feature set"
    }
]

mapping_df = pd.DataFrame(mapping)

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("MAPPING VALIDATION")
print("=" * 70)

for item in mapping:

    cic_feature = item["CICIDS2017"]
    nsl_feature = item["NSLKDD"]

    print(f"\nConcept: {item['Concept']}")
    print(f"  CICIDS2017 : {cic_feature}")
    print(f"  NSL-KDD    : {nsl_feature}")
    print(f"  Comparable : {item['Comparable']}")
    print(f"  Reason     : {item['Reason']}")

# ============================================================
# COUNT COMPARABLE FEATURES
# ============================================================

comparable = mapping_df[mapping_df["Comparable"] == "YES"]

print("\n" + "=" * 70)
print("COMPARABLE FEATURE SUMMARY")
print("=" * 70)

print("\nNumber of proposed semantic concepts:", len(mapping_df))
print("Number of directly comparable concepts:", len(comparable))

print("\nDirectly comparable features:")

for i, row in comparable.iterrows():
    print(
        f"{row['Concept']}: "
        f"{row['CICIDS2017']} <-> {row['NSLKDD']}"
    )

# ============================================================
# SAVE MAPPING REPORT
# ============================================================

output_file = "data/robust_feature_mapping.csv"

mapping_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("STEP 26 COMPLETED")
print("=" * 70)

print("\nMapping report saved to:")
print(output_file)

print("\nIMPORTANT:")
print("Only semantically defensible features will be used")
print("for the final cross-dataset representation.")