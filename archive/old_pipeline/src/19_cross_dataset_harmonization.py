import pandas as pd

print("=" * 70)
print("STEP 19 - CROSS-DATASET FEATURE HARMONIZATION")
print("=" * 70)

# =========================================================
# 1. Load final datasets
# =========================================================

cic_file = "data/CICIDS2017_final_features.csv"
nsl_file = "data/NSLKDD_final_features.csv"

print("\nLoading CICIDS2017...")
cic = pd.read_csv(cic_file)

print("Loading NSL-KDD...")
nsl = pd.read_csv(nsl_file)

print("\nCICIDS2017 shape:")
print(cic.shape)

print("\nNSL-KDD shape:")
print(nsl.shape)


# =========================================================
# 2. Get feature names
# =========================================================

cic_features = [col for col in cic.columns if col != "Label"]
nsl_features = [col for col in nsl.columns if col != "Label"]


# =========================================================
# 3. Display CICIDS2017 features
# =========================================================

print("\n" + "=" * 70)
print("CICIDS2017 FINAL FEATURES")
print("=" * 70)

for i, feature in enumerate(cic_features, 1):
    print(f"{i:2}. {feature}")


# =========================================================
# 4. Display NSL-KDD features
# =========================================================

print("\n" + "=" * 70)
print("NSL-KDD FINAL FEATURES")
print("=" * 70)

for i, feature in enumerate(nsl_features, 1):
    print(f"{i:2}. {feature}")


# =========================================================
# 5. Define semantic feature mapping
# =========================================================

mapping = [

    {
        "Concept": "Forward Traffic Volume",
        "CICIDS2017_Feature": "Total Length of Fwd Packets",
        "NSLKDD_Feature": "src_bytes",
        "Comparable": "Yes"
    },

    {
        "Concept": "Backward Traffic Volume",
        "CICIDS2017_Feature": "Total Length of Bwd Packets",
        "NSLKDD_Feature": "dst_bytes",
        "Comparable": "Yes"
    },

    {
        "Concept": "Forward Packet Size",
        "CICIDS2017_Feature": "Fwd Packet Length Max",
        "NSLKDD_Feature": "src_bytes",
        "Comparable": "Partial"
    },

    {
        "Concept": "Backward Packet Size",
        "CICIDS2017_Feature": "Bwd Packet Length Mean",
        "NSLKDD_Feature": "dst_bytes",
        "Comparable": "Partial"
    },

    {
        "Concept": "Packet Size Variation",
        "CICIDS2017_Feature": "Packet Length Variance",
        "NSLKDD_Feature": None,
        "Comparable": "No"
    },

    {
        "Concept": "Packet Size Standard Deviation",
        "CICIDS2017_Feature": "Packet Length Std",
        "NSLKDD_Feature": None,
        "Comparable": "No"
    },

    {
        "Concept": "Average Packet Size",
        "CICIDS2017_Feature": "Average Packet Size",
        "NSLKDD_Feature": None,
        "Comparable": "No"
    },

    {
        "Concept": "Destination Port",
        "CICIDS2017_Feature": "Destination Port",
        "NSLKDD_Feature": None,
        "Comparable": "No"
    },

    {
        "Concept": "Initial Forward Window",
        "CICIDS2017_Feature": "Init_Win_bytes_forward",
        "NSLKDD_Feature": None,
        "Comparable": "No"
    },

    {
        "Concept": "Initial Backward Window",
        "CICIDS2017_Feature": "Init_Win_bytes_backward",
        "NSLKDD_Feature": None,
        "Comparable": "No"
    },

    {
        "Concept": "Forward Inter-Arrival Time",
        "CICIDS2017_Feature": "Fwd IAT Max",
        "NSLKDD_Feature": None,
        "Comparable": "No"
    },

    {
        "Concept": "Forward Header Length",
        "CICIDS2017_Feature": "Fwd Header Length.1",
        "NSLKDD_Feature": None,
        "Comparable": "No"
    },

    {
        "Concept": "Backward Header Length",
        "CICIDS2017_Feature": "Bwd Header Length",
        "NSLKDD_Feature": None,
        "Comparable": "No"
    },

    {
        "Concept": "Service Type",
        "CICIDS2017_Feature": None,
        "NSLKDD_Feature": "service",
        "Comparable": "No"
    },

    {
        "Concept": "Connection Flag",
        "CICIDS2017_Feature": None,
        "NSLKDD_Feature": "flag",
        "Comparable": "No"
    },

    {
        "Concept": "Same Service Rate",
        "CICIDS2017_Feature": None,
        "NSLKDD_Feature": "same_srv_rate",
        "Comparable": "No"
    },

    {
        "Concept": "Different Service Rate",
        "CICIDS2017_Feature": None,
        "NSLKDD_Feature": "diff_srv_rate",
        "Comparable": "No"
    },

    {
        "Concept": "Destination Host Service Count",
        "CICIDS2017_Feature": None,
        "NSLKDD_Feature": "dst_host_srv_count",
        "Comparable": "No"
    },

    {
        "Concept": "Login Status",
        "CICIDS2017_Feature": None,
        "NSLKDD_Feature": "logged_in",
        "Comparable": "No"
    },

    {
        "Concept": "Connection Count",
        "CICIDS2017_Feature": None,
        "NSLKDD_Feature": "count",
        "Comparable": "No"
    },

    {
        "Concept": "Server Error Rate",
        "CICIDS2017_Feature": None,
        "NSLKDD_Feature": "srv_serror_rate",
        "Comparable": "No"
    },

    {
        "Concept": "Error Rate",
        "CICIDS2017_Feature": None,
        "NSLKDD_Feature": "serror_rate",
        "Comparable": "No"
    }
]


# =========================================================
# 6. Convert mapping to DataFrame
# =========================================================

mapping_df = pd.DataFrame(mapping)


# =========================================================
# 7. Validate mapped CICIDS features
# =========================================================

print("\n" + "=" * 70)
print("VALIDATING CICIDS2017 MAPPING")
print("=" * 70)

for feature in mapping_df["CICIDS2017_Feature"].dropna():

    if feature in cic_features:
        print(f"[OK] {feature}")

    else:
        print(f"[WARNING] {feature} not found")


# =========================================================
# 8. Validate mapped NSL-KDD features
# =========================================================

print("\n" + "=" * 70)
print("VALIDATING NSL-KDD MAPPING")
print("=" * 70)

for feature in mapping_df["NSLKDD_Feature"].dropna():

    if feature in nsl_features:
        print(f"[OK] {feature}")

    else:
        print(f"[WARNING] {feature} not found")


# =========================================================
# 9. Display semantic mapping
# =========================================================

print("\n" + "=" * 70)
print("SEMANTIC FEATURE MAPPING")
print("=" * 70)

print(
    mapping_df.to_string(index=False)
)


# =========================================================
# 10. Save mapping
# =========================================================

output_file = "data/CICIDS2017_NSLKDD_feature_mapping.csv"

mapping_df.to_csv(
    output_file,
    index=False
)


# =========================================================
# 11. Summary
# =========================================================

print("\n" + "=" * 70)
print("STEP 19 COMPLETED")
print("=" * 70)

print("\nMapping file saved to:")
print(output_file)

print("\nTotal semantic concepts:")
print(len(mapping_df))

print("\nDirectly comparable concepts:")

comparable = mapping_df[
    mapping_df["Comparable"] == "Yes"
]

print(comparable[
    [
        "Concept",
        "CICIDS2017_Feature",
        "NSLKDD_Feature"
    ]
].to_string(index=False))

print(
    f"\nNumber of directly comparable concepts: "
    f"{len(comparable)}"
)