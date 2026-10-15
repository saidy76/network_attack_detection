import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skew

# ================================================================
# STEP 31 - CROSS-DATASET DISTRIBUTION ANALYSIS
#                  AND ROBUST FEATURE TRANSFORMATION
# ================================================================

print("=" * 70)
print("STEP 31 - CROSS-DATASET DISTRIBUTION ANALYSIS")
print("=" * 70)


# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CICIDS_INPUT = DATA_DIR / "CICIDS2017_cross_dataset_common.csv"
NSLKDD_INPUT = DATA_DIR / "NSLKDD_cross_dataset_common.csv"

CICIDS_OUTPUT = DATA_DIR / "CICIDS2017_cross_dataset_transformed.csv"
NSLKDD_OUTPUT = DATA_DIR / "NSLKDD_cross_dataset_transformed.csv"

STATS_OUTPUT = DATA_DIR / "cross_dataset_distribution_analysis.csv"


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

print("\nLoading CICIDS2017...")
cicids = pd.read_csv(CICIDS_INPUT)

print("Loading NSL-KDD...")
nslkdd = pd.read_csv(NSLKDD_INPUT)


print("\nOriginal shapes:")
print("CICIDS2017:", cicids.shape)
print("NSL-KDD   :", nslkdd.shape)


# ------------------------------------------------
# COMMON FEATURES
# ------------------------------------------------

features = [
    "forward_bytes",
    "backward_bytes",
    "total_bytes",
    "traffic_ratio"
]


# ------------------------------------------------
# FEATURE VALIDATION
# ------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE VALIDATION")
print("=" * 70)

for feature in features:

    if feature not in cicids.columns:
        raise ValueError(
            f"CICIDS2017 missing feature: {feature}"
        )

    if feature not in nslkdd.columns:
        raise ValueError(
            f"NSL-KDD missing feature: {feature}"
        )

    print(f"[OK] {feature}")


# ------------------------------------------------
# DATA VALIDATION
# ------------------------------------------------

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

for name, df in [
    ("CICIDS2017", cicids),
    ("NSL-KDD", nslkdd)
]:

    missing = int(
        df[features].isnull().sum().sum()
    )

    infinity = int(
        np.isinf(
            df[features].to_numpy()
        ).sum()
    )

    print(f"\n{name}:")
    print("Missing values :", missing)
    print("Infinity values:", infinity)

    if missing > 0:
        raise ValueError(
            f"{name} contains missing values."
        )

    if infinity > 0:
        raise ValueError(
            f"{name} contains infinity values."
        )


# ------------------------------------------------
# DISTRIBUTION ANALYSIS
# ------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE DISTRIBUTION ANALYSIS")
print("=" * 70)


results = []


for feature in features:

    cic_values = pd.to_numeric(
        cicids[feature],
        errors="coerce"
    )

    nsl_values = pd.to_numeric(
        nslkdd[feature],
        errors="coerce"
    )

    cic_skew = skew(
        cic_values,
        nan_policy="omit"
    )

    nsl_skew = skew(
        nsl_values,
        nan_policy="omit"
    )

    cic_median = cic_values.median()
    nsl_median = nsl_values.median()

    cic_mean = cic_values.mean()
    nsl_mean = nsl_values.mean()

    cic_std = cic_values.std()
    nsl_std = nsl_values.std()

    print("\n" + "-" * 70)
    print(f"Feature: {feature}")

    print("\nCICIDS2017:")
    print(f"Mean   : {cic_mean:.6f}")
    print(f"Median : {cic_median:.6f}")
    print(f"Std    : {cic_std:.6f}")
    print(f"Skew   : {cic_skew:.6f}")

    print("\nNSL-KDD:")
    print(f"Mean   : {nsl_mean:.6f}")
    print(f"Median : {nsl_median:.6f}")
    print(f"Std    : {nsl_std:.6f}")
    print(f"Skew   : {nsl_skew:.6f}")


    # ------------------------------------------------
    # SKEWNESS DECISION
    # ------------------------------------------------

    max_abs_skew = max(
        abs(cic_skew),
        abs(nsl_skew)
    )

    if max_abs_skew > 1.0:

        transformation = "log1p"

        print(
            "\nTransformation decision:"
        )

        print(
            "HIGH SKEWNESS -> log1p recommended"
        )

    else:

        transformation = "none"

        print(
            "\nTransformation decision:"
        )

        print(
            "Moderate skewness -> no log transformation required"
        )


    results.append({

        "Feature": feature,

        "CICIDS2017_Mean": cic_mean,
        "CICIDS2017_Median": cic_median,
        "CICIDS2017_Std": cic_std,
        "CICIDS2017_Skewness": cic_skew,

        "NSLKDD_Mean": nsl_mean,
        "NSLKDD_Median": nsl_median,
        "NSLKDD_Std": nsl_std,
        "NSLKDD_Skewness": nsl_skew,

        "Transformation": transformation

    })


# ------------------------------------------------
# SAVE DISTRIBUTION ANALYSIS
# ------------------------------------------------

analysis_df = pd.DataFrame(results)

analysis_df.to_csv(
    STATS_OUTPUT,
    index=False
)


# ------------------------------------------------
# APPLY DOMAIN-INDEPENDENT TRANSFORMATION
# ------------------------------------------------

print("\n" + "=" * 70)
print("APPLYING DOMAIN-INDEPENDENT TRANSFORMATION")
print("=" * 70)

print(
    "\nImportant:"
)

print(
    "Transformation is applied independently to both datasets."
)

print(
    "No scaling parameters are learned from test data."
)


# ------------------------------------------------
# COPY DATA
# ------------------------------------------------

cicids_transformed = cicids.copy()
nslkdd_transformed = nslkdd.copy()


# ------------------------------------------------
# LOG TRANSFORMATION
# ------------------------------------------------

for feature in features:

    transformation = analysis_df.loc[
        analysis_df["Feature"] == feature,
        "Transformation"
    ].iloc[0]

    if transformation == "log1p":

        print(
            f"[LOG1P] {feature}"
        )

        cicids_transformed[feature] = np.log1p(
            cicids_transformed[feature]
        )

        nslkdd_transformed[feature] = np.log1p(
            nslkdd_transformed[feature]
        )

    else:

        print(
            f"[UNCHANGED] {feature}"
        )


# ------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


for name, df in [
    ("CICIDS2017", cicids_transformed),
    ("NSL-KDD", nslkdd_transformed)
]:

    missing = int(
        df[features].isnull().sum().sum()
    )

    infinity = int(
        np.isinf(
            df[features].to_numpy()
        ).sum()
    )

    print(f"\n{name}:")

    print("Shape:", df.shape)

    print("Missing values:", missing)

    print("Infinity values:", infinity)

    if missing > 0:
        raise ValueError(
            f"{name} contains missing values after transformation."
        )

    if infinity > 0:
        raise ValueError(
            f"{name} contains infinity after transformation."
        )


# ------------------------------------------------
# SAVE TRANSFORMED DATA
# ------------------------------------------------

print("\n" + "=" * 70)
print("SAVING TRANSFORMED DATA")
print("=" * 70)


cicids_transformed.to_csv(
    CICIDS_OUTPUT,
    index=False
)

nslkdd_transformed.to_csv(
    NSLKDD_OUTPUT,
    index=False
)


# ------------------------------------------------
# FINAL FEATURE STATISTICS
# ------------------------------------------------

print("\n" + "=" * 70)
print("TRANSFORMED FEATURE SUMMARY")
print("=" * 70)

print("\nCICIDS2017:")

print(
    cicids_transformed[
        features
    ].describe()
)

print("\nNSL-KDD:")

print(
    nslkdd_transformed[
        features
    ].describe()
)


# ------------------------------------------------
# COMPLETION
# ------------------------------------------------

print("\n" + "=" * 70)
print("STEP 31 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nDistribution analysis saved to:")
print(STATS_OUTPUT)

print("\nCICIDS2017 transformed dataset saved to:")
print(CICIDS_OUTPUT)

print("\nNSL-KDD transformed dataset saved to:")
print(NSLKDD_OUTPUT)

print("\nIMPORTANT:")
print(
    "This step performs distribution analysis and "
    "domain-independent log transformation."
)

print(
    "No scaler was fitted using test data."
)

print(
    "Scaling parameters will be learned only from "
    "training data in a later modeling step."
)