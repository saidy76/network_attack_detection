import os
import json
import numpy as np
import pandas as pd


# ==============================================================
# STEP 5 - DATA CLEANING & FEATURE QUALITY ANALYSIS
# ==============================================================

print("=" * 75)
print("STEP 5 - CICIoT2023 DATA CLEANING & FEATURE QUALITY ANALYSIS")
print("=" * 75)


# ==============================================================
# 1. PATH CONFIGURATION
# ==============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "data", "modeling")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiments")

os.makedirs(EXPERIMENT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(
    MODEL_DIR,
    "CICIoT2023_train.csv"
)

VALIDATION_PATH = os.path.join(
    MODEL_DIR,
    "CICIoT2023_validation.csv"
)

TEST_PATH = os.path.join(
    MODEL_DIR,
    "CICIoT2023_test.csv"
)


# ==============================================================
# 2. OUTPUT FILES
# ==============================================================

CLEAN_TRAIN_PATH = os.path.join(
    MODEL_DIR,
    "CICIoT2023_train_clean.csv"
)

CLEAN_VALIDATION_PATH = os.path.join(
    MODEL_DIR,
    "CICIoT2023_validation_clean.csv"
)

CLEAN_TEST_PATH = os.path.join(
    MODEL_DIR,
    "CICIoT2023_test_clean.csv"
)

QUALITY_REPORT_PATH = os.path.join(
    EXPERIMENT_DIR,
    "step5_feature_quality_report.csv"
)

MEDIAN_REPORT_PATH = os.path.join(
    EXPERIMENT_DIR,
    "step5_training_medians.csv"
)

CONSTANT_FEATURE_PATH = os.path.join(
    EXPERIMENT_DIR,
    "step5_constant_features.csv"
)

SUMMARY_PATH = os.path.join(
    EXPERIMENT_DIR,
    "step5_cleaning_summary.json"
)


# ==============================================================
# 3. CHECK INPUT FILES
# ==============================================================

print("\n" + "=" * 75)
print("CHECKING INPUT FILES")
print("=" * 75)

required_files = [
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH
]

for file_path in required_files:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"[ERROR] File not found:\n{file_path}"
        )

    print("[OK]", file_path)


# ==============================================================
# 4. LOAD DATA
# ==============================================================

print("\n" + "=" * 75)
print("LOADING MODELING DATASETS")
print("=" * 75)

print("\nLoading training data...")
train_df = pd.read_csv(TRAIN_PATH)

print("Loading validation data...")
validation_df = pd.read_csv(VALIDATION_PATH)

print("Loading test data...")
test_df = pd.read_csv(TEST_PATH)


print("\nOriginal shapes:")

print(f"CICIoT2023 Train      : {train_df.shape}")
print(f"CICIoT2023 Validation : {validation_df.shape}")
print(f"CICIoT2023 Test       : {test_df.shape}")


# ==============================================================
# 5. CHECK REQUIRED LABEL COLUMNS
# ==============================================================

print("\n" + "=" * 75)
print("VALIDATING LABEL COLUMNS")
print("=" * 75)

required_columns = [
    "Label",
    "Attack_Family"
]

for column in required_columns:

    for name, df in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df)
    ]:

        if column not in df.columns:
            raise ValueError(
                f"[ERROR] {column} missing from {name} dataset."
            )

    print(f"[OK] {column}")


# ==============================================================
# 6. IDENTIFY FEATURE COLUMNS
# ==============================================================

print("\n" + "=" * 75)
print("IDENTIFYING MODEL FEATURES")
print("=" * 75)

excluded_columns = [
    "Label",
    "Attack_Family"
]

feature_columns = [
    column
    for column in train_df.columns
    if column not in excluded_columns
]


print(f"Total columns       : {len(train_df.columns)}")
print(f"Target columns      : {len(excluded_columns)}")
print(f"Feature columns     : {len(feature_columns)}")

print("\nFeatures:")

for i, column in enumerate(feature_columns, start=1):
    print(f"{i:02d}. {column}")


# ==============================================================
# 7. DATA TYPE VALIDATION
# ==============================================================

print("\n" + "=" * 75)
print("DATA TYPE VALIDATION")
print("=" * 75)

non_numeric_features = []

for column in feature_columns:

    if not pd.api.types.is_numeric_dtype(
        train_df[column]
    ):
        non_numeric_features.append(column)


if len(non_numeric_features) == 0:

    print(
        "[OK] All model features are numeric."
    )

else:

    print(
        "[WARNING] Non-numeric features detected:"
    )

    for column in non_numeric_features:
        print(" -", column)


# ==============================================================
# 8. ORIGINAL DATA QUALITY CHECK
# ==============================================================

print("\n" + "=" * 75)
print("ORIGINAL DATA QUALITY")
print("=" * 75)


def quality_check(df, name):

    feature_data = df[feature_columns]

    missing_count = int(
        feature_data.isna().sum().sum()
    )

    infinity_count = int(
        np.isinf(
            feature_data.select_dtypes(
                include=np.number
            ).to_numpy()
        ).sum()
    )

    print(f"\n{name}")

    print(
        f"Missing values   : {missing_count}"
    )

    print(
        f"Infinity values  : {infinity_count}"
    )

    return missing_count, infinity_count


train_missing, train_inf = quality_check(
    train_df,
    "TRAIN"
)

val_missing, val_inf = quality_check(
    validation_df,
    "VALIDATION"
)

test_missing, test_inf = quality_check(
    test_df,
    "TEST"
)


# ==============================================================
# 9. REPLACE INFINITY WITH NaN
# ==============================================================

print("\n" + "=" * 75)
print("CONVERTING INFINITY VALUES")
print("=" * 75)

numeric_features = [
    column
    for column in feature_columns
    if pd.api.types.is_numeric_dtype(
        train_df[column]
    )
]

for df in [
    train_df,
    validation_df,
    test_df
]:

    df[numeric_features] = (
        df[numeric_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

print(
    "[OK] +Infinity and -Infinity converted to NaN."
)


# ==============================================================
# 10. TRAINING-ONLY MEDIAN IMPUTATION
# ==============================================================

print("\n" + "=" * 75)
print("TRAINING-ONLY MEDIAN IMPUTATION")
print("=" * 75)

print(
    "\nIMPORTANT:"
)

print(
    "Median values are calculated ONLY from training data."
)

print(
    "Validation and test data are never used to calculate medians."
)


training_medians = {}

for column in numeric_features:

    median_value = train_df[column].median()

    training_medians[column] = median_value


median_df = pd.DataFrame({
    "feature": list(training_medians.keys()),
    "training_median": list(training_medians.values())
})

median_df.to_csv(
    MEDIAN_REPORT_PATH,
    index=False
)


# Apply training-derived medians to all datasets

for column in numeric_features:

    median_value = training_medians[column]

    train_df[column] = (
        train_df[column]
        .fillna(median_value)
    )

    validation_df[column] = (
        validation_df[column]
        .fillna(median_value)
    )

    test_df[column] = (
        test_df[column]
        .fillna(median_value)
    )


print(
    "[OK] Training-derived median imputation completed."
)


# ==============================================================
# 11. POST-IMPUTATION VALIDATION
# ==============================================================

print("\n" + "=" * 75)
print("POST-IMPUTATION VALIDATION")
print("=" * 75)


def post_cleaning_check(df, name):

    feature_data = df[feature_columns]

    missing_count = int(
        feature_data.isna().sum().sum()
    )

    infinity_count = int(
        np.isinf(
            feature_data.select_dtypes(
                include=np.number
            ).to_numpy()
        ).sum()
    )

    print(f"\n{name}")

    print(
        f"Missing values   : {missing_count}"
    )

    print(
        f"Infinity values  : {infinity_count}"
    )

    return missing_count, infinity_count


train_missing_after, train_inf_after = (
    post_cleaning_check(
        train_df,
        "TRAIN"
    )
)

val_missing_after, val_inf_after = (
    post_cleaning_check(
        validation_df,
        "VALIDATION"
    )
)

test_missing_after, test_inf_after = (
    post_cleaning_check(
        test_df,
        "TEST"
    )
)


# ==============================================================
# 12. CONSTANT FEATURE DETECTION
# ==============================================================

print("\n" + "=" * 75)
print("CONSTANT FEATURE ANALYSIS")
print("=" * 75)

constant_features = []

for column in feature_columns:

    unique_count = train_df[column].nunique(
        dropna=False
    )

    if unique_count <= 1:

        constant_features.append(column)


if len(constant_features) == 0:

    print(
        "[OK] No constant features found."
    )

else:

    print(
        f"Constant features found: "
        f"{len(constant_features)}"
    )

    for column in constant_features:
        print(
            f" - {column}"
        )


# Save constant feature report

constant_df = pd.DataFrame({
    "constant_feature": constant_features
})

constant_df.to_csv(
    CONSTANT_FEATURE_PATH,
    index=False
)


# ==============================================================
# 13. REMOVE CONSTANT FEATURES
# ==============================================================

if len(constant_features) > 0:

    train_df.drop(
        columns=constant_features,
        inplace=True
    )

    validation_df.drop(
        columns=constant_features,
        inplace=True
    )

    test_df.drop(
        columns=constant_features,
        inplace=True
    )

    feature_columns = [
        column
        for column in feature_columns
        if column not in constant_features
    ]

    print(
        "\n[OK] Constant features removed."
    )

else:

    print(
        "\n[OK] No features needed to be removed."
    )


# ==============================================================
# 14. DUPLICATE CHECK AFTER CLEANING
# ==============================================================

print("\n" + "=" * 75)
print("DUPLICATE CHECK")
print("=" * 75)

train_duplicates = int(
    train_df.duplicated().sum()
)

validation_duplicates = int(
    validation_df.duplicated().sum()
)

test_duplicates = int(
    test_df.duplicated().sum()
)

print(
    f"TRAIN duplicates      : {train_duplicates}"
)

print(
    f"VALIDATION duplicates : {validation_duplicates}"
)

print(
    f"TEST duplicates       : {test_duplicates}"
)

print(
    "\n[INFO] No additional duplicate rows "
    "will be removed at this stage."
)


# ==============================================================
# 15. FINAL FEATURE COUNT
# ==============================================================

print("\n" + "=" * 75)
print("FINAL FEATURE SET")
print("=" * 75)

print(
    f"Original feature count : "
    f"{len(feature_columns) + len(constant_features)}"
)

print(
    f"Removed constant       : "
    f"{len(constant_features)}"
)

print(
    f"Final feature count    : "
    f"{len(feature_columns)}"
)


# ==============================================================
# 16. FEATURE QUALITY REPORT
# ==============================================================

print("\n" + "=" * 75)
print("CREATING FEATURE QUALITY REPORT")
print("=" * 75)

quality_records = []

for column in feature_columns:

    train_series = train_df[column]

    quality_records.append({

        "feature": column,

        "dtype": str(
            train_series.dtype
        ),

        "unique_values": int(
            train_series.nunique()
        ),

        "missing_train": int(
            train_series.isna().sum()
        ),

        "missing_validation": int(
            validation_df[column].isna().sum()
        ),

        "missing_test": int(
            test_df[column].isna().sum()
        ),

        "min_train": float(
            train_series.min()
        ),

        "max_train": float(
            train_series.max()
        ),

        "mean_train": float(
            train_series.mean()
        ),

        "std_train": float(
            train_series.std()
        )

    })


quality_df = pd.DataFrame(
    quality_records
)

quality_df.to_csv(
    QUALITY_REPORT_PATH,
    index=False
)

print(
    f"[OK] Feature quality report saved:"
)

print(
    QUALITY_REPORT_PATH
)


# ==============================================================
# 17. FINAL LABEL VALIDATION
# ==============================================================

print("\n" + "=" * 75)
print("FINAL LABEL VALIDATION")
print("=" * 75)

for name, df in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df)
]:

    print(f"\n{name}")

    print(
        f"Raw attack classes: "
        f"{df['Label'].nunique()}"
    )

    print(
        f"Attack families: "
        f"{df['Attack_Family'].nunique()}"
    )

    print(
        f"Missing labels: "
        f"{df['Label'].isna().sum()}"
    )

    print(
        f"Missing attack families: "
        f"{df['Attack_Family'].isna().sum()}"
    )


# ==============================================================
# 18. SAVE CLEANED DATASETS
# ==============================================================

print("\n" + "=" * 75)
print("SAVING CLEANED DATASETS")
print("=" * 75)

train_df.to_csv(
    CLEAN_TRAIN_PATH,
    index=False
)

print(
    "[OK] Training dataset saved."
)

validation_df.to_csv(
    CLEAN_VALIDATION_PATH,
    index=False
)

print(
    "[OK] Validation dataset saved."
)

test_df.to_csv(
    CLEAN_TEST_PATH,
    index=False
)

print(
    "[OK] Test dataset saved."
)


# ==============================================================
# 19. FINAL VALIDATION
# ==============================================================

print("\n" + "=" * 75)
print("FINAL VALIDATION")
print("=" * 75)


def final_validation(df, name):

    feature_data = df[feature_columns]

    missing = int(
        feature_data.isna().sum().sum()
    )

    infinity = int(
        np.isinf(
            feature_data.select_dtypes(
                include=np.number
            ).to_numpy()
        ).sum()
    )

    print(f"\n{name}")

    print(
        f"Shape             : {df.shape}"
    )

    print(
        f"Features          : {len(feature_columns)}"
    )

    print(
        f"Missing values    : {missing}"
    )

    print(
        f"Infinity values   : {infinity}"
    )

    print(
        f"Raw classes       : {df['Label'].nunique()}"
    )

    print(
        f"Attack families   : "
        f"{df['Attack_Family'].nunique()}"
    )

    return missing, infinity


final_train_missing, final_train_inf = (
    final_validation(
        train_df,
        "TRAIN"
    )
)

final_val_missing, final_val_inf = (
    final_validation(
        validation_df,
        "VALIDATION"
    )
)

final_test_missing, final_test_inf = (
    final_validation(
        test_df,
        "TEST"
    )
)


# ==============================================================
# 20. SAVE SUMMARY
# ==============================================================

summary = {

    "step": 5,

    "dataset": "CICIoT2023",

    "train_shape": list(
        train_df.shape
    ),

    "validation_shape": list(
        validation_df.shape
    ),

    "test_shape": list(
        test_df.shape
    ),

    "final_feature_count": len(
        feature_columns
    ),

    "constant_features_removed": (
        constant_features
    ),

    "training_median_imputation": True,

    "scaler_fitted": False,

    "smote_applied": False,

    "feature_selection_applied": False,

    "model_training": False,

    "final_missing_values": {

        "train": final_train_missing,

        "validation": final_val_missing,

        "test": final_test_missing

    },

    "final_infinity_values": {

        "train": final_train_inf,

        "validation": final_val_inf,

        "test": final_test_inf

    }

}


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ==============================================================
# COMPLETE
# ==============================================================

print("\n" + "=" * 75)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nCleaned datasets:")

print(
    f"Train:\n{CLEAN_TRAIN_PATH}"
)

print(
    f"\nValidation:\n{CLEAN_VALIDATION_PATH}"
)

print(
    f"\nTest:\n{CLEAN_TEST_PATH}"
)

print("\nReports:")

print(
    f"Feature quality:\n{QUALITY_REPORT_PATH}"
)

print(
    f"\nTraining medians:\n{MEDIAN_REPORT_PATH}"
)

print(
    f"\nConstant features:\n{CONSTANT_FEATURE_PATH}"
)

print(
    f"\nSummary:\n{SUMMARY_PATH}"
)

print("\n" + "=" * 75)
print("IMPORTANT")
print("=" * 75)

print(
    "1. Infinity values were converted to NaN."
)

print(
    "2. Missing values were imputed using training-only medians."
)

print(
    "3. Validation/test data were never used to calculate medians."
)

print(
    "4. Constant features were detected using training data."
)

print(
    "5. No SMOTE was applied."
)

print(
    "6. No scaling was applied."
)

print(
    "7. No feature selection was applied."
)

print(
    "8. No model was trained."
)

print(
    "9. Labels were not modified."
)

print(
    "10. Attack-family labels were preserved."
)

print("\nNext stage: FEATURE SELECTION")