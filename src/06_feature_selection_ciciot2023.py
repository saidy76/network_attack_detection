import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    mutual_info_classif,
    f_classif
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

warnings.filterwarnings("ignore")


# ==============================================================
# STEP 6 - RESEARCH-GRADE MULTICLASS FEATURE SELECTION
# ==============================================================

print("=" * 75)
print("STEP 6 - MULTI-METHOD MULTICLASS FEATURE SELECTION")
print("=" * 75)


# ==============================================================
# 1. PATH CONFIGURATION
# ==============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "modeling"
)

EXPERIMENT_DIR = os.path.join(
    BASE_DIR,
    "experiments"
)

os.makedirs(
    EXPERIMENT_DIR,
    exist_ok=True
)


TRAIN_PATH = os.path.join(
    MODEL_DIR,
    "CICIoT2023_train_clean.csv"
)


# ==============================================================
# OUTPUT FILES
# ==============================================================

MI_REPORT = os.path.join(
    EXPERIMENT_DIR,
    "step6_mutual_information.csv"
)

ANOVA_REPORT = os.path.join(
    EXPERIMENT_DIR,
    "step6_anova_f_scores.csv"
)

RF_REPORT = os.path.join(
    EXPERIMENT_DIR,
    "step6_random_forest_importance.csv"
)

ET_REPORT = os.path.join(
    EXPERIMENT_DIR,
    "step6_extra_trees_importance.csv"
)

CONSENSUS_REPORT = os.path.join(
    EXPERIMENT_DIR,
    "step6_feature_selection_consensus.csv"
)

CORRELATION_REPORT = os.path.join(
    EXPERIMENT_DIR,
    "step6_feature_correlation.csv"
)

SUMMARY_PATH = os.path.join(
    EXPERIMENT_DIR,
    "step6_feature_selection_summary.json"
)


# ==============================================================
# 2. LOAD TRAINING DATA ONLY
# ==============================================================

print("\n" + "=" * 75)
print("LOADING TRAINING DATA")
print("=" * 75)

if not os.path.exists(TRAIN_PATH):

    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_PATH}"
    )


print(
    "Loading CICIoT2023 training data..."
)

train_df = pd.read_csv(
    TRAIN_PATH
)

print(
    f"[OK] Training shape: "
    f"{train_df.shape}"
)


# ==============================================================
# 3. TARGET SELECTION
# ==============================================================

print("\n" + "=" * 75)
print("MULTICLASS TARGET")
print("=" * 75)

TARGET = "Label"

if TARGET not in train_df.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


print(
    f"Target: {TARGET}"
)

print(
    f"Number of raw attack classes: "
    f"{train_df[TARGET].nunique()}"
)


# ==============================================================
# 4. PREPARE FEATURES
# ==============================================================

excluded_columns = [
    "Label",
    "Attack_Family"
]

feature_columns = [
    column
    for column in train_df.columns
    if column not in excluded_columns
]


X = train_df[
    feature_columns
].copy()

y = train_df[
    TARGET
].copy()


print("\n" + "=" * 75)
print("FEATURE DATA")
print("=" * 75)

print(
    f"Number of features: "
    f"{len(feature_columns)}"
)

print(
    f"Number of samples: "
    f"{len(X)}"
)

print(
    f"Number of classes: "
    f"{y.nunique()}"
)


# ==============================================================
# 5. REMOVE ONLY NON-FINITE VALUES IF ANY
# ==============================================================

print("\n" + "=" * 75)
print("FINITE VALUE VALIDATION")
print("=" * 75)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

if X.isna().sum().sum() > 0:

    print(
        "[WARNING] NaN values detected."
    )

    X = X.fillna(
        X.median()
    )

else:

    print(
        "[OK] No missing values."
    )


# ==============================================================
# 6. CLASS DISTRIBUTION
# ==============================================================

print("\n" + "=" * 75)
print("MULTICLASS DISTRIBUTION")
print("=" * 75)

class_counts = y.value_counts()

print(
    class_counts.to_string()
)


# ==============================================================
# 7. MUTUAL INFORMATION
# ==============================================================

print("\n" + "=" * 75)
print("METHOD 1 - MUTUAL INFORMATION")
print("=" * 75)

print(
    "Calculating nonlinear feature relevance..."
)

mi_scores = mutual_info_classif(
    X,
    y,
    random_state=42,
    n_neighbors=3
)

mi_df = pd.DataFrame({

    "feature": feature_columns,

    "mutual_information": mi_scores

})

mi_df = mi_df.sort_values(
    "mutual_information",
    ascending=False
).reset_index(
    drop=True
)

mi_df[
    "MI_rank"
] = np.arange(
    1,
    len(mi_df) + 1
)

mi_df.to_csv(
    MI_REPORT,
    index=False
)

print(
    "[OK] Mutual Information completed."
)

print(
    "\nTop features:"
)

print(
    mi_df[
        [
            "feature",
            "mutual_information"
        ]
    ].head(15).to_string(
        index=False
    )
)


# ==============================================================
# 8. ANOVA F-TEST
# ==============================================================

print("\n" + "=" * 75)
print("METHOD 2 - ANOVA F-TEST")
print("=" * 75)

print(
    "Calculating multiclass statistical relevance..."
)

f_scores, p_values = f_classif(
    X,
    y
)

f_scores = np.nan_to_num(
    f_scores,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

p_values = np.nan_to_num(
    p_values,
    nan=1.0,
    posinf=1.0,
    neginf=1.0
)

anova_df = pd.DataFrame({

    "feature": feature_columns,

    "F_score": f_scores,

    "p_value": p_values

})

anova_df = anova_df.sort_values(
    "F_score",
    ascending=False
).reset_index(
    drop=True
)

anova_df[
    "ANOVA_rank"
] = np.arange(
    1,
    len(anova_df) + 1
)

anova_df.to_csv(
    ANOVA_REPORT,
    index=False
)

print(
    "[OK] ANOVA analysis completed."
)

print(
    "\nTop features:"
)

print(
    anova_df[
        [
            "feature",
            "F_score",
            "p_value"
        ]
    ].head(15).to_string(
        index=False
    )
)


# ==============================================================
# 9. RANDOM FOREST IMPORTANCE
# ==============================================================

print("\n" + "=" * 75)
print("METHOD 3 - RANDOM FOREST IMPORTANCE")
print("=" * 75)

print(
    "Training Random Forest for feature importance..."
)

rf_model = RandomForestClassifier(

    n_estimators=150,

    max_depth=None,

    min_samples_leaf=2,

    class_weight="balanced_subsample",

    n_jobs=-1,

    random_state=42

)


rf_model.fit(
    X,
    y
)


rf_importance = rf_model.feature_importances_


rf_df = pd.DataFrame({

    "feature": feature_columns,

    "RF_importance": rf_importance

})

rf_df = rf_df.sort_values(
    "RF_importance",
    ascending=False
).reset_index(
    drop=True
)

rf_df[
    "RF_rank"
] = np.arange(
    1,
    len(rf_df) + 1
)

rf_df.to_csv(
    RF_REPORT,
    index=False
)

print(
    "[OK] Random Forest importance completed."
)

print(
    "\nTop features:"
)

print(
    rf_df[
        [
            "feature",
            "RF_importance"
        ]
    ].head(15).to_string(
        index=False
    )
)


# ==============================================================
# 10. EXTRA TREES IMPORTANCE
# ==============================================================

print("\n" + "=" * 75)
print("METHOD 4 - EXTRA TREES IMPORTANCE")
print("=" * 75)

print(
    "Training Extra Trees for independent feature importance..."
)

et_model = ExtraTreesClassifier(

    n_estimators=150,

    min_samples_leaf=2,

    class_weight="balanced",

    n_jobs=-1,

    random_state=42

)


et_model.fit(
    X,
    y
)


et_importance = (
    et_model.feature_importances_
)


et_df = pd.DataFrame({

    "feature": feature_columns,

    "ET_importance": et_importance

})

et_df = et_df.sort_values(
    "ET_importance",
    ascending=False
).reset_index(
    drop=True
)

et_df[
    "ET_rank"
] = np.arange(
    1,
    len(et_df) + 1
)

et_df.to_csv(
    ET_REPORT,
    index=False
)

print(
    "[OK] Extra Trees importance completed."
)

print(
    "\nTop features:"
)

print(
    et_df[
        [
            "feature",
            "ET_importance"
        ]
    ].head(15).to_string(
        index=False
    )
)


# ==============================================================
# 11. MERGE ALL RANKINGS
# ==============================================================

print("\n" + "=" * 75)
print("CREATING FEATURE CONSENSUS")
print("=" * 75)


consensus_df = mi_df[
    [
        "feature",
        "mutual_information",
        "MI_rank"
    ]
].copy()


consensus_df = consensus_df.merge(

    anova_df[
        [
            "feature",
            "F_score",
            "p_value",
            "ANOVA_rank"
        ]
    ],

    on="feature"

)


consensus_df = consensus_df.merge(

    rf_df[
        [
            "feature",
            "RF_importance",
            "RF_rank"
        ]
    ],

    on="feature"

)


consensus_df = consensus_df.merge(

    et_df[
        [
            "feature",
            "ET_importance",
            "ET_rank"
        ]
    ],

    on="feature"

)


# ==============================================================
# 12. NORMALIZED RANK
# ==============================================================

max_rank = len(
    feature_columns
)

consensus_df[
    "MI_rank_score"
] = (
    1 -
    (
        consensus_df["MI_rank"] - 1
    ) / (max_rank - 1)
)


consensus_df[
    "ANOVA_rank_score"
] = (
    1 -
    (
        consensus_df["ANOVA_rank"] - 1
    ) / (max_rank - 1)
)


consensus_df[
    "RF_rank_score"
] = (
    1 -
    (
        consensus_df["RF_rank"] - 1
    ) / (max_rank - 1)
)


consensus_df[
    "ET_rank_score"
] = (
    1 -
    (
        consensus_df["ET_rank"] - 1
    ) / (max_rank - 1)
)


# ==============================================================
# 13. CONSENSUS SCORE
# ==============================================================

consensus_df[
    "consensus_score"
] = (

    0.25 *
    consensus_df["MI_rank_score"]

    +

    0.25 *
    consensus_df["ANOVA_rank_score"]

    +

    0.25 *
    consensus_df["RF_rank_score"]

    +

    0.25 *
    consensus_df["ET_rank_score"]

)


# ==============================================================
# 14. FINAL CONSENSUS RANK
# ==============================================================

consensus_df = consensus_df.sort_values(

    "consensus_score",

    ascending=False

).reset_index(
    drop=True
)


consensus_df[
    "consensus_rank"
] = np.arange(
    1,
    len(consensus_df) + 1
)


# ==============================================================
# 15. TOP-K INDICATORS
# ==============================================================

for k in [10, 15, 20, 25, 30]:

    consensus_df[
        f"top_{k}"
    ] = (
        consensus_df[
            "consensus_rank"
        ] <= k
    )


# ==============================================================
# 16. SAVE CONSENSUS REPORT
# ==============================================================

consensus_df.to_csv(
    CONSENSUS_REPORT,
    index=False
)


print(
    "[OK] Consensus feature ranking saved."
)

print(
    f"\nReport:\n{CONSENSUS_REPORT}"
)


# ==============================================================
# 17. DISPLAY FINAL RANKING
# ==============================================================

print("\n" + "=" * 75)
print("FINAL FEATURE RANKING")
print("=" * 75)

display_columns = [

    "consensus_rank",

    "feature",

    "consensus_score",

    "MI_rank",

    "ANOVA_rank",

    "RF_rank",

    "ET_rank"

]


print(
    consensus_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ==============================================================
# 18. CORRELATION ANALYSIS
# ==============================================================

print("\n" + "=" * 75)
print("FEATURE CORRELATION ANALYSIS")
print("=" * 75)

correlation_matrix = X[
    feature_columns
].corr(
    method="spearman"
)


correlation_records = []

for i in range(
    len(feature_columns)
):

    for j in range(
        i + 1,
        len(feature_columns)
    ):

        feature_a = feature_columns[i]

        feature_b = feature_columns[j]

        correlation = correlation_matrix.loc[
            feature_a,
            feature_b
        ]

        correlation_records.append({

            "feature_1": feature_a,

            "feature_2": feature_b,

            "spearman_correlation": correlation,

            "absolute_correlation": abs(
                correlation
            )

        })


correlation_df = pd.DataFrame(
    correlation_records
)


correlation_df = correlation_df.sort_values(
    "absolute_correlation",
    ascending=False
)


correlation_df.to_csv(
    CORRELATION_REPORT,
    index=False
)


print(
    "[OK] Correlation analysis completed."
)

print(
    "\nMost highly correlated feature pairs:"
)

print(
    correlation_df.head(15).to_string(
        index=False
    )
)


# ==============================================================
# 19. FEATURE SELECTION SUMMARY
# ==============================================================

top10 = consensus_df.head(
    10
)["feature"].tolist()

top15 = consensus_df.head(
    15
)["feature"].tolist()

top20 = consensus_df.head(
    20
)["feature"].tolist()


summary = {

    "step": 6,

    "dataset": "CICIoT2023",

    "target": "Label",

    "problem_type": "34-class multiclass classification",

    "original_feature_count": len(
        feature_columns
    ),

    "methods": [

        "Mutual Information",

        "ANOVA F-test",

        "Random Forest importance",

        "Extra Trees importance"

    ],

    "all_methods_training_only": True,

    "validation_used_for_selection": False,

    "test_used_for_selection": False,

    "smote_applied": False,

    "scaling_applied": False,

    "top_10_features": top10,

    "top_15_features": top15,

    "top_20_features": top20

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
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 75)

print(
    "\nIMPORTANT:"
)

print(
    "1. Feature selection used TRAINING DATA ONLY."
)

print(
    "2. Validation data was not used."
)

print(
    "3. Test data was not used."
)

print(
    "4. Four independent feature-ranking methods were used."
)

print(
    "5. Consensus ranking was generated."
)

print(
    "6. No final feature subset was permanently selected yet."
)

print(
    "7. Correlation analysis was also completed."
)

print(
    "\nNext stage:"
)

print(
    "Evaluate candidate feature subsets using validation data."
)

print("=" * 75)