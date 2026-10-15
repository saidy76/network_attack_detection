from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "step21_rf_tuning"
    / "step21_best_random_forest.joblib"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "CICIDS2018_validation_selected.csv"
)

FEATURE_REFERENCE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "step14_cicids2018_selected_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step41_shap_explainability"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Number of validation samples used for SHAP.
# Large enough for stable global interpretation while
# keeping computation practical for a 150-tree multiclass RF.
SHAP_SAMPLE_SIZE = 5000

# Number of examples per class for class-specific SHAP analysis.
PER_CLASS_SAMPLE_SIZE = 300

# Number of local examples retained.
LOCAL_EXAMPLE_COUNT = 20


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def json_safe(value):
    """Convert NumPy/Pandas values into JSON-safe Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, default=json_safe)


def normalize_feature_name(name):
    return str(name).strip()


def extract_feature_names():
    """
    Obtain the frozen 30-feature schema.

    Priority:
        1. Step 14 selected-feature CSV
        2. Validation CSV columns excluding Label
    """

    if FEATURE_REFERENCE_PATH.exists():
        try:
            ref = pd.read_csv(FEATURE_REFERENCE_PATH)

            # Common possible column names
            candidates = [
                "Feature",
                "feature",
                "Selected Feature",
                "selected_feature",
                "features",
            ]

            for col in candidates:
                if col in ref.columns:
                    features = [
                        normalize_feature_name(x)
                        for x in ref[col].dropna().tolist()
                    ]

                    if len(features) == 30:
                        return features

            # If the file is a one-column CSV
            if ref.shape[1] == 1:
                features = [
                    normalize_feature_name(x)
                    for x in ref.iloc[:, 0].dropna().tolist()
                ]

                if len(features) == 30:
                    return features

        except Exception:
            pass

    # Fallback
    validation_header = pd.read_csv(
        VALIDATION_PATH,
        nrows=0
    )

    features = [
        c for c in validation_header.columns
        if c != "Label"
    ]

    if len(features) != 30:
        raise ValueError(
            f"Expected 30 model features, found {len(features)}."
        )

    return features


# ============================================================
# START
# ============================================================

print("=" * 72)
print("STEP 41 — SHAP EXPLAINABILITY ANALYSIS")
print("=" * 72)

print(f"Project root:      {PROJECT_ROOT}")
print(f"Model:             {MODEL_PATH}")
print(f"Validation data:   {VALIDATION_PATH}")
print(f"Output directory:  {OUTPUT_DIR}")
print()


# ============================================================
# 1. LOAD FROZEN MODEL
# ============================================================

print("[1/9] Loading frozen Step 21 Random Forest...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

if not hasattr(model, "n_features_in_"):
    raise ValueError("Loaded model does not expose n_features_in_.")

if model.n_features_in_ != 30:
    raise ValueError(
        f"Expected 30 model features, got {model.n_features_in_}."
    )

model_classes = [str(x) for x in model.classes_]

print(f"Model type:       {type(model).__name__}")
print(f"Features:         {model.n_features_in_}")
print(f"Classes:          {len(model_classes)}")
print(f"SHAP version:     {shap.__version__}")
print()


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

print("[2/9] Loading CIC-IDS2018 validation data...")

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(
        f"Validation dataset not found: {VALIDATION_PATH}"
    )

feature_names = extract_feature_names()

df = pd.read_csv(VALIDATION_PATH)

if "Label" not in df.columns:
    raise ValueError("Validation dataset must contain 'Label'.")

missing_features = [
    f for f in feature_names
    if f not in df.columns
]

if missing_features:
    raise ValueError(
        "Missing model features:\n"
        + "\n".join(missing_features)
    )

X_full = df[feature_names].copy()
y_full = df["Label"].astype(str).copy()

print(f"Validation rows:  {len(df):,}")
print(f"Feature count:    {len(feature_names)}")
print()


# ============================================================
# 3. REPRODUCIBLE SHAP SAMPLE
# ============================================================

print("[3/9] Creating reproducible validation SHAP sample...")

rng = np.random.default_rng(RANDOM_STATE)

sample_size = min(SHAP_SAMPLE_SIZE, len(X_full))

sample_indices = rng.choice(
    len(X_full),
    size=sample_size,
    replace=False
)

sample_indices = np.sort(sample_indices)

X_shap = X_full.iloc[sample_indices].copy()
y_shap = y_full.iloc[sample_indices].copy()

print(f"SHAP sample size: {len(X_shap):,}")
print()


# ============================================================
# 4. MODEL PREDICTIONS
# ============================================================

print("[4/9] Generating frozen-model predictions...")

predicted_labels = model.predict(X_shap)

probabilities = model.predict_proba(X_shap)

predicted_indices = np.argmax(probabilities, axis=1)

predicted_confidence = probabilities[
    np.arange(len(probabilities)),
    predicted_indices
]

correct = predicted_labels == y_shap.to_numpy()

print(
    f"Correct predictions in SHAP sample: "
    f"{correct.sum():,}/{len(correct):,}"
)

print()


# ============================================================
# 5. CREATE SHAP EXPLAINER
# ============================================================

print("[5/9] Initializing SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

print("TreeExplainer initialized.")
print()


# ============================================================
# 6. COMPUTE SHAP VALUES
# ============================================================

print("[6/9] Computing SHAP values...")
print("This may take some time for 5,000 samples × 14 classes.")

shap_result = explainer(X_shap)

# SHAP >= 0.52 can represent multiclass output as:
#     samples × features × classes
#
# Older versions can return:
#     list[class] -> samples × features

raw_values = shap_result.values

if isinstance(raw_values, list):

    # list of [samples, features]
    shap_values = np.stack(raw_values, axis=-1)

elif isinstance(raw_values, np.ndarray):

    if raw_values.ndim == 3:
        shap_values = raw_values

    elif raw_values.ndim == 2:
        # Binary/single-output fallback
        shap_values = raw_values[:, :, np.newaxis]

    else:
        raise ValueError(
            f"Unexpected SHAP array shape: {raw_values.shape}"
        )

else:
    raise ValueError(
        f"Unsupported SHAP result type: {type(raw_values)}"
    )


print(f"SHAP array shape: {shap_values.shape}")

if shap_values.shape[0] != len(X_shap):
    raise ValueError("SHAP sample dimension mismatch.")

if shap_values.shape[1] != len(feature_names):
    raise ValueError("SHAP feature dimension mismatch.")

print()


# ============================================================
# 7. GLOBAL FEATURE IMPORTANCE
# ============================================================

print("[7/9] Computing global SHAP feature importance...")

# Mean absolute SHAP value across:
#   samples AND output classes
global_mean_abs = np.mean(
    np.abs(shap_values),
    axis=(0, 2)
)

global_rank = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": global_mean_abs
}).sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)

global_rank.insert(
    0,
    "rank",
    np.arange(1, len(global_rank) + 1)
)

global_rank.to_csv(
    OUTPUT_DIR / "step41_global_feature_importance.csv",
    index=False
)


# ============================================================
# 8. CLASS-SPECIFIC SHAP IMPORTANCE
# ============================================================

print("[8/9] Computing class-specific SHAP importance...")

class_rows = []

for class_index, class_name in enumerate(model_classes):

    class_importance = np.mean(
        np.abs(shap_values[:, :, class_index]),
        axis=0
    )

    order = np.argsort(
        class_importance
    )[::-1]

    for rank, feature_index in enumerate(order, start=1):

        class_rows.append({
            "class": class_name,
            "rank": rank,
            "feature": feature_names[feature_index],
            "mean_abs_shap": float(
                class_importance[feature_index]
            )
        })

class_rank = pd.DataFrame(class_rows)

class_rank.to_csv(
    OUTPUT_DIR / "step41_classwise_feature_importance.csv",
    index=False
)


# ============================================================
# 9. TOP FEATURES BY CLASS MATRIX
# ============================================================

top_n = 10

top_class_rows = []

for class_name in model_classes:

    subset = class_rank[
        class_rank["class"] == class_name
    ].head(top_n)

    row = {
        "class": class_name
    }

    for _, item in subset.iterrows():
        row[f"rank_{int(item['rank'])}"] = item["feature"]

    top_class_rows.append(row)

top_class_df = pd.DataFrame(top_class_rows)

top_class_df.to_csv(
    OUTPUT_DIR / "step41_top_features_by_class.csv",
    index=False
)


# ============================================================
# GLOBAL SHAP BAR PLOT
# ============================================================

print("Creating global SHAP importance plot...")

plt.figure(figsize=(10, 8))

global_plot = global_rank.head(20).sort_values(
    "mean_abs_shap",
    ascending=True
)

plt.barh(
    global_plot["feature"],
    global_plot["mean_abs_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title(
    "Global Feature Importance — Frozen Random Forest"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "step41_global_shap_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP SUMMARY / BEESWARM
# ============================================================

print("Creating SHAP summary plot...")

# Aggregate across classes for a model-wide explanation.
#
# For multiclass models, each feature can have different
# contributions to different attack classes. We therefore
# aggregate absolute contributions across classes for
# global ranking while preserving class-specific results.

mean_abs_per_sample = np.mean(
    np.abs(shap_values),
    axis=2
)

# Signed aggregate is not mathematically equivalent to
# multiclass probability contribution, so we use the
# class-averaged signed SHAP values only as a visualization
# aid, while the official ranking remains mean absolute SHAP.

signed_mean = np.mean(
    shap_values,
    axis=2
)

signed_explanation = shap.Explanation(
    values=signed_mean,
    base_values=np.mean(
        shap_result.base_values,
        axis=1
    ) if np.asarray(shap_result.base_values).ndim > 1
    else shap_result.base_values,
    data=X_shap.to_numpy(),
    feature_names=feature_names
)

plt.figure(figsize=(10, 8))

shap.summary_plot(
    signed_explanation,
    X_shap,
    show=False,
    max_display=20
)

plt.title(
    "SHAP Summary — Frozen Random Forest "
    "(Class-Averaged Explanation)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "step41_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TOP FEATURES TABLE
# ============================================================

top_global = global_rank.head(20).copy()

top_global.to_csv(
    OUTPUT_DIR / "step41_top_20_features.csv",
    index=False
)


# ============================================================
# PREDICTION + EXPLANATION METADATA
# ============================================================

prediction_metadata = pd.DataFrame({
    "row_index": sample_indices,
    "true_label": y_shap.to_numpy(),
    "predicted_label": predicted_labels,
    "prediction_confidence": predicted_confidence,
    "correct_prediction": correct
})

prediction_metadata.to_csv(
    OUTPUT_DIR / "step41_shap_sample_predictions.csv",
    index=False
)


# ============================================================
# LOCAL EXPLANATIONS
# ============================================================

print("Generating local explanations...")

# Select high-confidence correct examples from different classes
# where possible. This makes the examples useful for explaining
# why the frozen model reaches a particular decision.

local_candidates = pd.DataFrame({
    "sample_position": np.arange(len(X_shap)),
    "true_label": y_shap.to_numpy(),
    "predicted_label": predicted_labels,
    "confidence": predicted_confidence,
    "correct": correct
})

local_candidates = local_candidates[
    local_candidates["correct"]
].sort_values(
    "confidence",
    ascending=False
)

selected_local = []

used_classes = set()

# First pass: one high-confidence example per class
for _, row in local_candidates.iterrows():

    cls = row["predicted_label"]

    if cls not in used_classes:

        selected_local.append(row)

        used_classes.add(cls)

    if len(selected_local) >= min(
        LOCAL_EXAMPLE_COUNT,
        len(model_classes)
    ):
        break


# Second pass: fill remaining slots
if len(selected_local) < LOCAL_EXAMPLE_COUNT:

    selected_positions = {
        int(x["sample_position"])
        for x in selected_local
    }

    for _, row in local_candidates.iterrows():

        pos = int(row["sample_position"])

        if pos not in selected_positions:
            selected_local.append(row)

        if len(selected_local) >= LOCAL_EXAMPLE_COUNT:
            break


local_rows = []

for row in selected_local:

    pos = int(row["sample_position"])
    pred_class = predicted_labels[pos]

    class_index = model_classes.index(
        str(pred_class)
    )

    values = shap_values[pos, :, class_index]

    order = np.argsort(
        np.abs(values)
    )[::-1]

    for rank, feature_index in enumerate(
        order[:15],
        start=1
    ):

        local_rows.append({
            "sample_position": pos,
            "true_label": y_shap.iloc[pos],
            "predicted_label": pred_class,
            "prediction_confidence": float(
                predicted_confidence[pos]
            ),
            "rank": rank,
            "feature": feature_names[feature_index],
            "shap_value": float(values[feature_index]),
            "feature_value": float(
                X_shap.iloc[pos, feature_index]
            )
            if pd.notna(
                X_shap.iloc[pos, feature_index]
            )
            else None
        })

local_df = pd.DataFrame(local_rows)

local_df.to_csv(
    OUTPUT_DIR / "step41_local_explanations.csv",
    index=False
)


# ============================================================
# FEATURE IMPORTANCE COMPARISON
# ============================================================

print("Comparing SHAP importance with RF native importance...")

native_importance = pd.DataFrame({
    "feature": feature_names,
    "rf_gini_importance": model.feature_importances_
})

importance_comparison = global_rank.merge(
    native_importance,
    on="feature",
    how="left"
)

importance_comparison[
    "shap_rank"
] = importance_comparison["rank"]

importance_comparison[
    "rf_gini_rank"
] = (
    importance_comparison["rf_gini_importance"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

importance_comparison[
    "rank_difference_abs"
] = (
    importance_comparison["shap_rank"]
    - importance_comparison["rf_gini_rank"]
).abs()

importance_comparison.to_csv(
    OUTPUT_DIR / "step41_shap_vs_rf_importance.csv",
    index=False
)


# ============================================================
# CLASS COVERAGE
# ============================================================

class_distribution = (
    y_shap.value_counts()
    .rename_axis("class")
    .reset_index(name="sample_count")
)

class_distribution["sample_percentage"] = (
    class_distribution["sample_count"]
    / len(y_shap)
    * 100
)

class_distribution.to_csv(
    OUTPUT_DIR / "step41_shap_sample_class_distribution.csv",
    index=False
)


# ============================================================
# RESEARCH SUMMARY
# ============================================================

top_feature = global_rank.iloc[0]["feature"]

top_10_features = global_rank.head(10)["feature"].tolist()

class_top_features = {}

for class_name in model_classes:

    class_top_features[class_name] = (
        class_rank[
            class_rank["class"] == class_name
        ]
        .head(5)["feature"]
        .tolist()
    )


summary = {
    "step": 41,
    "title": "SHAP Explainability Analysis",
    "purpose": (
        "Post-hoc explainability of the frozen Step 21 "
        "Random Forest using CIC-IDS2018 validation data."
    ),
    "research_integrity": {
        "frozen_model_used": True,
        "model_retrained": False,
        "hyperparameter_optimization": False,
        "test_set_used": False,
        "validation_set_used": True,
        "feature_selection_changed": False,
        "data_balancing_performed": False,
        "model_selection_performed": False
    },
    "model": {
        "type": type(model).__name__,
        "features": int(model.n_features_in_),
        "classes": model_classes,
        "n_estimators": int(model.n_estimators)
        if hasattr(model, "n_estimators")
        else None
    },
    "shap": {
        "version": shap.__version__,
        "explainer": "TreeExplainer",
        "sample_size": int(len(X_shap)),
        "random_state": RANDOM_STATE,
        "array_shape": list(shap_values.shape)
    },
    "global_explainability": {
        "top_feature": top_feature,
        "top_10_features": top_10_features
    },
    "classwise_top_features": class_top_features,
    "outputs": {
        "global_feature_importance":
            "step41_global_feature_importance.csv",
        "classwise_feature_importance":
            "step41_classwise_feature_importance.csv",
        "top_features_by_class":
            "step41_top_features_by_class.csv",
        "top_20_features":
            "step41_top_20_features.csv",
        "global_shap_plot":
            "step41_global_shap_importance.png",
        "shap_summary_plot":
            "step41_shap_summary.png",
        "local_explanations":
            "step41_local_explanations.csv",
        "shap_vs_rf_importance":
            "step41_shap_vs_rf_importance.csv",
        "sample_predictions":
            "step41_shap_sample_predictions.csv"
    },
    "interpretation_note": (
        "SHAP values provide post-hoc feature-attribution evidence "
        "for the frozen Random Forest. Global importance is computed "
        "as mean absolute SHAP magnitude across validation samples "
        "and output classes. Class-specific rankings preserve "
        "attack-class differences. SHAP explanations do not imply "
        "causality."
    )
}

save_json(
    OUTPUT_DIR / "step41_shap_explainability_summary.json",
    summary
)


# ============================================================
# PROTOCOL
# ============================================================

protocol = f"""
STEP 41 — SHAP EXPLAINABILITY PROTOCOL
======================================

Objective
---------
Provide rigorous post-hoc explainability for the final frozen
Random Forest selected during Step 21.

Frozen model
------------
{MODEL_PATH}

Validation dataset
------------------
{VALIDATION_PATH}

Model characteristics
---------------------
Model type: {type(model).__name__}
Number of features: {model.n_features_in_}
Number of classes: {len(model_classes)}
Number of trees: {getattr(model, "n_estimators", "N/A")}

SHAP configuration
------------------
Explainer: TreeExplainer
SHAP version: {shap.__version__}
Validation SHAP sample: {len(X_shap)}
Random state: {RANDOM_STATE}

Research integrity
------------------
1. The Step 21 Random Forest was loaded without retraining.
2. The CIC-IDS2018 validation set was used for explanation.
3. The untouched CIC-IDS2018 test set was not accessed.
4. No hyperparameter optimization was performed.
5. No feature selection was changed.
6. No class balancing was performed.
7. No model selection decision was based on SHAP.
8. SHAP was used only for interpretation.

Global explanation
------------------
Global feature importance is based on mean absolute SHAP
magnitude across validation samples and all 14 output classes.

Class-specific explanation
--------------------------
For every attack class, feature importance was calculated
independently from the SHAP values associated with that class.

Local explanation
-----------------
High-confidence correctly classified validation observations
were selected to demonstrate sample-level model reasoning.

Important limitation
--------------------
SHAP explains the behavior of the trained Random Forest; it does
not establish causal relationships between network features and
attacks. Feature attribution should therefore be interpreted as
model behavior evidence rather than causal security evidence.

Top global feature
------------------
{top_feature}

Top 10 global features
----------------------
{chr(10).join(f"{i+1}. {f}" for i, f in enumerate(top_10_features))}
"""

with open(
    OUTPUT_DIR / "step41_shap_explainability_protocol.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(protocol.strip())


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 72)
print("STEP 41 COMPLETED SUCCESSFULLY")
print("=" * 72)

print(f"SHAP sample:       {len(X_shap):,}")
print(f"Classes explained: {len(model_classes)}")
print(f"Features explained:{len(feature_names)}")
print()
print("Top 10 global SHAP features:")

for i, feature in enumerate(top_10_features, start=1):
    value = global_rank.iloc[i - 1]["mean_abs_shap"]
    print(f"  {i:2d}. {feature}: {value:.6f}")

print()
print(f"Outputs saved to:")
print(f"  {OUTPUT_DIR}")
print()
print(
    "STATUS: STEP 41 EXPLAINABILITY ANALYSIS COMPLETE"
)