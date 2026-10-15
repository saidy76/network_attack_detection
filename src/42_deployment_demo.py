# src/42_deployment_demo.py

"""
STEP 42 — Deployment-Oriented IDS Demonstration
================================================

Purpose:
    Demonstrate that the frozen Step-21 Random Forest can be
    operationalized as a deployable multiclass IDS inference pipeline.

Pipeline:
    Input flow features
        ↓
    Schema validation
        ↓
    Frozen Random Forest
        ↓
    Attack classification
        ↓
    Confidence estimation
        ↓
    Security alert
        ↓
    JSON/CSV deployment logs

Research integrity:
    - Uses frozen Step-21 Random Forest.
    - No retraining.
    - No hyperparameter optimization.
    - No test-set model selection.
    - Uses validation samples only for demonstration.
    - Does not claim packet capture itself is implemented.
    - Demonstrates model-serving/inference readiness.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import time

import joblib
import numpy as np
import pandas as pd


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step42_deployment_demo"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42

# Number of real validation flows used for deployment simulation.
DEMO_SAMPLE_SIZE = 1000

# Security alert threshold.
#
# The classifier still returns its predicted class below this
# threshold, but confidence is used to distinguish high-confidence
# alerts from lower-confidence predictions.
HIGH_CONFIDENCE_THRESHOLD = 0.80

MEDIUM_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# FIXED 30-FEATURE SCHEMA
# ============================================================

FEATURES = [
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            default=lambda x: (
                x.item()
                if isinstance(x, np.generic)
                else x
            ),
        )


def confidence_level(confidence):
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH"

    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "MEDIUM"

    return "LOW"


def alert_type(predicted_label):
    if predicted_label == "BENIGN":
        return "NORMAL"

    return "ATTACK_DETECTED"


def top_alternative_predictions(probabilities, classes, top_n=3):
    order = np.argsort(probabilities)[::-1][:top_n]

    result = []

    for index in order:
        result.append(
            {
                "class": str(classes[index]),
                "probability": float(probabilities[index]),
            }
        )

    return result


# ============================================================
# START
# ============================================================

print("=" * 72)
print("STEP 42 — DEPLOYMENT-ORIENTED IDS DEMONSTRATION")
print("=" * 72)

print(f"Project root:     {PROJECT_ROOT}")
print(f"Frozen model:     {MODEL_PATH}")
print(f"Validation data:  {VALIDATION_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print()


# ============================================================
# 1. LOAD FROZEN MODEL
# ============================================================

print("[1/8] Loading frozen Step-21 Random Forest...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Frozen model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

if model.n_features_in_ != 30:
    raise ValueError(
        f"Expected 30 model features, "
        f"found {model.n_features_in_}"
    )

model_classes = [
    str(x)
    for x in model.classes_
]

print(
    f"Model: {type(model).__name__}"
)

print(
    f"Trees: {getattr(model, 'n_estimators', 'N/A')}"
)

print(
    f"Features: {model.n_features_in_}"
)

print(
    f"Classes: {len(model_classes)}"
)

print()


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

print("[2/8] Loading deployment demonstration data...")

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(
        f"Validation file not found: {VALIDATION_PATH}"
    )

df = pd.read_csv(
    VALIDATION_PATH
)

if "Label" not in df.columns:
    raise ValueError(
        "Validation dataset must contain Label."
    )

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "Missing required deployment features:\n"
        + "\n".join(missing_features)
    )

X = df[FEATURES].copy()
y = df["Label"].astype(str).copy()

print(
    f"Available validation flows: {len(X):,}"
)

print(
    f"Deployment feature count: {len(FEATURES)}"
)

print()


# ============================================================
# 3. CREATE REPRODUCIBLE DEMONSTRATION SAMPLE
# ============================================================

print("[3/8] Preparing deployment simulation...")

rng = np.random.default_rng(
    RANDOM_STATE
)

sample_size = min(
    DEMO_SAMPLE_SIZE,
    len(X)
)

indices = rng.choice(
    len(X),
    size=sample_size,
    replace=False
)

indices = np.sort(indices)

X_demo = X.iloc[indices].copy()
y_demo = y.iloc[indices].copy()

print(
    f"Simulated incoming flows: {len(X_demo):,}"
)

print()


# ============================================================
# 4. SCHEMA VALIDATION
# ============================================================

print("[4/8] Validating deployment schema...")

schema_valid = (
    list(X_demo.columns) == FEATURES
)

if not schema_valid:
    raise ValueError(
        "Deployment schema validation failed."
    )

numeric_check = all(
    pd.api.types.is_numeric_dtype(
        X_demo[column]
    )
    for column in FEATURES
)

if not numeric_check:
    raise ValueError(
        "One or more deployment features are non-numeric."
    )

missing_values = int(
    X_demo.isna().sum().sum()
)

if missing_values > 0:
    raise ValueError(
        f"Deployment input contains "
        f"{missing_values} missing values."
    )

print("Schema validation: PASS")
print("Numeric validation: PASS")
print("Missing-value validation: PASS")
print()


# ============================================================
# 5. REALISTIC INFERENCE LOOP
# ============================================================

print("[5/8] Running deployment-style inference...")

records = []

latencies_ms = []

for position in range(len(X_demo)):

    row = X_demo.iloc[[position]]

    start = time.perf_counter()

    probabilities = model.predict_proba(
        row
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_label = str(
        model_classes[predicted_index]
    )

    confidence = float(
        probabilities[predicted_index]
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    latencies_ms.append(
        elapsed_ms
    )

    true_label = str(
        y_demo.iloc[position]
    )

    records.append(
        {
            "event_id": position + 1,
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "prediction_confidence": confidence,
            "confidence_level": confidence_level(
                confidence
            ),
            "alert_type": alert_type(
                predicted_label
            ),
            "correct_prediction": (
                predicted_label == true_label
            ),
            "top_alternatives": json.dumps(
                top_alternative_predictions(
                    probabilities,
                    model_classes,
                    top_n=3
                )
            ),
        }
    )


results = pd.DataFrame(records)

print(
    f"Inference completed for "
    f"{len(results):,} flows."
)

print()


# ============================================================
# 6. PERFORMANCE ANALYSIS
# ============================================================

print("[6/8] Calculating deployment performance...")

latencies = np.asarray(
    latencies_ms,
    dtype=float
)

accuracy = float(
    results["correct_prediction"].mean()
)

attack_detected = (
    results["predicted_label"] != "BENIGN"
)

normal_predicted = (
    results["predicted_label"] == "BENIGN"
)

high_confidence = (
    results["prediction_confidence"]
    >= HIGH_CONFIDENCE_THRESHOLD
)

medium_or_higher = (
    results["prediction_confidence"]
    >= MEDIUM_CONFIDENCE_THRESHOLD
)

metrics = {
    "sample_size": int(len(results)),
    "correct_predictions": int(
        results["correct_prediction"].sum()
    ),
    "accuracy_on_demo_sample": accuracy,
    "attack_alert_count": int(
        attack_detected.sum()
    ),
    "benign_prediction_count": int(
        normal_predicted.sum()
    ),
    "high_confidence_predictions": int(
        high_confidence.sum()
    ),
    "medium_or_higher_confidence_predictions": int(
        medium_or_higher.sum()
    ),
    "latency_ms": {
        "mean": float(np.mean(latencies)),
        "median": float(np.median(latencies)),
        "p95": float(
            np.percentile(latencies, 95)
        ),
        "p99": float(
            np.percentile(latencies, 99)
        ),
        "max": float(np.max(latencies)),
        "min": float(np.min(latencies)),
    },
}


print(
    f"Demo accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Mean inference latency: "
    f"{metrics['latency_ms']['mean']:.4f} ms"
)

print(
    f"Median inference latency: "
    f"{metrics['latency_ms']['median']:.4f} ms"
)

print(
    f"P95 inference latency: "
    f"{metrics['latency_ms']['p95']:.4f} ms"
)

print()


# ============================================================
# 7. ATTACK / CLASS DISTRIBUTION
# ============================================================

print("[7/8] Generating deployment reports...")

predicted_distribution = (
    results["predicted_label"]
    .value_counts()
    .rename_axis("predicted_class")
    .reset_index(
        name="count"
    )
)

predicted_distribution[
    "percentage"
] = (
    predicted_distribution["count"]
    / len(results)
    * 100
)

predicted_distribution.to_csv(
    OUTPUT_DIR
    / "step42_predicted_class_distribution.csv",
    index=False
)


# Per-class inference counts
class_rows = []

for class_name in model_classes:

    class_mask = (
        results["predicted_label"]
        == class_name
    )

    class_true = (
        results["true_label"]
        == class_name
    )

    class_rows.append(
        {
            "class": class_name,
            "predicted_count": int(
                class_mask.sum()
            ),
            "true_count": int(
                class_true.sum()
            ),
            "correct_count": int(
                (
                    class_mask
                    & class_true
                ).sum()
            ),
        }
    )

class_distribution = pd.DataFrame(
    class_rows
)

class_distribution.to_csv(
    OUTPUT_DIR
    / "step42_classwise_inference_summary.csv",
    index=False
)


# Save complete inference log
results.to_csv(
    OUTPUT_DIR
    / "step42_deployment_inference_log.csv",
    index=False
)


# Save latency measurements
latency_df = pd.DataFrame(
    {
        "event_id": np.arange(
            1,
            len(latencies) + 1
        ),
        "latency_ms": latencies,
    }
)

latency_df.to_csv(
    OUTPUT_DIR
    / "step42_inference_latency.csv",
    index=False
)


# ============================================================
# 8. FINAL RESEARCH/DEPLOYMENT SUMMARY
# ============================================================

print("[8/8] Saving deployment documentation...")

summary = {
    "step": 42,
    "title": (
        "Deployment-Oriented IDS Demonstration"
    ),
    "purpose": (
        "Demonstrate operational inference using "
        "the frozen Step-21 Random Forest."
    ),
    "model": {
        "type": type(model).__name__,
        "model_path": str(MODEL_PATH),
        "n_features": int(
            model.n_features_in_
        ),
        "n_classes": int(
            len(model_classes)
        ),
        "n_estimators": int(
            getattr(
                model,
                "n_estimators",
                0
            )
        ),
    },
    "deployment_input": {
        "dataset": "CIC-IDS2018 validation set",
        "sample_size": int(
            len(X_demo)
        ),
        "feature_count": len(FEATURES),
        "schema_validation": True,
        "numeric_validation": True,
        "missing_value_validation": True,
    },
    "inference": {
        "accuracy_on_demo_sample": accuracy,
        "attack_alert_count": int(
            attack_detected.sum()
        ),
        "high_confidence_predictions": int(
            high_confidence.sum()
        ),
    },
    "latency_ms": metrics["latency_ms"],
    "thresholds": {
        "high_confidence": HIGH_CONFIDENCE_THRESHOLD,
        "medium_confidence": MEDIUM_CONFIDENCE_THRESHOLD,
    },
    "research_integrity": {
        "frozen_model": True,
        "retraining": False,
        "hyperparameter_optimization": False,
        "model_selection": False,
        "test_set_used": False,
        "validation_data_used": True,
        "feature_schema_changed": False,
    },
    "deployment_capability": {
        "batch_inference": True,
        "single_flow_inference": True,
        "multiclass_prediction": True,
        "confidence_estimation": True,
        "attack_alert_generation": True,
        "structured_logging": True,
        "latency_measurement": True,
        "packet_capture": False,
        "network_sensor_integration": False,
        "production_service": False,
    },
    "important_limitation": (
        "This step demonstrates deployable model inference "
        "using pre-extracted CIC-IDS2018 flow features. "
        "It does not claim that live packet capture, flow "
        "generation, or production network integration has "
        "already been implemented."
    ),
}

save_json(
    OUTPUT_DIR
    / "step42_deployment_summary.json",
    summary
)


protocol = f"""
STEP 42 — DEPLOYMENT-ORIENTED IDS DEMONSTRATION
================================================

Objective
---------
Demonstrate operational inference using the frozen Random Forest
selected during Step 21.

Frozen model
------------
{MODEL_PATH}

Input
-----
CIC-IDS2018 validation flows.

Feature schema
--------------
30 frozen CIC-IDS2018 features.

Demonstration sample
--------------------
{len(X_demo)} validation flows.

Inference pipeline
------------------
1. Receive flow feature vector.
2. Validate feature schema.
3. Validate numeric values.
4. Validate missing values.
5. Pass feature vector to frozen Random Forest.
6. Generate multiclass probabilities.
7. Select predicted attack/benign class.
8. Calculate prediction confidence.
9. Generate security alert status.
10. Store structured inference log.

Confidence thresholds
---------------------
High confidence: >= {HIGH_CONFIDENCE_THRESHOLD}
Medium confidence: >= {MEDIUM_CONFIDENCE_THRESHOLD}

Observed inference performance
------------------------------
Demo accuracy: {accuracy:.6f}
Mean latency: {metrics['latency_ms']['mean']:.6f} ms
Median latency: {metrics['latency_ms']['median']:.6f} ms
P95 latency: {metrics['latency_ms']['p95']:.6f} ms
P99 latency: {metrics['latency_ms']['p99']:.6f} ms
Maximum latency: {metrics['latency_ms']['max']:.6f} ms

Research integrity
------------------
- Frozen Step-21 model used.
- No retraining.
- No hyperparameter optimization.
- No model selection.
- No test-set access.
- No feature schema modification.
- Validation data used only for deployment simulation.

Deployment interpretation
-------------------------
The result demonstrates that the selected Random Forest can
operate as a low-latency multiclass inference component when
provided with pre-extracted network-flow features.

The demonstration does NOT yet constitute a complete live IDS
sensor because packet capture and online flow extraction are not
implemented in this step.

Next deployment-oriented work should integrate:
- live/network flow acquisition,
- online feature extraction,
- prediction service,
- alert generation,
- and monitoring.

This step therefore establishes the ML inference layer required
for a real-world IDS prototype.
"""

with open(
    OUTPUT_DIR
    / "step42_deployment_protocol.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        protocol.strip()
    )


# ============================================================
# SAMPLE ALERTS
# ============================================================

alerts = results[
    results["predicted_label"] != "BENIGN"
].copy()

alerts.head(100).to_csv(
    OUTPUT_DIR
    / "step42_sample_attack_alerts.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 72)
print("STEP 42 COMPLETED SUCCESSFULLY")
print("=" * 72)

print(
    f"Flows simulated:       {len(results):,}"
)

print(
    f"Correct predictions:   "
    f"{int(results['correct_prediction'].sum()):,}"
)

print(
    f"Demo accuracy:         {accuracy:.4f}"
)

print(
    f"Mean latency:          "
    f"{metrics['latency_ms']['mean']:.4f} ms"
)

print(
    f"Median latency:        "
    f"{metrics['latency_ms']['median']:.4f} ms"
)

print(
    f"P95 latency:           "
    f"{metrics['latency_ms']['p95']:.4f} ms"
)

print(
    f"Attack alerts:         "
    f"{int(attack_detected.sum()):,}"
)

print()
print(
    "Outputs saved to:"
)
print(
    f"  {OUTPUT_DIR}"
)

print()
print(
    "STATUS: STEP 42 DEPLOYMENT DEMONSTRATION COMPLETE"
)