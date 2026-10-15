from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ============================================================
# STEP 26 - LSTM MODEL TRAINING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LSTM_DIR = PROJECT_ROOT / "experiments" / "step25_lstm"
REPORT_DIR = PROJECT_ROOT / "experiments" / "step26_reports"
MODEL_DIR = PROJECT_ROOT / "experiments" / "step26_lstm"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 78)
print("STEP 26 - LSTM MODEL TRAINING")
print("=" * 78)

print(f"Project root : {PROJECT_ROOT}")
print(f"LSTM data dir: {LSTM_DIR}")
print(f"Report dir   : {REPORT_DIR}")
print(f"Model dir    : {MODEL_DIR}")


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 10
FEATURE_COUNT = 30
NUM_CLASSES = 14

RANDOM_STATE = 42

BATCH_SIZE = 512
EPOCHS = 30

LEARNING_RATE = 0.001

LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.30

PATIENCE = 5


# ============================================================
# INPUT FILES
# ============================================================

X_TRAIN_FILE = LSTM_DIR / "step25_X_train_sequences.npy"
Y_TRAIN_FILE = LSTM_DIR / "step25_y_train_sequences.npy"

X_VAL_FILE = LSTM_DIR / "step25_X_validation_sequences.npy"
Y_VAL_FILE = LSTM_DIR / "step25_y_validation_sequences.npy"

SCALER_FILE = LSTM_DIR / "step25_lstm_scaler.joblib"
ENCODER_FILE = LSTM_DIR / "step25_lstm_label_encoder.joblib"
FEATURE_FILE = LSTM_DIR / "step25_lstm_features.json"


# ============================================================
# CHECK INPUTS
# ============================================================

print("\n" + "=" * 78)
print("CHECKING INPUT FILES")
print("=" * 78)

required_files = [
    X_TRAIN_FILE,
    Y_TRAIN_FILE,
    X_VAL_FILE,
    Y_VAL_FILE,
    SCALER_FILE,
    ENCODER_FILE,
    FEATURE_FILE,
]

for file_path in required_files:
    print(f"Checking: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required Step 26 input not found:\n{file_path}"
        )

print("\nAll required input files found.")


# ============================================================
# LOAD SEQUENCES
# ============================================================

print("\n" + "=" * 78)
print("LOADING LSTM SEQUENCES")
print("=" * 78)

X_train = np.load(X_TRAIN_FILE)
y_train = np.load(Y_TRAIN_FILE)

X_validation = np.load(X_VAL_FILE)
y_validation = np.load(Y_VAL_FILE)

print(f"Training sequence shape   : {X_train.shape}")
print(f"Training label shape      : {y_train.shape}")
print(f"Validation sequence shape : {X_validation.shape}")
print(f"Validation label shape    : {y_validation.shape}")


# ============================================================
# VALIDATE SHAPES
# ============================================================

print("\n" + "=" * 78)
print("VALIDATING SEQUENCE SHAPES")
print("=" * 78)

if X_train.ndim != 3:
    raise ValueError("Training sequences must be 3-dimensional.")

if X_validation.ndim != 3:
    raise ValueError("Validation sequences must be 3-dimensional.")

if X_train.shape[1] != SEQUENCE_LENGTH:
    raise ValueError(
        f"Unexpected training sequence length: {X_train.shape[1]}"
    )

if X_validation.shape[1] != SEQUENCE_LENGTH:
    raise ValueError(
        f"Unexpected validation sequence length: {X_validation.shape[1]}"
    )

if X_train.shape[2] != FEATURE_COUNT:
    raise ValueError(
        f"Unexpected training feature count: {X_train.shape[2]}"
    )

if X_validation.shape[2] != FEATURE_COUNT:
    raise ValueError(
        f"Unexpected validation feature count: {X_validation.shape[2]}"
    )

if len(X_train) != len(y_train):
    raise ValueError("Training X/y sample counts do not match.")

if len(X_validation) != len(y_validation):
    raise ValueError("Validation X/y sample counts do not match.")

print("Sequence shapes verified.")


# ============================================================
# CHECK DATA QUALITY
# ============================================================

print("\n" + "=" * 78)
print("VALIDATING DATA QUALITY")
print("=" * 78)

print(f"Training NaN values     : {np.isnan(X_train).sum()}")
print(f"Validation NaN values   : {np.isnan(X_validation).sum()}")

print(f"Training infinite values   : {np.isinf(X_train).sum()}")
print(f"Validation infinite values : {np.isinf(X_validation).sum()}")

if np.isnan(X_train).any() or np.isnan(X_validation).any():
    raise ValueError("NaN values detected in sequence data.")

if np.isinf(X_train).any() or np.isinf(X_validation).any():
    raise ValueError("Infinite values detected in sequence data.")


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

print("\n" + "=" * 78)
print("LOADING LABEL ENCODER")
print("=" * 78)

label_encoder = joblib.load(ENCODER_FILE)

class_names = list(label_encoder.classes_)

print(f"Number of classes: {len(class_names)}")

for class_id, class_name in enumerate(class_names):
    print(f"{class_id:02d} -> {class_name}")

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, found {len(class_names)}."
    )


# ============================================================
# LABEL VALIDATION
# ============================================================

if y_train.min() < 0 or y_train.max() >= NUM_CLASSES:
    raise ValueError("Invalid training class IDs detected.")

if y_validation.min() < 0 or y_validation.max() >= NUM_CLASSES:
    raise ValueError("Invalid validation class IDs detected.")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 78)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 78)

train_counts = np.bincount(
    y_train.astype(int),
    minlength=NUM_CLASSES
)

validation_counts = np.bincount(
    y_validation.astype(int),
    minlength=NUM_CLASSES
)

distribution_rows = []

for class_id in range(NUM_CLASSES):

    distribution_rows.append(
        {
            "Class_ID": class_id,
            "Label": class_names[class_id],
            "Training_Sequences": int(train_counts[class_id]),
            "Validation_Sequences": int(validation_counts[class_id]),
        }
    )

distribution_df = pd.DataFrame(distribution_rows)

print(distribution_df.to_string(index=False))


# ============================================================
# BUILD LSTM MODEL
# ============================================================

print("\n" + "=" * 78)
print("BUILDING LSTM MODEL")
print("=" * 78)

model = Sequential(
    [
        LSTM(
            LSTM_UNITS_1,
            return_sequences=True,
            input_shape=(SEQUENCE_LENGTH, FEATURE_COUNT),
        ),

        Dropout(DROPOUT_RATE),

        LSTM(
            LSTM_UNITS_2,
            return_sequences=False,
        ),

        Dropout(DROPOUT_RATE),

        Dense(
            NUM_CLASSES,
            activation="softmax",
        ),
    ]
)


# ============================================================
# COMPILE MODEL
# ============================================================

optimizer = Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(model.summary())


# ============================================================
# CALLBACKS
# ============================================================

BEST_MODEL_FILE = MODEL_DIR / "step26_best_lstm.keras"

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),

    ModelCheckpoint(
        filepath=BEST_MODEL_FILE,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 78)
print("STARTING LSTM TRAINING")
print("=" * 78)

print(f"Sequence length : {SEQUENCE_LENGTH}")
print(f"Features/step   : {FEATURE_COUNT}")
print(f"Classes         : {NUM_CLASSES}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Maximum epochs  : {EPOCHS}")
print(f"Learning rate   : {LEARNING_RATE}")
print(f"LSTM units      : {LSTM_UNITS_1} -> {LSTM_UNITS_2}")
print(f"Dropout         : {DROPOUT_RATE}")
print(f"Early stopping  : patience={PATIENCE}")

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_validation,
        y_validation,
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
    shuffle=True,
)

training_time = time.time() - start_time


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 78)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 78)

validation_probabilities = model.predict(
    X_validation,
    batch_size=BATCH_SIZE,
    verbose=1,
)

validation_predictions = np.argmax(
    validation_probabilities,
    axis=1,
)


# ============================================================
# OVERALL VALIDATION METRICS
# ============================================================

print("\n" + "=" * 78)
print("VALIDATION EVALUATION")
print("=" * 78)

accuracy = accuracy_score(
    y_validation,
    validation_predictions,
)

balanced_accuracy = balanced_accuracy_score(
    y_validation,
    validation_predictions,
)

macro_precision = precision_score(
    y_validation,
    validation_predictions,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    y_validation,
    validation_predictions,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_validation,
    validation_predictions,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_validation,
    validation_predictions,
    average="weighted",
    zero_division=0,
)

print(f"Accuracy           : {accuracy:.6f}")
print(f"Balanced Accuracy  : {balanced_accuracy:.6f}")
print(f"Macro Precision    : {macro_precision:.6f}")
print(f"Macro Recall       : {macro_recall:.6f}")
print(f"Macro F1           : {macro_f1:.6f}")
print(f"Weighted F1        : {weighted_f1:.6f}")


# ============================================================
# TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history.history)

history_df.insert(
    0,
    "Epoch",
    range(1, len(history_df) + 1),
)

history_df.to_csv(
    REPORT_DIR / "step26_lstm_training_history.csv",
    index=False,
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

print("\n" + "=" * 78)
print("PER-CLASS VALIDATION RESULTS")
print("=" * 78)

per_class_precision = precision_score(
    y_validation,
    validation_predictions,
    average=None,
    labels=list(range(NUM_CLASSES)),
    zero_division=0,
)

per_class_recall = recall_score(
    y_validation,
    validation_predictions,
    average=None,
    labels=list(range(NUM_CLASSES)),
    zero_division=0,
)

per_class_f1 = f1_score(
    y_validation,
    validation_predictions,
    average=None,
    labels=list(range(NUM_CLASSES)),
    zero_division=0,
)

per_class_rows = []

for class_id in range(NUM_CLASSES):

    per_class_rows.append(
        {
            "Class_ID": class_id,
            "Label": class_names[class_id],
            "Precision": float(per_class_precision[class_id]),
            "Recall": float(per_class_recall[class_id]),
            "F1": float(per_class_f1[class_id]),
            "Support": int(validation_counts[class_id]),
        }
    )

per_class_df = pd.DataFrame(per_class_rows)

print(
    per_class_df.to_string(
        index=False
    )
)

per_class_df.to_csv(
    REPORT_DIR / "step26_lstm_validation_per_class_metrics.csv",
    index=False,
)


# ============================================================
# SAVE OVERALL METRICS
# ============================================================

metrics_df = pd.DataFrame(
    [
        {
            "Accuracy": accuracy,
            "Balanced_Accuracy": balanced_accuracy,
            "Macro_Precision": macro_precision,
            "Macro_Recall": macro_recall,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1,
            "Training_Time_Seconds": training_time,
            "Epochs_Completed": len(history_df),
        }
    ]
)

metrics_df.to_csv(
    REPORT_DIR / "step26_lstm_validation_metrics.csv",
    index=False,
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

configuration = {
    "experiment": "Step26_LSTM_Model_Training",
    "dataset": "CIC-IDS2018",
    "training_representation": "B1_undersampled",
    "sequence_length": SEQUENCE_LENGTH,
    "feature_count": FEATURE_COUNT,
    "class_count": NUM_CLASSES,
    "lstm_units_1": LSTM_UNITS_1,
    "lstm_units_2": LSTM_UNITS_2,
    "dropout_rate": DROPOUT_RATE,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "maximum_epochs": EPOCHS,
    "early_stopping_patience": PATIENCE,
    "random_state": RANDOM_STATE,
    "optimizer": "Adam",
    "loss": "sparse_categorical_crossentropy",
    "primary_metric": "Macro_F1",
    "validation_used_for_training_monitoring": True,
    "test_set_used": False,
    "training_time_seconds": training_time,
    "epochs_completed": len(history_df),
}

with open(
    REPORT_DIR / "step26_lstm_training_configuration.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        configuration,
        f,
        indent=4,
    )


# ============================================================
# SAVE PROTOCOL
# ============================================================

protocol = f"""
STEP 26 - LSTM MODEL TRAINING

Dataset:
CIC-IDS2018

Training representation:
B1 undersampled

Sequence length:
{SEQUENCE_LENGTH}

Features per timestep:
{FEATURE_COUNT}

Number of classes:
{NUM_CLASSES}

Architecture:
LSTM({LSTM_UNITS_1})
Dropout({DROPOUT_RATE})
LSTM({LSTM_UNITS_2})
Dropout({DROPOUT_RATE})
Dense({NUM_CLASSES}, softmax)

Optimizer:
Adam

Learning rate:
{LEARNING_RATE}

Loss:
Sparse Categorical Crossentropy

Batch size:
{BATCH_SIZE}

Maximum epochs:
{EPOCHS}

Early stopping patience:
{PATIENCE}

Primary evaluation metric:
Macro F1

Validation Metrics:

Accuracy:
{accuracy:.6f}

Balanced Accuracy:
{balanced_accuracy:.6f}

Macro Precision:
{macro_precision:.6f}

Macro Recall:
{macro_recall:.6f}

Macro F1:
{macro_f1:.6f}

Weighted F1:
{weighted_f1:.6f}

Training time:
{training_time:.2f} seconds

Epochs completed:
{len(history_df)}

Important:
The scaler was fitted during Step 25 using training data only.

The validation sequences were generated using the training-fitted scaler.

No test dataset was used in Step 26.

Early stopping and model checkpointing were used.

The best validation-loss model was retained.
"""

with open(
    REPORT_DIR / "step26_lstm_training_protocol.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(protocol)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_FILE = MODEL_DIR / "step26_final_lstm.keras"

model.save(FINAL_MODEL_FILE)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 78)
print("STEP 26 COMPLETED")
print("=" * 78)

print("Generated reports:")

print("  - step26_lstm_training_history.csv")
print("  - step26_lstm_validation_metrics.csv")
print("  - step26_lstm_validation_per_class_metrics.csv")
print("  - step26_lstm_training_configuration.json")
print("  - step26_lstm_training_protocol.txt")

print("\nGenerated models:")

print("  - step26_best_lstm.keras")
print("  - step26_final_lstm.keras")

print("\nPrimary validation result:")

print(f"  Macro F1 : {macro_f1:.6f}")

print("\nNEXT:")
print("STEP 27 - LSTM VALIDATION & ERROR ANALYSIS")