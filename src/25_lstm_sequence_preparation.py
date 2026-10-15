from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


# ============================================================
# STEP 25 - LSTM SEQUENCE MODEL PREPARATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "balanced"
    / "CICIDS2018_train_B1_undersampled.csv"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "CICIDS2018_validation_selected.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step25_reports"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step25_lstm"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 78)
print("STEP 25 - LSTM SEQUENCE MODEL PREPARATION")
print("=" * 78)

print(f"Project root : {PROJECT_ROOT}")
print(f"Training file: {TRAIN_FILE}")
print(f"Validation   : {VALIDATION_FILE}")
print(f"Output dir   : {OUTPUT_DIR}")
print(f"Model dir    : {MODEL_DIR}")


# ============================================================
# CHECK INPUTS
# ============================================================

print("\n" + "=" * 78)
print("CHECKING INPUT FILES")
print("=" * 78)

for file_path in [TRAIN_FILE, VALIDATION_FILE]:

    print(f"Checking: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{file_path}"
        )

print("\nAll required input files found.")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 78)
print("LOADING DATA")
print("=" * 78)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print(f"Training shape   : {train_df.shape}")
print(f"Validation shape : {validation_df.shape}")


# ============================================================
# IDENTIFY LABEL COLUMN
# ============================================================

print("\n" + "=" * 78)
print("IDENTIFYING LABEL COLUMN")
print("=" * 78)

possible_label_columns = [
    "Label",
    "label",
    "Attack",
    "attack",
    "Class",
    "class"
]

label_column = None

for column in possible_label_columns:
    if column in train_df.columns:
        label_column = column
        break

if label_column is None:
    raise ValueError(
        "Could not identify the target label column."
    )

if label_column not in validation_df.columns:
    raise ValueError(
        f"Validation dataset does not contain label column: {label_column}"
    )

print(f"Label column: {label_column}")


# ============================================================
# IDENTIFY FEATURES
# ============================================================

print("\n" + "=" * 78)
print("IDENTIFYING FEATURES")
print("=" * 78)

feature_columns = [
    column
    for column in train_df.columns
    if column != label_column
]

if len(feature_columns) != 30:

    raise ValueError(
        f"Expected exactly 30 features, found {len(feature_columns)}."
    )

missing_validation_features = [
    column
    for column in feature_columns
    if column not in validation_df.columns
]

if missing_validation_features:

    raise ValueError(
        "Validation dataset is missing required features:\n"
        + "\n".join(missing_validation_features)
    )

print(f"Feature count: {len(feature_columns)}")

print("\nSelected features:")

for index, feature in enumerate(feature_columns, start=1):
    print(f"{index:02d}. {feature}")


# ============================================================
# VERIFY CLASS ALIGNMENT
# ============================================================

print("\n" + "=" * 78)
print("VERIFYING CLASS ALIGNMENT")
print("=" * 78)

train_classes = sorted(
    train_df[label_column].astype(str).unique()
)

validation_classes = sorted(
    validation_df[label_column].astype(str).unique()
)

print(f"Training classes   : {len(train_classes)}")
print(f"Validation classes : {len(validation_classes)}")

if train_classes != validation_classes:

    raise ValueError(
        "Training and validation classes do not match."
    )

print("\nClass alignment verified.")


# ============================================================
# ENCODE LABELS
# ============================================================

print("\n" + "=" * 78)
print("ENCODING CLASS LABELS")
print("=" * 78)

label_encoder = LabelEncoder()

all_classes = np.array(train_classes)

label_encoder.fit(all_classes)

y_train = label_encoder.transform(
    train_df[label_column].astype(str)
)

y_validation = label_encoder.transform(
    validation_df[label_column].astype(str)
)

print(f"Number of classes: {len(label_encoder.classes_)}")

for class_id, class_name in enumerate(
    label_encoder.classes_
):

    print(f"{class_id:02d} -> {class_name}")


# ============================================================
# EXTRACT FEATURES
# ============================================================

X_train = train_df[feature_columns].copy()
X_validation = validation_df[feature_columns].copy()


# ============================================================
# NUMERIC VALIDATION
# ============================================================

print("\n" + "=" * 78)
print("VALIDATING FEATURE DATA")
print("=" * 78)

for column in feature_columns:

    X_train[column] = pd.to_numeric(
        X_train[column],
        errors="coerce"
    )

    X_validation[column] = pd.to_numeric(
        X_validation[column],
        errors="coerce"
    )

train_nan_count = int(
    X_train.isna().sum().sum()
)

validation_nan_count = int(
    X_validation.isna().sum().sum()
)

print(f"Training NaN values   : {train_nan_count}")
print(f"Validation NaN values : {validation_nan_count}")

if train_nan_count > 0 or validation_nan_count > 0:

    raise ValueError(
        "NaN values detected in LSTM feature data."
    )


# ============================================================
# HANDLE INFINITE VALUES
# ============================================================

train_inf_count = int(
    np.isinf(X_train.to_numpy()).sum()
)

validation_inf_count = int(
    np.isinf(X_validation.to_numpy()).sum()
)

print(f"Training infinite values   : {train_inf_count}")
print(f"Validation infinite values : {validation_inf_count}")

if train_inf_count > 0 or validation_inf_count > 0:

    raise ValueError(
        "Infinite values detected in LSTM feature data."
    )


# ============================================================
# FEATURE SCALING
# ============================================================

print("\n" + "=" * 78)
print("FITTING FEATURE SCALER")
print("=" * 78)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_validation_scaled = scaler.transform(
    X_validation
)

print("Scaler fitted on training data only.")
print("Validation data transformed using training scaler.")


# ============================================================
# LSTM SEQUENCE DESIGN
# ============================================================

print("\n" + "=" * 78)
print("LSTM SEQUENCE DESIGN")
print("=" * 78)

SEQUENCE_LENGTH = 10
SEQUENCE_STRIDE = 1

print(f"Sequence length : {SEQUENCE_LENGTH}")
print(f"Sequence stride : {SEQUENCE_STRIDE}")
print(f"Features/step   : {len(feature_columns)}")


# ============================================================
# SEQUENCE CREATION FUNCTION
# ============================================================

def create_sequences(X, y, sequence_length, stride):

    sequences = []
    labels = []

    for start in range(
        0,
        len(X) - sequence_length + 1,
        stride
    ):

        end = start + sequence_length

        sequences.append(
            X[start:end]
        )

        labels.append(
            y[end - 1]
        )

    if not sequences:

        raise ValueError(
            "No sequences could be created."
        )

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.int64)
    )


# ============================================================
# CREATE TRAINING SEQUENCES
# ============================================================

print("\n" + "=" * 78)
print("CREATING TRAINING SEQUENCES")
print("=" * 78)

X_train_seq, y_train_seq = create_sequences(
    X_train_scaled,
    y_train,
    SEQUENCE_LENGTH,
    SEQUENCE_STRIDE
)

print(f"Training sequence shape : {X_train_seq.shape}")
print(f"Training label shape    : {y_train_seq.shape}")


# ============================================================
# CREATE VALIDATION SEQUENCES
# ============================================================

print("\n" + "=" * 78)
print("CREATING VALIDATION SEQUENCES")
print("=" * 78)

X_validation_seq, y_validation_seq = create_sequences(
    X_validation_scaled,
    y_validation,
    SEQUENCE_LENGTH,
    SEQUENCE_STRIDE
)

print(
    f"Validation sequence shape : "
    f"{X_validation_seq.shape}"
)

print(
    f"Validation label shape    : "
    f"{y_validation_seq.shape}"
)


# ============================================================
# SAVE NUMPY ARRAYS
# ============================================================

print("\n" + "=" * 78)
print("SAVING LSTM SEQUENCES")
print("=" * 78)

np.save(
    MODEL_DIR / "step25_X_train_sequences.npy",
    X_train_seq
)

np.save(
    MODEL_DIR / "step25_y_train_sequences.npy",
    y_train_seq
)

np.save(
    MODEL_DIR / "step25_X_validation_sequences.npy",
    X_validation_seq
)

np.save(
    MODEL_DIR / "step25_y_validation_sequences.npy",
    y_validation_seq
)

print("Saved training sequences.")
print("Saved validation sequences.")


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    MODEL_DIR / "step25_lstm_scaler.joblib"
)

print("Saved training-fitted scaler.")


# ============================================================
# SAVE LABEL ENCODER
# ============================================================

joblib.dump(
    label_encoder,
    MODEL_DIR / "step25_lstm_label_encoder.joblib"
)

print("Saved label encoder.")


# ============================================================
# SAVE FEATURE LIST
# ============================================================

with open(
    MODEL_DIR / "step25_lstm_features.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        feature_columns,
        f,
        indent=4
    )

print("Saved feature list.")


# ============================================================
# SEQUENCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 78)
print("SEQUENCE LABEL DISTRIBUTION")
print("=" * 78)

train_sequence_distribution = (
    pd.Series(y_train_seq)
    .value_counts()
    .sort_index()
)

validation_sequence_distribution = (
    pd.Series(y_validation_seq)
    .value_counts()
    .sort_index()
)

distribution_rows = []

for class_id, class_name in enumerate(
    label_encoder.classes_
):

    train_count = int(
        train_sequence_distribution.get(
            class_id,
            0
        )
    )

    validation_count = int(
        validation_sequence_distribution.get(
            class_id,
            0
        )
    )

    distribution_rows.append(
        {
            "Class_ID": class_id,
            "Label": class_name,
            "Training_Sequences": train_count,
            "Validation_Sequences": validation_count,
        }
    )

distribution_df = pd.DataFrame(
    distribution_rows
)

print(
    distribution_df.to_string(
        index=False
    )
)

distribution_df.to_csv(
    OUTPUT_DIR
    / "step25_lstm_sequence_class_distribution.csv",
    index=False
)


# ============================================================
# SAVE PREPARATION SUMMARY
# ============================================================

summary = {

    "experiment":
        "Step25_LSTM_Sequence_Model_Preparation",

    "dataset":
        "CIC-IDS2018",

    "training_representation":
        "B1_undersampled",

    "training_file":
        str(TRAIN_FILE),

    "validation_file":
        str(VALIDATION_FILE),

    "feature_count":
        len(feature_columns),

    "class_count":
        len(label_encoder.classes_),

    "sequence_length":
        SEQUENCE_LENGTH,

    "sequence_stride":
        SEQUENCE_STRIDE,

    "training_samples":
        int(len(train_df)),

    "validation_samples":
        int(len(validation_df)),

    "training_sequences":
        int(len(X_train_seq)),

    "validation_sequences":
        int(len(X_validation_seq)),

    "input_shape":
        [
            SEQUENCE_LENGTH,
            len(feature_columns)
        ],

    "scaling":
        "StandardScaler fitted on training data only",

    "validation_used_for_scaler_fitting":
        False,

    "test_set_used":
        False,

    "class_labels":
        label_encoder.classes_.tolist()
}


with open(
    OUTPUT_DIR
    / "step25_lstm_preparation_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# SAVE PROTOCOL
# ============================================================

protocol = f"""
STEP 25 - LSTM SEQUENCE MODEL PREPARATION

Dataset:
CIC-IDS2018

Training representation:
B1 undersampled

Training samples:
{len(train_df)}

Validation samples:
{len(validation_df)}

Feature count:
{len(feature_columns)}

Class count:
{len(label_encoder.classes_)}

Sequence length:
{SEQUENCE_LENGTH}

Sequence stride:
{SEQUENCE_STRIDE}

Training sequence shape:
{X_train_seq.shape}

Validation sequence shape:
{X_validation_seq.shape}

Scaling:
StandardScaler

Scaler fitting:
Training data only

Validation data:
Transformed using the training-fitted scaler

Test data:
Not used

Purpose:
Prepare the existing 30-feature CIC-IDS2018 representation
for a sequence-based LSTM model without changing the
established dataset split or feature-selection pipeline.

No hyperparameter optimization was performed in Step 25.
No test data was used.
"""


with open(
    OUTPUT_DIR
    / "step25_lstm_preparation_protocol.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(protocol)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 78)
print("STEP 25 COMPLETED")
print("=" * 78)

print("Generated files:")

print(
    "  - step25_lstm_sequence_class_distribution.csv"
)

print(
    "  - step25_lstm_preparation_summary.json"
)

print(
    "  - step25_lstm_preparation_protocol.txt"
)

print("\nLSTM files:")

print(
    "  - step25_X_train_sequences.npy"
)

print(
    "  - step25_y_train_sequences.npy"
)

print(
    "  - step25_X_validation_sequences.npy"
)

print(
    "  - step25_y_validation_sequences.npy"
)

print(
    "  - step25_lstm_scaler.joblib"
)

print(
    "  - step25_lstm_label_encoder.joblib"
)

print(
    "  - step25_lstm_features.json"
)

print("\nNEXT:")
print("STEP 26 - LSTM MODEL TRAINING")