import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ================================================================
# STEP 29 - DATASET-SPECIFIC BASELINE MODELS
# ================================================================

print("=" * 70)
print("STEP 29 - DATASET-SPECIFIC BASELINE MODELS")
print("=" * 70)


# ================================================================
# CONFIGURATION
# ================================================================

datasets = {
    "CICIDS2017": {
        "train": "data/CICIDS2017_model_train.csv",
        "test": "data/CICIDS2017_model_test.csv"
    },

    "NSL-KDD": {
        "train": "data/NSLKDD_model_train.csv",
        "test": "data/NSLKDD_model_test.csv"
    }
}


# ================================================================
# MODEL DIRECTORY
# ================================================================

os.makedirs("models", exist_ok=True)


# ================================================================
# FUNCTION FOR EVALUATION
# ================================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc
    }


# ================================================================
# STORE RESULTS
# ================================================================

all_results = []


# ================================================================
# PROCESS EACH DATASET
# ================================================================

for dataset_name, paths in datasets.items():

    print("\n")
    print("=" * 70)
    print(f"PROCESSING {dataset_name}")
    print("=" * 70)


    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------

    print("\nLoading training data...")

    train_df = pd.read_csv(paths["train"])

    print("Loading testing data...")

    test_df = pd.read_csv(paths["test"])


    print("\nTraining shape:")
    print(train_df.shape)

    print("\nTesting shape:")
    print(test_df.shape)


    # ------------------------------------------------------------
    # SEPARATE FEATURES AND LABEL
    # ------------------------------------------------------------

    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]


    # ------------------------------------------------------------
    # DATA VALIDATION
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    print("\nTraining missing values:")
    print(X_train.isnull().sum().sum())

    print("Testing missing values:")
    print(X_test.isnull().sum().sum())


    train_inf = np.isinf(X_train.select_dtypes(include=np.number)).sum().sum()
    test_inf = np.isinf(X_test.select_dtypes(include=np.number)).sum().sum()

    print("\nTraining infinity values:")
    print(train_inf)

    print("Testing infinity values:")
    print(test_inf)


    # ------------------------------------------------------------
    # LABEL DISTRIBUTION
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)

    print("\nTraining:")
    print(y_train.value_counts())

    print("\nTesting:")
    print(y_test.value_counts())


    # ============================================================
    # MODEL 1 - RANDOM FOREST
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf_model.fit(X_train, y_train)

    print("Random Forest training completed.")


    # ------------------------------------------------------------
    # RF EVALUATION
    # ------------------------------------------------------------

    rf_results = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    print("\nRandom Forest Results:")

    for metric, value in rf_results.items():
        print(f"{metric}: {value:.4f}")


    # ------------------------------------------------------------
    # SAVE RANDOM FOREST
    # ------------------------------------------------------------

    rf_path = (
        f"models/"
        f"{dataset_name.replace('-', '').replace(' ', '_')}"
        f"_random_forest_baseline.joblib"
    )

    import joblib

    joblib.dump(
        rf_model,
        rf_path
    )

    print("\nRandom Forest saved to:")
    print(rf_path)


    # ============================================================
    # MODEL 2 - XGBOOST
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)


    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )


    xgb_model.fit(
        X_train,
        y_train
    )

    print("XGBoost training completed.")


    # ------------------------------------------------------------
    # XGBOOST EVALUATION
    # ------------------------------------------------------------

    xgb_results = evaluate_model(
        xgb_model,
        X_test,
        y_test
    )

    print("\nXGBoost Results:")

    for metric, value in xgb_results.items():
        print(f"{metric}: {value:.4f}")


    # ------------------------------------------------------------
    # SAVE XGBOOST
    # ------------------------------------------------------------

    xgb_path = (
        f"models/"
        f"{dataset_name.replace('-', '').replace(' ', '_')}"
        f"_xgboost_baseline.joblib"
    )

    joblib.dump(
        xgb_model,
        xgb_path
    )

    print("\nXGBoost saved to:")
    print(xgb_path)


    # ============================================================
    # STORE RESULTS
    # ============================================================

    all_results.append({
        "Dataset": dataset_name,
        "Model": "Random Forest",
        **rf_results
    })

    all_results.append({
        "Dataset": dataset_name,
        "Model": "XGBoost",
        **xgb_results
    })


# ================================================================
# SAVE RESULTS
# ================================================================

results_df = pd.DataFrame(all_results)

results_path = "data/baseline_model_results.csv"

results_df.to_csv(
    results_path,
    index=False
)


# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n")
print("=" * 70)
print("STEP 29 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nBaseline Model Results:")
print(results_df.to_string(index=False))

print("\nResults saved to:")
print(results_path)

print("\nModels saved inside:")
print("models/")