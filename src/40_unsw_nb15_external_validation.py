"""
========================================================================================
PROJECT: Behaviour Based Network Attack Detection System using Machine Learning
STEP 40 — UNSW-NB15 EXTERNAL / GENERALIZATION VALIDATION
========================================================================================

Purpose:
    Evaluate whether the frozen CIC-IDS2018 final model can be legitimately validated
    on the external UNSW-NB15 dataset.

Research principle:
    External validation MUST NOT fabricate feature mappings.

    The final CIC-IDS2018 Random Forest was trained using a frozen set of 30
    CIC-IDS2018-specific selected features.

    UNSW-NB15 uses a different feature schema and attack taxonomy.

    Therefore this script FIRST performs strict schema compatibility analysis.

    If the required feature schema is not directly compatible:
        - No prediction is performed.
        - No retraining is performed.
        - No artificial feature mapping is performed.
        - No fabricated external accuracy is reported.
        - The incompatibility is documented as a research limitation.

    If strict compatibility is unexpectedly satisfied:
        - External prediction/evaluation may proceed.

IMPORTANT RESEARCH-INTEGRITY RULES:
    - CIC-IDS2018 test set is NOT used.
    - No model training.
    - No hyperparameter optimization.
    - No model selection using UNSW-NB15.
    - Frozen final model remains unchanged.
    - No artificial feature mapping.
========================================================================================
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd


# ======================================================================================
# CONFIGURATION
# ======================================================================================

PROJECT_ROOT = Path(r"D:\network_attack_detection")

UNSW_DIR = PROJECT_ROOT / "data" / "external" / "UNSW-NB15"

RF_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "step21_rf_tuning"
    / "step21_best_random_forest.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "step40_unsw_nb15_external_validation"
)

# CIC-IDS2018 frozen selected-feature file.
# This is used ONLY to recover the exact frozen feature schema.
CIC_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "CICIDS2018_validation_selected.csv"
)

UNSW_TRAINING_PATH = UNSW_DIR / "UNSW_NB15_training-set.csv"
UNSW_TESTING_PATH = UNSW_DIR / "UNSW_NB15_testing-set.csv"
UNSW_FEATURES_PATH = UNSW_DIR / "NUSW-NB15_features.csv"
UNSW_GT_PATH = UNSW_DIR / "NUSW-NB15_GT.csv"
UNSW_DESCRIPTION_PATH = UNSW_DIR / "The UNSW-NB15 description.pdf"

# Expected final CIC project class mapping.
PROJECT_CLASS_MAPPING = {
    0: "BENIGN",
    1: "Bot",
    2: "Brute Force -Web",
    3: "Brute Force -XSS",
    4: "DDOS attack-HOIC",
    5: "DDOS attack-LOIC-UDP",
    6: "DoS attacks-GoldenEye",
    7: "DoS attacks-Hulk",
    8: "DoS attacks-SlowHTTPTest",
    9: "DoS attacks-Slowloris",
    10: "FTP-BruteForce",
    11: "Infiltration",
    12: "SQL Injection",
    13: "SSH-Bruteforce",
}

EXPECTED_CIC_FEATURE_COUNT = 30

RANDOM_SEED = 42


# ======================================================================================
# UTILITY FUNCTIONS
# ======================================================================================

def section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def subsection(title: str) -> None:
    print()
    print("-" * 88)
    print(title)
    print("-" * 88)


def fail(message: str) -> None:
    raise RuntimeError(f"\n[STEP 40 FATAL ERROR]\n{message}\n")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def normalize_feature_name(name: str) -> str:
    """
    Conservative normalization for comparison only.

    IMPORTANT:
        This normalization does NOT establish semantic compatibility.
        It is used to identify exact/near-exact naming overlap.

    No automatic semantic mapping is performed.
    """
    value = str(name).strip().lower()
    value = value.replace(" ", "_")
    value = value.replace("-", "_")
    return value


def safe_read_header(path: Path) -> List[str]:
    """
    Read CSV header only.
    """
    try:
        df = pd.read_csv(path, nrows=0)
        return list(df.columns)
    except Exception as exc:
        fail(f"Could not read CSV header:\n{path}\nError: {exc}")
        return []


def get_label_candidates(columns: List[str]) -> List[str]:
    candidates = []

    preferred = [
        "label",
        "Label",
        "attack_cat",
        "Attack_cat",
        "attack category",
        "Attack Category",
        "id",
        "ID",
    ]

    for col in preferred:
        if col in columns and col not in candidates:
            candidates.append(col)

    for col in columns:
        lower = str(col).strip().lower()

        if (
            "attack_cat" in lower
            or lower == "label"
            or lower == "attack category"
            or lower == "attack_category"
        ):
            if col not in candidates:
                candidates.append(col)

    return candidates


def inspect_csv_basic(path: Path) -> Dict[str, Any]:
    """
    Inspect basic metadata without loading the entire large CSV.
    """
    result: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": None,
        "columns": [],
        "column_count": 0,
        "label_candidates": [],
    }

    if not path.exists():
        return result

    result["size_bytes"] = path.stat().st_size

    columns = safe_read_header(path)

    result["columns"] = columns
    result["column_count"] = len(columns)
    result["label_candidates"] = get_label_candidates(columns)

    return result


def print_columns(title: str, columns: List[str]) -> None:
    subsection(title)

    print(f"Column count: {len(columns)}")

    for idx, col in enumerate(columns, start=1):
        print(f"{idx:3d}. {col}")


# ======================================================================================
# MAIN
# ======================================================================================

def main() -> None:

    start_time = time.time()

    section(
        "STEP 40 — UNSW-NB15 EXTERNAL / GENERALIZATION VALIDATION"
    )

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("UNSW-NB15 directory:")
    print(UNSW_DIR)

    print()
    print("Frozen final model:")
    print(RF_MODEL_PATH)

    print()
    print("CIC-IDS2018 frozen feature reference:")
    print(CIC_VALIDATION_PATH)

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    subsection("RESEARCH INTEGRITY")

    print("CIC-IDS2018 test set used: NO")
    print("UNSW-NB15 used as external dataset: YES")
    print("Training performed: NO")
    print("Hyperparameter optimization: NO")
    print("Model selection using UNSW-NB15: NO")
    print("Feature remapping: NO")
    print("Artificial semantic mapping: NO")
    print("Frozen final candidate: Tuned Random Forest")
    print("Primary external-validation principle: STRICT FEATURE COMPATIBILITY")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==================================================================================
    # [1/10] REQUIRED FILE CHECK
    # ==================================================================================

    subsection("[1/10] Checking required input files...")

    required_files = {
        "UNSW training set": UNSW_TRAINING_PATH,
        "UNSW testing set": UNSW_TESTING_PATH,
        "UNSW feature description": UNSW_FEATURES_PATH,
        "UNSW ground truth": UNSW_GT_PATH,
        "UNSW description PDF": UNSW_DESCRIPTION_PATH,
        "CIC validation feature reference": CIC_VALIDATION_PATH,
        "Frozen Random Forest": RF_MODEL_PATH,
    }

    file_status = {}

    for name, path in required_files.items():
        exists = path.exists()
        file_status[name] = {
            "path": str(path),
            "exists": exists,
        }

        if exists:
            print(f"[OK] {name}: {path}")
        else:
            print(f"[MISSING] {name}: {path}")

    # The feature description and GT file may not be strictly required for
    # schema comparison, but we record their status.
    mandatory = [
        "UNSW training set",
        "UNSW testing set",
        "CIC validation feature reference",
        "Frozen Random Forest",
    ]

    missing_mandatory = [
        name for name in mandatory
        if not file_status[name]["exists"]
    ]

    if missing_mandatory:
        fail(
            "Mandatory files are missing:\n"
            + "\n".join(
                f"- {name}: {file_status[name]['path']}"
                for name in missing_mandatory
            )
        )

    print()
    print("Mandatory input files: VERIFIED")

    # ==================================================================================
    # [2/10] INSPECT UNSW SCHEMAS
    # ==================================================================================

    subsection("[2/10] Inspecting UNSW-NB15 CSV schemas...")

    unsw_training_info = inspect_csv_basic(UNSW_TRAINING_PATH)
    unsw_testing_info = inspect_csv_basic(UNSW_TESTING_PATH)

    print(
        f"UNSW training columns: "
        f"{unsw_training_info['column_count']}"
    )

    print(
        f"UNSW testing columns: "
        f"{unsw_testing_info['column_count']}"
    )

    if unsw_training_info["column_count"] == 0:
        fail("UNSW training CSV contains no readable columns.")

    if unsw_testing_info["column_count"] == 0:
        fail("UNSW testing CSV contains no readable columns.")

    print_columns(
        "UNSW-NB15 TRAINING-SET COLUMNS",
        unsw_training_info["columns"],
    )

    print_columns(
        "UNSW-NB15 TESTING-SET COLUMNS",
        unsw_testing_info["columns"],
    )

    # ==================================================================================
    # [3/10] VERIFY UNSW TRAINING / TESTING SCHEMA CONSISTENCY
    # ==================================================================================

    subsection("[3/10] Verifying UNSW training/testing schema consistency...")

    train_columns = unsw_training_info["columns"]
    test_columns = unsw_testing_info["columns"]

    train_set = set(train_columns)
    test_set = set(test_columns)

    train_only = [c for c in train_columns if c not in test_set]
    test_only = [c for c in test_columns if c not in train_set]

    exact_same_schema = (
        train_columns == test_columns
    )

    same_column_set = (
        train_set == test_set
    )

    print(f"Exact column order match: {exact_same_schema}")
    print(f"Same column set: {same_column_set}")

    if train_only:
        print("Training-only columns:")
        for c in train_only:
            print(f"  - {c}")

    if test_only:
        print("Testing-only columns:")
        for c in test_only:
            print(f"  - {c}")

    if exact_same_schema:
        print("[OK] UNSW training/testing schema is identical.")
    elif same_column_set:
        print(
            "[WARNING] UNSW training/testing contain the same columns "
            "but in different order."
        )
    else:
        print(
            "[WARNING] UNSW training/testing column sets differ."
        )

    # ==================================================================================
    # [4/10] RECOVER FROZEN CIC FEATURE SCHEMA
    # ==================================================================================

    subsection(
        "[4/10] Recovering frozen CIC-IDS2018 30-feature schema..."
    )

    try:
        cic_header = pd.read_csv(
            CIC_VALIDATION_PATH,
            nrows=0,
        ).columns.tolist()
    except Exception as exc:
        fail(
            f"Could not read CIC-IDS2018 validation header:\n"
            f"{CIC_VALIDATION_PATH}\n"
            f"Error: {exc}"
        )

    if "Label" not in cic_header:
        fail(
            "CIC-IDS2018 validation reference does not contain "
            "the expected 'Label' column."
        )

    cic_features = [
        col for col in cic_header
        if col != "Label"
    ]

    print(f"CIC total columns: {len(cic_header)}")
    print(f"CIC frozen feature count: {len(cic_features)}")

    if len(cic_features) != EXPECTED_CIC_FEATURE_COUNT:
        fail(
            "Frozen CIC feature count does not match project specification.\n"
            f"Expected: {EXPECTED_CIC_FEATURE_COUNT}\n"
            f"Found: {len(cic_features)}"
        )

    print("[OK] Frozen CIC feature count = 30.")

    print_columns(
        "FROZEN CIC-IDS2018 SELECTED FEATURES",
        cic_features,
    )

    # ==================================================================================
    # [5/10] EXACT FEATURE NAME OVERLAP
    # ==================================================================================

    subsection(
        "[5/10] Comparing frozen CIC features with UNSW-NB15 features..."
    )

    unsw_train_features = [
        c for c in train_columns
        if c not in get_label_candidates(train_columns)
    ]

    unsw_test_features = [
        c for c in test_columns
        if c not in get_label_candidates(test_columns)
    ]

    print(f"UNSW training candidate features: {len(unsw_train_features)}")
    print(f"UNSW testing candidate features: {len(unsw_test_features)}")

    exact_overlap = [
        feature
        for feature in cic_features
        if feature in unsw_train_features
    ]

    print()
    print("EXACT FEATURE NAME OVERLAP")
    print(f"Matching features: {len(exact_overlap)} / {len(cic_features)}")

    if exact_overlap:
        for feature in exact_overlap:
            print(f"  [MATCH] {feature}")

    missing_exact = [
        feature
        for feature in cic_features
        if feature not in unsw_train_features
    ]

    print()
    print("CIC FEATURES NOT PRESENT IN UNSW")
    print(f"Missing: {len(missing_exact)} / {len(cic_features)}")

    for feature in missing_exact:
        print(f"  [MISSING] {feature}")

    # ==================================================================================
    # [6/10] CONSERVATIVE NORMALIZED NAME COMPARISON
    # ==================================================================================

    subsection(
        "[6/10] Performing conservative normalized-name comparison..."
    )

    normalized_unsw: Dict[str, List[str]] = {}

    for feature in unsw_train_features:
        normalized = normalize_feature_name(feature)

        normalized_unsw.setdefault(
            normalized,
            []
        ).append(feature)

    normalized_matches = []

    for cic_feature in cic_features:
        normalized = normalize_feature_name(cic_feature)

        if normalized in normalized_unsw:
            normalized_matches.append(
                {
                    "cic_feature": cic_feature,
                    "unsw_feature": normalized_unsw[normalized],
                }
            )

    print(
        f"Normalized-name matches: "
        f"{len(normalized_matches)} / {len(cic_features)}"
    )

    if normalized_matches:
        for item in normalized_matches:
            print(
                f"  [NORMALIZED MATCH] "
                f"{item['cic_feature']} -> "
                f"{item['unsw_feature']}"
            )

    print()
    print(
        "IMPORTANT: normalized-name overlap is NOT treated as "
        "semantic feature compatibility."
    )

    # ==================================================================================
    # [7/10] LABEL / ATTACK TAXONOMY INSPECTION
    # ==================================================================================

    subsection(
        "[7/10] Inspecting UNSW label and attack taxonomy..."
    )

    training_label_candidates = get_label_candidates(train_columns)
    testing_label_candidates = get_label_candidates(test_columns)

    print("Training label candidates:")
    for label in training_label_candidates:
        print(f"  - {label}")

    print("Testing label candidates:")
    for label in testing_label_candidates:
        print(f"  - {label}")

    # Load only label columns when possible.
    unsw_label_summary: Dict[str, Any] = {}

    for dataset_name, path, label_candidates in [
        (
            "training",
            UNSW_TRAINING_PATH,
            training_label_candidates,
        ),
        (
            "testing",
            UNSW_TESTING_PATH,
            testing_label_candidates,
        ),
    ]:

        if not label_candidates:
            unsw_label_summary[dataset_name] = {
                "status": "No obvious label column detected."
            }
            continue

        preferred_label = None

        for candidate in [
            "attack_cat",
            "label",
            "Label",
            "Attack_cat",
        ]:
            if candidate in label_candidates:
                preferred_label = candidate
                break

        if preferred_label is None:
            preferred_label = label_candidates[0]

        print()
        print(
            f"Loading UNSW {dataset_name} label column: "
            f"{preferred_label}"
        )

        try:
            label_df = pd.read_csv(
                path,
                usecols=[preferred_label],
            )

            counts = (
                label_df[preferred_label]
                .astype(str)
                .value_counts(dropna=False)
            )

            print(
                f"UNSW {dataset_name} rows: "
                f"{len(label_df):,}"
            )

            print(
                f"UNSW {dataset_name} unique values: "
                f"{len(counts)}"
            )

            print(counts.to_string())

            unsw_label_summary[dataset_name] = {
                "label_column": preferred_label,
                "row_count": int(len(label_df)),
                "unique_value_count": int(len(counts)),
                "class_distribution": {
                    str(k): int(v)
                    for k, v in counts.items()
                },
            }

            del label_df

        except Exception as exc:
            print(
                f"[WARNING] Could not inspect {dataset_name} labels: "
                f"{exc}"
            )

            unsw_label_summary[dataset_name] = {
                "label_column": preferred_label,
                "error": str(exc),
            }

    # ==================================================================================
    # [8/10] STRICT EXTERNAL COMPATIBILITY DECISION
    # ==================================================================================

    subsection(
        "[8/10] Determining strict external-validation compatibility..."
    )

    compatibility_reasons = []

    # Feature count comparison
    if len(unsw_train_features) != EXPECTED_CIC_FEATURE_COUNT:
        compatibility_reasons.append(
            "UNSW-NB15 does not contain exactly the frozen 30 CIC features."
        )

    # Exact feature set comparison
    if set(cic_features) != set(unsw_train_features):
        compatibility_reasons.append(
            "UNSW-NB15 feature names do not exactly match the frozen "
            "CIC-IDS2018 feature set."
        )

    # Exact overlap
    if len(exact_overlap) != len(cic_features):
        compatibility_reasons.append(
            f"Only {len(exact_overlap)} of {len(cic_features)} "
            "frozen CIC features have exact-name matches."
        )

    # CIC label taxonomy vs UNSW taxonomy
    unsw_attack_cat = None

    for dataset_name in ["training", "testing"]:
        info = unsw_label_summary.get(dataset_name, {})

        if info.get("label_column") == "attack_cat":
            unsw_attack_cat = info
            break

    if unsw_attack_cat is not None:
        compatibility_reasons.append(
            "UNSW-NB15 uses its own attack taxonomy ('attack_cat'), "
            "which is not identical to the 14-class CIC-IDS2018 taxonomy."
        )

    strict_feature_compatibility = (
        len(cic_features) == EXPECTED_CIC_FEATURE_COUNT
        and len(unsw_train_features) == EXPECTED_CIC_FEATURE_COUNT
        and set(cic_features) == set(unsw_train_features)
        and len(exact_overlap) == len(cic_features)
    )

    print()
    print("STRICT FEATURE COMPATIBILITY:")
    print(
        "PASS"
        if strict_feature_compatibility
        else "FAIL"
    )

    if strict_feature_compatibility:
        print()
        print(
            "The UNSW feature schema appears structurally identical "
            "to the frozen CIC schema."
        )
    else:
        print()
        print(
            "External prediction will NOT be performed because the "
            "UNSW-NB15 feature schema is not strictly compatible "
            "with the frozen CIC-IDS2018 model input."
        )

    # ==================================================================================
    # [9/10] MODEL COMPATIBILITY / SAFETY CHECK
    # ==================================================================================

    subsection(
        "[9/10] Frozen Random Forest compatibility safety check..."
    )

    model_info: Dict[str, Any] = {
        "model_path": str(RF_MODEL_PATH),
        "loaded": False,
        "model_type": None,
        "n_features_in": None,
        "classes": None,
    }

    try:
        import joblib

        rf_model = joblib.load(RF_MODEL_PATH)

        model_info["loaded"] = True
        model_info["model_type"] = type(rf_model).__name__

        n_features_in = getattr(
            rf_model,
            "n_features_in_",
            None,
        )

        model_info["n_features_in"] = (
            int(n_features_in)
            if n_features_in is not None
            else None
        )

        classes = getattr(
            rf_model,
            "classes_",
            None,
        )

        if classes is not None:
            model_info["classes"] = [
                str(value)
                for value in classes
            ]

        print(
            f"Random Forest type: "
            f"{model_info['model_type']}"
        )

        print(
            f"Random Forest feature count: "
            f"{model_info['n_features_in']}"
        )

        if n_features_in is not None:
            if int(n_features_in) != EXPECTED_CIC_FEATURE_COUNT:
                fail(
                    "Frozen Random Forest does not contain the expected "
                    f"{EXPECTED_CIC_FEATURE_COUNT} input features.\n"
                    f"Found: {n_features_in}"
                )

            print(
                f"[OK] Frozen Random Forest expects "
                f"{EXPECTED_CIC_FEATURE_COUNT} features."
            )

        print(
            "Model loading performed for compatibility verification only."
        )

    except Exception as exc:
        fail(
            "Could not load frozen Random Forest for compatibility check.\n"
            f"Error: {exc}"
        )

    # ==================================================================================
    # [10/10] DECISION + ARTIFACTS
    # ==================================================================================

    subsection(
        "[10/10] Saving Step 40 external-validation research artifacts..."
    )

    external_validation_performed = False

    if strict_feature_compatibility:

        decision = (
            "COMPATIBLE — external prediction may be methodologically "
            "possible, subject to label-taxonomy compatibility."
        )

        methodology_status = (
            "STRICT FEATURE COMPATIBILITY PASSED. "
            "However, UNSW-NB15 attack taxonomy must be handled separately "
            "and no automatic 14-class mapping is authorized."
        )

    else:

        decision = (
            "INCOMPATIBLE — external prediction NOT performed."
        )

        methodology_status = (
            "The frozen CIC-IDS2018 model cannot be directly evaluated "
            "on UNSW-NB15 without an unsupported feature transformation "
            "or retraining. Therefore external accuracy/F1 is intentionally "
            "not fabricated."
        )

    summary = {
        "step": 40,
        "title": "UNSW-NB15 External / Generalization Validation",
        "project_root": str(PROJECT_ROOT),
        "unsw_directory": str(UNSW_DIR),
        "frozen_model": str(RF_MODEL_PATH),
        "cic_feature_reference": str(CIC_VALIDATION_PATH),

        "research_integrity": {
            "cic_test_set_used": False,
            "unsw_external_dataset_used": True,
            "training_performed": False,
            "hyperparameter_optimization": False,
            "model_selection_using_unsw": False,
            "feature_remapping": False,
            "artificial_semantic_mapping": False,
            "post_test_model_selection": False,
        },

        "input_files": file_status,

        "unsw_schema": {
            "training_column_count": len(train_columns),
            "testing_column_count": len(test_columns),
            "exact_training_testing_schema_match": exact_same_schema,
            "same_training_testing_column_set": same_column_set,
            "training_columns": train_columns,
            "testing_columns": test_columns,
        },

        "frozen_cic_schema": {
            "feature_count": len(cic_features),
            "features": cic_features,
        },

        "feature_compatibility": {
            "expected_cic_feature_count": EXPECTED_CIC_FEATURE_COUNT,
            "unsw_candidate_feature_count": len(unsw_train_features),
            "exact_overlap_count": len(exact_overlap),
            "exact_overlap_features": exact_overlap,
            "missing_exact_cic_features": missing_exact,
            "normalized_match_count": len(normalized_matches),
            "normalized_matches": normalized_matches,
            "strict_feature_compatibility": strict_feature_compatibility,
        },

        "label_taxonomy": unsw_label_summary,

        "model_compatibility": model_info,

        "external_validation_performed": external_validation_performed,

        "decision": decision,

        "methodology_status": methodology_status,

        "compatibility_reasons": compatibility_reasons,

        "final_statement": (
            "UNSW-NB15 is retained as an external/generalization dataset. "
            "No fabricated cross-dataset prediction is reported when the "
            "frozen CIC-IDS2018 feature schema is incompatible."
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "step40_unsw_external_validation_summary.json"
    )

    save_json(summary_path, summary)

    # ------------------------------------------------------------------------------
    # Feature comparison CSV
    # ------------------------------------------------------------------------------

    feature_rows = []

    unsw_exact_set = set(unsw_train_features)

    for idx, feature in enumerate(cic_features, start=1):

        feature_rows.append(
            {
                "cic_feature_index": idx,
                "cic_feature": feature,
                "exact_match_in_unsw": feature in unsw_exact_set,
                "normalized_name": normalize_feature_name(feature),
            }
        )

    feature_comparison_df = pd.DataFrame(feature_rows)

    feature_comparison_path = (
        OUTPUT_DIR
        / "step40_cic_vs_unsw_feature_compatibility.csv"
    )

    feature_comparison_df.to_csv(
        feature_comparison_path,
        index=False,
    )

    # ------------------------------------------------------------------------------
    # UNSW schema CSV
    # ------------------------------------------------------------------------------

    schema_rows = []

    for idx, col in enumerate(train_columns, start=1):

        schema_rows.append(
            {
                "column_index": idx,
                "unsw_training_column": col,
                "present_in_testing": col in test_set,
                "candidate_feature": col in unsw_train_features,
                "possible_label_column": col in training_label_candidates,
            }
        )

    schema_df = pd.DataFrame(schema_rows)

    schema_path = (
        OUTPUT_DIR
        / "step40_unsw_schema_inventory.csv"
    )

    schema_df.to_csv(
        schema_path,
        index=False,
    )

    # ------------------------------------------------------------------------------
    # Protocol TXT
    # ------------------------------------------------------------------------------

    protocol_path = (
        OUTPUT_DIR
        / "step40_external_validation_protocol.txt"
    )

    with open(
        protocol_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "STEP 40 — UNSW-NB15 EXTERNAL / GENERALIZATION VALIDATION\n"
        )
        f.write("=" * 88 + "\n\n")

        f.write("PROJECT\n")
        f.write(
            "Behaviour Based Network Attack Detection System using "
            "Machine Learning\n\n"
        )

        f.write("FINAL PRIMARY MODEL\n")
        f.write("Tuned Random Forest\n\n")

        f.write("EXTERNAL DATASET\n")
        f.write("UNSW-NB15\n\n")

        f.write("RESEARCH-INTEGRITY RULES\n")
        f.write("- CIC-IDS2018 test set was NOT used.\n")
        f.write("- No training was performed.\n")
        f.write("- No hyperparameter optimization was performed.\n")
        f.write("- No model selection was performed using UNSW-NB15.\n")
        f.write("- No artificial feature mapping was performed.\n")
        f.write("- No unsupported label mapping was performed.\n\n")

        f.write("FROZEN CIC FEATURE SCHEMA\n")
        f.write(f"Feature count: {len(cic_features)}\n\n")

        for idx, feature in enumerate(cic_features, start=1):
            f.write(f"{idx:3d}. {feature}\n")

        f.write("\n")
        f.write("UNSW FEATURE SCHEMA\n")
        f.write(f"Candidate feature count: {len(unsw_train_features)}\n\n")

        for idx, feature in enumerate(
            unsw_train_features,
            start=1,
        ):
            f.write(f"{idx:3d}. {feature}\n")

        f.write("\n")
        f.write("EXACT FEATURE OVERLAP\n")
        f.write(
            f"{len(exact_overlap)} / "
            f"{len(cic_features)}\n\n"
        )

        for feature in exact_overlap:
            f.write(f"MATCH: {feature}\n")

        f.write("\n")
        f.write("STRICT COMPATIBILITY DECISION\n")
        f.write(
            "PASS\n"
            if strict_feature_compatibility
            else "FAIL\n"
        )

        f.write("\n")
        f.write("FINAL METHODOLOGICAL DECISION\n")
        f.write(decision + "\n\n")

        f.write("INTERPRETATION\n")
        f.write(methodology_status + "\n\n")

        if not strict_feature_compatibility:

            f.write(
                "WHY EXTERNAL PREDICTION WAS NOT PERFORMED\n"
            )
            f.write(
                "The frozen CIC-IDS2018 Random Forest expects the "
                "30 selected CIC-IDS2018 features. UNSW-NB15 uses a "
                "different feature representation. Predicting UNSW "
                "samples by merely selecting similarly named or vaguely "
                "related features would change the meaning of the "
                "model inputs and invalidate the external evaluation.\n\n"
            )

            f.write(
                "RESEARCH CONSEQUENCE\n"
            )
            f.write(
                "UNSW-NB15 remains an important external dataset for "
                "generalization analysis, but direct evaluation of the "
                "frozen CIC-IDS2018 model is not methodologically "
                "supported under the current frozen feature pipeline.\n"
            )

        else:

            f.write(
                "IMPORTANT LABEL WARNING\n"
            )
            f.write(
                "Even when feature compatibility is satisfied, UNSW-NB15 "
                "uses a different attack taxonomy. No automatic conversion "
                "to the 14 CIC-IDS2018 classes is authorized without a "
                "scientifically justified mapping.\n"
            )

    # ------------------------------------------------------------------------------
    # Human-readable result TXT
    # ------------------------------------------------------------------------------

    result_path = (
        OUTPUT_DIR
        / "step40_external_validation_result.txt"
    )

    runtime = time.time() - start_time

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "STEP 40 — UNSW-NB15 EXTERNAL / GENERALIZATION VALIDATION\n"
        )
        f.write("=" * 88 + "\n\n")

        f.write(f"Strict feature compatibility: ")
        f.write(
            "PASS\n"
            if strict_feature_compatibility
            else "FAIL\n"
        )

        f.write(
            f"Frozen CIC features: {len(cic_features)}\n"
        )

        f.write(
            f"UNSW candidate features: {len(unsw_train_features)}\n"
        )

        f.write(
            f"Exact feature overlap: "
            f"{len(exact_overlap)} / {len(cic_features)}\n"
        )

        f.write(
            f"External prediction performed: "
            f"{external_validation_performed}\n\n"
        )

        f.write("DECISION\n")
        f.write(decision + "\n\n")

        f.write("RESEARCH-INTEGRITY STATUS\n")
        f.write("CIC-IDS2018 test set used: NO\n")
        f.write("Training performed: NO\n")
        f.write("Hyperparameter optimization: NO\n")
        f.write("UNSW-based model selection: NO\n")
        f.write("Artificial feature mapping: NO\n")
        f.write("Fabricated external metrics: NO\n\n")

        f.write("OUTPUT DIRECTORY\n")
        f.write(str(OUTPUT_DIR) + "\n\n")

        f.write(f"Runtime: {runtime:.2f} seconds\n")

    # ==================================================================================
    # FINAL CONSOLE OUTPUT
    # ==================================================================================

    section("STEP 40 COMPLETE")

    print()
    print("EXTERNAL DATASET")
    print("UNSW-NB15")

    print()
    print("FROZEN CIC FEATURE COUNT")
    print(len(cic_features))

    print()
    print("UNSW CANDIDATE FEATURE COUNT")
    print(len(unsw_train_features))

    print()
    print("EXACT FEATURE OVERLAP")
    print(
        f"{len(exact_overlap)} / {len(cic_features)}"
    )

    print()
    print("STRICT FEATURE COMPATIBILITY")
    print(
        "PASS"
        if strict_feature_compatibility
        else "FAIL"
    )

    print()
    print("EXTERNAL PREDICTION")
    print(
        "PERFORMED"
        if external_validation_performed
        else "NOT PERFORMED — SCHEMA INCOMPATIBILITY"
    )

    print()
    print("FINAL DECISION")
    print(decision)

    print()
    print("METHODOLOGICAL INTERPRETATION")
    print(methodology_status)

    print()
    print("RESEARCH-INTEGRITY STATUS")
    print("CIC-IDS2018 test set: NOT USED")
    print("Training: NOT PERFORMED")
    print("Hyperparameter optimization: NOT PERFORMED")
    print("UNSW-based model selection: NOT PERFORMED")
    print("Artificial feature mapping: NOT PERFORMED")

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    print()
    print(f"Runtime: {runtime:.2f} seconds")

    print("=" * 88)


# ======================================================================================
# ENTRY POINT
# ======================================================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as exc:

        print()
        print("=" * 88)
        print("STEP 40 FAILED")
        print("=" * 88)
        print(str(exc))
        print()
        traceback.print_exc()
        print("=" * 88)

        sys.exit(1)