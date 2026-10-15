"""
STEP 3 - CICIoT2023 ATTACK TAXONOMY AND LABEL MAPPING

Purpose:
1. Read the Step-2 audit report.
2. Discover all raw attack labels.
3. Map raw labels into meaningful attack families.
4. Validate that every non-null label is assigned.
5. Save the taxonomy for reproducible downstream modeling.

IMPORTANT:
- No rows are removed.
- No SMOTE.
- No scaling.
- No feature selection.
- No model training.

This step only defines and validates the label taxonomy.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "CICIoT2023"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

LABEL_COLUMN = "Label"

OUTPUT_FILE = EXPERIMENTS_DIR / "ciciot2023_attack_taxonomy.csv"
SUMMARY_FILE = EXPERIMENTS_DIR / "ciciot2023_attack_family_summary.csv"


# ============================================================
# ATTACK FAMILY MAPPING
# ============================================================

def map_attack_family(label):
    """
    Map CICIoT2023 raw labels into research-oriented
    attack families.

    The mapping is based on the semantic meaning of
    the original CICIoT2023 labels.
    """

    if pd.isna(label):
        return "UNKNOWN_LABEL"

    label = str(label).strip().upper()

    # --------------------------------------------------------
    # BENIGN
    # --------------------------------------------------------

    if label == "BENIGN":
        return "BENIGN"

    # --------------------------------------------------------
    # DDoS
    # --------------------------------------------------------

    if label.startswith("DDOS-"):
        return "DDOS"

    # --------------------------------------------------------
    # DoS
    # --------------------------------------------------------

    if label.startswith("DOS-"):
        return "DOS"

    # --------------------------------------------------------
    # BOTNET / MIRAI
    # --------------------------------------------------------

    if label.startswith("MIRAI-"):
        return "BOTNET"

    # --------------------------------------------------------
    # RECONNAISSANCE
    # --------------------------------------------------------

    if label.startswith("RECON-"):
        return "RECONNAISSANCE"

    # --------------------------------------------------------
    # SPOOFING / MITM
    # --------------------------------------------------------

    if label in {
        "MITM-ARPSPOOFING",
        "DNS_SPOOFING",
    }:
        return "SPOOFING_MITM"

    # --------------------------------------------------------
    # WEB / APPLICATION ATTACKS
    # --------------------------------------------------------

    if label in {
        "XSS",
        "SQLINJECTION",
        "COMMANDINJECTION",
        "UPLOADING_ATTACK",
        "BROWSERHIJACKING",
    }:
        return "WEB_APPLICATION"

    # --------------------------------------------------------
    # BRUTE FORCE
    # --------------------------------------------------------

    if label in {
        "DICTIONARYBRUTEFORCE",
    }:
        return "BRUTE_FORCE"

    # --------------------------------------------------------
    # BACKDOOR / MALWARE
    # --------------------------------------------------------

    if label in {
        "BACKDOOR_MALWARE",
    }:
        return "MALWARE_BACKDOOR"

    # --------------------------------------------------------
    # VULNERABILITY SCANNING
    # --------------------------------------------------------

    if label in {
        "VULNERABILITYSCAN",
    }:
        return "VULNERABILITY_SCAN"

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return "OTHER"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 3 - CICIoT2023 ATTACK TAXONOMY")
    print("=" * 70)

    print("\nDataset directory:")
    print(RAW_DIR)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{RAW_DIR}"
        )

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in CICIoT2023 directory."
        )

    print(f"\nCSV files found: {len(csv_files)}")

    # ========================================================
    # DISCOVER RAW LABELS
    # ========================================================

    print("\n" + "=" * 70)
    print("DISCOVERING RAW LABELS")
    print("=" * 70)

    label_counts = {}

    for index, csv_file in enumerate(csv_files, start=1):

        print(
            f"[{index}/{len(csv_files)}] "
            f"Reading {csv_file.name} ..."
        )

        try:

            df = pd.read_csv(
                csv_file,
                usecols=[LABEL_COLUMN]
            )

        except ValueError:

            print(
                f"[WARNING] {LABEL_COLUMN} not found "
                f"in {csv_file.name}"
            )
            continue

        counts = df[LABEL_COLUMN].value_counts(
            dropna=False
        )

        for label, count in counts.items():

            if pd.isna(label):
                key = "__NULL__"
            else:
                key = str(label).strip()

            label_counts[key] = (
                label_counts.get(key, 0) + int(count)
            )

        del df

    # ========================================================
    # RAW LABEL TABLE
    # ========================================================

    raw_rows = []

    for label, count in label_counts.items():

        if label == "__NULL__":
            raw_label = np.nan
        else:
            raw_label = label

        family = map_attack_family(raw_label)

        raw_rows.append({
            "raw_label": raw_label,
            "attack_family": family,
            "sample_count": count
        })

    taxonomy_df = pd.DataFrame(raw_rows)

    taxonomy_df = taxonomy_df.sort_values(
        by="sample_count",
        ascending=False
    ).reset_index(drop=True)

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("TAXONOMY VALIDATION")
    print("=" * 70)

    non_null_taxonomy = taxonomy_df[
        taxonomy_df["raw_label"].notna()
    ]

    unmapped = non_null_taxonomy[
        non_null_taxonomy["attack_family"].isin(
            ["OTHER", "UNKNOWN_LABEL"]
        )
    ]

    print(
        f"\nTotal raw labels: "
        f"{len(taxonomy_df)}"
    )

    print(
        f"Non-null raw labels: "
        f"{len(non_null_taxonomy)}"
    )

    print(
        f"Unmapped/other labels: "
        f"{len(unmapped)}"
    )

    if len(unmapped) > 0:

        print("\nLabels requiring review:")

        for label in unmapped["raw_label"]:
            print(f"  [REVIEW] {label}")

    else:

        print(
            "\n[OK] All known raw attack labels "
            "have a defined attack family."
        )

    # ========================================================
    # FAMILY SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("ATTACK FAMILY DISTRIBUTION")
    print("=" * 70)

    family_summary = (
        taxonomy_df[
            taxonomy_df["attack_family"] != "UNKNOWN_LABEL"
        ]
        .groupby("attack_family", as_index=False)
        ["sample_count"]
        .sum()
        .sort_values(
            by="sample_count",
            ascending=False
        )
    )

    total_samples = family_summary["sample_count"].sum()

    family_summary["percentage"] = (
        family_summary["sample_count"]
        / total_samples
        * 100
    )

    print(
        family_summary.to_string(
            index=False,
            formatters={
                "percentage": "{:.4f}".format
            }
        )
    )

    # ========================================================
    # SAVE TAXONOMY
    # ========================================================

    print("\n" + "=" * 70)
    print("SAVING TAXONOMY")
    print("=" * 70)

    taxonomy_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    family_summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        f"\nTaxonomy saved to:\n{OUTPUT_FILE}"
    )

    print(
        f"\nFamily summary saved to:\n{SUMMARY_FILE}"
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nResearch taxonomy:")

    for family in family_summary["attack_family"]:
        print(f"  - {family}")

    print("\nIMPORTANT:")
    print("1. Raw CICIoT2023 labels were preserved.")
    print("2. Attack families were created for modeling.")
    print("3. No samples were removed.")
    print("4. No SMOTE was applied.")
    print("5. No feature transformation was performed.")
    print("6. No model was trained.")
    print("7. This taxonomy will be reused throughout the pipeline.")


if __name__ == "__main__":
    main()