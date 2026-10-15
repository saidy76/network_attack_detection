import pandas as pd
import numpy as np


# ============================================================
# COMBINED FEATURE SELECTION
# Mutual Information + Random Forest + Correlation
# ============================================================

print("=" * 70)
print("COMBINED FEATURE SELECTION")
print("=" * 70)


# ============================================================
# 1. Load Mutual Information results
# ============================================================

mi_file = "data/CICIDS2017_mutual_information.csv"

mi_df = pd.read_csv(mi_file)

mi_df = mi_df[
    ["Feature", "Mutual_Information"]
].copy()


print("\nMutual Information features:",
      len(mi_df))


# ============================================================
# 2. Load Random Forest results
# ============================================================

rf_file = "data/CICIDS2017_random_forest_importance.csv"

rf_df = pd.read_csv(rf_file)

rf_df = rf_df[
    ["Feature", "Importance"]
].copy()


print("Random Forest features:",
      len(rf_df))


# ============================================================
# 3. Merge MI + Random Forest
# ============================================================

combined_df = pd.merge(
    mi_df,
    rf_df,
    on="Feature",
    how="inner"
)


# ============================================================
# 4. Create rankings
# ============================================================

combined_df["MI_Rank"] = (
    combined_df["Mutual_Information"]
    .rank(
        ascending=False,
        method="min"
    )
)

combined_df["RF_Rank"] = (
    combined_df["Importance"]
    .rank(
        ascending=False,
        method="min"
    )
)


# ============================================================
# 5. Calculate combined rank
# ============================================================

combined_df["Combined_Rank"] = (
    combined_df["MI_Rank"] +
    combined_df["RF_Rank"]
)


# ============================================================
# 6. Calculate normalized scores
# ============================================================

mi_max = combined_df["Mutual_Information"].max()

rf_max = combined_df["Importance"].max()


combined_df["MI_Score"] = (
    combined_df["Mutual_Information"] /
    mi_max
)


combined_df["RF_Score"] = (
    combined_df["Importance"] /
    rf_max
)


# ============================================================
# 7. Final combined score
# ============================================================

combined_df["Combined_Score"] = (
    0.5 * combined_df["MI_Score"] +
    0.5 * combined_df["RF_Score"]
)


# ============================================================
# 8. Sort by combined score
# ============================================================

combined_df = combined_df.sort_values(
    "Combined_Score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 9. Assign final ranking
# ============================================================

combined_df["Final_Rank"] = (
    np.arange(len(combined_df)) + 1
)


# ============================================================
# 10. Display complete ranking
# ============================================================

print("\n" + "=" * 70)
print("COMBINED FEATURE RANKING")
print("=" * 70)

display_columns = [
    "Final_Rank",
    "Feature",
    "Mutual_Information",
    "Importance",
    "MI_Rank",
    "RF_Rank",
    "Combined_Score"
]

print(
    combined_df[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# 11. Top 30 features
# ============================================================

print("\n" + "=" * 70)
print("TOP 30 COMBINED FEATURES")
print("=" * 70)

print(
    combined_df[
        display_columns
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 12. Identify features appearing in both Top 30
# ============================================================

mi_top30 = set(
    mi_df
    .sort_values(
        "Mutual_Information",
        ascending=False
    )
    .head(30)["Feature"]
)


rf_top30 = set(
    rf_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(30)["Feature"]
)


both_top30 = sorted(
    mi_top30.intersection(rf_top30)
)


print("\n" + "=" * 70)
print("FEATURES PRESENT IN BOTH MI TOP 30 AND RF TOP 30")
print("=" * 70)

for i, feature in enumerate(
    both_top30,
    start=1
):

    print(
        f"{i:2d}. {feature}"
    )


print(
    "\nNumber of common Top-30 features:",
    len(both_top30)
)


# ============================================================
# 13. Add a flag for common Top 30
# ============================================================

combined_df["In_Both_Top30"] = (
    combined_df["Feature"]
    .isin(both_top30)
)


# ============================================================
# 14. Save complete combined ranking
# ============================================================

output_file = (
    "data/CICIDS2017_combined_feature_ranking.csv"
)

combined_df.to_csv(
    output_file,
    index=False
)

print("\nCombined ranking saved to:")
print(output_file)


# ============================================================
# 15. Save common Top 30 features
# ============================================================

common_df = combined_df[
    combined_df["In_Both_Top30"] == True
].copy()


common_output = (
    "data/CICIDS2017_common_top_features.csv"
)

common_df.to_csv(
    common_output,
    index=False
)

print("\nCommon Top-30 features saved to:")
print(common_output)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("COMBINED FEATURE SELECTION COMPLETED")
print("=" * 70)