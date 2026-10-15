import pandas as pd
import numpy as np

print("=" * 70)
print("NSL-KDD COMBINED FEATURE SELECTION")
print("=" * 70)

# ============================================================
# 1. LOAD MUTUAL INFORMATION RESULTS
# ============================================================

print("\nLoading Mutual Information results...")

mi_file = "data/NSLKDD_mutual_information.csv"
rf_file = "data/NSLKDD_random_forest_importance.csv"

mi_df = pd.read_csv(mi_file)
rf_df = pd.read_csv(rf_file)

print("Mutual Information features:", len(mi_df))
print("Random Forest features:", len(rf_df))

# ============================================================
# 2. PREPARE MUTUAL INFORMATION DATA
# ============================================================

mi_df = mi_df[[
    "Feature",
    "Mutual_Information",
    "MI_Rank"
]].copy()

# ============================================================
# 3. PREPARE RANDOM FOREST DATA
# ============================================================

rf_df = rf_df[[
    "Feature",
    "Importance",
    "RF_Rank"
]].copy()

# ============================================================
# 4. MERGE BOTH RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MERGING FEATURE SELECTION RESULTS")
print("=" * 70)

combined = pd.merge(
    mi_df,
    rf_df,
    on="Feature",
    how="inner"
)

print("\nCombined features:", len(combined))

# ============================================================
# 5. NORMALIZE MI SCORES
# ============================================================

mi_min = combined["Mutual_Information"].min()
mi_max = combined["Mutual_Information"].max()

if mi_max > mi_min:
    combined["MI_Normalized"] = (
        (combined["Mutual_Information"] - mi_min)
        / (mi_max - mi_min)
    )
else:
    combined["MI_Normalized"] = 0

# ============================================================
# 6. NORMALIZE RANDOM FOREST IMPORTANCE
# ============================================================

rf_min = combined["Importance"].min()
rf_max = combined["Importance"].max()

if rf_max > rf_min:
    combined["RF_Normalized"] = (
        (combined["Importance"] - rf_min)
        / (rf_max - rf_min)
    )
else:
    combined["RF_Normalized"] = 0

# ============================================================
# 7. CALCULATE COMBINED SCORE
# ============================================================

# Equal weight:
# 50% Mutual Information
# 50% Random Forest

combined["Combined_Score"] = (
    0.5 * combined["MI_Normalized"]
    + 0.5 * combined["RF_Normalized"]
)

# ============================================================
# 8. FINAL RANKING
# ============================================================

combined = combined.sort_values(
    by="Combined_Score",
    ascending=False
).reset_index(drop=True)

combined["Final_Rank"] = (
    combined.index + 1
)

# ============================================================
# 9. DISPLAY COMBINED RANKING
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
    combined[display_columns].to_string(index=False)
)

# ============================================================
# 10. TOP 30 FEATURES
# ============================================================

top_30 = combined.head(30)

print("\n" + "=" * 70)
print("TOP 30 COMBINED NSL-KDD FEATURES")
print("=" * 70)

print(
    top_30[display_columns].to_string(index=False)
)

# ============================================================
# 11. FEATURES PRESENT IN BOTH TOP 30
# ============================================================

mi_top_30 = set(
    mi_df.sort_values(
        by="MI_Rank"
    ).head(30)["Feature"]
)

rf_top_30 = set(
    rf_df.sort_values(
        by="RF_Rank"
    ).head(30)["Feature"]
)

common_features = sorted(
    mi_top_30.intersection(rf_top_30)
)

print("\n" + "=" * 70)
print("FEATURES PRESENT IN BOTH MI TOP 30 AND RF TOP 30")
print("=" * 70)

for i, feature in enumerate(common_features, 1):
    print(f"{i:2d}. {feature}")

print("\nNumber of common Top-30 features:", len(common_features))

# ============================================================
# 12. SAVE COMBINED RANKING
# ============================================================

combined_output = (
    "data/NSLKDD_combined_feature_ranking.csv"
)

combined.to_csv(
    combined_output,
    index=False
)

# ============================================================
# 13. SAVE COMMON FEATURES
# ============================================================

common_output = (
    "data/NSLKDD_common_top_features.csv"
)

common_df = pd.DataFrame({
    "Feature": common_features
})

common_df.to_csv(
    common_output,
    index=False
)

# ============================================================
# 14. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("NSL-KDD COMBINED FEATURE SELECTION COMPLETED")
print("=" * 70)

print("\nCombined ranking saved to:")
print(combined_output)

print("\nCommon Top-30 features saved to:")
print(common_output)

print("\nTotal features analyzed:", len(combined))
print("Top 30 features reported:", len(top_30))
print("Common Top-30 features:", len(common_features))