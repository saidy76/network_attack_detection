import pandas as pd
import numpy as np

# Load cleaned dataset
input_file = "data/CICIDS2017_clean.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("CHECKING INVALID NUMERICAL VALUES")
print("=" * 60)

print("Dataset shape:", df.shape)


# ==========================================
# 1. Check negative values
# ==========================================

print("\n" + "=" * 60)
print("NEGATIVE VALUES")
print("=" * 60)

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:

    if column == "Label":
        continue

    negative_count = (df[column] < 0).sum()

    if negative_count > 0:
        print(f"{column}: {negative_count}")


# ==========================================
# 2. Check zero-heavy features
# ==========================================

print("\n" + "=" * 60)
print("ZERO VALUE ANALYSIS")
print("=" * 60)

for column in numeric_columns:

    if column == "Label":
        continue

    zero_count = (df[column] == 0).sum()
    zero_percentage = (zero_count / len(df)) * 100

    if zero_percentage >= 95:
        print(
            f"{column}: "
            f"{zero_count} zeros "
            f"({zero_percentage:.2f}%)"
        )


# ==========================================
# 3. Check very large values
# ==========================================

print("\n" + "=" * 60)
print("EXTREME VALUES")
print("=" * 60)

for column in numeric_columns:

    if column == "Label":
        continue

    max_value = df[column].max()

    print(f"{column}: max = {max_value}")


print("\n" + "=" * 60)
print("INVALID VALUE CHECK COMPLETED")
print("=" * 60)