import pandas as pd
import glob
import os

input_folder = "data/CICIDS2017"

files = glob.glob(os.path.join(input_folder, "*.csv"))

print("Total files:", len(files))

dataframes = []

for file in files:
    print("Reading:", file)

    df = pd.read_csv(file)

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    dataframes.append(df)

cicids = pd.concat(dataframes, ignore_index=True)

print("Final shape:", cicids.shape)

cicids.to_csv(
    "data/CICIDS2017_merged.csv",
    index=False
)

print("Merge completed!")
