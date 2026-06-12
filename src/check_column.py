import pandas as pd

df = pd.read_csv("data/processed/cleaned_data.csv")

print(df.columns.tolist())
print("Total Columns:", len(df.columns))