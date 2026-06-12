import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==========================
# 1. Load Dataset
# ==========================
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully")
print("Original Shape:", df.shape)

# ==========================
# 2. Remove customerID
# ==========================
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# ==========================
# 3. Fix TotalCharges
# ==========================
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# ==========================
# 4. Encode ALL Object Columns
# ==========================
label_encoder = LabelEncoder()

# Encode all non-numeric columns
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = label_encoder.fit_transform(
            df[col].astype(str)
        )

# ==========================
# 5. Verify Data Types
# ==========================
print("\nData Types:")
print(df.dtypes)

# Check if any object columns remain
object_cols = df.select_dtypes(include=["object"]).columns

print("\nRemaining Object Columns:")
print(list(object_cols))

# ==========================
# 6. Save Cleaned Dataset
# ==========================
df.to_csv(
    "data/processed/cleaned_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("Location: data/processed/cleaned_data.csv")

print("\nFirst 5 Rows:")
print(df.head())