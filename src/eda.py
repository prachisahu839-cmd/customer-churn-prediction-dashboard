import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

print("Dataset Shape:", df.shape)

# Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.savefig("visuals/churn_distribution.png")
plt.show()

print("Chart saved successfully!")