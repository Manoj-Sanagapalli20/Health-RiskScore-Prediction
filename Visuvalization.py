import pandas as pd
import numpy as np
df = pd.read_csv("healthdataset.csv")
df["health_risk_score"] = (
    0.03 * df["Age"] +
    0.04 * df["BMI"] +
    1.5  * df["HighBP"] +
    1.3  * df["HighChol"] +
    1.8  * df["Smoker"] -
    1.2  * df["PhysActivity"] +
    0.05 * df["MentHlth"] +
    0.06 * df["PhysHlth"] +
    np.random.normal(0, 1, len(df)) #making data real noisy
)
# EDA
# 1)IS THE DATA REALISTIC OR NOT 
import matplotlib.pyplot as plt
X = df[
    ["Age", "BMI", "HighBP", "HighChol", "Smoker",
     "PhysActivity", "MentHlth", "PhysHlth"]
]
X.hist(figsize=(12, 10), bins=30)
# 2) IS THE TARGET SUITABLE FOR PERFORMING THE REGRESSION
plt.suptitle("Distribution of Health Indicators")
plt.show()
plt.figure(figsize=(6,4))
plt.hist(df["health_risk_score"], bins=30)
plt.title("Health Risk Score Distribution")
plt.xlabel("Health Risk Score")
plt.ylabel("Count")
plt.show()
# 3)TO KNOW WHICH INPUT IS EFFECTING HS HOW LIKE 
# COLLERATION - To have the values of effection 
correlation = df[
    ["Age", "BMI", "HighBP", "HighChol", "Smoker",
     "PhysActivity", "MentHlth", "PhysHlth", "health_risk_score"]].corr()
print(correlation["health_risk_score"].sort_values(ascending=False))
# HEAPMAP - to viaualze the colleration by graph 
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
sns.heatmap(df["Age", "BMI", "HighBP", "HighChol", "Smoker","PhysActivity", "MentHlth", "PhysHlth", "health_risk_score"].corr(),annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
