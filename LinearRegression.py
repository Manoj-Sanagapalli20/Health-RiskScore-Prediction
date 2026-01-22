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
features=["Age", "BMI", "HighBP", "HighChol","Smoker", "PhysActivity", "MentHlth", "PhysHlth"]
X = df[features]
y = df["health_risk_score"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1)
# -------------------------------------------------- SCALING 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)
# Use same scaler on test data
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
baseline_value = y_train.mean()
baseline_predictions = np.full(shape=len(y_test),fill_value=baseline_value)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
print("Baseline RMSE:", baseline_rmse)
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("R² Score:", r2)
coef_df = pd.DataFrame({"Feature": features,"Coefficient": model.coef_}).sort_values(by="Coefficient", ascending=False)
print(coef_df)
new_person = pd.DataFrame([{
    "Age": 45,
    "BMI": 31,
    "HighBP": 1,
    "HighChol": 1,
    "Smoker": 0,
    "PhysActivity": 1,
    "MentHlth": 5,
    "PhysHlth": 10
}])
new_person_scaled = scaler.transform(new_person)
predicted_risk = model.predict(new_person_scaled)
print("Predicted Health Risk Score:", predicted_risk[0])
############################################## OUTPUT VISUVALIZATIONS #############################################################
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red")
plt.xlabel("Actual Health Risk Score")
plt.ylabel("Predicted Health Risk Score")
plt.title("Actual vs Predicted Health Risk")
plt.show()
# 
residuals = y_test - y_test_pred
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30)
plt.xlabel("Residual (Error)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()
# 
plt.figure(figsize=(6, 4))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(0, color="red")
plt.xlabel("Predicted Health Risk")
plt.ylabel("Residual")
plt.title("Residuals vs Predictions")
plt.show()
