
# ============================================================
# CRISP-DM: 50 Startups Profit Prediction
# Algorithm: Multiple Linear Regression
# Library: scikit-learn
# ============================================================

# 0. Install if needed:
# pip install pandas numpy scikit-learn matplotlib joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ------------------------------------------------------------
# Gate 1: Business Understanding
# ------------------------------------------------------------
# Objective: Predict startup profit using company spending
# and state.
#
# Target: Profit
# Features: R&D Spend, Administration, Marketing Spend, State
# Success metrics: R2, RMSE, MAE

# ------------------------------------------------------------
# Gate 2: Data Understanding
# ------------------------------------------------------------

df = pd.read_csv("50_Startups.csv")

# Normalize column names to handle accidental extra spaces
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types and non-null counts:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())
print("\nStatistical summary:")
print(df.describe(include="all"))

# Confirm expected columns exist
expected_columns = [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State",
    "Profit"
]

missing_columns = set(expected_columns) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Remove rows with missing target because the model
# cannot train without known Profit values.
df = df.dropna(subset=["Profit"]).copy()

# ------------------------------------------------------------
# Gate 3: Data Preparation
# ------------------------------------------------------------

X = df[[
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State"
]]

y = df["Profit"]

numeric_features = [
    "R&D Spend",
    "Administration",
    "Marketing Spend"
]

categorical_features = ["State"]

# Numeric data: fill missing values with the median
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical data: fill missing values with the most
# frequent category, then convert categories to numbers
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Apply appropriate preprocessing to each column type
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Split BEFORE fitting preprocessing to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# Gate 4: Modeling
# ------------------------------------------------------------

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Fit preprocessing and model using training data only
model.fit(X_train, y_train)

print("\nModel training completed.")

# ------------------------------------------------------------
# Gate 5: Evaluation
# ------------------------------------------------------------

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== Test Set Evaluation =====")
print(f"R^2  : {r2:.4f}")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")

# Compare actual and predicted profits
results = pd.DataFrame({
    "Actual Profit": y_test,
    "Predicted Profit": y_pred,
    "Residual": y_test.to_numpy() - y_pred
})

print("\nPrediction comparison:")
print(results.round(2).to_string(index=False))

results.to_csv("test_predictions.csv", index=False)

# Plot actual vs predicted values
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# Residual plot: look for systematic patterns
residuals = y_test.to_numpy() - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Profit")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150)
plt.show()

# Inspect coefficients after preprocessing
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = model.named_steps[
    "regressor"
].coef_

coef_table = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

print("\nRegression coefficients:")
print(coef_table.to_string(index=False))

print(
    "\nIntercept:",
    round(model.named_steps["regressor"].intercept_, 2)
)

# ------------------------------------------------------------
# Gate 6: Deployment
# ------------------------------------------------------------

# Save the entire pipeline, including preprocessing
joblib.dump(model, "startup_profit_model.joblib")

print("\nModel saved to startup_profit_model.joblib")
print("Predictions saved to test_predictions.csv")

# Example: predict profit for a new startup
new_startup = pd.DataFrame([{
    "R&D Spend": 100000,
    "Administration": 120000,
    "Marketing Spend": 150000,
    "State": "California"
}])

predicted_profit = model.predict(new_startup)[0]

print(
    f"\nPredicted profit for new startup: "
    f"{predicted_profit:,.2f}"
)
