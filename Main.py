# Import required libraries
from hmac import digest_size

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset

data = pd.read_csv("LAEI_Final_Dataset.csv")
print(data.head())
print(data.info())

# Data Preprocessing
# Example: predicting PM2.5 Total emissions
target = 'PM25_tonnes'
X = data.drop(columns=['PM25_tonnes','PM25_log','Borough','Zone','Source_Category','Sector','Grid ID 2019'], errors='ignore')
y = data[target]
print("Numeric features used for prediction:\n", X.columns)

# Split dataset into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor

rf_model = RandomForestRegressor(
    n_estimators=200, # number of trees
    max_depth=5, # prevents overfitting
    random_state=42
)
rf_model.fit(X_train, y_train)

# Make predictions on test set
y_pred_raw = rf_model.predict(X_test)

# Physical constraint: emissions can't be negative, cap slightly above training max
y_pred = np.clip(y_pred_raw, 0.0, y_train.max()*1.10)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Random Forest Results (test set, with physical constraints):")
print(f"{'Metric':<12} {'PM2.5':>10}")
print(f"{'R²':<12} {r2:>10.4f}")
print(f"{'RMSE':<12} {rmse:>10.4f}")
print(f"{'MAE':<12} {mae:>10.4f}")



# Feature importance
importances = rf_model.feature_importances_
features = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("/nTop 10 important features:")
print(feature_importance_df.head(10))

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual PM2.5 Total Emissions (tonnes)')
plt.ylabel('Predicted PM2.5 Total Emissions (tonnes)')
plt.title('Random Forest Predictions vs Actual PM2.5')
plt.show()

# Feature Importance Plot
plt.figure(figsize=(10,6))
sns.barplot(
    x='Importance', y='Feature',
    data=feature_importance_df.head(20),
    palette='viridis'
)
plt.title('Random Forest Feature Importance (Top 20)')
plt.show()

