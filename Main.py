# Import required libraries
from hmac import digest_size

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset

data = pd.read_csv("LAEI_Final_Dataset.csv")
print(data.head())
print(data.info())

# Data Preprocessing
# Example: predicting PM2.5 Total emissions
target = 'pm25_total'

# Drop non-numeric or text columns (like Borough) and target itself from
X = data.drop(columns=['Borough',
target,
'pm25_concentration_mean'
])
y = data[target]

# Optional : Inspect columns used
print("Features used for prediction:\n", X.columns)

# Split dataset into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor

rf_model = RandomForestRegressor(
    n_estimators=200, # number of trees
    max_depth=5, # prevent overfitting for small dataset
    random_state=42
)
rf_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = rf_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Mean Squared Error (MSE) : {mse:.2f}")
print(f"Random Forest Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Random Forest R² score: {r2:.2f}")

# Feature importance
importances = rf_model.feature_importances_
features = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print("Feature Importance Ranking:\n", feature_importance_df)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual PM2.5 Total Emissions (tonnes)')
plt.ylabel('Predicted PM2.5 Total Emissions (tonnes)')
plt.title('Random Forest Predictions vs Actual PM2.5')
plt.show()
