# -*- coding: utf-8 -*-
"""
Read docs.md for an in-depth explanation of the practice.

P8: Forecasting

Linear Forecasting.

Create a model using linear regression to predict future values, using a time
series.

Forecast the number of reviews per week using linear regression, 
then evaluate performance with MSE, MAE, and MAPE.

Martín Alexis Martínez Andrade
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../Dataset/modified_clean_data.csv")

# use Date column as datetime type
df["Date"] = pd.to_datetime(df["Date"])

# number of reviews per week
df['Week'] = df['Date'].dt.to_period('W')
# group by week
weekly_reviews = df.groupby('Week').size().reset_index(name='NumReviews')

# ensure correct chronological order
weekly_reviews = weekly_reviews.sort_values('Week').reset_index(drop=True)
weekly_reviews['WeekNum'] = np.arange(len(weekly_reviews))

# define features and target
X = weekly_reviews['WeekNum'].values.reshape(-1, 1) # 2D array with weeks x 1 shape
y = weekly_reviews['NumReviews'].values

# 80% train, 20% test
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

lr = LinearRegression().fit(X_train, y_train)
print("Regression coefficients:")
print(f"\tSlope: {lr.coef_[0]:.2f}, Intercept: {lr.intercept_:.2f}")

# forecast for the test set
y_pred = lr.predict(X_test)

# metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\nModel evaluation on test set:")
print(f"\tMean Squared Error (MSE): {mse:.2f}")
print(f"\tMean Absolute Error (MAE): {mae:.2f}")
print(f"\tMean Absolute Percentage Error (MAPE): {mape:.2f}%")

# forecast future weeks
future_weeks = np.arange(len(weekly_reviews), len(weekly_reviews)+20).reshape(-1, 1)
future_preds = lr.predict(future_weeks)
print("\nForecast for the next 20 weeks:")
for i, pred in enumerate(future_preds, 1):
    print(f"\tWeek {len(weekly_reviews) + i} (last +{i}): {pred:.2f} reviews (predicted)")

# plot
figure, ax = plt.subplots()
# actual values
ax.plot(weekly_reviews['WeekNum'], y, label='Actual')
# predicted values in test subset
ax.plot(weekly_reviews['WeekNum'].iloc[split_idx:], y_pred, label='Predicted (test)', linestyle='--')
# forecasted values in future weeks not present in the dataset
ax.plot(future_weeks.reshape(-1), future_preds, label="Forecast")

# if X ticks are kept, the labels look like a long black bar below the X axis
ax.set_xticks([])

plt.xlabel('Week')
plt.ylabel('Number of reviews')
plt.title('Weekly Reviews Forecasting with Linear Regression')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("Forecast.png")