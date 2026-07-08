import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

#df_result = pd.read_csv(os.path.join(directory, 'preprocessed_meteohydrodata2026-05-26.csv'), index_col='TIMESTAMP')
df = pd.read_csv('mod_feature_meteohydrodata2026-06-19.csv', index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

# define reduced data set
df = df[["water_level",'summer_seasonality','lag_1']]
#normalization
df_norm = (df - df.min()) / (df.max() - df.min())

#Data split
# sorting acc. to time and remove time zone
df_norm = df_norm.sort_index()
df_norm.index = df_norm.index.tz_localize(None)

# Split acc. to fixed date
train_end_date = '2022-12-31'

train_df = df_norm[:train_end_date]  # Die ersten 10 Jahre (2013-2022)
test_df = df_norm[train_end_date:]   # Die letzten 2 Jahre (2023-2024)

target_col = "water_level"

# y ist die Zielspalte
y_test = test_df[target_col]
y_train = train_df[target_col]

# X sind alle Spalten, außer der Zielspalte
X_test = test_df.drop(columns=[target_col])
X_train = train_df.drop(columns=[target_col])


print(f"Training set: {train_df.index.min()} bis {train_df.index.max()} ({len(train_df)} Zeilen)")
print(f"Test set:     {test_df.index.min()} bis {test_df.index.max()} ({len(test_df)} Zeilen)")


# --- Kontrolle für Ihr Team ---
print(f"Features für das Training (X_train) enthalten: {list(X_train.columns)}")
print(f"Target für das Training (y_train) ist die Spalte: {y_train.name}")


# model impelentation
model = LinearRegression()
model.fit(X_train, y_train)


# model run
y_pred = model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
print(f'Erreur quadratique moyenne (RMSE) : {rmse}')


#Analysis

# generate series type
y_pred_series = pd.Series(y_pred, index=y_test.index)


plt.figure(figsize=(12, 6))

# 1. Train data (z.B. in Blau)
plt.plot(y_train.index, y_train, 
         label='Historical Data (Train)', color='#1f77b4', linewidth=2)

# 2. Prediction data (z.B. in Orange)
plt.plot(y_pred_series.index, y_pred_series, 
         label='Model Prediction (Forecast)', color='#ff7f0e', linestyle='--', linewidth=1, alpha=0.8)

plt.plot(y_test.index, y_test, 
         label='Actual Data (y_test)', color='#1f77b4', linewidth=2, alpha=0.5)


# . Diagram-Details 
plt.title('Water Level Forecast with Distinct Data Phases')
plt.xlabel('Date')
plt.ylabel('Water Level')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.show()

