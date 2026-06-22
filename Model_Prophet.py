import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime

from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

#df_result = pd.read_csv(os.path.join(directory, 'preprocessed_meteohydrodata2026-05-26.csv'), index_col='TIMESTAMP')
df = pd.read_csv('mod_feature_meteohydrodata2026-06-19.csv', index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

#normalization
df_norm = (df - df.min()) / (df.max() - df.min())

#Data split
# sort data acc. to time and remove time zone
df_norm = df_norm.sort_index()
df_norm.index = df_norm.index.tz_localize(None)

#Split acc. to fix dates
train_start_date = '2020-01-01'
train_end_date = '2023-10-31'
test_end_date = '2024-10-30'
test_end_dt = pd.to_datetime(test_end_date)

test_start_dt = pd.to_datetime(train_end_date) + pd.Timedelta(days=1)
test_start_date = test_start_dt.strftime('%Y-%m-%d')

#train and test data
train_df = df_norm[train_start_date:train_end_date]
test_df = df_norm[test_start_date:test_end_date]

print(f"Training set: {train_df.index.min()} bis {train_df.index.max()} ({len(train_df)} Zeilen)")
print(f"Test set:     {test_df.index.min()} bis {test_df.index.max()} ({len(test_df)} Zeilen)")

#Target and discibtive variables
target_col = "water_level"

# y is target
y_test = test_df[target_col]
y_train = train_df[target_col]

# X are exploratory variables
X_test = test_df.drop(columns=[target_col])
X_train = train_df.drop(columns=[target_col])

# --- Kontrolle für Ihr Team ---
print(f"Features für das Training (X_train) enthalten: {list(X_train.columns)}")
print(f"Target für das Training (y_train) ist die Spalte: {y_train.name}")

# Training-DataFrame for Prophet: fix format for Model
df_train = pd.DataFrame({"ds": y_train.index, "y": y_train.values})
df_train["ds"] = pd.to_datetime(df_train["ds"]).dt.tz_localize(None)

# implement model
df_train_prophet = pd.DataFrame({
    'ds': train_df.index,
    'y': train_df['water_level'],       # Target
    'lag_1': train_df['lag_1'],   #  exogene Feature
    'summer_seasonality': train_df['summer_seasonality']   #  exogene Feature
})
df_train_prophet['ds'] = pd.to_datetime(df_train_prophet['ds']).dt.tz_localize(None)


# run model
model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
model.add_seasonality(name='yearly', period=365.25, fourier_order=15)
# add regressors
model.add_regressor('lag_1')
model.add_regressor('summer_seasonality')

model.fit(df_train_prophet)

#Define predection time period
tage_bis_ende = (test_end_dt - df_train['ds'].max()).days

# no overlap train and test period
if tage_bis_ende <= 0:
    raise ValueError("Das gewünschte Enddatum liegt vor oder auf dem Ende der Trainingsdaten!")

future = model.make_future_dataframe(periods=tage_bis_ende, freq='D')
temp_gesamt = pd.concat([train_df['lag_1'], test_df['lag_1']])
future['lag_1'] = future['ds'].map(temp_gesamt)
temp_gesamt = pd.concat([train_df['summer_seasonality'], test_df['summer_seasonality']])
future['summer_seasonality'] = future['ds'].map(temp_gesamt)
forecast = model.predict(future)

# Prediction data seperation
y_pred = forecast['yhat'].iloc[-tage_bis_ende:]                     

# RMSE 
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Prophet Modell RMSE: {rmse:.2f}")


#Analysis
plt.figure(figsize=(12, 6))

# 1. Tatsächliche Daten plotten
# WICHTIG: Nutze index und values vom SELBEN gekürzten Objekt (y_test_truncated)
plt.plot(
    y_test.index, 
    y_test.values, 
    label='Tatsächliche Daten', 
    color='blue', 
    alpha=0.7
)

# 2. Prophet Vorhersage plotten
# WICHTIG: Nutze 'ds' und 'yhat' aus dem SELBEN gekürzten DataFrame (test_forecast)
plt.plot(
    forecast['ds'], 
    forecast['yhat'], 
    label='Prophet Vorhersage', 
    color='orange',
    alpha=0.5,
    linewidth=2
)

plt.title("Meteorologische Vorhersage - Eingegrenzter Testzeitraum")
plt.xlabel("Datum")
plt.ylabel("Messwert")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
