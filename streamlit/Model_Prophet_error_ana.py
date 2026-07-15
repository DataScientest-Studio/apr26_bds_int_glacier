import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import datetime

from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error,r2_score

#df_result = pd.read_csv(os.path.join(directory, 'preprocessed_meteohydrodata2026-05-26.csv'), index_col='TIMESTAMP')
df = pd.read_csv(r"C:\Users\HWHah\OneDrive\Desktop\Python_Prog\DS_Glacier\ML_Model\mod_feature_meteohydrodata_hourly.csv", index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

#Data split
# sort data acc. to time and remove time zone
df = df.sort_index()
df.index = df.index.tz_localize(None)


#Split acc. to fix dates
train_start_date = '2019-12-31'
train_end_date = '2023-04-30'

test_end_date = '2023-10-30'
test_end_dt = pd.to_datetime(test_end_date)

test_start_dt = pd.to_datetime(train_end_date) + pd.Timedelta(days=1)
test_start_date = test_start_dt.strftime('%Y-%m-%d')

#train and test data
train_df = df[train_start_date:train_end_date]
test_df = df[test_start_date:test_end_date]
train_df = train_df.drop(columns=['temp_lag_2', 'temp_lag_3', 'SWD_lag_3', 'SWD_lag_4', 'WS_lag_1', 'WS_lag_2', 'precip_lag_1'])
test_df = test_df.drop(columns=['temp_lag_2', 'temp_lag_3', 'SWD_lag_3', 'SWD_lag_4', 'WS_lag_1', 'WS_lag_2', 'precip_lag_1'])
#train_df = train_df.drop(columns=["melt"])
#train_df = train_df.drop(columns=["melt"])

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

# normalisation of the data (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Kontrolle für Ihr Team ---
print(f"Features für das Training (X_train) enthalten: {list(X_train.columns)}")
print(f"Target für das Training (y_train) ist die Spalte: {y_train.name}")

# ==========================================
# 1. Training-DataFrame for Prophet Setup
# ==========================================
# Aufbau des strukturierten Trainings-Dataframes inklusive exogener Features
df_train_prophet = pd.DataFrame({
    'ds': y_train.index,
    'y': y_train,                                          # Target (water_level)
    'summer_seasonality': X_train['summer_seasonality'],   # Exogenes Feature 1
    'hour_sin': X_train['hour_sin'],                       # Exogenes Feature 2
    'hour_cos': X_train['hour_cos']                        # Exogenes Feature 3
})

# KRITISCH: Zeitzonen entfernen, um Typen-Konflikte in Prophet zu vermeiden
df_train_prophet['ds'] = pd.to_datetime(df_train_prophet['ds']).dt.tz_localize(None)

# Sättigungsgrenzen (Cap/Floor) für das logistische Wachstum definieren
df_train_prophet['floor'] = 0
df_train_prophet['cap'] = df_train_prophet['y'].max() * 1.5  

# ==========================================
# 2. Model Definition & Training
# ==========================================
model = Prophet(
    growth='logistic',
    daily_seasonality=False,   # Deaktiviert, da wir unten eine hochauflösende hinzufügen
    yearly_seasonality=True,   # Gut funktionierende Jahressaisonalität
    changepoint_prior_scale=0.02,
    changepoint_range=0.80
)

# Tägliche Saisonalität hochdynamisch hinzufügen für scharfe Peak-Kurven
model.add_seasonality(
    name='daily_high_res',
    period=1,                  # 1 Tag (da Eingabedaten stündlich/täglich basieren)
    fourier_order=20,          # Erlaubt steile Peak-Kurven
    prior_scale=25.0           # Gibt der Saisonalität viel Freiheit gegenüber dem Gesamttrend
)

# Exogene Regressoren registrieren
model.add_regressor('summer_seasonality')
model.add_regressor('hour_sin')
model.add_regressor('hour_cos')

# Modell fitten
model.fit(df_train_prophet)

# ==========================================
# 3. Future-DataFrame Generation (Train + Test Combined)
# ==========================================
#  Wir führen X_train und X_test zusammen, um Lücken am Ende des Testzeitraums zu verhindern.
X_gesamt = pd.concat([X_train, X_test])

# 2. Wir ziehen den Datetime-Index heraus und erstellen die 'ds'-Spalte direkt daraus
# Das verhindert jegliche KeyErrors, egal ob der Index benannt ist oder nicht!
X_gesamt['ds'] = X_gesamt.index

# 3. Falls der Index kein Datetime-Index war, sondern eine normale Spalte existierte:
if 'ds' not in X_gesamt.columns:
    # Falls Ihre Zeitspalte in X_train z.B. 'date' hieß, aktivieren Sie diese Zeile:
    # X_gesamt = X_gesamt.rename(columns={'Hier_Ihre_Zeitspalte_Eintragen': 'ds'})
    pass

# 4. Jetzt existiert 'ds' garantiert. Wir entfernen die Zeitzone:
X_gesamt['ds'] = pd.to_datetime(X_gesamt['ds']).dt.tz_localize(None)

# 5. Nur die für das Modell relevanten Spalten extrahieren
future = X_gesamt[['ds', 'summer_seasonality', 'hour_sin', 'hour_cos']].copy()

# Sättigungsgrenzen exakt analog zum Training setzen
future['floor'] = 0
future['cap'] = df_train_prophet['y'].max() * 1.5 

# ==========================================
# 4. Predict & Align
# ==========================================
forecast = model.predict(future)

# Die Vorhersage 'forecast' enthält nun nahtlos alle Zeitstempel aus Train und Test.
# Sie können jetzt direkt mit dem zuvor erstellten Visualisierungs- und Fehlercode fortfahren.

# ==========================================
# Prediction Data Separation & Alignment
# ==========================================

# 1. Nutzung des echten Zeitstempels zur Ausrichtung (Verhindert Timeline-Shifts)
# Prophet gibt 'ds' als Spalte aus. Wir setzen sie als Index für das Matching.
forecast_indexed = forecast.set_index('ds')

# 2. Filtern der Vorhersagen exakt auf die Indizes (Zeitstempel) von y_train und y_test
y_train_pred = forecast_indexed.loc[y_train.index, 'yhat']
y_test_pred = forecast_indexed.loc[y_test.index, 'yhat']

# ==========================================
# Performance Evaluation
# ==========================================

# Berechnung der RMSE Metriken
rmse_train = root_mean_squared_error(y_train, y_train_pred)
rmse_test = root_mean_squared_error(y_test, y_test_pred)

# Berechnung der R² Metriken
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Saubere, strukturierte Ausgabe
print(f"--- Prophet Model Evaluation ---")
print(f"Train - RMSE: {rmse_train:.3f} | R²: {r2_train:.3f}")
print(f"Test  - RMSE: {rmse_test:.3f}  | R²: {r2_test:.3f}")

#==========================================
# 1. Visualization (Forecast Overview)
# ==========================================
plt.figure(figsize=(12, 6))

# 1. Plot training data (Historical Baseline)
plt.plot(y_train.index, y_train, 
         label='Historical Data (Train)', color='#1f77b4', linewidth=2)

# 2. Plot test data (Actual Measured Values)
plt.plot(y_test.index, y_test, 
         label='Actual Data (y_test)', color='#2ca02c', linewidth=2, alpha=0.6)

# 3. Plot prediction data (Model Forecast)
plt.plot(y_test_pred.index, y_test_pred, 
         label=f'Model Prediction (Forecast)\nRMSE = {rmse_test:.3f}, R² = {r2_test:.3f}', 
         color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.9)

# Chart layout and formatting
plt.title('Water Level Forecast with Distinct Data Phases', fontsize=12, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Water Level')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.show()


# ==========================================
# 2. Error Analysis (Peak Underestimations)
# ==========================================

# 1. Create analysis DataFrame using aligned indices
error_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred
}, index=y_test.index).sort_index()

# Calculate local deviation (Negative = Model underestimates the actual value/peak)
error_df['Deviation'] = error_df['Predicted'] - error_df['Actual']

# 2. Identify the top 5 largest underestimations (most negative deviations)
top_5_errors = error_df.sort_values(by='Deviation').head(5)

# 3. Initialize error plot
plt.figure(figsize=(15, 7))

# Plot time series for actual and predicted values over the test period
plt.plot(error_df.index, error_df['Actual'], label='Actual (Measured)', color='#1f77b4', alpha=0.6, linewidth=1.5)
plt.plot(error_df.index, error_df['Predicted'], label='Predicted (Model)', color='#ff7f0e', alpha=0.8, linewidth=1.5)

# 4. Highlight and annotate the top 5 peak errors
for idx, row in top_5_errors.iterrows():
    # Place a red dot on the actual peak value
    plt.scatter(idx, row['Actual'], color='red', s=60, zorder=5, edgecolors='black')
    
    # Create two-line label containing the actual value and local deviation
    label_text = f"Act: {row['Actual']:.1f}\nΔ: {row['Deviation']:.1f}"
    
    
# Configure text box position next to the marker
    plt.annotate(
        label_text, 
        xy=(idx, row['Actual']), 
        xytext=(15, 15),              # Slightly increased offset for better visibility
        textcoords='offset points', 
        fontsize=8, 
        fontweight='bold',
        color='red',
        arrowprops=dict(arrowstyle="->", color='red', lw=0.8), # Added an arrow pointing exactly to the peak
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
    )

# 5. Chart layout and formatting
plt.title(f"Glacier Modeling: Visualization of Maximum Peak Underestimations\nTest RMSE = {rmse_test:.3f}", fontsize=12, fontweight='bold')
plt.xlabel("Date / Time")
plt.ylabel("Water Level / Value")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left")

plt.tight_layout()
plt.show()