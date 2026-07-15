# Parameter optimization SAMIRAX
import numpy as np
import pandas as pd
import statsmodels.api as sm
import itertools
from sklearn.metrics import root_mean_squared_error, r2_score

#Data import
df = pd.read_csv(r"C:\Users\HWHah\OneDrive\Desktop\Python_Prog\DS_Glacier\ML_Model\mod_feature_meteohydrodata_hourly.csv", index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)
#df = df.asfreq('D')

# Data splitting
#Split acc. to fix dates
train_start_date = '2019-12-31'
train_end_date = '2023-04-30'

test_end_date = '2023-10-30'
test_end_dt = pd.to_datetime(test_end_date)

test_start_dt = pd.to_datetime(train_end_date) + pd.Timedelta(days=1)
test_start_date = test_start_dt.strftime('%Y-%m-%d')

#Define train and test data
train_df = df[train_start_date:train_end_date]
test_df = df[test_start_date:test_end_date]

print(f"Training set: {train_df.index.min()} bis {train_df.index.max()} ({len(train_df)} Zeilen)")
print(f"Test set:     {test_df.index.min()} bis {test_df.index.max()} ({len(test_df)} Zeilen)")

#Target and discibtive variables
target_col = "water_level"

# y is target (only variable for this model)
y_test = test_df[target_col]
y_train = train_df[target_col]

# Datumsfenster aus Ihrem Code extrahieren
start_date = y_test.index[0]
end_date = y_test.index[-1]

# --- DATEN-CHECK VORAB ---
# Sicherstellen, dass der Index eine saubere Frequenz hat, um Verschiebungen zu vermeiden
if y_train.index.freq is None:
    inferred_freq = pd.infer_freq(y_train.index)
    y_train.index.freq = inferred_freq
    y_test.index.freq = inferred_freq

# Start- und Enddatum für die Vorhersage definieren
start_date = y_test.index[0]
end_date = y_test.index[-1]

# --- DATEN-CHECK FÜR STÜNDLICHE DATEN ---
if y_train.index.freq is None:
    y_train.index.freq = 'H'  # 'H' erzwingt stündliche Frequenz
    y_test.index.freq = 'H'

start_date = y_test.index[0]
end_date = y_test.index[-1]

# --- SUCHRAUM FÜR STÜNDLICHE DATEN ---
# Da s=168 zu groß für 'seasonal_order' ist, fangen wir die stündliche und 
# tägliche Dynamik über die klassischen p- und q-Lags ab.
p = [1, 2, 24]  # 24 fängt den exakten Wert vom Vortag zur selben Stunde ab
d = [0, 1]
q = [0, 1, 2]

pdq_combinations = list(itertools.product(p, d, q))
results_list = []

print(f"Starte stündliche Optimierung über {len(pdq_combinations)} Kombinationen...")

for param in pdq_combinations:
    try:
        # Wir nutzen SARIMAX ohne seasonal_order, bauen aber 24-Stunden-Lags über 'p' auf
        model = sm.tsa.statespace.SARIMAX(
            y_train,  
            order=param,
            seasonal_order=(0, 0, 0, 0),  # Deaktiviert, da s=168 mathematisch zu groß ist
            trend='c',
            enforce_stationarity=True,
            enforce_invertibility=True
        )
        sarima = model.fit(disp=False, method='lbfgs')
        
        # Vorhersage generieren
        y_test_pred = sarima.predict(start=start_date, end=end_date)
        
        if np.isnan(y_test_pred).any():
            continue
            
        # Metriken berechnen
        r2_test = r2_score(y_test, y_test_pred)
        rmse_test = root_mean_squared_error(y_test, y_test_pred)
        
        # Speichern, wenn das Modell besser als der Mittelwert ist
        if r2_test > 0:
            results_list.append({
                'p,d,q': param,
                'R2_Test': r2_test,
                'RMSE_Test': rmse_test
            })
            
    except Exception as e:
        continue

# --- ERGEBNISSE AUSGEBEN ---
df_search = pd.DataFrame(results_list)

if not df_search.empty:
    df_search = df_search.sort_values(by='R2_Test', ascending=False)
    print("\n--- Gefundene stündliche Parameter sortiert nach bestem Test-R² ---")
    print(df_search.to_string(index=False))
else:
    print("\nKein stündliches Modell mit positivem R² gefunden.")
    print("Hinweis: Bei stündlichen Daten über ein halbes Jahr Testzeitraum (ca. 4380 Schritte) "
          "ohne exogene Features verpufft die Wirkung von ARIMAX-Lags nach wenigen Tagen.")