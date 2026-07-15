import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Daten auf Tagesmittelwerte aggregieren (Resampling)
# Das reduziert die 1,2 Millionen Punkte auf ca. 4.380 Punkte.
# Voraussetzung: Der Index muss ein valider DatetimeIndex sein.
ts_daily = df_result_grouped_year['Lufttemperatur'].resample('D').mean()

# Falls es kleine Lücken im Datensatz gibt, füllen wir diese linear auf
ts_daily = ts_daily.interpolate(method='linear')
 2. Zerlegung für den Jahresverlauf durchführen
# Da wir nun Tagesdaten haben, entspricht ein Jahreszyklus period=365
result_longterm = seasonal_decompose(ts_daily, model='additive', period=365)

# 3. Visualisierung des reinen Klimatrends
fig, ax = plt.subplots(figsize=(12, 5))

# Original-Tageswerte im Hintergrund (leicht transparent)
ax.plot(ts_daily.index, ts_daily.values, color='gray',
        alpha=0.3, label='Tagesmittelwerte')

# Der isolierte, langfristige Klimatrend (ohne Jahreszeiten-Effekt)
ax.plot(result_longterm.trend.index, result_longterm.trend.values, color='red',
        linewidth=2.5, label='Langfristiger Klimatrend')
# Grafik-Styling (Einheitliche Beschriftungen)
ax.tick_params(axis='both', labelsize=10)
ax.set_title('12-Jähriger Klimatrend (Bereinigt um Saisonalität)', fontsize=14)
ax.set_ylabel('Temperatur (°C)', fontsize=11)
ax.set_xlabel('Jahr', fontsize=11)
ax.grid(True, alpha=0.3)        
