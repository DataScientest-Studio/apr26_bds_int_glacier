import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Read data
df = pd.read_csv('preprocessed_meteohydrodata2026-06-04.csv',
                 index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

#df = df.ffill()
#diff = df['PM_temperature'].pct_change()
#diff = df['PM_temperature'].interpolate().pct_change(fill_method=None)
diff_abs_t = df['PM_temperature'].diff()
diff_abs_p= df['PM_atmospheric_pressure'].diff()
diff_abs_h= df['PM_relative_humidity'].diff()
diff_abs_s= df['PM_snow_height'].diff()
diff_abs_ws= df['PM_wind_speed'].diff()
diff_abs_wd= df['PM_wind_direction'].diff()
diff_abs_Precip= df['PM_precipitation'].diff()
diff_abs_Precip_cum= df['PM_precipitation_cum'].diff()
diff_abs_SWD= df['PM_SWD'].diff()
diff_abs_SWU= df['PM_SWU'].diff()

# PLOT
y = pd.DataFrame()
x = df.index
y['1'] = diff_abs_t
y['2'] = diff_abs_p
y['3'] = diff_abs_h
y['4'] = diff_abs_s
y['5'] = diff_abs_ws
y['6'] = diff_abs_wd
y['7'] = diff_abs_Precip
y['8'] = diff_abs_Precip_cum
y['9'] = diff_abs_SWD
y['10'] = diff_abs_SWU
#y['1']= df['PM_temperature
#y['3']= df['PM_relative_humidity']
#y['4']= df['PM_snow_height']
#y['5']= df['PM_wind_speed']
#y['6']= df['PM_wind_direction']
#y['7']= df['PM_precipitation']
#y['8']= df['PM_precipitation_cum']
#y['9']= df['PM_SWD']
#y['10']= df['PM_SWU']

# 1. Liste mit den echten Namen Ihrer Variablen definieren (Reihenfolge passend zu 1-8)
variablen_namen = [
    "Temperature [°C]",
    "Atmospheric pressure [hPa]",
    "Relative humidity [%]",
    "Snow height [m]", 
    "Wind speed [m/s]",
    "Wind direction [°]", # Für Plot 6 (Index 5)
    "Precipitation [mm]",
    "Precipitation_cum [mm]", 
    "Radiation D [W/m²]",
    "Radiation U [W/m²]",
]

# Eine "Figure" (Leinwand) und ein 4x2 Raster aus "Axes" (Plots) erstellen
#fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 22))  # Höhe leicht erhöht
fig, axes = plt.subplots(5, 2, figsize=(15, 15), constrained_layout=True)
axes_flat = axes.flatten()

# Das 2D-Array der Axes flachdrücken (1D), um einfacher durchzuloopen
axes_flat = axes.flatten()

# Die 10 Plots in einer Schleife befüllen
for i in range(10):

    spalten_schluessel = str(i + 1)
    y_data = y[spalten_schluessel]

    # Auf das jeweilige Unterdiagramm zeichnen
    if i == 5:
        axes_flat[i].scatter(x, y_data, c="blue", alpha=0.01, marker=".", s=1)
    else:
        axes_flat[i].plot(x, y_data, color=f"C{i}", linewidth=1)

    # DYNAMISCHE ÜBERSCHRIFT: Holt den Namen aus der Liste basierend auf dem Index i
    axes_flat[i].set_title(variablen_namen[i], fontsize=11, fontweight="bold")
    axes_flat[i].grid(True)
#X-Achsen-Beschriftung lesbar machen
    axes_flat[i].tick_params(axis="x", rotation=30)

# Manuelle, großzügige Abstände setzen (ersetzt tight_layout)
plt.subplots_adjust(hspace=0.5, wspace=0.25)
# Layout optimieren, damit sich Beschriftungen nicht überschneiden
#plt.tight_layout()

# Diagramm anzeigen
plt.show()



