import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Read data
df_result = pd.read_csv('meteohydrodata.csv', index_col='TIMESTAMP')
df_result.index = pd.to_datetime(df_result.index)

# Group by index month and calculate the average
#monats_mittelwerte = df_result.groupby(df_result.index.month).mean()

# Optional: Rename Index 
#monats_mittelwerte.index.name = "Monat"

# Mean for every month in each year ('ME' Month End)
chronologisch_mittel = df_result.resample("ME").mean()

# Forward fill (ffill) with last value before gap
chronologisch_mittel["water_level_ffill"] = chronologisch_mittel["water_level"].ffill()

x = chronologisch_mittel.index
y1 = chronologisch_mittel['water_level_ffill']
y2 = chronologisch_mittel['water_level']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 2. Erstes Diagramm (links) konfigurieren
ax1.plot(x, y1, color='blue')
ax1.set_title('Water Level Forward Fill')
ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

# 3. Zweites Diagramm (rechts) konfigurieren
ax2.plot(x, y2, color='orange')
ax2.set_title('Water Level')
ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

plt.show()


