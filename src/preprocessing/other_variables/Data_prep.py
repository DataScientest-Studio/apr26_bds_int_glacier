# First run comment lines 15 to 40 until CHECK
# Cope / Paste time data from console to start / end variables (lines #19/20
# Then successive elemination of gaps with data from the past LINE# 21
# Stop when gap size is sufficiant

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Read data
#dateipfad =r"C:\Users\HWHah\OneDrive\Desktop\Python_Prog\DS_Glacier/preprocessed_meteohydrodata2026-05-26.csv"
df = pd.read_csv('mod_meteohydrodata.csv',
                 index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

# Generate old status
#df_pn = pd.read_csv('preprocessed_meteohydrodata2026-05-26.csv,
#                    index_col='TIMESTAMP')
#df_pn.index = pd.to_datetime(df.index)

#df['PM_precipitation_cum'] = df_pn['PM_precipitation_cum']

# Variable analysied and modified
var = 'PM_atmospheric_pressure'

# 1. Definition which time period should be recovered
start = '2017-08-15 18:17:30+00:00'
end  = '2017-08-16 06:42:30+00:00'

# 2. DF shifted by one day (or (5)year/-1825D with precipitation)
ay_before = df[var].shift(freq='2D')
                           
# 3. Missing time period
time_period = (df.index >= start) & (df.index <= end)

#PLOT
#x_werte = df.index[(df.index >= start) & (df.index <= end)]
#y_werte = day_before[(day_before.index >= start) & (day_before.index <= end)]

# 4. Plot erstellen
#plt.figure(figsize=(10, 5))
#plt.plot(
#    x_werte, y_werte, label=f"{var} (day_before)"
#)  # Nutzt echten Zeitindex für X und Werte für Y

# Formatierung
#plt.xlabel("time")
#plt.ylabel(var)
#plt.title(f"Course from {start} to {end}")
#plt.xticks(rotation=35)
#plt.grid(True)
#plt.legend()

#plt.tight_layout()
#plt.show()  # <-- Die Klammern () sind zwingend notwendig!

print("--- Diese Zeilen werden überprüft ---")
print(df.loc[time_period, [var]])
# 4. Filling data with data from days before
#df.loc[time_period, var] = (df.loc[time_period, var].interpolate(method='time',
#                                                        limit_direction='both'))

#df.loc[time_period, var] = df.loc[time_period, var].ffill()
df.loc[time_period, var] = (df.loc[time_period, var].fillna(day_before))
#df.loc[time_period, var] = (df.loc[time_period, var].fillna(0))
#df.loc[time_period, var] = np.nan
#print(df.loc[time_period, [var]])

#df = df.ffill()
#df[var] = df[var] + 700.0 # atmothpric pressure correction for the glacier height

#Save modified data to new csv file
# 1. Den Index wieder in eine normale Spalte umwandeln
df_bereit = df.reset_index()

# 2. Die Spalte umbenennen (falls gewünscht, z.B. zurück zu 'Datum_Zeit')
#df_bereit = df_bereit.rename(columns={'index': 'TIMESTAMP'})
if "index" in df_bereit.columns:
    df_bereit.rename(columns={"index": "TIMESTAMP"}, inplace=True)
# Hinweis: Wenn Ihr Index vorher einen Namen hatte, müssen Sie 'index' durch diesen Namen ersetzen.

# 3. Als CSV-Datei speichern (jetzt mit index=False, da das Datum eine eigene Spalte ist!)
df_bereit.to_csv('mod_meteohydrodata.csv', encoding='utf-8', index=False)
#df.to_csv('mod_meteohydrodata.csv', encoding='utf-8', index=True)

#CHECK
df_result = df
# Find gaps
gesamte_luecken = df_result[var].isnull().sum()
prozent = (gesamte_luecken / len(df_result)) * 100

print(f"Lücken gesamt: {gesamte_luecken} Zeilen ({prozent:.2f}%)")

# Erstellt eine Serie, die True bei Lücken (NaN) enthält
ist_luecke = df_result[var].isnull()
luecken_id = (ist_luecke != ist_luecke.shift()).cumsum()

# Gruppiert aufeinanderfolgende True-Werte (Lücken-Blöcke)
luecken_bloecke = (ist_luecke != ist_luecke.shift()).cumsum()

# 3. Nur die echten Lücken-Blöcke betrachten und deren Zeilen zählen
luecken_stats = df_result[ist_luecke].groupby(luecken_id)

# 4. Den Block mit den meisten fehlenden Zeilen ermitteln
groesste_luecke_id = luecken_stats.size().idxmax()
groesste_luecke_daten = luecken_stats.get_group(groesste_luecke_id)

# 5. Auswertung berechnen
anzahl_punkte = len(groesste_luecke_daten)
dauer_minuten = anzahl_punkte * 5  # 5-Minuten-Takt
dauer_tage = dauer_minuten / (24 * 60)

start_zeit = groesste_luecke_daten.index.min()
end_zeit = groesste_luecke_daten.index.max()
# Ergebnis-Ausgabe
print("--- GRÖSSTE LÜCKE IM DATENSATZ ---")
print(f"Startet am:        {start_zeit}")
print(f"Endet am:          {end_zeit}")
print(f"Fehlende Messwerte: {anzahl_punkte} Punkte am Stück")
print(f"Dauer der Lücke:    {dauer_minuten} Minuten (~ {dauer_tage:.1f} Tage)")

# Zählt, wie viele Zeilen in jedem Lücken-Block stecken
luecken_groessen = ist_luecke.groupby(luecken_bloecke).sum()

# Filtere Blöcke heraus, die keine Lücken waren (Größe 0)
nur_echte_luecken = luecken_groessen[luecken_groessen > 0]

# Umrechnung in Minuten (da 1 Zeile = 5 Minuten entspricht)
luecken_in_minuten = nur_echte_luecken * 5

# Statistische Übersicht der Lückengrößen anzeigen
print(luecken_in_minuten.describe())

df_luecken = df_result[ist_luecke]
print("Beispielhafte Zeitpunkte mit fehlenden Werten:")
print(df_luecken.index[:5])
