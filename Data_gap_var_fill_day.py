# First run comment lines 15 to 40 until CHECK
# Cope / Paste time data from console to start / end variables (lines #19/20
# Then successive elemination of gaps with data from the past LINE# 21
# Stop when gap size is sufficiant

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Read data
df = pd.read_csv('mod_meteohydrodata.csv',
                 index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

# Variable analysied and modified
var = 'SK_temperature'

# 1. Definition which time period should be recovered
start = '2013-01-01 00:02:30+00:00'
end  = '2015-07-22 11:52:30+00:00'

# 2. DF shifted by one day (or (5)year/-1825D with precipitation)
day_before = df[var].shift(freq='2D')
                           
# 3. Missing time period
time_period = (df.index >= start) & (df.index <= end)
print("--- Diese Zeilen werden überprüft ---")
print(df.loc[time_period, [var]])
# 4. Filling data with data from days before
df.loc[time_period, var] = (df.loc[time_period, var].fillna(day_before))
#df.loc[time_period, var] = (df.loc[time_period, var].fillna(0))
print(df.loc[time_period, [var]])

# 1. Den Index wieder in eine normale Spalte umwandeln
df_bereit = df.reset_index()

# 2. Die Spalte umbenennen (falls gewünscht, z.B. zurück zu 'Datum_Zeit')
df_bereit = df_bereit.rename(columns={'index': 'TIMESTAMP'}) 
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
