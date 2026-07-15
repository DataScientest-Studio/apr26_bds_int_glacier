import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#df_result = pd.read_csv(os.path.join(directory, 'preprocessed_meteohydrodata2026-05-26.csv'), index_col='TIMESTAMP')
df = pd.read_csv(r'C:\Users\HWHah\OneDrive\Desktop\Python_Prog\DS_Glacier/preprocessed_meteohydrodata2026-06-04.csv', index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

# remove SK data
df.drop(columns=['SK_temperature', 'SK_relative_humidity', 'SK_wind_speed','SK_wind_direction'], inplace=True)

# remove unstable data
#df.drop(columns=['water_level_ultrasound','PM_precipitation_cum'], inplace=True)
df.drop(columns=['water_level_ultrasound'], inplace=True)

#cummulation to 1h
df_hourly = df.resample('1h').mean()

# 3. Display the first few rows of the new daily dataset
print(df_hourly.head())

# Lag variables generation
df_hourly['temp_lag_2'] = df_hourly['PM_temperature'].shift(2)
df_hourly['temp_lag_3'] = df_hourly['PM_temperature'].shift(3)
df_hourly = df_hourly.dropna(subset=['temp_lag_2', 'temp_lag_3'])

df_hourly['SWD_lag_3'] = df_hourly['PM_SWD'].shift(3)
df_hourly['SWD_lag_4'] = df_hourly['PM_SWD'].shift(4)
df_hourly = df_hourly.dropna(subset=['SWD_lag_3', 'SWD_lag_4'])

df_hourly['WS_lag_1'] = df_hourly['PM_wind_speed'].shift(1)
df_hourly['WS_lag_2'] = df_hourly['PM_wind_speed'].shift(2)
df_hourly = df_hourly.dropna(subset=['WS_lag_1', 'WS_lag_2'])

df_hourly['precip_lag_1'] = df_hourly['PM_precipitation'].shift(1)
df_hourly = df_hourly.dropna(subset=['precip_lag_1'])

#snow memory
k = 0.1

df_hourly["melt"] = k * np.maximum(
    df_hourly["PM_temperature"],
    0
)

storage = np.nan
snow_storage = []

for _, row in df_hourly.iterrows():

    if not np.isnan(row["PM_snow_height"]):
        storage = row["PM_snow_height"]
    if row["PM_snow_height"] > 0:
        storage = row["PM_snow_height"]

    if not np.isnan(storage):
        storage -= row["melt"]
        storage = max(storage, 0)

    snow_storage.append(storage)

df_hourly["SnowStorage"] = snow_storage

# calculation summer_seasonality

hour_of_year = (df_hourly.index.dayofyear - 1) * 24 + df_hourly.index.hour

# 2. Start (1. Mai, 00:00) und Ende (30. Sept, 23:00) in Stunden umrechnen
# Tag 121 (1. Mai) startet bei Stunde: (121 - 1) * 24 = 2880
# Tag 273 (30. Sep) endet nach Stunde: (273 * 24) - 1 = 6551
start_hour = (121 - 1) * 24  # 2880
end_hour = (273 * 24) - 1  # 6551
duration_hours = end_hour - start_hour

# 3. Berechnung der Sommerkurve auf Stundenbasis (Sinus-Welle von 0 bis Pi)
summer_wave_hourly = np.sin(
    np.pi * (hour_of_year - start_hour) / duration_hours
)

# 4. Kombination der Zeitintervalle auf Stundenbasis
df_hourly['summer_seasonality'] = np.where(
    (hour_of_year >= start_hour) & (hour_of_year <= end_hour),
    summer_wave_hourly,
    0.0,
)

#calculation daily period
hour_of_day = df_hourly.index.hour
df_hourly['day_of_year'] = df_hourly.index.dayofyear

df_hourly['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24.0)
df_hourly['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24.0)

# Modifikationen nach Feature Engineering speichern
df_hourly = df_hourly.reset_index()
df_hourly.to_csv(
    'mod_feature_meteohydrodata_hourly.csv', encoding='utf-8', index=False
)

