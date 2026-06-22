import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#df_result = pd.read_csv(os.path.join(directory, 'preprocessed_meteohydrodata2026-05-26.csv'), index_col='TIMESTAMP')
df = pd.read_csv('../preprocessed_meteohydrodata2026-06-04.csv', index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

# remove SK data
df.drop(columns=['SK_temperature', 'SK_relative_humidity', 'SK_wind_speed','SK_wind_direction'], inplace=True)

# remove unstable data
df.drop(columns=['water_level_ultrasound'], inplace=True)

df_daily = df.resample('D').mean()

# 3. Display the first few rows of the new daily dataset
print(df_daily.head())

# Lag wl 1 day
df_daily['lag_1'] = df_daily['water_level'].shift(1)
df_daily['lag_1'] = df_daily['water_level'].dropna()

# calculation summer_seasonality

# 1. Extraction day of the year (1 bis 365)
day_of_year = df_daily.index.dayofyear

# 2. Define Start (May) and end (September) for summer high

start_day = 121 #1.Mai
end_day = 273 #30. Sep.
duration = end_day - start_day

# 3. Calculation summer curve (Sinus-Welle von 0 bis Pi)
# exlusive values are zero
summer_wave = np.sin(np.pi * (day_of_year - start_day) / duration)

# 4. Combination time intervalls
df_daily['summer_seasonality'] = np.where(
    (day_of_year >= start_day) & (day_of_year <= end_day), 
    summer_wave, 
    0.0 )

# Safe modifications after feature engineering
df_daily = df_daily.reset_index()
df_daily.to_csv('mod_feature_meteohydrodata2026-06-19.csv', encoding='utf-8', index=False)



