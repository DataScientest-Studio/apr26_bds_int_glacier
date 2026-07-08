import numpy as np
import pandas as pd

df = pd.read_csv('/home/ubuntu/preprocessed_meteohydrodata2026-06-04.csv', index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

# Drop columns of Schwarzkoegele and others
df_preprocessed = df.drop(columns=['PM_precipitation', 'SK_temperature', 'SK_relative_humidity', 'SK_wind_speed', 'SK_wind_direction', 'water_level_ultrasound'])

# Convert wind direction and speed to wind speed in x and y direction
df_preprocessed['PM_wind_direction_in_radians'] = df_preprocessed['PM_wind_direction'] * np.pi / 180
df_preprocessed['PM_wind_speed_X_direction'] = df_preprocessed['PM_wind_speed'] * np.cos(df_preprocessed['PM_wind_direction_in_radians'])
df_preprocessed['PM_wind_speed_Y_direction'] = df_preprocessed['PM_wind_speed'] * np.sin(df_preprocessed['PM_wind_direction_in_radians'])
df_preprocessed = df_preprocessed.drop(columns=['PM_wind_direction','PM_wind_speed','PM_wind_direction_in_radians'])

# Convert precipitation_cum to remove yearly jump
df_preprocessed['dummy'] = df_preprocessed['PM_precipitation_cum'] - df_preprocessed['PM_precipitation_cum'].shift(-1)
df_preprocessed.loc[df_preprocessed['dummy'] > 100, 'dummy'] = 0
df_preprocessed['precipitation'] = df_preprocessed['dummy'].map(lambda x: x if x > 0 else 0)
df_preprocessed['evaporation'] = df_preprocessed['dummy'].map(lambda x: x if x < 0 else 0)
df_preprocessed = df_preprocessed.drop(columns=['dummy'])

df_preprocessed.to_csv('/home/ubuntu/data.csv')
