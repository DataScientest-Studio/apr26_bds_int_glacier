# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from datetime import datetime

import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

from itertools import groupby

# Create variable for constant water_level in winter in cm
water_level_winter = 5

# Create variable directory for directory with 'meteohydrodata.csv'
directory = '/home/theresa/Liora/Projets_fiche/Data/'

# Impport 'meteohydrodata.csv' and convert index to datetime
df_result = pd.read_csv(os.path.join(directory, 'meteohydrodata2026-05-31.csv'), index_col='TIMESTAMP')
df_result.index = pd.to_datetime(df_result.index)

# Import the events at Pegelstation
with open(os.path.join(directory, 'Daten_Liora', 'events_pm.pkl'), 'rb') as file:
    df_events_pm = pickle.load(file)

# Get list consecutive_missing with missing values grouped when consecutive
consecutive_missing = []
for k, g in groupby(enumerate(df_result['water_level'].isna()), lambda x: x[1]):
    if k:
        consecutive_missing.append(list(map(lambda x: x[0], list(g))))
        
# Duplicate df_result
df_result_preprocessed = df_result.copy()

# Replace missing values in winter with water_level_winter
for i in range(len(consecutive_missing)):
    if len(consecutive_missing[i]) > 10000:
        start = df_result_preprocessed.index[consecutive_missing[i][0]]
        stop = df_result_preprocessed.index[consecutive_missing[i][-1]]
        df_result_preprocessed.loc[start:stop, 'water_level'] = water_level_winter
        
# Update list consecutive_missing with missing values of df_result_preprocessed
consecutive_missing = []
for k, g in groupby(enumerate(df_result_preprocessed['water_level'].isna()), lambda x: x[1]):
    if k:
        consecutive_missing.append(list(map(lambda x: x[0], list(g))))
        
df_result_preprocessed2 = df_result_preprocessed.copy()
        
# Replace missing values with higher numbers of consecutive missing values with moving average and level adjusted data from 'water_level_ultrasound'        
list_i = [6, 22, 24, 33, 38, 44, 49, 74]
for i in list_i:
    plot_start = df_result_preprocessed2.index[consecutive_missing[i][0]] - pd.Timedelta('1d')
    plot_stop = df_result_preprocessed2.index[consecutive_missing[i][-1]] + pd.Timedelta('1d')
    df_result_preprocessed2.loc[plot_start:plot_stop, ['water_level', 'water_level_ultrasound', 'PM_temperature']].plot(marker='+', linewidth=0)
    
    start = df_result_preprocessed2.index[consecutive_missing[i][0]]
    stop = df_result_preprocessed2.index[consecutive_missing[i][-1]]
    window = 3
    start_rolling = df_result_preprocessed2.index[consecutive_missing[i][0] - window // 2 - 1]
    stop_rolling = df_result_preprocessed2.index[consecutive_missing[i][-1] + window // 2 + 1]
    dummy = df_result_preprocessed2.loc[start_rolling:stop_rolling, 'water_level_ultrasound'].rolling(window, center=True).mean()
    
    start_minus_1 = df_result_preprocessed2.index[consecutive_missing[i][0] - 1]
    stop_plus_1 = df_result_preprocessed2.index[consecutive_missing[i][-1] + 1]
    comparison_start = df_result_preprocessed2.loc[start_minus_1, 'water_level'] - dummy[start_minus_1]
    comparison_stop = df_result_preprocessed2.loc[stop_plus_1, 'water_level'] - dummy[stop_plus_1]
    
    df_result_preprocessed2.loc[start:stop, 'water_level'] = dummy[start:stop] + comparison_stop
        
    df_result_preprocessed2.loc[plot_start:plot_stop, ['water_level', 'PM_temperature']].plot()
    
# special case 
i = 48
plot_start = df_result_preprocessed2.index[consecutive_missing[i][0]] - pd.Timedelta('1d')
plot_stop = df_result_preprocessed2.index[consecutive_missing[i][-1]] + pd.Timedelta('1d')
df_result_preprocessed2.loc[plot_start:plot_stop, ['water_level', 'water_level_ultrasound', 'PM_temperature']].plot(marker='+', linewidth=0)

start = df_result_preprocessed2.index[consecutive_missing[i][0]]
stop = df_result_preprocessed2.index[consecutive_missing[i][-1]]
window = 3
start_rolling = df_result_preprocessed2.index[consecutive_missing[i][0] - window // 2 - 1]
stop_rolling = df_result_preprocessed2.index[consecutive_missing[i][-1] + window // 2 + 1]
dummy = df_result_preprocessed2.loc[start_rolling:stop_rolling, 'water_level_ultrasound'].rolling(window, center=True).mean()

start_minus_1 = df_result_preprocessed2.index[consecutive_missing[i][0] - 1]
stop_plus_1 = df_result_preprocessed2.index[consecutive_missing[i][-1] + 1]
comparison_start = df_result_preprocessed2.loc[start_minus_1, 'water_level'] - dummy[start_minus_1]
comparison_stop = df_result_preprocessed2.loc[stop_plus_1, 'water_level'] - dummy[stop_plus_1]

df_result_preprocessed2.loc[start:stop, 'water_level'] = dummy[start:stop] - 7
    
df_result_preprocessed2.loc[plot_start:plot_stop, ['water_level', 'PM_temperature']].plot()

# Update list consecutive_missing with missing values of df_result_preprocessed2
consecutive_missing = []
for k, g in groupby(enumerate(df_result_preprocessed2['water_level'].isna()), lambda x: x[1]):
    if k:
        consecutive_missing.append(list(map(lambda x: x[0], list(g))))
        
# Interpolate remaining missing values
df_result_preprocessed2['water_level'] = df_result_preprocessed2['water_level'].interpolate()

plt.figure()
df_result_preprocessed2['water_level'].plot()

today_string = datetime.today().strftime('%Y-%m-%d')

df_result_preprocessed2.to_csv('/home/theresa/Liora/Projets_fiche/Data/preprocessed_meteohydrodata' + today_string + '.csv')
