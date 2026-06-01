#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 08:11:06 2026

@author: theresa
"""

from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import os
import pandas as pd

from itertools import groupby

def get_missing_list(df, column):
    """
    returns a list consecutive_missing with consecutive missing values of df[column]
    """
    consecutive_missing = []
    for k, g in groupby(enumerate(df[column].isna()), lambda x: x[1]):
        if k:
            consecutive_missing.append(list(map(lambda x: x[0], list(g))))
    return consecutive_missing

def plot_missing_values(df, column, consecutive_missing, timeframe):
    """
    plots df[column] for each entry of consecutive_missing with added timeframe before and after the index
    """
    for i in range(len(consecutive_missing)):
        plt.figure()
        plot_start = df.index[consecutive_missing[i][0]] - pd.Timedelta(timeframe)
        plot_stop = df.index[consecutive_missing[i][-1]] + pd.Timedelta(timeframe)
        df.loc[plot_start:plot_stop, column].plot(x_compat=True)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))

def interpolate_missing_values(df, column, consecutive_missing, threshold):
    """
    interpolates linearly the missing values in consecutive_missing that are not more than threshold
    """
    for i in range(len(consecutive_missing)):
        if len(consecutive_missing[i]) <= threshold:
            start = df.index[consecutive_missing[i][0] - 1]
            stop = df.index[consecutive_missing[i][-1] + 1]
            df.loc[start:stop, column] = df.loc[start:stop, column].interpolate()

# Create variable directory for directory with 'meteohydrodata.csv'
directory = '/home/theresa/Liora/Projets_fiche/Data/'

# Impport 'meteohydrodata.csv' and convert index to datetime
df_result = pd.read_csv(os.path.join(directory, 'meteohydrodata.csv'), index_col='TIMESTAMP')
df_result.index = pd.to_datetime(df_result.index)

# Get list consecutive_missing with missing values grouped when consecutive
column = 'PM_wind_speed'
consecutive_missing = get_missing_list(df_result, column)

# Duplicate df_result
df_result_preprocessed = df_result.copy()

# Interpolate missing values
threshold = 250
interpolate_missing_values(df_result_preprocessed, column, consecutive_missing, threshold)

# Plot the remaining missing values
consecutive_missing = get_missing_list(df_result_preprocessed, column)
timeframe = '1d'
plot_missing_values(df_result_preprocessed, [column, 'SK_wind_speed'], consecutive_missing, timeframe)
