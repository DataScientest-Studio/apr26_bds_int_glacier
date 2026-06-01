#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:19:02 2026

@author: theresa
"""

from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import os
import pandas as pd

# Create variable for directory for directory with 'meteohydrodata.csv'
directory = '/home/theresa/Liora/Projets_fiche/Data/'

# Create variable for directory for plots
plot_directory = '/home/theresa/apr26_bds_int_glacier/reports/plots/'

# Impport 'meteohydrodata.csv' and convert index to datetime
df_result = pd.read_csv(os.path.join(directory, 'meteohydrodata2026-05-31.csv'), index_col='TIMESTAMP')
df_result.index = pd.to_datetime(df_result.index)

df_result1 = df_result.loc[(df_result.index.year == 2024) &
                              (df_result.index.month == 1) &
                              (df_result.index.day == 1), :]

df_result2 = df_result.loc[(df_result.index.year == 2024) &
                              (df_result.index.month == 7) &
                              (df_result.index.day == 1), :]

# Create plot for air temperature of Pegelstation Meteorologie and Schwarzkögele
fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
df_result1[['PM_temperature', 'SK_temperature']].plot(color=['r','b'], legend=True, ax=ax1)
ax1.set_ylabel('Temperature in °C')
ax1.set_title('Januar 1, 2024')
ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
df_result2[['PM_temperature', 'SK_temperature']].plot(color=['r','b'], legend=True, ax=ax2)
ax2.set_ylabel('Temperature in °C')
ax2.set_title('July 1, 2024')
ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig(plot_directory + 'overview_temperature_2days.png')
plt.close()

# Create plot for atmospheric pressure of Pegelstation Meteorologie
fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
df_result1['PM_atmospheric_pressure'].plot(color='r', legend=True, ax=ax1)
ax1.set_ylabel('Atmospheric Pressure in hPa')
ax1.set_title('Januar 1, 2024')
ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
df_result2['PM_atmospheric_pressure'].plot(color='r', legend=True, ax=ax2)
ax2.set_ylabel('Atmospheric Pressure in hPa')
ax2.set_title('July 1, 2024')
ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig(plot_directory + 'overview_pressure_2days.png')
plt.close()

# Create plot for relative humidity of Pegelstation Meteorologie and Schwarzkögele
fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
df_result1[['SK_relative_humidity', 'PM_relative_humidity']].plot(color=['r','b'], ax=ax1)
ax1.set_ylabel('Relative humidity in %')
ax1.set_title('Januar 1, 2024')
ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
df_result2[['SK_relative_humidity', 'PM_relative_humidity']].plot(color=['r','b'], ax=ax2)
ax2.set_ylabel('Relative humidity in %')
ax2.set_title('July 1, 2024')
ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig(plot_directory + 'overview_humidity_2days.png')
plt.close()

# Create plot for precipitation and precipitation_cum of Pegelstation Meteorologie
fig, (ax1, ax3) = plt.subplots(1,2,sharey=True)
df_result1['PM_precipitation'].plot(color='r', linewidth=0.1, legend=False, ax=ax1)
ax1.set_ylabel('Precipitation in mm/(5 min)')
ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
ax1.legend(loc='upper left')
ax2 = ax1.twinx() 
df_result1['PM_precipitation_cum'].plot(color='r', legend=False, ax=ax2)
ax2.set_title('Januar 1, 2024')
plt.autoscale(enable=True, axis='x', tight=True)
ax2.legend(loc='center right')
df_result2['PM_precipitation'].plot(color='r', linewidth=0.1, legend=False, ax=ax3)
ax3.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
ax3.legend(loc='upper left')
ax4 = ax3.twinx() 
df_result2['PM_precipitation_cum'].plot(color='r', legend=False, ax=ax4)
ax4.set_ylabel('cumulative Precipitation in mm')
ax4.set_title('July 1, 2024')
plt.autoscale(enable=True, axis='x', tight=True)
ax4.legend(loc='center right')
plt.savefig(plot_directory + 'overview_precipitation_2days.png')
plt.close()

# Create plot for snow height of Pegelstation Meteorologie
fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
df_result1['PM_snow_height'].plot(color='r', legend=True, ax=ax1)
ax1.set_ylabel('Snow Height in hPa')
ax1.set_title('Januar 1, 2024')
ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
df_result2['PM_snow_height'].plot(color='r', legend=True, ax=ax2)
ax2.set_ylabel('Snow Height in hPa')
ax2.set_title('July 1, 2024')
ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig(plot_directory + 'overview_snow_2days.png')
plt.close()

# Create plot for wind speed of Pegelstation Meteorologie and Schwarzkögele
fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
df_result1[['PM_wind_speed', 'SK_wind_speed']].plot(linewidth=1, color=['r','b'], ax=ax1)
ax1.set_ylabel('Wind Speed in m/s')
ax1.set_title('Januar 1, 2024')
ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
df_result2[['PM_wind_speed', 'SK_wind_speed']].plot(linewidth=1, color=['r','b'], ax=ax2)
ax2.set_ylabel('Wind Speed in m/s')
ax2.set_title('July 1, 2024')
ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig(plot_directory + 'overview_wind_speed_2days.png')
plt.close()

# Create plot for wind direction of Pegelstation Meteorologie and Schwarzkögele
df_result1['PM_wind_direction_shifted'] = df_result1['PM_wind_direction']
df_result1.loc[df_result1['PM_wind_direction'] < 90, 'PM_wind_direction_shifted'] = df_result1.loc[df_result1['PM_wind_direction'] < 90, 'PM_wind_direction'] + 360
df_result1['SK_wind_direction_shifted'] = df_result1['SK_wind_direction']
df_result1.loc[df_result1['SK_wind_direction'] < 90, 'SK_wind_direction_shifted'] = df_result1.loc[df_result1['SK_wind_direction'] < 90, 'SK_wind_direction'] + 360
df_result2['PM_wind_direction_shifted'] = df_result2['PM_wind_direction']
df_result2.loc[df_result2['PM_wind_direction'] < 90, 'PM_wind_direction_shifted'] = df_result2.loc[df_result2['PM_wind_direction'] < 90, 'PM_wind_direction'] + 360
df_result2['SK_wind_direction_shifted'] = df_result2['SK_wind_direction']
df_result2.loc[df_result2['SK_wind_direction'] < 90, 'SK_wind_direction_shifted'] = df_result2.loc[df_result2['SK_wind_direction'] < 90, 'SK_wind_direction'] + 360
fig, axs = plt.subplots(2,2,sharey=True)
df_result1['PM_wind_direction_shifted'].plot(linewidth=0, marker='.', ax=axs[0,0], legend=True, color='r')
axs[0,0].set_ylabel('Wind Direction in degrees')
axs[0,0].xaxis.set_major_formatter(DateFormatter("%H:%M"))
axs[0,0].set_title('January 1, 2024')
df_result2['PM_wind_direction_shifted'].plot(linewidth=0, marker='.', ax=axs[0,1], legend=True, color='r')
axs[0,1].xaxis.set_major_formatter(DateFormatter("%H:%M"))
axs[0,1].set_title('July 1, 2024')
df_result1['SK_wind_direction_shifted'].plot(linewidth=0, marker='.', ax=axs[1,0], legend=True, sharex=True, color='b')
axs[1,0].set_ylabel('Wind Direction in degrees')
axs[1,0].xaxis.set_major_formatter(DateFormatter("%H:%M"))
df_result2['SK_wind_direction_shifted'].plot(linewidth=0, marker='.', ax=axs[1,1], legend=True, sharex=True, color='b')
axs[1,1].xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.savefig(plot_directory + 'overview_wind_direction_2days.png')
plt.close()

# Create plot for shortwave radiation of Pegelstation Meteorologie
fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
df_result1['PM_SWD'].plot(color='b', marker='.', linewidth=0, ax=ax1)
df_result1['PM_SWU'].plot(color='r', marker='.', linewidth=0, ax=ax1)
ax1.set_ylabel('Shortwave Radiation in W/m**2')
ax1.set_title('Januar 1, 2024')
ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
ax1.legend()
df_result2['PM_SWD'].plot(color='b', marker='.', linewidth=0, ax=ax2)
df_result2['PM_SWU'].plot(color='r', marker='.', linewidth=0, ax=ax2)
ax2.set_title('July 1, 2024')
ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
ax2.legend()
plt.savefig(plot_directory + 'overview_radiation_2days.png')
plt.close()

# Create plot for water level of Pegelstation Hydrologie
fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
df_result1['water_level'].plot(color='k', marker='.', ax=ax1)
ax1.set_ylabel('Water Level in m')
ax1.set_title('Januar 1, 2024')
df_result1['water_level_ultrasound'].plot(color='g', marker='.', ax=ax1)
ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
ax1.legend()
df_result2['water_level'].plot(color='k', marker='.', ax=ax2)
df_result2['water_level_ultrasound'].plot(color='g', marker='.', ax=ax2)
ax2.set_title('July 1, 2024')
ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.autoscale(enable=True, axis='x', tight=True)
ax2.legend()
plt.savefig(plot_directory + 'overview_water_level_2days.png')
plt.close()