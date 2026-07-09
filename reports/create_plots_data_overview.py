#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:19:02 2026

@author: theresa
"""

import matplotlib.pyplot as plt
import os
import pandas as pd

# Create variable for directory for directory with 'meteohydrodata.csv'
directory = '/home/theresa/Liora/Projets_fiche/Data/'

# Create variable for directory for plots
plot_directory = '/home/theresa/apr26_bds_int_glacier/reports/plots/'

# Import 'meteohydrodata.csv' and convert index to datetime
df_result = pd.read_csv(os.path.join(directory, 'meteohydrodata2026-05-31.csv'), index_col='TIMESTAMP')
df_result.index = pd.to_datetime(df_result.index)

# Create plot for air temperature of Pegelstation Meteorologie and Schwarzkögele
df_result[['PM_temperature', 'SK_temperature']].plot(color=['r','b'], legend=True, linewidth=0.2, alpha=0.5)
plt.ylabel('Temperature in °C')
plt.title('Air Temperature at Pegelstation Meteorologie and Schwarzkögele')
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig(plot_directory + 'overview_temperature.png')
plt.close()

# Create plot for atmospheric pressure of Pegelstation Meteorologie
df_result['PM_atmospheric_pressure'].plot(color='r', legend=True, linewidth=0.2, alpha=0.5)
plt.ylabel('Atmospheric Pressure in hPa')
plt.title('Atmospheric Pressure at Pegelstation Meteorologie')
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig(plot_directory + 'overview_pressure.png')
plt.close()

# Create plot for relative humidity of Pegelstation Meteorologie and Schwarzkögele
df_result[['SK_relative_humidity', 'PM_relative_humidity']].plot(color=['r','b'], linewidth=0.2, alpha=0.5)
plt.ylabel('Relative humidity in %')
plt.title('Relative Humidity at Pegelstation Meteorologie and Schwarzkögele')
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig(plot_directory + 'overview_humidity.png')
plt.close()

# Create plot for precipitation and precipitation_cum of Pegelstation Meteorologie
fig, ax1 = plt.subplots()
df_result['PM_precipitation'].plot(color='r', linewidth=0.1, legend=False, ax=ax1)
ax1.set_ylabel('Precipitation in mm/(5 min)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(loc='upper left')
ax2 = ax1.twinx() 
df_result['PM_precipitation_cum'].plot(color='r', legend=False, ax=ax2)
ax2.set_ylabel('cumulative Precipitation in mm')
fig.suptitle('Precipitation at Pegelstation Meteorologie')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend()
plt.savefig(plot_directory + 'overview_precipitation.png')
plt.close()

# Create plot for snow height of Pegelstation Meteorologie
df_result['PM_snow_height'].plot(color='r', legend=True, linewidth=0.2, alpha=0.5)
plt.ylabel('Snow Height in hPa')
plt.title('Snow Height at Pegelstation Meteorologie')
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig(plot_directory + 'overview_snow.png')
plt.close()

# Create plot for wind speed of Pegelstation Meteorologie and Schwarzkögele
df_result[['PM_wind_speed', 'SK_wind_speed']].plot(linewidth=0.1, color=['r','b'])
plt.ylabel('Wind Speed in m/s')
plt.title('Wind Speed at Pegelstation Meteorologie and Schwarzkögele')
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig(plot_directory + 'overview_wind_speed.png')
plt.close()

# Create plot for wind direction of Pegelstation Meteorologie and Schwarzkögele
df_result['PM_wind_direction_shifted'] = df_result['PM_wind_direction']
df_result.loc[df_result['PM_wind_direction'] < 90, 'PM_wind_direction_shifted'] = df_result.loc[df_result['PM_wind_direction'] < 90, 'PM_wind_direction'] + 360
df_result['SK_wind_direction_shifted'] = df_result['SK_wind_direction']
df_result.loc[df_result['SK_wind_direction'] < 90, 'SK_wind_direction_shifted'] = df_result.loc[df_result['SK_wind_direction'] < 90, 'SK_wind_direction'] + 360
fig, (ax1, ax2) = plt.subplots(2,1)
df_result['PM_wind_direction_shifted'].plot(linewidth=0, marker=',', ax=ax1, legend=True, color='r')
ax1.set_ylabel('Wind Direction in degrees')
ax1.set_xlim(df_result.index[0], df_result.index[-1])
df_result['SK_wind_direction_shifted'].plot(linewidth=0, marker=',', ax=ax2, legend=True, sharex=True, color='b')
ax2.set_ylabel('Wind Direction in degrees')
ax2.set_xlim(df_result.index[0], df_result.index[-1])
plt.suptitle('Wind Direction at Pegelstation Meteorologie and Schwarzkögele')
plt.savefig(plot_directory + 'overview_wind_direction.png')
plt.close()

# Create plot for shortwave radiation of Pegelstation Meteorologie
df_result['PM_SWD'].plot(color='b', marker=',', linewidth=0)
plt.ylabel('Shortwave Radiation in W/m**2')
df_result['PM_SWU'].plot(color='r', marker=',', linewidth=0)
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend()
plt.title('Shortwave Radiation Upward and Downward at Pegelstation Meteorologie')
plt.savefig(plot_directory + 'overview_radiation.png')
plt.close()

# Create plot for water level of Pegelstation Hydrologie
df_result['water_level_ultrasound'].plot(color='g', marker=',', linewidth=0.2, alpha=0.5)
plt.ylabel('Water Level Ultrasound in m')
df_result['water_level'].plot(color='k', marker=',', linewidth=0.2, alpha=0.5)
plt.ylabel('Water Level in m')
plt.title('Water Level at Pegelstation Hydrologie')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend()
plt.savefig(plot_directory + 'overview_water_level.png')
plt.close()