#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 00:00:17 2026

@author: theresa
"""

import os
import pandas as pd

# Path to the directory containing the file with the data (has to be adjusted)
directory = '/home/theresa/Liora/Projets_fiche/Data/'

df_result = pd.read_csv(os.path.join(directory, 'meteohydrodata.csv'), index_col='TIMESTAMP')
df_result.index = pd.to_datetime(df_result.index)
    
# Create a Dataframe df with the column names being the column names of the excel-sheet to be created
df = pd.DataFrame(columns=['number of column',
                           'name of column',
                           'name of variable',
                           'name of measurement station',
                           'measurement instrument',
                           'measurement unit of variable',
                           'type of variable',
                           'percentage of missing values',
                           'categorical/quantitative',
                           'distribution',
                           'comments'])

# Fill the column 'name of column' of df with the column names of df_result
df['name of column'] = df_result.columns

# Fill the column 'number of column' of df
df['number of column'] = range(1, len(df) + 1)

# Fill the column 'type of variable' of df with float
df['type of variable'] = 'float'

# Calculate the column 'percentage of missing values' of df from df_result
df['percentage of missing values'] = df['name of column'].map(lambda x: df_result[x].isna().sum()/len(df_result))

# Fill the column 'categorical/quantitative' of df with quantitative
df['categorical/quantitative'] = 'quantitative'

# variable for current measurement station
measurement_station = 'Pegelstation Meteorologie, height 2640 m'

# Add information to df about column 'PM_temperature'
column_name = 'PM_temperature'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Air temperature at 2 m height'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = 'Pt-100 temperature sensor with ventilation fan'
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = '°C'
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'PM_atmospheric_pressure'
column_name = 'PM_atmospheric_pressure'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Pressure, atmospheric'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = 'Barometric pressure sensor, Druck, RPT 410h'
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = 'hPa'
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'PM_relative_humidity'
column_name = 'PM_relative_humidity'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Humidity, relative'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = 'Hair hygrometer, Thies Clima'
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = '%'
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'PM_precipitation'
column_name = 'PM_precipitation'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Precipitation'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = 'Tipping bucket, Gertsch, unheated'
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = 'mm/5 min'
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'PM_precipitation_cum'
column_name = 'PM_precipitation_cum'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Precipitation, sum'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = 'Weighing rain gauge, Belfort'
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = 'mm'
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'PM_snow_height'
column_name = 'PM_snow_height'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Snow height'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = 'Sonic Ranging Sensor, Campbell Scientific, SR50'
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = 'm'
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'PM_wind_speed'
column_name = 'PM_wind_speed'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Wind speed at 2 m height'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = 'Anemometer, Thies Clima'
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = 'm/s'
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'PM_wind_direction'
column_name = 'PM_wind_direction'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Wind direction at 2 m height'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = 'Wind vane, Thies Clima'
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = 'deg'
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'PM_SWD'
column_name = 'PM_SWD'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Short-wave downward (GLOBAL) radiation'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = 'Albedometer, Kipp & Zonen, CM7B, unventilated'
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = 'W/m**2'
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'PM_SWU'
column_name = 'PM_SWU'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Short-wave upward (REFLEX) radiation'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = 'Albedometer, Kipp & Zonen, CM7B, unventilated'
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = 'W/m**2'
df.loc[df['name of column'] == column_name, 'description'] = ''

# variable for current measurement station
measurement_station = 'Schwarzkögele, height 3075 m'

# Add information to df about column 'SK_temperature'
column_name = 'SK_temperature'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Air temperature at 2 m height'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = 'Pt-100 temperature sensor'
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = '°C'
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'SK_relative_humidity'
column_name = 'SK_relative_humidity'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Humidity, relative'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = ''
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = '%'
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'SK_wind_speed'
column_name = 'SK_wind_speed'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Wind speed at 2 m height'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = ''
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = 'm/s'
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'SK_wind_direction'
column_name = 'SK_wind_direction'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'Wind direction at 2 m height'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = ''
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = 'deg'
df.loc[df['name of column'] == column_name, 'description'] = ''

# variable for current measurement station
measurement_station = 'Pegelstation Hydrologie, height 2640 m'

# Add information to df about column 'water_level'
column_name = 'water_level'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'water level'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = ''
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = ''
df.loc[df['name of column'] == column_name, 'description'] = ''

# Add information to df about column 'water_level_ultrasound'
column_name = 'water_level_ultrasound'
df.loc[df['name of column'] == column_name, 'name of variable'] = 'water level measured with ultrasound'
df.loc[df['name of column'] == column_name, 'name of measurement station'] = measurement_station
df.loc[df['name of column'] == column_name, 'measurement instrument'] = ''
df.loc[df['name of column'] == column_name, 'measurement unit of variable'] = ''
df.loc[df['name of column'] == column_name, 'description'] = ''

with pd.ExcelWriter(os.path.join(directory, 'datasheet.xlsx')) as writer:
    df.to_excel(writer, sheet_name='columns', index=False)