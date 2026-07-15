#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:57:48 2026

@author: theresa
"""
import os
import streamlit as st
import pandas as pd

directory = '/home/theresa/Liora/Projets_fiche/Data/'
df = pd.read_csv(os.path.join(directory, 'preprocessed_meteohydrodata2026-06-04.csv'), index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

st.title('Modelling the Water Level')
st.header('of a Glacier Stream using Meteorological Data')
st.sidebar.title("Table of contents")
pages = ['Project Description', 
         'Data Description',
         'Data Visualization',
         'Data Preprocessing',
         'Data Analysis',
         'Feature Engineering',
         'Modelling',
         'Conclusion']
page = st.sidebar.radio("Go to", pages)

if page == pages[0]:
    st.header("Project Description")
    st.image('csm_8_Vernagtxxxx_6853154684.jpg', caption='Vernagtferner in Tyrol (2900 - 3600 m)')
    st.subheader('Goal')
    st.write('To set up a model to compute the water level of the stream Vernagtbach from the meteorological variables')
    st.markdown('- air temperature')
    st.markdown('- relative humidity')
    st.markdown('- wind speed and direction')
    st.markdown('- precipitation and snow height')
    st.markdown('- short wave radiation downwards and upwards')
    st.write('measured at two measurement stations at 2640 m and 3080 m in the vicinity of the glacier.')
    st.subheader('Issues')
    st.write('The problem is not a conventional data science problem:')
    st.markdown('- It is not a typical regression problem because the time dependence spoils the independence between the observations.')
    st.markdown('- It is not a typical time series problem because it is not the prediction of the water level from the previous water levels.')
    
if page == pages[1]:
    st.header("Data Description")
    st.write('Data from January 1, 2013 to December 31, 2024 in steps of 5 minutes')
    st.write('Data in a text file with comma-separated values of 160.3 MB')    
    st.subheader('Hydrological Data from one measurement station')
    st.markdown('Measurement station at 2640 m (Pegelstation Hydrologie)')
    st.markdown('- water level from a gauge measurement (water_level)')
    st.markdown('- water level from a measurement with ultrasound (water_level_ultrasound)')
    st.subheader('Meteorological Data from two measurement stations')
    st.markdown('Measurement station at 2640 m (Pegelstation Meteorologie')
    st.markdown('- air temperature (PM_temperature)')
    st.markdown('- atmospheric pressure (PM_atmospheric_pressure)')
    st.markdown('- relative humidity (PM_relative_humidity)')
    st.markdown('- precipitation (PM_precipitation)')
    st.markdown('- cumulative precipitation (PM_precipitation_cum)')
    st.markdown('- snow height (PM_snow_height)')
    st.markdown('- wind speed (PM_wind_speed)')
    st.markdown('- wind direction (PM_wind_direction)')
    st.markdown('- short wave radiation downwards (PM_SWD)')
    st.markdown('- short wave radiation upwards (PM_SWU)')
    st.markdown('Measurement station at 3080 m (Schwarzkögele)')
    st.markdown('- air temperature (SK_temperature)')
    st.markdown('- relative humidity (SK_relative_humidity)')
    st.markdown('- wind speed (SK_wind_speed)')
    st.markdown('- wind direction (SK_wind_direction)')

if page == pages[2]: 
    st.header("Data Visualization")
    
    choice = ['Water Level', 'Air Temperatures', 'Atmospheric Pressure', 'Relative Humidity', 'Precipitation', 'Snow Height', 'Wind Speed', 'Wind Direction', 'Short Wave Radiation']
    option = st.selectbox('Variable', choice)
    display = st.radio('Measurement interval', ('all data', 'one day'))
    if display == 'all data':
        if option == 'Water Level':
            st.image('../reports/plots/overview_water_level.png')
        elif option == 'Air Temperatures':
            st.image('../reports/plots/overview_temperature.png')
        elif option == 'Atmospheric Pressure':
            st.image('../reports/plots/overview_pressure.png')
        elif option == 'Relative Humidity':
            st.image('../reports/plots/overview_humidity.png')
        elif option == 'Precipitation':
            st.image('../reports/plots/overview_precipitation.png')
        elif option == 'Snow Height':
            st.image('../reports/plots/overview_snow.png')
        elif option == 'Wind Speed':
            st.image('../reports/plots/overview_wind_speed.png')
        elif option == 'Wind Direction':
            st.image('../reports/plots/overview_wind_direction.png')
        elif option == 'Short Wave Radiation':
            st.image('../reports/plots/overview_radiation.png')
    elif display == 'one day':
        if option == 'Water Level':
            st.image('../reports/plots/overview_water_level_2days.png')
        elif option == 'Air Temperatures':
            st.image('../reports/plots/overview_temperature_2days.png')
        elif option == 'Atmospheric Pressure':
            st.image('../reports/plots/overview_pressure_2days.png')
        elif option == 'Relative Humidity':
            st.image('../reports/plots/overview_humidity_2days.png')
        elif option == 'Precipitation':
            st.image('../reports/plots/overview_precipitation_2days.png')
        elif option == 'Snow Height':
            st.image('../reports/plots/overview_snow_2days.png')
        elif option == 'Wind Speed':
            st.image('../reports/plots/overview_wind_speed_2days.png')
        elif option == 'Wind Direction':
            st.image('../reports/plots/overview_wind_direction_2days.png')
        elif option == 'Short Wave Radiation':
            st.image('../reports/plots/overview_radiation_2days.png')
    
if page == pages[3]:
    st.header("Data Preprocessing")
    st.subheader('Preprocessing of the target variable water_level')
    st.markdown('- large time gaps of water_level are filled with adjusted values of water_level_ultrasound')
    st.markdown('- remaining small gaps are interpolated')
    st.subheader('Preprocessing of the other variables')
    st.markdown('- Forward Shifting for gaps of less than a day')
    st.markdown('- Forward Filling for longer gaps')
    
if page == pages[4]:
    st.header("Data Analysis")
    st.image('Corr_matrix.png', caption='Correlation Matrix', width='stretch')

if page == pages[5]:
    st.header("Feature Engineering")
    st.subheader("Feature Engineering for Machine Learning Models")
    st.markdown('-')
    
    st.subheader('Feature Engineering for Deep Learning Models')
    st.markdown('- Data is left in steps of 5 minutes')
    st.markdown('- Creation of a new time variable to keep through the modell as time reference')
    st.markdown('- Conversion of the wind variables of speed and direction in speed in X direction and speed in Y direction')
    st.markdown('- Conversion of the cumulative precipitation in variables precipitation and evaporation')
    st.markdown('- Deletion of variables because of faulty and redundant measurements')

if page == pages[6]:
    st.header("Modelling")
    st.subheader('Model Application of Machine Learning Models')
    st.markdown('-')
    
    st.subheader('Model Application of Deep Learning Models')
    st.markdown('- ')
    
if page == pages[7]:
    st.header("Conclusion")
        