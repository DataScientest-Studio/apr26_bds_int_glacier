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
pages = ['Context and Scope of the Project', 
         'Data Visualization',
         'Modelling']
page = st.sidebar.radio("Go to", pages)

if page == pages[1]: 
    st.write("### Data Visualization")
    
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
        
        