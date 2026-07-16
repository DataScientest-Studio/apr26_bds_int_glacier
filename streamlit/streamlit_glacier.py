#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:57:48 2026

@author: theresa
"""

import pandas as pd
import streamlit as st

st.title('Modelling the Water Level')
st.header('of a Glacier Stream using Meteorological Data')
st.sidebar.title("Table of contents")
pages = ['Project Description', 
         'Data Visualization',
         'Data Preprocessing',
         'Data Analysis',
         'Feature Engineering',
         'Modeling',
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
    st.write('measured at two measurement stations at 2640 m and 3080 m in the vicinity of the glacier from January 1, 2013 to December 31, 2024 in steps of 5 minutes.')
    st.subheader('Issues')
    st.write('The problem of the project is not a conventional data science problem:')
    st.markdown('- It is not a typical regression problem because the time dependence spoils the independence between the observations.')
    st.markdown('- It is not a typical time series problem because it is not the prediction of the water level from the previous water levels.')

if page == pages[1]: 
    st.header("Data Visualization of some variables")
    
    st.subheader('Water Level')
    st.image('plots/overview_water_level.png')
    
    st.subheader('Air Temperatures')
    st.image('plots/overview_temperature.png')
    
    st.subheader('Snow Height')
    st.image('plots/overview_snow.png')
    
    st.subheader('Wind Direction')
    st.image('plots/overview_wind_direction_2days.png')
    
if page == pages[2]:
    st.header("Data Preprocessing")
    st.subheader('Preprocessing of the target variable water_level')
    st.markdown('- water_level is set to 5 cm in winter')
    st.markdown('- other large time gaps of water_level are filled with adjusted values of water_level_ultrasound')
    st.markdown('- remaining small gaps are interpolated')
    st.subheader('Preprocessing of the other variables')
    st.markdown('- Forward Shifting for gaps of less than a day')
    st.markdown('- Forward Filling for longer gaps')
    
if page == pages[3]:
    st.header("Data Analysis")
    st.image('Corr_matrix.png', width='stretch')

if page == pages[4]:
    st.header("Feature Engineering")
    st.subheader("Feature Engineering for Machine Learning Models")
    st.markdown('-')
    
    st.subheader('Feature Engineering for Deep Learning Models')
    st.markdown('- Data is left in steps of 5 minutes')
    st.markdown('- Creation of a new time variable to keep through the modell as time reference')
    st.markdown('- Conversion of the wind variables of speed and direction in speed in X direction and speed in Y direction')
    st.markdown('- Conversion of the cumulative precipitation in variables precipitation and evaporation')
    st.markdown('- Deletion of variables because of faulty and redundant measurements')

if page == pages[5]:
    st.header("Modeling")
    st.subheader('Model Application of Machine Learning Models')
    st.markdown('-')
    
    st.subheader('Model Application of Deep Learning Models')
    st.markdown('- To feed data to a neural net a CustomDataset was defined in Pytorch:')
    st.markdown('- It divides the input data into continuous chunks of data on the time axis with a given time span.')
    st.markdown('- The chunks overlap each other.')
    st.markdown('- A model is trained with these chunks.')
    st.markdown('- For the metric and the output of the model an output of the model is computed by averaging the output chunks of the model on their position of the time axis.')
    st.markdown('- The chosen metric for validation and testing is mean absolute error.')
    st.markdown('- Training Dataset: 70\%, Validation Dataset: 15\%, Test Dataset: 15\%')
    st.markdown('- Architecture of first model:')

    st.write('Sequential(')
    st.write('(0): Conv2d(1, 2, kernel_size=(3, 1), stride=(1, 1))')
    st.write('(1): ReLU()')
    st.write('(2): MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=0, dilation=1, ceil_mode=False)')
    st.write('(3): Conv2d(2, 4, kernel_size=(7, 1), stride=(1, 1))')
    st.write('(4): ReLU()')
    st.write('(5): MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=0, dilation=1, ceil_mode=False)')
    st.write('(6): Flatten(start_dim=1, end_dim=-1)')
    st.write('(7): Linear(in_features=768, out_features=288, bias=True)')
    
    st.markdown('- Graphical presentation of the result of the first model:')
    st.image('model_PM_0_result.png', width='stretch')
    
    st.markdown('- Results for 4 different models:')
    results = pd.DataFrame(
        {
        "model_PM_0": [8.73, 10.74, '1 day'],
        "model_all_0": [6.7, 9.2, '1 day'],
        "model_pm_1": [5.87, 8.41, '7 days'],
        "model_all_1": [6.49, 9.38, '7 days'],
        },
        index=["mean absolute error", "rmse", "time span"])
    st.table(results)

if page == pages[6]:
    st.header("Conclusion and Outlook")
    st.subheader('Conclusion for Machine Learning Models')
    
    st.subheader('Conclusion on Deep Learning Models')
    st.markdown('- It was possible to get a Deep Learning Model running.')
    st.markdown('- The results are only as good a linear regression.')
    st.markdown('- There is very likely improvement on the data loading part possible leading to much lower computation times.')
    st.markdown('- There are a lot of model architectures to be tried out.')
    st.subheader('Outlook')
    st.markdown('- Gain more experience on neural networks and their set up')
    st.markdown('- Try again')
        