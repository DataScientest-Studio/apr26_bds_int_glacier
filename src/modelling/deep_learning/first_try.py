#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:02:15 2026

@author: theresa
"""

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
from torchsummary import summary

# Path to the directory containing the file with the data (has to be adjusted)
directory = '/home/theresa/Liora/Projets_fiche/Data/'

#
df = pd.read_csv(os.path.join(directory, 'preprocessed_meteohydrodata2026-06-04.csv'), index_col='TIMESTAMP')
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

# Account for time
df_preprocessed['time'] = df.index
df_preprocessed['time_s'] = df_preprocessed['time'].map(pd.Timestamp.timestamp) - pd.Timestamp('2013-01-01 00:00:00+00:00').timestamp()
time_min = df_preprocessed['time_s'].min() # 150
time_max = df_preprocessed['time_s'].max() # 378687450
df_preprocessed['time_s_norm'] = (df_preprocessed['time_s'] - time_min) / (time_max - time_min)
df_preprocessed = df_preprocessed.drop(columns=['time', 'time_s'])

X = df_preprocessed.drop(columns=['water_level'])
y = df_preprocessed['water_level']

factor = 0.8
last_index_train = round(len(X)*factor)
X_train = X.iloc[:last_index_train].values
X_test = X.iloc[last_index_train:].values
y_train = y.iloc[:last_index_train].values
y_test = y.iloc[last_index_train:].values

scaler = StandardScaler()
X_train[:,:-1] = scaler.fit_transform(X_train[:,:-1])
X_test[:,:-1] = scaler.transform(X_test[:,:-1])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, length):
        self.X = X
        self.y = y
        self.length = length

    def __getitem__(self, idx):
        # For all data, return [x, y]
        # if idx + self.length > len(self.X):
        #     x = np.zeros((1, 288, 12))
        #     y = np.zeros(288)
        #     # x_dummy = np.expand_dims(self.X[idx:idx + self.length,:], 0)
        #     # x[:,idx:idx + self.length,:] = x_dummy
        #     y[idx:idx + self.length] = self.y[idx:idx + self.length]
        # else:
        x = np.expand_dims(self.X[idx:idx + self.length,:], 0)
        y = self.y[idx:idx + self.length]
        return [x, y]

    def __len__(self):
        return len(self.X) - self.length
    
    def plot_item(self, idx):
        [x, y] = self.__getitem__(idx)
        x_data = np.squeeze(x)
        time = x_data[:,-1]
        time = time * (378687450 - 150) + 150
        time = pd.to_datetime(time, unit='s', origin=pd.Timestamp('2013-01-01 00:00:00+00:00').timestamp())
        plt.figure(figsize=(15,10))
        ax1 = plt.subplot(211)
        labels = ['PM_temperature',
                  'PM_atmospheric_pressure',
                  'PM_relative_humidity',
                  'PM_precipitation_cum',
                  'PM_snow_height',
                  'PM_SWD',
                  'PM_SWU',
                  'PM_wind_speed_X_direction',
                  'PM_wind_speed_Y_direction',
                  'precipitation',
                  'evaporation']
        plt.plot(time, x_data[:,:-1], label=labels)
        plt.autoscale(enable=True, axis='x', tight=True)
        #plt.xticks(rotation=90)
        ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d\n%H:%M"))
        plt.legend()
        plt.xlabel('time')
        ax2 = plt.subplot(212)
        labels = 'water_level'
        plt.plot(time, y, label=labels)
        plt.autoscale(enable=True, axis='x', tight=True)
        #plt.xticks(rotation=90)
        ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d\n%H:%M"))
        plt.legend()
        plt.xlabel('time')

class CustomDatasetTest(torch.utils.data.Dataset):
    def __init__(self, X, y, length):
        self.X = X
        self.y = y
        self.length = length

    def __getitem__(self, idx):
        # For all data, return [x, y]
        # if idx + self.length > len(self.X):
        #     x = np.zeros((1, 288, 12))
        #     y = np.zeros(288)
        #     # x_dummy = np.expand_dims(self.X[idx:idx + self.length,:], 0)
        #     # x[:,idx:idx + self.length,:] = x_dummy
        #     y[idx:idx + self.length] = self.y[idx:idx + self.length]
        # else:
        x = np.expand_dims(self.X[idx*self.length:idx*self.length + self.length,:], 0)
        y = self.y[idx*self.length:idx*self.length + self.length]
        return [x, y]

    def __len__(self):
        return len(self.X) // self.length
    
    def plot_item(self, idx):
        [x, y] = self.__getitem__(idx)
        x_data = np.squeeze(x)
        time = x_data[:,-1]
        time = time * (378687450 - 150) + 150
        time = pd.to_datetime(time, unit='s', origin=pd.Timestamp('2013-01-01 00:00:00+00:00').timestamp())
        plt.figure(figsize=(15,10))
        ax1 = plt.subplot(211)
        labels = ['PM_temperature',
                  'PM_atmospheric_pressure',
                  'PM_relative_humidity',
                  'PM_precipitation_cum',
                  'PM_snow_height',
                  'PM_SWD',
                  'PM_SWU',
                  'PM_wind_speed_X_direction',
                  'PM_wind_speed_Y_direction',
                  'precipitation',
                  'evaporation']
        plt.plot(time, x_data[:,:-1], label=labels)
        plt.autoscale(enable=True, axis='x', tight=True)
        #plt.xticks(rotation=90)
        ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d\n%H:%M"))
        plt.legend()
        plt.xlabel('time')
        ax2 = plt.subplot(212)
        labels = 'water_level'
        plt.plot(time, y, label=labels)
        plt.autoscale(enable=True, axis='x', tight=True)
        #plt.xticks(rotation=90)
        ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d\n%H:%M"))
        plt.legend()
        plt.xlabel('time')

time_span = 12*24 # 288

train_set = CustomDataset(X_train, y_train, time_span)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=15, shuffle=True)

test_set = CustomDatasetTest(X_test, y_test, time_span)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=876)

dummy1, dummy2 = next(iter(train_loader))

result = pd.DataFrame(columns=['test'], index=y.iloc[last_index_train:].index)

dummy1t, dummy2t = next(iter(test_loader))

dummy_time = dummy1t.numpy()
time_vector = []
for i in range(dummy_time.shape[0]):
    time_vector.extend(dummy_time[i, :, :, -1])

time_vector = np.concatenate(time_vector).ravel()

time_vector = time_vector * (378687450 - 150) + 150
time = pd.to_datetime(time_vector, unit='s', origin=pd.Timestamp('2013-01-01 00:00:00+00:00').timestamp(), utc=True)

# Fill result
result.loc[time, 'test'] = dummy2t.detach().numpy().ravel()


#device = "cpu"

# model = torch.nn.Sequential(
#     # input 1xtime_spanx12
#     torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 1)), # output 288-3+1=286x12x2
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(kernel_size=(4, 1)), # output 286/4=71x12x2
    
#     torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(7, 1)), # output 71-7+1=65x12x4
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(kernel_size=(4, 1)), # output 65/4=16x12x4
    
#     torch.nn.Flatten(), # 1x768
#     torch.nn.Linear(16*12*4, time_span)
# )
#model.to(device)
model = torch.load(os.path.join(directory, "model2026_07_07_14_00.pt"), weights_only=False)


model(dummy1.to(torch.float32)).detach().numpy()

summary(model, input_data=dummy1.to(torch.float32), verbose=2)

epochs = 3
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

# for epoch in range(epochs):
#     model.train()
#     for batch in train_loader:
#         X_batch, y_batch = batch

# #        X_batch = X_batch.to(device)
# #        y_batch = y_batch.to(device)
        
#         model.zero_grad()

#         y_pred = model(X_batch.to(torch.float32))

#         loss = criterion(y_pred, y_batch)

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
model.eval()

result = pd.DataFrame(columns=['test', 'prediction'], index=y.iloc[last_index_train:].index)
dummy1t, dummy2t = next(iter(test_loader))
with torch.no_grad():
    y_pred = model(dummy1t.to(torch.float32))

dummy_time = dummy1t.numpy()
time_vector = []
for i in range(dummy_time.shape[0]):
    time_vector.extend(dummy_time[i, :, :, -1])

time_vector = np.concatenate(time_vector).ravel()

time_vector = time_vector * (378687450 - 150) + 150
time = pd.to_datetime(time_vector, unit='s', origin=pd.Timestamp('2013-01-01 00:00:00+00:00').timestamp(), utc=True)

# Fill result
result.loc[time, 'test'] = dummy2t.detach().numpy().ravel()
result.loc[time, 'prediction'] = y_pred.numpy().ravel()


result.plot()
(result['test'] - result['prediction']).abs().mean()
    
    

for i in range(len(test_loader)):
    # Get batch from test_loader
    dummy1t, dummy2t = next(iter(test_loader))
    
    # Get time of batch
    time = np.squeeze(dummy1t).numpy()[:,-1]
    time = time * (378687450 - 150) + 150
    time = pd.to_datetime(time, unit='s', origin=pd.Timestamp('2013-01-01 00:00:00+00:00').timestamp(), utc=True)
    
    # Make prediction on batch
    pred_dummy1t = model(dummy1t.to(torch.float32))

    # Fill result
    result.loc[time, 'test'] = np.squeeze(dummy2t.detach().numpy())
    result.loc[time, 'prediction'] = np.squeeze(pred_dummy1t.detach().numpy())


loss_val_total = 0
predictions, true_vals = [], []
for batch in test_loader:
    X_batch, y_batch = batch
    with torch.no_grad():
        y_pred = model(torch.tensor(X_batch, dtype=torch.float32))
    y = torch.tensor(y_batch, dtype=torch.float32)
    loss = criterion(y_pred, y)
    loss_val_total += loss.item()
    predictions.extend(y_pred.detach().cpu().numpy())
    true_vals.extend(y_batch.cpu().numpy())
    
loss_test_avg = loss_val_total / len(test_loader)
predictions = np.array(predictions)
true_vals = np.array(true_vals)
print(loss_test_avg, mean_squared_error(true_vals, predictions), mean_absolute_error(true_vals, predictions))

# Save model
torch.save(model, os.path.join(directory, "model2026_07_07_14_00.pt"))

