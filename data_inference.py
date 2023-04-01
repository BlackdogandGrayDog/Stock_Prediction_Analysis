#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 00:21:55 2023

"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers
from keras.optimizers import Adagrad
import matplotlib.pyplot as plt
from data_exploration import data_read_cleaned
#%%
# Create function to acquire training and validation datasets
def acquire_training_validation_data(train_df, look_back, pred_period, validation_split, multiple):
    '''This function is used to convert and generate training and validation data.'''
    train_data = train_df.values
    scaler = MinMaxScaler(feature_range=(0,1))
    if multiple is True:
        num_shape = train_data.shape[1]
    else:
        num_shape = 1
    train_data = scaler.fit_transform(train_data.reshape(-1, num_shape))
    x_train = []
    y_train = []
    for i in range(look_back, len(train_data)-pred_period):
        x_train.append(train_data[i-look_back : i, 0])
        y_train.append(train_data[i:i+pred_period, 0])
    x_train = np.array(x_train)
    y_train= np.array(y_train)
    split = int(len(x_train) * validation_split)
    return x_train[:split], y_train[:split], x_train[split:], y_train[split:]
#%%
def lstm_model(x_train):
    '''This function is used to build a lstm model.'''
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(50, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.summary()
    optimizer = Adagrad(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
#%%
def test_model(model):
    '''This function is used to test lstm model by test dataset.'''
    print('\n Testing in progress...')
    print('Predicting May 2022 Adj Close Price')
    aapl_test_path = os.path.join('Dataset', 'AAPL_test.csv')
    aapl_test = pd.read_csv(aapl_test_path).set_index('Date')
    scaler = MinMaxScaler(feature_range=(0,1))
    y_test = aapl_test['Adj Close'].values
    x_test = scaler.fit_transform(y_test.reshape(-1, 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    print('The Root Mean Sqaured Error of testing is:' + str(rmse))
    return aapl_test, y_test, predictions
#%%
def test_plot(aapl_test, y_test, predictions, name):
    '''This function is used to generate the performance plot.'''
    print('\nGenerating Test plot...')
    _, ax_s = plt.subplots(figsize=(10, 7))
    # Plot the real values
    ax_s.plot(aapl_test.index, y_test, label='Real Values')
    ax_s.fill_between(aapl_test.index, y_test*0.98, y_test*1.02, color='blue', alpha=0.1)
    # Plot the predictions
    ax_s.plot(aapl_test.index, predictions, label='Predictions')
    # Add a legend to the plot
    ax_s.legend()
    ax_s.grid()
    # Set the x-axis label
    ax_s.set_xlabel('Date', fontsize = 12, fontweight='bold')
    # Set the y-axis label
    ax_s.set_ylabel('Adj Close', fontsize = 12, fontweight='bold')
    plt.xticks(aapl_test.index[::3],fontsize = 12)
    plt.yticks(fontsize = 12, fontweight='bold')
    plt.title('Test and Prediction Plot', fontsize = 12, fontweight='bold')
    file_name = 'Test ' + name + '.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')
#%%
def residual_plt(aapl_test, y_test, predictions, name):
    '''This function is used to generate residual plot between real and prediction values.'''
    print('\nGenerating Residual plot...')
    residuals = np.subtract(y_test.reshape(-1, 1), predictions)
    plt.figure(figsize=(10, 7))
    plt.plot(aapl_test.index, residuals, label = 'Residual', c = 'm')
    plt.xlabel('Time', fontsize = 12, fontweight='bold')
    plt.ylabel('Residual (Price)', fontsize = 12, fontweight='bold')
    plt.title('Residual Plot', fontsize = 12, fontweight='bold')
    plt.xticks(aapl_test.index[::3],fontsize = 12)
    plt.yticks(fontsize = 12, fontweight='bold')
    plt.grid()
    plt.legend()
    file_name = 'Residual ' + name + '.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')
#%%
def model_saving(name, model):
    '''This function is used to save the lstm models.'''
    path = os.path.join('Models', name)
    # check if file already exists at the designated path
    if os.path.exists(path):
        os.remove(path)
    # save the model
    model.save(path)
#%%
def data_inference():
    ''' This function summarises and execute all functions in this module'''
    stock_dc, inflation_dc, gdp_dc, interest_dc = data_read_cleaned()
    x_train, y_train, x_val, y_val = acquire_training_validation_data(stock_dc['Adj Close'],
                                                                      10, 30, 0.8, False)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    model = lstm_model(x_train)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=16)
    model_saving('Stock_LSTM_Model.h5', model)
    aapl_test, y_test, predictions = test_model(model)
    test_plot(aapl_test, y_test, predictions, 'with only stock data')
    residual_plt(aapl_test, y_test, predictions, 'with only stock data')
    multi_train_df = pd.concat([stock_dc['Adj Close'], inflation_dc, gdp_dc, interest_dc], axis=1)
    x_train, y_train, x_val, y_val = acquire_training_validation_data(multi_train_df,
                                                                      10, 30, 0.8, False)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    model = lstm_model(x_train)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=16)
    model_saving('Auxiliary_LSTM_Model.h5', model)
    aapl_test, y_test, predictions = test_model(model)
    test_plot(aapl_test, y_test, predictions, 'with auxiliary data')
    residual_plt(aapl_test, y_test, predictions, 'with auxiliary data')
