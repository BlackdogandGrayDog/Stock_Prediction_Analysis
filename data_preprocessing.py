#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:47:24 2023

@author: ericwei
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_acquisition_store import save_to_csv
from data_acquisition_store import save_to_bitio
#%%
def data_read_alignment():
    '''This function reads csv files stored from previous module.'''
    stock_csv = os.path.join('Dataset', 'AAPL.csv')
    inflation_csv = os.path.join('Dataset', 'Inflation_data.csv')
    gdp_csv = os.path.join('Dataset', 'GDP_data.csv')
    interest_csv = os.path.join('Dataset', 'Interest_data.csv')
    stock_df = pd.read_csv(stock_csv)
    inflation_df = pd.read_csv(inflation_csv)
    gdp_df = pd.read_csv(gdp_csv)
    interest_df = pd.read_csv(interest_csv)
    stock_df = stock_df.set_index(['Date'])
    inflation_df = inflation_df.set_index(['DATE'])
    gdp_df = gdp_df.set_index(['DATE'])
    interest_df = interest_df.set_index(['DATE'])
    inflation_df = inflation_df.reindex(stock_df.index)
    gdp_df = gdp_df.reindex(stock_df.index)
    interest_df = interest_df.reindex(stock_df.index)
    save_to_bitio(inflation_df, 'inflation_aligned')
    save_to_bitio(gdp_df, 'gdp_aligned')
    save_to_bitio(interest_df, 'interest_aligned')
    save_to_csv('inflation_aligned.csv', inflation_df)
    save_to_csv('gdp_aligned.csv', gdp_df)
    save_to_csv('interest_aligned.csv', interest_df)
    return stock_df, inflation_df, gdp_df, interest_df

#%%
def outlier_detect(df_frame, df_name):
    '''This function used for detect outliers.'''
    print('\n')
    print('Processing ' + df_name + ' data...')
    # Calculate quartiles (IQR Test)
    q1_value = df_frame.quantile(0.25)
    q3_value = df_frame.quantile(0.75)
    # Calculate IQR
    iqr_value = q3_value - q1_value
    outliers = (df_frame < (q1_value - 1.5 * iqr_value)) | (df_frame > (q3_value + 1.5 * iqr_value))
    # Calculate quartiles (IQR Test)
    outliers_in_df = df_frame.loc[outliers]
    print('The outliers from IQR Test in ' + df_name + ' is: ')
    print(outliers_in_df)
    print('Total Number of outliers fron IQR Test: ' + str(len(outliers_in_df)))
    # Outliers from Z-Score
    df_z_scores = np.abs(stats.zscore(df_frame))
    outliers = df_z_scores > 3
    z_outliers_in_df = df_z_scores.loc[outliers]
    print('\nThe outliers from Z-Score Test in ' + df_name + ' is: ')
    print(z_outliers_in_df)
    print('Total Number of outliers fron Z-Score Test is: ' + str(len(z_outliers_in_df)))
    # Detect Missing values
    num_miss = df_frame.isnull().sum()
    print('\nNumber of missing values in ' + df_name + ' is: \n' + str(num_miss))
    return outliers_in_df, z_outliers_in_df, num_miss
#%%
def stock_data_plot(stock_df, z_stock_volume_outlier, stock_volume_outlier):
    '''This function used for visualise data.'''
    print('\nGenerating AAPL stock price on Open, High, Low, Close, Adj Close plot')
    plt.figure(figsize=(10,8))
    # Plot the closing price
    plt.plot(stock_df['Open'], label = 'Open', lw = 1.7)
    plt.plot(stock_df['High'], label = 'High', lw = 0.9)
    plt.plot(stock_df['Low'], label = 'Low', lw = 1.1)
    plt.plot(stock_df['Close'], label = 'Close', lw = 1.3)
    plt.plot(stock_df['Adj Close'], label = 'Adj Close', lw = 1.5)
    # Add labels and title
    plt.xlabel('Date', fontsize = 12, fontweight='bold')
    plt.ylabel('Price', fontsize = 12, fontweight='bold')
    plt.title('AAPL Stock Price', fontsize = 15, fontweight='bold')
    plt.xticks(stock_df.index[::120], rotation=30,fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.grid()
    plt.legend()
    file_name = 'AAPL_plot.png'
    file_path = os.path.join('Images', file_name)
    # Check if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')

    print('\nGenerating AAPL stock price on Volume and outlier points plot')
    plt.figure(figsize=(10,20))
    plt.subplot(2,1,1)
    plt.plot(stock_df['Volume'], label = 'Volume', lw = 1.5)
    # labels and title
    plt.xlabel('Date', fontsize = 12, fontweight='bold')
    plt.ylabel('Price', fontsize = 12, fontweight='bold')
    plt.title('AAPL Stock Volume Price', fontsize = 15, fontweight='bold')
    plt.xticks(stock_df.index[::120], rotation=30,fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.scatter(z_stock_volume_outlier.index,
                stock_df['Volume'][z_stock_volume_outlier.index],
                c='red', label = 'Z-Score Outliers')
    plt.grid()
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(stock_df['Volume'], label = 'Volume', lw = 1.5)
    # labels and title
    plt.xlabel('Date', fontsize = 12, fontweight='bold')
    plt.ylabel('Price', fontsize = 12, fontweight='bold')
    plt.title('AAPL Stock Volume Price', fontsize = 15, fontweight='bold')
    plt.xticks(stock_df.index[::120], rotation=30,fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.scatter(stock_volume_outlier.index,
                stock_df['Volume'][stock_volume_outlier.index], c='green', label = 'IQR Outliers')
    plt.grid()
    plt.legend()
    file_name = 'AAPL_volume_plot.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')
    print('\nGenerating AAPL stock price Boxplot on Open, High, Low, Close, Adj Close plot')
    plt.figure(figsize=(10,7))
    sns.boxplot(data = stock_df[['Open','Close','High','Low']])
    plt.title("AAPL Stock Box Plots", fontsize = 15, fontweight='bold')
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)

    # Text and arrow label
    q1 = stock_df['Open'].quantile(0.25)
    q3 = stock_df['Open'].quantile(0.75)
    iqr = q3 - q1
    plt.text(x=0.3, y = q1,  s='Q1', fontweight='bold', color='red', fontsize = 15)
    plt.text(x=0.3, y = q3,  s='Q3', fontweight='bold', color='red', fontsize = 15)
    plt.text(x=0.3, y = iqr, s='IQR', fontweight='bold', color='red', fontsize = 15)
    plt.arrow(x = 0.2, y = q1 + (iqr * 0.5), dx = 0, dy = 0.5 * iqr,
              color = 'orange', width = 0.05,
              head_length = 0.5, head_width = 0.2)
    plt.arrow(x = 0.2, y = q1 + (iqr * 0.5), dx = 0, dy = -0.5 * iqr,
              color = 'orange', width = 0.05,
              head_length = 0.5, head_width = 0.2)

    file_name = 'AAPL_boxplot.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')
    print('\nGenerating AAPL stock price Boxplot on volume plot')
    plt.figure(figsize=(10,7))
    sns.boxplot(data = stock_df[['Volume']], orient="h")
    plt.title("AAPL Volume Box Plots", fontsize = 15, fontweight='bold')
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    file_name = 'AAPL_Volume_boxplot.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')
#%%
def outlier_interpolate(stock_df, z_stock_volume_outlier):
    '''This function used for remove outliers by interpolation and save to sql and csv file.'''
    print('\nRemoving outliers from Z_score on volume of AAPL...')
    stock_df_interpolated = stock_df
    stock_df['Volume'][z_stock_volume_outlier.index] = np.nan
    stock_df_interpolated['Volume'] = stock_df['Volume'].interpolate()
    print('Removed Successful')
    print('\nGenerating AAPL stock price plot on Volume of orginal and interpolated one...')
    plt.figure(figsize = (10,10))
    plt.plot(stock_df.index, stock_df_interpolated['Volume'],
             label = 'Interpolated', lw = 1.5, c = 'r')
    plt.plot(stock_df.index, stock_df['Volume'], label = 'Original', lw = 2, c = 'b')

    plt.xlabel('Date', fontsize = 12, fontweight='bold')
    plt.ylabel('Price', fontsize = 12, fontweight='bold')
    plt.title('AAPL Stock Volume Price', fontsize = 15, fontweight='bold')
    plt.xticks(stock_df.index[::120], rotation=30,fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.grid()
    plt.legend()

    file_name = 'AAPL_Volume_boxplot.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')
    print('\nSaving in progress...')
    save_to_csv('AAPL_clean.csv', stock_df_interpolated)
    save_to_bitio(stock_df_interpolated, 'aapl_clean')
    print('Saving successful.')
    return stock_df_interpolated

#%%
def df_scaler(df_frame, df_name):
    '''This function used for scaling dataframe by set mean to 0 and unit variance.'''
    print('\nRemoving the mean and scaling to unit variance in ' + df_name + '...')
    scaler = StandardScaler()
    scaler.fit(df_frame)
    df_scaled = scaler.transform(df_frame)
    print('Scaling successed.')
    df_scaled = pd.DataFrame(df_scaled, index = df_frame.index, columns = df_frame.columns)
    print('\nSaving in progress...')
    save_to_bitio(df_scaled, df_name)
    df_name = df_name + '.csv'
    save_to_csv(df_name, df_scaled)
    print('Saving successful.')
    return df_scaled

#%%
def pca_reduce(stock_df_clean_scaled):
    '''This function used for PCA reducing and save reduced dataset tp csv and sql.'''
    pca = PCA().fit(stock_df_clean_scaled)
    plt.figure(figsize = (7,7))
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('number of components', fontsize = 12, fontweight='bold')
    plt.ylabel('explained variance ratio', fontsize = 12, fontweight='bold')
    plt.title('Explained Variance Ratio', fontsize = 15, fontweight='bold')
    plt.grid()
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)

    file_name = 'Explained Variance Ratio.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the plot
    plt.savefig(file_path)
    plt.close()
    pca = PCA(n_components = 2)
    pca.fit(stock_df_clean_scaled)
    stock_df_clean_scaled_reduced = pca.transform(stock_df_clean_scaled)
    plt.figure(figsize = (10,7))
    plt.scatter(stock_df_clean_scaled_reduced[:, 0],
                stock_df_clean_scaled_reduced[:, 1], marker='x', s=50)
    plt.title('Scatter Plot of PCA Components', fontsize = 15, fontweight='bold')
    plt.xlabel('PCA Component 1', fontsize = 12, fontweight='bold')
    plt.ylabel('PCA Component 2', fontsize = 12, fontweight='bold')
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.grid()
    plt.legend()
    file_name = 'PCA_plot.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save the plot
    plt.savefig(file_path)
    plt.close()
    stock_df_clean_scaled_reduced = pd.DataFrame(np.array(stock_df_clean_scaled_reduced)
                                                 , columns=['PCA1', 'PCA2'])
    save_to_csv('AAPL_clean_scaled_reduced.csv',
                stock_df_clean_scaled_reduced)
    save_to_bitio(stock_df_clean_scaled_reduced, 'aapl_clean_scaled_reduced')
    return stock_df_clean_scaled_reduced
#%%
def data_preprocessing():
    ''' This function summarises and execute all functions in this module'''
    stock_df, inflation_df, gdp_df, interest_df = data_read_alignment()
    _, _, _ = outlier_detect(stock_df['Open'], 'Stock Open')
    _, _, _ = outlier_detect(stock_df['High'], 'Stock High')
    _, _, _ = outlier_detect(stock_df['Low'], 'Stock Low')
    _, _, _ = outlier_detect(stock_df['Close'], 'Stock Close')
    _, _, _ = outlier_detect(stock_df['Adj Close'], 'Stock Adj Close')
    stock_volume_outlier, z_stock_volume_outlier, _ = outlier_detect(stock_df['Volume']
                                                                     , 'Stock Volume')
    _, _, _ = outlier_detect(inflation_df['Inflation Rate'], 'Inflation')
    _, _, _ = outlier_detect(gdp_df['GDP'], 'GDP')
    _, _, _ = outlier_detect(interest_df['Interest Rate'], 'Interest')
    stock_data_plot(stock_df, z_stock_volume_outlier, stock_volume_outlier)
    stock_df_clean = outlier_interpolate(stock_df, z_stock_volume_outlier)
    stock_df_clean_scaled = df_scaler(stock_df_clean, 'aapl_clean_scaled')
    inflation_df_clean_scaled = df_scaler(inflation_df, 'inflation_clean_scaled')
    gdp_df_clean_scaled = df_scaler(gdp_df, 'gdp_clean_scaled')
    interest_df_clean_scaled = df_scaler(interest_df, 'interest_clean_scaled')
    stock_df_clean_scaled_reduced = pca_reduce(stock_df_clean_scaled)
    return stock_df_clean_scaled, inflation_df_clean_scaled, gdp_df_clean_scaled, interest_df_clean_scaled, stock_df_clean_scaled_reduced
