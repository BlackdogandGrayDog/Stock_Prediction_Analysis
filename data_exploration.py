#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 02:18:11 2023

"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
#%%
def data_read_scaled():
    ''' This function reads csv file of scaled and clean preprocessed data from previous stage'''
    stock_dcs_path = os.path.join('Dataset', 'aapl_clean_scaled.csv')
    inflation_dcs_path = os.path.join('Dataset', 'inflation_clean_scaled.csv')
    gdp_dcs_path = os.path.join('Dataset', 'gdp_clean_scaled.csv')
    interest_dcs_path = os.path.join('Dataset', 'interest_clean_scaled.csv')
    stock_dcs = pd.read_csv(stock_dcs_path).set_index('Date')
    inflation_dcs = pd.read_csv(inflation_dcs_path).set_index('Date')
    gdp_dcs = pd.read_csv(gdp_dcs_path).set_index('Date')
    interest_dcs = pd.read_csv(interest_dcs_path).set_index('Date')
    return stock_dcs, inflation_dcs, gdp_dcs, interest_dcs
#%%
def data_read_cleaned():
    ''' This function reads csv file of unscaled but clean preprocessed data from previous stage'''
    stock_dc_path = os.path.join('Dataset', 'AAPL_clean.csv')
    inflation_dc_path = os.path.join('Dataset', 'inflation_aligned.csv')
    gdp_dc_path = os.path.join('Dataset', 'gdp_aligned.csv')
    interest_dc_path = os.path.join('Dataset', 'interest_aligned.csv')
    stock_dc = pd.read_csv(stock_dc_path).set_index('Date')
    inflation_dc = pd.read_csv(inflation_dc_path).set_index('Date')
    gdp_dc = pd.read_csv(gdp_dc_path).set_index('Date')
    interest_dc = pd.read_csv(interest_dc_path).set_index('Date')
    return stock_dc, inflation_dc, gdp_dc, interest_dc
#%%
def trend_analysis_plot(stock_dcs, inflation_dcs, gdp_dcs, interest_dcs, df_name):
    ''' This function plots the trend of each input dataframe'''
    print('\nGenerating ' + df_name + ' Trend plot:')
    # figure and axes
    _, axs = plt.subplots(5, 1, figsize=(10, 7), sharex = True)
    # plot data on each subplot
    axs[0].plot(stock_dcs['Adj Close'], label = 'Adj Close', c = 'b', lw = 2)
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(stock_dcs['Volume'], label = 'Volume', c = 'r', lw = 2)
    axs[1].grid()
    axs[1].legend()
    axs[2].plot(inflation_dcs['Inflation Rate'], label = 'Inflation Rate', c = 'y', lw = 2)
    axs[2].grid()
    axs[2].legend()
    axs[3].plot(gdp_dcs['GDP'], label = 'GDP', c = 'g', lw = 2)
    axs[3].grid()
    axs[3].legend()
    axs[4].plot(interest_dcs['Interest Rate'], label = 'Interest Rate', c = 'm', lw = 2)
    axs[4].grid()
    axs[4].legend()
    # set x-tick labels
    plt.xticks(stock_dcs.index[::120], rotation=10,fontsize = 8)
    plt.xlabel('Date', fontsize = 10, fontweight='bold')
    file_name = df_name + ' Trend_plot.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')
#%%
def seasonality_analysis_plot(df_frame, df_name, stock_dcs):
    ''' This function plots the seasonality of each input dataframe'''
    print('\nGenerating ' + df_name + ' seasonality plot:')
    result = seasonal_decompose(df_frame, model='additive', period = 90)
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid
    _, axs = plt.subplots(3, 1, figsize=(10, 7), sharex = True)
    axs[0].plot(trend, label = 'Trend', c = 'b', lw = 1)
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(seasonal, label = 'Seasonal', c = 'r', lw = 1)
    axs[1].grid()
    axs[1].legend()
    axs[2].plot(residual, label = 'Residual', c = 'g', lw = 1)
    axs[2].grid()
    axs[2].legend()
    plt.xticks(stock_dcs.index[::120], rotation=10,fontsize = 8)
    plt.xlabel('Date', fontsize = 10, fontweight='bold')
    file_name = df_name + ' Seasonality_plot.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')

#%%
def features_correlation(stock_dcs):
    ''' This function returns and plots the correlation between each feature dataframe'''
    print('\nGenerating feature correlation plot:')
    sns.pairplot(stock_dcs, kind="scatter")
    file_name = 'features_correlation.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')
    corr_adj_open = stock_dcs['Adj Close'].corr(stock_dcs['Open'], method='pearson')
    corr_adj_high = stock_dcs['Adj Close'].corr(stock_dcs['High'], method='pearson')
    corr_adj_low = stock_dcs['Adj Close'].corr(stock_dcs['Low'], method='pearson')
    corr_adj_close = stock_dcs['Adj Close'].corr(stock_dcs['Close'], method='pearson')
    corr_adj_volume = stock_dcs['Adj Close'].corr(stock_dcs['Volume'], method='pearson')
    print('The correlation (Pearson Coefficient) between Adj Close and Open is: '
          + str(corr_adj_open))
    print('The correlation (Pearson Coefficient) between Adj Close and High is: '
          + str(corr_adj_high))
    print('The correlation (Pearson Coefficient) between Adj Close and Low is: '
          + str(corr_adj_low))
    print('The correlation (Pearson Coefficient) between Adj Close and Close is: '
          + str(corr_adj_close))
    print('The correlation (Pearson Coefficient) between Adj Close and Volume is: '
          + str(corr_adj_volume))
    print('\nGenerating covariance matrix plot:')
    cov_matrix = np.cov(stock_dcs[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].transpose())
    plt.figure(figsize = (10, 7))
    sns.heatmap(cov_matrix, annot=True, fmt='.5f',
                xticklabels = stock_dcs.columns, yticklabels = stock_dcs.columns)
    file_name = 'feature_covariance_matrix.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')

#%%
def external_correlation(stock_dcs, inflation_dcs, gdp_dcs, interest_dcs):
    ''' This function returns and plots the correlation between input features'''
    print('\nGenerating external correlation plot:')
    _, axs = plt.subplots(3, 1, figsize=(10, 7))
    axs[0].scatter(stock_dcs['Adj Close'],
                   inflation_dcs['Inflation Rate'],
                   label = 'Correlation Adj Close and Inflation Rate', c = 'b')
    axs[0].grid()
    axs[0].legend()
    axs[1].scatter(stock_dcs['Adj Close'],
                   gdp_dcs['GDP'],
                   label = 'Correlation Adj Close and GDP', c = 'g')
    axs[1].grid()
    axs[1].legend()
    axs[2].scatter(stock_dcs['Adj Close'],
                   interest_dcs['Interest Rate'],
                   label = 'Correlation Adj Close and Interest Rate', c = 'r')
    axs[2].grid()
    axs[2].legend()
    file_name = 'external_correlation.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')
    corr_adj_inf = stock_dcs['Adj Close'].corr(inflation_dcs['Inflation Rate'], method='pearson')
    corr_adj_gdp = stock_dcs['Adj Close'].corr(gdp_dcs['GDP'], method='pearson')
    corr_adj_int = stock_dcs['Adj Close'].corr(interest_dcs['Interest Rate'], method='pearson')
    print('The correlation (Pearson Coefficient) between Adj Close and Inflation Rate is: '
          + str(corr_adj_inf))
    print('The correlation (Pearson Coefficient) between Adj Close and GDP is: '
          + str(corr_adj_gdp))
    print('The correlation (Pearson Coefficient) between Adj Close and Interest Rate is: '
          + str(corr_adj_int))
    corr_adj_inf_k = stock_dcs['Adj Close'].corr(inflation_dcs['Inflation Rate'], method='kendall')
    corr_adj_gdp_k = stock_dcs['Adj Close'].corr(gdp_dcs['GDP'], method='kendall')
    corr_adj_int_k = stock_dcs['Adj Close'].corr(interest_dcs['Interest Rate'], method='kendall')
    print('The correlation (Kendall rank correlation) between Adj Close and Inflation Rate is: '
          + str(corr_adj_inf_k))
    print('The correlation (Kendall rank correlation) between Adj Close and GDP is: '
          + str(corr_adj_gdp_k))
    print('The correlation (Kendall rank correlation) between Adj Close and Interest Rate is: '
          + str(corr_adj_int_k))
    print('\nGenerating covariance matrix plot:')
    cov_matrix_adj_inf = np.cov(stock_dcs['Adj Close'], inflation_dcs['Inflation Rate'])
    cov_matrix_adj_gdp = np.cov(stock_dcs['Adj Close'], gdp_dcs['GDP'])
    cov_matrix_adj_int = np.cov(stock_dcs['Adj Close'], interest_dcs['Interest Rate'])
    _, axs = plt.subplots(3, 1, figsize=(5, 12))
    sns.heatmap(cov_matrix_adj_inf, annot=True, fmt='.5f',
                xticklabels = ['Adj Close', 'Inflation'],
                yticklabels = ['Adj Close', 'Inflation'], ax = axs[0])
    sns.heatmap(cov_matrix_adj_gdp, annot=True, fmt='.5f',
                xticklabels = ['Adj Close', 'GDP'],
                yticklabels = ['Adj Close', 'GDP'], ax = axs[1])
    sns.heatmap(cov_matrix_adj_int, annot=True, fmt='.5f',
                xticklabels = ['Adj Close', 'Interest Rate'],
                yticklabels = ['Adj Close', 'Interest Rate'], ax = axs[2])
    file_name = 'external_covariance_matrix.png'
    file_path = os.path.join('Images', file_name)
    # Detect if file already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the plot
    plt.savefig(file_path)
    plt.close()
    print('Generate Successful')
#%%
def hytest_significant_relation(df_frame, df_name, stock_dcs):
    ''' This function do the hypothesis test on significant relation
        between adj close and other auxilliary data
    '''
    print('\nHypothesis test on significant relation between Adj close and '
          + df_name + ' in progress...')
    stock_dcs['Adj Close Class'] = ['High' if x > 0 else 'Low' for x in stock_dcs['Adj Close']]
    df_frame['Class'] = ['High' if x > 0 else 'Low' for x in df_frame[df_name]]
    hy_df = pd.crosstab(stock_dcs['Adj Close Class'], df_frame['Class'])
    chi2, p_value, dof, expected = chi2_contingency(hy_df)
    print('Chi-squared test statistic: ', chi2)
    print('p-value: ', p_value)
    print('Degree of freedom: ', dof)
    print('Expected values: ', expected)
    # Set the significance level
    alpha = 0.05
    # Check if the p-value is less than the significance level 95%
    if p_value < alpha:
        print('There is a significant relationship between Adj Close and ' + df_name + '.')
    else:
        print('There is no significant relationship between Adj Close and ' + df_name + '.')
#%%
def hytest_significant_relation_change(df_frame, df_name, stock_dcs):
    ''' This function do the hypothesis test on significant relation
        between daily Open and adj close price
        and percentage change in other auxilliary data
    '''
    print('\nHypothesis test on relation between Change in Adj close and Percentage Change in '
          + df_name + ' in progress...')
    stock_dcs['Diff.'] = stock_dcs['Adj Close'] - stock_dcs['Open']
    df_frame['pct. change'] = df_frame[df_name].pct_change()
    stock_dcs['Diff. Sign'] = ['positive' if x > 0 else 'negative' for x in stock_dcs['Diff.']]
    df_frame['pct. change Sign'] = ['positive' if x > 0 else 'negative' for x in df_frame['pct. change']]
    hy_df = pd.crosstab(stock_dcs['Diff. Sign'], df_frame['pct. change Sign'])
    chi2, p_value, dof, expected = chi2_contingency(hy_df)
    print('Chi-squared test statistic: ', chi2)
    print('p-value: ', p_value)
    print('Degree of freedom: ', dof)
    print('Expected values: ', expected)
    # Set the significance level
    alpha = 0.05
    # Check if the p-value is less than the significance level 95%
    if p_value < alpha:
        print('There is a significant relationship between Change in Adj Close and '+ df_name +'.')
    else:
        print('There is no significant relationship between Change in Adj Close and '+ df_name +'.')
#%%
def data_exploration():
    ''' This function summarises and execute all functions in this module'''
    stock_dcs, inflation_dcs, gdp_dcs, interest_dcs = data_read_scaled()
    stock_dc, inflation_dc, gdp_dc, interest_dc = data_read_cleaned()
    trend_analysis_plot(stock_dcs, inflation_dcs, gdp_dcs, interest_dcs, 'Clean and Scaled')
    trend_analysis_plot(stock_dc, inflation_dc, gdp_dc, interest_dc, 'Clean')
    seasonality_analysis_plot(stock_dcs['Adj Close'], 'Clean and Scaled Adj Close', stock_dcs)
    seasonality_analysis_plot(stock_dcs['Volume'], 'Clean and Scaled Volume', stock_dcs)
    seasonality_analysis_plot(inflation_dcs['Inflation Rate'],
                              'Clean and Scaled Inflation Rate', stock_dcs)
    seasonality_analysis_plot(gdp_dcs['GDP'],
                              'Clean and Scaled GDP', stock_dcs)
    seasonality_analysis_plot(interest_dcs['Interest Rate'],
                              'Clean and Scaled Interest Rate', stock_dcs)
    seasonality_analysis_plot(stock_dc['Adj Close'],
                              'Clean Adj Close', stock_dcs)
    seasonality_analysis_plot(stock_dc['Volume'],
                              'Clean Volume', stock_dcs)
    seasonality_analysis_plot(inflation_dc['Inflation Rate'],
                              'Clean Inflation Rate', stock_dcs)
    seasonality_analysis_plot(gdp_dc['GDP'], 'Clean GDP', stock_dcs)
    seasonality_analysis_plot(interest_dc['Interest Rate'],
                              'Clean Interest Rate', stock_dcs)
    features_correlation(stock_dcs)
    external_correlation(stock_dcs, inflation_dcs, gdp_dcs, interest_dcs)
    hytest_significant_relation(inflation_dcs, 'Inflation Rate', stock_dcs)
    hytest_significant_relation(gdp_dcs, 'GDP', stock_dcs)
    hytest_significant_relation(interest_dcs, 'Interest Rate', stock_dcs)
    hytest_significant_relation_change(inflation_dcs, 'Inflation Rate', stock_dcs)
    hytest_significant_relation_change(gdp_dcs, 'GDP', stock_dcs)
    hytest_significant_relation_change(interest_dcs, 'Interest Rate', stock_dcs)
