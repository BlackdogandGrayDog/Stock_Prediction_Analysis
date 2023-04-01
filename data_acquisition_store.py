#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 02:04:57 2023

"""
import os
import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
import tweepy
from textblob import TextBlob
from sqlalchemy import create_engine
#%%
def save_to_csv(df_name, df_frame):
    ''' This function is used for saving dataframes to local csv file'''
    # check if file already exists
    des = os.path.join('Dataset', df_name)
    if os.path.exists(des):
        print(f"{df_name} already exists.")
    else:
        # store dataframe to local csv file
        df_frame.to_csv(des, index=True)
        print(f"{df_name} has been created.")

#%%
def save_to_bitio(df_frame, df_name):
    ''' This function is used for saving dataframes to online bit.io sql db'''
    print('\n')
    print('Connecting to Bit.io SQL DB...')
    eng = create_engine('postgresql://ericwei09:v2_3y3XG_Zy6Y3PgkCMbdhL2FnhrzdSD@db.bit.io:5432/ericwei09/MagnusManatee',
                        isolation_level="AUTOCOMMIT")
    try:
        # Try connecting to the database
        eng.connect()
        print("Connection successful.")
        # Insert dataframe to SQL
        print('Inserting ' + df_name + ' to Bit.io SQL DB...')
        df_frame.to_sql(df_name, eng, if_exists='replace')
        print("Insert successfully.")
    except Exception:
        # If connection fails, print the error message and terminates
        print("Connection failed, please check APIs or might due to subscription ends.")
        return df_frame
    # Read dataframe to SQL
    print('Reading' + df_name + ' from Bit.io SQL DB...')
    df_r = pd.read_sql_query('SELECT * FROM '+ df_name, eng)
    print("Read successfully.")
    print(df_r)
    return df_r
#%%
def acquire_stock_api(ticker, start_date, end_date):
    ''' This function is used for getting stock data from yfinance API'''
    print('\n')
    print('Acquring Stock data from Yahoo API:')  # using yfinance package
    stock_data_api = yf.download(ticker, start = start_date, end = end_date)
    print('Successul Acquired.')
    print(stock_data_api)
    num_observations = stock_data_api.shape[0]
    num_features = stock_data_api.shape[1]
    print('The dataset has ' + str(num_observations)
          + ' observations, and ' + str(num_features) + ' features.')
    print('The name of each feature is:')
    print(list(stock_data_api.columns))
    # store dataframe to local csv file
    save_to_csv('AAPL_API.csv', stock_data_api)
    # store dataframe to Bit.io SQL DB
    save_to_bitio(stock_data_api, 'stock_data_api')
    return stock_data_api

#%%
def acquire_stock_csv(csv_file):
    ''' This function is used for getting stock data from downloaded csv file'''
    # Read the CSV file into a DataFrame
    print('\n')
    print('Acquring Stock data from Local CSV file:')
    stock_data_csv  = pd.read_csv(csv_file, index_col = 0)
    print('Successul Acquired.')
    print(stock_data_csv)
    # store dataframe to local csv file
    save_to_csv('AAPL.csv', stock_data_csv)
    # store dataframe to Bit.io SQL DB
    save_to_bitio(stock_data_csv, 'stock_data_csv')
    return stock_data_csv
#%%
def acquire_stock_url(ticker, start_date, end_date):
    ''' This function is used for getting stock data from url provided'''
    print('\n')
    print('Acquring Stock data using Yahoo URL:')
    start = pd.to_datetime([start_date]).astype(int)[0]//10**9 # convert to unix timestamp.
    end = pd.to_datetime([end_date]).astype(int)[0]//10**9 # convert to unix timestamp.
    url = 'https://query1.finance.yahoo.com/v7/finance/download/' + ticker + '?period1=' + str(start) + '&period2=' + str(end) + '&interval=1d&events=history'
    stock_data_url = pd.read_csv(url, index_col = 0)
    print('Successul Acquired.')
    print(stock_data_url)
    # store dataframe to local csv file
    save_to_csv('AAPL_URL.csv', stock_data_url)
    # store dataframe to Bit.io SQL DB
    save_to_bitio(stock_data_url, 'stock_data_url')
    return stock_data_url
#%%
def acquire_inflation_data(start_date, end_date):
    ''' This function is used for getting inflation data from FRED API'''
    print('\n')
    print('Acquring Inflation data from Federal Reserve Economic Data (FRED):')
    # Get the inflation rate data from FRED for specific date range
    inflation_data = pdr.get_data_fred('CPIAUCNS', start = start_date, end = end_date)
    print('Successul Acquired.')

    print('Resample to daily basis.')
    # Upsample the data to daily frequency via interpolate
    inflation_data = inflation_data.resample('D').interpolate()
    inflation_data.columns = ['Inflation Rate']
    print('Successful.')
    print(inflation_data)
    # store dataframe to local csv file
    save_to_csv('Inflation_data.csv', inflation_data)
    # store dataframe to Bit.io SQL DB
    save_to_bitio(inflation_data, 'inflation_data')
    return inflation_data
#%%
def acquire_gdp_data(start_date, end_date):
    ''' This function is used for getting gdp data from FRED API'''
    print('\n')
    print('Acquring GDP data from Federal Reserve Economic Data (FRED) API:')
    # Download GDP data for the specified date range
    gdp_data = pdr.get_data_fred('GDPC1', start = start_date, end = end_date)
    print('Successul Acquired.')
    print('Resample to daily basis.')
    # Upsample GDP data to daily basis
    gdp_data = gdp_data.resample('D').interpolate()
    gdp_data.columns = ['GDP']
    print('Successful.')
    print(gdp_data)
    # store dataframe to local csv file
    save_to_csv('GDP_data.csv', gdp_data)
    # store dataframe to Bit.io SQL DB
    save_to_bitio(gdp_data, 'gdp_data')
    return gdp_data

#%%
def acquire_interest_data(start_date, end_date):
    ''' This function is used for getting interest rate data from FRED API'''
    print('\n')
    print('Acquring Interest rate data from Federal Reserve Economic Data (FRED) API:')
    # Download Interest data for the specified date range
    interest_data = pdr.get_data_fred('FEDFUNDS', start = start_date, end = end_date)
    print('Successul Acquired.')
    print('Resample to daily basis.')
    # Upsample GDP data to daily basis
    interest_data = interest_data.resample('D').interpolate()
    interest_data.columns = ['Interest Rate']
    print('Successful.')
    print(interest_data)
    # store dataframe to local csv file
    save_to_csv('Interest_data.csv', interest_data)
    # store dataframe to Bit.io SQL DB
    save_to_bitio(interest_data, 'interest_data')
    return interest_data
#%%
def acquire_tweet_data():
    ''' This function is used for scrapy tweets from twitter API'''
    print('\n')
    print('Acquring Twitter data from Tweepy:')
    # set api key and secret and access token and token secret
    api_key = 'PFDE5FwOC955b1gp7TrAGxDux'
    api_secret = 'aqOGQqa5Pv47NVd5JA4n4NXaWtdCPJ5spImC5ozriuCrYn823u'
    access_token = '1614432674982019072-QIjocg9Vre4XVvScBYfrG5wlLWBaWo'
    access_token_secret = '3qMGfJF8fKjPkmorJFQW81M2wijyfuqEt2denPOPGpUy9'
    try:
        print('Data extraction in progess:')
        # authenticate to the twitter API
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        # create an empty pandas DataFrame to store the tweets
        tweets_df = pd.DataFrame(columns=['date', 'text','polarity'])
        # search for tweets
        for tweet in tweepy.Cursor(api.search_tweets, q='AAPL since:2017-04-01 until:2023-06-01'
                                   , lang='en').items(200):
            date = tweet.created_at
            text = tweet.text
            polarity = TextBlob(text).sentiment.polarity
            # Append the extracted data to the DataFrame
            tweets_df = tweets_df.append({'date': date,
                                          'text': text,'polarity':polarity}, ignore_index=True)
    except Exception:
        # If connection fails, print the error message and terminates
        print("Authentication failed, please check APIs.")
        return 0
    # Convert the date column to a datetime type
    tweets_df['date'] = pd.to_datetime(tweets_df['date'])
    # Set the date column as the index of the DataFrame
    tweets_df.set_index('date', inplace=True)
    print('Data extraction successful.')
    print('Only past 7 days tweets are provided by Twitter.')
    print(tweets_df)
    print('Not use this data in this case.')
    # store dataframe to local csv file
    save_to_csv('Tweets_data.csv', tweets_df)
    # store dataframe to Bit.io SQL DB
    save_to_bitio(tweets_df, 'tweets_data')
    return tweets_df
#%%
def data_aquisition_store(start_date, end_date, ticker):
    ''' This function summarises and execute all functions in this module'''
    stock_data_api = acquire_stock_api(ticker, start_date, end_date)
    stock_data_csv = acquire_stock_csv("Dataset/AAPL.csv")
    stock_data_url = acquire_stock_url(ticker, start_date, end_date)
    inflation_data = acquire_inflation_data(start_date, end_date)
    gdp_data = acquire_gdp_data(start_date, end_date)
    interest_data = acquire_interest_data(start_date, end_date)
    tweets_data = acquire_tweet_data()
    return stock_data_api, stock_data_csv, stock_data_url, inflation_data, gdp_data, interest_data, tweets_data
