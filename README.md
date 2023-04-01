# Data Acquisition and Processing Systems (DAPS) (ELEC0136)

Welcome to the final assignment of the _Data Acquisition and Processing Systems_ (DAPS) course ELEC0136 at UCL.

In this assignment, we simulated a real-world data analysis on stock price. The assignment is divided into five individual tasks: i) Data aquisition, ii) Data storage, iii) Data preprocessing, iv) Data exploration and Data inference. Task i. and ii. are combined into one python files of data_aquisition_store.py, while others are implmented in the corresponding files.

The all fundamental packages and environment settings are detailed in the environment.yml file, which can be automatically installed via 'conda env create -f environment.yml' command. Please be advised, the 'spyder-kernels' package is used for Spyder IDE, which is not mandatroy if using other IDEs. The following list gives the require packages:

```
    - numpy
    - scipy
    - pandas
    - scikit-learn
    - matplotlib
    - seaborn
    - wbdata
    - missingpy
    - pymongo
    - keras
    - tensorflow
    - datetime
    - yfinance
    - fredapi
    - pandas_datareader
    - tweepy
    - textblob
    - sqlalchemy
    - psycopg2-binary
    - statsmodels
    - pylint
    - spyder-kernels
```

The main.py file will automatically run through all the tasks and functions with printed intructions, including stock data, auxilliary data acquisition, storage, preprocessing, visualisation, transformation, trend analysis, hypothesis testing and final inference on future adjusted close price.

## Task 1 Data Aquisition
In this task, three different methods for acquiring AAPL (Apple.inc) stock historical prices from April 2017 to April 2022 are being presented, including from Yahoo finance API, pre-downloaded csv file, and acquiring from URL. Also, auxilliary dataset such as inflation rate, GDP, twitter sentiment and interest rate are acquired for future prediction.

## Task 2 Data Storage
In this task, we presented different storage methods for dataset, that are stored on local PC and online SQL database named Bit.io.

## Task 3 Data Preprocessing
In this task, outliers and missing values are being detected and replaced. Then, datasets are being visulaised and we standardise the data and reduce the dimensionality. Processed datasets are also being saved on both local PC as csv files and online bit.io database.

## Task 4 Data Exploration
In this task, trand, seasonality, random noise are being analysised. Also, correlation between different features within AAPL stock price dataset and Adjusted close price with other auxilliary datasets are being calculated and plotted. Any unusual behaviours are being justified and we finally present the hypothesis testing.

## Task 5 Data Inference
In this task, we built two LSTM models which takes previous 10 days Adj close price data or attached with auxilliary data for training, and then predict the future 30 days (a month period) of price. Then the model is tested via a new test dataset stored in pre-downloaded csv file, which is stock price in May 2022. Both models are being save in .h5 format.

## Dataset Folder
Two pre-downloaded dataset AAPL.csv and AAPL_test.csv being stored at initial, when run the main.py file, more datasets will being stored in this folder.

## Images Folder
Initially empty, when run the main.py file, more plotted images will being stored in this folder.

## Models Folder
Initially empty, when run the main.py file, two LSTM models will being stored in this folder.

