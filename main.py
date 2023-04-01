""" Main.py file, run all functions in this assignment through it """
import data_acquisition_store as das # Module provide function on data acquisiton and store
import data_preprocessing as dp # Module provide function on data preprocessing
import data_exploration as de # Module provide function on data exploration
import data_inference as di # # Module provide function on data inference
def main():
    ''' This is main module, this module will automatically run through
        all the functions and tasks in this assignment with printed
        instructions.
    '''
    # Define specific time range for acquiring data
    start_date = '2017-04-01'
    end_date = '2022-04-01'
    ticker = 'AAPL'
    # acquire the necessary data and store data in Bit.io SQL database and local csv file
    _, _, _, _, _, _, _ = das.data_aquisition_store(start_date, end_date, ticker)
    # format, project and clean the data read from lcoal csv file and visulisation
    _, _, _, _, _ = dp.data_preprocessing()
    # perform exploratory data analysis on trend, seasonality, correlation and hypothesis testing
    de.data_exploration()
    di.data_inference()

if __name__ == "__main__":
    main()
