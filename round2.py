import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn.metrics
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pmdarima import auto_arima

# ################
# Hyperparameters
# ################

# The number of days to test on. 
# eg. All but the last 7 days are trained on, the last 7 days will be reserved
# for testing data. 
NUM_DAYS_TESTING = 7

# The number of days to validate on. 
# eg. All but the last 7 days are trained on, the last 7 days will be reserved
# for validateing data. 
# TODO: tune this guy to make it smaller 
NUM_DAYS_VALIDATION = 7

# Models are validated with different "training start dates". 
# These values determine the beginnings and endings of the dates to check over. 
HOLT_START_DATE = 0
HOLT_END_DATE = 80
LINREG_START_DATE = 120
LINREG_END_DATE = 220

train = None


def MAPE(predicted, actual):
    assert len(predicted) == len(actual)
    res = 0
    for i in range(len(predicted)):
        diff = np.abs(predicted[i] - actual[i]) / np.abs(actual[i])
        res += diff
    return (res/len(predicted)) * 100


def holt_start_dates(param, num_days_validation, train_data):
    '''Returns the optimal start dates for the HOLT simulation.

    param: Attribute to optimize (eg. "Deaths")
    num_days_validation: How many days to validate on. 
    train_data: The dataset to train HOLT over. 
    '''

    start_date = None
    total_err = 0
    min_mape = 100
    for i in range(HOLT_START_DATE, HOLT_END_DATE):
        split_train = train_data[param].iloc[i:-num_days_validation].to_numpy() 
        split_valid = train_data[param].iloc[-num_days_validation:].to_numpy()
        model = Holt(split_train)
        model_fit = model.fit()
        predicted_cases = model_fit.forecast(num_days_validation)
        mape = MAPE(predicted_cases, split_valid)
        if mape < min_mape:
            min_mape = mape
            start_date = i
    total_err += len(predicted_cases) * min_mape

    return start_date, total_err/(num_days_validation)


def linreg_start_dates(param, num_days_validation, train_data):
    '''Returns the optimal start dates for the Linear Regression simulation.

    param: Attribute to optimize (eg. "Deaths")
    num_days_validation: How many days to validate on. 

    regression. 
    train_data: The dataset to train HOLT over. 
    '''

    optimal_start_date = None
    total_err = 0
    min_mape = 100
    for i in range(LINREG_START_DATE, LINREG_END_DATE): 
        split_train = train_data[param].iloc[i:-NUM_DAYS_VALIDATION].to_numpy()
        split_test = train_data[param].iloc[-NUM_DAYS_VALIDATION:].to_numpy()

        x_axis = len(split_train)
        cal_lin_train_x = np.linspace(0, x_axis, x_axis).reshape(-1, 1)
        cal_lin_test_x = np.linspace(0, x_axis + num_days_validation, 
                x_axis + num_days_validation).reshape(-1, 1)

        model = LinearRegression().fit(cal_lin_train_x, split_train)
        predicted_y = model.predict(cal_lin_test_x)
        predicted_cases = predicted_y 

        mape = MAPE(predicted_cases[-num_days_validation::], split_test)
        if mape < min_mape:
            min_mape = mape
            optimal_start_date = i
            
    total_err += len(predicted_cases[-num_days_validation::]) * min_mape

    return optimal_start_date, total_err/(num_days_validation)


def linreg(param, start_date, train_data):
    '''Perform linear regression over the given state. 
    param: Attribute to optimize (eg. "Deaths").
    start_date: The day to start the linear regression. Days before this point
        will not be trained on. 
    state: The state data to regress over.
    '''
    
    split_train = train_data[param].iloc[start_date::].to_numpy()
    x_axis = len(split_train)
    # x axis is days after april 1st + start_date
    ids = np.linspace(start_date, x_axis+start_date,x_axis)
    cal_lin_train_x = ids.reshape(-1, 1)

    model = LinearRegression().fit(cal_lin_train_x, split_train)
    # TODO: We use NUM_DAYS_TESTING because we want to "predict" 7 days. This
    # works when we want to "predict" our test dataset, but will FAIL if we use
    # this method to check our validation dataset (eg. "is linear or holt
    # better?"). We'll want to consolidate these later, but for now mind the
    # gap. 
    future = np.linspace(0, 
            start_date + x_axis + NUM_DAYS_TESTING, 
            start_date + x_axis + NUM_DAYS_TESTING)
    cal_lin_test_x = future.reshape(-1, 1)

    predicted_cases = model.predict(cal_lin_test_x)

    return predicted_cases[-NUM_DAYS_TESTING:]

def train_arima(param, num_days_validation, train_data):
    '''
    Do not need any hyperparameters 
    :param: Attribute to train on
    :num_days_validation: the size of validating
    returns the MAPE of the ARIMA
    '''
    train = train_data[param].values[:-num_days_validation]
    validation = train_data[param].values[-num_days_validation:]

    arima_model = auto_arima(train, seasonal=False, 
            supress_warnings=True, start_p=1, start_q=0, max_p=10)
    predicted_cases = arima_model.predict(num_days_validation)
    arima_MAPE = MAPE(predicted_cases, validation)
    return arima_MAPE


def train_full(param, num_days_validation, train_data):
    res = {}
    
    for state in np.unique(train['Province_State']):
        data_df = train_data.loc[train['Province_State'] == state]
        lr_start, lr_mape = linreg_start_dates(param, num_days_validation, data_df)
        holt_start, holt_mape = holt_start_dates(param, num_days_validation, data_df)
        print("{} MAPE for LR:".format(state), lr_mape)
        print("{} MAPE for HOLT:".format(state), holt_mape)
        arima_mape = train_arima(param, NUM_DAYS_VALIDATION, data_df)
        print("{} MAPE for ARIMA:".format(state), arima_mape)

        min_mape = min(lr_mape, holt_mape, arima_mape)
        pred_values = []
        if arima_mape == min_mape:
            print("{} will use ARIMA".format(state))
            arima_train = data_df[param].to_numpy()
            arima_model = auto_arima(arima_train, seasonal=False,
                    supress_warnings=True, start_p=1, start_q=0, 
                    max_p=10)
            pred_values = arima_model.predict(NUM_DAYS_TESTING)
        elif lr_mape == min_mape:
            print("{} will use Linear Regression".format(state))
            print("{} will be the start day".format(lr_start))
            pred_values = linreg(param, lr_start, data_df)
        else:
            print("{} will use Holt".format(state))
            print("{} will be the start day".format(holt_start))
            holt_train_data = data_df[param].to_numpy()[holt_start:]
            model = Holt(holt_train_data)
            model_fit = model.fit()
            pred_values = model_fit.forecast(NUM_DAYS_TESTING)
        print()
        res[state] = pred_values
    return res


if __name__ == "__main__":
    # Ignore matrix warnings, etc. 
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter('ignore', ConvergenceWarning)
    simplefilter('ignore', UserWarning)

    # Segregate the training and testing data
    train = pd.read_csv('./data/train_full.csv')
    test = train[-NUM_DAYS_TESTING * 50:]
    train = train[:-NUM_DAYS_TESTING * 50]

    # Train the models. 
    print("Training Deaths")
    death_results = train_full('Deaths', 
            NUM_DAYS_VALIDATION, train)
    print('Training Confirmed')
    conf_results = train_full('Confirmed', 
            NUM_DAYS_VALIDATION, train)

    # Output to CSV 
    test = pd.read_csv('./data/test_round2.csv')
    test['Confirmed'] = test['Confirmed'].astype(float)
    test['Deaths'] = test['Deaths'].astype(float)
    # Save confirmed cases. 
    for index, state in enumerate(np.unique(train['Province_State'])):
        predicted_cases = conf_results[state]
        for j in range(len(predicted_cases)):
            cur_index = index + j * 50
            test.at[cur_index, 'Confirmed'] = predicted_cases[j]
    # Save deaths. 
    for index, state in enumerate(np.unique(train['Province_State'])):
        predicted_cases = death_results[state]
        for j in range(len(predicted_cases)):
            cur_index = index + j * 50
            test.at[cur_index, 'Deaths'] = predicted_cases[j]
    submission = test
    submission = submission.drop(['Province_State', 'Date'], axis = 1)
    submission.to_csv('./data/round2.csv', index=False)

    # Print results
    conf_mape = 0
    death_mape = 0
    for state in np.unique(test['Province_State']):
        actual_df = test.loc[test['Province_State'] == state]
        actual_c = actual_df['Confirmed'].to_numpy()
        actual_d = actual_df['Deaths'].to_numpy()
        state_d = MAPE(death_results[state], actual_d)
        state_c = MAPE(conf_results[state], actual_c)
        print(state)
        print("MAPE death:", state_d)
        print("MAPE conf:", state_c)
        death_mape += state_d
        conf_mape += state_c
    conf_mape /= 50
    death_mape /= 50
    print('TOTAL conf mape:', conf_mape)
    print('TOTAL death mape:', death_mape)
