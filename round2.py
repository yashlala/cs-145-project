import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn.metrics
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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
NUM_DAYS_VALIDATION = 7

# Models are validated with different "training start dates". 
# These values determine the middle of the range of starting dates to check
# over. 
# 
# TODO: triple check this logic, switch to a "start date" and a "end date"
HOLT_START_DATE = 40
LINREG_START_DATE = 170

# These values determine the range of days to check over. 
# TODO: triple check as above. 
HOLT_START_DATE_RANGE = 80
LINREG_START_DATE_RANGE = 100

train = None


def MAPE(predicted, actual):
    assert len(predicted) == len(actual)
    res = 0
    for i in range(len(predicted)):
        diff = np.abs(predicted[i] - actual[i]) / np.abs(actual[i])
        res += diff
    return (res/len(predicted)) * 100


def holt_start_dates(param, num_days_validation, day_range):
    '''Returns the optimal start dates for the HOLT simulation

    param: Attribute to optimize (eg. "Deaths")
    num_days_validation: How many days to validate on. 
    day_range: The number of days to train over
    '''

    opt_split = {}
    total_err = 0
    for state in np.unique(train['Province_State']):
        split = train.loc[train['Province_State'] == state]
        min_mape = 100
        for i in range(HOLT_START_DATE - day_range//2, 
                HOLT_START_DATE + day_range//2):
          split_train = split[param].iloc[i:-num_days_validation].to_numpy() # why start at 40? should this be tuned?
          split_valid = split[param].iloc[-num_days_validation:].to_numpy()
          model = Holt(split_train)
          model_fit = model.fit()
          predicted_cases = model_fit.forecast(num_days_validation)
          mape = MAPE(predicted_cases, split_valid)
          if mape < min_mape:
            min_mape = mape
            opt_split[state] = i
        total_err += len(predicted_cases) * min_mape
    return total_err/(num_days_validation*50)


def linreg_start_dates(param, t_size, day_range, opt_starts, states):
    total_err = 0
    for state in states:
        min_mape = 100
        for i in range(states[state] - day_range, states[state] + day_range):
            split = train.loc[train['Province_State'] == state]
            split_train = split[param].iloc[i:-t_size].to_numpy()
            split_test = split[param].iloc[-t_size:].to_numpy()

            x_axis = len(split_train)
            #x axis is days after april 1st + start_date
            ids = np.linspace(states[state], x_axis+states[state],x_axis)
            cal_lin_train_x = ids.reshape(-1,1)

            model = LinearRegression().fit(cal_lin_train_x, split_train)
            future = np.linspace(0,states[state] + x_axis+t_size,states[state]+x_axis+t_size)
            cal_lin_test_x = future.reshape(-1,1)

            predicted_y = model.predict(cal_lin_test_x)
            predicted_cases = predicted_y 

            mape = MAPE(predicted_cases[-t_size::], split_test)
            if mape < min_mape:
                min_mape = mape
                opt_starts[state] = i
                
        total_err += len(predicted_cases[-t_size::]) * min_mape
    return total_err/(t_size*50)

def linreg(param, start_date, state):
    split = train.loc[train['Province_State'] == state]
    
    split_train = split[param].iloc[start_date::].to_numpy()
    x_axis = len(split_train)
    #x axis is days after april 1st + start_date
    ids = np.linspace(start_date, x_axis+start_date,x_axis)
    cal_lin_train_x = ids.reshape(-1,1)

    model = LinearRegression().fit(cal_lin_train_x, split_train)
    future = np.linspace(0,start_date + x_axis+7,start_date+x_axis+7)
    cal_lin_test_x = future.reshape(-1,1)

    predicted_cases = model.predict(cal_lin_test_x)

    pred_data = predicted_cases[-7:]
    return pred_data

def train_full(param, t_size, start_dates_conf_holt, start_dates_death_holt):
    res = {}
    
    start_conf = {'Hawaii': 80, 'Wyoming': 170, 'Arizona': 170, 'Alaska':170, 'Florida' : 170,
                 'Oklahoma':150, 'Maine': 170, 'Vermont': 170}
    start_death = {'Arizona': 170, 'Idaho': 170, 'Wyoming': 170, 'Vermont': 170, 'Maine': 170}
    lin_states = {}

    if param == 'Deaths':
        lin_states = start_death
    else: 
        lin_states = start_conf

    opt_lin_start = {}
    linreg_start_dates(param, t_size, LIN_REG_RANGE, opt_lin_start, lin_states)

    
    for state in np.unique(train['Province_State']):
        
        if state in lin_states:
            res[state] = linreg(param, opt_lin_start[state], state)
        else:
            # Holt Model
            split = train.loc[train['Province_State'] == state]
            if param == 'Deaths':
                split_train = split[param].to_numpy()[start_dates_death_holt[state]:]
            else:
                split_train = split[param].to_numpy()[start_dates_conf_holt[state]:] 
            model = Holt(split_train)
            model_fit = model.fit()
            predicted_cases = model_fit.forecast(NUM_DAYS_TESTING)
            res[state] = predicted_cases
    return res

if __name__ == "__main__":
    # Ignore matrix warnings, etc. 
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter('ignore', ConvergenceWarning)

    # Segregate the training and testing data
    train = pd.read_csv('./data/train_full.csv')
    test = train[-NUM_DAYS_TESTING * 50:]
    train = train[:-NUM_DAYS_TESTING * 50]

    # Find optimal training start-dates for our models. 
    # Optimal dates to start training for confirmed and death cases. 
    conf_start_dates_conf = {}
    death_start_dates_death = {}
    pure_holt = holt_start_dates('Deaths', VALIDATION_DATA_SIZE, HOLT_RANGE, start_dates_death)
    pure_d_holt = holt_start_dates('Confirmed', VALIDATION_DATA_SIZE, HOLT_RANGE, start_dates_conf)

    # Train the models given the optimized start dates. 
    conf_results = train_full('Confirmed', VALIDATION_DATA_SIZE, start_dates_conf, start_dates_death)
    death_results = train_full('Deaths', VALIDATION_DATA_SIZE, start_dates_conf, start_dates_death)

    # TODO: Output to CSV 

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
