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
    '''Returns the optimal start dates for the HOLT simulation.

    param: Attribute to optimize (eg. "Deaths")
    num_days_validation: How many days to validate on. 
    day_range: The number of days to train over.
    '''

    start_dates = {}
    total_err = 0
    for state in np.unique(train['Province_State']):
        split = train.loc[train['Province_State'] == state]
        min_mape = 100
        for i in range(HOLT_START_DATE - day_range//2, 
                HOLT_START_DATE + day_range//2):
          split_train = split[param].iloc[i:-num_days_validation].to_numpy() 
          split_valid = split[param].iloc[-num_days_validation:].to_numpy()
          model = Holt(split_train)
          model_fit = model.fit()
          predicted_cases = model_fit.forecast(num_days_validation)
          mape = MAPE(predicted_cases, split_valid)
          if mape < min_mape:
            min_mape = mape
            start_dates[state] = i
        total_err += len(predicted_cases) * min_mape

    # TODO: Return individualized state MAPE -- probably go state by state in
    # toplevel method? 
    return start_dates, total_err/(num_days_validation*50)


def linreg_start_dates(param, num_days_validation, day_range, states):
    '''Returns the optimal start dates for the Linear Regression simulation.

    param: Attribute to optimize (eg. "Deaths")
    num_days_validation: How many days to validate on. 
    day_range: The number of days to train over.
    states: hardcoded dictionary of which states to predict over with linear
    regression. 
    '''
    # TODO: Remove state logic, move to toplevel run_all. 

    start_dates = {}
    total_err = 0
    for state in states:
        min_mape = 100
        for i in range(states[state] - day_range//2, states[state] + day_range//2):
            split = train.loc[train['Province_State'] == state]
            split_train = split[param].iloc[i:-num_days_validation].to_numpy()
            split_test = split[param].iloc[-num_days_validation:].to_numpy()

            x_axis = len(split_train)
            #x axis is days after april 1st + start_date
            ids = np.linspace(states[state], x_axis+states[state],x_axis)
            cal_lin_train_x = ids.reshape(-1,1)

            model = LinearRegression().fit(cal_lin_train_x, split_train)
            future = np.linspace(0, 
                    states[state] + x_axis + num_days_validation,
                    states[state] + x_axis + num_days_validation)
            cal_lin_test_x = future.reshape(-1,1)

            predicted_y = model.predict(cal_lin_test_x)
            predicted_cases = predicted_y 

            mape = MAPE(predicted_cases[-num_days_validation::], split_test)
            if mape < min_mape:
                min_mape = mape
                start_dates[state] = i
                
        total_err += len(predicted_cases[-num_days_validation::]) * min_mape

    # TODO: Return individualized state MAPE -- probably go state by state in
    # toplevel method? 
    return start_dates, total_err/(num_days_validation*50)


def linreg(param, start_date, state):
    '''Perform linear regression over the given state. 
    param: Attribute to optimize (eg. "Deaths").
    start_date: The day to start the linear regression. Days before this point
        will not be trained on. 
    state: The state data to regress over.
    '''
    # TODO: states, you know the jam
    
    split = train.loc[train['Province_State'] == state]
    
    split_train = split[param].iloc[start_date::].to_numpy()
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

    pred_data = predicted_cases[-NUM_DAYS_TESTING:]
    return pred_data



def train_full(param, num_days_validation, start_dates):

    # TODO: train each state-by-state, implement MAPE comparisons *here*. 
    res = {}
    
    start_conf = {'Hawaii': 80, 'Wyoming': 170, 'Arizona': 170, 'Alaska':170, 'Florida' : 170,
                 'Oklahoma':150, 'Maine': 170, 'Vermont': 170}
    start_death = {'Arizona': 170, 'Idaho': 170, 'Wyoming': 170, 'Vermont': 170, 'Maine': 170}
    lin_states = {}

    if param == 'Deaths':
        lin_states = start_death
    else: 
        lin_states = start_conf

    # TODO: change up state logic.
    linreg_start_dates_dict = linreg_start_dates(param, num_days_validation,
            LINREG_START_DATE_RANGE, lin_states)
    
    for state in np.unique(train['Province_State']):
        if state in lin_states:
            res[state] = linreg(param, linreg_start_dates_dict[state], state)
        else:
            # Holt Model
            split = train.loc[train['Province_State'] == state]
            split_train = split[param].to_numpy()[start_dates[state]:]
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
    conf_start_dates = {}
    death_start_dates = {}
    # TODO: record MAPE below
    death_holt_start_dates, _ = holt_start_dates('Deaths', 
            NUM_DAYS_VALIDATION,
            HOLT_START_DATE_RANGE)
    conf_holt_start_dates, _ = holt_start_dates('Confirmed',
            NUM_DAYS_VALIDATION,
            HOLT_START_DATE_RANGE)

    # Train the models given the optimized start dates. 
    death_results = train_full('Deaths', 
            NUM_DAYS_VALIDATION, death_holt_start_dates)
    conf_results = train_full('Confirmed', 
            NUM_DAYS_VALIDATION, conf_holt_start_dates)

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
