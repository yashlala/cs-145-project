import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sklearn.metrics
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ConvergenceWarning

LIN_REG_RANGE = 50
HOLT_RANGE = 40
T_SIZE = 7
train2 = None

def MAPE(predicted, actual):
    assert len(predicted) == len(actual)
    res = 0
    for i in range(len(predicted)):
        diff = np.abs(predicted[i] - actual[i]) / np.abs(actual[i])
        res += diff
    return (res/len(predicted)) * 100

def get_opt_toy(param, t_size, day_range, opt_split):
    total_err = 0
    for state in np.unique(train2['Province_State']):
        split = train2.loc[train2['Province_State'] == state]
        min_mape = 100
        for i in range(40 - day_range, 41 + day_range):
          split_train = split[param].iloc[i:-t_size].to_numpy() # why start at 40? should this be tuned?
          split_test = split[param].iloc[-t_size:].to_numpy()
          model = Holt(split_train)
          model_fit = model.fit()
          predicted_cases = model_fit.forecast(t_size)
          mape = MAPE(predicted_cases, split_test)
          if mape < min_mape:
            min_mape = mape
            opt_split[state] = i
        total_err += len(predicted_cases) * min_mape
    return total_err/(t_size*50)

def linreg_opt_start(param, t_size, day_range, opt_starts, states):
    total_err = 0
    for state in states:
        min_mape = 100
        for i in range(states[state] - day_range, states[state] + day_range):
            split = train2.loc[train2['Province_State'] == state]
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
    split = train2.loc[train2['Province_State'] == state]
    
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

def train_full(param, opt_conf_start_holt, opt_death_start_holt):
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
    linreg_opt_start(param, T_SIZE, LIN_REG_RANGE, opt_lin_start, lin_states)

    
    for state in np.unique(train2['Province_State']):
        
        if state in lin_states:
            res[state] = linreg(param, opt_lin_start[state], state)
        else:
            # Holt Model
            split = train2.loc[train2['Province_State'] == state]
            if param == 'Deaths':
                split_train = split[param].to_numpy()[opt_death_start_holt[state]:]
            else:
                split_train = split[param].to_numpy()[opt_conf_start_holt[state]:] 
            model = Holt(split_train)
            model_fit = model.fit()
            predicted_cases = model_fit.forecast(7)
            res[state] = predicted_cases
    return res

if __name__ == "__main__":
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter('ignore', ConvergenceWarning)


    opt_conf_start = {}
    opt_death_start = {}
    train2 = pd.read_csv('./data/train_full.csv')
    train2_test = train2[-350:]
    train2 = train2[:-350]

    print("Finding optimal start dates for holt")
    pure_holt = get_opt_toy('Deaths', T_SIZE, HOLT_RANGE, opt_death_start)
    print("Got optimal start dates for deaths")
    pure_d_holt = get_opt_toy('Confirmed', T_SIZE, HOLT_RANGE, opt_conf_start)
    print('Got holt optimal start dates')


    print("Beginning to train")
    conf_results = train_full('Confirmed', opt_conf_start, opt_death_start)
    print("Finished training on confirmed")
    death_results = train_full('Deaths', opt_conf_start, opt_death_start)

    conf_mape = 0
    death_mape = 0
    for state in np.unique(train2_test['Province_State']):
        actual_df = train2_test.loc[train2_test['Province_State'] == state]
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





