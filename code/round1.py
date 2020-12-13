import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sklearn.metrics
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import Holt
from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ConvergenceWarning


train = None

def linreg(param, start_date, state):
    '''
    Trains a linear regression model on a state after inputted start date
    returns the predicted data
    '''
    p_days = 167 # 0 based
    
    #x axis is days after april 1st + start_date
    ids = np.linspace(start_date, 141,142-start_date)
    train_x = ids.reshape(-1,1)
    future = np.linspace(0,p_days,p_days+1)
    test_x = future.reshape(-1,1)

    
    res = {}
    data = train.loc[train['Province_State'] == state]   
    train_y = data[param][start_date::]
    model = LinearRegression().fit(train_x, train_y)


    predicted_y = model.predict(test_x) #integer numbers
    last_day = int(train_y.iloc[[len(train_y)-1]])

    if(last_day < 10000 and param != 'Deaths'):
        temp = predicted_y + (last_day-predicted_y[141])
    elif(predicted_y[141] < last_day):
        m, b = np.polyfit([142,p_days],[last_day, predicted_y[p_days-1]],1)
        temp = m*future + b
    else:
        temp = predicted_y - (predicted_y[141]-last_day)

    pred_data = temp[142::]
    return pred_data

def train_full(param):
    '''
    Trains traditional Holt and Linear reg models on the round1 data
    returns a hash map of the data where the keys are the states
    the buckets are a 26-indexed array that are the predicted values for 9-01 to 9-26
    '''
    res = {}
    for state in np.unique(train['Province_State']):
        if (param == 'Confirmed'):#using linear regression on these states
            if (state == 'Hawaii'):
                res[state] = linreg(param, 80, state)
                continue
            if (state == 'Wyoming'):
                res[state] = linreg(param, 120, state)
                continue
            if (state == 'Arizona'):
                res[state] = linreg(param, 125, state)
                continue
            if (state == 'Alaska'):
                res[state] = linreg(param, 120, state)
                continue
            if (state == 'Florida'):
                res[state] = linreg(param, 130, state)
                continue
            if (state == 'Montana'):
                res[state] = linreg(param, 135, state)
                continue
            if (state == 'Oklahoma'):
                res[state] = linreg(param, 138, state)
                continue
        if (param == 'Deaths'):#using linear regression on these states
            if (state == 'Arizona'):
                res[state] = linreg(param, 80, state)
                continue
            if (state == 'Idaho'):
                res[state] = linreg(param, 60, state)
                continue
            if (state == 'Hawaii'):
                res[state] = linreg(param, 130, state)
                continue
            if (state == 'West Virginia'):
                res[state] = linreg(param, 130, state)
                continue
            if (state == 'Virginia'):
                res[state] = linreg(param, 0, state)
                continue
            if (state == 'Wyoming'):
                res[state] = linreg(param, 120, state)
                continue
            if (state == 'Montana'):
                res[state] = linreg(param, 125, state)
                continue
            
        split = train.loc[train['Province_State'] == state]
        split_train = split[param].to_numpy()[40::]
        model = Holt(split_train)
        model_fit = model.fit()
        predicted_cases = model_fit.forecast(26)
        res[state] = predicted_cases
    return res

if __name__ == "__main__":
    #ignore future warnings
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter('ignore', ConvergenceWarning)
    train = pd.read_csv('./data/train.csv')
    pred_conf = train_full('Confirmed')
    pred_death = train_full('Deaths')
    test = pd.read_csv('./data/test.csv')
    test['Confirmed'] = test['Confirmed'].astype(float)
    test['Deaths'] = test['Deaths'].astype(float)
    for index, state in enumerate(np.unique(train['Province_State'])):
        predicted_cases = pred_conf[state]
        for j in range(len(predicted_cases)):
            cur_index = index + j * 50
            test.at[cur_index, 'Confirmed'] = predicted_cases[j]
            # test['Confirmed'].iloc[cur_index] = predicted_cases[j]
    for index, state in enumerate(np.unique(train['Province_State'])):
        predicted_cases = pred_death[state]
        for j in range(len(predicted_cases)):
            cur_index = index + j * 50
            test.at[cur_index, 'Deaths'] = predicted_cases[j]
    submission = test
    submission = submission.drop(['Province_State', 'Date'], axis = 1)
    submission.to_csv('./round1.csv', index=False)
