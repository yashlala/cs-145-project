import pandas as pd 
import csv
import requests
from io import StringIO

import datetime
CSV_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'

days = [
    '11-23-2020.csv',
    '11-24-2020.csv',
    '11-25-2020.csv',
    '11-26-2020.csv',
    '11-27-2020.csv',
    '11-28-2020.csv',
    '11-29-2020.csv',
    '11-30-2020.csv',
    '12-01-2020.csv',
    '12-02-2020.csv',
    '12-03-2020.csv',
    '12-04-2020.csv',
    '12-05-2020.csv'
]
df = None
def convert_date(x):
    date = x[:10]
    date = datetime.datetime.strptime(date, '%Y-%m-%d')
    output = date - datetime.timedelta(days=1)
    return output.strftime('%m-%d-%Y')
all_days = list()
cols_needed = ['Confirmed', 'Deaths', 'Date', 'Province_State']
with requests.Session() as s:
    for day in days:
        print(day)
        download = s.get(CSV_URL + day)

        decoded_content = download.content.decode('utf-8')
        df = pd.read_csv(StringIO(decoded_content))
        df = df.loc[df['Province_State'] != 'American Samoa']
        df = df.loc[df['Province_State'] != 'Virgin Islands']
        df = df.loc[df['Province_State'] != 'Guam']
        df = df.loc[df['Province_State'] != 'Diamond Princess']
        df = df.loc[df['Province_State'] != 'Grand Princess']
        df = df.loc[df['Province_State'] != 'Northern Mariana Islands']
        df = df.loc[df['Province_State'] != 'Puerto Rico']
        df = df.loc[df['Province_State'] != 'District of Columbia']
        df['Date'] = df['Last_Update'].apply(convert_date)
        df = df.drop(df.columns.difference(cols_needed), axis=1)
        cols = df.columns.tolist()
        cols = [cols[0]] + [cols[-1]] + cols[1:3]
        df = df[cols] 
        all_days.append(df)


all_df = pd.concat(all_days).reset_index(drop=True)
all_df.index.name = 'ID'

all_df.to_csv("train_afternov22.csv")

df = pd.read_csv('train_round2.csv')


all_df = pd.concat([df, all_df]).reset_index(drop=True)

all_df.drop('ID', axis=1, inplace=True)
all_df.index.name = 'ID'
all_df.to_csv("train_full.csv")

