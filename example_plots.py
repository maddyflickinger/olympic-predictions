# -*- coding: utf-8 -*-
#import the necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from datetime import datetime

#%%
#import the data set
data = pd.read_csv('swimming.csv')

#drop NA values and reset the index
data = data.dropna()
data = data[data['Rank'] != 0]
data = data.reset_index(drop = True)

#format the results string to get all times in the same format
for i in range(len(data)):
    item = data.iloc[i,8]
    if item.count('.') == 0:
        data.iloc[i,8] = item+'.00'

for i in range(len(data)):
    item = data.iloc[i,8].split('.')[0] + '.' + data.iloc[i,8].split('.')[1]
    if item.count(':')==0:
        data.iloc[i,8]='00:00:'+item
    elif item.count(':')==1:
        data.iloc[i,8]='00:'+item
    elif item.count(':')==2:
        data.iloc[i,8]=item

#remove rows not formatted correctly
data = data[~data['Results'].str.contains('est')]

#create the date time columns
data['Date_Time'] = pd.to_datetime(data['Results'], format='%H:%M:%S.%f')
data['Time'] = data['Date_Time'].dt.time


for i in range(len(data)):
    data.iloc[i,11] = data.iloc[i,11].minute * 60 + data.iloc[i,11].second

data['Time'] = data['Time'].astype(str)
data['Final_Time'] = [None] * len(data)

#convert all of the times into seconds and milliseconds
for i in range(len(data)):
    if data.iloc[i,1] == 2020:
        ms = data.iloc[i,8].split('.')[1]
        ms = float(ms)
        ms = ms/100
        ms = str(ms)
        ms = ms.split('.')[1]
        data.iloc[i,12] = data.iloc[i,11] + '.' + ms
    else:
        string = data.iloc[i,8].split('.')[1]
        if len(string) == 3:
            ms = float(string)
            ms = ms/1000
            ms = str(ms)
            ms = ms.split('.')[1]
            data.iloc[i,12] = data.iloc[i,11] + '.' + ms
        if len(string) == 6:
            ms = float(string)
            ms = ms/1000000
            ms = str(ms)
            ms = ms.split('.')[1]
            data.iloc[i,12] = data.iloc[i,11] + '.' + ms
     
#change the final time to a numeric
data['Final_Time'] = data['Final_Time'].astype(float)
   
#get separate data frames for bronze, silver, gold finishers
gold = data[data['Rank'] == 1]
silver = data[data['Rank'] == 2]
bronze = data[data['Rank'] == 3]

#%%
#focus on one event: 100m backstroke
#format to start from the year 1948
#women
w100bk = gold[(gold["Gender"] == 'Women') & (gold['Distance (in meters)'] == '100m') & (gold['Stroke'] == 'Backstroke')]
w100bk = w100bk.sort_values(by = 'Year')
w100bk = w100bk[['Year', 'Date_Time', 'Final_Time']]
w100bk = w100bk.drop(3603) #drop a duplicate time
w100bk = w100bk.reset_index(drop = True)
w100bk = w100bk.iloc[4:23]

#men
m100bk = gold[(gold['Gender'] == 'Men') & (gold['Distance (in meters)'] == '100m') & (gold['Stroke'] == 'Backstroke')]
m100bk = m100bk.sort_values(by = 'Year')
m100bk = m100bk[['Year', 'Date_Time', 'Final_Time']]
m100bk = m100bk.reset_index(drop = True)

#add missing year as average of surrounding years
newrow = {'Year': 1964, 'Date_Time': pd.to_datetime('00:01:01.05', format='%H:%M:%S.%f'), 'Final_Time': 61.05} 
m100bk.loc[len(m100bk)] = newrow #add to the end of the dataframe
m100bk = m100bk.sort_values(by = 'Year')
m100bk = m100bk.reset_index(drop = True)
m100bk = m100bk.iloc[5:24]


#%%
#plot the data to look at the time series
year = m100bk['Year'].tolist()
wtime = w100bk['Date_Time'].tolist()
mtime = m100bk['Date_Time'].tolist()
plt.plot(year, wtime, color = 'blue', linewidth = 5, label = 'Women')
plt.plot(year, mtime, color = 'red', linewidth = 5, label = 'Men')
plt.xlabel('Olympic Year')
plt.ylabel('Time')
plt.legend(loc = 'upper right', ncol = 1)
plt.title('100m Backstroke')
plt.show()

#%% 100 backstroke graph
#create an exponential smoothing model for the series
for i in range(len(w100bk)):
    year = w100bk.iloc[i,0]
    date = datetime(year,12,31).date()
    w100bk.iloc[i,0] = date
    
for i in range(len(m100bk)):
    year = m100bk.iloc[i,0]
    date = datetime(year,12,31).date()
    m100bk.iloc[i,0] = date
    
w100bk = w100bk.drop('Date_Time', axis = 1)
w100bk['Year'] = pd.to_datetime(w100bk['Year'])
w100bk = w100bk.set_index('Year')

m100bk = m100bk.drop('Date_Time', axis = 1)
m100bk['Year'] = pd.to_datetime(m100bk['Year'])
m100bk = m100bk.set_index('Year')

#womens result
model_w = ExponentialSmoothing(w100bk['Final_Time'], trend='add', seasonal=None, damped_trend = False, freq='4A')
w100bk_result = model_w.fit()
w100bk_result.summary()
fcastw = w100bk_result.forecast(1).rename("Women")
print(fcastw)

#mens result
model_m = ExponentialSmoothing(m100bk['Final_Time'], trend='add', seasonal=None, damped_trend = True, freq='4A')
m100bk_result = model_m.fit()
m100bk_result.summary()
fcastm = m100bk_result.forecast(1).rename("Men")
print(fcastm)

#create a graph of the forecasted times
plt.figure(figsize=(12, 8))
plt.plot(w100bk, marker="o", color="blue")
plt.plot(m100bk, marker="o", color="red")
plt.plot(w100bk_result.fittedvalues, color="blue")
(line1,) = plt.plot(fcastw, marker="*", color="blue", ms = 15)
plt.plot(m100bk_result.fittedvalues, color="red")
(line2,) = plt.plot(fcastm, marker="*", color="red", ms = 15)
#plt.legend([line1, line2], [fcastw.name, fcastm.name])
plt.xlabel('Olympic Year')
plt.ylabel('Time (Seconds)')
plt.title('100m Backstroke')

#%% follow the same process for the 400 free event to show the impact of covid
w400free = gold[(gold["Gender"] == 'Women') & (gold['Distance (in meters)'] == '400m') & (gold['Stroke'] == 'Freestyle')]
w400free = w400free.sort_values(by = 'Year')
w400free = w400free[['Year', 'Date_Time', 'Final_Time']]
w400free = w400free.reset_index(drop = True)
w400free = w400free.iloc[4:23]

#men
m400free = gold[(gold['Gender'] == 'Men') & (gold['Distance (in meters)'] == '400m') & (gold['Stroke'] == 'Freestyle')]
m400free = m400free.sort_values(by = 'Year')
m400free = m400free[['Year', 'Date_Time', 'Final_Time']]
m400free = m400free.reset_index(drop = True)
m400free = m400free.iloc[5:24]

#create an exponential smoothing model for the series
for i in range(len(w400free)):
    year = w400free.iloc[i,0]
    date = datetime(year,12,31).date()
    w400free.iloc[i,0] = date
    
for i in range(len(m400free)):
    year = m400free.iloc[i,0]
    date = datetime(year,12,31).date()
    m400free.iloc[i,0] = date
    
w400free = w400free.drop('Date_Time', axis = 1)
w400free['Year'] = pd.to_datetime(w400free['Year'])
w400free = w400free.set_index('Year')

m400free = m400free.drop('Date_Time', axis = 1)
m400free['Year'] = pd.to_datetime(m400free['Year'])
m400free = m400free.set_index('Year')

#womens result
model_w = ExponentialSmoothing(w400free['Final_Time'], trend='add', seasonal=None, damped_trend = True, freq='4A')
w400free_result = model_w.fit()
w400free_result.summary()
fcastw = w400free_result.forecast(1).rename("Women")
print(fcastw)

#mens result
model_m = ExponentialSmoothing(m400free['Final_Time'], trend='add', seasonal=None, damped_trend = False, freq='4A')
m400free_result = model_m.fit()
m400free_result.summary()
fcastm = m400free_result.forecast(1).rename("Men")
print(fcastm)

#create a graph of the forecasted times
plt.figure(figsize=(12, 8))
plt.plot(w400free, marker="o", color="blue")
plt.plot(m400free, marker="o", color="red")
plt.plot(w400free_result.fittedvalues, color="blue")
(line1,) = plt.plot(fcastw, marker="*", color="blue", ms = 15)
plt.plot(m400free_result.fittedvalues, color="red")
(line2,) = plt.plot(fcastm, marker="*", color="red", ms = 15)
#plt.legend([line1, line2], [fcastw.name, fcastm.name])
plt.xlabel('Olympic Year')
plt.ylabel('Time (Seconds)')
plt.title('400m Freestyle')
