# -*- coding: utf-8 -*-
"""
simulate and visualize the raindall data from Germany rainfall stations with Python


Created on Thu Mar  1 20:16:43 2018
#Spatial and Temporal
@author: Chi
"""
import numpy as np
from numpy.linalg import inv
import pandas as pd
import tarfile
import os
import timeit
import datetime
from datetime import timedelta
from datetime import datetime
import time
import glob
import pickle
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
import calendar

main_dir = os.path.join(r'C:\Users\Chi\Desktop\Dokument\Mater_Stuttgart\python')
os.chdir(main_dir)

StationPath = os.path.join(main_dir, r'oneStationReutlingen.txt');
assert StationPath

print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer()  # to get the runtime of the program


# =============================================================================
# input a txt file  and test , index_col=0
# =============================================================================

#divided by 10
StationData_row = pd.read_csv(StationPath, sep=';', header = None
                              , names=['value'], index_col= 0) / 10 

StationData_row.index = pd.to_datetime(StationData_row.index
                                       , format = "%Y-%m-%d %H:%M:%S")

#resample 1 hr
StationData_row = StationData_row.resample('1H').sum()

#make sure it start from 01-01-year (fulfill with Nan)
def fullyear(df):
    df.index = pd.to_datetime(df.index)
    y = df.index.year
    idx = pd.date_range('{}-01-01'.format(y.min()), '{}-12-31'.format(y.max()), freq='H')
    return df.reindex(idx)

StationData_row = fullyear(StationData_row)

StationData = pd.DataFrame()
StationData = StationData_row.reindex(pd.date_range(StationData_row.index.floor('D').min(), 
                 StationData_row.index.ceil('D').max(), freq='H'))[:-1]

StationData = StationData.set_index([StationData.index.floor('D')
                                    , StationData.index.hour])['value']\
                                    .unstack()#.apply(lambda x : str(list(x)), axis=1)


averY = StationData.groupby(StationData.index.strftime("%j")).mean()

# =============================================================================
# plot (normal) - amount (mm) and probability (%)(annual) with choosing frequency
# =============================================================================

def meshplot_amount(period_days):
    #set index Leap year and datatime format
    timeseries = pd.date_range('2016-01-01', periods=366, freq='D').to_series()
    df = averY.copy()
    df = df.set_index(timeseries)
    freq_string = '%d' %(period_days) + 'D'
    df_frequency = df.groupby(pd.TimeGrouper(freq = freq_string)).mean()
    
    plt.figure(1)

    plt.pcolor(df_frequency.values.transpose())
    #CS = plt.contourf(averY.values.transpose())

    plt.colorbar()

    plt.grid(linestyle='--', color='k', linewidth=0.05, alpha=0.35)
    #plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('Day', fontsize=14)
    plt.xticks(fontsize=14, rotation=0)

    plt.ylabel('Hour', fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.ylim(ymin = 0)

    plt.title('Hourly rainfall amount', fontsize=16)
    #plt.savefig(os.path.join(main_dir, 'dem_deu.pdf'))
    plt.show()

meshplot_amount(10)

def meshplot_prob(period_days):
    #set index Leap year and datatime format
    timeseries = pd.date_range('2016-01-01', periods=366, freq='D').to_series()
    df = averY.copy()
    df = df.set_index(timeseries)
    freq_string = '%d' %(period_days) + 'D'
    df_frequency = df.groupby(pd.TimeGrouper(freq = freq_string)).mean()
    
    #convert to pdf
    count = 0
    for index, row in df_frequency.iterrows():
        df_frequency.iloc[ count ] = row/row.sum()
        count = count + 1
    
    plt.figure(2)

    plt.pcolor(df_frequency.values.transpose())
    #CS = plt.contourf(averY.values.transpose())

    plt.colorbar(cmap='bone')

    plt.grid(linestyle='--', color='k', linewidth=0.05, alpha=0.35)
    #plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('Day', fontsize=14)
    plt.xticks(fontsize=14, rotation=0)

    plt.ylabel('Hour', fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.ylim(ymin = 0)

    plt.title('Hourly rainfall probability', fontsize=16)
    #plt.savefig(os.path.join(main_dir, 'dem_deu.pdf'))
    plt.show()

meshplot_prob(10)
# =============================================================================
# plot -> amount with choosing year 
# =============================================================================

def plotting_amount(ch_year, period_days):
    
    df_year = StationData[(StationData.index.year == ch_year)]
    
    #know the days of choosing year 
    if calendar.isleap(ch_year) == False:
        days = 365
    else:
        days = 366
    
    # fulfill the dataframe (insufficient date)
    def fullyear(df):
        df.index = pd.to_datetime(df.index)
        y = df.index.year
        idx = pd.date_range('{}-01-01'.format(y.min()), '{}-12-31'.format(y.max()))
        return df.reindex(idx)
    
    #check if the choosing year is incomplete
    if len(df_year) < days:
        df_year = fullyear(df_year)
    else:
        pass 
             
    
    freq_string = '%d' %(period_days) + 'D'
    df_frequency = df_year.groupby(pd.TimeGrouper(freq = freq_string)).mean()
    
    plt.figure (3) 
    plt.pcolor(df_frequency.values.transpose())
    
    plt.colorbar()
    plt.grid(linestyle='--', color='k', linewidth=0.05, alpha=0.35)
    
    plt.xlabel('Period (%s' %period_days +'Days)', fontsize=14)
    plt.xticks(fontsize=14, rotation=0)
    #plt.xticks(np.arange(0,len(df_frequency.index)))
    
    plt.ylabel('Hour', fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.ylim(ymin = 0)
    
    plt.title('Hourly rainfall amount -year %s'%ch_year, fontsize=16)
    #plt.savefig(os.path.join(main_dir, 'dem_deu.pdf'))
    plt.show()
    
 
plotting_amount(2015,10)

# =============================================================================
# plot  -> probability with choosing year  
# =============================================================================

def plotting_probability(ch_year, period_days):
    
    df_year_pdf = StationData[(StationData.index.year == ch_year)]
    
    #know the days of choosing year 
    if calendar.isleap(ch_year) == False:
        days = 365
    else:
        days = 366
    
    # fulfill the dataframe (insufficient date)
    def fullyear(df):
        df.index = pd.to_datetime(df.index)
        y = df.index.year
        idx = pd.date_range('{}-01-01'.format(y.min()), '{}-12-31'.format(y.max()))
        return df.reindex(idx)
    
    #check if the choosing year is incomplete
    if len(df_year_pdf) < days:
        df_year_pdf = fullyear(df_year_pdf)
    else:
        pass 
             
    
    freq_string = '%d' %(period_days) + 'D'
    df_frequency = df_year_pdf.groupby(pd.TimeGrouper(freq = freq_string)).mean()
    
    
    count = 0
    for index, row in df_frequency.iterrows():
        df_frequency.iloc[ count ] = row/row.sum()
        count = count + 1
    
    plt.figure (4)
    plt.pcolor(df_frequency.values.transpose())
    
    plt.colorbar()
    plt.grid(linestyle='--', color='k', linewidth=0.05, alpha=0.35)
    
    plt.xlabel('Period (%s' %period_days +'Days)', fontsize=14)
    plt.xticks(fontsize=14, rotation=0)
    #plt.xticks(np.arange(0,len(df_frequency.index)))
    
    plt.ylabel('Hour', fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.ylim(ymin = 0)
    
    plt.title('Hourly rainfall probability -year %s'%ch_year, fontsize=16)
    #plt.savefig(os.path.join(main_dir, 'dem_deu.pdf'))
    plt.show()


plotting_probability(2015,10)

STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP-START)))


