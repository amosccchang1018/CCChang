# -*- coding: utf-8 -*-
"""
simualte and visualize the rainfall data from Central Weather Bureau with Python 

Created on Wed Mar 14 00:18:44 2018
@author: Chi
"""

import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import pandas as pd
import tarfile
import os
import timeit
import datetime
from datetime import timedelta

import time
import glob
import pickle
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
import calendar

main_dir = os.path.join(r'C:\Users\Chi\Desktop\Dokument\Mater_Stuttgart\python\Taiwan_19stations')
os.chdir(main_dir)

print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer()  # to get the runtime of the program


# =============================================================================
#  function of plotting -> amount
# =============================================================================

def plotting_amount(station, period_days):
    
    StationPath = os.path.join(main_dir, r'%d.mat' %(station));
    assert StationPath
    
    matlab_data = sio.loadmat(StationPath)
    matlab_data = matlab_data['st']

    ymdh = matlab_data[:,0].astype(int)
    y, mdh = divmod(ymdh, 1000000)
    m, dh = divmod(mdh, 10000)
    d, h = divmod(dh, 100)

    timestep = [datetime.datetime(int(y[i]), int(m[i]), int(d[i]), int(h[i]-1)) \
                for i in range(0, len(matlab_data[:,0]))]
    
    matlab_row = pd.DataFrame(matlab_data[:,1],index = timestep, columns=['value'] ) / 10

    matlab_StationData = pd.DataFrame()
    matlab_StationData = matlab_row.reindex(pd.date_range(matlab_row.index.floor('D').min(), 
                                            matlab_row.index.ceil('D').max(), freq='H'))[:-1]

    matlab_StationData = matlab_StationData.set_index([matlab_StationData.index.floor('D')
                                                      , matlab_StationData.index.hour])['value']\
                                                      .unstack()

    averY = matlab_StationData.groupby(matlab_StationData.index.strftime("%j")).mean() 


    # fulfill the dataframe (insufficient date)
    def fullyear(df):
        df.index = pd.to_datetime(df.index)
        y = df.index.year
        idx = pd.date_range('{}-01-01'.format(y.min()), '{}-12-31'.format(y.max()))
        return df.reindex(idx)

    df_frequency = averY.groupby(np.arange(len(averY.index)) //period_days).mean()

    plt.pcolor(df_frequency.values.transpose())

    plt.colorbar()

    plt.grid(linestyle='--', color='k', linewidth=0.05, alpha=0.35)
    #plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('Period (%s' %period_days +'Days)', fontsize=14)
    plt.xticks(fontsize=14, rotation=0)

    plt.ylabel('Hour', fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.ylim(ymin = 0)

    plt.title('Hourly rainfall amount', fontsize=16)
    #plt.savefig(os.path.join(main_dir, 'dem_deu.pdf'))
    plt.show()    

plt.figure(1) 
plotting_amount(466930,30)

# =============================================================================
#  function of plotting -> probability
# =============================================================================
def plotting_probability(station, period_days):
    
    StationPath = os.path.join(main_dir, r'%d.mat' %(station));
    assert StationPath
    
    matlab_data = sio.loadmat(StationPath)
    matlab_data = matlab_data['st']

    ymdh = matlab_data[:,0].astype(int)
    y, mdh = divmod(ymdh, 1000000)
    m, dh = divmod(mdh, 10000)
    d, h = divmod(dh, 100)

    timestep = [datetime.datetime(int(y[i]), int(m[i]), int(d[i]), int(h[i]-1)) \
                for i in range(0, len(matlab_data[:,0]))]
    
    matlab_row = pd.DataFrame(matlab_data[:,1],index = timestep, columns=['value'] )

    matlab_StationData = pd.DataFrame()
    matlab_StationData = matlab_row.reindex(pd.date_range(matlab_row.index.floor('D').min(), 
                                            matlab_row.index.ceil('D').max(), freq='H'))[:-1]

    matlab_StationData = matlab_StationData.set_index([matlab_StationData.index.floor('D')
                                                      , matlab_StationData.index.hour])['value']\
                                                      .unstack()

    averY = matlab_StationData.groupby(matlab_StationData.index.strftime("%j")).mean() 


    # fulfill the dataframe (insufficient date)
    def fullyear(df):
        df.index = pd.to_datetime(df.index)
        y = df.index.year
        idx = pd.date_range('{}-01-01'.format(y.min()), '{}-12-31'.format(y.max()))
        return df.reindex(idx)

    df_frequency = averY.groupby(np.arange(len(averY.index)) //period_days).mean()
    
    count = 0
    for index, row in df_frequency.iterrows():
        df_frequency.iloc[ count ] = row/row.sum()
        count = count + 1

    plt.pcolor(df_frequency.values.transpose())

    plt.colorbar()

    plt.grid(linestyle='--', color='k', linewidth=0.05, alpha=0.35)
    #plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('Period (%s' %period_days +'Days)', fontsize=14)
    plt.xticks(fontsize=14, rotation=0)

    plt.ylabel('Hour', fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.ylim(ymin = 0)

    plt.title('Hourly rainfall amount', fontsize=16)
    #plt.savefig(os.path.join(main_dir, 'dem_deu.pdf'))
    plt.show()    
 
plt.figure(2)
plotting_probability(466930,30)

STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP-START)))
