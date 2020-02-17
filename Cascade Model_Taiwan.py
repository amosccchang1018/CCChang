# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 20:07:02 2018

@author: Chi
"""
import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import pandas as pd
import tarfile
import os, fnmatch
import timeit
import datetime
from datetime import timedelta
import datetime
import time
import glob
import pickle
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
import calendar
from pandas.tseries.frequencies import to_offset
from scipy.stats import beta
from scipy.special import gamma as gammaf
from scipy.optimize import minimize
from sklearn import linear_model
import math

main_dir = os.path.join(r'C:\Users\Chi\Desktop\Dokument\Mater_Stuttgart\python\Taiwan_19stations')
os.chdir(main_dir)

WvaluePath = os.path.join(main_dir, r'W_values');
assert WvaluePath

print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer()  # to get the runtime of the program


# =============================================================================
#  data process  #  input Taiwan data # casade
# =============================================================================

class Casade:
    threshold = 0.3
    
    def datas(self, filename_str):
        StationPath = os.path.join(main_dir, filename_str + '.mat');
        assert StationPath
        
        matlab_data = sio.loadmat(StationPath)
        matlab_data = matlab_data['st']

        ymdh = matlab_data[:,0].astype(int)
        y, mdh = divmod(ymdh, 1000000)
        m, dh = divmod(mdh, 10000)
        d, h = divmod(dh, 100)

        timestep = [datetime.datetime(int(y[i]), int(m[i]), int(d[i]), int(h[i]-1)) \
                    for i in range(0, len(matlab_data[:,0]))]

        matlab_row = pd.DataFrame(matlab_data[:,1]
                                  ,index = timestep
                                  , columns=['value'] )/ 10
        
        # the minus value
        # delete it (replace by nan)
        matlab_row[matlab_row.value < 0] = np.nan
        
        return matlab_row

    def grouppe(self, df, timestep):
        df_unter = df.copy()
        df_unter = df.resample(timestep, label='right').sum()    
        return df_unter  
    
    #check the sum
    def sum_check(self, df1, df2):
        if abs(df1.value.sum() - df2.value.sum()) <= 10**-4:
            pass
        else:
            print ('error')
    
    #fulfill the year 
    def fulfill_er(self, df_oben, df_unten, timestep_unten):
        df_fill = df_unten.copy()
        if 2*len(df_oben) - len(df_unten) == 1:
            #df_fill = df_unten.copy()
            df_fill.loc[df_fill.index[0] - pd.offsets.Minute(timestep_unten)
                   , 'value'] = df_oben.value[0] - df_fill.value[0]
            df_fill = df_fill.sort_index()
        
        elif 2*len(df_oben) - len(df_unten) == 3:
            #df_fill = df_unten.copy()
            # third
            df_fill.loc[df_fill.index[0] - pd.offsets.Minute(timestep_unten)
                   , 'value'] = df_oben.value[1] - df_fill.value[0]    
            df_fill = df_fill.sort_index()
    
            #second and one
            for i in range(0,2):
                df_fill.loc[df_fill.index[0] - pd.offsets.Minute(timestep_unten)
                       , 'value'] = df_oben.value[0] / 2     
                df_fill = df_fill.sort_index() 
        else:
            pass

        return df_fill

    
    # calculate w1 and w2 
    def para_W1(self, df_oben, df_unten):
        W1 = df_unten.iloc[np.arange(len(df_oben)) * 2].value / df_oben.value
        return W1
    #W1_1er = para_W1(S_12hr,S_6hr)
    #W1_2er = para_W1(S_6hr,S_3hr)

    def para_W2(self, df_oben, df_unten):
        W2 = df_unten.iloc[np.arange(len(df_oben))*2 + 1].value / df_oben.value
        return W2
    #W2_1er = para_W2(S_12hr,S_6hr)
    #W2_2er = para_W2(S_6hr,S_3hr)
    
    # calculate P01 
    def valueP01(self, df_oben, df_unten):
        dic = {'value_unten' : df_unten.iloc[np.arange(len(df_oben))*2 + 1].value
                , 'value_above' : df_oben.value}
        df = pd.DataFrame(dic)
        df2 = df[df.value_above >= self.threshold] 
        w = df2.value_unten / df2.value_above
        P01 = len(w[(w == 0) | (w == 1)]) /len(w)
        return P01
    
    #plot 0 < w < 1
    def W_innerhalb(self, w_layer, station_oben, station_unten):
        df = pd.concat([ w_layer
                        ,station_unten.iloc[np.arange(len(station_oben))*2 + 1]
                        ,station_oben.value]
                        , axis = 1)
        df.columns = ['percent', 'amount','threshold_check']
        w_in = df[df.threshold_check >= self.threshold]
        w_in = df[(df.percent  > 0) & (df.percent < 1)]
        return w_in

    
    def plotting(self, w_in):
        plt.plot(w_in.percent, w_in.amount,'ro')
        plt.xlabel('W value')
        plt.ylabel('Amount')
        plt.show()

Cas=Casade()
S_3hr = Cas.grouppe(Cas.datas('467530'), '3H')
S_6hr = Cas.grouppe(Cas.datas('467530'), '6H')
S_12hr = Cas.grouppe(Cas.datas('467530'), '12H')

Cas.sum_check(Cas.datas('467530'), S_3hr)
Cas.sum_check(Cas.datas('467530'), S_6hr)
Cas.sum_check(Cas.datas('467530'), S_12hr)

p01_1 = Cas.valueP01(S_12hr, S_6hr)
p01_2 = Cas.valueP01(S_6hr, S_3hr)

W2_1er = Cas.para_W2(S_12hr,S_6hr)
W2_2er = Cas.para_W2(S_6hr,S_3hr)

plt.figure(1)
Cas.plotting(Cas.W_innerhalb(W2_1er, S_12hr, S_6hr))
#fig = plt.figure(1)
#fig.savefig(os.path.join(WvaluePath, r'W_1er_%s.png'%filename_str))
#plt.clf()
    
plt.figure(2)
Cas.plotting(Cas.W_innerhalb(W2_2er, S_6hr, S_3hr))    
#fig = plt.figure(2)
#fig.savefig(os.path.join(WvaluePath, r'W_2er_%s.png'%filename_str))
#plt.clf()



# =============================================================================
#  load all stations (19 stations) and store the plot -> 0<W<1
# =============================================================================

# find name in file
for (path, dirs, files) in os.walk(main_dir):
    files_name = files
    del path, dirs, files
    break

# remove the substring of name 
files_name_num = []
for word in files_name:
    if word.endswith(".mat"):
        word = word[:-4]
        files_name_num.append(word)

# run casade model

#write a running model
#casade step -> plot 0 < w < 1
def CasadesWinPlot(name):
    Cas=Casade()
    
    S_3hr = Cas.grouppe(Cas.datas(name), '3H')
    S_6hr = Cas.grouppe(Cas.datas(name), '6H')
    S_12hr = Cas.grouppe(Cas.datas(name), '12H')
    
    Cas.sum_check(Cas.datas(name), S_3hr)
    Cas.sum_check(Cas.datas(name), S_6hr)
    Cas.sum_check(Cas.datas(name), S_12hr)
    
    S_3hr = Cas.fulfill_er(S_6hr, S_3hr, '3H')
    S_6hr = Cas.fulfill_er(S_12hr, S_6hr, '6H')
        
    W2_1er = Cas.para_W2(S_12hr,S_6hr)
    W2_2er = Cas.para_W2(S_6hr,S_3hr)
    
    plt.figure(1)
    Cas.plotting(Cas.W_innerhalb(W2_1er, S_12hr, S_6hr))
    fig = plt.figure(1)
    fig.savefig(os.path.join(WvaluePath, r'W_1er_%s.png'%name))
    plt.clf()
    
    plt.figure(2)
    Cas.plotting(Cas.W_innerhalb(W2_2er, S_6hr, S_3hr))    
    fig = plt.figure(2)
    fig.savefig(os.path.join(WvaluePath, r'W_2er_%s.png'%name))
    plt.clf()

for stationnames in files_name_num:
    CasadesWinPlot(stationnames)

# =============================================================================
#  log regression fit
# =============================================================================

# collect P01 of 19 stations 

def P01process(name):
    Cas=Casade()

    S_3hr = Cas.grouppe(Cas.datas(name), '3H')
    S_6hr = Cas.grouppe(Cas.datas(name), '6H')
    S_12hr = Cas.grouppe(Cas.datas(name), '12H')
    
    Cas.sum_check(Cas.datas(name), S_3hr)
    Cas.sum_check(Cas.datas(name), S_6hr)
    Cas.sum_check(Cas.datas(name), S_12hr)
    
    S_3hr = Cas.fulfill_er(S_6hr, S_3hr, '3H')
    S_6hr = Cas.fulfill_er(S_12hr, S_6hr, '6H')  
    
    p01_1 = Cas.valueP01(S_12hr, S_6hr)
    p01_2 = Cas.valueP01(S_6hr, S_3hr)
    
    return p01_1, p01_2

allP01_1er = []
allP01_2er = []

for stationnames in files_name_num:
    a, b= P01process(stationnames)
    allP01_1er.append(a)
    allP01_2er.append(b)
    print(P01process(stationnames))

# =============================================================================
#  beta fit and plot 
# =============================================================================

#先選一站出來試驗
Cas=Casade()
S_3hr = Cas.grouppe(Cas.datas('466990'), '3H')
S_6hr = Cas.grouppe(Cas.datas('466990'), '6H')
S_12hr = Cas.grouppe(Cas.datas('466990'), '12H')

Cas.sum_check(Cas.datas('466990'), S_3hr)
Cas.sum_check(Cas.datas('466990'), S_6hr)
Cas.sum_check(Cas.datas('466990'), S_12hr)
    
S_3hr = Cas.fulfill_er(S_6hr, S_3hr, '3H')
S_6hr = Cas.fulfill_er(S_12hr, S_6hr, '6H')

W2_1er = Cas.para_W2(S_12hr,S_6hr)
W2_2er = Cas.para_W2(S_6hr,S_3hr)

win1 = Cas.W_innerhalb(W2_1er, S_12hr, S_6hr)
#win2 = Cas.W_innerhalb(W2_2er, S_6hr, S_3hr)

def obj_logbetafun(x, sign=1.0):
    df_beta = win1.percent.copy()
    summa = - np.log(beta.pdf(df_beta, x[0], x[1])).sum() #min sum
    return summa

cons ={'type': 'eq',
        'fun' : lambda x : np.array(x[0] - x[1])}

result = minimize(obj_logbetafun
                  , [20, 20]
                  , constraints = cons
                  , method='SLSQP'
                  , options={'disp': True})

def betafun(x, a, b):
    return gammaf(a+b)/gammaf(a)/gammaf(b)*x**(a-1)*(1-x)**(b-1)

fig = plt.figure(3)
plt.scatter(win1.percent
            , betafun(win1.percent, result.x[0], result.x[1])
            , c='r', zorder=1)
plt.hist(win1.percent, bins = 50
         ,color='blue',align='mid', normed=True, zorder=0)


STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP-START)))
        