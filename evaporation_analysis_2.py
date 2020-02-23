# -*- coding: utf-8 -*-
"""
visualize rainfall in specific area by Geoinformation in Taiwan with Python 

Created on Thu Feb  1 22:48:27 2018
@author: Chi
"""
import numpy as np
import pandas as pd
import xarray as xr
import tarfile
import os
#os.mkdir('evap_test') #創一個空資料夾 (跟code位置一樣)
import fnmatch  #用來找有沒有後綴是某個類型檔案的
from  tempfile  import  mkdtemp 

import timeit
import time
import datetime 
from dateutil.relativedelta import relativedelta
import glob
import pickle
import matplotlib.pyplot as plt

main_dir = os.path.join(r'C:\Users\Chi\Desktop\Dokument\Mater_Stuttgart\python\germany_evaporation')
os.chdir(main_dir)

ascii_grid_path = os.path.join(main_dir, r'grids_germany_daily_evapo_p_19910101.asc')
assert ascii_grid_path

tar_path = os.path.join(main_dir, r'grids_germany_daily_evapo_p_199101.tgz')
assert tar_path

print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer()  # to get the runtime of the program

# =============================================================================
# try one file simply (try memmap)
# =============================================================================
ascii_grid = np.loadtxt(ascii_grid_path
                        , skiprows = 6);#讀 asc without first six string 

# =============================================================================
# load data with all tarfile and store to dataframe 
# =============================================================================

def replace_minus(ndarray):
    ndarray[ndarray == -9999] = 0
    return ndarray

def dele_firstsevenrow(ndarray):
    return [ndarray.pop(0) for i in range(0, 6)]
   
firstmonth = datetime.date(1991, 1, 1)

#做出dataframe 準備裝檔案
#date 排好 x,y座標排好 ，之後再新增column 放 nd array用

GridColumn = 654
GridRow = 866 
tgzCounter = len(glob.glob1(main_dir
                            ,"*.tgz"))

'''get number of file name '''
minthnumbers_data = []
for months in range(0, tgzCounter):
    minthnumbers_data.append(firstmonth + relativedelta( months =+ months ))

'''time to number'''
minthnumbers_data_num = []
for i in minthnumbers_data:
    minthnumbers_data_num.append(i.strftime("%Y%m"))

empty_ndarray = [np.full((GridRow, GridColumn)
                 ,0
                 ,dtype=float) for i in range(0,tgzCounter)]

df = pd.DataFrame( minthnumbers_data
                  ,columns=['Date'])

df['Grid Evaporation'] = empty_ndarray

#set a mmap file
n_array = 0
for dates in minthnumbers_data_num:
    
    tar_path = os.path.join(main_dir, r'grids_germany_daily_evapo_p_%s.tgz' %dates)
    assert tar_path
    
    tar = tarfile.open(tar_path ,"r")
    n_array += len(tar.getmembers())

filename = os.path.join(main_dir, 'mmap.dat')
fp = np.memmap(filename, mode='w+', shape=(n_array, GridRow, GridColumn))

#迴圈讀tarfile用
count = 0
count2 = 0
for dates in minthnumbers_data_num:
    
    tar_path = os.path.join(main_dir, r'grids_germany_daily_evapo_p_%s.tgz' %dates)
    assert tar_path
    
    tar = tarfile.open(tar_path ,"r")
    #tar.extractall('evap') #load 出來的資料放的資料夾名稱 (跟tgz位置一樣)

    mean_lines = np.empty((GridRow, GridColumn))
    for member in tar.getmembers():
        file = tar.extractfile(member)
        file_contents = file.readlines()
    
        dele_firstsevenrow(file_contents)

        lines = []
        for line in file_contents:
            lines.append(line.decode("utf-8").split())    

        lines = np.asarray(lines, dtype=float)
    
        replace_minus(lines)
    
        mean_lines = mean_lines + lines
        
        #save to mmap
        fp[count2,:,:] = lines
        count2 += 1
        print (count2)
        #break
        
    
    mean_lines = mean_lines / len(tar.getmembers())  
    
    #save file to dataframe 
    df.ix[count,'Grid Evaporation'] = mean_lines 
    print (count)
    count += 1
    #break
    
    #呼叫某個值
    #df.ix[minthnumbers_data[0],'Grid Evaporation'][439,3]

    tar.close()
#save file to directory
df.to_pickle('evaporation datas')


STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP-START)))

# =============================================================================
# load data with all tarfile and store to dataframe 
# =============================================================================

df2 = pd.read_pickle('evaporation datas')

def Evapplot(Gridx,Gridy):
    
    X = minthnumbers_data  
    Y = [df2.ix[i,'Grid Evaporation'][Gridx,Gridy] for i in range(0,len(df.index))]

    plt.plot(X, Y)
    
    plt.grid(linestyle='--', color='k', linewidth=0.05, alpha=0.35)
    #plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('Time (Month)', fontsize=14)
    plt.xticks(fontsize=14, rotation=0)

    plt.ylabel('Evaporation (mm / Month)', fontsize=14)
    plt.yticks(fontsize=14, rotation=0)

    plt.title("Evaporation - Time Series\n(Grid Point [%s , %s])"%(Gridx,Gridy), fontsize=16)
    #plt.savefig(os.path.join(main_dir, 'dem_deu.pdf'))
    plt.show()

Evapplot(439,3)

