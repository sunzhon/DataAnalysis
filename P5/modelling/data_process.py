import pandas as pd
import pdb
import h5py

import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import gridspec
import os
import pdb
import termcolor
import gnureadline
plt.rc('font',family='Arial')
import pandas as pd
import re
import time as localtimepkg
from brokenaxes import brokenaxes
from mpl_toolkits import mplot3d
from matplotlib import animation
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import warnings
from matplotlib.animation import FuncAnimation


def loadData(objectName,folderName="/media/suntao/DATA/Research/Human_walking_date/processed_data"):
    '''  
    load data from a file
    fileName: the name of file that you want to read
    columnsName: it the column name of the file
    Note: the args of sys is file_id and date of the file
    ''' 
    #1) load data from file
    data_file = folderName +"/"+ objectName + "/combined/baseline.csv"
    resource_data = pd.read_csv(data_file, index_col=0,header=0,skip_blank_lines=True,dtype=str)
    read_rows=resource_data.shape[0]-1
    fine_data = resource_data.iloc[0:read_rows,:].astype(float)# 数据行对齐
    return fine_data


def saveH5Data():
    data_file="/media/suntao/DATA/Research/Human_walking_date/processed_data/subject_info.csv"
    subject_info = pd.read_csv(data_file, index_col=None,header=1,skip_blank_lines=False,dtype=str)
    subjects=subject_info['subject id'][0:17].values
    
    fileName="/media/suntao/DATA/Research/Human_walking_date/processed_data/suntao_all_17_subjects.h5"
    with h5py.File(fileName, "w") as h5_data_set:
        #rawData_group=h5_data_set.create_group("rawData_group")
        for idx, subjectName in enumerate(subjects):
            subjectID="subject_"+str(idx+1)
            print(subjectName, subjectID)
            #pdb.set_trace()
            dataset=loadData(subjectName)
            columns=dataset.columns.values
            #print(columns)
            #print(dataset.shape)
            #print(dataset["plate_1_force_x"])
            #plt.plot(dataset['plate_1_force_x'][200:],'r')
            #lt.plot(dataset['plate_1_force_y'][200:],"g")
            #plt.plot(dataset['plate_1_force_z'][1000:1400],"b")
            
            h5_data_set.create_dataset(name=subjectID,data=dataset.values)
            print(h5_data_set[subjectID].shape)
            h5_data_set[subjectID].attrs.create("dataItem",columns)
            h5_data_set[subjectID].attrs.create("subjectName",subjectName)
            #pdb.set_trace()
            #data_set['subject_01']
        print(h5_data_set.keys())
            

def loadH5Data():
    input_variables=["plate_1_force_x", "plate_1_force_y","plate_1_force_z", "plate_2_force_x", "plate_2_force_y", "plate_2_force_z"]
    output_variables=["RFT_X","RFT_Y","RFT_Z","LFT_X","LFT_Y","LFT_Z"]

    fileName="/media/suntao/DATA/Research/Human_walking_date/processed_data/suntao_all_17_subjects.h5"
    with h5py.File(fileName, "r") as h5_data_set:
        for idx in range(1,len(h5_data_set.keys()):
            subject_id="subject_"+str(idx)
            dataItem=h5_data_set[subject_id].attrs.get('dataItem')
            input_variable_idx=[]
            output_variable_idx=[]
            for input_variable in input_variables:
                input_variable_idx.append(np.where(dataItem==input_variable)[0][0])
            for output_variable in input_variables:
                output_variable_idx.append(np.where(dataItem==output_variable)[0][0])
            # retrieve input and output data for training NN
            training_data=h5_data_set[subject_id][:,input_variable_idx]
            training_labels= h5_data_set[subject_id][:,output_variable_idx]


        print(h5_data_set['subject_1'])
        print(h5_data_set['subject_1'].shape)
        print(h5_data_set['subject_1'].size)
        print(h5_data_set['subject_1'].dtype)
        print(h5_data_set['subject_1'].ndim)
        print(h5_data_set['subject_1'].attrs)
        print(h5_data_set['subject_1'].attrs.keys())
        print(h5_data_set['subject_1'].attrs.values())
        print(h5_data_set['subject_1'].attrs.items())
        
        print(h5_data_set.attrs.keys())
        
        print(h5_data_set['subject_1'][0,0])
        
        print(len(h5_data_set.attrs.items()))
        if len(h5_data_set.attrs.items()):
            print("{} contains: ".format(fileName))
            print("Root attributes:")
    
        for key, value in h5_data_set.attrs.items():
            print(" {}: {}".format(key, value)) # 输出储存在File类中的attrs信息，一般是各层的名称
        print("len",len(value))
        for layer, g in data_set.items(): # 读取各层的名称以及包含层信息的Group类
            print(" {}".format(layer))
            print("  Attributes:")
            for key, value in g.attrs.items(): # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
                print("   {}: {}".format(key, value)) 
                print("sss:{}".format(len(value)))
            print("  Dataset:")
            
    
if __name__=="__main__":
    #saveH5Data()
    loadH5Data()
    
    

