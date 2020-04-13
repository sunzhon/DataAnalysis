#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import os
import pdb 
plt.rc('font',family='Arial')
import pandas as pd
import re
from brokenaxes import brokenaxes

def stsubplot(fig,position,number,gs):
    axprops = dict(xticks=[], yticks=[])
    width_p=position.x1-position.x0; height_p=(position.y1-position.y0)/number
    left_p=position.x0;bottom_p=position.y1-height_p;
    ax=[]
    for idx in range(number):
        #ax.append(fig.add_axes([left_p,bottom_p-idx*height_p,width_p,height_p], **axprops))
        ax.append(brokenaxes(xlims=((76, 116), (146, 160)), hspace=.05, despine=True,fig=fig,subplot_spec=gs))
        pdb.set_trace()
        #ax[-1].set_xticks([])
        #ax[-1].set_xticklabels(labels=[])
    return ax

def loadData(fileName,columnsName,folderName="/media/suntao/DATA/Research/P1_workspace/Experiments/Experiment_data/SR/0101022943"):
#def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/1230175458"):
    '''
    load data from a file
    fileName: the name of file that you want to read
    columnsName: it the column name of the file
    Note: the args of sys is file_id and date of the file
    '''
        
    #1) load data from file
    data_file = folderName +"/"+ fileName + ".csv"
    resource_data = pd.read_csv(data_file, sep='\t', index_col=0,header=None, names=columnsName, skip_blank_lines=True,dtype=str)

    read_rows=resource_data.shape[0]-1
    fine_data = resource_data.iloc[0:read_rows,:].astype(float)# 数据行对齐
    return fine_data



def lowPassFilter(data,gamma):
    '''
    #filter 1
    filterData=gamma*data+(1.0-gamma)*np.append(data[-1],data[0:-1])
    return filterData
    '''
    '''
    #filter 2
    filterData=[]
    for idx, value in enumerate(data):
    filterData.append(sum(data[0:idx])/(idx+1))
    return np.array(filterData)

    '''
    #filter 3
    filterData=[]
    setpoint=20
    for idx, value in enumerate(data):
        if(idx<setpoint):
            count=idx+1
            filterData.append(sum(data[0:idx])/(count))
        else:
            count=setpoint
            filterData.append(sum(data[idx-count:idx])/(count))

    return np.array(filterData)

def EnergyCost(I,U,Fre):
    if ((type(U) is np.ndarray) and (type(I) is np.ndarray)):
        E=sum(U*np.absolute(I/1000.0)*1.0/Fre)
    else:
        print("input data type is wrong, please use numpy array")
    return E


def SpecificResistance(current,voltage,Fre):
    M=2.5
    G=9.8
    D=1.0
    E=[]
    for idx in range(current.shape[1]):
        E.append(EnergyCost(current[:,idx],voltage[:,idx],Fre))

    sumE=sum(E)
    epsilon=sumE/M/G/D
    return epsilon

def StatisticSR_F(current_data, voltage_data, freq):
    epsilons=[]
    start=65*freq; end=int(round((70+5/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=75*freq; end=int(round((80-6/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=84*freq; end=int(round((89+17/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=95*freq; end=int(round((100+1/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=105*freq; end=int(round((110-1/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    return epsilons

def StatisticSR_B(current_data, voltage_data, freq):
    epsilons=[]
    start=49*freq; end=int(round((54+51/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=60*freq; end=int(round((65+10/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=70*freq; end=int(round((75+57/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=80*freq; end=int(round((85+18/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=89*freq; end=int(round((93+6/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    return epsilons

def StatisticSR_O(current_data, voltage_data, freq):
    epsilons=[]
    start=72*freq; end=int(round((79+9/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=84*freq; end=int(round((90+10/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=94*freq; end=int(round((100+10/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=105*freq; end=int(round((111+21/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=115*freq; end=int(round((121+9/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    return epsilons



def StatisticSR_I(current_data, voltage_data, freq):
    epsilons=[]
    start=56*freq; end=int(round((62+49/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=68*freq; end=int(round((73+50/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=80*freq; end=int(round((85+52/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=90*freq; end=int(round((94+56/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    start=101*freq; end=int(round((105+54/60.0)*freq))
    epsilons.append(SpecificResistance(current_data[start:end,:],voltage_data[start:end,:],freq))
    return epsilons



def readData(folderName,freq):
    #1) Load data
    fileName_CPGs="controlfile_CPGs"
    fileName_commands='controlfile_commands'
    fileName_modules='controlfile_modules'
    fileName_parameters='parameterfile_parameters'
    fileName_joints='sensorfile_joints'

    columnsName_CPGs=['RFO1','RFO2','RHO1','RHO2','LFO1','LFO2','LHO1','LKO2']
    columnsName_GRFs=['RF','RH','LF','LH']
    columnsName_POSEs=['roll','picth','yaw', 'x','y','z']
    columnsName_jointPositions=['p1','p2','p3','p4','p5','p6', 'p7','p8','p9','p10','p11','p12']
    columnsName_jointVelocities=['v1','v2','v3','v4','v5','v6', 'v7','v8','v9','v10','v11','v12']
    columnsName_jointCurrents=['c1','c2','c3','c4','c5','c6', 'c7','c8','c9','c10','c11','c12']
    columnsName_jointVoltages=['vol1','vol2','vol3','vol4','vol5','vol6', 'vol7','vol8','vol9','vol10','vol11','vol12']
    columnsName_modules=['ss']
    columnsName_parameters=['MI']
    columnsName_commands=['c1','c2','c3','c4','c5','c6', 'c7','c8','c9','c10','c11','c12']


    columnsName_joints = columnsName_jointPositions + columnsName_jointVelocities + columnsName_jointCurrents + columnsName_jointVoltages + columnsName_POSEs + columnsName_GRFs

    cpg_data=loadData(fileName_CPGs,columnsName_CPGs,folderName)    
    cpg_data=cpg_data.values

    command_data=loadData(fileName_commands,columnsName_commands,folderName)    
    command_data=command_data.values

    module_data=loadData(fileName_modules,columnsName_modules,folderName)    
    module_data=module_data.values

    parameter_data=loadData(fileName_parameters,columnsName_parameters,folderName)    
    parameter_data=parameter_data.values

    jointsensory_data=loadData(fileName_joints,columnsName_joints,folderName)    
    grf_data=jointsensory_data[columnsName_GRFs].values
    pose_data=jointsensory_data[columnsName_POSEs].values
    current_data=jointsensory_data[columnsName_jointCurrents].values
    voltage_data=jointsensory_data[columnsName_jointVoltages].values

    #2) Time
    read_rows=min([jointsensory_data.shape[0], cpg_data.shape[0], command_data.shape[0], parameter_data.shape[0], module_data.shape[0]])
    start_point=1
    end_point=read_rows
    time = np.linspace(int(1.0*start_point/freq),int(1.0*end_point/freq),end_point-start_point)
    
    data={}
    data['time']=time
    data['cpg_data']=cpg_data
    data['command_data']=command_data
    data['pose_data'] =pose_data
    data['current_data']=current_data
    data['voltage_data']=voltage_data
    data['grf_data']=grf_data
    data['parameter_data']=parameter_data
    data['module_data']=module_data

    return data


if __name__=="__main__":
    all_SR=[]
    #1) read data
    folderName="/media/suntao/DATA/Research/P1_workspace/Experiments/Experiment_data/SR/0101022943"
    data=readData(folderName,freq=60)
    time=data['time']
    cpg_data=data['cpg_data']
    command_data=data['command_data']
    pose_data=data['pose_data'] 
    current_data=data['current_data']
    voltage_data=data['voltage_data']
    grf_datat=data['grf_data']
    parameter_data=data['parameter_data']
    module_data=data['module_data']
    all_SR.append(StatisticSR_F(current_data,voltage_data,freq=60))

    folderName="/media/suntao/DATA/Research/P1_workspace/Experiments/Experiment_data/SR/0101023548"
    data=readData(folderName,freq=60)
    time=data['time']
    cpg_data=data['cpg_data']
    command_data=data['command_data']
    pose_data=data['pose_data'] 
    current_data=data['current_data']
    voltage_data=data['voltage_data']
    grf_datat=data['grf_data']
    parameter_data=data['parameter_data']
    module_data=data['module_data']
    all_SR.append(StatisticSR_B(current_data,voltage_data,freq=60))

    folderName="/media/suntao/DATA/Research/P1_workspace/Experiments/Experiment_data/SR/0101025431"
    data=readData(folderName,freq=60)
    time=data['time']
    cpg_data=data['cpg_data']
    command_data=data['command_data']
    pose_data=data['pose_data'] 
    current_data=data['current_data']
    voltage_data=data['voltage_data']
    grf_datat=data['grf_data']
    parameter_data=data['parameter_data']
    module_data=data['module_data']
    all_SR.append(StatisticSR_O(current_data,voltage_data,freq=60))


    folderName="/media/suntao/DATA/Research/P1_workspace/Experiments/Experiment_data/SR/0101024834"
    data=readData(folderName,freq=60)
    time=data['time']
    cpg_data=data['cpg_data']
    command_data=data['command_data']
    pose_data=data['pose_data'] 
    current_data=data['current_data']
    voltage_data=data['voltage_data']
    grf_datat=data['grf_data']
    parameter_data=data['parameter_data']
    module_data=data['module_data']
    all_SR.append(StatisticSR_I(current_data,voltage_data,freq=60))
    print(all_SR)
    #2) process
    read_rows=min([pose_data.shape[0], cpg_data.shape[0], command_data.shape[0], parameter_data.shape[0], module_data.shape[0]])
    start_point=1
    end_point=read_rows
    #3) plot
    font_legend = {'family' : 'Arial',
                   'weight' : 'light',
                   'size'   : 10,
                   'style'  :'italic'
                  }
    font_label = {'family' : 'Arial',
                  'weight' : 'light',
                  'size'   : 12,
                  'style'  :'normal'
                 }

    font_title = {'family' : 'Arial',
                  'weight' : 'light',
                  'size'   : 12,
                  'style'  :'normal'
                 }

    #figsize=(10.5118,7.1244)
    figsize=(5.5118,5.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(1,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:1,0]))

    xtickstep=4
    #3.1) plot 
    idx=0
    N = len(all_SR)
    means=[]
    stds=[]
    for i in range(N):
        means.append(np.mean(all_SR[i]))
        stds.append(np.std(all_SR[i]))
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    
    print("menas:",means)
    print("std:",stds)
    print(all_SR)
    for i in range(4):
        print(min(all_SR[i]),max(all_SR[i]))
    axs[idx].bar(ind, means, width, yerr=stds)
    
    confi=['all knee', 'all-elbow', 'outward','inward']
    axs[idx].set_ylabel('Specific resistance')
    axs[idx].set_xlabel('Legs orientations')
    axs[idx].set_xticks(ind, confi)
    axs[idx].set_xticklabels(labels=confi)
    axs[idx].set_yticks(np.arange(0, 6, 0.5))
    #plt.legend((p1[0], p2[0]), ('Men', 'Women'))



    plt.savefig('/media/suntao/DATA/Research/P1_workspace/Figures/FigSR/FigSR_source.svg')
    plt.show()

