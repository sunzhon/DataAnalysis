#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import gridspec
import os
import gnureadline
import pdb 
plt.rc('font',family='Arial')
import pandas as pd
import re
from brokenaxes import brokenaxes
import time as localtimepkg
import termcolor
import seaborn as sns

#def loadData(fileName,columnsName,folderName="/media/suntao/DATA/Research/P1_workspace/Experiments/Experiment_data/fig12/1231060335"):
#def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/0127113800"):
def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/0127113800"):
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
    #filter 1 low pass filter
    filterData=gamma*data+(1.0-gamma)*np.append(data[-1],data[0:-1])
    return filterData
    #filter 2
    filterData=[]
    for idx, value in enumerate(data):
    filterData.append(sum(data[0:idx])/(idx+1))
    return np.array(filterData)

    '''
    #filter 3 moveing average
    filterData=[]
    setpoint=40
    for idx, value in enumerate(data):
        if(idx<setpoint):
            count=idx+1
            filterData.append(sum(data[0:idx])/(count))
        else:
            count=setpoint
            filterData.append(sum(data[idx-count:idx])/(count))

    return np.array(filterData)
    
def neural_preprocessing(data):
    '''
    This is use a recurrent neural network to preprocessing the data,
    it works like a filter
    '''
    new_data=[]
    w_i=20
    w_r=7.2
    bias=-6.0
    new_d_old =0.0
    for d in data:
        new_a=w_i*d+w_r*new_d_old + bias
        new_d=1.0/(1.0+math.exp(-new_a))
        new_data.append(new_d)
        new_d_old=new_d

    return np.array(new_data)

def read_data(freq,start_point,end_point,folder_name):
    '''
    read data from file cut a range data

    '''
    #1) Load data
    fileName_CPGs="controlfile_CPGs"
    fileName_commands='controlfile_commands'
    fileName_modules='controlfile_modules'
    fileName_parameters='parameterfile_parameters'
    fileName_joints='sensorfile_joints'

    columnsName_CPGs=['RFO1','RFO2','RHO1','RHO2','LFO1','LFO2','LHO1','LKO2']
    columnsName_GRFs=['RF','RH','LF','LH']
    columnsName_POSEs=['roll','picth','yaw', 'x','y','z','vx','vy','vz']
    columnsName_jointPositions=['p1','p2','p3','p4','p5','p6', 'p7','p8','p9','p10','p11','p12']
    columnsName_jointVelocities=['v1','v2','v3','v4','v5','v6', 'v7','v8','v9','v10','v11','v12']
    columnsName_jointCurrents=['c1','c2','c3','c4','c5','c6', 'c7','c8','c9','c10','c11','c12']
    columnsName_jointVoltages=['vol1','vol2','vol3','vol4','vol5','vol6', 'vol7','vol8','vol9','vol10','vol11','vol12']
    columnsName_modules=['ss']
    columnsName_parameters=['MI']
    columnsName_commands=['c1','c2','c3','c4','c5','c6', 'c7','c8','c9','c10','c11','c12']


    columnsName_joints = columnsName_jointPositions + columnsName_jointVelocities + columnsName_jointCurrents + columnsName_jointVoltages + columnsName_POSEs + columnsName_GRFs

    cpg_data=loadData(fileName_CPGs,columnsName_CPGs,folder_name)    
    cpg_data=cpg_data.values

    command_data=loadData(fileName_commands,columnsName_commands,folder_name)    
    command_data=command_data.values

    module_data=loadData(fileName_modules,columnsName_modules,folder_name)    
    module_data=module_data.values

    parameter_data=loadData(fileName_parameters,columnsName_parameters,folder_name)    
    parameter_data=parameter_data.values

    jointsensory_data=loadData(fileName_joints,columnsName_joints,folder_name)    
    grf_data=jointsensory_data[columnsName_GRFs].values
    pose_data=jointsensory_data[columnsName_POSEs].values
    position_data=jointsensory_data[columnsName_jointPositions].values
    velocity_data=jointsensory_data[columnsName_jointVelocities].values
    current_data=jointsensory_data[columnsName_jointCurrents].values
    voltage_data=jointsensory_data[columnsName_jointVoltages].values


    #2) postprecessing 
    read_rows=min([4000000,jointsensory_data.shape[0], cpg_data.shape[0], command_data.shape[0], parameter_data.shape[0], module_data.shape[0]])
    if end_point>read_rows:
        print(termcolor.colored("Warning:end_point out the data bound, please use a small one","yellow"))
    time = np.linspace(int(start_point/freq),int(end_point/freq),end_point-start_point)
    return cpg_data[start_point:end_point,:], command_data[start_point:end_point,:], module_data[start_point:end_point,:], parameter_data[start_point:end_point,:], grf_data[start_point:end_point,:], pose_data[start_point:end_point,:], position_data[start_point:end_point,:],velocity_data[start_point:end_point,:],current_data[start_point:end_point,:],voltage_data[start_point:end_point,:], time



    
    jmc=Te[1]
    Te=Te[0]
    axs=[]
    if ax==None:
        figsize=(7.1,6.1244)
        fig = plt.figure(figsize=figsize,constrained_layout=False)
        gs1=gridspec.GridSpec(6,1)#13
        gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.15,right=0.98)
        axs.append(fig.add_subplot(gs1[0:6,0]))
    else:
        axs.append(ax)

    time=np.linspace(0,len(Te[0])/60,len(Te[0]))
    axs[0].plot(time,Te[3][:,leg_id],'r*')
    axs[0].plot(time,Te[4][:,leg_id],'ro')
    axs[0].plot(time,Te[5][:,leg_id],'b*')
    axs[0].plot(time,Te[6][:,leg_id],'bo')
    #axs[0].plot(Te[2][:,leg_id],'k:')

    axs[0].plot(time,0.5*Te[0][:,leg_id],'b-.')
    axs[0].plot(time,0.5*Te[1][:,leg_id],'r-.')
    #axs[0].plot(time,0.5*jmc[:,3*leg_id+1],'g-.')
    axs[0].legend(['Te1','Te2', 'Te3', 'Te4','Expected GRF','Actual GRF'],ncol=3,loc='center',bbox_to_anchor=(0.5,0.8))
    axs[0].grid(which='both',axis='x',color='k',linestyle=':')
    axs[0].grid(which='both',axis='y',color='k',linestyle=':')
    axs[0].set_xlabel('Time [s]')
    ticks=np.arange(int(max(time)))
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels([str(tick) for tick in ticks])



    # 1) read data

    #1) load data from file
    data_file = data_file_dic +"slopeWalking.log"
    data_date = pd.read_csv(data_file, sep='\t',header=None, names=['file_name','inclination'], skip_blank_lines=True,dtype=str)
    
    data_date_inclination=data_date.groupby('inclination')
    labels= data_date_inclination.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    Te={}
    Te1={}
    Te2={}
    Te3={}
    Te4={}

    for name, inclination in data_date_inclination: #name is a inclination names
        Te[name] =[]
        Te1[name]=[]  #inclination is the table of the inclination name
        Te2[name]=[]
        Te3[name]=[]
        Te4[name]=[]
        for idx in inclination.index: # how many time experiments one inclination
            #folder_name= "/home/suntao/workspace/experiment_data/" + inclination['file_name'][idx]
            folder_name= data_file_dic + inclination['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            Te_temp=touch_difference(cpg_data[:,[0,2,4,6]], grf_data)
            Te1[name].append(sum(Te_temp[3])/freq)
            Te2[name].append(sum(Te_temp[4])/freq)
            Te3[name].append(sum(Te_temp[5])/freq)
            Te4[name].append(sum(Te_temp[6])/freq)
            Te[name].append([Te_temp,command_data])

    return Te, Te1, Te2,Te3, Te4, labels


    # 1) get tocuh data
    Te, Te1, Te2,Te3,Te4, labels = slopeWalking_getTouchData(data_file_dic,start_point,end_point,freq)

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

    figsize=(7.1,6.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.15,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))



    ind= np.arange(len(labels))
    width=0.15

    #3.1) plot 
    Te1_mean,Te1_std=[],[]
    Te2_mean,Te2_std=[],[]
    Te3_mean,Te3_std=[],[]
    Te4_mean,Te4_std=[],[]
    for i in labels: # various inclinations
        Te1_mean.append(np.mean(Te1[i]))
        Te1_std.append(np.std(Te1[i]))
        Te2_mean.append(np.mean(Te2[i]))
        Te2_std.append(np.std(Te2[i]))
        Te3_mean.append(np.mean(Te3[i]))
        Te3_std.append(np.std(Te3[i]))
        Te4_mean.append(np.mean(Te4[i]))
        Te4_std.append(np.std(Te4[i]))


    idx=0
    axs[idx].bar(ind-1.5*width,Te1_mean,width,yerr=Te1_std,label=r'Te1')
    axs[idx].bar(ind-0.5*width,Te2_mean,width,yerr=Te2_std,label=r'Te2')
    axs[idx].bar(ind+0.5*width,Te3_mean,width,yerr=Te3_std,label=r'Te3')
    axs[idx].bar(ind+1.5*width,Te4_mean,width,yerr=Te4_std,label=r'Te4')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    axs[idx].set_yticks([-0.5,0,1,2])
    axs[idx].set_xticklabels([ str(round(180*float(l)/3.1415))+ u'\u00b0' for l in labels])
    axs[idx].legend()
    axs[idx].set_xlabel(r'Inclination of the floor: $\alpha$')


    axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    
    figPath=data_file_dic+ '../../' + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithCOGReflexes_TouchDiff.svg'
    #plt.savefig(figPath)
    plt.show()
    


    Te, Te1, Te2,Te3,Te4, labels = slopeWalking_getTouchData(data_file_dic,start_point,end_point,freq)

    figsize=(9.1,11.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(8,len(inclinations))#13
    gs1.update(hspace=0.24,top=0.95,bottom=0.07,left=0.1,right=0.98)
    axs=[]
    for idx in range(len(inclinations)):# how many columns, depends on the inclinations
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))
        axs.append(fig.add_subplot(gs1[4:6,idx]))
        axs.append(fig.add_subplot(gs1[6:8,idx]))

    for idx,inclination in enumerate(inclinations):
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=0,ax=axs[4*idx])
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=1,ax=axs[4*idx+1])
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=2,ax=axs[4*idx+2])
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=3,ax=axs[4*idx+3])
    plt.show()


def plot_pose(data_file_dic,start_point=90,end_point=1200,freq=60.0,inclinations=['0.0']):
    '''
    Experiment data analysis for pose

    '''
    # 1) read data
    #1) load data from file
    data_file = data_file_dic +"slopeWalking.log"
    data_date = pd.read_csv(data_file, sep='\t',header=None, names=['file_name','inclination'], skip_blank_lines=True,dtype=str)
    
    data_date_inclination=data_date.groupby('inclination')
    labels= data_date_inclination.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    pose={}

    for name, inclination in data_date_inclination: #name is a inclination names
        pose[name]=[]  #inclination is the table of the inclination name
        for idx in inclination.index:
            folder_name= data_file_dic + inclination['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            pose[name].append(pose_data)

    
    # plot
    plt.plot(lowPassFilter(pose['-0.32'][0][:,1],0.02),'r')
    plt.plot(pose['-0.32'][0][:,1],'g')
    #plt.plot(pose['0.32'][0][:,1],'b')
    plt.show()
    figPath=data_file_dic + '../../' + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithVestibularandCOGReflex_singleCurve.svg'
    #plt.savefig(figPath)
    plt.show()

if __name__=="__main__":
    data_file_dic= "~/workspace/experiment_data/"
    plot_pose(data_file_dic,start_point=100,end_point=4700,freq=60.0,inclinations=['0.0'])
    
