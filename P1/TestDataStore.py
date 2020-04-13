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

#def loadData(fileName,columnsName,folderName="/media/suntao/DATA/Research/P1_workspace/Experiments/Experiment_data/fig12/1226042908"):
def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/1230175458"):
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


if __name__=="__main__":
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

    cpg_data=loadData(fileName_CPGs,columnsName_CPGs)    
    cpg_data=cpg_data.values

    command_data=loadData(fileName_commands,columnsName_commands)    
    command_data=command_data.values

    module_data=loadData(fileName_modules,columnsName_modules)    
    module_data=module_data.values

    parameter_data=loadData(fileName_parameters,columnsName_parameters)    
    parameter_data=parameter_data.values

    jointsensory_data=loadData(fileName_joints,columnsName_joints)    
    grf_data=jointsensory_data[columnsName_GRFs].values
    pose_data=jointsensory_data[columnsName_POSEs].values
    current_data=jointsensory_data[columnsName_jointCurrents].values
    voltage_data=jointsensory_data[columnsName_jointVoltages].values

    #2) postprecessing 
    freq=60.0 # 60Hz,
    read_rows=min([4000000,jointsensory_data.shape[0], cpg_data.shape[0], command_data.shape[0], parameter_data.shape[0], module_data.shape[0]])
    start_point=1
    end_point=read_rows
    time = np.linspace(int(start_point/freq),int(end_point/freq),end_point-start_point)

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
    figsize=(10.5118,10.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(10,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))
    axs.append(fig.add_subplot(gs1[2:3,0]))
    axs.append(fig.add_subplot(gs1[3:4,0]))
    axs.append(fig.add_subplot(gs1[4:5,0]))
    axs.append(fig.add_subplot(gs1[5:6,0]))
    axs.append(fig.add_subplot(gs1[6:8,0]))
    axs.append(fig.add_subplot(gs1[8:10,0]))

    #3.1) plot 

    idx=0
    axs[idx].plot(time,-1.0*lowPassFilter(pose_data[start_point:end_point,1],0.1),'b')
    axs[idx].plot(time,-1.0*pose_data[start_point:end_point,1],'r')
    #axs[idx].plot(time,joint_data[run_id].iloc[start_point:end_point,4],'b')
    axs[idx].legend([r'Pitch'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.2,0.5],'tight')
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_yticks([-.2,0.0,.2,0.5])
    #axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    axs[idx].set_ylabel('Pitch [rad]',font_label)
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1],10,dtype='int')])
    #axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')
    #axs[idx].set_title("CPG outputs of the right front leg",font2)



    idx=1
    # plot the gait diagram
    #bax = brokenaxes(xlims=((76, 116), (146, 160)), hspace=.05, despine=False)
    #axs[idx]=bax


    #axs[idx].plot(time,current_data.iloc[start_point:end_point,26]/1000.0,'r')
    axs[idx].plot(time,lowPassFilter(current_data[start_point:end_point,2],0.1)/1000,'b')
    axs[idx].legend([r'RF'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.35,0.45],'tight')
    axs[idx].set_xticklabels(labels=[])
    #axs[idx].set_yticks([-1.0,0.0,1.0])
    #axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1],10,dtype='int')])
    #axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')
    #axs[idx].set_title("Adaptive control input term of the right front leg")

    idx=idx+1
    axs[idx].plot(time,lowPassFilter(current_data[start_point:end_point,5],0.1)/1000,'b')
    axs[idx].legend([r'RH'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.35,0.45],'tight')
    axs[idx].set_xticklabels(labels=[])
    #axs[idx].set_yticks([-1.0,0.0,1.0])
    #axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
    #axs[idx].set_title("CPG outputs of the right front leg",font2)
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1],10,dtype='int')])
    #axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')

    idx=idx+1
    axs[idx].plot(time,-1.0*lowPassFilter(current_data[start_point:end_point,8],0.1)/1000,'b')
    axs[idx].legend([r'LF'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.35,0.45],'tight')
    axs[idx].set_xticklabels(labels=[])
    #axs[idx].set_yticks([-1.0,0.0,1.0])
    #axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
    #axs[idx].set_title("CPG outputs of the right front leg",font2)
    #axs[idx].set_xlabel('Time [s]',font_label)
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1],10,dtype='int')])
    #axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')
    #axs[idx].set_title("Adaptive control input term of the right front leg")

    idx=idx+1
    axs[idx].plot(time,-1.0*lowPassFilter(current_data[start_point:end_point,11],0.1)/1000,'b')
    axs[idx].legend([r'LH'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.35,0.45],'tight')
    #axs[idx].set_yticks([-1.0,0.0,1.0])
    #axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
    #axs[idx].set_title("CPG outputs of the right front leg",font2)
    axs[idx].set_xlabel('Time [s]',font_label)
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1],10,dtype='int')])
    axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')

#000000000000000000000000000000000000000-------------------------------

    idx=idx+1
    axs[idx].plot(time, cpg_data[start_point:end_point,0],'r')
    axs[idx].plot(time, cpg_data[start_point:end_point,1],'b')
    axs[idx].legend([r'LH'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-1.2,1.2],'tight')
    #axs[idx].set_yticks([-1.0,0.0,1.0])
    #axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
    #axs[idx].set_title("CPG outputs of the right front leg",font2)
    axs[idx].set_xlabel('Time [s]',font_label)
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1],10,dtype='int')])
    axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')


    idx=idx+1
    axs[idx].plot(time, cpg_data[start_point:end_point,2],'r')
    axs[idx].plot(time, parameter_data[start_point:end_point,0],'r')
    axs[idx].plot(time, command_data[start_point:end_point,1],'k')
    axs[idx].plot(time, current_data[start_point:end_point,1]/1000,'b')
    axs[idx].plot(time, voltage_data[start_point:end_point,1]/10.,'g:')
    axs[idx].legend([r'LH'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-1.2,1.5],'tight')
    #axs[idx].set_yticks([-1.0,0.0,1.0])
    #axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
    #axs[idx].set_title("CPG outputs of the right front leg",font2)
    axs[idx].set_xlabel('Time [s]',font_label)
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1],10,dtype='int')])
    axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')

    plt.savefig('/media/suntao/DATA/Research/P1_workspace/Figures/Fig12/Fig12_source2.svg')
    plt.show()

