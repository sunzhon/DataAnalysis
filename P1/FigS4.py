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
        ax.append(fig.add_axes([left_p,bottom_p-idx*height_p,width_p,height_p], **axprops))
        #ax.append(brokenaxes(xlims=((76, 116), (146, 160)), hspace=.05, despine=True,fig=fig,subplot_spec=gs))
        #ax[-1].set_xticks([])
        #ax[-1].set_xticklabels(labels=[])
    return ax



#
#def stsubplot(position,number):
#    axprops = dict(xticks=[], yticks=[])
#    width_p=position.x1-position.x0; height_p=(position.y1-position.y0)/number
#    left_p=position.x0;bottom_p=position.y1-height_p;
#    ax=[]
#    for idx in range(number):
#        ax.append(fig.add_axes([left_p,bottom_p-idx*height_p,width_p,height_p], **axprops))
#    return ax

def loadData(fileName,columnsName,folderName="/media/suntao/DATA/Research/P1_workspace/Experiments/Experiment_data/figS4/0109150006"):
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

def EnergyCost(U,I,Fre):
    if ((type(U) is np.ndarray) and (type(I) is np.ndarray)):
        E=sum(U*I*1/Fre)
    else:
        print("input data type is wrong, please use numpy array")
    return E


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
    position_data=jointsensory_data[columnsName_jointPositions].values
    current_data=jointsensory_data[columnsName_jointCurrents].values
    voltage_data=jointsensory_data[columnsName_jointVoltages].values

    #2) postprecessing 
    freq=60.0 # 60Hz,
    read_rows=min([4000000,jointsensory_data.shape[0], cpg_data.shape[0], command_data.shape[0], parameter_data.shape[0], module_data.shape[0]])
    start_point= 1#2100
    end_point= read_rows
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

    figsize=(5.1,7.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.15,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))
    axs.append(fig.add_subplot(gs1[2:4,0]))
    axs.append(fig.add_subplot(gs1[4:6,0]))

    #3.1) plot 

    idx=0




    #--------------------CPG-------------------------#
    idx=0
    axs[idx].set_yticklabels(labels=[])
    axs[idx].set_yticks([])
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1]+0.1,5,dtype='int')])
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_title('CPG',fontdict=font_label)
    #axs[idx].text(text_x,1,'(A)',fontdict=font_label,rotation='horizontal')


    # plot the gait diagram
    position=axs[idx].get_position()
    ax=stsubplot(fig,position,4,gs1)

    for idx_sub in range(4):
        ax[idx_sub].plot(time,cpg_data[start_point:end_point,0+2*idx_sub],'r')
        ax[idx_sub].plot(time,cpg_data[start_point:end_point,1+2*idx_sub],'b')
        #ax[idx_sub].legend([columnsName_GRF[idx_sub]], loc='upper left',prop=font_legend)
        #ax[idx_sub].legend(['N1', 'N2'], loc='upper left',prop=font_legend)
        ax[idx_sub].grid(which='both',axis='x',color='k',linestyle=':')
        ax[idx_sub].axis([time[0],time[-1],-1.1,1.1],'tight')
        ax[idx_sub].set_xticklabels(labels=[])
        ax[idx_sub].set_yticks([0.0,1.0])
        ax[idx_sub].set_yticklabels(labels=['0.0','1.0'],fontweight='light')
        ax[idx_sub].set_xticks([t for t in np.arange(time[0],time[-1],10,dtype='int')])


    idx=idx+1
    axs[idx].set_yticklabels(labels=[])
    axs[idx].set_yticks([])
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1]+0.1,5,dtype='int')])
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_title('GRF',fontdict=font_label)
    #axs[idx].text(text_x,1,'(A)',fontdict=font_label,rotation='horizontal')


    # plot the gait diagram
    position=axs[idx].get_position()
    ax=stsubplot(fig,position,4,gs1)

    for idx_sub in range(4):
        ax[idx_sub].plot(time,grf_data[start_point:end_point,idx_sub],'r')
        #ax[idx_sub].legend([columnsName_GRF[idx_sub]], loc='upper left',prop=font_legend)
        #ax[idx_sub].legend(['N1', 'N2'], loc='upper left',prop=font_legend)
        ax[idx_sub].grid(which='both',axis='x',color='k',linestyle=':')
        ax[idx_sub].axis([time[0],time[-1],-1.1,1.1],'tight')
        ax[idx_sub].set_xticklabels(labels=[])
        ax[idx_sub].set_yticks([0.0,1.0])
        ax[idx_sub].set_yticklabels(labels=['0.0','1.0'],fontweight='light')
        ax[idx_sub].set_xticks([t for t in np.arange(time[0],time[-1],10,dtype='int')])




    idx=idx+1
    axs[idx].set_yticklabels(labels=[])
    axs[idx].set_yticks([])
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1]+0.1,5,dtype='int')])
    axs[idx].set_title("Gait")
    #axs[idx].text(text_x,1.05,'(D)',fontdict=font_label,rotation='horizontal')

    # preprocess of the data
    threshold = 0.1
    data=grf_data[start_point:end_point,:]
    state= data>threshold
    #pdb.set_trace()
    #state=state.values

    # plot the gait diagram
    position=axs[idx].get_position()
    ax=stsubplot(fig,position,4,gs1)
    xx=[]
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest',vmin=0.0,vmax=1.0)
    for idx_sub in range(4):
        ax[idx_sub].set_yticks([0.1*(idx_sub+1)])
        xx.append(np.where(state[:,idx_sub]>0.7,1.0,0.0))
        ax[idx_sub].imshow(xx[idx_sub].reshape((1,-1)),**barprops)
        #ax[idx_sub].set_ylabel(LegName[idx_sub])
        ax[idx_sub].set_yticklabels(labels=[])
        ax[idx_sub].set_xticklabels(labels=[])




    plt.savefig('/media/suntao/DATA/Research/P1_workspace/Figures/FigS4/FigS4_source222_position.svg')
    plt.show()

