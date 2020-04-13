#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec
import numpy as np
import scipy 
import os
import pdb 
import re
import pandas as pd
plt.rc('font',family='Arial')

def loadData(fileName,columnsName,folderName="/media/suntao/DATA/Research/P2_workspace/Experiments/Experiment_data/fig14/122201742"):
    '''
    load data from a file
    fileName: the name of file that you want to read
    columnsName: it the column name of the file
    Note: the args of sys is file_id and date of the file
    '''
        
    #1) load data from file
    resource_data=[]
    rows_num=[]
    data_file = folderName +"/"+ fileName + ".csv"
    resource_data.append(pd.read_csv(data_file, sep='\t', index_col=0,header=None, names=columnsName, skip_blank_lines=True,dtype=str))
    rows_num.append(resource_data[-1].shape[0])# how many rows

    fine_data = []
    min_rows=min(rows_num)
    read_rows=min_rows-1

    for data in resource_data:
        fine_data.append(data.iloc[0:read_rows,:].astype(float))# 数据行对齐
    return fine_data


def stsubplot(fig,position,number):
    axprops = dict(xticks=[], yticks=[])
    width_p=position.x1-position.x0; height_p=(position.y1-position.y0)/number
    left_p=position.x0;bottom_p=position.y1-height_p;
    ax=[]
    for idx in range(number):
        ax.append(fig.add_axes([left_p,bottom_p-idx*height_p,width_p,height_p], **axprops))
        ax[-1].set_xticks([])
        ax[-1].set_xticklabels(labels=[])
    return ax


def Exp4():
    columnsName_CPG=['RFO1','RFO2','RHO1','RHO2','LFO1','LFO2','LHO1','LKO2',
                     'RFSA','RHSA','LFSA','LHSA',
                     'RFACITerm0','RFACITerm1','RHACITerm0','RHACITerm1','LFACITerm0','LFACITerm1','LHACITerm0','LHACITerm1',
                     'RFSFTerm0','RFSFTerm1','RHSFTerm0','RHSFTerm1','LFSFTerm0','LFSFTerm1','LHSFTerm0','LHSFTerm1',
                     'RFFM','RHFM','LFFM','LHFM']
    fileName_CPG="controlfile_CPG"
    columnsName_GRF=['RF','RH','LF','LH']
    fileName_GRF='sensorfile_GRF'
    fileName_joint='sensorfile_joint'
    columnsName_joint=['j1','j2','j3','j4','j5,''j6',
                        'j7','j8','j9','j10','j11','j12','s'
                      ]*2
    fileName_ANC='controlfile_ANC'
    columnsName_ANC=['\u03A6'+'$_{12}$','\u03A6'+'$_{13}$','\u03A6'+'$_{14}$','variance']

    fileName_GRFFMI='feedbackfile_GRFF_MI'
    columnsName_GRFFMI=['RF','RH','LF','LH','MI']

    fileName_pose='sensorfile_POSE'
    columnsName_pose=["roll","pitch","yaw","x","y","z"]

    freq=40.0 # 40Hz,
    cpg_data=loadData(fileName_CPG,columnsName_CPG)
    grf_data=loadData(fileName_GRF,columnsName_GRF)
    joint_data=loadData(fileName_joint,columnsName_joint)
    ANC_data=loadData(fileName_ANC,columnsName_ANC)
    DL_data=loadData(fileName_CPG,columnsName_CPG)
    grffmi_data=loadData(fileName_GRFFMI,columnsName_GRFFMI)
    pose_data=loadData(fileName_pose,columnsName_pose)

    #2) postprecessing 
    if len(sys.argv)>=2:
        run_id = int(sys.argv[1]) # The id of the experiments
    else:
        run_id = 0

    read_rows=min([cpg_data[run_id].shape[0],grf_data[run_id].shape[0],joint_data[run_id].shape[0],ANC_data[run_id].shape[0],pose_data[run_id].shape[0],grffmi_data[run_id].shape[0]])
    start_point=8820
    end_point=10180#read_rows
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


    
    figsize=(6,4.5)#8.6614
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(4,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:1,0]))
    axs.append(fig.add_subplot(gs1[1:2,0]))
    axs.append(fig.add_subplot(gs1[2:3,0]))
    axs.append(fig.add_subplot(gs1[3:4,0]))

    
    xticks=list(range(int(time[0]),int(time[-1])+1,2))
    LegName=["RF","RH","LF","LH"]
    text_x=-1.
    
    #---------------------RF----------------------------------#
    plot_idx=0
    yticks=[-0.8,-0.2,0.4]
    axs[plot_idx].plot(time,joint_data[run_id].iloc[start_point:end_point,1],'b',linestyle='-')
    axs[plot_idx].plot(time,joint_data[run_id].iloc[start_point:end_point,2],'r',linestyle='-')
    axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[plot_idx].axis([time[0],time[-1],yticks[0],yticks[-1]])
    axs[plot_idx].legend([r'$\theta_1$', r'$\theta_2$'], loc='upper left',prop=font_legend, ncol=2)
    axs[plot_idx].set_yticks(yticks)
    axs[plot_idx].set_yticklabels(labels=[str(yt) for yt in yticks],fontweight='light')
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[],fontweight='light')
    axs[plot_idx].set_ylabel(LegName[plot_idx]+" motor\ncommands")
    axs[plot_idx].axvline(x=89, color='k', linestyle='--')
    axs[plot_idx].axvline(x=107, color='k', linestyle='--')

    #---------------------RH----------------------------------#
    plot_idx=plot_idx+1
    yticks=[-0.8,-0.2,0.4]
    axs[plot_idx].plot(time,joint_data[run_id].iloc[start_point:end_point,7],'b',linestyle='-')
    axs[plot_idx].plot(time,joint_data[run_id].iloc[start_point:end_point,8],'r',linestyle='-')
    axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[plot_idx].axis([time[0],time[-1],yticks[0],yticks[-1]])
    axs[plot_idx].legend([r'$\theta_1$', r'$\theta_2$'], loc='upper left',prop=font_legend, ncol=2)
    axs[plot_idx].set_yticks(yticks)
    axs[plot_idx].set_yticklabels(labels=[str(yt) for yt in yticks],fontweight='light')
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[],fontweight='light')
    axs[plot_idx].set_ylabel(LegName[plot_idx]+" motor\ncommands")
    axs[plot_idx].axvline(x=89, color='k', linestyle='--')
    axs[plot_idx].axvline(x=107, color='k', linestyle='--')

    #---------------------LF----------------------------------#
    plot_idx=plot_idx+1
    yticks=[-0.8,-0.2,0.4]
    axs[plot_idx].plot(time,joint_data[run_id].iloc[start_point:end_point,13],'b',linestyle='-')
    axs[plot_idx].plot(time,joint_data[run_id].iloc[start_point:end_point,14],'r',linestyle='-')
    axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[plot_idx].axis([time[0],time[-1],yticks[0],yticks[-1]])
    axs[plot_idx].legend([r'$\theta_1$', r'$\theta_2$'], loc='upper left',prop=font_legend, ncol=2)
    axs[plot_idx].set_yticks(yticks)
    axs[plot_idx].set_yticklabels(labels=[str(yt) for yt in yticks],fontweight='light')
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[],fontweight='light')
    axs[plot_idx].set_ylabel(LegName[plot_idx]+" motor\ncommands")
    axs[plot_idx].axvline(x=89, color='k', linestyle='--')
    axs[plot_idx].axvline(x=107, color='k', linestyle='--')

    #---------------------LH----------------------------------#
    plot_idx=plot_idx+1
    yticks=[-0.8,-0.2,0.4]
    axs[plot_idx].plot(time,joint_data[run_id].iloc[start_point:end_point,19],'b',linestyle='-')
    axs[plot_idx].plot(time,joint_data[run_id].iloc[start_point:end_point,20],'r',linestyle='-')
    axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[plot_idx].axis([time[0],time[-1],yticks[0],yticks[-1]])
    axs[plot_idx].legend([r'$\theta_1$', r'$\theta_2$'], loc='upper left',prop=font_legend, ncol=2)
    axs[plot_idx].set_yticks(yticks)
    axs[plot_idx].set_yticklabels(labels=[str(yt) for yt in yticks],fontweight='light')
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_ylabel(LegName[plot_idx]+" motor\ncommands")
    axs[plot_idx].axvline(x=89, color='k', linestyle='--')
    axs[plot_idx].axvline(x=107, color='k', linestyle='--')
    #axs[plot_idx].set_title(LegName[plot_idx],pad=3)

    axs[plot_idx].set_xlabel('Time [s]',font_label)

    axs[0].text(time[0]+2,0.5,'Walking forward',fontdict=font_label,rotation='horizontal')
    axs[0].text((time[0]+time[-1])/2,0.5,'Turning right',fontdict=font_label,rotation='horizontal')
    axs[0].text(time[-1]-6,0.5,'Turning left',fontdict=font_label,rotation='horizontal')

if __name__=="__main__":
    Exp4()
    plt.savefig('/media/suntao/DATA/Research/P2_workspace/Experiments/P2Figs/Fig14_resource.svg')
    plt.show()
