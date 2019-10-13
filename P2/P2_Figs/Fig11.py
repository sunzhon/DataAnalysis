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
loaddatapath=os.getenv("HOME")+'/PythonProjects/PyPro3/DataAnalysis/P2'
sys.path.append(loaddatapath)
import loaddata as LD
import pdb 
plt.rc('font',family='Arial')


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
    cpg_data=LD.loadData(fileName_CPG,columnsName_CPG)
    grf_data=LD.loadData(fileName_GRF,columnsName_GRF)
    joint_data=LD.loadData(fileName_joint,columnsName_joint)
    ANC_data=LD.loadData(fileName_ANC,columnsName_ANC)
    DL_data=LD.loadData(fileName_CPG,columnsName_CPG)
    grffmi_data=LD.loadData(fileName_GRFFMI,columnsName_GRFFMI)
    pose_data=LD.loadData(fileName_pose,columnsName_pose)

    #2) postprecessing 
    if len(sys.argv)>=2:
        run_id = int(sys.argv[1]) # The id of the experiments
    else:
        run_id = 0

    read_rows=min([cpg_data[run_id].shape[0],grf_data[run_id].shape[0],joint_data[run_id].shape[0],ANC_data[run_id].shape[0],pose_data[run_id].shape[0],grffmi_data[run_id].shape[0]])
    start_point=1360
    end_point=3060#read_rows
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


    
    figsize=(6,4)#8.6614
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(2,1)#13
    gs1.update(hspace=0.35,top=0.95,bottom=0.13,left=0.11,right=0.92)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:1,0]))
    axs.append(fig.add_subplot(gs1[1:2,0]))

    
    xticks=list(range(int(time[0]),int(time[-1])+1,2))
    LegName=["RF","RH","LF","LH"]
    text_x=-1.
    
    #---------------------MI----------------------------------#
    plot_idx=0
    yticks=[0.0,0.15,0.3,0.4,0.45]
    axs[plot_idx].plot(time,grffmi_data[run_id].iloc[start_point:end_point,4],'k',linestyle='-')
    axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[plot_idx].axis([time[0],time[-1],yticks[0],yticks[-1]])
    axs[plot_idx].set_yticks(yticks)
    axs[plot_idx].set_yticklabels(labels=[str(yt) for yt in yticks],fontweight='light')
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_ylabel("Value")
    axs[plot_idx].set_title("MI",pad=3)
    #---------------------Speed------------------------#
    plot_idx=plot_idx+1
    pose=pose_data[run_id].iloc[start_point:end_point,3]
    speed=(pose[1:].values-pose[0:-1].values)*freq

    smooth_speed = np.convolve(speed, np.ones(20)/20, mode='same')
    yticks=[0.0,0.15,0.3]
    axs[plot_idx].plot(time[0:-1],speed,'k',linestyle='--')
    axs[plot_idx].plot(time[0:-1],smooth_speed,'r',linestyle='-')
    axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[plot_idx].axis([time[0],time[-1],yticks[0],yticks[-1]])
    axs[plot_idx].set_yticks(yticks)
    axs[plot_idx].set_yticklabels(labels=[str(yt) for yt in yticks],fontweight='light',color='r')
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_ylabel("v [m/s]",color='r')
    axs[plot_idx].set_title("Speed & Displacement",pad=3)

    axs[plot_idx].set_xlabel('Time [s]',font_label)
    
    axs_twinx=axs[plot_idx].twinx();dispcolor='b'
    axs_twinx.set_ylabel("x[m]",color=dispcolor)
    axs_twinx.plot(time[0:-1],pose[0:-1],color=dispcolor,linestyle='-.')
    axs_twinx.tick_params(axis='y',labelcolor=dispcolor)

if __name__=="__main__":
    Exp4()
    plt.savefig('/media/suntao/DATA/P2 workspace/Experimental Figs/P2Figs/Fig11.eps')
    plt.show()
