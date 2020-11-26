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
loaddatapath=os.getenv("PWD")+'/../'
sys.path.append(loaddatapath)

loaddatapath=os.getenv("PWD")+'/P2/'
sys.path.append(loaddatapath)
import loaddata as LD
import pdb 
import DataProcess as dp

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


def imuplot(folder):
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

    freq=50.0 # 50Hz,
    pose_data=LD.loadData(fileName_pose,columnsName_pose,folder)

    #2) postprecessing 
    if len(sys.argv)>=2:
        run_id = int(sys.argv[1]) # The id of the experiments
    else:
        run_id = 0

    start_point=500
    end_point=7500#read_rows
    time = np.linspace(int(start_point/freq),int(end_point/freq),end_point-start_point)

    lowPassFliter = dp.DataProcess()
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


    
    figsize=(10,4.5)#8.6614
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(4,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.08,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))
    axs.append(fig.add_subplot(gs1[2:4,0]))

    
    xticks=list(range(int(time[0]),int(time[-1])+1,10))
    text_x=-1.

    #---------------------ROLL----------------------------------#
    plot_idx=0
    yticks=[-0.15,0.0,0.3]
    axs[plot_idx].plot(time,pose_data[run_id].iloc[start_point:end_point,0],'b',linestyle='-')
    axs[plot_idx].plot(time,lowPassFliter.ydpprocess(pose_data[run_id].iloc[start_point:end_point,0]),'y',linestyle=':')
    axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[plot_idx].axis([time[0],time[-1],yticks[0],yticks[-1]])
    axs[plot_idx].legend([r'$\alpha$'], loc='upper left',prop=font_legend, ncol=1)
    axs[plot_idx].set_yticks(yticks)
    axs[plot_idx].set_yticklabels(labels=[str(yt) for yt in yticks],fontweight='light')
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[],fontweight='light')
    axs[plot_idx].set_ylabel("Roll [rad]")

    #---------------------PITCH----------------------------------#
    plot_idx=plot_idx+1
    yticks=[-0.8,0.0,0.8]
    axs[plot_idx].plot(time,pose_data[run_id].iloc[start_point:end_point,1],'r',linestyle='-')
    axs[plot_idx].plot(time,lowPassFliter.ydpprocess(pose_data[run_id].iloc[start_point:end_point,1]),'y',linestyle=':')
    axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[plot_idx].axis([time[0],time[-1],yticks[0],yticks[-1]])
    axs[plot_idx].legend([r'$\beta$'], loc='upper left',prop=font_legend, ncol=1)
    axs[plot_idx].set_yticks(yticks)
    axs[plot_idx].set_yticklabels(labels=[str(yt) for yt in yticks],fontweight='light')
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[str(xt) for xt in xticks],fontweight='light')
    axs[plot_idx].set_ylabel("Pitch [rad]")
    axs[plot_idx].set_xlabel('Time [s]',font_label)
    axs[0].set_title('Body attitude angle')

if __name__=="__main__":
    imuplot('12323125')
    plt.savefig('/media/suntao/DATA/Research/P1_workspace/Figures/BodyAttitudeAnglesSlope.svg')
    imuplot('123204016')
    plt.savefig('/media/suntao/DATA/Research/P1_workspace/Figures/BodyAttitudeAnglesFlat.svg')
    plt.show()

