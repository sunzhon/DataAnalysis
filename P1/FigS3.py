#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
loaddatapath=os.getenv("PWD")+'/../'
sys.path.append(loaddatapath)
import loaddata as LD
import pdb 
plt.rc('font',family='Arial')

if __name__=="__main__":
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
    fileName_POSE='sensorfile_POSE'
    columnsName_POSE=['roll','picth','yaw', 'x','y','z']
    freq=60.0 # 60Hz,
    pose_data=LD.loadData(fileName_POSE,columnsName_POSE,'122501132')
    joint_data=LD.loadData(fileName_joint,columnsName_joint,'122501132')
    pose_data2=LD.loadData(fileName_POSE,columnsName_POSE,'122513652')

    #2) postprecessing 
    if len(sys.argv)>=2:
        run_id = int(sys.argv[1]) # The id of the experiments
    else:
        run_id = 0

    read_rows=min([40000,pose_data[run_id].shape[0]])
    start_point=1000
    end_point=read_rows
    time = np.linspace(int(start_point/freq),int(end_point/freq),end_point-start_point)

    read_rows2=min([400000,pose_data2[run_id].shape[0]])
    end_point2=read_rows2
    time2 = np.linspace(int(start_point/freq),int(end_point2/freq),end_point2-start_point)
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

    figsize=(10.5118,4.1244)
    fig,axs = plt.subplots(2,1,figsize=figsize,constrained_layout=False)
    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(left=0.14)
    fig.subplots_adjust(bottom=0.11)
    fig.subplots_adjust(right=0.98)
    fig.subplots_adjust(top=0.98)
    xticks=list(range(int(time[0]),int(time[-1])+1,1))

    idx=0
    axs[idx].plot(time,pose_data[run_id].iloc[start_point:end_point,1],'r')
    #axs[idx].plot(time,joint_data[run_id].iloc[start_point:end_point,4],'b')
    axs[idx].legend([r'Pitch'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-3,3],'tight')
    axs[idx].set_xticks(xticks)
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_yticks([-1.0,0.0,1.0])
    #axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    axs[idx].set_ylabel('CPG',font_label)
    #axs[idx].set_title("CPG outputs of the right front leg",font2)

    idx=1
    axs[idx].plot(time2,pose_data2[run_id].iloc[start_point:end_point2,1],'b')
    axs[idx].legend([r'Pitch'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-1.0,1.0],'tight')
    axs[idx].set_xticks(xticks)
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_yticks([-1.0,0.0,1.0])
    axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    axs[idx].set_ylabel('CPG',font_label)
    #axs[idx].set_title("CPG outputs of the right front leg",font2)
    axs[idx].set_xlabel('Time [s]',font_label)
    #axs[idx].set_title("Adaptive control input term of the right front leg")

    

    '''add color block '''
    for i in range(idx+1):
        axs[i].axvspan(9.48, 9.65, facecolor='#eee1d3ff', alpha=0.7)
        axs[i].axvspan(10.40, 10.7, facecolor='#b3e5e2ff', alpha=0.7)

    plt.savefig('/media/suntao/DATA/Research/P1_workspace/Figures/Fig12_source.svg')
    plt.show()

