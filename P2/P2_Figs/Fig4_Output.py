#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
loaddatapath=os.getenv("HOME")+'/PythonProjects/PyPro3/DataAnalysis/P2'
sys.path.append(loaddatapath)
import loaddata as LD
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
    freq=40.0 # 40Hz,
    cpg_data=LD.loadData(fileName_CPG,columnsName_CPG)
    grf_data=LD.loadData(fileName_GRF,columnsName_GRF)
    joint_data=LD.loadData(fileName_joint,columnsName_joint)


    #2) postprecessing 
    if len(sys.argv)>=2:
        run_id = int(sys.argv[1]) # The id of the experiments
    else:
        run_id = 0
    read_rows=min([cpg_data[run_id].shape[0],grf_data[run_id].shape[0],joint_data[run_id].shape[0]])
    start_point=0
    end_point=320#read_rows
    time = np.linspace(int(start_point/freq),int(end_point/freq),end_point-start_point)
    #3) plot---------------------------------
    font_legend = {'family' : 'Arial',
    'weight' : 'light',
    'size'   : 10,
    'style'  :'italic'
    }
    font_tick = {'family' : 'Arial',
    'weight' : 'light',
    'size'   : 10,
    #'style'  :'normal'
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

    figsize=(5.5118,4.7244)
    fig,axs = plt.subplots(5,1,figsize=figsize,constrained_layout=False)
    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(left=0.14)
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(right=0.98)
    fig.subplots_adjust(top=0.98)
    xticks=list(range(int(time[0]),int(time[-1])+1,1))

    idx=0
    axs[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,0],'r')
    axs[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,1],'b')
    axs[idx].legend([r'$O_1$', r'$O_2$'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-1.0,1.0],'tight')
    axs[idx].set_xticks(xticks)
    axs[idx].set_ylabel('CPG',fontdict=font_label)
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_yticks([-1.0,0.0,1.0])
    axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    #axs[idx].set_title("CPG outputs of the right front leg",font_title)


    idx=idx+1
    axs[idx].plot(time,joint_data[run_id].iloc[start_point:end_point,2],'b')
    axs[idx].plot(time,joint_data[run_id].iloc[start_point:end_point,1],'r')
    axs[idx].legend([r'$\theta_1$', r'$\theta_2$'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.6,0.3])
    axs[idx].set_yticks([-0.5,-0.1,0.2])
    axs[idx].set_yticklabels(labels=['-0.5','-0.1','0.2'],fontweight='light')
    axs[idx].set_xticks(xticks)
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_ylabel('MNs',fontdict=font_label)
    #axs[idx].set_title("The joint commands for the right front leg")


    idx=idx+1
    axs[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,28],'g')
    axs[idx].plot(time,grf_data[run_id].iloc[start_point:end_point,0],'--y')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].legend(['Exp. GRF', 'Act. GRF'], loc='upper left',prop=font_legend)
    axs[idx].axis([time[0],time[-1],-0.2,1.2])
    axs[idx].set_xticks(xticks)
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_yticks([0.0,0.5,1.0])
    axs[idx].set_yticklabels(labels=['0.0','0.5','1.0'],fontweight='light')
    axs[idx].set_ylabel('FM',fontdict=font_label)
    #axs[idx].set_title("The joint commands for the right front leg")

    '''
    idx=idx+1
    axs[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,28],'r')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.01,1.0])
    #axs[idx].set_title("The forward model output of the right front leg")
    axs[idx].tick_params(labelsize=10)
    labels = axs[idx].get_xticklabels() + axs[idx].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    idx=idx+1
    axs[idx].plot(time,grf_data[run_id].iloc[start_point:end_point,0])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.1,1])
    #axs[idx].set_title("Ground reaction force of the right front leg")
    axs[idx].tick_params(labelsize=10)
    labels = axs[idx].get_xticklabels() + axs[idx].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    '''    
    
    idx=idx+1
    axs[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,8])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.01,0.12])
    axs[idx].legend([r'$\gamma$'], loc='upper left',prop=font_legend)
    axs[idx].set_ylabel('DL',font_label)
    axs[idx].set_yticks([0.0,0.05,0.10])
    axs[idx].set_yticklabels(labels=['0.0','0.05','0.10'],fontweight='light')
    axs[idx].set_xticks(xticks)
    axs[idx].set_xticklabels(labels=[])
    #axs[idx].set_title("Adaptive sensory feedback gain of the right front leg")


    idx=idx+1
    axs[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,20],'r')
    axs[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,21],'b')
    axs[idx].legend([r'$f_1$', r'$f_2$'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].set_xticks(xticks)
    axs[idx].set_xticklabels(labels= [ temp for temp in map(str,xticks)],fontweight='light')
    axs[idx].axis([time[0],time[-1],-0.1,0.11])
    axs[idx].set_ylabel('SFM',fontdict=font_label)
    axs[idx].set_yticks([-0.1,0.0,0.10])
    axs[idx].set_yticklabels(labels=['-0.1','0.0','0.10'],fontweight='light',fontstyle='normal')
    axs[idx].set_xlabel('Time [s]',font_label)
    #axs[idx].set_title("Sensory feedback term of the right front leg")
    
    # add color block at sepcific aera
    for i in range(idx+1):
        axs[i].axvspan(3.75, 4.29, facecolor='#2ca02c', alpha=0.2)

    plt.savefig('/media/suntao/DATA/P2 workspace/Experimental Figs/P2Figs/fig4.eps')
    plt.show()
