#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
loaddatapath=os.getenv("HOME")+'/PythonProjects/PyPro3/DataAnalysis/P2'
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
    fileName_ANC='controlfile_ANC'
    columnsName_ANC=['\u03A6'+'$_{12}$','\u03A6'+'$_{13}$','\u03A6'+'$_{14}$','variance']
    freq=40.0 # 40Hz,
    cpg_data=LD.loadData(fileName_CPG,columnsName_CPG)
    grf_data=LD.loadData(fileName_GRF,columnsName_GRF)
    joint_data=LD.loadData(fileName_joint,columnsName_joint)
    ANC_data=LD.loadData(fileName_ANC,columnsName_ANC)

    #2) postprecessing 
    if len(sys.argv)>=2:
        run_id = int(sys.argv[1]) # The id of the experiments
    else:
        run_id = 0

    read_rows=min([4000,cpg_data[run_id].shape[0],grf_data[run_id].shape[0],joint_data[run_id].shape[0],ANC_data[run_id].shape[0]])
    start_point=240
    end_point=560#read_rows
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


    figsize=(5.5118,4.1244)
    fig,axs = plt.subplots(4,1,figsize=figsize,constrained_layout=False)
    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(left=0.14)
    fig.subplots_adjust(bottom=0.11)
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
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_yticks([-1.0,0.0,1.0])
    axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    axs[idx].set_ylabel('CPG',font_label)
    #axs[idx].set_title("CPG outputs of the right front leg",font2)


    idx=idx+1
    axs[idx].plot(time,abs(ANC_data[run_id].iloc[start_point:end_point,0]),'r')
    axs[idx].plot(time,abs(ANC_data[run_id].iloc[start_point:end_point,1]),'g')
    axs[idx].plot(time,abs(ANC_data[run_id].iloc[start_point:end_point,2]),'b')
    axs[idx].legend(columnsName_ANC[0:3], loc='upper left',prop=font_legend,ncol=2)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-.5,3.5])
    axs[idx].set_yticks([0.0,1.57,3.14])
    axs[idx].set_yticklabels(labels=['0.0','1.57','3.14'],fontweight='light')
    axs[idx].set_xticks(xticks)
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_ylabel('CPD',font_label)
    #axs[idx].set_title("The relative phases")


    idx=idx+1
    axs[idx].plot(time,ANC_data[run_id].iloc[start_point:end_point,3],'k')
    temp_data=ANC_data[run_id].iloc[start_point:end_point,3];threshold=0.4
    temp_data2=[ int(temp)  for temp in temp_data < threshold]
    temp_data3=[temp_data2[i-1] for i, temp in enumerate(temp_data2)];
    temp_data4=np.array(temp_data2)-np.array(temp_data3)
    axs[idx].plot(time,[ ([0.0,1.0][idx >= list(temp_data4).index(1)]) for idx,temp in enumerate(temp_data4)],'y',linestyle='-')
    axs[idx].plot(time,threshold*np.ones(len(time),dtype='float32'),'r',linestyle=':')
    axs[idx].scatter(time[list(temp_data4).index(1)], threshold, color='', marker='o', edgecolors='g', s=60)
    axs[idx].legend(['d',r'$\kappa$','threshold value','activation point'],loc='upper right',prop=font_legend,ncol=2)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.2,3.5])
    axs[idx].set_xticks(xticks)
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_yticks([0.0,2.0,3.2])
    axs[idx].set_yticklabels(labels=['0.0','2.0','3.2'],fontweight='light')
    axs[idx].set_ylabel('EPD',font_label)
    #axs[idx].set_title("The joint commands for the right front leg")

    idx=idx+1
    axs[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,12],'r')
    axs[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,13],'b')
    axs[idx].legend([r'$g_1$', r'$g_2$'], loc='upper left',prop=font_legend,ncol=2)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].set_xticks(xticks)
    axs[idx].set_xticklabels(labels= [ temp for temp in map(str,xticks)],fontweight='light')
    axs[idx].axis([time[0],time[-1],-0.1,0.11])
    axs[idx].set_ylabel('ACI',font_label)
    axs[idx].set_yticks([-0.1,0.0,0.10])
    axs[idx].set_yticklabels(labels=['-0.1','0.0','0.10'],fontweight='light')
    axs[idx].set_xlabel('Time [s]',font_label)
    #axs[idx].set_title("Adaptive control input term of the right front leg")

    '''add color block '''
    for i in range(idx+1):
        axs[i].axvspan(9.48, 9.65, facecolor='#eee1d3ff', alpha=0.7)
        axs[i].axvspan(10.40, 10.7, facecolor='#b3e5e2ff', alpha=0.7)

    plt.savefig('/media/suntao/DATA/P2 workspace/Experimental Figs/P2Figs/fig5.eps')
    plt.show()

