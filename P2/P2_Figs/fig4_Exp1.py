#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
loaddatapath=os.getcwd()+'/../'
sys.path.append(loaddatapath)
import loaddata as LD
import pdb 
plt.rc('font',family='Times New Roman')

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
    columnsName_ANC=['\u03A6'+'12','\u03A6'+'13','\u03A6'+'14','variance']
    freq=40.0 # 40Hz,
    cpg_data=LD.loadData(fileName_CPG,columnsName_CPG)
    grf_data=LD.loadData(fileName_GRF,columnsName_GRF)
    joint_data=LD.loadData(fileName_joint,columnsName_joint)
    ANC_data=LD.loadData(fileName_ANC,columnsName_ANC)
    DL_data=LD.loadData(fileName_CPG,columnsName_CPG)

    #2) postprecessing 
    if len(sys.argv)>=2:
        run_id = int(sys.argv[1]) # The id of the experiments
    else:
        run_id = 0

    read_rows=min([cpg_data[run_id].shape[0],grf_data[run_id].shape[0],joint_data[run_id].shape[0],ANC_data[run_id].shape[0]])
    start_point=0
    end_point=400#read_rows
    time = np.linspace(int(start_point/freq),int(end_point/freq),end_point-start_point)
    #3) plot
    font_legend = {'family' : 'Times New Roman',
    'weight' : 'light',
    'size'   : 10,
    'style'  :'italic'
    }
    font_label = {'family' : 'Times New Roman',
    'weight' : 'light',
    'size'   : 12,
    'style'  :'normal'
    }

    font_title = {'family' : 'Times New Roman',
    'weight' : 'light',
    'size'   : 12,
    'style'  :'normal'
    }


    figsize=(5.5118,4.1244)
    fig,axs = plt.subplots(4,1,figsize=figsize,constrained_layout=False)
    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(left=0.14)
    fig.subplots_adjust(bottom=0.12)
    fig.subplots_adjust(right=0.98)
    fig.subplots_adjust(top=0.98)
    xticks=list(range(int(time[0]),int(time[-1])+1,1))

    idx=0
    axs[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,0],'r')
    axs[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,1],'b')
    axs[idx].legend(['N1', 'N2'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-1.0,1.0],'tight')
    axs[idx].set_xticks(xticks)
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_yticks([-1.0,0.0,1.0])
    axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    axs[idx].set_ylabel('CPG',font_label)
    #axs[idx].set_title("CPG outputs of the right front leg",font2)


    idx=idx+1
    axs[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,8])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.01,0.12])
    axs[idx].legend(['Adaptive feedback gain'], loc='upper left',prop=font_legend)
    axs[idx].set_ylabel('DL',font_label)
    axs[idx].set_yticks([-0.01,0.05,0.10])
    axs[idx].set_yticklabels(labels=['-0.01','0.05','0.10'],fontweight='light')
    axs[idx].set_xticks(xticks)
    axs[idx].set_xticklabels(labels=[])


    idx=idx+1
    axs[idx].plot(time,abs(ANC_data[run_id].iloc[start_point:end_point,0]),'r')
    axs[idx].plot(time,abs(ANC_data[run_id].iloc[start_point:end_point,1]),'g')
    axs[idx].plot(time,abs(ANC_data[run_id].iloc[start_point:end_point,2]),'b')
    axs[idx].legend(columnsName_ANC[0:3], loc='upper left',prop=font_legend,ncol=2)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-.5,3.5])
    axs[idx].set_yticks([-0.5,1.57,3.14])
    axs[idx].set_yticklabels(labels=['-0.5','1.57','3.14'],fontweight='light')
    axs[idx].set_xticks(xticks)
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_ylabel('CPD',font_label)
    #axs[idx].set_title("The relative phases")

    idx=idx+1
    axs[idx].set_yticklabels(labels=[])
    axs[idx].set_yticks([])
    axs[idx].set_xticks(xticks)
    axs[idx].text(-1.3,0.5,'Gait',fontdict=font_label,rotation='vertical')

    position=axs[idx].get_position()
    # preprocess of the data
    threshold = 0.01
    data=grf_data[0]
    state = np.zeros(data.shape, int)
    for i in range(0,4):
        for j in range(len(data)):
            if data.iloc[j,i] < threshold:
                state[j, i] = 0
            else:
                state[j, i] = 1

    # plot the gait diagram
    axprops = dict(xticks=[], yticks=[])
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest')
    width_p=position.x1-position.x0; height_p=(position.y1-position.y0)/4
    left_p=position.x0;bottom_p=position.y1-height_p;

    ax1 = fig.add_axes([left_p,bottom_p,width_p,height_p], **axprops)
    ax2 = fig.add_axes([left_p,bottom_p-height_p,width_p,height_p], **axprops)
    ax3 = fig.add_axes([left_p,bottom_p-2*height_p,width_p,height_p], **axprops)
    ax4 = fig.add_axes([left_p,bottom_p-3*height_p,width_p,height_p], **axprops)

    ax1.set_yticks([0.1]); ax1.set_yticklabels(["RF"])
    ax2.set_yticks([0.2]); ax2.set_yticklabels(["RH"])
    ax3.set_yticks([0.3]); ax3.set_yticklabels(["LF"])
    ax4.set_yticks([0.4]); ax4.set_yticklabels(["LH"])
    x1 = np.where(state[:,0] > 0.7, 1.0, 0.0)
    x2 = np.where(state[:,1] > 0.7, 1.0, 0.0)
    x3 = np.where(state[:,2] > 0.7, 1.0, 0.0)
    x4 = np.where(state[:,3] > 0.7, 1.0, 0.0)

    ax1.imshow(x1.reshape((1, -1)), **barprops)
    ax2.imshow(x2.reshape((1, -1)), **barprops)
    ax3.imshow(x3.reshape((1, -1)), **barprops)
    ax4.imshow(x4.reshape((1, -1)), **barprops)
    

    axs[idx].set_xlabel('Time[s]',font_label)
    #axs[idx].set_title("Adaptive control input term of the right front leg")
    plt.savefig('/media/suntao/DATA/Figs/P2Figs/fig4.eps')
    plt.show()
