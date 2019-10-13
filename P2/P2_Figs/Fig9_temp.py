#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec
import numpy as np
import os
loaddatapath=os.getenv("HOME")+'/PythonProjects/PyPro3/DataAnalysis/P2'
sys.path.append(loaddatapath)
import loaddata as LD
import pdb 
plt.rc('font',family='Arial')

def stsubplot(fig,position,number):
    axprops = dict(xticks=[], yticks=[])
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest')
    width_p=position.x1-position.x0; height_p=(position.y1-position.y0)/number
    left_p=position.x0;bottom_p=position.y1-height_p;
    ax=[]
    for idx in range(number):
        ax.append(fig.add_axes([left_p,bottom_p-idx*height_p,width_p,height_p], **axprops))
        ax[idx].set_xticklabels(labels=[])
        ax[idx].set_yticklabels(labels=[])
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
    return ax


def Exp3():
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
    start_point=600
    end_point=min(1200,read_rows)
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


    
    figsize=(3.6614,8.6614)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(13,1)
    gs1.update(hspace=0.8,top=0.95,bottom=0.065,left=0.16,right=0.94)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:3,0]))
    axs.append(fig.add_subplot(gs1[3:6,0]))
    axs.append(fig.add_subplot(gs1[6:9,0]))
    axs.append(fig.add_subplot(gs1[9:11,0]))
    axs.append(fig.add_subplot(gs1[11:13,0]))

    xticks=list(range(int(time[0]),int(time[-1])+1,1))
    LegName=["RF","RH","LF","LH"]
    text_x=-1.
    

    #--------------------GRFF and MI-------------------------#
    plot_idx=0
    axs[plot_idx].set_yticklabels(labels=[])
    axs[plot_idx].set_yticks([])
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[])
    axs[plot_idx].set_title("Ground reaction force feedback")
    #axs[plot_idx].text(text_x,0.5,'GRFF',fontdict=font_label,rotation='vertical')

    # plot the gait diagram
    position=axs[plot_idx].get_position()
    ax=stsubplot(fig,position,5)
    for idx in range(5):
        ax[idx].plot(time,grffmi_data[run_id].iloc[start_point:end_point,idx],'r')
        ax[idx].grid(which='both',axis='x',color='k',linestyle=':')
        ax[idx].axis([time[0],time[-1],-.1,.3],'tight')
        ax[idx].set_xticks(xticks)
        ax[idx].set_xticklabels(labels=[])
        ax[idx].set_yticks([0.0,0.15,.3])
        ax[idx].set_yticklabels(labels=['0.0','0.15',''],fontweight='light')
        ax[idx].set_ylabel(columnsName_GRFFMI[idx])
        #ax[idx].text(text_x,3,'GRF',fontdict=font_label,rotation='vertical')
    #ax[4].set_title("GRF")


    #--------------------CPG-------------------------#
    plot_idx=plot_idx+1
    axs[plot_idx].set_yticklabels(labels=[])
    axs[plot_idx].set_yticks([])
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[])
    axs[plot_idx].set_title('CPG',fontdict=font_label)

    # plot the gait diagram
    position=axs[plot_idx].get_position()
    ax=stsubplot(fig,position,4)

    for idx in range(4):
        ax[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,0+2*idx],'r')
        ax[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,1+2*idx],'b')
        #ax[idx].legend([columnsName_GRF[idx]], loc='upper left',prop=font_legend)
        #ax[idx].legend(['N1', 'N2'], loc='upper left',prop=font_legend)
        ax[idx].grid(which='both',axis='x',color='k',linestyle=':')
        ax[idx].axis([time[0],time[-1],-1.1,1.1],'tight')
        ax[idx].set_xticks(xticks)
        ax[idx].set_xticklabels(labels=[])
        ax[idx].set_yticks([-1.0,0.0,1.0])
        ax[idx].set_yticklabels(labels=['','0.0',''],fontweight='light')
        ax[idx].set_ylabel(LegName[idx])
        #ax[idx].text(text_x,0,LegName[0],fontdict=font_label,rotation='vertical')


    #---------------------ANC------------------------#
    plot_idx=plot_idx+1
    axs[plot_idx].plot(time,abs(ANC_data[run_id].iloc[start_point:end_point,0]),'r')
    axs[plot_idx].plot(time,abs(ANC_data[run_id].iloc[start_point:end_point,1]),'g')
    axs[plot_idx].plot(time,abs(ANC_data[run_id].iloc[start_point:end_point,2]),'b')
    axs[plot_idx].legend(columnsName_ANC[0:3], loc='upper left',prop=font_legend,ncol=2)
    axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[plot_idx].axis([time[0],time[-1],-.5,3.5])
    axs[plot_idx].set_yticks([-0.5,1.57,3.14])
    axs[plot_idx].set_yticklabels(labels=['-0.5','1.57','3.14'],fontweight='light')
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[])
    axs[plot_idx].set_ylabel('\u03A6 [rad]',fontdict=font_label,rotation='vertical')
    axs[plot_idx].set_title("Relative phases")
    '''
    #---------------------Gait------------------------#
    plot_idx=plot_idx+1
    axs[plot_idx].set_yticklabels(labels=[])
    axs[plot_idx].set_yticks([])
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[])
    axs[plot_idx].set_title("Gait")
    #axs[plot_idx].text(text_x,0.5,'Gait',fontdict=font_label,rotation='vertical')

    # preprocess of the data
    threshold = 0.0
    data=grf_data[run_id].iloc[start_point:end_point,:]
    state= data>threshold
    state=state.values

    # plot the gait diagram
    position=axs[plot_idx].get_position()
    ax=stsubplot(fig,position,4)
    xx=[]
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest')
    for idx in range(4):
        ax[idx].set_yticks([0.1*(idx+1)])
        xx.append(np.where(state[:,idx]>0.7,1.0,0.0))
        ax[idx].imshow(xx[idx].reshape((1,-1)),**barprops)
        ax[idx].set_ylabel(LegName[idx])
    '''



    #---------------------Gait------------------------#
    plot_idx=plot_idx+1
    axs[plot_idx].set_yticklabels(labels=[])
    axs[plot_idx].set_yticks([])
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[])
    axs[plot_idx].set_title("Gait",loc="left")
    #axs[plot_idx].text(text_x,0.5,'Gait',fontdict=font_label,rotation='vertical')

    # preprocess of the data
    threshold = 0.0
    data=grf_data[run_id].iloc[start_point:end_point,:]
    state= data>threshold
    state=state.values

    # plot the gait diagram
    position=axs[plot_idx].get_position()
    ax=stsubplot(fig,position,4)
    xx=[]
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest')
    for idx in range(4):
        ax[idx].set_yticks([0.1*(idx+1)])
        xx.append(np.where(state[:,idx]>0.7,1.0,0.0))
        ax[idx].imshow(xx[idx].reshape((1,-1)),**barprops)
        ax[idx].set_ylabel(LegName[idx])

    # gait legend
    p1 = mpatches.Rectangle(
        (0.25,1.1), 0.03, 0.06,
        fill=True,color='k', transform=axs[plot_idx].transAxes, clip_on=False
        )
    axs[plot_idx].text(0.3, 1.05, 'Stance phase',fontdict=font_legend,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=axs[plot_idx].transAxes)
    axs[plot_idx].add_patch(p1)

    p2 = mpatches.Rectangle(
        (0.65,1.1), 0.03, 0.06,
        fill=False, transform=axs[plot_idx].transAxes, clip_on=False
        )
    axs[plot_idx].text(0.7, 1.05, 'Swing phase',fontdict=font_legend,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=axs[plot_idx].transAxes)
    axs[plot_idx].add_patch(p2)

    #---------------------Displacement------------------------#
    plot_idx=plot_idx+1
    axs[plot_idx].plot(time,pose_data[run_id].iloc[start_point:end_point,3],'k',linestyle='--')
    axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[plot_idx].axis([time[0],time[-1],0.8,1.4])
    axs[plot_idx].set_yticks([0.8,1.1,1.4])
    axs[plot_idx].set_yticklabels(labels=['0.8','1.1','1.4'],fontweight='light')
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_ylabel("x [m]")
    axs[plot_idx].set_title("Dispacement of robot")

    axs[plot_idx].set_xlabel('Time [s]',font_label)

if __name__=="__main__":
    Exp3()
    plt.savefig('/media/suntao/DATA/P2 workspace/Experimental Figs/P2Figs/fig81.eps')
    plt.show()
