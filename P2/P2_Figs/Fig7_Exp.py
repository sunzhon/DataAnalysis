#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import numpy as np
import os
loaddatapath=os.getenv("HOME")+'/PythonProjects/PyPro3/DataAnalysis/P2'
sys.path.append(loaddatapath)
import loaddata as LD
import pdb 
plt.rc('font',family='Arial')

def stsubplot(position,number):
    axprops = dict(xticks=[], yticks=[])
    width_p=position.x1-position.x0; height_p=(position.y1-position.y0)/number
    left_p=position.x0;bottom_p=position.y1-height_p;
    ax=[]
    for idx in range(number):
        ax.append(fig.add_axes([left_p,bottom_p-idx*height_p,width_p,height_p], **axprops))
    return ax


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


    
    figsize=(5.5118,7.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(10,1)
    gs1.update(hspace=0.7,top=0.95,bottom=0.09,left=0.12,right=0.94)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:3,0]))
    axs.append(fig.add_subplot(gs1[3:6,0]))
    axs.append(fig.add_subplot(gs1[6:8,0]))
    axs.append(fig.add_subplot(gs1[8:10,0]))

    xticks=list(range(int(time[0]),int(time[-1])+1,1))
    LegName=["RF","RH","LF","LH"]
    text_x=-1.4
    



    #--------------------CPG-------------------------#
    plot_idx=0
    axs[plot_idx].set_yticklabels(labels=[])
    axs[plot_idx].set_yticks([])
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[])
    axs[plot_idx].set_title('CPG',fontdict=font_label)
    axs[plot_idx].text(text_x,1,'(A)',fontdict=font_label,rotation='horizontal')


    # plot the gait diagram
    position=axs[plot_idx].get_position()
    ax=stsubplot(position,4)

    for idx in range(4):
        ax[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,0+2*idx],'r')
        ax[idx].plot(time,cpg_data[run_id].iloc[start_point:end_point,1+2*idx],'b')
        #ax[idx].legend([columnsName_GRF[idx]], loc='upper left',prop=font_legend)
        #ax[idx].legend(['N1', 'N2'], loc='upper left',prop=font_legend)
        ax[idx].grid(which='both',axis='x',color='k',linestyle=':')
        ax[idx].axis([time[0],time[-1],-1.1,1.1],'tight')
        ax[idx].set_xticks(xticks)
        ax[idx].set_xticklabels(labels=[])
        ax[idx].set_yticks([0.0,1.0])
        ax[idx].set_yticklabels(labels=['0.0','1.0'],fontweight='light')
        ax[idx].set_ylabel(LegName[idx])

    # CPG legend
    l1 = mlines.Line2D([0.6,0.64],[1.1,1.1],color='r', lw=2,transform=axs[plot_idx].transAxes,clip_on=False)
    axs[plot_idx].text(0.65, 1.02, '$O_1$',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=axs[plot_idx].transAxes)
    axs[plot_idx].add_line(l1)

    l2 = mlines.Line2D([0.76,0.80],[1.1,1.1],color='b', lw=2,transform=axs[plot_idx].transAxes,clip_on=False)
    axs[plot_idx].text(0.81, 1.02, '$O_2$',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=axs[plot_idx].transAxes)
    axs[plot_idx].add_line(l2)
    #--------------------GRFF-------------------------#
    plot_idx=plot_idx+1
    axs[plot_idx].set_yticklabels(labels=[])
    axs[plot_idx].set_yticks([])
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[])
    axs[plot_idx].set_title("Ground reaction force (GRF) feedback")
    axs[plot_idx].text(text_x,1,'(B)',fontdict=font_label,rotation='horizontal')

    # plot the gait diagram
    position=axs[plot_idx].get_position()
    ax=stsubplot(position,4)
    for idx in range(4):
        ax[idx].plot(time,grf_data[run_id].iloc[start_point:end_point,idx],'r')
        ax[idx].grid(which='both',axis='x',color='k',linestyle=':')
        ax[idx].axis([time[0],time[-1],-.1,1.0],'tight')
        ax[idx].set_xticks(xticks)
        ax[idx].set_xticklabels(labels=[])
        ax[idx].set_yticks([0.0,0.5])
        ax[idx].set_yticklabels(labels=['0.0','0.5'],fontweight='light')
        ax[idx].set_ylabel(LegName[idx])
        #ax[idx].text(text_x,3,'GRF',fontdict=font_label,rotation='vertical')
    #ax[4].set_title("GRF")
    


    #---------------------ANC------------------------#
    plot_idx=plot_idx+1
    axs[plot_idx].plot(time,abs(ANC_data[run_id].iloc[start_point:end_point,0]),'r')
    axs[plot_idx].plot(time,abs(ANC_data[run_id].iloc[start_point:end_point,1]),'g')
    axs[plot_idx].plot(time,abs(ANC_data[run_id].iloc[start_point:end_point,2]),'b')
    axs[plot_idx].legend(columnsName_ANC[0:3], loc='upper left',prop=font_legend,ncol=2)
    axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[plot_idx].axis([time[0],time[-1],-.5,3.5])
    axs[plot_idx].set_yticks([0.0,1.57,3.14])
    axs[plot_idx].set_yticklabels(labels=['0.0','1.57','3.14'],fontweight='light')
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_xticklabels(labels=[])
    axs[plot_idx].set_ylabel('\u03A6 [rad]',fontdict=font_label,rotation='vertical')
    axs[plot_idx].set_title("Relative phases")
    axs[plot_idx].text(text_x,3.5,'(C)',fontdict=font_label,rotation='horizontal')


    #---------------------Gait------------------------#
    plot_idx=plot_idx+1
    axs[plot_idx].set_yticklabels(labels=[])
    axs[plot_idx].set_yticks([])
    axs[plot_idx].set_xticks(xticks)
    axs[plot_idx].set_title("Gait")
    axs[plot_idx].text(text_x,1.05,'(D)',fontdict=font_label,rotation='horizontal')

    # preprocess of the data
    threshold = 0.0
    data=grf_data[run_id].iloc[start_point:end_point,:]
    state= data>threshold
    state=state.values

    # plot the gait diagram
    position=axs[plot_idx].get_position()
    ax=stsubplot(position,4)
    xx=[]
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest',vmin=0.0,vmax=1.0)
    for idx in range(4):
        ax[idx].set_yticks([0.1*(idx+1)])
        xx.append(np.where(state[:,idx]>0.7,1.0,0.0))
        ax[idx].imshow(xx[idx].reshape((1,-1)),**barprops)
        ax[idx].set_ylabel(LegName[idx])
        ax[idx].set_yticklabels(labels=[])
        ax[idx].set_xticklabels(labels=[])

    axs[idx].set_xlabel('Time [s]',font_label)
    
    # gait legend
    p1 = mpatches.Rectangle(
        (0.55,1.1), 0.03, 0.06,
        fill=True,color='k', transform=axs[plot_idx].transAxes, clip_on=False
        )
    axs[plot_idx].text(0.59, 1.03, 'Stance phase',fontdict=font_legend,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=axs[plot_idx].transAxes)
    axs[plot_idx].add_patch(p1)

    p2 = mpatches.Rectangle(
        (0.79,1.1), 0.03, 0.06,
        fill=False, transform=axs[plot_idx].transAxes, clip_on=False
        )
    axs[plot_idx].text(0.83, 1.03, 'Swing phase',fontdict=font_legend,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=axs[plot_idx].transAxes)
    axs[plot_idx].add_patch(p2)

    plt.savefig('/media/suntao/DATA/P2 workspace/Experimental Figs/P2Figs/fig7.eps')
    plt.show()
