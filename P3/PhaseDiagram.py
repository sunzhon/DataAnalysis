#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import gridspec
import os
import pdb 
plt.rc('font',family='Arial')
import pandas as pd
import re
from brokenaxes import brokenaxes
from mpl_toolkits import mplot3d
from matplotlib import animation

def stsubplot(fig,position,number,gs):
    axprops = dict(xticks=[], yticks=[])
    width_p=position.x1-position.x0; height_p=(position.y1-position.y0)/number
    left_p=position.x0;bottom_p=position.y1-height_p;
    ax=[]
    for idx in range(number):
        #ax.append(fig.add_axes([left_p,bottom_p-idx*height_p,width_p,height_p], **axprops))
        ax.append(brokenaxes(xlims=((76, 116), (146, 160)), hspace=.05, despine=True,fig=fig,subplot_spec=gs))
        pdb.set_trace()
        #ax[-1].set_xticks([])
        #ax[-1].set_xticklabels(labels=[])
    return ax

#def loadData(fileName,columnsName,folderName="/media/suntao/DATA/Research/P1_workspace/Experiments/Experiment_data/fig12/1231060335"):
#def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/0116122819"):
def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/0131103820"):# phase reset Version 1
#def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/0116134200"):# sensory feedback
#def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/0116140352"):# sensory feedback and phase reset version 1
#def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/0116143724"):# sensory feedback and phase reset Vsrion 2
#def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/0116144615"):# phase reset Vsrion 2
# def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/0116145957"):# phase reset Vsrion 2
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
    start_point= 1
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

#    figsize=(5.1,7.1244)
#    fig = plt.figure(figsize=figsize,constrained_layout=False)
#    gs1=gridspec.GridSpec(6,1)#13
#    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.15,right=0.98)
#    axs=[]
#    axs.append(fig.add_subplot(gs1[0:2,0]))
#    axs.append(fig.add_subplot(gs1[2:3,0]))
#    axs.append(fig.add_subplot(gs1[3:4,0]))
#    axs.append(fig.add_subplot(gs1[4:5,0]))
#    axs.append(fig.add_subplot(gs1[5:6,0]))
#
#    #3.1) plot 
#
#    idx=0
#    axs[idx].plot(time,-1.0*pose_data[start_point:end_point,1],'b')
#    #axs[idx].plot(time,joint_data[run_id].iloc[start_point:end_point,4],'b')
#    axs[idx].legend([r'Pitch'], loc='upper left',prop=font_legend)
#    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
#    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
#    axs[idx].axis([time[0],time[-1],-0.15,0.35],'tight')
#    axs[idx].set_xticklabels(labels=[])
#    yticks=[-.15,0.0,.15,0.35]
#    axs[idx].set_yticks(yticks)
#    axs[idx].set_yticklabels(labels=[str(ytick) for ytick in yticks],fontweight='light')
#    #axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
#    axs[idx].set_ylabel('Pitch [rad]',font_label)
#    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1],10,dtype='int')])
#    #axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')
#    #axs[idx].set_title("CPG outputs of the right front leg",font2)
#
#
#
#    idx=1
#    # plot the gait diagram
#    #bax = brokenaxes(xlims=((76, 116), (146, 160)), hspace=.05, despine=False)
#    #axs[idx]=bax
#
#
#    #axs[idx].plot(time,current_data.iloc[start_point:end_point,26]/1000.0,'r')
#    axs[idx].plot(time,lowPassFilter(current_data[start_point:end_point,2],0.1)/1000,'b')
#    axs[idx].plot(time,lowPassFilter(position_data[start_point:end_point,2],0.1)/5.0,'r')
#    axs[idx].legend([r'RF'], loc='upper left',prop=font_legend)
#    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
#    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
#    axs[idx].axis([time[0],time[-1],-0.1,0.51],'tight')
#    axs[idx].set_xticklabels(labels=[])
#    yticks=[-.1,0.0,0.25,0.5]
#    axs[idx].set_yticks(yticks)
#    axs[idx].set_yticklabels(labels=[str(ytick) for ytick in yticks],fontweight='light')
#    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
#    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1]+0.1,5,dtype='int')])
#    #axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')
#    #axs[idx].set_title("Adaptive control input term of the right front leg")
#
#    idx=idx+1
#    axs[idx].plot(time,lowPassFilter(current_data[start_point:end_point,5],0.1)/1000,'b')
#    axs[idx].plot(time,lowPassFilter(position_data[start_point:end_point,5],0.1)/5.0,'r')
#    axs[idx].legend([r'RH'], loc='upper left',prop=font_legend)
#    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
#    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
#    axs[idx].axis([time[0],time[-1],-0.1,0.51],'tight')
#    axs[idx].set_xticklabels(labels=[])
#    yticks=[-.1,0.0,0.25,0.5]
#    axs[idx].set_yticks(yticks)
#    axs[idx].set_yticklabels(labels=[str(ytick) for ytick in yticks],fontweight='light')
#    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
#    #axs[idx].set_title("CPG outputs of the right front leg",font2)
#    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1]+0.1,5,dtype='int')])
#    #axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')
#
#    idx=idx+1
#    axs[idx].plot(time,-1.0*lowPassFilter(current_data[start_point:end_point,8],0.1)/1000,'b')
#    axs[idx].plot(time,lowPassFilter(position_data[start_point:end_point,8],0.1)/5.0,'r')
#    axs[idx].legend([r'LF'], loc='upper left',prop=font_legend)
#    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
#    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
#    axs[idx].axis([time[0],time[-1],-0.1,0.51],'tight')
#    yticks=[-.1,0.0,0.25,0.5]
#    axs[idx].set_yticks(yticks)
#    axs[idx].set_yticklabels(labels=[str(ytick) for ytick in yticks],fontweight='light')
#    axs[idx].set_xticklabels(labels=[])
#    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
#    #axs[idx].set_title("CPG outputs of the right front leg",font2)
#    #axs[idx].set_xlabel('Time [s]',font_label)
#    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1]+0.1,5,dtype='int')])
#    #axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')
#    #axs[idx].set_title("Adaptive control input term of the right front leg")
#
#    idx=idx+1
#    axs[idx].plot(time,-1.0*lowPassFilter(current_data[start_point:end_point,11],0.1)/1000,'b')
#    axs[idx].plot(time,lowPassFilter(position_data[start_point:end_point,11],0.1)/5.0,'r')
#    axs[idx].legend([r'LH'], loc='upper left',prop=font_legend)
#    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
#    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
#    axs[idx].axis([time[0],time[-1],-0.1,0.51],'tight')
#    yticks=[-.1,0.0,0.25,0.5]
#    axs[idx].set_yticks(yticks)
#    axs[idx].set_yticklabels(labels=[str(ytick) for ytick in yticks],fontweight='light')
#    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
#    #axs[idx].set_title("CPG outputs of the right front leg",font2)
#    axs[idx].set_xlabel('Time [s]',font_label)
#    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1]+0.1,5,dtype='int')])
#    axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1])+0.1,5,dtype='int')],fontweight='light')
#

#
#    #plt.savefig('/media/suntao/DATA/Research/P3_workspace/Figures/FigPhase/FigPhase_source222_position.svg')
#    plt.show()

#
#    fig = plt.figure()
#    ax = plt.axes(projection="3d")
#
#    ax.plot3D(time,cpg_data[start_point:end_point,0],cpg_data[start_point:end_point,1],'gray')
#    ax.scatter3D(time,cpg_data[start_point:end_point,0],cpg_data[start_point:end_point,1],cmap='green')
#
#    ax.plot3D(time,cpg_data[start_point:end_point,2],cpg_data[start_point:end_point,3],'gray')
#    ax.scatter3D(time,cpg_data[start_point:end_point,2],cpg_data[start_point:end_point,3],cmap='blue')
#
#
#    ax.plot3D(time,cpg_data[start_point:end_point,4],cpg_data[start_point:end_point,5],'gray')
#    ax.scatter3D(time,cpg_data[start_point:end_point,4],cpg_data[start_point:end_point,5],cmap='red')
#
#
#    ax.plot3D(time,cpg_data[start_point:end_point,6],cpg_data[start_point:end_point,7],'gray')
#    ax.scatter3D(time,cpg_data[start_point:end_point,6],cpg_data[start_point:end_point,7],'+',cmap='yellow')
#    #if cpg_data[start_point:end_point,1]==
#    #if cpg_data[start_point:end_point,1]==
#    ax.set_xlabel('Time[s]',font_label)
#    ax.set_ylabel(r'$O_{1}$',font_label)
#    ax.set_zlabel(r'$O_{2}$',font_label)
#    ax.grid(which='both',axis='x',color='k',linestyle=':')
#    ax.grid(which='both',axis='y',color='k',linestyle=':')
#    ax.grid(which='both',axis='z',color='k',linestyle=':')
#
#
#    fig2=plt.figure()
#    ax2=fig2.add_subplot(1,1,1)
#    ax2.plot(cpg_data[start_point:end_point,0],cpg_data[start_point:end_point,1],'r')
#    ax2.plot(cpg_data[start_point:end_point,2],cpg_data[start_point:end_point,3],'g')
#    ax2.plot(cpg_data[start_point:end_point,4],cpg_data[start_point:end_point,5],'b')
#    ax2.plot(cpg_data[start_point:end_point,6],cpg_data[start_point:end_point,7],'y')
#    ax2.plot(time,cpg_data[start_point:end_point,0],'r')
#    ax2.plot(time,cpg_data[start_point:end_point,2],'g')
#    ax2.plot(time,cpg_data[start_point:end_point,4],'b')
#    ax2.plot(time,cpg_data[start_point:end_point,6],'y')
#    ax2.grid(which='both',axis='x',color='k',linestyle=':')
#    ax2.grid(which='both',axis='y',color='k',linestyle=':')
#    ax2.set_xlabel(r'$O_{1}$',font_label)
#    ax2.set_ylabel(r'$O_{2}$',font_label)
#
#    fig3=plt.figure()
#    ax31=fig3.add_subplot(4,1,1)
#    ax31.plot(time, grf_data[start_point:end_point,0],'r')
#    ax32=fig3.add_subplot(4,1,2)
#    ax32.plot(time, grf_data[start_point:end_point,1],'r')
#    ax33=fig3.add_subplot(4,1,3)
#    ax33.plot(time, grf_data[start_point:end_point,2],'r')
#    ax34=fig3.add_subplot(4,1,4)
#    ax34.plot(time, grf_data[start_point:end_point,3],'r')
#
#    plt.show()
#


    """
    Matplotlib Animation Example

    author: Jake Vanderplas
    email: vanderplas@astro.washington.edu
    website: http://jakevdp.github.com
    license: BSD
    Please feel free to use and modify this, but keep the above information. Thanks!
    """


    # First set up the figure, the axis, and the plot element we want to animate
    figsize=(7.1244,7.1244)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    ax.set_xlabel(r'$O_{1}$')
    ax.set_ylabel(r'$O_{2}$')
    ax.set_title(r'Phase diagram')
    line1, = ax.plot([], [], 'r-',lw=1)
    line2, = ax.plot([], [], 'g-',lw=1)
    line3, = ax.plot([], [], 'b-',lw=1)
    line4, = ax.plot([], [], 'y-',lw=1)

    point1, = ax.plot([], [], 'ro-',lw=1,markersize=6)
    point2, = ax.plot([], [], 'go-',lw=1,markersize=6)
    point3, = ax.plot([], [], 'bo-',lw=1,markersize=6)
    point4, = ax.plot([], [], 'yo-',lw=1,markersize=6)
    ax.grid(which='both',axis='x',color='k',linestyle=':')
    ax.grid(which='both',axis='y',color='k',linestyle=':')
    ax.legend((point1,point2,point3,point4),['RF','RH','LF','LH'],ncol=4)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    line_length=60
    ax.text(-0.45,0.2,r'$a_{i}(t)=\sum_{j=1}^2 w_{ij}*o_{i}(t-1)+b_{i}+f_{i},i=1,2$')
    ax.text(-0.45,0.0,r'$o_{i}(t)=\tanh(a_{1,2})$')
    #ax.text(-0.45,-0.2,r'$f_{1}=-\gamma*GRF*cos(o_{1}(t-1))$')
    #ax.text(-0.45,-0.3,r'$f_{2}=-\gamma*GRF*sin(o_{2}(t-1))$')
    ax.text(-0.45,-0.2,r'$f_{1}=(1-a_{1}(t))*Dirac$')
    ax.text(-0.45,-0.3,r'$f_{2}=(-a_{2}(t))*Dirac$')
    ax.text(-0.45,-0.45,r'$Dirac = 1, GRF > 0.2; 0, otherwise $')

    # initialization function: plot the background of each frame
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])

        point1.set_data([], [])
        point2.set_data([], [])
        point3.set_data([], [])
        point4.set_data([], [])

        time_text.set_text('')
        energy_text.set_text('')
        return line1, line2, line3, line4, point1, point2, point3, point4, time_text, energy_text

    # animation function.  This is called sequentially
    def animate(i):
        index=i%(end_point-start_point-line_length)
        a_start=index; a_end=index+line_length;

        line1.set_data(cpg_data[a_start:a_end,0], cpg_data[a_start:a_end,1])
        point1.set_data(cpg_data[a_end,0], cpg_data[a_end,1])

        line2.set_data(cpg_data[a_start:a_end,2], cpg_data[a_start:a_end,3])
        point2.set_data(cpg_data[a_end,2], cpg_data[a_end,3])

        line3.set_data(cpg_data[a_start:a_end,4], cpg_data[a_start:a_end,5])
        point3.set_data(cpg_data[a_end,4], cpg_data[a_end,5])

        line4.set_data(cpg_data[a_start:a_end,6], cpg_data[a_start:a_end,7])
        point4.set_data(cpg_data[a_end,6], cpg_data[a_end,7])

        time_text.set_text('Time = %.1f [s]' % (index/60.0))
        angle1=math.atan2(cpg_data[a_end,1],cpg_data[a_end,0])-math.atan2(cpg_data[a_end,3],cpg_data[a_end,2])
        if angle1<0.0:
            angle1+=2*np.pi
        energy_text.set_text(r'$\Theta_{1,2}$ = %.2f' % (angle1))
        return line1, point1 ,line2, point2, line3, point3, line4, point4, time_text, energy_text

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=(end_point-start_point-line_length), interval=50, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html

    #anim.save('non-continuous modulation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()
