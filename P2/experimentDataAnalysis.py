#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
This module was developed to analyze and plot experimental data for P2 paper

This module depends on a user package CRCF (Custmization robot control framework) package

Author: suntao
Email: suntao.hn@gmail.com
Created Date: probably on 1-1-2019
'''

import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import gridspec
import os
import pdb 
import termcolor
import gnureadline

#from fontTools.ttLib import TTFont
#font = TTFont('/path/to/font.ttf')

plt.rc('font',family='Arial')
import pandas as pd
import re
import time as localtimepkg
from brokenaxes import brokenaxes
from mpl_toolkits import mplot3d
from matplotlib import animation
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import warnings
from matplotlib.animation import FuncAnimation
import matplotlib.font_manager

# import customization robot control framework
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR+"/../")
from CRCF.metrics import *
from CRCF.data_manager import *

warnings.simplefilter('always', UserWarning)


from statannotations.Annotator import Annotator

'''
Global parameters:

Robot configurations and parameters of Lilibot

'''



def stsubplot(fig,position,number,gs):
    axprops = dict(xticks=[], yticks=[])
    width_p=position.x1-position.x0; height_p=(position.y1-position.y0)/number
    left_p=position.x0;bottom_p=position.y1-height_p;
    ax=[]
    for idx in range(number):
        ax.append(fig.add_axes([left_p,bottom_p-idx*height_p,width_p,height_p], **axprops))
        #ax.append(brokenaxes(xlims=((76, 116), (146, 160)), hspace=.05, despine=True,fig=fig,subplot_spec=gs))
        ax[-1].set_xticks([])
        ax[-1].set_xticklabels(labels=[])
    return ax



def cpg_diagram(fig,axs,gs,grf_data,time):
    '''
    plot ground reaction forces of four legs in curves
    '''
    position=axs.get_position()
    axs.set_yticks([])
    axs.set_yticklabels(labels=[])
    axs.set_xticks([])
    axs.set_xticklabels(labels=[])
    axs.set_xlim([int(min(time)),int(max(time))])


    c4_1color=(46/255.0, 77/255.0, 129/255.0)
    c4_2color=(0/255.0, 198/255.0, 156/255.0)
    c4_3color=(255/255.0, 1/255.0, 118/255.0)
    c4_4color=(225/255.0, 213/255.0, 98/255.0)
    colors=[c4_1color, c4_2color, c4_3color, c4_4color]
    ax=stsubplot(fig,position,4,gs)
    LegName=['RF','RH', 'LF', 'LH']
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest',vmin=0.0,vmax=1.0)
    for idx in range(4):
        ax[idx].plot(time,grf_data[:,idx],color=colors[idx])
        ax[idx].set_ylabel(LegName[idx])
        ax[idx].set_yticks([0,1])
        ax[idx].set_yticklabels(labels=[0,1])
        ax[idx].set_ylim([-1.1,1.1])
        ax[idx].set_xlim([int(min(time)),int(max(time))])
        #xticks=np.arange(int(min(time)),int(max(time))+1,5)
        #ax[idx].set_xticks(xticks)
        #ax[idx].set_xticklabels([])
        ax[idx].grid(which='both',axis='x',color='k',linestyle=':')
        ax[idx].grid(which='both',axis='y',color='k',linestyle=':')


def grf_diagram(fig,axs,gs,grf_data,time):
    '''
    plot ground reaction forces of four legs in curves
    '''
    position=axs.get_position()
    axs.set_yticks([])
    axs.set_yticklabels(labels=[])
    axs.set_xticks([])
    axs.set_xticklabels(labels=[])

    c4_1color=(46/255.0, 77/255.0, 129/255.0)
    c4_2color=(0/255.0, 198/255.0, 156/255.0)
    c4_3color=(255/255.0, 1/255.0, 118/255.0)
    c4_4color=(225/255.0, 213/255.0, 98/255.0)
    colors=[c4_1color, c4_2color, c4_3color, c4_4color]
    ax=stsubplot(fig,position,4,gs)
    LegName=['RF','RH', 'LF', 'LH']
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest',vmin=0.0,vmax=1.0)
    for idx in range(4):
        ax[idx].plot(time,grf_data[:,idx],color=colors[idx])
        ax[idx].set_ylabel(LegName[idx])
        ax[idx].set_yticks([0.5,1])
        ax[idx].set_yticklabels(labels=[0.5,1])
        ax[idx].set_ylim([-0.1,1.1])
        ax[idx].set_xlim([int(min(time)),int(max(time))])
        #xticks=np.arange(int(min(time)),int(max(time))+1,5)
        #ax[idx].set_xticks(xticks)
        #ax[idx].set_xticklabels([])
        ax[idx].grid(which='both',axis='x',color='k',linestyle=':')
        ax[idx].grid(which='both',axis='y',color='k',linestyle=':')

def gait_diagram(fig,axs,gs,gait_data):
    '''
    plot gait diagram using while and black block to indicate swing and stance phase
    '''
    position=axs.get_position()
    axs.set_yticks([])
    axs.set_yticklabels(labels=[])
    axs.set_xticks([])
    axs.set_xticklabels(labels=[])
    #axs.set_title("Gait",loc="left",pad=2)



    # colors
    c4_1color=(46/255.0, 77/255.0, 129/255.0)
    c4_2color=(0/255.0, 198/255.0, 156/255.0)
    c4_3color=(255/255.0, 1/255.0, 118/255.0)
    c4_4color=(225/255.0, 213/255.0, 98/255.0)
    colors=[c4_1color, c4_2color, c4_3color, c4_4color]
    cmap = (mpl.colors.ListedColormap(['white', 'cyan', 'yellow', 'royalblue']).with_extremes(over='red', under='blue'))
    ax=stsubplot(fig,position,4,gs)
    xx=[]
    LegName=['RF','RH', 'LF', 'LH']
    #barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest',vmin=0.0,vmax=1.0)
    barprops = dict(aspect='auto', cmap=cmap, interpolation='nearest',vmin=0.0,vmax=1.0)
    for idx in range(4):
        ax[idx].set_yticks([0.1*(idx+1)])
        xx.append(np.where(gait_data[:,idx]>0.2*max(gait_data[:,idx]),1.0,0.0)) # > 0.2 times of max_GRF, then leg on stance phase
        ax[idx].imshow(xx[idx].reshape((1,-1)),**barprops)
        ax[idx].set_ylabel(LegName[idx])
        ax[idx].set_yticklabels(labels=[])

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


def touch_convergence_moment_identification(grf_data,cpg_data,time):
    '''
    Identify the touch moment and phase convergence moment, and output phi_std
    '''
    touch_idx,convergence_idx=calculate_touch_idx_phaseConvergence_idx(time,grf_data,cpg_data)
    phi_std=calculate_phase_diff_std(cpg_data[touch_idx:,:],time[touch_idx:]) # 相位差的标准差
    return touch_idx, convergence_idx, phi_std


def AvgerageGaitBeta(beta):
    '''
    Calculate the average duty factors of every legs within a walking period.
    '''
    average_beta=[]
    for i in range(len(beta)):
        average_beta.append(np.mean(beta[i]))

    return average_beta

def Animate_phase_transition(cpg_data):
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
        index=i%(cpg_data.shape[0]-line_length)
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
                                   frames=(cpg_data.shape[0]-line_length), interval=50, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html

    #anim.save('non-continuous modulation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()

def plot_phase_transition_animation(data_file_dic,start_time=5,end_time=30,freq=60.0,experiment_categories=['0.0'],trial_id=0):
    ''' 
    This is for plot the phase diff transition that is shown by a small mive
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_id, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''
    # 1) read data
    titles_files_categories, categories=load_data_log(data_file_dic)

    cpg={}

    for category, files_name in titles_files_categories: #category is a files_name categorys
        cpg[category]=[]  #files_name is the table of the files_name category
        if category in experiment_categories:
            print(category)
            for idx in files_name.index:
                folder_category= data_file_dic + files_name['data_files'][idx]
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                # 2)  data process
                print(folder_category)
                cpg[category].append(cpg_data)

    #3) plot
    Animate_phase_transition(cpg[experiment_categories[0]][trial_id])
    
def PhaseAnalysis(data_file_dic,start_time=90,end_time=1200,freq=60.0,experiment_categories=['0.0'],trial_id=0):
    # 1) read data
    titles_files_categories, categories =load_data_log(data_file_dic)

    cpg={}
    current={}
    position={}

    for category, files_name in titles_files_categories: #category is a files_name categorys
        cpg[category]=[]  #files_name is the table of the files_name category
        current[category]=[]
        position[category]=[]
        print(category)
        for idx in files_name.index:
            folder_category= data_file_dic + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
            # 2)  data process
            print(folder_category)
            cpg[category].append(cpg_data)
            current[category].append(current_data)
            position[category].append(position_data)

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

    figsize=(5.1,7.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.15,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))
    axs.append(fig.add_subplot(gs1[2:3,0]))
    axs.append(fig.add_subplot(gs1[3:4,0]))
    axs.append(fig.add_subplot(gs1[4:5,0]))
    axs.append(fig.add_subplot(gs1[5:6,0]))

    #3.1) plot 

    idx=0
    axs[idx].plot(time,-1.0*pose_data[:,1],'b')
    #axs[idx].plot(time,joint_data[run_id].iloc[:,4],'b')
    axs[idx].legend([r'Pitch'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.15,0.35],'tight')
    axs[idx].set_xticklabels(labels=[])
    yticks=[-.15,0.0,.15,0.35]
    axs[idx].set_yticks(yticks)
    axs[idx].set_yticklabels(labels=[str(ytick) for ytick in yticks],fontweight='light')
    #axs[idx].set_yticklabels(labels=['-1.0','0.0','1.0'],fontweight='light')
    axs[idx].set_ylabel('Pitch [rad]',font_label)
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1],10,dtype='int')])
    #axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')
    #axs[idx].set_title("CPG outputs of the right front leg",font2)



    idx=1
    # plot the gait diagram
    #bax = brokenaxes(xlims=((76, 116), (146, 160)), hspace=.05, despine=False)
    #axs[idx]=bax


    #axs[idx].plot(time,current_data.iloc[start_time:end_time,26]/1000.0,'r')
    axs[idx].plot(time,lowPassFilter(current_data[:,2],0.1)/1000,'b')
    axs[idx].plot(time,lowPassFilter(position_data[:,2],0.1)/5.0,'r')
    axs[idx].legend([r'RF'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.1,0.51],'tight')
    axs[idx].set_xticklabels(labels=[])
    yticks=[-.1,0.0,0.25,0.5]
    axs[idx].set_yticks(yticks)
    axs[idx].set_yticklabels(labels=[str(ytick) for ytick in yticks],fontweight='light')
    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1]+0.1,5,dtype='int')])
    #axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')
    #axs[idx].set_title("Adaptive control input term of the right front leg")

    idx=idx+1
    axs[idx].plot(time,lowPassFilter(current_data[:,5],0.1)/1000,'b')
    axs[idx].plot(time,lowPassFilter(position_data[:,5],0.1)/5.0,'r')
    axs[idx].legend([r'RH'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.1,0.51],'tight')
    axs[idx].set_xticklabels(labels=[])
    yticks=[-.1,0.0,0.25,0.5]
    axs[idx].set_yticks(yticks)
    axs[idx].set_yticklabels(labels=[str(ytick) for ytick in yticks],fontweight='light')
    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
    #axs[idx].set_title("CPG outputs of the right front leg",font2)
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1]+0.1,5,dtype='int')])
    #axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')

    idx=idx+1
    axs[idx].plot(time,-1.0*lowPassFilter(current_data[:,8],0.1)/1000,'b')
    axs[idx].plot(time,lowPassFilter(position_data[:,8],0.1)/5.0,'r')
    axs[idx].legend([r'LF'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.1,0.51],'tight')
    yticks=[-.1,0.0,0.25,0.5]
    axs[idx].set_yticks(yticks)
    axs[idx].set_yticklabels(labels=[str(ytick) for ytick in yticks],fontweight='light')
    axs[idx].set_xticklabels(labels=[])
    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
    #axs[idx].set_title("CPG outputs of the right front leg",font2)
    #axs[idx].set_xlabel('Time [s]',font_label)
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1]+0.1,5,dtype='int')])
    #axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1]),10,dtype='int')],fontweight='light')
    #axs[idx].set_title("Adaptive control input term of the right front leg")

    idx=idx+1
    axs[idx].plot(time,-1.0*lowPassFilter(current_data[:,11],0.1)/1000,'b')
    axs[idx].plot(time,lowPassFilter(position_data[:,11],0.1)/5.0,'r')
    axs[idx].legend([r'LH'], loc='upper left',prop=font_legend)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].axis([time[0],time[-1],-0.1,0.51],'tight')
    yticks=[-.1,0.0,0.25,0.5]
    axs[idx].set_yticks(yticks)
    axs[idx].set_yticklabels(labels=[str(ytick) for ytick in yticks],fontweight='light')
    axs[idx].set_ylabel('Knee joint\ncurrent [A]',font_label)
    #axs[idx].set_title("CPG outputs of the right front leg",font2)
    axs[idx].set_xlabel('Time [s]',font_label)
    axs[idx].set_xticks([t for t in np.arange(time[0],time[-1]+0.1,5,dtype='int')])
    axs[idx].set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1])+0.1,5,dtype='int')],fontweight='light')



    #plt.savefig('/media/suntao/DATA/Research/P3_workspace/Figures/FigPhase/FigPhase_source222_position.svg')
    plt.show()


    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.plot3D(time,cpg_data[:,0],cpg_data[:,1],'gray')
    ax.scatter3D(time,cpg_data[:,0],cpg_data[:,1],cmap='green')

    ax.plot3D(time,cpg_data[:,2],cpg_data[:,3],'gray')
    ax.scatter3D(time,cpg_data[:,2],cpg_data[:,3],cmap='blue')


    ax.plot3D(time,cpg_data[:,4],cpg_data[:,5],'gray')
    ax.scatter3D(time,cpg_data[:,4],cpg_data[:,5],cmap='red')


    ax.plot3D(time,cpg_data[:,6],cpg_data[:,7],'gray')
    ax.scatter3D(time,cpg_data[:,6],cpg_data[:,7],'+',cmap='yellow')
    #if cpg_data[:,1]==
    #if cpg_data[:,1]==
    ax.set_xlabel('Time[s]',font_label)
    ax.set_ylabel(r'$O_{1}$',font_label)
    ax.set_zlabel(r'$O_{2}$',font_label)
    ax.grid(which='both',axis='x',color='k',linestyle=':')
    ax.grid(which='both',axis='y',color='k',linestyle=':')
    ax.grid(which='both',axis='z',color='k',linestyle=':')


    fig2=plt.figure()
    ax2=fig2.add_subplot(1,1,1)
    ax2.plot(cpg_data[:,0],cpg_data[:,1],'r')
    ax2.plot(cpg_data[:,2],cpg_data[:,3],'g')
    ax2.plot(cpg_data[:,4],cpg_data[:,5],'b')
    ax2.plot(cpg_data[:,6],cpg_data[:,7],'y')
    ax2.plot(time,cpg_data[:,0],'r')
    ax2.plot(time,cpg_data[:,2],'g')
    ax2.plot(time,cpg_data[:,4],'b')
    ax2.plot(time,cpg_data[:,6],'y')
    ax2.grid(which='both',axis='x',color='k',linestyle=':')
    ax2.grid(which='both',axis='y',color='k',linestyle=':')
    ax2.set_xlabel(r'$O_{1}$',font_label)
    ax2.set_ylabel(r'$O_{2}$',font_label)

    fig3=plt.figure()
    ax31=fig3.add_subplot(4,1,1)
    ax31.plot(time, grf_data[:,0],'r')
    ax32=fig3.add_subplot(4,1,2)
    ax32.plot(time, grf_data[:,1],'r')
    ax33=fig3.add_subplot(4,1,3)
    ax33.plot(time, grf_data[:,2],'r')
    ax34=fig3.add_subplot(4,1,4)
    ax34.plot(time, grf_data[:,3],'r')

    plt.show()

def Phase_Gait(data_file_dic,start_time=90,end_time=1200,freq=60.0,experiment_categories=['0.0'],trial_id=0):
    ''' 
    This is for experiment two, for the first figure with one trail on inclination
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_id, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''
    # 1) read data
    titles_files_categories, categories=load_data_log(data_file_dic)

    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    pose={}
    jmc={}
    cpg={}
    noise={}

    for category, files_name in titles_files_categories: #category is a files_name categorys
        gamma[category]=[]  #files_name is the table of the files_name category
        gait_diagram_data[category]=[]
        beta[category]=[]
        pose[category]=[]
        jmc[category]=[]
        cpg[category]=[]
        noise[category]=[]
        print(category)
        for idx in files_name.index:
            folder_category= data_file_dic + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
            # 2)  data process
            print(folder_category)
            gamma[category].append(COG_distribution(grf_data))

            gait_diagram_data_temp, beta_temp=calculate_gait(grf_data)
            gait_diagram_data[category].append(gait_diagram_data_temp); beta[category].append(beta_temp)

            pose[category].append(pose_data)
            jmc[category].append(command_data)
            cpg[category].append(cpg_data)
            noise[category].append(module_data)
            
            temp_1=min([len(bb) for bb in beta_temp]) #minimum steps of all legs
            beta_temp2=np.array([beta_temp[0][:temp_1],beta_temp[1][:temp_1],beta_temp[2][:temp_1],beta_temp[3][0:temp_1]]) # transfer to np array
            if(beta_temp2 !=[]):
                print("Coordination:",1.0/max(np.std(beta_temp2, axis=0)))
            else:
                print("Coordination:",0.0)

            print("Stability:",1.0/np.std(pose_data[:,0],axis=0))
            print("Displacemment:",np.sqrt(pow(pose_data[-1,3]-pose_data[0,3],2)+pow(pose_data[-1,5]-pose_data[0,5],2))) #Displacement on slopes 
    #3) plot
    figsize=(6,3.0)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(2,len(experiment_categories))#13
    gs1.update(hspace=0.12,top=0.95,bottom=0.2,left=0.12,right=0.98)
    axs=[]
    for idx in range(len(experiment_categories)):# how many columns, depends on the experiment_categories
        axs.append(fig.add_subplot(gs1[0:1,idx]))
        axs.append(fig.add_subplot(gs1[1:2,idx]))
    
    #3.1) plot 

    for idx, inclination in enumerate(experiment_categories):


        axs[0].set_ylabel(u'Phase diff. [rad]')
        phi=calculate_phase_diff(cpg[inclination][trial_id],time)
        axs[3*idx+0].plot(phi['time'],phi['phi_12'],color=(77/255,133/255,189/255))
        axs[3*idx+0].plot(phi['time'],phi['phi_13'],color=(247/255,144/255,61/255))
        axs[3*idx+0].plot(phi['time'],phi['phi_14'],color=(89/255,169/255,90/255))
        axs[3*idx+0].legend([u'$\phi_{12}$',u'$\phi_{13}$',u'$\phi_{14}$'])
        axs[3*idx+0].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+0].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+0].set_yticks([0.0,1.5,3.0])
        axs[3*idx+0].set_xticklabels([])
        axs[3*idx+0].set(xlim=[min(time),max(time)])

        axs[1].set_ylabel(r'Gait')
        gait_diagram(fig,axs[3*idx+1],gs1,gait_diagram_data[inclination][trial_id])
        axs[3*idx+1].set_xlabel(u'Time [s]')
        xticks=np.arange(int(min(time)),int(max(time))+1,2)
        axs[3*idx+1].set_xticks(xticks)
        axs[3*idx+1].set_xticklabels([str(xtick) for xtick in xticks])
        axs[1].yaxis.set_label_coords(-0.07,.5)
        axs[3*idx+1].set(xlim=[min(time),max(time)])

    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'experiment2_1.svg'
    plt.savefig(figPath)

    plt.show()

def Phase_Gait_ForNoiseFeedback(data_file_dic,start_time=90,end_time=1200,freq=60.0,experiment_categories=['0.0'],trial_id=0):
    ''' 
    This is for experiment two, for the first figure with one trail on inclination
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_id, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''
    # 1) read data
    titles_files_categories, categories=load_data_log(data_file_dic)

    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    pose={}
    jmc={}
    cpg={}
    noise={}

    for category, files_name in titles_files_categories[titles_files_categories['experiment_categories']==experiment_categories[0]]: #category is a files_name categorys
        gamma[category]=[]  #files_name is the table of the files_name category
        gait_diagram_data[category]=[]
        beta[category]=[]
        pose[category]=[]
        jmc[category]=[]
        cpg[category]=[]
        noise[category]=[]
        print(category)
        for idx in files_name.index:
            folder_category = data_file_dic + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
            # 2)  data process
            print(folder_category)
            gamma[category].append(COG_distribution(grf_data-module_data[:,1:]))

            gait_diagram_data_temp, beta_temp=gait(grf_data-module_data[:,1:])
            gait_diagram_data[category].append(gait_diagram_data_temp); beta[category].append(beta_temp)

            pose[category].append(pose_data)
            jmc[category].append(command_data)
            cpg[category].append(cpg_data)
            noise[category].append(module_data)
            
            temp_1=min([len(bb) for bb in beta_temp]) #minimum steps of all legs
            beta_temp2=np.array([beta_temp[0][:temp_1],beta_temp[1][:temp_1],beta_temp[2][:temp_1],beta_temp[3][0:temp_1]]) # transfer to np array
            if(beta_temp2 !=[]):
                print("Coordination:",1.0/max(np.std(beta_temp2, axis=0)))
            else:
                print("Coordination:",0.0)

            print("Stability:",1.0/np.std(pose_data[:,0],axis=0))
            print("Displacemment:",np.sqrt(pow(pose_data[-1,3]-pose_data[0,3],2)+pow(pose_data[-1,5]-pose_data[0,5],2))) #Displacement on slopes 
    #3) plot
    figsize=(6,3.0)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(2,len(experiment_categories))#13
    gs1.update(hspace=0.12,top=0.95,bottom=0.2,left=0.12,right=0.98)
    axs=[]
    for idx in range(len(experiment_categories)):# how many columns, depends on the experiment_categories
        axs.append(fig.add_subplot(gs1[0:1,idx]))
        axs.append(fig.add_subplot(gs1[1:2,idx]))
    
    #3.1) plot 

    for idx, inclination in enumerate(experiment_categories):
        axs[0].set_ylabel(u'Phase diff. [rad]')
        phi=calculate_phase_diff(cpg[inclination][trial_id],time)
        axs[3*idx+0].plot(phi['time'],phi['phi_12'],color=(77/255,133/255,189/255))
        axs[3*idx+0].plot(phi['time'],phi['phi_13'],color=(247/255,144/255,61/255))
        axs[3*idx+0].plot(phi['time'],phi['phi_14'],color=(89/255,169/255,90/255))
        axs[3*idx+0].legend([u'$\phi_{12}$',u'$\phi_{13}$',u'$\phi_{14}$'])
        axs[3*idx+0].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+0].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+0].set_yticks([0.0,1.5,3.0])
        axs[3*idx+0].set_xticklabels([])
        axs[3*idx+0].set(xlim=[min(time),max(time)])

        axs[1].set_ylabel(r'Gait')
        gait_diagram(fig,axs[3*idx+1],gs1,gait_diagram_data[inclination][trial_id])
        axs[3*idx+1].set_xlabel(u'Time [s]')
        xticks=np.arange(int(min(time)),int(max(time))+1,2)
        axs[3*idx+1].set_xticks(xticks)
        axs[3*idx+1].set_xticklabels([str(xtick) for xtick in xticks])
        axs[1].yaxis.set_label_coords(-0.07,.5)
        axs[3*idx+1].set(xlim=[min(time),max(time)])

    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'experiment2_1.svg'
    plt.savefig(figPath)

    plt.show()

def neural_preprocessing(data):
    '''
    This is use a recurrent neural network to preprocessing the data,
    it works like a filter
    '''
    new_data=[]
    w_i=20
    w_r=7.2
    bias=-6.0
    new_d_old =0.0
    for d in data:
        new_a=w_i*d+w_r*new_d_old + bias
        new_d=1.0/(1.0+math.exp(-new_a))
        new_data.append(new_d)
        new_d_old=new_d
    return np.array(new_data)

def forceForwardmodel(data):
    '''
    This is a forward model, it use a joint command (hip joint) to 
    map an expected ground reaction force
    '''
    new_data=[]
    alpha = 1.0
    gamma = 0.99
    d_old=data[0]
    out_old=0.0
    for d in data:
        if d < d_old:
            G=1.0
        else:
            G=0.0
        out= alpha*(gamma*G + (1-gamma)*out_old)
        d_old=d
        out_old=out
        new_data.append(out)

    return np.array(new_data)

def touch_difference(joint_cmd, actual_grf):
    '''
    Compare the diff between the expected grf and actual grf, where expected grf is calculated by the hip jointmovoemnt commands

    '''

    assert(joint_cmd.shape[1]==4)
    assert(actual_grf.shape[1]==4)

    new_expected_grf = np.zeros(joint_cmd.shape)
    new_actual_grf = np.zeros(actual_grf.shape)
    joint_cmd_middle_position=[]

    #
    for idx in range(4):
        joint_cmd_middle_position.append((np.amax(joint_cmd[:,idx])+np.amin(joint_cmd[:,idx]))/2.0)
        new_expected_grf[:,idx] = forceForwardmodel(joint_cmd[:,idx])
        new_actual_grf[:,idx] = neural_preprocessing(actual_grf[:,idx])
    # pre process again
    threshold=0.2
    new_expected_grf = new_expected_grf > threshold
    new_actual_grf = new_actual_grf > threshold
    #
    move_stage = joint_cmd > joint_cmd_middle_position

    # transfer the GRF into bit , to get stance and swing phase
    new_actual_grf=new_actual_grf.astype(np.int)
    new_expected_grf=new_expected_grf.astype(np.int)
    # calculate the difference between the expected and actual swing/stance
    diff = new_actual_grf-new_expected_grf

    Te1=(diff*move_stage==1)
    Te3=(diff*move_stage==-1)
    Te2=(diff*(~move_stage)==1)
    Te4=(diff*(~move_stage)==-1)
    return [new_expected_grf, new_actual_grf, diff, Te1, Te2, Te3, Te4]

def getTouchData(data_file_dic,start_time=900,end_time=1200,freq=60,experiment_categories=['0.0']):
    '''
    calculating touch momemnt
    '''
    # 1) read data
    titles_files_categories, categories=load_data_log(data_file_dic)
    Te={}
    Te1={}
    Te2={}
    Te3={}
    Te4={}
    for category, files_name in titles_files_categories: #category is a files_name categorys
        Te[category] =[]
        Te1[category]=[]  #inclination is the table of the inclination name
        Te2[category]=[]
        Te3[category]=[]
        Te4[category]=[]
        print(category)
        for idx in files_name.index:
            folder_category= data_file_dic + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
            # 2)  data process
            print(folder_category)
            Te_temp=touch_difference(cpg_data[:,[0,2,4,6]], grf_data)
            Te1[category].append(sum(Te_temp[3])/freq)
            Te2[category].append(sum(Te_temp[4])/freq)
            Te3[category].append(sum(Te_temp[5])/freq)
            Te4[category].append(sum(Te_temp[6])/freq)
            Te[category].append([Te_temp,command_data])

    return Te, Te1, Te2,Te3, Te4

def plot_comparasion_of_expected_actual_grf(Te, leg_id,ax=None):
    jmc=Te[1]
    Te=Te[0]
    axs=[]
    if ax==None:
        figsize=(7.1,6.1244)
        fig = plt.figure(figsize=figsize,constrained_layout=False)
        gs1=gridspec.GridSpec(6,1)#13
        gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.15,right=0.98)
        axs.append(fig.add_subplot(gs1[0:6,0]))
    else:
        axs.append(ax)

    time=np.linspace(0,len(Te[0])/60,len(Te[0]))
    axs[0].plot(time,Te[3][:,leg_id],'r*')
    axs[0].plot(time,Te[4][:,leg_id],'ro')
    axs[0].plot(time,Te[5][:,leg_id],'b*')
    axs[0].plot(time,Te[6][:,leg_id],'bo')
    #axs[0].plot(Te[2][:,leg_id],'k:')

    axs[0].plot(time,0.5*Te[0][:,leg_id],'b-.')
    axs[0].plot(time,0.5*Te[1][:,leg_id],'r-.')
    #axs[0].plot(time,0.5*jmc[:,3*leg_id+1],'g-.')
    axs[0].legend(['Te1','Te2', 'Te3', 'Te4','Expected GRF','Actual GRF'],ncol=3,loc='center',bbox_to_anchor=(0.5,0.8))
    axs[0].grid(which='both',axis='x',color='k',linestyle=':')
    axs[0].grid(which='both',axis='y',color='k',linestyle=':')
    axs[0].set_xlabel('Time [s]')
    ticks=np.arange(int(max(time)))
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels([str(tick) for tick in ticks])

def plot_comparasion_expected_actual_grf_all_leg(data_file_dic,start_time=600,end_time=1200,freq=60,experiment_categories=['0.0']):
    '''
    compare expected and actual ground reaction force

    '''
    Te, Te1, Te2,Te3,Te4 = getTouchData(data_file_dic,start_time,end_time,freq,experiment_categories)

    figsize=(9.1,11.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(8,len(experiment_categories))#13
    gs1.update(hspace=0.24,top=0.95,bottom=0.07,left=0.1,right=0.98)
    axs=[]
    for idx in range(len(experiment_categories)):# how many columns, depends on the experiment_categories
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))
        axs.append(fig.add_subplot(gs1[4:6,idx]))
        axs.append(fig.add_subplot(gs1[6:8,idx]))

    for idx,inclination in enumerate(experiment_categories):
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=0,ax=axs[4*idx])
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=1,ax=axs[4*idx+1])
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=2,ax=axs[4*idx+2])
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=3,ax=axs[4*idx+3])
    plt.show()


def plot_phase_diff(data_file_dic,start_time=90,end_time=1200,freq=60.0,experiment_categories=['0.0'],trial_id=0):
    ''' 
    This is for plot the phase diff by a plot curve
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_id, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''
    # 1) read data
    titles_files_categories, categories=load_data_log(data_file_dic)

    cpg={}

    for category, files_name in titles_files_categories: #category is a files_name categorys
        cpg[category]=[]  #files_name is the table of the files_name category
        if category in experiment_categories:
            print(category)
            for idx in files_name.index:
                folder_category= data_file_dic + files_name['data_files'][idx]
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
        # 2)  data process
                print(folder_category)
                cpg[category].append(cpg_data)


    # 2) calculate the phase angles
    phi=pd.DataFrame(columns=['time','phi_12','phi_13','phi_14'])
    for idx in range(len(cpg[experiment_categories[0]])):
        trial_id=idx
        phi=pd.concat([phi,calculate_phase_diff(cpg[experiment_categories[0]][trial_id],time)])

    #3) plot
    figsize=(5.5,7.)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(9,len(experiment_categories))#13
    gs1.update(hspace=0.13,top=0.95,bottom=0.1,left=0.11,right=0.98)
    axs=[]
    for idx in range(len(experiment_categories)):# how many columns, depends on the experiment_categories
        axs.append(fig.add_subplot(gs1[0:9,idx]))


    sns.lineplot(x='time',y='phi_12',data=phi,ax=axs[0])
    sns.lineplot(x='time',y='phi_13',data=phi,ax=axs[0])
    sns.lineplot(x='time',y='phi_14',data=phi,ax=axs[0])
    axs[0].legend(['$\phi_{12}$','$\phi_{13}$','$\phi_{14}$'])

    plt.show()

def plot_actual_grf_all_leg(data_file_dic,start_time=600,end_time=1200,freq=60,experiment_categories=['0'], trial_id=0):
    '''
    Comparing all legs' actual ground reaction forces

    '''

    # 1) read data
    titles_files_categories, categories=load_data_log(data_file_dic)
    grf={}
    for category, files_name in titles_files_categories: #category is a files_name categorys
        if category in experiment_categories:
            print(category)
            grf[category]=[]
            for idx in files_name.index:
                folder_category= data_file_dic + files_name['data_files'][idx]
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                # 2)  data process
                grf[category].append(grf_data)
                print(folder_category)
    
    # 2) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(8,len(experiment_categories))#13
    gs1.update(hspace=0.24,top=0.95,bottom=0.07,left=0.1,right=0.99)
    axs=[]
    for idx in range(len(experiment_categories)):# how many columns, depends on the experiment_categories
        axs.append(fig.add_subplot(gs1[0:8,idx]))

    for idx,inclination in enumerate(experiment_categories):
        axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
        grf_diagram(fig,axs[idx],gs1, grf[inclination][idx],time)
    plt.show()

def GeneralDisplay(data_file_dic,start_time=90,end_time=1200,freq=60.0,experiment_categories=['0.0'],trial_id=0):
    ''' 
    This is for experiment two, for the first figure with one trail on inclination
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_id, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''
    # 1) read data
    titles_files_categories, categories=load_data_log(data_file_dic)

    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    pose={}
    jmc={}
    jmp={}
    jmv={}
    jmf={}
    cpg={}
    noise={}

    for category, files_name in titles_files_categories: #category is a files_name categorys
        gamma[category]=[]  #files_name is the table of the files_name category
        gait_diagram_data[category]=[]
        beta[category]=[]
        pose[category]=[]
        jmc[category]=[]
        jmp[category]=[]
        jmv[category]=[]
        jmf[category]=[]
        cpg[category]=[]
        noise[category]=[]
        if category in experiment_categories:
            print(category)
            for idx in files_name.index:
                folder_category= data_file_dic + files_name['data_files'][idx]
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                # 2)  data process
                print(folder_category)
                gamma[category].append(COG_distribution(grf_data))
                gait_diagram_data_temp, beta_temp = gait(grf_data)
                gait_diagram_data[category].append(gait_diagram_data_temp); beta[category].append(beta_temp)
                pose[category].append(pose_data)
                jmc[category].append(command_data)
                jmp[category].append(position_data)
                velocity_data=calculate_joint_velocity_data(position_data,freq)
                jmv[category].append(velocity_data)
                jmf[category].append(current_data)
                cpg[category].append(cpg_data)
                noise[category].append(module_data)
                temp_1=min([len(bb) for bb in beta_temp]) #minimum steps of all legs
                beta_temp2=np.array([beta_temp[0][:temp_1],beta_temp[1][:temp_1],beta_temp[2][:temp_1],beta_temp[3][0:temp_1]]) # transfer to np array
                if(beta_temp2 !=[]):
                    print("Coordination:",1.0/max(np.std(beta_temp2, axis=0)))
                else:
                    print("Coordination:",0.0)

                print("Stability:", 1.0/np.std(pose_data[:,0],axis=0) + 1.0/np.std(pose_data[:,1], axis=0))
                print("Displacemment:",calculate_displacement(pose_data)) #Displacement on slopes 
                print("Energy cost:", calculate_energy_cost(velocity_data,current_data,freq))

            
    #3) plot
    figsize=(8.2,6.5)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,len(experiment_categories))#13
    gs1.update(hspace=0.12,top=0.95,bottom=0.09,left=0.12,right=0.98)
    axs=[]
    for idx in range(len(experiment_categories)):# how many columns, depends on the experiment_categories
        axs.append(fig.add_subplot(gs1[0:1,idx]))
        axs.append(fig.add_subplot(gs1[1:2,idx]))
        axs.append(fig.add_subplot(gs1[2:3,idx]))
        axs.append(fig.add_subplot(gs1[3:4,idx]))
        axs.append(fig.add_subplot(gs1[4:5,idx]))
        axs.append(fig.add_subplot(gs1[5:6,idx]))
    
    #3.1) plot 
    situations={'0':'Normal', '1':'Noisy feedback', '2':'Malfunction leg', '3':'Carrying load','0.9':'0.9'}
    
    experiment_category=experiment_categories[0]# The first category of the input parameters (arg)
    idx=0
    axs[idx].plot(time,cpg[experiment_category][trial_id][:,1], color=(46/255.0, 77/255.0, 129/255.0))
    axs[idx].plot(time,cpg[experiment_category][trial_id][:,3], color=(0/255.0, 198/255.0, 156/255.0))
    axs[idx].plot(time,cpg[experiment_category][trial_id][:,5], color=(255/255.0, 1/255.0, 118/255.0))
    axs[idx].plot(time,cpg[experiment_category][trial_id][:,7], color=(225/255.0, 213/255.0, 98/255.0))
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_ylabel(u'CPGs')
    axs[idx].set_yticks([-1.0,0.0,1.0])
    axs[idx].legend(['RF','RH','LF', 'LH'],ncol=4)
    axs[idx].set_xticklabels([])
    axs[idx].set_title(data_file_dic[79:-1]+": " + situations[experiment_categories[0]] +" "+str(trial_id))
    axs[idx].set(xlim=[min(time),max(time)])

    idx=1
    axs[idx].set_ylabel(u'Atti. [deg]')
    axs[idx].plot(time,pose[experiment_category][trial_id][:,0]*-57.3,color=(129/255.0,184/255.0,223/255.0))
    axs[idx].plot(time,pose[experiment_category][trial_id][:,1]*-57.3,color=(254/255.0,129/255.0,125/255.0))
    axs[idx].plot(time,pose[experiment_category][trial_id][:,2]*-57.3,color=(86/255.0,169/255.0,90/255.0))
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_yticks([-5.0,0.0,5.0])
    axs[idx].legend(['Roll','Pitch','Yaw'],loc='upper left')
    axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])


    idx=2
    axs[idx].set_ylabel(u'Disp. [m]')
    displacement_x = pose[experiment_category][trial_id][:,3] #Displacement on slopes 
    displacement_y = pose[experiment_category][trial_id][:,4] #Displacement on slopes 
    axs[idx].plot(time,displacement_x,'r')
    axs[idx].plot(time,displacement_y,'b')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].legend(['X-axis','Y-axis'],loc='upper left')
    axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])

    idx=3
    axs[idx].set_ylabel(u'Phase diff. [rad]')
    phi=calculate_phase_diff(cpg[experiment_category][trial_id],time)
    axs[idx].plot(phi['time'],phi['phi_12'],color=(77/255,133/255,189/255))
    axs[idx].plot(phi['time'],phi['phi_13'],color=(247/255,144/255,61/255))
    axs[idx].plot(phi['time'],phi['phi_14'],color=(89/255,169/255,90/255))
    axs[idx].legend([u'$\phi_{12}$',u'$\phi_{13}$',u'$\phi_{14}$'])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_yticks([0.0,1.5,3.0])
    axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])
#
#
#    idx=4
#    axs[idx].set_ylabel(u'Noise [rad]')
#    axs[idx].plot(time,noise[experiment_category][trial_id][:,1], 'r')
#    axs[idx].plot(time,noise[experiment_category][trial_id][:,2], 'g')
#    axs[idx].plot(time,noise[experiment_category][trial_id][:,3], 'y')
#    axs[idx].plot(time,noise[experiment_category][trial_id][:,4], 'y')
#    axs[idx].legend(['N1','N2','N3','N4'])
#    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
#    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
#    axs[idx].set_yticks([-0.3, 0.0, 0.3])
#    axs[idx].set_xticklabels([])
#    axs[idx].set(xlim=[min(time),max(time)])
#

    idx=4
    axs[idx].set_ylabel(u'Joint poistion and 1000*velocity [rad]')
    axs[idx].plot(time,jmf[experiment_category][trial_id][:,1], 'r')
    axs[idx].plot(time,jmv[experiment_category][trial_id][:,1]*1000, 'g')
    axs[idx].legend(['N1','N2'])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    #axs[idx].set_yticks([-0.3, 0.0, 0.3])
    axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])

    idx=5
    axs[idx].set_ylabel(r'Gait')
    gait_diagram(fig,axs[idx],gs1,gait_diagram_data[experiment_category][trial_id])
    axs[idx].set_xlabel(u'Time [s]')
    xticks=np.arange(int(min(time)),int(max(time))+1,2)
    axs[idx].set_xticklabels([str(xtick) for xtick in xticks])
    axs[idx].set_xticks(xticks)
    axs[idx].yaxis.set_label_coords(-0.09,.5)
    axs[idx].set(xlim=[min(time),max(time)])

    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'general_display.svg'
    plt.savefig(figPath)

    plt.show()

def plot_runningSuccess_statistic(data_file_dic,start_time=10,end_time=400,freq=60,experiment_categories=['0.0']):
    '''
    Statistical of running success
    

    '''
    # 1) read data
    #1.1) read  local data of phase modulation
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)
    pose_phaseModulation={}
    for category, files_name in titles_files_categories: #name is a inclination names
        pose_phaseModulation[category] =[]
        print(category)
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
            print(folder_name)
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            stability_temp= 1.0/np.std(pose_data[:,0],axis=0) + 1.0/np.std(pose_data[:,1],axis=0) #+ 1.0/np.std(pose_data[:,2]) +1.0/np.std(pose_data[:,5])
            pose_phaseModulation[category].append(stability_temp)
            print(pose_phaseModulation[category][-1])

    #1.2) read local data of phase reset
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories, categories =load_data_log(data_file_dic_phaseReset)
    pose_phaseReset={}
    for category, files_name in titles_files_categories: #name is a inclination names
        pose_phaseReset[category] =[]
        print(category)
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
            print(folder_name)
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            stability_temp= 1.0/np.std(pose_data[:,0],axis=0) + 1.0/np.std(pose_data[:,1],axis=0) #+ 1.0/np.std(pose_data[:,2]) + 1.0/np.std(pose_data[:,5])
            pose_phaseReset[category].append(stability_temp)
            print(pose_phaseReset[category][-1])

            
    #2) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))
    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15

    #3.1) plot 

    idx=0
    axs[idx].bar(ind-0.5*width,[len(pose_phaseModulation[ll]) for ll in labels], width,label=r'Phase modulation')
    axs[idx].bar(ind+0.5*width,[len(pose_phaseReset[ll]) for ll in labels],width,label=r'Phase reset')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    axs[idx].set(ylim=[0,10])
    axs[idx].legend(loc='center right')
    axs[idx].set_ylabel(r'Success [count]')
    axs[idx].set_xlabel(r'Situations')
    axs[idx].set_xticklabels(situations)

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'runningSuccess.svg'
    plt.savefig(figPath)
    plt.show()

def percentage_plot_runningSuccess_statistic(data_file_dic,start_time=10,end_time=400,freq=60,experiment_categories=['0.0']):
    '''
    Statistical of running success
    

    '''
    # 1) read data
    #1.1) read  local data of phase modulation
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)
    pose_phaseModulation={}
    for category, files_name in titles_files_categories: #name is a inclination names
        pose_phaseModulation[category] =[]
        print(category)
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
            print(folder_name)
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            stability_temp= 1.0/np.std(pose_data[:,0],axis=0) + 1.0/np.std(pose_data[:,1],axis=0) #+ 1.0/np.std(pose_data[:,2]) +1.0/np.std(pose_data[:,5])
            pose_phaseModulation[category].append(stability_temp)
            print(pose_phaseModulation[category][-1])

    #1.2) read local data of phase reset
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories, categories =load_data_log(data_file_dic_phaseReset)
    pose_phaseReset={}
    for category, files_name in titles_files_categories: #name is a inclination names
        pose_phaseReset[category] =[]
        print(category)
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
            print(folder_name)
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            stability_temp= 1.0/np.std(pose_data[:,0],axis=0) + 1.0/np.std(pose_data[:,1],axis=0) #+ 1.0/np.std(pose_data[:,2]) + 1.0/np.std(pose_data[:,5])
            pose_phaseReset[category].append(stability_temp)
            print(pose_phaseReset[category][-1])

            
    #2) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']
    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15

    #3.1) plot 

    idx=0
    colors = ['lightblue', 'lightgreen']
    axs[idx].bar(ind-0.5*width,[len(pose_phaseModulation[ll])/10.0*100 for ll in labels], width,label=r'Phase modulation', color=colors[0])
    axs[idx].bar(ind+0.5*width,[len(pose_phaseReset[ll])/10.0*100 for ll in labels],width,label=r'Phase reset',color=colors[1])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[0,10])
    axs[idx].legend(loc='center right')
    axs[idx].set_ylabel(r'Success rate[%]')
    axs[idx].set_xlabel(r'Situations')
    axs[idx].set_xticklabels(situations)

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'runningSuccess.svg'
    plt.savefig(figPath)
    plt.show()

def plot_stability_statistic(data_file_dic,start_time=10,end_time=400,freq=60,experiment_categories=['0.0']):
    '''
    Stability of statistic

    '''
    # 1) read data
    #1.1) read loacal data of phase modulation
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)
    pose_phaseModulation={}
    for category, files_name in titles_files_categories: #name is a inclination names
        if category in categories:
            print(category)
            pose_phaseModulation[category] =[]
            for idx in files_name.index: # how many time experiments one inclination
                folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
                print(folder_name)
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
                # 2)  data process
                stability_temp= 1.0/np.std(pose_data[:,0],axis=0) + 1.0/np.std(pose_data[:,1],axis=0) #+ 1.0/np.std(pose_data[:,2]) +1.0/np.std(pose_data[:,5])
                pose_phaseModulation[category].append(stability_temp)
                print(pose_phaseModulation[category][-1])

    #1.2) read loacal data of phase reset
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories, categories =load_data_log(data_file_dic_phaseReset)
    pose_phaseReset={}
    for category, files_name in titles_files_categories: #name is a inclination names
        if category in categories:
            print(category)
            pose_phaseReset[category] =[]
            for idx in files_name.index: # how many time experiments one inclination
                folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
                print(folder_name)
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
                # 2)  data process
                stability_temp= 1.0/np.std(pose_data[:,0],axis=0) + 1.0/np.std(pose_data[:,1],axis=0) #+ 1.0/np.std(pose_data[:,2]) + 1.0/np.std(pose_data[:,5])
                pose_phaseReset[category].append(stability_temp)
                print(pose_phaseReset[category][-1])


    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15
    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']

    #3.1) plot 
    angular_phaseReset_mean, angular_phaseReset_std=[],[]
    angular_phaseModulation_mean, angular_phaseModulation_std=[],[]
    for i in labels: #inclination
        angular_phaseReset_mean.append(np.mean(pose_phaseReset[i]))
        angular_phaseReset_std.append(np.std(pose_phaseReset[i]))

        angular_phaseModulation_mean.append(np.mean(pose_phaseModulation[i]))
        angular_phaseModulation_std.append(np.std(pose_phaseModulation[i]))

    idx=0
    axs[idx].bar(ind-0.5*width,angular_phaseModulation_mean, width, yerr=angular_phaseModulation_std,label=r'Phase modulation')
    axs[idx].bar(ind+0.5*width,angular_phaseReset_mean,width,yerr=angular_phaseReset_std,label=r'Phase reset')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels(situations)
    axs[idx].legend()
    axs[idx].set_ylabel(r'Stability')
    axs[idx].set_xlabel(r'Situations')

    # save plot
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'stabilityStatistic.svg'
    plt.savefig(figPath)
    plt.show()

def plot_coordination_statistic(data_file_dic,start_time=60,end_time=900,freq=60.0,experiment_categories=['0.0']):
    '''
    @description: this is for experiment two, plot coordination statistic
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @return: show and save a data figure.

    '''
    #1) load data from file
    
    #1.1) local data of phase modualtion
    data_file_dic_phaseModulation = data_file_dic + "PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)

    coordination_phaseModulation={}

    for category, files_name in titles_files_categories: #name is a inclination names
        coordination_phaseModulation[category]=[]  #inclination is the table of the inclination name
        print(category)
        for idx in files_name.index:
            folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            print(folder_name)
            gait_diagram_data, beta=gait(grf_data)
            temp_1=min([len(bb) for bb in beta]) #minimum steps of all legs
            beta=np.array([beta[0][:temp_1],beta[1][:temp_1],beta[2][:temp_1],beta[3][0:temp_1]]) # transfer to np array
            
            if(beta !=[]):
                coordination_phaseModulation[category].append(1.0/max(np.std(beta, axis=0)))# 
            else:
                coordination_phaseModulation[category].append(0.0)

            print(coordination_phaseModulation[category][-1])
    
    #1.2) local data of phase reset
    data_file_dic_phaseReset = data_file_dic + "PhaseReset/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseReset)
    
    coordination_phaseReset={}

    for category, files_name in titles_files_categories: #name is a inclination names
        coordination_phaseReset[category]=[]  #inclination is the table of the inclination name
        print(category)
        for idx in files_name.index:
            folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            print(folder_name)
            gait_diagram_data, beta=gait(grf_data)
            temp_1=min([len(bb) for bb in beta]) #minimum steps of all legs
            beta=np.array([beta[0][:temp_1],beta[1][:temp_1],beta[2][:temp_1],beta[3][0:temp_1]]) # transfer to np array
            if(beta !=[]):
                coordination_phaseReset[category].append(1.0/max(np.std(beta, axis=0)))# 
            else:
                coordination_phaseReset[category].append(0.0)
            print(coordination_phaseReset[category][-1])


    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15
    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']

    #3.1) plot 
    coordinationPhaseModulation_mean,coordinationPhaseModulation_std=[],[]
    coordinationPhaseReset_mean,coordinationPhaseReset_std=[],[]
    for i in labels:
        coordinationPhaseModulation_mean.append(np.mean(coordination_phaseModulation[i]))
        coordinationPhaseModulation_std.append(np.std(coordination_phaseModulation[i]))
        coordinationPhaseReset_mean.append(np.mean(coordination_phaseReset[i]))
        coordinationPhaseReset_std.append(np.std(coordination_phaseReset[i]))

    idx=0
    axs[idx].bar(ind-0.5*width,coordinationPhaseModulation_mean, width, yerr=coordinationPhaseModulation_std,label=r'Phase modulation')
    axs[idx].bar(ind+0.5*width,coordinationPhaseReset_mean,width,yerr=coordinationPhaseReset_std,label=r'Phase reset')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels(situations)
    axs[idx].legend()
    axs[idx].set_ylabel(r'Coordination')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coordinationStatistic.svg'
    plt.savefig(figPath)
    plt.show()

def plot_COT_statistic(data_file_dic,start_time=60,end_time=900,freq=60.0,experiment_categories=['0.0']):
    '''
    @description: this is for comparative investigation, plot cost of transport statistic
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @return: show and save a data figure.

    '''
    #1) load data from file
    
    #1.1) local COG reflex data
    data_file_dic_phaseModulation = data_file_dic + "PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)

    COT_phaseModulation={}

    for category, files_name in titles_files_categories: #name is a inclination names
        COT_phaseModulation[category]=[]  #inclination is the table of the inclination name
        print(category)
        for idx in files_name.index:
            folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            print(folder_name)
            velocity_data=(position_data[1:,:]-position_data[0:-1,:])*freq
            initial_velocity=[0,0,0, 0,0,0, 0,0,0, 0,0,0]
            velocity_data=np.vstack([initial_velocity,velocity_data])
            d=np.sqrt(pow(pose_data[-1,3]-pose_data[0,3],2)+pow(pose_data[-1,4]-pose_data[0,4],2)) #Displacement
            COT=calculate_COT(velocity_data,current_data,freq,d)
            COT_phaseModulation[category].append(COT)# 
            print(COT_phaseModulation[category][-1])
    
    #1.2) local vestibular reflex data
    data_file_dic_phaseReset = data_file_dic + "PhaseReset/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseReset)
    
    COT_phaseReset={}

    for category, files_name in titles_files_categories: #name is a inclination names
        COT_phaseReset[category]=[]  #inclination is the table of the inclination name
        print(category)
        for idx in files_name.index:
            folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            print(folder_name)
            velocity_data=(position_data[1:,:]-position_data[0:-1,:])*freq
            initial_velocity=[0,0,0, 0,0,0, 0,0,0, 0,0,0]
            velocity_data=np.vstack([initial_velocity,velocity_data])
            d=calculate_displacement(pose_data)
            COT=calculate_COT(velocity_data,current_data,freq,d)
            COT_phaseReset[category].append(COT)# 
            print(COT_phaseReset[category][-1])

            
    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15

    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']
    #3.1) plot 
    COTPhaseModulation_mean,COTPhaseModulation_std=[],[]
    COTPhaseReset_mean,COTPhaseReset_std=[],[]
    for i in labels:
        COTPhaseModulation_mean.append(np.mean(COT_phaseModulation[i]))
        COTPhaseModulation_std.append(np.std(COT_phaseModulation[i]))
        COTPhaseReset_mean.append(np.mean(COT_phaseReset[i]))
        COTPhaseReset_std.append(np.std(COT_phaseReset[i]))

    idx=0
    axs[idx].bar(ind-0.5*width,COTPhaseModulation_mean, width, yerr=COTPhaseModulation_std,label=r'Phase modulation')
    axs[idx].bar(ind+0.5*width,COTPhaseReset_mean,width,yerr=COTPhaseReset_std,label=r'Phase reset')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels(situations)
    axs[idx].legend()
    axs[idx].set_ylabel(r'COT')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'COTStatistic.svg'
    plt.savefig(figPath)
    plt.show()

def plot_energyCost_statistic(data_file_dic,start_time=60,end_time=900,freq=60.0,experiment_categories=['0.0']):
    '''
    @description: this is for compative investigation to plot energy cost statistic
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @return: show and save a data figure.

    '''
    #1) load data from file
    
    #1.1) local data of phase modulation
    data_file_dic_phaseModulation = data_file_dic + "PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)
    energy_phaseModulation={}
    for category, files_name in titles_files_categories: #name is a inclination names
        energy_phaseModulation[category]=[]  #inclination is the table of the inclination name
        print(category)
        for idx in files_name.index:
            folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            print(folder_name)
            velocity_data=(position_data[1:,:]-position_data[0:-1,:])*freq
            initial_velocity=[0,0,0, 0,0,0, 0,0,0, 0,0,0]
            velocity_data=np.vstack([initial_velocity,velocity_data])
            energy=calculate_energy_cost(velocity_data,current_data,freq)
            energy_phaseModulation[category].append(energy)# 
            print(energy_phaseModulation[category][-1])
    
    #1.2) local data of phase reset
    data_file_dic_phaseReset = data_file_dic + "PhaseReset/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseReset)
    energy_phaseReset={}
    for category, files_name in titles_files_categories: #name is a inclination names
        energy_phaseReset[category]=[]  #inclination is the table of the inclination name
        print(category)
        for idx in files_name.index:
            folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            print(folder_name)
            velocity_data=(position_data[1:,:]-position_data[0:-1,:])*freq
            initial_velocity=[0,0,0, 0,0,0, 0,0,0, 0,0,0]
            velocity_data=np.vstack([initial_velocity,velocity_data])
            energy=calculate_energy_cost(velocity_data,current_data,freq)
            energy_phaseReset[category].append(energy)# 
            print(energy_phaseReset[category][-1])


    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15
    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']

    #3.1) plot 
    energyPhaseModulation_mean,energyPhaseModulation_std=[],[]
    energyPhaseReset_mean,energyPhaseReset_std=[],[]
    for i in labels:
        energyPhaseModulation_mean.append(np.mean(energy_phaseModulation[i]))
        energyPhaseModulation_std.append(np.std(energy_phaseModulation[i]))
        energyPhaseReset_mean.append(np.mean(energy_phaseReset[i]))
        energyPhaseReset_std.append(np.std(energy_phaseReset[i]))

    idx=0
    axs[idx].bar(ind-0.5*width,energyPhaseModulation_mean, width, yerr=energyPhaseModulation_std,label=r'Phase modulation')
    axs[idx].bar(ind+0.5*width,energyPhaseReset_mean,width,yerr=energyPhaseReset_std,label=r'Phase reset')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels(situations)
    axs[idx].legend()
    axs[idx].set_ylabel(r'Energy')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'energyStatistic.svg'
    plt.savefig(figPath)
    plt.show()

def plot_slipping_statistic(data_file_dic,start_time=60,end_time=900,freq=60.0,experiment_categories=['0.0']):
    '''
    @description: this is for experiment two, plot slipping statistic
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @return: show and save a data figure.

    '''
    #1) load data from file
    
    #1.1) local COG reflex data
    data_file_dic_phaseModulation = data_file_dic + "PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)
    COT_phaseModulation={}
    for category, files_name in titles_files_categories: #name is a inclination names
        COT_phaseModulation[category]=[]  #inclination is the table of the inclination name
        print(category)
        for idx in files_name.index:
            folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            print(folder_name)
            expected_actual_motion_states=touch_difference(cpg_data[:,[0,2,4,6]], grf_data)
            expected_actual_motion_states_equal = expected_actual_motion_states[0]==expected_actual_motion_states[1]

            COT_phaseModulation[category].append(COT)# 
            print(COT_phaseModulation[category][-1])
    
    #1.2) local vestibular reflex data
    data_file_dic_phaseReset = data_file_dic + "PhaseReset/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseReset)
    
    COT_phaseReset={}

    for category, files_name in titles_files_categories: #name is a inclination names
        COT_phaseReset[category]=[]  #inclination is the table of the inclination name
        print(category)
        for idx in files_name.index:
            folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            print(folder_name)
            expected_actual_motion_states=touch_difference(cpg_data[:,[0,2,4,6]], grf_data)
            expected_actual_motion_states_equal = expected_actual_motion_states[0]==expected_actual_motion_states[1]

            COT_phaseReset[category].append(COT)# 
            print(COT_phaseReset[category][-1])



    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15
    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']

    #3.1) plot 
    COTPhaseModulation_mean,COTPhaseModulation_std=[],[]
    COTPhaseReset_mean,COTPhaseReset_std=[],[]
    for i in labels:
        COTPhaseModulation_mean.append(np.mean(COT_phaseModulation[i]))
        COTPhaseModulation_std.append(np.std(COT_phaseModulation[i]))
        COTPhaseReset_mean.append(np.mean(COT_phaseReset[i]))
        COTPhaseReset_std.append(np.std(COT_phaseReset[i]))

    idx=0
    axs[idx].bar(ind-0.5*width,COTPhaseModulation_mean, width, yerr=COTPhaseModulation_std,label=r'Phase modulation')
    axs[idx].bar(ind+0.5*width,COTPhaseReset_mean,width,yerr=COTPhaseReset_std,label=r'Phase reset')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels(situations)
    axs[idx].legend()
    axs[idx].set_ylabel(r'COT')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'COTStatistic.svg'
    plt.savefig(figPath)
    plt.show()

def plot_distance_statistic(data_file_dic,start_time=10,end_time=400,freq=60,experiment_categories=['0.0']):
    '''
    Plot distance statistic, it calculates all movement, oscillation of the body during locomotion, it cannot express the transporyability of the locomotion
    

    '''
    # 1) read data
    # 1.1) read Phase reset data
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)

    pose_phaseModulation={}

    for category, files_name in titles_files_categories: #name is a inclination names
        pose_phaseModulation[category] =[]
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
            print(folder_name)
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            d=0
            for step_index in range(pose_data.shape[0]-1):
                d+=np.sqrt(pow(pose_data[step_index+1,3]-pose_data[step_index,3],2)+pow(pose_data[step_index+1,4]-pose_data[step_index,4],2))
            pose_phaseModulation[category].append(d) #Displacement on slopes 
            print(pose_phaseModulation[category][-1])

    #1.2) read continuous phase modulation data
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseReset)

    pose_phaseReset={}
    for category, files_name in titles_files_categories: #name is a inclination names
        pose_phaseReset[category] =[]
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
            print(folder_name)
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            d=0
            for step_index in range(pose_data.shape[0]-1):
                d+=np.sqrt(pow(pose_data[step_index+1,3]-pose_data[step_index,3],2)+pow(pose_data[step_index+1,4]-pose_data[step_index,4],2))
            pose_phaseReset[category].append(d) #Displacement on slopes 
            print(pose_phaseReset[category][-1])


    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15
    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']

    #3.1) plot 
    disp_phaseReset_mean, disp_phaseReset_std=[],[]
    disp_phaseModulation_mean, disp_phaseModulation_std=[],[]
    for i in labels: #inclination
        disp_phaseReset_mean.append(np.mean(pose_phaseReset[i]))
        disp_phaseReset_std.append(np.std(pose_phaseReset[i]))

        disp_phaseModulation_mean.append(np.mean(pose_phaseModulation[i]))
        disp_phaseModulation_std.append(np.std(pose_phaseModulation[i]))

    idx=0
    axs[idx].bar(ind-0.5*width,disp_phaseModulation_mean, width, yerr=disp_phaseModulation_std,label=r'Phase modulation')
    axs[idx].bar(ind+0.5*width,disp_phaseReset_mean,width,yerr=disp_phaseReset_std,label=r'Phase reset')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels(situations)
    axs[idx].legend()
    axs[idx].set_ylabel(r'Distance [m]')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Phase reset')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'distance.svg'
    plt.savefig(figPath)
    plt.show()

def plot_displacement_statistic(data_file_dic,start_time=10,end_time=400,freq=60,experiment_categories=['0.0']):
    '''
    plot displacement statistic, it can indicates the actual traverability of the locomotion
    

    '''
    # 1) read data
    # 1.1) read Phase reset data
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)

    pose_phaseModulation={}
    for category, files_name in titles_files_categories: #name is a inclination names
        pose_phaseModulation[category] =[]
        print(category)
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
            print(folder_name)
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            disp=np.sqrt(pow(pose_data[-1,3]-pose_data[0,3],2)+pow(pose_data[-1,4]-pose_data[0,4],2))#Displacement
            pose_phaseModulation[category].append(disp) #Displacement on slopes 
            print(pose_phaseModulation[category][-1])
            
    #1.2) read continuous phase modulation data
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseReset)

    pose_phaseReset={}
    for category, files_name in titles_files_categories: #name is a inclination names
        pose_phaseReset[category] =[]
        print(category)
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
            print(folder_name)
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            disp=np.sqrt(pow(pose_data[-1,3]-pose_data[0,3],2)+pow(pose_data[-1,4]-pose_data[0,4],2))#Displacement
            pose_phaseReset[category].append(disp) #Displacement on slopes 
            print(pose_phaseReset[category][-1])


    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15
    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']

    #3.1) plot 
    disp_phaseReset_mean, disp_phaseReset_std=[],[]
    disp_phaseModulation_mean, disp_phaseModulation_std=[],[]
    for i in labels: #inclination
        disp_phaseReset_mean.append(np.mean(pose_phaseReset[i]))
        disp_phaseReset_std.append(np.std(pose_phaseReset[i]))

        disp_phaseModulation_mean.append(np.mean(pose_phaseModulation[i]))
        disp_phaseModulation_std.append(np.std(pose_phaseModulation[i]))

    idx=0
    axs[idx].bar(ind-0.5*width,disp_phaseModulation_mean, width, yerr=disp_phaseModulation_std,label=r'Phase modulation')
    axs[idx].bar(ind+0.5*width,disp_phaseReset_mean,width,yerr=disp_phaseReset_std,label=r'Phase reset')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels(situations)
    axs[idx].legend()
    axs[idx].set_ylabel(r'Displacement [m]')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Phase reset')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'displacement.svg'
    plt.savefig(figPath)
    plt.show()


def phase_formTime_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0']):
    '''
    Plot convergence time statistic, it can indicates the actual traverability of the locomotion

    '''
    # 1) read data
    # 1.1) read Phase reset data
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)
    phi_phaseModulation={}
    for category, files_name in titles_files_categories: #name is a inclination names
        if category in categories:
            print(category)
            phi_phaseModulation[category] =[]
            for idx in files_name.index: # how many time experiments one inclination
                folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
                print(folder_name)
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
                # 2)  data process
                touch, covergence_idx, phi_std = touch_convergence_moment_identification(grf_data,cpg_data,time)
                convergenTime=convergence_idx/freq
                phi_phaseModulation[category].append(convergenTime) #Displacement on slopes 
                print(phi_phaseModulation[category][-1])
                print("Convergence idx", convergence_idx)

    #1.2) read continuous phase modulation data
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseReset)
    phi_phaseReset={}
    for category, files_name in titles_files_categories: #name is a inclination names
        if category in categories:
            print(category)
            phi_phaseReset[category] =[]
            for idx in files_name.index: # how many time experiments one inclination
                folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
                print(folder_name)
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
                # 2)  data process
                touch, covergence_idx, phi_std = touch_convergence_moment_identification(grf_data,cpg_data,time)
                convergenTime=convergence_idx/freq
                phi_phaseModulation[category].append(convergenTime) #Displacement on slopes 
                print(phi_phaseModulation[category][-1])
                print("Convergence idx", convergence_idx)

    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15
    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']

    #3.1) plot 
    phi_phaseReset_mean, phi_phaseReset_std=[],[]
    phi_phaseModulation_mean, phi_phaseModulation_std=[],[]
    for i in labels: #inclination
        phi_phaseReset_mean.append(np.mean(phi_phaseReset[i]))
        phi_phaseReset_std.append(np.std(phi_phaseReset[i]))

        phi_phaseModulation_mean.append(np.mean(phi_phaseModulation[i]))
        phi_phaseModulation_std.append(np.std(phi_phaseModulation[i]))

    idx=0
    axs[idx].bar(ind-0.5*width,phi_phaseModulation_mean, width, yerr=phi_phaseModulation_std,label=r'Phase modulation')
    axs[idx].bar(ind+0.5*width,phi_phaseReset_mean,width,yerr=phi_phaseReset_std,label=r'Phase reset')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels(situations)
    axs[idx].legend()
    axs[idx].set_ylabel(r'Phase convergence time [s]')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Phase reset')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'phase_convergence_time.svg'
    plt.savefig(figPath)
    plt.show()

def phase_stability_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0']):
    '''
    Plot formed phase stability statistic, it can indicates the actual traverability of the locomotion

    '''
    # 1) read data

    # 1.1) read Phase reset data
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)

    phi_phaseModulation={}
    
    for category, files_name in titles_files_categories: #name is a experiment class names
        if category in categories:
            print(category)
            phi_phaseModulation[category] =[]
            for idx in files_name.index: # how many time experiments one inclination
                folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
                print(folder_name)
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
                # 2)  data process
                touch_idx, convergence_idx, phi_stability =touch_convergence_moment_identification(grf_data,cpg_data,time)
                formedphase_stability=np.mean(phi_stability[convergence_idx:])
                phi_phaseModulation[category].append(1.0/formedphase_stability) #phase stability of the formed phase diff, inverse of the std
                print(phi_phaseModulation[category][-1])

    #1.2) read continuous phase modulation data
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseReset)

    phi_phaseReset={}
    for category, files_name in titles_files_categories: #name is a experiment class names
        if category in categories:
            print(category)
            phi_phaseReset[category] =[]
            for idx in files_name.index: # how many time experiments one inclination
                folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
                print(folder_name)
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
                # 2)  data process
                touch_idx, convergence_idx, phi_stability =touch_convergence_moment_identification(grf_data,cpg_data,time)
                formedphase_stability=np.mean(phi_stability[convergence_idx:])
                phi_phaseReset[category].append(1.0/formedphase_stability) #phase stability of the formed phase diff, inverse of the std
                print(phi_phaseReset[category][-1])

    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15
    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']

    #3.1) plot 
    phi_phaseReset_mean, phi_phaseReset_std=[],[]
    phi_phaseModulation_mean, phi_phaseModulation_std=[],[]
    for i in labels: #inclination
        phi_phaseReset_mean.append(np.mean(phi_phaseReset[i]))
        phi_phaseReset_std.append(np.std(phi_phaseReset[i]))

        phi_phaseModulation_mean.append(np.mean(phi_phaseModulation[i]))
        phi_phaseModulation_std.append(np.std(phi_phaseModulation[i]))

    idx=0
    axs[idx].bar(ind-0.5*width,phi_phaseModulation_mean, width, yerr=phi_phaseModulation_std,label=r'Phase modulation')
    axs[idx].bar(ind+0.5*width,phi_phaseReset_mean,width,yerr=phi_phaseReset_std,label=r'Phase reset')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels(situations)
    axs[idx].legend()
    axs[idx].set_ylabel(r'Phase stability')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Phase reset')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'Phase_stability.svg'
    plt.savefig(figPath)
    plt.show()

def percentage_plot_runningSuccess_statistic_2(data_file_dic,start_time=10,end_time=400,freq=60,experiment_categories=['0.0']):
    '''
    Statistical of running success
    

    '''
    # 1) read data
    #1.1) read  local data of phase modulation
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)
    pose_phaseModulation={}
    for category, files_name in titles_files_categories: #name is a inclination names
        pose_phaseModulation[category] =[]
        print(category)
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
            print(folder_name)
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            stability_temp= 1.0/np.std(pose_data[:,0],axis=0) + 1.0/np.std(pose_data[:,1],axis=0) #+ 1.0/np.std(pose_data[:,2]) +1.0/np.std(pose_data[:,5])
            pose_phaseModulation[category].append(stability_temp)
            print(pose_phaseModulation[category][-1])

    #1.2) read local data of phase reset
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories, categories =load_data_log(data_file_dic_phaseReset)
    pose_phaseReset={}
    for category, files_name in titles_files_categories: #name is a inclination names
        pose_phaseReset[category] =[]
        print(category)
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
            print(folder_name)
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            stability_temp= 1.0/np.std(pose_data[:,0],axis=0) + 1.0/np.std(pose_data[:,1],axis=0) #+ 1.0/np.std(pose_data[:,2]) + 1.0/np.std(pose_data[:,5])
            pose_phaseReset[category].append(stability_temp)
            print(pose_phaseReset[category][-1])

            
    #2) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15
    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']
    #3.1) plot 

    idx=0
    colors = ['lightblue', 'lightgreen']
    axs[idx].bar(ind-0.5*width,[len(pose_phaseModulation[ll])/10.0*100 for ll in labels], width,label=r'Phase modulation', color=colors[0])
    axs[idx].bar(ind+0.5*width,[len(pose_phaseReset[ll])/10.0*100 for ll in labels],width,label=r'Phase reset',color=colors[1])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[0,10])
    axs[idx].legend(loc='center right')
    axs[idx].set_ylabel(r'Success rate[%]')
    axs[idx].set_xlabel(r'Situations')
    axs[idx].set_xticklabels(situations)

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'runningSuccess.svg'
    plt.savefig(figPath)
    plt.show()



'''  The follwoing is the for the final version for paper data process  '''
'''                             牛逼                                    '''
'''---------------------------------------------------------------------'''

def GeneralDisplay_All(data_file_dic,start_time=90,end_time=1200,freq=60.0,experiment_categories=['0.0'],trial_id=0):
    ''' 
    This is for experiment two, for the first figure with one trail on inclination
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_id, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''
    # 1) read data
    titles_files_categories, categories=load_data_log(data_file_dic)

    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    pose={}
    jmc={}
    jmp={}
    jmv={}
    jmf={}
    grf={}
    cpg={}
    noise={}


    for category, files_name in titles_files_categories: #category is a files_name categorys
        gamma[category]=[]  #files_name is the table of the files_name category
        gait_diagram_data[category]=[]
        beta[category]=[]
        pose[category]=[]
        jmc[category]=[]
        jmp[category]=[]
        jmv[category]=[]
        jmf[category]=[]
        grf[category]=[]
        cpg[category]=[]
        noise[category]=[]
        if category in experiment_categories:
            print(category)
            for idx in files_name.index:
                folder_category= data_file_dic + files_name['data_files'][idx]
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                # 2)  data process
                print(folder_category)
                gait_diagram_data_temp, beta_temp = gait(grf_data)
                gait_diagram_data[category].append(gait_diagram_data_temp); beta[category].append(beta_temp)
                pose[category].append(pose_data)
                jmc[category].append(command_data)
                jmp[category].append(position_data)
                velocity_data=calculate_joint_velocity(position_data,freq)
                jmv[category].append(velocity_data)
                jmf[category].append(current_data)
                grf[category].append(grf_data)
                cpg[category].append(cpg_data)
                noise[category].append(module_data)
                temp_1=min([len(bb) for bb in beta_temp]) #minimum steps of all legs
                beta_temp2=np.array([beta_temp[0][:temp_1],beta_temp[1][:temp_1],beta_temp[2][:temp_1],beta_temp[3][0:temp_1]]) # transfer to np array
                if(beta_temp2 !=[]):
                    print("Coordination:",1.0/max(np.std(beta_temp2, axis=0)))
                else:
                    print("Coordination:",0.0)

                print("Stability:",calculate_stability(pose_data,grf_data))
                print("Balance:",calculate_body_balance(pose_data))
                displacement= calculate_displacement(pose_data)
                print("Displacemment:",displacement) #Displacement
                print("Distance:",calculate_distance(pose_data)) #Distance 
                print("Energy cost:", calculate_energy_cost(velocity_data,current_data,freq))
                print("COT:", calculate_COT(velocity_data,current_data,freq,displacement))
                print("Convergence time:",calculate_phase_convergence_time(time,grf_data,cpg_data,freq))



    #3) plot
    figsize=(8.2,9)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(23,len(experiment_categories))#13
    gs1.update(hspace=0.14,top=0.97,bottom=0.06,left=0.1,right=0.98)
    axs=[]
    for idx in range(len(experiment_categories)):# how many columns, depends on the experiment_categories
        axs.append(fig.add_subplot(gs1[0:3,idx]))
        axs.append(fig.add_subplot(gs1[3:6,idx]))
        axs.append(fig.add_subplot(gs1[6:9,idx]))
        axs.append(fig.add_subplot(gs1[9:12,idx]))
        axs.append(fig.add_subplot(gs1[12:15,idx]))
        axs.append(fig.add_subplot(gs1[15:18,idx]))
        axs.append(fig.add_subplot(gs1[18:21,idx]))
        axs.append(fig.add_subplot(gs1[21:23,idx]))

    #3.1) plot 
    #situations
    situations={'0':'Normal', '1':'Noisy feedback', '2':'Malfunction leg', '3':'Carrying load','0.9':'0.9'}
    # colors
    c4_1color=(46/255.0, 77/255.0, 129/255.0)
    c4_2color=(0/255.0, 198/255.0, 156/255.0)
    c4_3color=(255/255.0, 1/255.0, 118/255.0)
    c4_4color=(225/255.0, 213/255.0, 98/255.0)
    colors=[c4_1color, c4_2color, c4_3color, c4_4color]
    
    #experiment class
    experiment_category=experiment_categories[0]# The first category of the input parameters (arg)
    if not situations.__contains__(experiment_category):
        situations[experiment_category]=experiment_category

    #control method
    control_method=titles_files_categories['titles'].apply(lambda x: x.iat[0]).values[trial_id]

    #plotting
    idx=0
    axs[idx].plot(time,cpg[experiment_category][trial_id][:,1], color=c4_1color)
    axs[idx].plot(time,cpg[experiment_category][trial_id][:,3], color=c4_2color)
    axs[idx].plot(time,cpg[experiment_category][trial_id][:,5], color=c4_3color)
    axs[idx].plot(time,cpg[experiment_category][trial_id][:,7], color=c4_4color)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_ylabel(u'CPGs')
    axs[idx].set_yticks([-1.0,0.0,1.0])
    axs[idx].legend(['RF','RH','LF', 'LH'],ncol=4)
    axs[idx].set_xticklabels([])
    axs[idx].set_title(control_method+": " + situations[experiment_categories[0]] +" "+str(trial_id))
    axs[idx].set(xlim=[min(time),max(time)])


    idx=1
    axs[idx].set_ylabel(u'Phase diff. [rad]')
    phi=calculate_phase_diff(cpg[experiment_category][trial_id],time)
    phi_std=calculate_phase_diff_std(cpg[experiment_category][trial_id],time); 
    axs[idx].plot(phi['time'],phi['phi_12'],color=(77/255,133/255,189/255))
    axs[idx].plot(phi['time'],phi['phi_13'],color=(247/255,144/255,61/255))
    axs[idx].plot(phi['time'],phi['phi_14'],color=(89/255,169/255,90/255))
    axs[idx].plot(phi['time'],phi_std,color='k')
    axs[idx].legend([u'$\phi_{12}$',u'$\phi_{13}$',u'$\phi_{14}$',u'$\phi^{std}$'],ncol=2)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_yticks([0.0,0.7,1.5,2.1,3.0])
    axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])

    idx=2
    axs[idx].set_ylabel(u'Torque [Nm]')
    axs[idx].plot(time,jmf[experiment_category][trial_id][:,0], color=(129/255.0,184/255.0,223/255.0) )
    axs[idx].plot(time,jmf[experiment_category][trial_id][:,2],  color=(254/255.0,129/255.0,125/255.0))
    axs[idx].legend(['RF hip','RF knee'])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    #axs[idx].set_yticks([-0.3, 0.0, 0.3])
    axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])


    idx=3
    axs[idx].set_ylabel(u'Atti. [deg]')
    axs[idx].plot(time,pose[experiment_category][trial_id][:,0]*-57.3,color=(129/255.0,184/255.0,223/255.0))
    axs[idx].plot(time,pose[experiment_category][trial_id][:,1]*-57.3,color=(254/255.0,129/255.0,125/255.0))
    #axs[idx].plot(time,pose[experiment_category][trial_id][:,2]*-57.3,color=(86/255.0,169/255.0,90/255.0))
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_yticks([-5.0,0.0,5.0])
    axs[idx].legend(['Roll','Pitch'],loc='upper left')
    axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])


    idx=4
    axs[idx].set_ylabel(u'Disp. [m]')
    displacement_x = pose[experiment_category][trial_id][:,3]  -  pose[experiment_category][trial_id][0,3] #Displacement along x axis
    displacement_y = pose[experiment_category][trial_id][:,4]  -  pose[experiment_category][trial_id][0,4] #Displacement along y axis
    axs[idx].plot(time,displacement_x, color=(129/255.0,184/255.0,223/255.0))
    axs[idx].plot(time,displacement_y, color=(254/255.0,129/255.0,125/255.0))
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].legend(['X-axis','Y-axis'],loc='upper left')
    axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])

    idx=5
    axs[idx].set_ylabel(u'GRFs')
    if situations[experiment_categories[0]] == "Noisy feedback":
        grf_feedback_rf = grf[experiment_category][trial_id][:,0] + noise[experiment_category][trial_id][:,1]
        grf_feedback_rh = grf[experiment_category][trial_id][:,1] + noise[experiment_category][trial_id][:,2]
    else:
        grf_feedback_rf = grf[experiment_category][trial_id][:,0]
        grf_feedback_rh = grf[experiment_category][trial_id][:,1]
    
    if  control_method == "PhaseModulation":
        axs[idx].plot(time,grf_feedback_rf, color=c4_1color)
        axs[idx].plot(time,grf_feedback_rh, color=c4_2color)
        axs[idx].legend(['RF','RH'])

    if  control_method == "PhaseReset":
        axs[idx].plot(time,grf_feedback_rf, color=c4_1color)
        axs[idx].plot(time,grf_feedback_rh, color=c4_2color)
        axs[idx].plot(time,parameter_data[200,3]*np.ones(len(time)),'-.k') # Force threshold line, here it is 0.2, details can be see in synapticplasticityCPG.cpp
        axs[idx].legend(['Filtered RF','Filtered RH','Reset threshold'], ncol=2,loc='right')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    #axs[idx].set_yticks([-0.3, 0.0, 0.3])
    axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])


    idx=6
    axs[idx].set_ylabel(u'GRFs')
    if situations[experiment_categories[0]] == "Noisy feedback":
        grf_feedback_lf = grf[experiment_category][trial_id][:,2] + noise[experiment_category][trial_id][:,3]
        grf_feedback_lh = grf[experiment_category][trial_id][:,3] + noise[experiment_category][trial_id][:,4]
    else:
        grf_feedback_lf = grf[experiment_category][trial_id][:,2]
        grf_feedback_lh = grf[experiment_category][trial_id][:,3]
    
    if control_method == "PhaseModulation":
        axs[idx].plot(time,grf_feedback_lf, color=c4_3color)
        axs[idx].plot(time,grf_feedback_lh, color=c4_4color)
        axs[idx].legend(['LF','LH'])

    if control_method == "PhaseReset":
        axs[idx].plot(time,grf_feedback_lf, color=c4_3color)
        axs[idx].plot(time,grf_feedback_lh, color=c4_4color)
        axs[idx].plot(time,parameter_data[200,3]*np.ones(len(time)),'-.k') # Force threshold line, here it is 0.2, details can be see in synapticplasticityCPG.cpp
        axs[idx].legend(['Filtered LF','Filtered LH','Reset threshold'], ncol=2, loc='right')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    #axs[idx].set_yticks([-0.3, 0.0, 0.3])
    axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])


    idx=7
    axs[idx].set_ylabel(r'Gait')
    gait_diagram(fig,axs[idx],gs1,gait_diagram_data[experiment_category][trial_id])
    axs[idx].set_xlabel(u'Time [s]')
    xticks=np.arange(int(min(time)),int(max(time))+1,2)
    axs[idx].set_xticklabels([str(xtick) for xtick in xticks])
    axs[idx].set_xticks(xticks)
    axs[idx].yaxis.set_label_coords(-0.05,.5)
    axs[idx].set(xlim=[min(time),max(time)])

    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'general_display.svg'
    plt.savefig(figPath)
    plt.show()

'''  GRF patterns'''
def barplot_GRFs_patterns_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0','1','2','3']):
    '''
    plot GRFs patterns statistic after the locomotion is generated, it can indicates the features of the situations 
    

    '''

    # 1) read data
    # 1.1) read Phase reset data
    titles_files_categories, categories=load_data_log(data_file_dic)
    GRFs={}
    for category, files_name in titles_files_categories: #category is a files_name categorys
        GRFs[category]=[]
        control_method=files_name['titles'].iat[0]
        print(category)
        for idx in files_name.index:
            folder_category= data_file_dic + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
            # 2)  data process
            print(folder_category)
            if category=='1': # has noise 
                grf_data+=module_data[:,1:5]

            GRFs[category].append(grf_data)

    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    width=0.15
    trail_num= len(GRFs[experiment_categories[0]])
    ind=np.arange(4)#four legs
    situations=['S1\nNormal', 'S2\nNoisy feedback','S3\nMalfunction leg', 'S4\nCarrying load']
    #3.1) plot 
    RF_GRFs_mean, RF_GRFs_std= [], []
    RH_GRFs_mean, RH_GRFs_std= [], []
    LF_GRFs_mean, LF_GRFs_std= [], []
    LH_GRFs_mean, LH_GRFs_std= [], []
    for idx_situation in range(len(situations)):
        GRFs_mean, GRFs_std=[],[]
        for idx_leg in range(4): #four legs
            mean=[]
            std=[]
            for idx_trail in range(trail_num):
                mean.append(np.mean(GRFs[experiment_categories[idx_situation]][idx_trail][:,idx_leg]))
                std.append(np.std(GRFs[experiment_categories[idx_situation]][idx_trail][:,idx_leg]))
            GRFs_mean.append(np.mean(np.array(mean)))
            GRFs_std.append(np.std(np.array(std)))
        RF_GRFs_mean.append(GRFs_mean[0]);RF_GRFs_std.append(GRFs_std[0])
        RH_GRFs_mean.append(GRFs_mean[1]);RH_GRFs_std.append(GRFs_std[1])
        LF_GRFs_mean.append(GRFs_mean[2]);LF_GRFs_std.append(GRFs_std[2])
        LH_GRFs_mean.append(GRFs_mean[3]);LH_GRFs_std.append(GRFs_std[3])


    idx=0
    c4_1color=(46/255.0, 77/255.0, 129/255.0)
    c4_2color=(0/255.0, 198/255.0, 156/255.0)
    c4_3color=(255/255.0, 1/255.0, 118/255.0)
    c4_4color=(225/255.0, 213/255.0, 98/255.0)
    colors=[c4_1color, c4_2color, c4_3color, c4_4color]
    axs[idx].bar(ind-1.5*width,RF_GRFs_mean, width, yerr=RF_GRFs_std,label=r'RF',color=colors[0])
    axs[idx].bar(ind-0.5*width,RH_GRFs_mean, width, yerr=RH_GRFs_std,label=r'RH',color=colors[1])
    axs[idx].bar(ind+0.5*width,LF_GRFs_mean, width, yerr=LF_GRFs_std,label=r'LF',color=colors[2])
    axs[idx].bar(ind+1.5*width,LH_GRFs_mean, width, yerr=LH_GRFs_std,label=r'LH',color=colors[3])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].legend(ncol=2, loc='upper left')
    axs[idx].set_xticklabels(situations)
    axs[idx].set_ylabel(r'GRFs [N]')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Phase reset')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'GRFs_pattern.svg'
    plt.savefig(figPath)
    plt.show()

'''  Parameters analysis '''
def phaseModulation_parameter_investigation_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0'],trial_ids=range(15)):
    '''
    Plot convergence time of different parameter statistic, it can indicates the actual traverability of the locomotion

    '''
    # 1) read data
    # 1.1) read Phase reset data
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories=load_data_log(data_file_dic_phaseModulation)
    phi_phaseModulation={}
    for category, files_name in titles_files_categories: #category is a files_name categorys
        if category in experiment_categories:
            phi_phaseModulation[category] =[]
            control_method=files_name['titles'].iat[0]
            print("The experiment category: ",category)
            for idx in files_name.index: # the number of trials in a category
                if idx in np.array(files_name.index)[trial_ids]:# which one is to load
                    folder_category= data_file_dic_phaseModulation + files_name['data_files'][idx]
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    phi_phaseModulation[category].append(calculate_phase_convergence_time(time,grf_data,cpg_data,freq))
                    print("The number of trials:{idx}".format(idx=idx))
                    print("Convergence time: {:.2f}".format(phi_phaseModulation[category][-1]))

    #3) plot

    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95, bottom=0.14, left=0.12, right=0.88)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    if len(experiment_categories) > 1:
        labels=[str(ll) for ll in sorted([float(ll) for ll in experiment_categories])]
    else:
        labels=[str(ll) for ll in sorted([float(ll) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15

    #3.1) plot 


    phi_phaseModulation_mean, phi_phaseModulation_std=[],[]
    for i in labels: 
        phi_phaseModulation_mean.append(np.mean(phi_phaseModulation[i]))
        phi_phaseModulation_std.append(np.std(phi_phaseModulation[i]))

    idx=0
    color= 'tab:green'
    axs[idx].errorbar(ind,phi_phaseModulation_mean, yerr=phi_phaseModulation_std, fmt='-o', color=color)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_ylabel(r'Phase convergence time [s]', color=color)
    #axs[idx].set_ylim(-5,45)
    axs[idx].set_yticks([0,5,10,15,20,25,30])
    axs[idx].set_xticks(ind)
    axs[idx].set_xticklabels([round(float(ll),2) for ll in labels])
    axs[idx].set_xlabel(r'Threshold')
    axs[idx].tick_params(axis='y', labelcolor=color)


    success_rate=[]
    for i in labels: 
        success_count= np.array(phi_phaseModulation[i]) < 20.0
        success_rate.append(sum(success_count)/len(phi_phaseModulation[i])*100)
    ax2 = axs[idx].twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Success rate [%]', color=color)  # we already handled the x-label with ax1
    ax2.plot(ind, success_rate,'-h', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #ax2.set_ylim(-10,110)
    ax2.set_yticks([0, 20,40, 60, 80, 100])

    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'phaseModulation_parameter_convergence_time.svg'
    plt.savefig(figPath)
    plt.show()


def phaseReset_parameter_investigation_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0'],trial_ids=range(15)):
    '''
    Plot convergence time of different parameter statistic, it can indicates the actual traverability of the locomotion

    '''
    # 1) read data
    # 1.1) read Phase reset data
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories=load_data_log(data_file_dic_phaseReset)
    phi_phaseReset={}
    for category, files_name in titles_files_categories: #category is a files_name categorys
        if category in experiment_categories:
            phi_phaseReset[category] =[]
            control_method=files_name['titles'].iat[0]
            print("The experiment category: ",category)
            for idx in files_name.index: # the number of trials in a category
                if idx in np.array(files_name.index)[trial_ids]:# which one is to load
                    folder_category= data_file_dic_phaseReset + files_name['data_files'][idx]
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    phi_phaseReset[category].append(calculate_phase_convergence_time(time,grf_data,cpg_data,freq))
                    print("The number of trials:{idx}".format(idx=idx))
                    print("Convergence time: {:.2f}".format(phi_phaseReset[category][-1]))

    #3) plot

    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95, bottom=0.14, left=0.12, right=0.88)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([float(ll) for ll in experiment_categories])]
    ind= np.arange(len(labels))
    width=0.15

    #3.1) plot 


    phi_phaseReset_mean, phi_phaseReset_std=[],[]
    for i in labels: 
        phi_phaseReset_mean.append(np.mean(phi_phaseReset[i]))
        phi_phaseReset_std.append(np.std(phi_phaseReset[i]))

    idx=0
    color= 'tab:green'
    axs[idx].errorbar(ind,phi_phaseReset_mean, yerr=phi_phaseReset_std, fmt='-o', color=color)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_ylabel(r'Phase convergence time [s]', color=color)
    #axs[idx].set_ylim(-5,45)
    axs[idx].set_yticks([0,5,10,15,20,25,30])
    axs[idx].set_xticks(ind)
    axs[idx].set_xticklabels([round(float(ll),2) for ll in labels])
    axs[idx].set_xlabel(r'Threshold')
    axs[idx].tick_params(axis='y', labelcolor=color)


    success_rate=[]
    for i in labels: 
        success_count= np.array(phi_phaseReset[i]) < 20.0
        success_rate.append(sum(success_count)/len(phi_phaseReset[i])*100)
    ax2 = axs[idx].twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Success rate [%]', color=color)  # we already handled the x-label with ax1
    ax2.plot(ind, success_rate,'-h', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #ax2.set_ylim(-10,110)
    ax2.set_yticks([0, 20,40, 60, 80, 100])

    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'phaseReset_parameter_convergence_time.svg'
    plt.savefig(figPath)
    plt.show()

''' Boxplot for paper  '''
def boxplot_phase_formTime_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0']):
    '''
    Plot convergence time statistic, it can indicates the actual traverability of the locomotion

    '''
    # 1) read data
    # 1.1) read Phase reset data
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)
    phi_phaseModulation={}
    for category, files_name in titles_files_categories: #name is a inclination names
        if category in categories:
            print(category)
            phi_phaseModulation[category] =[]
            for idx in files_name.index: # how many time experiments one inclination
                folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
                print(folder_name)
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
                # 2)  data process
                phi_phaseModulation[category].append(calculate_phase_convergence_time(time,grf_data,cpg_data,freq))
                print(phi_phaseModulation[category][-1])

    #1.2) read phase reset data
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseReset)
    phi_phaseReset={}
    for category, files_name in titles_files_categories: #name is a inclination names
        if category in categories:
            print(category)
            phi_phaseReset[category] =[]
            for idx in files_name.index: # how many time experiments one inclination
                folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
                print(folder_name)
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
                # 2)  data process
                phi_phaseReset[category].append(calculate_phase_convergence_time(time,grf_data,cpg_data,freq))
                print(phi_phaseReset[category][-1])


    #3) plot
    figsize=(5.1,4.2)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.16,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))

    situations=['S1\nNormal', 'S2\nNoisy feedback', 'S3\nMalfunction leg', 'S3\nCarrying load']

    #3.1) plot 
    phi_phaseModulation_values=[]
    phi_phaseReset_values=[]
    for i in labels: #inclination
        phi_phaseModulation_values.append(phi_phaseModulation[i])
        phi_phaseReset_values.append(phi_phaseReset[i])

    idx=0
    boxwidth=0.2
    box1=axs[idx].boxplot(phi_phaseModulation_values,widths=boxwidth, positions=ind-boxwidth/2,vert=True,patch_artist=True,meanline=True,showmeans=True,showfliers=False)
    box2=axs[idx].boxplot(phi_phaseReset_values,widths=boxwidth, positions=ind+boxwidth/2,vert=True,patch_artist=True,meanline=True,showmeans=True,showfliers=False)

       # fill with colors
    colors = ['lightblue', 'lightgreen']
    for bplot, color in zip((box1, box2),colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticks(ind)
    axs[idx].set_xticklabels(situations)
    axs[idx].legend([box1['boxes'][0], box2['boxes'][0]],['Phase modulation','Phase reset'])
    axs[idx].set_ylabel(r'Phase convergence time [s]')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Phase reset')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'phase_convergence_time.svg'
    plt.savefig(figPath)
    plt.show()

def boxplot_phase_stability_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0']):
    '''
    Plot formed phase stability statistic, it can indicates the actual traverability of the locomotion

    '''
    # 1) read data

    # 1.1) read Phase reset data
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)

    phi_phaseModulation={}
    
    for category, files_name in titles_files_categories: #name is a experiment class names
        if category in categories:
            print(category)
            phi_phaseModulation[category] =[]
            for idx in files_name.index: # how many time experiments one inclination
                folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
                print(folder_name)
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
                # 2)  data process
                phase_stability=calculate_phase_diff_stability(grf_data,cpg_data,time)
                phi_phaseModulation[category].append(phase_stability) #phase stability of the formed phase diff, inverse of the std
                print(phi_phaseModulation[category][-1])

    #1.2) read continuous phase modulation data
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseReset)

    phi_phaseReset={}
    for category, files_name in titles_files_categories: #name is a experiment class names
        if category in categories:
            print(category)
            phi_phaseReset[category] =[]
            for idx in files_name.index: # how many time experiments one inclination
                folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
                print(folder_name)
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
                # 2)  data process
                phase_stability=calculate_phase_diff_stability(grf_data,cpg_data,time)
                phi_phaseReset[category].append(phase_stability) #phase stability of the formed phase diff, inverse of the std
                print(phi_phaseReset[category][-1])

    #3) plot
    figsize=(5.1,4.2)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.16,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))

    situations=['S1\nNormal', 'S2\nNoisy feedback', 'S3\nMalfunction leg', 'S4\nCarrying load']

    #3.1) plot 
    phi_phaseModulation_values=[]
    phi_phaseReset_values=[]
    for i in labels: #inclination
        phi_phaseModulation_values.append(phi_phaseModulation[i])
        phi_phaseReset_values.append(phi_phaseReset[i])

    idx=0
    boxwidth=0.2
    box1=axs[idx].boxplot(phi_phaseModulation_values,widths=boxwidth, positions=ind-boxwidth/2,vert=True,patch_artist=True,meanline=True,showmeans=True,showfliers=False)
    box2=axs[idx].boxplot(phi_phaseReset_values,widths=boxwidth, positions=ind+boxwidth/2,vert=True,patch_artist=True,meanline=True,showmeans=True,showfliers=False)

    # fill with colors
    colors = ['lightblue', 'lightgreen']
    for bplot, color in zip((box1, box2),colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].legend([box1['boxes'][0], box2['boxes'][0]],['Phase modulation','Phase reset'])
    axs[idx].set_xticklabels(situations)
    axs[idx].set_ylabel(r'Phase stability')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Phase reset')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'Phase_stability.svg'
    plt.savefig(figPath)
    plt.show()

def boxplot_displacement_statistic(data_file_dic,start_time=10,end_time=400,freq=60,experiment_categories=['0.0']):
    '''
    plot displacement statistic, it can indicates the actual traverability of the locomotion
    

    '''
    # 1) read data
    # 1.1) read Phase reset data
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)

    pose_phaseModulation={}
    for category, files_name in titles_files_categories: #name is a inclination names
        pose_phaseModulation[category] =[]
        print(category)
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
            print(folder_name)
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            disp=calculate_displacement(pose_data)
            pose_phaseModulation[category].append(disp) #Displacement on slopes 
            print(pose_phaseModulation[category][-1])
            
    #1.2) read continuous phase modulation data
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseReset)

    pose_phaseReset={}
    for category, files_name in titles_files_categories: #name is a inclination names
        pose_phaseReset[category] =[]
        print(category)
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
            print(folder_name)
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            disp=calculate_displacement(pose_data)
            pose_phaseReset[category].append(disp) #Displacement on slopes 
            print(pose_phaseReset[category][-1])


    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15
    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']


    #3.1) plot 
    disp_phaseReset_values=[]
    disp_phaseModulation_values=[]
    for i in labels: #inclination
        disp_phaseReset_values.append(pose_phaseReset[i])
        disp_phaseModulation_values.append(pose_phaseModulation[i])

    idx=0
    boxwidth=0.2
    box1=axs[idx].boxplot(disp_phaseModulation_values,widths=boxwidth, positions=ind-boxwidth/2,vert=True,patch_artist=True)
    box2=axs[idx].boxplot(disp_phaseReset_values,widths=boxwidth, positions=ind+boxwidth/2,vert=True,patch_artist=True)

    # fill with colors
    colors = ['lightblue', 'lightgreen']
    for bplot, color in zip((box1, box2),colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)


    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels(situations)
    axs[idx].legend([box1['boxes'][0], box2['boxes'][0]],['Phase modulation','Phase reset'])
    axs[idx].set_ylabel(r'Displacement [m]')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Phase reset')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'displacement.svg'
    plt.savefig(figPath)
    plt.show()

def boxplot_stability_statistic(data_file_dic,start_time=10,end_time=400,freq=60,experiment_categories=['0.0']):
    '''
    Stability of statistic

    '''
    # 1) read data
    #1.1) read loacal data of phase modulation
    data_file_dic_phaseModulation=data_file_dic+"PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)
    pose_phaseModulation={}
    for category, files_name in titles_files_categories: #name is a inclination names
        if category in categories:
            print(category)
            pose_phaseModulation[category] =[]
            for idx in files_name.index: # how many time experiments one inclination
                folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
                print(folder_name)
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
                # 2)  data process
                stability_temp=calculate_stability(pose_data,grf_data)
                pose_phaseModulation[category].append(stability_temp)
                print(pose_phaseModulation[category][-1])

    #1.2) read loacal data of phase reset
    data_file_dic_phaseReset=data_file_dic+"PhaseReset/"
    titles_files_categories, categories =load_data_log(data_file_dic_phaseReset)
    pose_phaseReset={}
    for category, files_name in titles_files_categories: #name is a inclination names
        if category in categories:
            print(category)
            pose_phaseReset[category] =[]
            for idx in files_name.index: # how many time experiments one inclination
                folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
                print(folder_name)
                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
                # 2)  data process
                stability_temp=calculate_stability(pose_data,grf_data)
                pose_phaseReset[category].append(stability_temp)
                print(pose_phaseReset[category][-1])


    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))
    width=0.15
    situations=['Normal', 'Noisy feedback', 'Malfunction leg', 'Carrying load']

    #3.1) plot 
    stability_phaseReset_values, stability_phaseModulation_values=[],[]
    for i in labels: #inclination
        stability_phaseReset_values.append(pose_phaseReset[i])
        stability_phaseModulation_values.append(pose_phaseModulation[i])


    idx=0
    boxwidth=0.2
    box1=axs[idx].boxplot(stability_phaseModulation_values,widths=boxwidth, positions=ind-boxwidth/2,vert=True,patch_artist=True)
    box2=axs[idx].boxplot(stability_phaseReset_values,widths=boxwidth, positions=ind+boxwidth/2,vert=True,patch_artist=True)

    # fill with colors
    colors = ['lightblue', 'lightgreen']
    for bplot, color in zip((box1, box2),colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)


    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels(situations)
    axs[idx].legend([box1['boxes'][0], box2['boxes'][0]],['Phase modulation','Phase reset'])
    axs[idx].set_ylabel(r'Stability')
    axs[idx].set_xlabel(r'Situations')

    # save plot
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'stabilityStatistic.svg'
    plt.savefig(figPath)
    plt.show()

def boxplot_COT_statistic(data_file_dic,start_time=60,end_time=900,freq=60.0,experiment_categories=['0.0'],trial_ids=[0]):
    '''
    @description: this is for comparative investigation, plot cost of transport statistic
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_ids, the trials that are included to computate
    @return: show and save a data figure.

    '''
    #1) load data from file
    
    #1.1) local COG reflex data
    data_file_dic_phaseModulation = data_file_dic + "PhaseModulation/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseModulation)

    COT_phaseModulation={}

    for category, files_name in titles_files_categories: #name is a inclination names
        COT_phaseModulation[category]=[]  #inclination is the table of the inclination name
        print(category)
        for idx in files_name.index:
            folder_name= data_file_dic_phaseModulation + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            print(folder_name)
            velocity_data=calculate_joint_velocity(position_data,freq)
            d=calculate_displacement(pose_data)
            COT=calculate_COT(velocity_data,current_data,freq,d)
            COT_phaseModulation[category].append(COT)# 
            print(COT_phaseModulation[category][-1])
    
    #1.2) local vestibular reflex data
    data_file_dic_phaseReset = data_file_dic + "PhaseReset/"
    titles_files_categories, categories=load_data_log(data_file_dic_phaseReset)
    
    COT_phaseReset={}

    for category, files_name in titles_files_categories: #name is a inclination names
        COT_phaseReset[category]=[]  #inclination is the table of the inclination name
        print(category)
        for idx in files_name.index:
            folder_name= data_file_dic_phaseReset + files_name['data_files'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_name)
            # 2)  data process
            print(folder_name)
            velocity_data=calculate_joint_velocity(position_data,freq)
            d=calculate_displacement(pose_data)
            COT=calculate_COT(velocity_data,current_data,freq,d)
            COT_phaseReset[category].append(COT)# 
            print(COT_phaseReset[category][-1])

            
    #3) plot
    figsize=(5.1,4.2)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.16,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels=[str(ll) for ll in sorted([ round(float(ll)) for ll in categories])]
    ind= np.arange(len(labels))

    situations=['S1\nNormal', 'S2\nNoisy feedback', 'S3\nMalfunction leg', 'S4\nCarrying load']


    #3.1) plot 
    COT_phaseModulation_values,COT_phaseReset_values=[],[]
    for i in labels:
        COT_phaseModulation_values.append(COT_phaseModulation[i])
        COT_phaseReset_values.append(COT_phaseReset[i])


    idx=0
    boxwidth=0.2
    box1=axs[idx].boxplot(COT_phaseModulation_values,widths=boxwidth, positions=ind-boxwidth/2,vert=True,patch_artist=True,meanline=True,showmeans=True,showfliers=False)
    box2=axs[idx].boxplot(COT_phaseReset_values,widths=boxwidth, positions=ind+boxwidth/2,vert=True,patch_artist=True,meanline=True,showmeans=True,showfliers=False)

    # fill with colors
    colors = ['lightblue', 'lightgreen']
    for bplot, color in zip((box1, box2),colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)


    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels(situations)
    axs[idx].legend([box1['boxes'][0], box2['boxes'][0]],['Phase modulation','Phase reset'])
    axs[idx].set_ylabel(r'$COT [J kg^{-1} m^{-1}$]')
    axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'COTStatistic.svg'
    plt.savefig(figPath)
    plt.show()


def plot_single_details(data_file_dic,start_time=5,end_time=30,freq=60.0,experiment_categories=['0.0'],trial_ids=[0],control_methods=['apnc'],investigation="parameter investigation"):
    ''' 
    This is for experiment two, for the first figure with one trail on inclination
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data, units: seconds
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_ids, it indicates which experiments (trials) among a inclination/situation/case experiments 
    @param: control_methods which indicate the control method used in the experiment
    @param: investigation, it indicates which experiment in the paper, Experiment I, experiment II, ...
    @return: show and save a data figure.
    '''
    # 1) read data
    experiment_data, metrics=metrics_calculatiions(data_file_dic, start_time, end_time, freq, experiment_categories, trial_ids=trial_ids, control_methods=control_methods)


    #2) Whether get right data
    for exp_idx in range(len(experiment_categories)):
        for trial_id in range(len(trial_ids)):
            experiment_category=experiment_categories[exp_idx]# The first category of the input parameters (arg)
            control_method=control_methods[0]
            cpg=experiment_data[experiment_category][control_method][trial_id]['cpg']
            grf=experiment_data[experiment_category][control_method][trial_id]['grf']
            time=experiment_data[experiment_category][control_method][trial_id]['time']
            gait_diagram_data=experiment_data[experiment_category][control_method][trial_id]['gait_diagram_data']

            #3) plot
            figsize=(6,6.5)
            fig = plt.figure(figsize=figsize,constrained_layout=False)
            plot_column_num=1# the columns of the subplot. here set it to one
            gs1=gridspec.GridSpec(14,plot_column_num)#13
            gs1.update(hspace=0.18,top=0.95,bottom=0.08,left=0.1,right=0.98)
            axs=[]
            for idx in range(plot_column_num):# how many columns, depends on the experiment_categories
                axs.append(fig.add_subplot(gs1[0:3,idx]))
                axs.append(fig.add_subplot(gs1[3:6,idx]))
                axs.append(fig.add_subplot(gs1[6:9,idx]))
                axs.append(fig.add_subplot(gs1[9:12,idx]))
                axs.append(fig.add_subplot(gs1[12:14,idx]))

            #3.1) plot 
            if investigation=="situation investigation":
                #experiment_variables={'0':'Normal', '1':'Noisy feedback', '2':'Malfunction leg', '3':'Carrying load','0.9':'0.9'}
                experiment_variables={'0':'Normal', '1':'Noisy feedback', '2':'Malfunction leg', '3':'Carrying load','0.9':'0.9'}

            if investigation=="parameter investigation":
                experiment_variables=['0.0','0.05','0.15','0.25','0.35','0.45','0.55']
    
            c4_1color=(46/255.0, 77/255.0, 129/255.0)
            c4_2color=(0/255.0, 198/255.0, 156/255.0)
            c4_3color=(255/255.0, 1/255.0, 118/255.0)
            c4_4color=(225/255.0, 213/255.0, 98/255.0)
            colors=[c4_1color, c4_2color, c4_3color, c4_4color]


            idx=0
            axs[idx].plot(time,cpg[:,1], color=c4_1color)
            axs[idx].plot(time,cpg[:,3], color=c4_2color)
            axs[idx].plot(time,cpg[:,5], color=c4_3color)
            axs[idx].plot(time,cpg[:,7], color=c4_4color)
            axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
            axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
            axs[idx].set_ylabel(u'CPGs')
            axs[idx].set_yticks([-1.0,0.0,1.0])
            axs[idx].legend(['RF','RH','LF', 'LH'],ncol=4, loc='right')
            axs[idx].set_xticklabels([])
            axs[idx].set_title(control_method+": " + experiment_category +" "+str(trial_id))
            axs[idx].set(xlim=[min(time),max(time)])
            axs[idx].set(ylim=[-1.1,1.1])


            plt.subplots_adjust(hspace=0.4)
            idx=1
            axs[idx].set_ylabel(u'Phase diff. [rad]')
            phi=calculate_phase_diff(cpg,time)
            phi_std=calculate_phase_diff_std(cpg,time,method_option=1); 
            axs[idx].plot(phi['time'],phi['phi_12'],color=(77/255,133/255,189/255))
            axs[idx].plot(phi['time'],phi['phi_13'],color=(247/255,144/255,61/255))
            axs[idx].plot(phi['time'],phi['phi_14'],color=(89/255,169/255,90/255))
            axs[idx].plot(phi['time'],phi_std,color='k')
            #axs[idx].plot(phi['time'],savgol_filter(phi['phi_12'],91,2,mode='nearest'),color='k',linestyle="-.")
            #ax2 = axs[idx].twinx()  # instantiate a second axes that shares the same x-axis
            #ax2.set_ylabel('Phase disance', color='tab:red')  # we already handled the x-label with ax1
            #ax2.plot(phi['time'],phi_std,color='red')
            axs[idx].legend([u'$\phi_{12}$',u'$\phi_{13}$',u'$\phi_{14}$',u'$\phi^{dis}$'],ncol=2,loc='center')
            axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
            axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
            axs[idx].set_yticks([0.0,1.5,3.0])
            axs[idx].set_xticklabels([])
            axs[idx].set(xlim=[min(time),max(time)])
            axs[idx].set(ylim=[-0.1,3.5])

            #ax2.plot(ind, success_rate,'-h', color=color)
            #ax2.tick_params(axis='y', labelcolor=color)
            #ax2.set_ylim(-10,110)
            #ax2.set_yticks([0, 20,40, 60, 80, 100])

            idx=2
            axs[idx].set_ylabel(u'GRFs')
            if experiment_category == "1": # noisy situation
                grf_feedback_rf = grf[:,0] + noise[experiment_category][trial_id][:,1]
                grf_feedback_rh = grf[:,1] + noise[experiment_category][trial_id][:,2]
                axs[idx].set(ylim=[-1,20.1])
            else:
                grf_feedback_rf = grf[:,0]
                grf_feedback_rh = grf[:,1]
                axs[idx].set(ylim=[-1,20.1])

            if  control_method in ["PhaseModulation","phase_modulation","apnc"] :
                axs[idx].plot(time,grf_feedback_rf, color=c4_1color)
                axs[idx].plot(time,grf_feedback_rh, color=c4_2color)
                axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
                axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
                axs[idx].legend(['RF','RH'])
                axs[idx].set_xticklabels([])
                axs[idx].set(xlim=[min(time),max(time)])

            if  control_method in ["PhaseReset", "phase_reset"]:
                axs[idx].plot(time,grf_feedback_rf, color=c4_1color)
                axs[idx].plot(time,grf_feedback_rh, color=c4_2color)
                GRF_threshold=experiment_data[experiment_category][control_method][trial_id]['rosparameter'][-1,3]*25/4*np.ones(len(time))
                axs[idx].plot(time,GRF_threshold,'-.k') # Force threshold line, here it is 0.2, details can be see in synapticplasticityCPG.cpp
                axs[idx].set_yticks([0,GRF_threshold[0],10,20])
                axs[idx].set_yticklabels(['0',str(round(GRF_threshold[0],2)),'10','20'])
                axs[idx].legend(['RF','RH','Threshold'], ncol=3,loc='right')
                axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
                axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
                axs[idx].set_xticklabels([])
                axs[idx].set(xlim=[min(time),max(time)])


            idx=3
            axs[idx].set_ylabel(u'GRFs')
            if experiment_category == "1": #noisy situation
                grf_feedback_lf = grf[:,2] + noise[experiment_category][trial_id][:,3]
                grf_feedback_lh = grf[:,3] + noise[experiment_category][trial_id][:,4]
                axs[idx].set(ylim=[-1,20.1])
            else:
                grf_feedback_lf = grf[:,2]
                grf_feedback_lh = grf[:,3]
                axs[idx].set(ylim=[-1,20.1])

            if  control_method in ["PhaseModulation","phase_modulation","apnc"] :
                axs[idx].plot(time,grf_feedback_lf, color=c4_3color)
                axs[idx].plot(time,grf_feedback_lh, color=c4_4color)
                axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
                axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
                axs[idx].legend(['RF','RH'])
                axs[idx].set_xticklabels([])
                axs[idx].set(xlim=[min(time),max(time)])

            if  control_method in ["PhaseReset", "phase_reset"]:
                axs[idx].plot(time,grf_feedback_lf, color=c4_3color)
                axs[idx].plot(time,grf_feedback_lh, color=c4_4color)
                GRF_threshold=experiment_data[experiment_category][control_method][trial_id]['rosparameter'][-1,3]*25/4*np.ones(len(time))
                axs[idx].plot(time,GRF_threshold,'-.k') # Force threshold line, here it is 0.2, details can be see in synapticplasticityCPG.cpp
                axs[idx].set_yticks([0,GRF_threshold[0],10,20])
                axs[idx].set_yticklabels(['0',str(round(GRF_threshold[0],2)),'10','20'])
                axs[idx].legend(['LF','LH','Threshold'], ncol=3, loc='right')
                axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
                axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
                #axs[idx].set_yticks([-0.3, 0.0, 0.3])
                axs[idx].set_xticklabels([])
                axs[idx].set(xlim=[min(time),max(time)])
            idx=4
            axs[idx].set_ylabel(r'Gait')
            gait_diagram(fig,axs[idx],gs1,gait_diagram_data)
            axs[idx].set_xlabel(u'Time [s]')
            xticks=np.arange(int(min(time)),int(max(time))+1,1)
            axs[idx].set_xticks(xticks)
            axs[idx].set_xticklabels([str(xtick) for xtick in xticks])
            axs[idx].yaxis.set_label_coords(-0.065,.5)
            axs[idx].set(xlim=[min(time),max(time)])
            # save figure
            folder_fig = os.path.join(data_file_dic, 'data_visulization/')
            if not os.path.exists(folder_fig):
                os.makedirs(folder_fig)

            figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime())) + experiment_category+"_"+control_method+'_general_display.svg'
            plt.savefig(figPath)
            plt.show()
            plt.close()


def plot_cpg_phase_portrait(data_file_dic,start_time=5,end_time=40,freq=60.0,experiment_categories=['0.0'],trial_ids=[0]):
    ''' 
    This is for plot CPG phase shift dynamics
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data, unit: second
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_ids, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''

    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    cpg={}
    grf={}
    for category, files_name in titles_files_categories: #category is a files_name categorys
        if category in experiment_categories:
            cpg[category]=[]
            grf[category]=[]
            control_method=files_name['titles'].iat[0]
            print(category)
            for idx in files_name.index:
                if idx in np.array(files_name.index)[trial_ids]:# which one is to load
                    folder_category= data_file_dic + files_name['data_files'][idx]
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    print(folder_category)
                    cpg[category].append(cpg_data)
                    grf[category].append(grf_data)

    #2) plot
    figsize=(4+4,4)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    markers=['g*','g*','g*','k^','y<','k>','ks','kp']

    gs1=gridspec.GridSpec(1,6)#13
    gs1.update(hspace=0.1,wspace=1.4,top=0.92,bottom=0.18,left=0.18,right=0.92)
    axs=[]
    axs.append(fig.add_subplot(gs1[0,0:3]))
    axs.append(fig.add_subplot(gs1[0,3:6]))

    titles=experiment_categories

    for exp_idx in range(len(experiment_categories)):
        for trial_id in range(len(trial_ids)):
            plot_idx=trial_id
            experiment_category=experiment_categories[exp_idx]# The first category of the input parameters (arg)
            if not cpg[experiment_category]:
                warnings.warn('Without proper data was read')
            cpg_idx=0
            #3.1) draw
            axs[plot_idx].set_aspect('equal', adjustable='box')
            axs[plot_idx].plot(cpg[experiment_category][trial_id][:100,0], cpg[experiment_category][trial_id][:100,1],'.',markersize=2)
            axs[plot_idx].set_xlabel(u'$o_{1k}$')
            axs[plot_idx].set_ylabel(u'$o_{2k}$')
            #plot reset point
            axs[plot_idx].plot([np.tanh(1)],[0],'ro')
            axs[plot_idx].annotate('Phase resetting point', xy=(np.tanh(1), 0), xytext=(-0.3, 0.1), arrowprops=dict(arrowstyle='->'))
            #plot CPG output of touch moment
            touch_idx, convergence_idx=calculate_touch_idx_phaseConvergence_idx(time, grf[experiment_category][trial_id],cpg[experiment_category][trial_id])
            print("touch_idx",touch_idx)
            #touch_point_markers=("b^","b>","bv","b<")
            touch_point_markers=("r.","g.","b.","k.")
            for cpg_idx in range(4):
                touch_point_x=[cpg[experiment_category][trial_id][touch_idx-3, 2*cpg_idx]]
                touch_point_y=[cpg[experiment_category][trial_id][touch_idx-3, 2*cpg_idx+1]]
                axs[plot_idx].plot(touch_point_x,touch_point_y,touch_point_markers[cpg_idx])
            #axs[plot_idx].annotate('Inital condition', xy=(touch_point_x[0], touch_point_y[0]), xytext=(-0.1,-0.3), arrowprops=dict(arrowstyle='->'))
            axs[plot_idx].set_xlim([-1,1])
            axs[plot_idx].set_ylim([-1,1])
            axs[plot_idx].set_yticks([-1, -0.5, 0.0, 0.5, 1.0])
            axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
            axs[plot_idx].grid(which='both',axis='y',color='k',linestyle=':')
    #axs[plot_idx].legend(experiment_categories)


    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'CPG_Phase_portrait.svg'
    plt.savefig(figPath)
    plt.show()
    

''' Boxplot for paper 2  '''
def boxplot_phase_convergenceTime_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0'],trial_ids=[0]):
    '''
    Plot convergence time statistic, it can indicates the actual traverability of the locomotion

    '''
    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    phi={}
    for category, files_name in titles_files_categories: #category is the thrid columns, files_name is the table
        if category in experiment_categories:
            phi[category]=[]
            control_method=files_name['titles'].iat[0]
            print("The experiment category:", category)
            for idx in files_name.index: #遍历某个分类category下的所有数据文件
                if idx in np.array(files_name.index)[trial_ids]:# which one is to load
                    folder_category= data_file_dic + files_name['data_files'][idx]
                    print(folder_category)
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    if calculate_phase_convergence_time(time,grf_data,cpg_data,freq)>0:
                        phi[category].append(calculate_phase_convergence_time(time,grf_data,cpg_data,freq))
                        print("Convergence time:{:.2f}".format(phi[category][-1]))

    #3) plot
    figsize=(5.1,4.0)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.16,left=0.12,right=0.88)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    roughness=[int(float(ll)*100) for ll in experiment_categories]
    print(roughness)
    ind= np.arange(len(experiment_categories))


    #3.1) plot 
    phi_values=[]
    for key in experiment_categories:
        phi_values.append(phi[key])
    
    idx=0
    boxwidth=0.2
    box1=axs[idx].boxplot(phi_values,widths=boxwidth, positions=ind,vert=True,patch_artist=True,meanline=True,showmeans=True,showfliers=False)

       # fill with colors
    colors = ['lightblue', 'lightgreen']
    #for bplot, color in zip((box1, box2),colors):
    #    for patch in bplot['boxes']:
    #        patch.set_facecolor(color)

    #axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    #axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    #axs[idx].legend([box1['boxes'][0], box2['boxes'][0]],['Phase modulation','Phase reset'])
    axs[idx].set_xticks(ind)
    axs[idx].set_xticklabels(roughness)
    axs[idx].set_ylabel(r'Phase convergence time [s]')
    axs[idx].set_xlabel(r'Roughness [%]')

    # plot the success_rate with twinx 
    ax2 = axs[idx].twinx()  # instantiate a second axes that shares the same x-axis
    success_rate=[]
    for idx in range(len(phi_values)):
        success_count= np.array(phi_values[idx]) < 20.0
        success_rate.append(len(phi_values[idx])/len(trial_ids)*100)
    color = 'tab:red'
    ax2.set_ylabel('Success rate [%]', color=color)  # we already handled the x-label with ax1
    ax2.plot(ind, success_rate,'-h', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #ax2.set_ylim(-10,110)
    ax2.set_yticks([0, 20,40, 60, 80, 100])

    #axs[idx].set_title('Phase reset')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'phase_convergence_time.svg'
    plt.savefig(figPath)
    plt.show()


def boxplot_phase_convergenceTime_sucessRateBar_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0'],trial_ids=[0]):
    '''
    Plot convergence time statistic, it can indicates the actual traverability of the locomotion

    '''
    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    phi={}
    for category, files_name in titles_files_categories: #category is the thrid columns, files_name is the table
        if category in experiment_categories:
            phi[category]=[]
            control_method=files_name['titles'].iat[0]
            print("The experiment category:", category)
            for idx in files_name.index: #遍历某个分类category下的所有数据文件
                if idx in np.array(files_name.index)[trial_ids]:# which one is to load
                    folder_category= data_file_dic + files_name['data_files'][idx]
                    print(folder_category)
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    if calculate_phase_convergence_time(time,grf_data,cpg_data,freq)>0:
                        phi[category].append(calculate_phase_convergence_time(time,grf_data,cpg_data,freq))
                        print("Convergence time:{:.2f}".format(phi[category][-1]))

    #3) plot
    figsize=(5.1,4.0)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.16,left=0.12,right=0.88)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    roughness=[int(float(ll)*100) for ll in experiment_categories]
    print(roughness)
    ind= np.arange(len(experiment_categories))


    #3.1) plot 
    phi_values=[]
    for key in experiment_categories:
        phi_values.append(phi[key])
    


    success_rate=[]
    for idx in range(len(phi_values)):
        success_count= np.array(phi_values[idx]) < 20.0
        success_rate.append(len(phi_values[idx])/len(trial_ids)*100)

    idx=0
    boxwidth=0.2
    box1=axs[idx].boxplot(phi_values,widths=boxwidth, positions=ind-boxwidth/2,vert=True,patch_artist=True,meanline=True,showmeans=True,showfliers=False)
    # plot the success_rate with twinx 
    ax2 = axs[idx].twinx()  # instantiate a second axes that shares the same x-axis
    #box2=ax2.boxplot(success_rate,widths=boxwidth, positions=ind+boxwidth/2,vert=True,patch_artist=True,meanline=True,showmeans=True,showfliers=False)
    box2=ax2.bar(ind+boxwidth/2,success_rate)

       # fill with colors
    colors = ['lightblue', 'lightgreen']
    # for bplot, color in zip((box1, box2),colors):
    #     for patch in bplot['boxes']:
    #         patch.set_facecolor(color)

    #axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    #axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    #axs[idx].legend([box1['boxes'][0], box2['boxes'][0]],['Phase modulation','Phase reset'])
    axs[idx].set_xticks(ind)
    axs[idx].set_xticklabels(roughness)
    axs[idx].set_ylabel(r'Phase convergence time [s]')
    axs[idx].set_xlabel(r'Roughness [%]')


    ax2.set_ylabel('Success rate [%]', color=colors[1])  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=colors[1])
    #ax2.set_ylim(-10,110)
    ax2.set_yticks([70, 80, 90, 100])

    #axs[idx].set_title('Phase reset')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'phase_convergence_time.svg'
    plt.savefig(figPath)
    plt.show()

def scatter_dutyFactors_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0'],trial_ids=[0]):
    '''
    Plot duty factors in statistic, it can indicates generated walking patterns

    '''
    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    beta={}
    for category, files_name in titles_files_categories: #category is the thrid columns, files_name is the table
        if category in experiment_categories:
            beta[category]=[]
            control_method=files_name['titles'].iat[0]
            print("The experiment category:", category)
            for idx in files_name.index: #遍历某个分类category下的所有数据文件
                if idx in np.array(files_name.index)[trial_ids]:# which one is to load
                    folder_category= data_file_dic + files_name['data_files'][idx]
                    print(folder_category)
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    state_temp, beta_temp=gait(grf_data)
                    number_beta=min([len(beta_temp[0]),len(beta_temp[1]),len(beta_temp[2]),len(beta_temp[3])])
                    beta[category].append([beta_temp[0][:number_beta],beta_temp[1][:number_beta],beta_temp[2][:number_beta],beta_temp[3][:number_beta]])
        
    
    #data processing
    pd_beta='NO'
    for category in experiment_categories:
        for trial_id in trial_ids:
            data_temp={'MI': [category]*len(beta[category][trial_id][0]),'trial_id':[trial_id]*len(beta[category][trial_id][0]),'beta1':beta[category][trial_id][0], 'beta2':beta[category][trial_id][1], 'beta3':beta[category][trial_id][2], 'beta4':beta[category][trial_id][3]}
            pd_beta_temp=pd.DataFrame(data=data_temp,columns=['MI','trial_id','beta1','beta2','beta3','beta4'])
            pd_beta_temp=pd_beta_temp.dropna()
            if(isinstance(pd_beta,str)):
                pd_beta=pd_beta_temp;
            else:
                pd_beta=pd_beta.append(pd_beta_temp,ignore_index=True)

    #3) plot
    #figsize=(5.1,4.0)
    #fig = plt.figure(figsize=figsize,constrained_layout=False)
    #gs1=gridspec.GridSpec(6,1)#13
    #gs1.update(hspace=0.18,top=0.95,bottom=0.16,left=0.12,right=0.88)
    #axs=[]
    #axs.append(fig.add_subplot(gs1[0:6,0]))
    #fig, ax = plt.subplots(figsize=figsize)


    #plotting 

    #sns.catplot(data=pd_beta, kind="bar", x="MI", y="Beta", hue="legs", ci="sd", palette="dark", alpha=.6, height=6)
    pd_beta['beta']=(pd_beta['beta1']+pd_beta['beta2']+pd_beta['beta3']+pd_beta['beta4'])/4
    g=sns.catplot(data=pd_beta, kind="bar", x="MI", y="beta", ci="sd", palette="dark", alpha=.5, height=6)
    g.set_axis_labels("MI", "Duty factor")
    g.set(ylim=(0.5, None),yticks=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0])

    plt.grid(axis='y',linestyle='--',color='black',linewidth='0.5')
    plt.axes()
    #axs[0].set_ylim(0.5,1.0)

    #sns.displot(pd_beta, x="beta1", y="beta3", hue="MI")
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'dutyFactors.svg'
    plt.savefig(figPath)
    
    plt.show()

def scatter_phaseShift_statistic(data_file_dic,start_time=800,end_time=2400,freq=60,experiment_categories=['0'],trial_ids=[0]):
    '''
    Plot duty factors in statistic, it can indicates generated walking patterns

    '''

    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    phi={}
    for category, files_name in titles_files_categories: #category is the thrid columns, files_name is the table
        if category in experiment_categories: # 仅仅读取入口参数指定的实验类别数据
            phi[category]=[]
            control_method=files_name['titles'].iat[0]
            print("The experiment category:", category)
            for idx in files_name.index: #遍历某个分类category下的所有数据文件
                if idx in np.array(files_name.index)[trial_ids]:# 仅仅读取指定trial_id的数据
                    folder_category= data_file_dic + files_name['data_files'][idx]
                    print(folder_category)
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    phi[category].append(calculate_phase_diff(cpg_data,time))
        

    # 3) Transfer the data into a pandas format
    pd_phi='NO'
    for category in experiment_categories:
        for trial_id in trial_ids:
            phi[category][trial_id]['MI']=category
            phi[category][trial_id]['trial_id']=trial_id
            phi[category][trial_id]=phi[category][trial_id].dropna()
            if(isinstance(pd_phi,str)):
                pd_phi=phi[category][trial_id];
            else:
                pd_phi=pd_phi.append(phi[category][trial_id],ignore_index=True)




    
    long_pd_phi=pd_phi.melt(id_vars=['time','MI','trial_id'])
    group_long_pd_phi=long_pd_phi.groupby('MI')

    #sns.lineplot(data=group_long_pd_phi,x='time',y='value',hue='variable')
    #sns.violinplot(data=long_pd_phi, x="MI", y="value", hue="variable", split=True, inner="quart", linewidth=1,palette={"Yes": "b", "No": ".85"})
    #g=sns.displot(data=pd_phi, x="phi_12", y="phi_13", hue='MI',kind='kde')
    g=sns.catplot(data=long_pd_phi, kind="bar", x="MI", y="value", ci="sd",hue='variable', alpha=.6, height=6)
    #g.set_axis_labels("MI", "Phase shift")
    #sns.displot(pd_beta, x="beta1", y="beta3", hue="MI")
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'MI_phaseShift.svg'
    plt.savefig(figPath)
    
    plt.show()


def scatter_COT_statistic(data_file_dic,start_time=60,end_time=900,freq=60.0,experiment_categories=['0.0'],trial_ids=[0]):
    '''
    @description: this is for comparative investigation, plot cost of transport statistic
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_ids, the trials that are included to computate
    @return: show and save a data figure.

    '''
    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    COT={}
    for category, files_name in titles_files_categories: #category is the thrid columns, files_name is the table
        if category in experiment_categories: # 仅仅读取入口参数指定的实验类别数据
            COT[category]=[]
            control_method=files_name['titles'].iat[0]
            print("The experiment category:", category)
            for idx in files_name.index: #遍历某个分类category下的所有数据文件
                if idx in np.array(files_name.index)[trial_ids]:# 仅仅读取指定trial_id的数据
                    folder_category= data_file_dic + files_name['data_files'][idx]
                    print(folder_category)
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    velocity_data=calculate_joint_velocity(position_data,freq)
                    d=calculate_displacement(pose_data)
                    COT_temp=calculate_COT(velocity_data,current_data,freq,d)
                    COT[category].append(COT_temp)# 
                    print(COT[category][-1])
                    
        
    pd_COT=pd.DataFrame(COT)
    pd_COT=pd_COT.melt()
    g=sns.catplot(data=pd_COT, kind="bar", x="variable", y="value", ci="sd",alpha=.6, height=6)
    #axs[idx].set_ylabel(r'$COT [J kg^{-1} m^{-1}$]')
    #axs[idx].set_xlabel(r'Situations')

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'COTStatistic.svg'
    plt.savefig(figPath)
    plt.show()



def boxplot_phase_convergenceTime_statistic_underMI(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0'],trial_ids=[0]):
    '''
    Plot convergence time statistic, it can indicates the actual traverability of the locomotion

    '''
    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    phi={}
    for category, files_name in titles_files_categories: #category is the thrid columns, files_name is the table
        if category in experiment_categories:
            phi[category]=[]
            control_method=files_name['titles'].iat[0]
            print("The experiment category:", category)
            for idx in files_name.index: #遍历某个分类category下的所有数据文件
                if idx in np.array(files_name.index)[trial_ids]:# which one is to load
                    folder_category= data_file_dic + files_name['data_files'][idx]
                    print(folder_category)
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    if calculate_phase_convergence_time(time,grf_data,cpg_data,freq)>0:
                        phi[category].append(calculate_phase_convergence_time(time,grf_data,cpg_data,freq))
                        print("Convergence time:{:.2f}".format(phi[category][-1]))

    #3) plot
    figsize=(6,4)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.16,left=0.12,right=0.89)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    MI=[float(ll) for ll in experiment_categories]
    print(MI)
    ind= np.arange(len(experiment_categories))


    #3.1) plot 
    phi_values=[]
    for key in experiment_categories:
        phi_values.append(phi[key])
    
    idx=0
    boxwidth=0.2
    box1=axs[idx].boxplot(phi_values,widths=boxwidth, positions=ind,vert=True,patch_artist=True,meanline=True,showmeans=True,showfliers=False)

       # fill with colors
    colors = ['lightblue', 'lightgreen']
    #for bplot, color in zip((box1, box2),colors):
    #    for patch in bplot['boxes']:
    #        patch.set_facecolor(color)

    #axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    #axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    #axs[idx].legend([box1['boxes'][0], box2['boxes'][0]],['Phase modulation','Phase reset'])
    axs[idx].set_xticks(ind)
    axs[idx].set_xticklabels(MI)
    axs[idx].set_ylabel(r'Phase convergence time [s]')
    axs[idx].set_xlabel(r'MI')

    # plot the success_rate with twinx 
    ax2 = axs[idx].twinx()  # instantiate a second axes that shares the same x-axis
    success_rate=[]
    for idx in range(len(phi_values)):
        success_count= np.array(phi_values[idx]) < 20.0
        success_rate.append(len(phi_values[idx])/len(trial_ids)*100)
    color = 'tab:red'
    ax2.set_ylabel('Success rate [%]', color=color)  # we already handled the x-label with ax1
    ax2.plot(ind, success_rate,'h', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #ax2.set_ylim(-10,110)
    ax2.set_yticks([0, 20,40, 60, 80, 100])

    #axs[idx].set_title('Phase reset')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'phase_convergence_time.svg'
    plt.savefig(figPath)
    plt.show()


def boxplot_phase_convergenceTime_statistic_threeMethod_underRoughness(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0'],trial_ids=[0]):
    '''
    Plot convergence time statistic in three different methods under different roughness, it can indicates the actual traverability of the locomotion

    '''
    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    phi={}
    for category, files_name in titles_files_categories: #category is the thrid columns, files_name is the table
        if category in experiment_categories:
            phi[category]={}
            for control_method, file_name in files_name.groupby('titles'): #control methods
                print("The experiment category is: {} of control method: {} with trial number: {}".format( category, control_method,len(file_name)))
                phi[category][control_method]=[]
                for idx in file_name.index: #遍历某个分类category下的所有数据文件, trials
                    try:
                        if idx in np.array(file_name.index)[trial_ids]:# which one is to load
                            folder_category= data_file_dic + file_name['data_files'][idx]
                            print(folder_category)
                            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                            # 2)  data process
                            phase_con_time=calculate_phase_convergence_time(time,grf_data,cpg_data,freq)
                            if phase_con_time>0:
                                phi[category][control_method].append(phase_con_time)
                                print("Convergence time:{:.2f}".format(phi[category][control_method][-1]))# print the convergence time of each trial
                    except IndexError:
                        print("category 类别数目没有trial_ids 列出的多, 请添加trials")
    
    #3) plot
    figsize=(6,4)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.16,left=0.12,right=0.89)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    Roughness=[int(float(ll)*100) for ll in experiment_categories]
    print(Roughness)
    ind= np.arange(len(experiment_categories))


    #3.1) plot 
    phi_values=[]
    control_methods=list(phi[experiment_categories[0]].keys())
    for idx,control_method in enumerate(control_methods):
        phi_values.append([])
        for category in experiment_categories:
            phi_values[idx].append(phi[category][control_method])
    
    idx=0
    boxwidth=0.2
    box=[]
    colors = ['tab:red', 'tab:green','tab:blue']
    for box_idx in range(len(control_methods)):
        box.append(axs[idx].boxplot(phi_values[box_idx],widths=boxwidth, positions=ind+(box_idx-int(len(control_methods)/2))*boxwidth ,boxprops={'color':'pink','facecolor':colors[box_idx]},vert=True, patch_artist=True,meanline=False,showmeans=False,showfliers=False,notch=False))

    # fill with colors
    #for bplot, color in zip(box,colors[0:len(box)]):
    #    for patch in bplot['boxes']:
    #        patch.set_facecolor(color)

    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    legend_names=['PR' if name=="phase_reset" else 'Tegotae' if name=='phase_modulation' else 'APNC' if name=='apnc' else name for name in control_methods]
    axs[idx].legend([bx['boxes'][0] for bx in box],legend_names[0:len(box)],loc='best')
    axs[idx].set_xticks(ind)
    axs[idx].set_xticklabels(Roughness)
    axs[idx].set_ylabel(r'Phase convergence time [s]')
    axs[idx].set_xlabel(r'Roughness [%]')

    # plot the success_rate with twinx 
    ax2 = axs[idx].twinx()  # instantiate a second axes that shares the same x-axis
    success_rate=[]
    for control_idx in range(len(control_methods)):
        success_rate.append([])
        for idx in range(len(phi_values[control_idx])):
            success_count= np.array(phi_values[control_idx][idx]) < 20.0
            success_rate[control_idx].append(sum(success_count)/len(trial_ids)*100)
            color = 'tab:orange'
    ax2.set_ylabel('Success rate [%]', color=color)  # we already handled the x-label with ax1
    for box_idx in range(len(control_methods)):
        ax2.plot(ind+(box_idx-int(len(control_methods)/2))*boxwidth, success_rate[box_idx],'h',markeredgecolor=color, color=colors[box_idx])
    ax2.tick_params(axis='y', labelcolor=color)
    #ax2.set_ylim(-10,110)
    ax2.set_yticks([0, 20,40, 60, 80, 100])

    #axs[idx].set_title('Phase reset')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime())) + 'phase_convergence_time.svg'
    plt.savefig(figPath)
    plt.show()


def boxplot_phase_convergenceTime_statistic_threeMethod_underMI(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0'],trial_ids=[0],**args):
    '''
    Plot convergence time statistic in three different methods under different MI, it can indicates the actual traverability of the locomotion

    '''
    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    phi={}
    for category, files_name in titles_files_categories: #category is the thrid columns, files_name is the table
        if category in experiment_categories:
            phi[category]={}
            for control_method, file_name in files_name.groupby('titles'): #control methods
                print("The experiment category is: {} of control method: {} with trial number: {}".format( category, control_method,len(file_name)))
                phi[category][control_method]=[]
                for idx in file_name.index: #遍历某个分类category下的所有数据文件, trials
                    try:
                        if idx in np.array(file_name.index)[trial_ids]:# which one is to be loaded
                            folder_category= data_file_dic + file_name['data_files'][idx]
                            print(folder_category)
                            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                            # 2)  data process
                            phase_con_time=calculate_phase_convergence_time(time,grf_data,cpg_data,freq)
                            if phase_con_time>0:
                                phi[category][control_method].append(phase_con_time)
                                print("Convergence time:{:.2f}".format(phi[category][control_method][-1]))# print the convergence time of each trial
                    except IndexError:
                        print("category 类别数目没有trial_ids 列出的多, 请添加trials")
    
    #3) plot
    figsize=(6,4)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.16,left=0.12,right=0.89)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    MI=[ str(ll) for ll in experiment_categories]
    print(MI)
    ind= np.arange(len(experiment_categories))


    #3.1) plot 
    phi_values=[]
    pd_phi_values_list=[]
    try:
        control_methods=list(phi[experiment_categories[0]].keys())
    except KeyError:
        warnings.warn(termcolor.colored("No trials loaded"))

    for idx,control_method in enumerate(control_methods):
        phi_values.append([])
        for category in experiment_categories:
            phi_values[idx].append(phi[category][control_method])
            temp=pd.DataFrame(data=phi[category][control_method],columns=["values"])
            temp.insert(1,'experiment_categories',category)
            temp.insert(2,'control_methods',control_method)
            pd_phi_values_list.append(temp)
    
    pd_phi_values=pd.concat(pd_phi_values_list)
    pd_phi_values=pd_phi_values.rename(columns={'control_methods':'Control methods'})
    pd_phi_values=pd_phi_values.replace('phase_modulation','Tegotae')
    pd_phi_values=pd_phi_values.replace('phase_reset','Phase reset')
    pd_phi_values=pd_phi_values.replace('apnc','APNC')





    if(args['plot_type']=='boxplot'):
        idx=0
        boxwidth=0.2
        box=[]
        for box_idx in range(len(control_methods)):
            box.append(axs[idx].boxplot(phi_values[box_idx],widths=boxwidth, positions=ind+(box_idx-int(len(control_methods)/2))*boxwidth ,vert=True,patch_artist=True,meanline=False,showmeans=False,showfliers=False,notch=False))

        # fill with colors
        colors = ['tab:red', 'tab:green','tab:blue']
        for bplot, color in zip(box,colors[0:len(box)]):
            for patch in bplot['boxes']:
                patch.set_facecolor(color)

        axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
        #axs[idx].set_yticks([0,0.1,0.2,0.3])
        #axs[idx].set(ylim=[-0.01,0.3])
        legend_names=['PR' if name=="phase_reset" else 'Tegotae' if name=='phase_modulation' else 'APNC' if name=='apnc' else name for name in control_methods]
        axs[idx].legend([bx['boxes'][0] for bx in box],legend_names[0:len(box)],loc='best')
        axs[idx].set_xticks(ind)
        axs[idx].set_xticklabels(MI)
        axs[idx].set_ylabel(r'Phase convergence time [s]')
        axs[idx].set_xlabel(r'MI')

        # plot the success_rate with twinx 
        ax2 = axs[idx].twinx()  # instantiate a second axes that shares the same x-axis
        success_rate=[]
        for control_idx in range(len(control_methods)):
            success_rate.append([])
            for idx in range(len(phi_values[control_idx])):
                success_count= np.array(phi_values[control_idx][idx]) < 20.0
                success_rate[control_idx].append(sum(success_count)/len(trial_ids)*100)
                
        color = 'tab:orange'
        ax2.set_ylabel('Success rate [%]', color=color)  # we already handled the x-label with ax1
        for box_idx in range(len(control_methods)):
            ax2.plot(ind+(box_idx-int(len(control_methods)/2))*boxwidth, success_rate[box_idx],'h', markeredgecolor=color,color=colors[box_idx])
        ax2.tick_params(axis='y', labelcolor=color)
        #ax2.set_ylim(-10,110)
        ax2.set_yticks([0, 20,40, 60, 80, 100])

        #axs[idx].set_title('Phase reset')
    
    if(args['plot_type']=='barplot'):
        ## barplot
        idx=0
        boxwidth=0.3
        colors = ['green','blue','red']
        axs[idx]=sns.barplot(x='experiment_categories',y='values',hue='Control methods', data=pd_phi_values,palette=colors,hue_order=['Tegotae','Phase reset','APNC'])
    
        # set art of the plot
        axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
        #axs[idx].set_yticks([0,0.1,0.2,0.3])
        #axs[idx].set(ylim=[-0.01,0.3])
        axs[idx].set_xticks(ind)
        axs[idx].set_xticklabels(MI)
        axs[idx].set_ylabel(r'Phase convergence time [s]')
        axs[idx].set_xlabel(r'MI')



        # plot the success_rate with twinx 
        ax2 = axs[idx].twinx()  # instantiate a second axes that shares the same x-axis
        success_rate=[]
        for control_idx in range(len(control_methods)):
            success_rate.append([])
            for idx in range(len(phi_values[control_idx])):
                success_count= np.array(phi_values[control_idx][idx]) < end_time
                success_rate[control_idx].append(sum(success_count)/len(trial_ids)*100)

        color = 'tab:orange'
        ax2.set_ylabel('Success rate [%]', color=color)  # we already handled the x-label with ax1
        for box_idx in range(len(control_methods)):
            ax2.plot(ind+(box_idx-int(len(control_methods)/2))*boxwidth, success_rate[box_idx],'h', markeredgecolor=color,color=colors[box_idx])
        ax2.tick_params(axis='y', labelcolor=color)
        #ax2.set_ylim(-10,110)
        ax2.set_yticks([0, 20,40, 60, 80, 100])



    if(args['plot_type']=='catplot'):
        ## barplot
        idx=0
        boxwidth=0.3
        colors = ['green','blue','red']
        axs[idx]=sns.catplot(x='experiment_categories',y='values',hue='Control methods', data=pd_phi_values,palette=colors,hue_order=['Tegotae','Phase reset','APNC'])
    
        # set art of the plot
        #axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
        #axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
        #axs[idx].set_yticks([0,0.1,0.2,0.3])
        #axs[idx].set(ylim=[-0.01,0.3])
        #axs[idx].set_xticks(ind)
        #axs[idx].set_xticklabels(MI)
        #axs[idx].set_ylabel(r'Phase convergence time [s]')
        #axs[idx].set_xlabel(r'MI')



        # plot the success_rate with twinx 
        ax2 = axs[idx].twinx()  # instantiate a second axes that shares the same x-axis
        success_rate=[]
        for control_idx in range(len(control_methods)):
            success_rate.append([])
            for idx in range(len(phi_values[control_idx])):
                success_count= np.array(phi_values[control_idx][idx]) < end_time
                success_rate[control_idx].append(sum(success_count)/len(trial_ids)*100)

        color = 'tab:orange'
        ax2.set_ylabel('Success rate [%]', color=color)  # we already handled the x-label with ax1
        for box_idx in range(len(control_methods)):
            ax2.plot(ind+(box_idx-int(len(control_methods)/2))*boxwidth, success_rate[box_idx],'h', markeredgecolor=color,color=colors[box_idx])
        ax2.tick_params(axis='y', labelcolor=color)
        #ax2.set_ylim(-10,110)
        ax2.set_yticks([0, 20,40, 60, 80, 100])


    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime())) + 'phase_convergence_time.svg'
    plt.savefig(figPath)
    plt.show()




def boxplot_phase_convergenceTime_statistic_threeMethod_underUpdateFrequency(data_file_dic,start_time=5,end_time=30,experiment_categories=['0'],trial_ids=[0],**args):
    '''
    Plot convergence time statistic in three different methods under different controlller update frequency, it can indicates the actual traverability of the locomotion

    '''
    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    phi={}
    for category, files_name in titles_files_categories: #category is the thrid columns, files_name is the table
        if category in experiment_categories:
            phi[category]={}
            for control_method, file_name in files_name.groupby('titles'): #control methods
                print("The experiment category is: {} of control method: {} with trial number: {}".format( category, control_method,len(file_name)))
                phi[category][control_method]=[]
                for idx in file_name.index: #遍历某个分类category下的所有数据文件, trials
                    try:
                        if idx in np.array(file_name.index)[trial_ids]:# which one is to load
                            folder_category= data_file_dic + file_name['data_files'][idx]
                            print(folder_category)
                            freq=int(category) # the category is the frequency
                            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                            # 2)  data process
                            phase_con_time=calculate_phase_convergence_time(time,grf_data,cpg_data,freq)
                            if phase_con_time>0:
                                phi[category][control_method].append(phase_con_time)
                                print("Convergence time:{:.2f}".format(phi[category][control_method][-1]))# print the convergence time of each trial
                    except IndexError:
                        print("category 类别数目没有trial_ids 列出的多, 请添加trials")
    
    #3) plot
    figsize=(6,4)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.16,left=0.12,right=0.89)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    MI=[ str(ll) for ll in experiment_categories]
    print(MI)
    ind= np.arange(len(experiment_categories))


    #3.1) plot 
    phi_values=[]
    control_methods=list(phi[experiment_categories[0]].keys())
    pd_phi_values_list=[]
    for idx,control_method in enumerate(control_methods):
        phi_values.append([])
        for category in experiment_categories:
            phi_values[idx].append(phi[category][control_method])
            temp=pd.DataFrame(data=phi[category][control_method],columns=["values"])
            temp.insert(1,'experiment_categories',category)
            temp.insert(2,'control_methods',control_method)
            pd_phi_values_list.append(temp)
    
    pd_phi_values=pd.concat(pd_phi_values_list)
    pd_phi_values=pd_phi_values.rename(columns={'control_methods':'Control methods'})
    pd_phi_values=pd_phi_values.replace('phase_modulation','Tegotae')
    pd_phi_values=pd_phi_values.replace('phase_reset','Phase reset')
    pd_phi_values=pd_phi_values.replace('apnc','APNC')


    if(args['plot_type']=='boxplot'):
        idx=0
        boxwidth=0.2
        box=[]
        ### There are two type of the plot, boxplot and barplot
        ## boxplot
        for box_idx in range(len(control_methods)):
            box.append(axs[idx].boxplot(phi_values[box_idx],widths=boxwidth, positions=ind+(box_idx-int(len(control_methods)/2))*boxwidth ,vert=True,patch_artist=True,meanline=False,showmeans=False,showfliers=False))


        # fill with colors
        colors = ['tab:red', 'tab:green','tab:blue']
        for bplot, color in zip(box,colors[0:len(box)]):
            for patch in bplot['boxes']:
                patch.set_facecolor(color)
        

        # set art of the plot
        axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
        #axs[idx].set_yticks([0,0.1,0.2,0.3])
        #axs[idx].set(ylim=[-0.01,0.3])
        legend_names=['PR' if name=="phase_reset" else 'Tegotae' if name=='phase_modulation' else 'APNC' if name=='apnc' else name for name in control_methods]
        axs[idx].legend([bx['boxes'][0] for bx in box],legend_names[0:len(box)])
        axs[idx].set_xticks(ind)
        axs[idx].set_xticklabels(MI)
        axs[idx].set_ylabel(r'Phase convergence time [s]')
        axs[idx].set_xlabel(r'Update frequency [Hz]')



        # plot the success_rate with twinx 
        ax2 = axs[idx].twinx()  # instantiate a second axes that shares the same x-axis
        success_rate=[]
        for control_idx in range(len(control_methods)):
            success_rate.append([])
            for idx in range(len(phi_values[control_idx])):
                success_count= np.array(phi_values[control_idx][idx]) < end_time
                success_rate[control_idx].append(sum(success_count)/len(trial_ids)*100)

        color = 'tab:orange'
        ax2.set_ylabel('Success rate [%]', color=color)  # we already handled the x-label with ax1
        for box_idx in range(len(control_methods)):
            ax2.plot(ind+(box_idx-int(len(control_methods)/2))*boxwidth, success_rate[box_idx],'h', markeredgecolor=color,color=colors[box_idx])
        ax2.tick_params(axis='y', labelcolor=color)
        #ax2.set_ylim(-10,110)
        ax2.set_yticks([0, 20,40, 60, 80, 100])

        #axs[idx].set_title('Phase reset')

    if(args['plot_type']=='barplot'):
        ## barplot
        idx=0
        boxwidth=0.3
        colors = ['lightgreen','lightblue','pink']
        axs[idx]=sns.barplot(x='experiment_categories',y='values',hue='Control methods', data=pd_phi_values,palette=colors,ci='sd',hue_order=['Tegotae','Phase reset','APNC'])
    
        # set art of the plot
        axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
        #axs[idx].set_yticks([0,0.1,0.2,0.3])
        #axs[idx].set(ylim=[-0.01,0.3])
        axs[idx].set_xticks(ind)
        axs[idx].set_xticklabels(MI)
        axs[idx].set_ylabel(r'Phase convergence time [s]')
        axs[idx].set_xlabel(r'Update frequency [Hz]')



        # plot the success_rate with twinx 
        ax2 = axs[idx].twinx()  # instantiate a second axes that shares the same x-axis
        success_rate=[]
        for control_idx in range(len(control_methods)):
            success_rate.append([])
            for idx in range(len(phi_values[control_idx])):
                success_count= np.array(phi_values[control_idx][idx]) < end_time
                success_rate[control_idx].append(sum(success_count)/len(trial_ids)*100)

        color = 'tab:orange'
        ax2.set_ylabel('Success rate [%]', color=color)  # we already handled the x-label with ax1
        for box_idx in range(len(control_methods)):
            ax2.plot(ind+(box_idx-int(len(control_methods)/2))*boxwidth, success_rate[box_idx],'h', markeredgecolor=color,color=colors[box_idx])
        ax2.tick_params(axis='y', labelcolor=color)
        #ax2.set_ylim(-10,110)
        ax2.set_yticks([0, 20,40, 60, 80, 100])

        #axs[idx].set_title('Phase reset')
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime())) + 'phase_convergence_time.svg'
    plt.savefig(figPath)
    plt.show()

def WalkingSpeed_GaitDiagram(data_file_dic,start_time=60,end_time=900,freq=60.0,experiment_categories=['0.0'],trial_ids=[0]):
    '''
    @description: this is for plot the gait diagram under the different walking speeds
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_ids, the trials that are included to computate
    @return: show and save a data figure.

    '''
    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    speed={}
    gait_diagram_data={}
    for category, files_name in titles_files_categories: #category is the thrid columns, files_name is the table
        if category in experiment_categories: # 仅仅读取入口参数指定的实验类别数据
            speed[category]=[]
            gait_diagram_data[category]=[]
            control_method=files_name['titles'].iat[0]
            print("The experiment category:", category)
            for idx in files_name.index: #遍历某个分类category下的所有数据文件
                if idx in np.array(files_name.index)[trial_ids]:# 仅仅读取指定trial_id的数据
                    folder_category= data_file_dic + files_name['data_files'][idx]
                    print(folder_category)
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    displacement= calculate_displacement(pose_data)
                    speed[category].append(displacement/(time[-1]-time[0]))
                    gait_diagram_data_temp, beta_temp = gait(grf_data)
                    gait_diagram_data[category].append(gait_diagram_data_temp);
                    #beta[category].append(beta_temp)
                    print(speed[category][-1])
                    
                

    #3) plot
    axs_column_num=2 # how many columns axis in the figure
    figsize=(2.1*5,2.97*5+0*len(experiment_categories)/axs_column_num)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(int(len(experiment_categories)/axs_column_num),axs_column_num)#13
    gs1.update(hspace=0.46,top=0.95,bottom=0.05,left=0.1,right=0.96)
    axs=[]
    for idx in range(int(len(experiment_categories)/axs_column_num)):
        axs.append(fig.add_subplot(gs1[idx:idx+1,0]))
        axs.append(fig.add_subplot(gs1[idx:idx+1,1]))

    for idx in range(len(experiment_categories)):
        experiment_category=experiment_categories[idx]
        trial_id=trial_ids[0]
        axs[idx].set_ylabel(r'Gait')
        axs[idx].yaxis.set_label_coords(-0.08,.5)
        gait_diagram(fig,axs[idx],gs1,gait_diagram_data[experiment_category][trial_id])
        axs[idx].set_xlabel(u'Time [s]')
        axs[idx].xaxis.set_label_coords(0.5,-0.11)
        xticks=np.arange(int(min(time)),int(max(time))+1,1)
        axs[idx].set_xticks(xticks)
        axs[idx].set_xticklabels([str(xtick) for xtick in xticks])
        axs[idx].set(xlim=[min(time),max(time)])
        axs[idx].set_title('MI: '+experiment_category+'; Walking speed: '+str(round(speed[experiment_category][trial_id]*100,1))+" cm/s")

    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'WalkingSpeed_GaitDiagram.svg'
    plt.savefig(figPath)
    plt.show()


def g_VideoText(data_file_dic,start_time=60,end_time=900,freq=60.0,experiment_categories=['0.0'],trial_ids=[0]):
    '''
    @description: this is for plot the g term of the neural couplings
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_ids, the trials that are included to computate
    @return: show and save a data figure.

    '''
    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    g_data={}
    for category, files_name in titles_files_categories: #category is the thrid columns, files_name is the table
        if category in experiment_categories: # 仅仅读取入口参数指定的实验类别数据
            g_data[category]=[]
            control_method=files_name['titles'].iat[0]
            print("The experiment category:", category)
            for idx in files_name.index: #遍历某个分类category下的所有数据文件
                if idx in np.array(files_name.index)[trial_ids]:# 仅仅读取指定trial_id的数据
                    folder_category= data_file_dic + files_name['data_files'][idx]
                    print(folder_category)
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    g_data[category].append(module_data)
                    
    # plot
    figsize=(7.2*3,2.5)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(2,1)#13
    gs1.update(hspace=0.22,top=0.96,bottom=0.18,left=0.11,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))

    colorList=['r','g','b','k','y','c','m']
    labels= titles_files_categories.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]

    for index, class_name in enumerate(labels): # name is a inclination names
        columnsName_modules=['ss','f1','f2','f3','f4','f5','f6','f7','f8','g1','g2','g3','g4','g5','g6','g7','g8','FM1','FM2','gama1','gamma2','phi_12','phi_13','phi_14']
        g_data_df=pd.DataFrame(g_data[class_name][0],columns=columnsName_modules)
        idx=0
        xdata, g1_ydata = [], []
        g2_ydata=[]; 
        g3_ydata=[];
        g4_ydata=[];
        ln1, = axs[idx].plot([], [], 'r', linewidth=1,markersize=0.5)
        ln2, = axs[idx].plot([], [], 'g', linewidth=1,markersize=0.5)
        ln3, = axs[idx].plot([], [], 'b', linewidth=1,markersize=0.5)
        ln4, = axs[idx].plot([], [], 'y', linewidth=1,markersize=0.5)

        def init():
            axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
            axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
            axs[idx].set_ylabel(r'$f_{1,2}$ and $g_{1,2}$ of the RF leg')
            axs[idx].legend([r'$f_1$',r'$f_2$',r'$g_1$',r'$g_2$'],ncol=4,loc='upper left')
            axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.04,0.04])
            yticks= [-0.04,-0.02,0.0,0.02,0.04]
            axs[idx].set_yticks(yticks)
            axs[idx].set_yticklabels([str(ytick) for ytick in yticks])
            xticks=np.arange(int(min(time)),int(max(time))+1,5)
            #xticks=np.arange(0,int(max(time)),10)
            axs[idx].set_xticklabels([str(xtick) for xtick in xticks])
            axs[idx].set_xticks(xticks)
            axs[idx].set_xlabel('Time [s]')
            return ln1,ln2,ln3,ln4

        def update(frame):
            xdata.append(time[frame])
            g1_ydata.append(g_data_df.iat[frame,1])
            g2_ydata.append(g_data_df.iat[frame,2])
            g3_ydata.append(g_data_df.iat[frame,9])
            g4_ydata.append(g_data_df.iat[frame,10])
            ln1.set_data(xdata, g1_ydata)
            ln2.set_data(xdata, g2_ydata)
            ln3.set_data(xdata, g3_ydata)
            ln4.set_data(xdata, g4_ydata)
            return ln1,ln2, ln3,ln4


        #axs[idx].plot(time,average_average_gamma,color=colorList[index])
        ani = FuncAnimation(fig, update, frames=range(0, len(time)-1, 1), init_func=init, interval=50,blit=True)
        #plt.show()


        #save animation
        folder_fig = data_file_dic + 'data_visulization/'
        if not os.path.exists(folder_fig):
            os.makedirs(folder_fig)
        aniPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) +str(class_name) + '.mp4'
        ani.save(aniPath,dpi=360)
        plt.cla()

def APC_ANC_plots(data_file_dic,start_time=60,end_time=900,freq=60.0,experiment_categories=['0.0'],trial_ids=[0]):
    '''
    @description: this is for plot the g, f term, gamma, xi, ...  term of the adaptive physical and neural couplings
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_ids, the trials that are included to computate
    @return: show and save a data figure.

    '''
    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    g_data={}
    f_data={}
    gamma_data={}
    xi_data={}
    phi_data={}
    cpg={}
    grf={}
    for category, files_name in titles_files_categories: #category is the thrid columns, files_name is the table
        if category in experiment_categories: # 仅仅读取入口参数指定的实验类别数据
            g_data[category]=[]
            f_data[category]=[]
            gamma_data[category]=[]
            xi_data[category]=[]
            phi_data[category]=[]
            cpg[category]=[]
            grf[category]=[]
            control_method=files_name['titles'].iat[0]
            print("The experiment category:", category)
            for idx in files_name.index: #遍历某个分类category下的所有数据文件
                if idx in np.array(files_name.index)[trial_ids]:# 仅仅读取指定trial_id的数据
                    folder_category= data_file_dic + files_name['data_files'][idx]
                    print(folder_category)
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    f_data[category].append(module_data[:,1:9])
                    g_data[category].append(module_data[:,9:17])
                    gamma_data[category].append(module_data[:,19:21])
                    phi_data[category].append(module_data[:,21:])
                    cpg[category].append(cpg_data)
                    grf[category].append(grf_data)
                    xi=np.zeros(len(module_data[:,9]))
                    xi[np.where(module_data[:,9]>0)[0][0]:]=0.01
                    xi_data[category].append(xi)

                    



    #2) Whether get right data
    for exp_idx in range(len(experiment_categories)):
        for trial_id in range(len(trial_ids)):
            experiment_category=experiment_categories[exp_idx]# The first category of the input parameters (arg)
            if not cpg[experiment_category]:
                warnings.warn('Without proper data was read')
            #3) plot
            figsize=(21/3,24/3)
            fig = plt.figure(figsize=figsize,constrained_layout=False)
            plot_column_num=1# the columns of the subplot. here set it to one
            gs1=gridspec.GridSpec(16,plot_column_num)#13
            gs1.update(hspace=0.18,top=0.95,bottom=0.08,left=0.1,right=0.98)
            axs=[]
            for idx in range(plot_column_num):# how many columns, depends on the experiment_categories
                axs.append(fig.add_subplot(gs1[0:3,idx]))
                axs.append(fig.add_subplot(gs1[3:6,idx]))
                axs.append(fig.add_subplot(gs1[6:8,idx]))
                axs.append(fig.add_subplot(gs1[8:10,idx]))
                axs.append(fig.add_subplot(gs1[10:12,idx]))
                axs.append(fig.add_subplot(gs1[12:14,idx]))
                axs.append(fig.add_subplot(gs1[14:16,idx]))

            c4_1color=(46/255.0, 77/255.0, 129/255.0)
            c4_2color=(0/255.0, 198/255.0, 156/255.0)
            c4_3color=(255/255.0, 1/255.0, 118/255.0)
            c4_4color=(225/255.0, 213/255.0, 98/255.0)
            colors=[c4_1color, c4_2color, c4_3color, c4_4color]


            idx=0# CPGs
            cpg_diagram(fig,axs[idx],gs1,cpg[experiment_category][trial_id],time)
            axs[idx].set_ylabel(u'CPGs')
            axs[idx].yaxis.set_label_coords(-0.08,.5)
            xticks=np.arange(int(min(time)),int(max(time))+1,5)
            axs[idx].set_xticks(xticks)
            axs[idx].set_xticklabels([])
            axs[idx].set(xlim=[min(time),max(time)])

            idx=1# GRFs
            grf_diagram(fig,axs[idx],gs1,grf[experiment_category][trial_id],time)
            axs[idx].set_ylabel(u'GRFs')
            axs[idx].yaxis.set_label_coords(-0.08,.5)
            xticks=np.arange(int(min(time)),int(max(time))+1,5)
            axs[idx].set_xticks(xticks)
            axs[idx].set_xticklabels([])
            axs[idx].set(xlim=[min(time),max(time)])


            #plt.subplots_adjust(hspace=0.4)
            idx=2
            axs[idx].set_ylabel(u'$\Phi$ [rad]')
            axs[idx].plot(time, phi_data[experiment_category][trial_id][:,0],color=(77/255,133/255,189/255))
            axs[idx].plot(time, phi_data[experiment_category][trial_id][:,1],color=(247/255,144/255,61/255))
            axs[idx].plot(time, phi_data[experiment_category][trial_id][:,2],color=(89/255,169/255,90/255))
            #axs[idx].plot(time, phi_std,color='k')
            axs[idx].legend([u'$\phi_{12}$',u'$\phi_{13}$',u'$\phi_{14}$'],ncol=3,loc='upper left')
            axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
            axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
            axs[idx].set(ylim=[-0.2,3.5])
            axs[idx].set_yticks([0.0,1.5,3.0])
            xticks=np.arange(int(min(time)),int(max(time))+1,5)
            axs[idx].set_xticks(xticks)
            axs[idx].set_xticklabels([])
            axs[idx].set(xlim=[min(time),max(time)])

            idx=3# parameters
            axs[idx].set_ylabel(r'$\gamma$')
            axs[idx].plot(time, gamma_data[experiment_category][trial_id][:,0],color=colors[0])
            axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
            axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
            axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.01,0.52])
            yticks= [0.0,0.2,0.4,0.5]
            axs[idx].set_yticks(yticks)
            axs[idx].set_yticklabels([str(ytick) for ytick in yticks])
            xticks=np.arange(int(min(time)),int(max(time))+1,5)
            axs[idx].set_xticks(xticks)
            axs[idx].set_xticklabels([])
            #axs[idx].legend([r'$\gamma$'],ncol=1,loc='upper left')


            idx=4# parameters
            #axs[idx].set_ylabel(u'$\xi$')
            axs[idx].set_ylabel(r'$\xi$')
            axs[idx].plot(time, xi_data[experiment_category][trial_id],color=colors[0])
            axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
            axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
            axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.001,0.012])
            yticks= [0.0,0.01]
            axs[idx].set_yticks(yticks)
            axs[idx].set_yticklabels([str(ytick) for ytick in yticks])
            xticks=np.arange(int(min(time)),int(max(time))+1,5)
            axs[idx].set_xticks(xticks)
            axs[idx].set_xticklabels([])
            #axs[idx].legend([r'$\xi$'],ncol=1,loc='upper left')

            idx=5# f and g
            axs[idx].set_ylabel(u'$f_i,g_i$')
            axs[idx].plot(time, f_data[experiment_category][trial_id][:,0],color=colors[0])
            axs[idx].plot(time, f_data[experiment_category][trial_id][:,1],color=colors[1])
            axs[idx].plot(time, g_data[experiment_category][trial_id][:,0],color=colors[2])
            axs[idx].plot(time, g_data[experiment_category][trial_id][:,1],color=colors[3])
            axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
            axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
            axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.02,0.031])
            yticks= [-0.02,0.0,0.013,0.03]
            axs[idx].set_yticks(yticks)
            axs[idx].set_yticklabels([str(ytick) for ytick in yticks])
            xticks=np.arange(int(min(time)),int(max(time))+1,5)
            axs[idx].set_xticks(xticks)
            axs[idx].set_xticklabels([])
            axs[idx].legend([r'$f_1$',r'$f_2$',r'$g_1$',r'$g_2$'],ncol=4,loc='upper left')


            idx=6
            axs[idx].set_ylabel(r'Gait')
            gait_diagram(fig,axs[idx],gs1,grf[experiment_category][trial_id])
            axs[idx].set_xlabel(u'Time [s]')
            xticks=np.arange(int(min(time)),int(max(time))+1,5)
            axs[idx].set_xticks(xticks)
            axs[idx].set_xticklabels([str(xtick) for xtick in xticks])
            axs[idx].yaxis.set_label_coords(-0.065,.5)
            axs[idx].set(xlim=[min(time),max(time)])

            # save figure
            folder_fig = data_file_dic + 'data_visulization/'
            if not os.path.exists(folder_fig):
                os.makedirs(folder_fig)

            figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'general_display.svg'
            plt.savefig(figPath)
            plt.show()
            plt.close()




def plot_phase_shift_dynamics(data_file_dic,start_time=90,end_time=1200,freq=60.0,experiment_categories=['0.0'],trial_ids=[0],control_methods=['apnc']):
    ''' 
    @description: This is for plot CPG phase shift dynamics
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_ids, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''

    # 1) read data
    titles_files_categories=load_data_log(data_file_dic)
    cpg={}
    for category, files_name in titles_files_categories: #category is a files_name categorys
        if category in experiment_categories:
            cpg[category]=[]
            control_method=files_name['titles'].iat[0]
            print(category)
            for idx in files_name.index:
                if idx in np.array(files_name.index)[trial_ids]:# which one is to load
                    folder_category= data_file_dic + files_name['data_files'][idx]
                    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                    # 2)  data process
                    print(folder_category)
                    cpg[category].append(cpg_data)

    #2) plot
    figsize=(8/3*len(experiment_categories),2.1)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    markers=['g*','g*','g*','g*','g*','g*','ks','kp']

    gs1=gridspec.GridSpec(1,2*len(experiment_categories))#2 per column
    gs1.update(hspace=0.1,wspace=0.4,top=0.9,bottom=0.11,left=0.02,right=0.94)
    axs=[]
    for idx in range(len(experiment_categories)):
        axs.append(fig.add_subplot(gs1[0,int(2*idx):int(2*(idx+1))],projection="3d"))

    titles=experiment_categories

    for exp_idx in range(len(experiment_categories)):
        for trial_id in range(len(trial_ids)):
            plot_idx=exp_idx
            try:
                experiment_category=experiment_categories[exp_idx]# The first category of the input parameters (arg)
            except IndexError:
                pdb.set_trace()
            if not cpg[experiment_category]:
                warnings.warn('Without proper data was read')
    
            #3.1) draw
            axs[plot_idx].plot([0],[0],[0],color='red',marker='X')
            axs[plot_idx].plot([3.14],[3.14],[0],color='blue',marker="D")
            phi=calculate_phase_diff(cpg[experiment_category][trial_id],time)
            phi_std=calculate_phase_diff_std(cpg[experiment_category][trial_id],time); 
            axs[plot_idx].plot(phi['phi_12'], phi['phi_13'], phi['phi_14'],markers[exp_idx],markersize='3')
            axs[plot_idx].view_init(12,-62)
            axs[plot_idx].set_xlabel(u'$\phi_{12}$[rad]')
            axs[plot_idx].set_ylabel(u'$\phi_{13}$[rad]')
            axs[plot_idx].set_zlabel(u'$\phi_{14}$[rad]')# specifying the distance betwwen the label and the axis
            #axs[plot_idx].xaxis._axinfo['label']['space_factor'] = 1
            #axs[plot_idx].yaxis._axinfo['label']['space_factor'] = 1
            axs[plot_idx].dist=15
            axs[plot_idx].set_xlim([-0.1,3.2])
            axs[plot_idx].set_ylim([-0.1,3.2])
            axs[plot_idx].set_zlim([-0.1,2.2])
            axs[plot_idx].set_xticks([0,1,2,3])
            axs[plot_idx].set_yticks([0,1,2,3])
            axs[plot_idx].set_zticks([0,1,2])
            axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
            axs[plot_idx].grid(which='both',axis='y',color='k',linestyle=':')
            axs[plot_idx].grid(which='both',axis='z',color='k',linestyle=':')
            axs[plot_idx].tick_params(axis='x', which='major', pad=-3)# reduce the distance between ticklabels and the axis
            axs[plot_idx].tick_params(axis='y', which='major', pad=-3)# reduce the distance between ticklabels and the axis
            axs[plot_idx].tick_params(axis='z', which='major', pad=-3)# reduce the distance between ticklabels and the axis

            if control_method=="PhaseReset":
                axs[plot_idx].set_title(u"$F_t=$"+titles[plot_idx])
            if control_method=="PhaseModulation":
                axs[plot_idx].set_title(u"R="+str(int(float(titles[plot_idx])*100))+r"%")
                #axs[plot_idx].set_title(u"MI="+titles[plot_idx])


    
    #axs[plot_idx].legend(experiment_categories)


    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'phase_shift_dynamics.svg'
    plt.savefig(figPath)
    plt.show()
    '''
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'general_display.svg'
    #plt.savefig(figPath)
    

    figsize=(6,6)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    x=phi_std
    y=np.gradient(phi_std,1.0/freq)
    plt.plot(0,0,'bo')
    plt.plot(x,y,'ro',markersize=1.5)

    figsize=(6,6)
    fig1 = plt.figure(figsize=figsize,constrained_layout=False)
    x=phi_std
    y=np.gradient(phi_std,1.0/freq)
    dx=2*np.sign(np.gradient(x,1.0/freq))
    dy=2*np.sign(np.gradient(y,1.0/freq))

    plt.plot(0,0,'bo')
    plt.quiver(x,y,dx,dy,angles='xy',color='r')
    plt.xlim([-0.1,4])
    plt.ylim([-0.1,4])
    plt.show()
    '''



def plot_phase_shift_dynamics_underThreeMethods(data_file_dic,start_time=5,end_time=40,freq=60.0,experiment_categories=['0.0'],trial_ids=[0],control_methods='apnc',investigation='update_frequency'):
    ''' 
    @description: This is for plot CPG phase shift dynamics
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data, unit : seconds
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: trial_ids, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''

    # 1) read data
    freq=freq
    titles_files_categories=load_data_log(data_file_dic)
    cpg={}
    phi={}
    phi_std={}
    for category, files_name in titles_files_categories: #category is a files_name categorys
        if category in experiment_categories:
            cpg[category]=[]
            phi[category]=[]
            phi_std[category]=[]
            for control_method, file_name in files_name.groupby('titles'): #control methods
                if(control_method in control_methods): # which control methoid is to be display
                    print("The experiment category: ", category, "control method is: ", control_method)
                    for idx in files_name.index: # trials for display here
                        if idx in np.array(files_name.index)[trial_ids]:# which one is to load
                            folder_category= data_file_dic + files_name['data_files'][idx]
                            if investigation=='update_frequency':
                                freq=int(category) # the category is the frequency
                            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_time,end_time,folder_category)
                            # 2)  data process
                            print(folder_category)
                            print("Convergence time:{:.2f}".format(calculate_phase_convergence_time(time,grf_data,cpg_data,freq)))
                            cpg[category].append(cpg_data)
                            phi[category].append(calculate_phase_diff(cpg_data,time))
                            phi_std[category].append(calculate_phase_diff_std(cpg_data,time)) 

    #2) plot
    if investigation=='update_frequency':
        figsize=(len(experiment_categories),3)
        fig = plt.figure(figsize=figsize,constrained_layout=False)

        gridspec_column=int(math.ceil(len(experiment_categories)/2))
        gridspec_row=2
        gs1=gridspec.GridSpec(gridspec_row,gridspec_column)#2 per column, experiment_category should be even
        gs1.update(hspace=0.01,wspace=0.01,top=0.9,bottom=0.11,left=0.02,right=0.94)
    axs=[]
    for row_idx in range(gridspec_row):
        for col_idx in range(gridspec_column):
            axs.append(fig.add_subplot(gs1[row_idx,col_idx:col_idx+1],projection="3d"))

    markers=['g*','g*','g*','g*','g*','g*']*3
    titles=experiment_categories
    control_method=control_methods[0]

    for exp_idx, category in enumerate(experiment_categories):
        for trial_idx in range(len(trial_ids)):
            plot_idx=exp_idx
            if not cpg[category]:
                warnings.warn('Without proper data was read')
    
            #3.1) draw
            axs[plot_idx].plot([0],[0],[0],color='red',marker='X')
            axs[plot_idx].plot([3.14],[3.14],[0],color='blue',marker="D")
            axs[plot_idx].plot(phi[category][trial_idx]['phi_12'], phi[category][trial_idx]['phi_13'], phi[category][trial_idx]['phi_14'],markers[exp_idx],markersize='3')
            axs[plot_idx].view_init(12,-62)
            axs[plot_idx].set_xlabel(u'$\phi_{12}$[rad]')
            axs[plot_idx].set_ylabel(u'$\phi_{13}$[rad]')
            axs[plot_idx].set_zlabel(u'$\phi_{14}$[rad]')# specifying the distance betwwen the label and the axis
            #axs[plot_idx].xaxis._axinfo['label']['space_factor'] = 1
            #axs[plot_idx].yaxis._axinfo['label']['space_factor'] = 1
            axs[plot_idx].dist=15
            axs[plot_idx].set_xlim([-0.1,3.2])
            axs[plot_idx].set_ylim([-0.1,3.2])
            axs[plot_idx].set_zlim([-0.1,2.2])
            axs[plot_idx].set_xticks([0,1,2,3])
            axs[plot_idx].set_yticks([0,1,2,3])
            axs[plot_idx].set_zticks([0,1,2])
            axs[plot_idx].grid(which='both',axis='x',color='k',linestyle=':')
            axs[plot_idx].grid(which='both',axis='y',color='k',linestyle=':')
            axs[plot_idx].grid(which='both',axis='z',color='k',linestyle=':')
            axs[plot_idx].tick_params(axis='x', which='major', pad=-3)# reduce the distance between ticklabels and the axis
            axs[plot_idx].tick_params(axis='y', which='major', pad=-3)# reduce the distance between ticklabels and the axis
            axs[plot_idx].tick_params(axis='z', which='major', pad=-3)# reduce the distance between ticklabels and the axis
            
            
            if investigation == "update_frequency":
                axs[plot_idx].set_title(u"$f=$"+titles[plot_idx]+" Hz",y=0.8)
            elif investigation in ["PhaseReset","phase_reset"]:
                axs[plot_idx].set_title(u"$F_t=$"+titles[plot_idx],y=0.8)
            elif investigation in ["PhaseModulation","phase_modulation",'apnc']:
                axs[plot_idx].set_title(u"R="+str(int(float(titles[plot_idx])*100))+r"%",y=0.8)
                #axs[plot_idx].set_title(u"MI="+titles[plot_idx])


    
    #axs[plot_idx].legend(experiment_categories)


    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime())) + 'phase_shift_dynamics.svg'
    plt.savefig(figPath)
    plt.show()
    '''
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'general_display.svg'
    #plt.savefig(figPath)
    

    figsize=(6,6)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    x=phi_std
    y=np.gradient(phi_std,1.0/freq)
    plt.plot(0,0,'bo')
    plt.plot(x,y,'ro',markersize=1.5)

    figsize=(6,6)
    fig1 = plt.figure(figsize=figsize,constrained_layout=False)
    x=phi_std
    y=np.gradient(phi_std,1.0/freq)
    dx=2*np.sign(np.gradient(x,1.0/freq))
    dy=2*np.sign(np.gradient(y,1.0/freq))

    plt.plot(0,0,'bo')
    plt.quiver(x,y,dx,dy,angles='xy',color='r')
    plt.xlim([-0.1,4])
    plt.ylim([-0.1,4])
    plt.show()
    '''


def plot_all_metrics(data_file_dic, start_time, end_time, freq, experiment_categories, trial_ids, control_methods,**args):
    '''
    Compare and Plot the metrics of different experiment catogries and experimental titles (i.e., control_methods)

    '''
    #1) calculate metrics
    experiment_data, metrics=metrics_calculatiions(data_file_dic, start_time, end_time, freq, experiment_categories, trial_ids=trial_ids, control_methods=control_methods)
    
    #2) tranfer metrics in dict into pandas Dataframe
    pd_metrics_list=[]
    for experiment_category_key, categories in metrics.items():
        for control_method_key, metrics_value in categories.items():
            temp=pd.DataFrame(metrics_value)
            temp['experiment_categories']=experiment_category_key
            temp['control_methods']=control_method_key
            pd_metrics_list.append(temp)

    pd_metrics=pd.concat(pd_metrics_list)

    pd_metrics.loc[pd_metrics['experiment_categories']=='normal_situation','experiment_categories']='S1'
    pd_metrics.loc[pd_metrics['experiment_categories']=='noisy_feedback','experiment_categories']='S2'
    pd_metrics.loc[pd_metrics['experiment_categories']=='leg_damage','experiment_categories']='S3'
    pd_metrics.loc[pd_metrics['experiment_categories']=='carrying_payload','experiment_categories']='S4'

    #3) plot
    figsize=(10,7)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(4,2)#13
    gs1.update(hspace=0.35,top=0.95,bottom=0.12,left=0.1,right=0.92)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))
    axs.append(fig.add_subplot(gs1[2:4,0]))
    axs.append(fig.add_subplot(gs1[0:2,1]))
    axs.append(fig.add_subplot(gs1[2:4,1]))
    #axs.append(fig.add_subplot(gs1[0:2,2]))
    #axs.append(fig.add_subplot(gs1[2:4,2]))


    test_method="Mann-Whitney"
    order=['S1','S2','S3','S4']
    hue_order=['apnc','phase_modulation','phase_reset']
    pairs=(
        [('S1','apnc'),('S1','phase_modulation')],
        [('S1','apnc'),('S1','phase_reset')],

        [('S2','apnc'),('S2','phase_modulation')],
        [('S2','apnc'),('S2','phase_reset')],

        [('S3','apnc'),('S3','phase_modulation')],
        [('S3','apnc'),('S3','phase_reset')],

        [('S4','apnc'),('S4','phase_modulation')],
        [('S4','apnc'),('S4','phase_reset')]
          )
    axs_id=0

    x='experiment_categories'; y='displacement'
    states_palette = sns.color_palette("YlGnBu", n_colors=5)
    hue_plot_params = {
    'data': pd_metrics,
    'x': x,
    'y': y,
    "order": order,
    "hue": "control_methods",
    "hue_order": hue_order
    #"palette": states_palette
    }

    sns.barplot(ax=axs[axs_id],**hue_plot_params)
    axs[axs_id].set_ylabel('Displacement [m]')

    annotator=Annotator(axs[axs_id],pairs=pairs,**hue_plot_params)
    annotator.configure(test=test_method, text_format='star', loc='inside')
    annotator.apply_and_annotate()


    axs_id=1
    x='experiment_categories'; y='balance'
    sns.barplot(ax=axs[axs_id],hue='control_methods', x=x, y=y, order=order,data=pd_metrics)
    axs[axs_id].set_ylabel('Balance [$rad^{-1}$]')

    annotator=Annotator(axs[axs_id],pairs=pairs,**hue_plot_params)
    annotator.configure(test=test_method, text_format='star', loc='inside')
    annotator.apply_and_annotate()

    axs_id=2
    x='experiment_categories'; y='coordination'
    sns.barplot(ax=axs[axs_id],hue='control_methods', x=x, y=y, order=order, data=pd_metrics)
    axs[axs_id].set_ylabel('Coordination')

    annotator=Annotator(axs[axs_id],pairs=pairs,**hue_plot_params)
    annotator.configure(test=test_method, text_format='star', loc='inside')
    annotator.apply_and_annotate()

    axs_id=3
    x='experiment_categories'; y='COT'
    sns.barplot(ax=axs[axs_id],hue='control_methods', x=x, y=y, order=order,data=pd_metrics)
    axs[axs_id].set_ylabel('COT [$JKg^{-1}m^{-1}$]')

    annotator=Annotator(axs[axs_id],pairs=pairs,**hue_plot_params)
    annotator.configure(test=test_method, text_format='star', loc='inside')
    annotator.apply_and_annotate()

    #sns.barplot(ax=axs[1],x='experiment_categories', y='distance',hue='control_methods', order=['S1','S2','S3','S4'],  data=pd_metrics)
    #sns.barplot(ax=axs[4],x='experiment_categories', y='stability',hue='control_methods', order=['S1','S2','S3','S4'],data=pd_metrics)

    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'metrics.svg'
    plt.savefig(figPath)



if __name__=="__main__":

    #test_neuralprocessing()
        
    ''' expected and actuall grf comparison'''
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/PhaseModulation/"
    #plot_comparasion_expected_actual_grf_all_leg(data_file_dic,start_time=1,end_time=1000,freq=60.0,experiment_categories=['0'])

    '''   The routines are called'''
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseTransition/Normal/"
    #PhaseAnalysis(data_file_dic, start_time=960, end_time=1560, freq=60.0, experiment_categories = ['-0.2'], trial_id=0)#1440-2160

    #plot_phase_transition_animation(data_file_dic,start_time=90,end_time=1200,freq=60.0,experiment_categories=['3'],trial_id=0)
    #plot_phase_diff(data_file_dic,start_time=240,end_time=2000,freq=60.0,experiment_categories=['0'],trial_id=0)

    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/PhaseModulation/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/PhaseReset/"
    #plot_actual_grf_all_leg(data_file_dic, start_time=100, end_time=2000, freq=60.0, experiment_categories=['3'], trial_id=0)
    ''' Display phase diff among CPGs and the gait diagram'''
    #Phase_Gait(data_file_dic,start_time=240+60,end_time=721+360+60,freq=60.0,experiment_categories=['0'],trial_id=0)

    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseReset/Normal/SingleExperiment/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseReset/Normal/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseReset/AbnormalLeg/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseReset/Payload/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseReset/NoiseFeedback/"

    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseTransition/Normal/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseTransition/AbnormalLeg/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseTransition/Payload/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseTransition/NoiseFeedback/"
    data_file_dic= "/home/suntao/workspace/experiment_data/"

    ''' Experiment I '''
    #GeneralDisplay(data_file_dic,start_time=240,end_time=721+240,freq=60.0,experiment_categories=['-0.2'],trial_id=0)

    ''' Display the general data of the convergence process '''
    data_file_dic= "/home/suntao/workspace/experiment_data/"
    #plot_actual_grf_all_leg(data_file_dic, start_time=0, end_time=2100, freq=60.0, experiment_categories=['0.9'], trial_id=1)

    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/PhaseReset/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/PhaseModulation/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation/PhaseReset/"
    #GeneralDisplay_All(data_file_dic,start_time=120,end_time=1200+900,freq=60.0,experiment_categories=['0.024'],trial_id=0)

    '''EXPERIMENT II'''
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/"
    #plot_runningSuccess_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0.0'])
    #plot_coordination_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['-0.2'])
    #plot_distance_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0'])
    #plot_energyCost_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['-0.2'])
    #plot_COT_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['-0.2'])
    #plot_displacement_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0'])
    #plot_stability_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['-0.2'])
    #phase_formTime_statistic(data_file_dic,start_time=0,end_time=1200+900,freq=60,experiment_categories=['0'])
    #phase_stability_statistic(data_file_dic,start_time=0,end_time=1200+900,freq=60,experiment_categories=['0'])

    #percentage_plot_runningSuccess_statistic(data_file_dic,start_time=1200,end_time=1200+900,freq=60,experiment_categories=['0.0'])
    #boxplot_displacement_statistic(data_file_dic,start_time=1000,end_time=1200+900,freq=60,experiment_categories=['0'])
    #boxplot_stability_statistic(data_file_dic,start_time=1000,end_time=1200+900,freq=60,experiment_categories=['-0.2'])

    '''EXPERIMENT III'''
    #boxplot_phase_formTime_statistic(data_file_dic,start_time=0,end_time=1200+900,freq=60,experiment_categories=['0'])
    #boxplot_phase_stability_statistic(data_file_dic,start_time=0,end_time=1200+700,freq=60,experiment_categories=['0'])
    #boxplot_COT_statistic(data_file_dic,start_time=1000,end_time=1200+700,freq=60,experiment_categories=['-0.2'])


    '''EXPERIMENT IV'''
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/MiddleLoad_PM/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/PhaseReset/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/PhaseModulation/"
    #barplot_GRFs_patterns_statistic(data_file_dic,start_time=400,end_time=1200+900,freq=60,experiment_categories=['0','1','2','3'])

    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation/"
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation_V2/"
    experiment_categories= ['0.0', '0.04', '0.12', '0.2', '0.28', '0.36', '0.4','0.44'] # ['0.0','0.05','0.15','0.25','0.35','0.45','0.55']#phase modulation
    #phaseModulation_parameter_investigation_statistic(data_file_dic,start_time=0,end_time=1200+900,freq=60,experiment_categories=experiment_categories,trial_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])

    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation_V2/"
    data_file_dic= "/home/suntao/workspace/experiment_data/"
    #experiment_categories= ['0.0', '0.09', '0.27', '0.45', '0.64', '0.82', '0.91', '1.0'] #['0.0','0.05','0.15','0.25','0.35','0.45','0.55']#phase reset
    experiment_categories=['0.0']
    #phaseReset_parameter_investigation_statistic(data_file_dic,start_time=0,end_time=1200+900,freq=60,experiment_categories=experiment_categories,trial_ids=range(15))

    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/PhaseReset/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/PhaseModulation/"
    #plot_single_details(data_file_dic,start_time=120,end_time=720,freq=60.0,experiment_categories=['0'],trial_id=[0])


    '''
    Dyanmics of phase shift among CPGs
    PM parameter sets: 
    gain_value in 0.008 0.01 0.012 0.014 0.016 0.018 0.02 0.024
    PR pramater sets: (not divded mg)
    threshold_value in 0.0 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 
    '''

    data_file_dic = "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation_V2/PhaseModulation/"
    data_file_dic= "/home/suntao/workspace/experiment_data/"
    experiment_categories=['0.0']
    #plot_single_details(data_file_dic, start_time=120, end_time=60*40, freq=60, experiment_categories=['0.0'], trial_ids=[0], investigation="paramater investigation")
    #plot_phase_shift_dynamics(data_file_dic,start_time=120,end_time=1900,freq=60.0,experiment_categories=['0.07'],trial_ids=1)
    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation_V2/PhaseReset/"
    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation_V2/PhaseModulation/"
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #experiment_categories= ['0.04', '0.12', '0.2', '0.28', '0.36', '0.4','0.44','0.52','0.6','0.7'] # ['0.0','0.05','0.15','0.25','0.35','0.45','0.55']#phase modulation
    #experiment_categories= ['0.0', '0.09', '0.27', '0.45', '0.64', '0.82', '0.91', '1.0'] #['0.0','0.05','0.15','0.25','0.35','0.45','0.55']#phase reset
    #experiment_categories=['0.04','0.36','0.44']
    experiment_categories=['0.0','0.36','1.2']
    #experiment_categories=['0.0','0.64','1.0']
    #experiment_categories=['0.0','0.64','1.5']
    #experiment_categories=['0.45']

    trial_ids=[0]
    experiment_categories=['1.0']
    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation_V2/PhaseReset/"
    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation_V2/PhaseModulation/"

    #plot_single_details(data_file_dic,start_time=120,end_time=1900-1000-300,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids, investigation="parameter investigation")
    trial_ids=[0]
    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation_V2/PhaseReset/"
    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation_V2/PhaseModulation/"
    experiment_categories=['0.0','0.36','0.4']
    #experiment_categories=['0.0','0.64','1.5']
    #plot_phase_shift_dynamics(data_file_dic,start_time=120,end_time=1900,freq=60.0,experiment_categories=experiment_categories,trial_ids=[0])


    ''' Plot CPG phase portrait and decoupled CPGs with PM/PR initial conditions which is a point in CPG limit cycle when robot dropped on teh ground'''

    #experiment_categories=['0.45']
    trial_ids=[0]
    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation_V2/PhaseReset/"
    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/StatisticData/ParameterInvestigation_V2/PhaseModulation/"
    experiment_categories=['0.0','0.36','1.0']
    #experiment_categories=['0.0','0.64','1.5']
    #plot_cpg_phase_portrait(data_file_dic,start_time=90,end_time=1200,freq=60.0,experiment_categories=experiment_categories,trial_ids=trial_ids)




    '''   Laikago test in real world  '''
    #data_file_dic="/media/suntao/DATA/Research/P2_workspace/Experiments/laikago_real_robot_experiments/laikago_experiment_data/"
    #experiment_categories=['1']
    #trial_ids=[0]
    #plot_single_details(data_file_dic,start_time=60*0,end_time=60*200,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids, investigation="parameter investigation")


    '''   Animate of g (neural coupings and phsyical communication)  '''
    data_file_dic="/media/suntao/DATA/Research/P2_workspace/submission/Revision/revised_version/kdenlive/SourceMeidas/f_g_curves_simulation/"
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #g_VideoText(data_file_dic,start_time=0,end_time=50*60,freq=60.0,experiment_categories=['0.08'],trial_ids=[0])


    ''' A complete figures to show the APC and ANC  '''
    data_file_dic="/media/suntao/DATA/Research/P2_workspace/submission/Revision/revised_version/kdenlive/SourceMeidas/f_g_curves_simulation/"
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #APC_ANC_plots(data_file_dic,start_time=10*60,end_time=45*60,freq=60.0,experiment_categories=['0.08'],trial_ids=[0])





    '''   PLOT For P2 '''

    ''' Various roughness '''
    data_file_dic= "/home/suntao/workspace/experiment_data/"
    data_file_dic= "/home/suntao/roughness_data/"
    data_file_dic= "/media/suntao/DATA/Research/P2_workspace/Experiments/Experiment_data/SupplementaryExperimentData/roughness_data/"
    data_file_dic= "/media/suntao/DATA/Onedrive/Researches/Papers_and_Thesis/P2_workspace/Experiments/Experiment_data/SupplementaryExperimentData/roughness_data/"
    trial_ids=[4]
    experiment_categories=['1.0']
    #plot_single_details(data_file_dic,start_time=60*10,end_time=60*35,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids, investigation="parameter investigation")
    trial_ids=[1]
    experiment_categories=['0.1','0.2','0.3','0.4','0.5']
    #experiment_categories=['0.6','0.7','0.8','0.9','1.0']
    #plot_phase_shift_dynamics(data_file_dic,start_time=120,end_time=1900,freq=60.0,experiment_categories=experiment_categories,trial_ids=[0])
    trial_ids=[0,1]
    trial_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    experiment_categories=['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
    #boxplot_phase_convergenceTime_sucessRateBar_statistic(data_file_dic,start_time=60*5,end_time=60*40,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids)

    ''' Various MI   '''

    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    data_file_dic= "/media/suntao/DATA/Research/P2_workspace/Experiments/Experiment_data/SupplementaryExperimentData/MI_data/"
    data_file_dic= "/media/suntao/DATA/Onedrive/Researches/Papers_and_Thesis/P2_workspace/Experiments/Experiment_data/SupplementaryExperimentData/MI_data/"
    trial_ids=[0]
    experiment_categories=['0.0','0.04','0.08','0.12','0.16','0.2']
    experiment_categories=['0.18']
    #plot_single_details(data_file_dic,start_time=60*10,end_time=60*35,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids, investigation="parameter investigation")
    #trial_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    experiment_categories=['0.0','0.02','0.04','0.06','0.08','0.1','0.12','0.14','0.16','0.18','0.2','0.22','0.24','0.26','0.28']
    #experiment_categories=['0.0','0.02','0.04','0.06','0.08']
    #experiment_categories=['0.1','0.12','0.14','0.16','0.18']
    #experiment_categories=['0.2','0.22','0.24','0.26','0.28']
    trial_ids=[0]
    trial_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    #plot_phase_shift_dynamics(data_file_dic,start_time=60*15,end_time=60*30,freq=60.0,experiment_categories=experiment_categories,trial_ids=[0])
    #boxplot_phase_convergenceTime_statistic_underMI(data_file_dic,start_time=60*5,end_time=60*40,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids)

    #scatter_dutyFactors_statistic(data_file_dic,start_time=60*14,end_time=60*40,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids)
    #scatter_phaseShift_statistic(data_file_dic,start_time=60*35,end_time=60*40,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids)
    #scatter_COT_statistic(data_file_dic,start_time=60*30,end_time=60*35,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids)

    #experiment_categories=['0.0', '0.04','0.08','0.12','0.16','0.2','0.24','0.28']
    #experiment_categories=['0.0','0.02','0.04','0.06','0.08','0.1','0.12','0.14','0.16','0.18','0.2','0.22','0.24','0.26','0.28','0.30']
    #trial_ids=[0]
    #WalkingSpeed_GaitDiagram(data_file_dic,start_time=60*30,end_time=60*35,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids)

    '''------------------------------------------------------------------------------------------------------------'''
    ''' Various roughness in three diffrent control methods (PR, PM, and APNC), the second round revision of P2,     '''
    ##------- Roughness 

    data_file_dic= "/media/suntao/DATA/Onedrive/Researches/Papers_and_Thesis/P2_workspace/Experiments/Experiment_data/SupplementaryExperimentData/3M_roughness_data/"
    data_file_dic="/media/suntao/DATA/Onedrive/Researches/Papers_and_Thesis/P2_workspace/Experiments/Experiment_data/SupplementaryExperimentData/roughness_data/"
    data_file_dic= "/home/suntao/workspace/experiment_data/"
    data_file_dic= "/media/sun/My Passport/DATA/Researches/Papers/P2_workspace/Experiments/Experiment_data/SupplementaryExperimentData/roughness_data_3M/"

    experiment_categories=['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
    trial_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    #trial_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    #trial_ids=[0,1]
    #boxplot_phase_convergenceTime_statistic_threeMethod_underRoughness(data_file_dic,start_time=5,end_time=30,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids)


    experiment_categories=['1.0']
    control_methods=['apnc']
    #plot_single_details(data_file_dic, start_time=5, end_time=40, freq=60, experiment_categories=experiment_categories, trial_ids=[2], control_methods=control_methods,investigation="paramater investigation")






    ##----- MI

    data_file_dic= "/home/suntao/workspace/experiment_data/"
    data_file_dic="/media/suntao/DATA/MI_3M/"
    data_file_dic= "/media/sun/My Passport/DATA/Researches/Papers/P2_workspace/Experiments/Experiment_data/SupplementaryExperimentData/MI_data_3M/"
    #experiment_categories=['0.0','0.02','0.04','0.06','0.08','0.1','0.12','0.14','0.16','0.18','0.2','0.22','0.24','0.26','0.28']
    experiment_categories=['0.02','0.04','0.06','0.08','0.1','0.12','0.14','0.16','0.18','0.2','0.22','0.24','0.26','0.28']
    #trial_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    #experiment_categories=['0.0','0.04']
    trial_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    trial_ids=[0,1,2]

    #boxplot_phase_convergenceTime_statistic_threeMethod_underMI(data_file_dic,start_time=5,end_time=30,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids,plot_type='catplot')
    #plot_phase_shift_dynamics_underThreeMethods(data_file_dic,start_time=10*60,end_time=1900,freq=60.0,experiment_categories=experiment_categories,trial_ids=[0],control_methods='apnc',investigation='MI')


    ##----- Update frequency

    data_file_dic= "/home/suntao/workspace/experiment_data/"
    data_file_dic= "/media/suntao/DATA/Onedrive/Researches/Papers_and_Thesis/P2_workspace/Experiments/Experiment_data/SupplementaryExperimentData/UpdateFrequency_data_3M/"
    data_file_dic= "/media/sun/DATA/Onedrive/Researches/Papers_and_Thesis/P2_workspace/Experiments/Experiment_data/SupplementaryExperimentData/UpdateFrequency_data_3M/"
    data_file_dic= "/media/sun/My Passport/DATA/Researches/Papers/P2_workspace/Experiments/Experiment_data/SupplementaryExperimentData/UpdateFrequency_data_3M/"

    #data_file_dic="/media/suntao/DATA/UpdateFrequency_3M/"
    experiment_categories=['5','10', '15', '20','25','30','35','40','45','50','55','60']
    #trial_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    #experiment_categories=['0.0','0.04']
    trial_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    #boxplot_phase_convergenceTime_statistic_threeMethod_underUpdateFrequency(data_file_dic,start_time=5,end_time=50,experiment_categories=experiment_categories,trial_ids=trial_ids,plot_type='barplot')

    experiment_categories=['5','10', '15', '20','25','30','35','40','45','50','55','60']
    #plot_phase_shift_dynamics_underThreeMethods(data_file_dic,start_time=5,end_time=40,freq=60.0,experiment_categories=experiment_categories,trial_ids=[0],control_methods=['apnc'],investigation='update_frequency')

    experiment_categories=['15']
    control_methods=['phase_reset']
    trial_ids=[2]
    #plot_single_details(data_file_dic, start_time=5, end_time=50, freq=60, experiment_categories=experiment_categories, trial_ids=trial_ids, control_methods=control_methods,investigation="update_frequency")



    '''------------------------------------------------------------------------------------------------------------'''
    ''' Various robot situations in three diffrent control methods (PR, PM, and APNC), the second round revision of P2,     '''
    ## various robot situations

    data_file_dic= os.environ['EXPERIMENT_DATA_FOLDER']

    control_methods=['apnc','phase_modulation','phase_reset']
    experiment_categories=['normal_situation','noisy_feedback','leg_damage','carrying_payload']
    trial_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    #boxplot_phase_convergenceTime_statistic_threeMethod_underRoughness(data_file_dic,start_time=40,end_time=60,freq=60,experiment_categories=experiment_categories,trial_ids=trial_ids)
    #plot_all_metrics(data_file_dic, start_time=40, end_time=60, freq=60, experiment_categories=experiment_categories, trial_ids=trial_ids,control_methods=control_methods,investigation="paramater investigation")
    
    control_methods=['apnc']
    experiment_categories=['carrying_payload']
    trial_ids=[0]
    plot_single_details(data_file_dic, start_time=40, end_time=60, freq=60, experiment_categories=experiment_categories, trial_ids=trial_ids, control_methods=control_methods, investigation="paramater investigation")
    

