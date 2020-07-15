#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import gridspec
import os
import pdb 
import termcolor
import gnureadline
plt.rc('font',family='Arial')
import pandas as pd
import re
import time as localtimepkg
from brokenaxes import brokenaxes
from mpl_toolkits import mplot3d
from matplotlib import animation
import seaborn as sns

import matplotlib as mpl

'''
###############################
Data loading and preprocessing functions

loadData()
read_data()
load_data_log()

'''
def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/0127113800"):
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


def read_data(freq,start_point,end_point,folder_name):
    '''
    read data from file cut a range data

    '''
    #1) Load data
    fileName_CPGs="controlfile_CPGs"
    fileName_commands='controlfile_commands'
    fileName_modules='controlfile_modules'
    fileName_parameters='parameterfile_rosparameters'
    fileName_joints='sensorfile_joints'

    columnsName_CPGs=['RFO1','RFO2','RHO1','RHO2','LFO1','LFO2','LHO1','LKO2']
    columnsName_GRFs=['RF','RH','LF','LH']
    columnsName_POSEs=['roll','picth','yaw', 'x','y','z','vx','vy','vz']
    columnsName_jointPositions=['p1','p2','p3','p4','p5','p6', 'p7','p8','p9','p10','p11','p12']
    columnsName_jointVelocities=['v1','v2','v3','v4','v5','v6', 'v7','v8','v9','v10','v11','v12']
    columnsName_jointCurrents=['c1','c2','c3','c4','c5','c6', 'c7','c8','c9','c10','c11','c12']
    columnsName_jointVoltages=['vol1','vol2','vol3','vol4','vol5','vol6', 'vol7','vol8','vol9','vol10','vol11','vol12']
    columnsName_modules=['ss','Noise1','Noise2','Noise3','Noise4']
    columnsName_parameters=['MI','CPGBeta','CPGType', \
                            'RF_PSN','RF_VRN_Hip','RF_VRN_Knee','RF_MN1','RF_MN2','RF_MN3',\
                            'RH_PSN','RH_VRN_Hip','RH_VRN_Knee','RH_MN1','RH_MN2','RH_MN3',\
                            'LF_PSN','LF_VRN_Hip','LF_VRN_Knee','LF_MN1','LF_MN2','LF_MN3',\
                            'LH_PSN','LH_VRN_Hip','LH_VRN_Knee','LH_MN1','LH_MN2','LH_MN3'
                           ]
    columnsName_commands=['c1','c2','c3','c4','c5','c6', 'c7','c8','c9','c10','c11','c12']


    columnsName_joints = columnsName_jointPositions + columnsName_jointVelocities + columnsName_jointCurrents + columnsName_jointVoltages + columnsName_POSEs + columnsName_GRFs

    cpg_data=loadData(fileName_CPGs,columnsName_CPGs,folder_name)    
    cpg_data=cpg_data.values

    command_data=loadData(fileName_commands,columnsName_commands,folder_name)    
    command_data=command_data.values

    module_data=loadData(fileName_modules,columnsName_modules,folder_name)    
    module_data=module_data.values

    parameter_data=loadData(fileName_parameters,columnsName_parameters,folder_name)    
    parameter_data=parameter_data.values

    jointsensory_data=loadData(fileName_joints,columnsName_joints,folder_name)    
    grf_data=jointsensory_data[columnsName_GRFs].values
    pose_data=jointsensory_data[columnsName_POSEs].values
    position_data=jointsensory_data[columnsName_jointPositions].values
    velocity_data=jointsensory_data[columnsName_jointVelocities].values
    current_data=jointsensory_data[columnsName_jointCurrents].values
    voltage_data=jointsensory_data[columnsName_jointVoltages].values


    #2) postprecessing 
    read_rows=min([4000000,jointsensory_data.shape[0], cpg_data.shape[0], command_data.shape[0], parameter_data.shape[0], module_data.shape[0]])
    if end_point>read_rows:
        print(termcolor.colored("Warning:end_point out the data bound, please use a small one","yellow"))
    time = np.linspace(int(start_point/freq),int(end_point/freq),end_point-start_point)
    #time = np.linspace(0,int(end_point/freq)-int(start_point/freq),end_point-start_point)
    return cpg_data[start_point:end_point,:], command_data[start_point:end_point,:], module_data[start_point:end_point,:], parameter_data[start_point:end_point,:], grf_data[start_point:end_point,:], pose_data[start_point:end_point,:], position_data[start_point:end_point,:],velocity_data[start_point:end_point,:],current_data[start_point:end_point,:],voltage_data[start_point:end_point,:], time

def load_data_log(data_file_dic):
    '''
    Load data log that stores data files

    '''
    #1.1) load file list 
    data_file_log = data_file_dic +"ExperimentDataLog.log"
    data_files = pd.read_csv(data_file_log, sep='\t',header=None, names=['file_name','experiment_classes'], skip_blank_lines=True,dtype=str)

    datas_of_experiment_classes=data_files.groupby('experiment_classes')
    labels= datas_of_experiment_classes.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    return datas_of_experiment_classes


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

def grf_diagram(fig,axs,gs,grf_data,time):
    '''
    plot ground reaction forces of four legs in curves
    '''
    position=axs.get_position()
    axs.set_yticklabels(labels=[])
    axs.set_yticks([])
    axs.set_xticks([t for t in np.arange(time[0],time[-1]+0.1,2,dtype='int')])
    axs.set_xticklabels(labels=[str(t) for t in np.arange(round(time[0]),round(time[-1])+0.1,2,dtype='int')],fontweight='light')
    axs.set_title("Ground reaction force",loc="left",pad=2)

    ax=stsubplot(fig,position,4,gs)
    LegName=['RF','RH', 'LF', 'LH']
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest',vmin=0.0,vmax=1.0)
    for idx in range(4):
        ax[idx].set_yticks([0.1*(idx+1)])
        ax[idx].plot(grf_data[:,idx])
        ax[idx].set_ylabel(LegName[idx])
        ax[idx].set_yticklabels(labels=[0.0,0.5,1.0])
        ax[idx].set_yticks([0.0,0.5,1.0])
        ax[idx].set_ylim([-0.1,1.1])
        ax[idx].set_xticks([])
        ax[idx].set_xticklabels(labels=[])

def gait_diagram(fig,axs,gs,gait_data):
    '''
    plot gait diagram using while and black block to indicate swing and stance phase
    '''
    position=axs.get_position()
    axs.set_yticklabels(labels=[])
    axs.set_yticks([])
    axs.set_xticks([])
    axs.set_xticklabels(labels=[])
    #axs.set_title("Gait",loc="left",pad=2)

    ax=stsubplot(fig,position,4,gs)
    xx=[]
    LegName=['RF','RH', 'LF', 'LH']
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest',vmin=0.0,vmax=1.0)
    for idx in range(4):
        ax[idx].set_yticks([0.1*(idx+1)])
        xx.append(np.where(gait_data[:,idx]>0.7,1.0,0.0))
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

def EnergyCost(U,I,Fre):
    if ((type(U) is np.ndarray) and (type(I) is np.ndarray)):
        E=sum(U*I*1/Fre)
    else:
        print("input data type is wrong, please use numpy array")
    return E


def COG_distribution(grf_data):
    '''
    division should have a large value, it indicates the foot on the ground.
    '''
    f_front=grf_data[:,0]+grf_data[:,2]
    f_hind=grf_data[:,1]+grf_data[:,3]
    sum_f=np.column_stack([f_front,f_hind])
    #sum_f=sum_f[sum_f[:,0]>0.1] # the front legs shoulb be on stance phase
    #sum_f=sum_f[sum_f[:,1]>0.1] # the hind legs shoulb be on stance phase
    if(len(sum_f)==0):
        print("robot fail down or no stance phase during the period")
        print("simulation is wrong, please redo this simulation")
        gamma=[0]
    else:
        gamma=sum_f[:,0]/sum_f[:,1]
    # set the gamma to zero when robot is fixed in the air
    temp=pd.DataFrame(np.column_stack([sum_f,gamma]),columns=['front','hind','gamma'])
    temp.loc[temp['hind']<0.015,'gamma']=0.0
    gamma=temp['gamma'].to_numpy().reshape((-1,1))
    if(len(grf_data)!=len(gamma)):
        print("COG distribution remove the no stance data")
    return gamma

def Average_COG_distribution(grf_data):
    '''
    calculate the average value of the gamma of during the robot walking
    '''
    data=COG_distribution(grf_data)
    return np.mean(data)

def gait(data):
    ''' Calculating the gait information including touch states and duty factor'''
    # binary the GRF value 
    threshold =0.06
    state=np.zeros(data.shape,int)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if data[j,i] < threshold:
                state[j,i] = 0
            else:
                state[j,i]=1
    
    # get gait info, count the touch and lift number steps
    gait_info=[]
    beta=[]
    state=np.vstack([state,abs(state[-1,:]-1)])
    for i in range(state.shape[1]): #each leg
        count_stance=0;
        count_swing=0;
        number_stance=0
        number_swing=0
        duty_info = {}
                
        for j in range(state.shape[0]-1): #every count
            if state[j,i] ==1:# stance 
                count_stance+=1
            else:#swing
                count_swing+=1

            if (state[j,i]==0) and (state[j+1,i]==1):
                duty_info[str(number_swing)+ "swing"]= count_swing
                count_swing=0
                number_swing+=1
            if (state[j,i]==1) and (state[j+1,i]==0):
                duty_info[str(number_stance) + "stance"]= count_stance
                count_stance=0
                number_stance+=1
            
        gait_info.append(duty_info)
    # calculate the duty factors of all legs
    for i in range(len(gait_info)): # each leg
        beta_singleleg=[]
        for j in range(int(len(gait_info[i])/2)): # each step, ignore the first stance or swing
            if (gait_info[i].__contains__(str(j)+'stance') and  gait_info[i].__contains__(str(j)+'swing')):
                beta_singleleg.append(gait_info[i][str(j)+"stance"]/(gait_info[i][str(j)+"stance"] + gait_info[i][str(j) + "swing"]))
            elif (gait_info[i].__contains__(str(j)+'stance') and  (not gait_info[i].__contains__(str(j)+'swing'))):
                beta_singleleg.append(1.0)
            elif ((not gait_info[i].__contains__(str(j)+'stance')) and  gait_info[i].__contains__(str(j)+'swing')):
                beta_singleleg.append(0.0)
        
        # remove the max and min beta of each legs during the whole locomotion
        if(beta_singleleg!=[]):
            beta_singleleg.remove(max(beta_singleleg))
        if(beta_singleleg!=[]):
            beta_singleleg.remove(min(beta_singleleg))

        beta.append(beta_singleleg)

    return state, beta

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


def plot_phase_transition(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['0.0'],trail_id=0):
    ''' 
    This is for plot the phase diff transition that is shown by a small mive
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_point, the start point (time) of all the data
    @param: end_point, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_classes, the conditions/cases/experiment_classes of the experimental data
    @param: trail_id, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''
    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)

    cpg={}

    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        cpg[class_name]=[]  #files_name is the table of the files_name class_name
        print(class_name)
        for idx in files_name.index:
            folder_class_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_class_name)
            # 2)  data process
            print(folder_class_name)
            cpg[class_name].append(cpg_data)

    #3) plot
    Animate_phase_transition(cpg[class_name][trail_id])
    

def PhaseAnalysis(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['0.0'],trail_id=0):
    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)

    cpg={}
    current={}
    position={}

    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        cpg[class_name]=[]  #files_name is the table of the files_name class_name
        current[class_name]=[]
        position[class_name]=[]
        print(class_name)
        for idx in files_name.index:
            folder_class_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_class_name)
            # 2)  data process
            print(folder_class_name)
            cpg[class_name].append(cpg_data)
            current[class_name].append(current_data)
            position[class_name].append(position_data)

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
    pdb.set_trace()
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


    #axs[idx].plot(time,current_data.iloc[start_point:end_point,26]/1000.0,'r')
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



def GeneralDisplay(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['0.0'],trail_id=0):
    ''' 
    This is for experiment two, for the first figure with one trail on inclination
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_point, the start point (time) of all the data
    @param: end_point, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_classes, the conditions/cases/experiment_classes of the experimental data
    @param: trail_id, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''
    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)

    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    pose={}
    jmc={}
    cpg={}
    noise={}

    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        gamma[class_name]=[]  #files_name is the table of the files_name class_name
        gait_diagram_data[class_name]=[]
        beta[class_name]=[]
        pose[class_name]=[]
        jmc[class_name]=[]
        cpg[class_name]=[]
        noise[class_name]=[]
        print(class_name)
        for idx in files_name.index:
            folder_class_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_class_name)
            # 2)  data process
            print(folder_class_name)
            gamma[class_name].append(COG_distribution(grf_data))

            gait_diagram_data_temp, beta_temp=gait(grf_data)
            gait_diagram_data[class_name].append(gait_diagram_data_temp); beta[class_name].append(beta_temp)

            pose[class_name].append(pose_data)
            jmc[class_name].append(command_data)
            cpg[class_name].append(cpg_data)
            noise[class_name].append(module_data)
            
            temp_1=min([len(bb) for bb in beta_temp]) #minimum steps of all legs
            beta_temp2=np.array([beta_temp[0][:temp_1],beta_temp[1][:temp_1],beta_temp[2][:temp_1],beta_temp[3][0:temp_1]]) # transfer to np array
            if(beta_temp2 !=[]):
                print("Coordination:",1.0/max(np.std(beta_temp2, axis=0)))
            else:
                print("Coordination:",0.0)

            print("Stability:",1.0/np.std(pose_data[:,0],axis=0))
            print("Displacemment:",np.sqrt(pow(pose_data[-1,3]-pose_data[0,3],2)+pow(pose_data[-1,5]-pose_data[0,5],2))) #Displacement on slopes 
    #3) plot
    figsize=(6.2,6.5)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(5,len(experiment_classes))#13
    gs1.update(hspace=0.12,top=0.95,bottom=0.09,left=0.12,right=0.98)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:1,idx]))
        axs.append(fig.add_subplot(gs1[1:2,idx]))
        axs.append(fig.add_subplot(gs1[2:3,idx]))
        axs.append(fig.add_subplot(gs1[3:4,idx]))
        axs.append(fig.add_subplot(gs1[4:5,idx]))
    
    #3.1) plot 

    for idx, inclination in enumerate(experiment_classes):
        axs[3*idx].plot(time,cpg[inclination][trail_id][:,1], color=(46/255.0, 77/255.0, 129/255.0))
        axs[3*idx].plot(time,cpg[inclination][trail_id][:,3], color=(0/255.0, 198/255.0, 156/255.0))
        axs[3*idx].plot(time,cpg[inclination][trail_id][:,5], color=(255/255.0, 1/255.0, 118/255.0))
        axs[3*idx].plot(time,cpg[inclination][trail_id][:,7], color=(225/255.0, 213/255.0, 98/255.0))
        axs[3*idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx].grid(which='both',axis='y',color='k',linestyle=':')
        axs[0].set_ylabel(u'CPGs')
        axs[3*idx].set_yticks([-1.0,0.0,1.0])
        axs[3*idx].legend(['RF','RH','LF', 'LH'],ncol=4)
        axs[3*idx].set_xticklabels([])
        axs[3*idx].set_title('Inclination of the slope:' + str(round(180*float(inclination)/3.1415))+ u'\u00b0')
        axs[3*idx].set(xlim=[min(time),max(time)])


        axs[1].set_ylabel(u'Atti. [deg]')
        axs[3*idx+1].plot(time,pose[inclination][trail_id][:,0]*-57.3,color=(129/255.0,184/255.0,223/255.0))
        axs[3*idx+1].plot(time,pose[inclination][trail_id][:,1]*-57.3,color=(254/255.0,129/255.0,125/255.0))
        #axs[3*idx+1].plot(time,pose[inclination][trail_id][:,2]*57.3,'b')
        axs[3*idx+1].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+1].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+1].set_yticks([-5.0,0.0,5.0])
        axs[3*idx+1].legend(['Roll','Pitch'],loc='upper left')
        axs[3*idx+1].set_xticklabels([])
        axs[3*idx+1].set(xlim=[min(time),max(time)])

        axs[2].set_ylabel(u'Disp. [m]')
        displacement = pose[inclination][trail_id][:,3] #Displacement on slopes 
        axs[3*idx+2].plot(time,displacement,'r')
        axs[3*idx+2].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+2].grid(which='both',axis='y',color='k',linestyle=':')
        #axs[3*idx+2].set_yticks([0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0])
        axs[3*idx+2].set_xticklabels([])
        axs[3*idx+2].set(xlim=[min(time),max(time)])


        axs[3].set_ylabel(u'Phase diff. [rad]')
        phi=calculate_phase_diff(cpg[inclination][trail_id],time)
        axs[3*idx+3].plot(phi['time'],phi['phi_12'],color=(77/255,133/255,189/255))
        axs[3*idx+3].plot(phi['time'],phi['phi_13'],color=(247/255,144/255,61/255))
        axs[3*idx+3].plot(phi['time'],phi['phi_14'],color=(89/255,169/255,90/255))
        axs[3*idx+3].legend([u'$\phi_{12}$',u'$\phi_{13}$',u'$\phi_{14}$'])
        axs[3*idx+3].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+3].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+3].set_yticks([0.0,1.5,3.0])
        axs[3*idx+3].set_xticklabels([])
        axs[3*idx+3].set(xlim=[min(time),max(time)])

        '''
        axs[3].set_ylabel(u'Noise [rad]')
        axs[3*idx+3].plot(time,noise[inclination][trail_id][:,0], 'r')
        axs[3*idx+3].plot(time,noise[inclination][trail_id][:,1], 'g')
        axs[3*idx+3].plot(time,noise[inclination][trail_id][:,2], 'y')
        axs[3*idx+3].plot(time,noise[inclination][trail_id][:,3], 'y')
        axs[3*idx+3].legend(['N1','N2','N3','N4'])
        axs[3*idx+3].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+3].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+3].set_yticks([0.0,0.3])
        axs[3*idx+3].set_xticklabels([])
        axs[3*idx+3].set(xlim=[min(time),max(time)])
        '''

        axs[4].set_ylabel(r'Gait')
        gait_diagram(fig,axs[3*idx+4],gs1,gait_diagram_data[inclination][trail_id])
        axs[3*idx+4].set_xlabel(u'Time [s]')
        xticks=np.arange(int(min(time)),int(max(time))+1,2)
        axs[3*idx+4].set_xticklabels([str(xtick) for xtick in xticks])
        axs[3*idx+4].set_xticks(xticks)
        axs[4].yaxis.set_label_coords(-0.09,.5)
        axs[3*idx+4].set(xlim=[min(time),max(time)])

    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'experiment2_1.svg'
    plt.savefig(figPath)

    plt.show()




def Phase_Gait(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['0.0'],trail_id=0):
    ''' 
    This is for experiment two, for the first figure with one trail on inclination
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_point, the start point (time) of all the data
    @param: end_point, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_classes, the conditions/cases/experiment_classes of the experimental data
    @param: trail_id, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''
    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)

    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    pose={}
    jmc={}
    cpg={}
    noise={}

    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        gamma[class_name]=[]  #files_name is the table of the files_name class_name
        gait_diagram_data[class_name]=[]
        beta[class_name]=[]
        pose[class_name]=[]
        jmc[class_name]=[]
        cpg[class_name]=[]
        noise[class_name]=[]
        print(class_name)
        for idx in files_name.index:
            folder_class_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_class_name)
            # 2)  data process
            print(folder_class_name)
            gamma[class_name].append(COG_distribution(grf_data))

            gait_diagram_data_temp, beta_temp=gait(grf_data)
            gait_diagram_data[class_name].append(gait_diagram_data_temp); beta[class_name].append(beta_temp)

            pose[class_name].append(pose_data)
            jmc[class_name].append(command_data)
            cpg[class_name].append(cpg_data)
            noise[class_name].append(module_data)
            
            temp_1=min([len(bb) for bb in beta_temp]) #minimum steps of all legs
            beta_temp2=np.array([beta_temp[0][:temp_1],beta_temp[1][:temp_1],beta_temp[2][:temp_1],beta_temp[3][0:temp_1]]) # transfer to np array
            if(beta_temp2 !=[]):
                print("Coordination:",1.0/max(np.std(beta_temp2, axis=0)))
            else:
                print("Coordination:",0.0)

            print("Stability:",1.0/np.std(pose_data[:,0],axis=0))
            print("Displacemment:",np.sqrt(pow(pose_data[-1,3]-pose_data[0,3],2)+pow(pose_data[-1,5]-pose_data[0,5],2))) #Displacement on slopes 
    #3) plot
    figsize=(6.2,3.0)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(2,len(experiment_classes))#13
    gs1.update(hspace=0.12,top=0.95,bottom=0.2,left=0.12,right=0.98)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:1,idx]))
        axs.append(fig.add_subplot(gs1[1:2,idx]))
    
    #3.1) plot 

    for idx, inclination in enumerate(experiment_classes):


        axs[0].set_ylabel(u'Phase diff. [rad]')
        phi=calculate_phase_diff(cpg[inclination][trail_id],time)
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
        gait_diagram(fig,axs[3*idx+1],gs1,gait_diagram_data[inclination][trail_id])
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


def Phase_Gait_ForNoiseFeedback(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['0.0'],trail_id=0):
    ''' 
    This is for experiment two, for the first figure with one trail on inclination
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_point, the start point (time) of all the data
    @param: end_point, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_classes, the conditions/cases/experiment_classes of the experimental data
    @param: trail_id, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''
    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)

    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    pose={}
    jmc={}
    cpg={}
    noise={}

    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        gamma[class_name]=[]  #files_name is the table of the files_name class_name
        gait_diagram_data[class_name]=[]
        beta[class_name]=[]
        pose[class_name]=[]
        jmc[class_name]=[]
        cpg[class_name]=[]
        noise[class_name]=[]
        print(class_name)
        for idx in files_name.index:
            folder_class_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_class_name)
            # 2)  data process
            print(folder_class_name)
            gamma[class_name].append(COG_distribution(grf_data-module_data[:,1:]))

            gait_diagram_data_temp, beta_temp=gait(grf_data-module_data[:,1:])
            gait_diagram_data[class_name].append(gait_diagram_data_temp); beta[class_name].append(beta_temp)

            pose[class_name].append(pose_data)
            jmc[class_name].append(command_data)
            cpg[class_name].append(cpg_data)
            noise[class_name].append(module_data)
            
            temp_1=min([len(bb) for bb in beta_temp]) #minimum steps of all legs
            beta_temp2=np.array([beta_temp[0][:temp_1],beta_temp[1][:temp_1],beta_temp[2][:temp_1],beta_temp[3][0:temp_1]]) # transfer to np array
            if(beta_temp2 !=[]):
                print("Coordination:",1.0/max(np.std(beta_temp2, axis=0)))
            else:
                print("Coordination:",0.0)

            print("Stability:",1.0/np.std(pose_data[:,0],axis=0))
            print("Displacemment:",np.sqrt(pow(pose_data[-1,3]-pose_data[0,3],2)+pow(pose_data[-1,5]-pose_data[0,5],2))) #Displacement on slopes 
    #3) plot
    figsize=(6.2,3.0)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(2,len(experiment_classes))#13
    gs1.update(hspace=0.12,top=0.95,bottom=0.2,left=0.12,right=0.98)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:1,idx]))
        axs.append(fig.add_subplot(gs1[1:2,idx]))
    
    #3.1) plot 

    for idx, inclination in enumerate(experiment_classes):


        axs[0].set_ylabel(u'Phase diff. [rad]')
        phi=calculate_phase_diff(cpg[inclination][trail_id],time)
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
        gait_diagram(fig,axs[3*idx+1],gs1,gait_diagram_data[inclination][trail_id])
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



def calculate_phase_diff(CPGs_output,time):
    '''
    calculating phase difference bewteen  C12, C13, C14
    @param: CPGs_output, numpy darray n*8
    @return phi, DataFrame N*4, [time, phi_12, phi_13, phi_14]

    '''
    phi=pd.DataFrame(columns=['time','phi_12','phi_13','phi_14'])
    C1=CPGs_output[:,0:2]
    C2=CPGs_output[:,2:4]
    C3=CPGs_output[:,4:6]
    C4=CPGs_output[:,6:8]

    # phi_12
    temp1=np.sum(C1*C2,axis=1)
    temp2=np.sqrt(np.sum(C1**2,axis=1))*np.sqrt(np.sum(C2**2,axis=1))
    cos=temp1/temp2
    cos = np.clip(cos, -1, 1) # set the cos in range [-1,1]
    phi_12=np.arccos(cos)

    # phi_13
    temp1=np.sum(C1*C3,axis=1)
    temp2=np.sqrt(np.sum(C1**2,axis=1))*np.sqrt(np.sum(C3**2,axis=1))
    cos=temp1/temp2
    cos = np.clip(cos, -1, 1) # set the cos in range [-1,1]
    phi_13=np.arccos(cos)

    # phi_14
    temp1=np.sum(C1*C4,axis=1)
    temp2=np.sqrt(np.sum(C1**2,axis=1))*np.sqrt(np.sum(C4**2,axis=1))
    cos=temp1/temp2
    cos = np.clip(cos, -1, 1) # set the cos in range [-1,1]
    phi_14=np.arccos(cos)

    data=np.concatenate((time.reshape((-1,1)),phi_12.reshape((-1,1)),phi_13.reshape((-1,1)),phi_14.reshape((-1,1))),axis=1)
    data=pd.DataFrame(data,columns=['time','phi_12','phi_13','phi_14'])
    phi=pd.concat([phi,data])

    return phi

def plot_phase_diff(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['0.0'],trail_id=0):
    ''' 
    This is for plot the phase diff by a plot curve
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_point, the start point (time) of all the data
    @param: end_point, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_classes, the conditions/cases/experiment_classes of the experimental data
    @param: trail_id, it indicates which experiment among a inclination/situation/case experiments 
    @return: show and save a data figure.
    '''
    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)

    cpg={}

    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        cpg[class_name]=[]  #files_name is the table of the files_name class_name
        print(class_name)
        for idx in files_name.index:
            folder_class_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_class_name)
        # 2)  data process
            print(folder_class_name)
            cpg[class_name].append(cpg_data)


    # 2) calculate the phase angles
    phi=pd.DataFrame(columns=['time','phi_12','phi_13','phi_14'])
    for idx in range(len(cpg[experiment_classes[0]])):
        trail_id=idx
        phi=pd.concat([phi,calculate_phase_diff(cpg[experiment_classes[0]][trail_id],time)])

    #3) plot
    figsize=(5.5,7.)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(9,len(experiment_classes))#13
    gs1.update(hspace=0.13,top=0.95,bottom=0.1,left=0.11,right=0.98)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:9,idx]))


    sns.lineplot(x='time',y='phi_12',data=phi,ax=axs[0])
    sns.lineplot(x='time',y='phi_13',data=phi,ax=axs[0])
    sns.lineplot(x='time',y='phi_14',data=phi,ax=axs[0])
    axs[0].legend(['$\phi_{12}$','$\phi_{13}$','$\phi_{14}$'])

    #pdb.set_trace()
    #plt.plot(phi_12)
    #plt.plot(phi_13)
    #plt.plot(phi_14)
    plt.show()


if __name__=="__main__":


    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseTransition/"
    #PhaseAnalysis(data_file_dic, start_point=960, end_point=1560, freq=60.0, experiment_classes = ['-0.2'], trail_id=3)#1440-2160


    data_file_dic= "/home/suntao/workspace/experiment_data/"
    #plot_phase_transition(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['-0.2'],trail_id=0)


    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseReset/Normal/SingleExperiment/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseReset/Normal/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseReset/AbnormalLeg/"
    data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseReset/Payload/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseReset/NoiseFeedback/"

    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseTransition/Normal/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseTransition/AbnormalLeg/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseTransition/Payload/"
    #data_file_dic= "/media/suntao/DATA/Research/P3_workspace/Figures/experiment_data/PhaseTransition/NoiseFeedback/"
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #plot_phase_diff(data_file_dic,start_point=240,end_point=1200,freq=60.0,experiment_classes=['-0.2'],trail_id=0)

    ''' The experiment one '''
    #GeneralDisplay(data_file_dic,start_point=240,end_point=721+240,freq=60.0,experiment_classes=['-0.2'],trail_id=1)

    ''' The experiment two'''
    Phase_Gait(data_file_dic,start_point=240+60,end_point=721+360+60,freq=60.0,experiment_classes=['-0.2'],trail_id=0)


