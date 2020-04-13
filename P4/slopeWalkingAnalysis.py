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
import time as localtimepkg
import termcolor

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
    plot gait diagram
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
        ax[idx].set_yticks([0.1,0.0,0.5,1.0,1.1])
        ax[idx].set_xticks([])
        ax[idx].set_xticklabels(labels=[])


def gait_diagram(fig,axs,gs,gait_data):
    '''
    plot gait diagram
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


#def loadData(fileName,columnsName,folderName="/media/suntao/DATA/Research/P1_workspace/Experiments/Experiment_data/fig12/1231060335"):
#def loadData(fileName,columnsName,folderName="/home/suntao/workspace/experiment_data/0127113800"):
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
    temp.loc[temp['hind']<0.01,'gamma']=0.0
    gamma=temp['gamma'].to_numpy()
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
    # binary the grf value 
    threshold =0.04
    state=np.zeros(data.shape,int)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if data[j,i] < threshold:
                state[j,i] = 0
            else:
                state[j,i]=1
    
    # get gait info, count the tocuh and lift number steps
    gait_info=[]
    beta=[]
    state=np.vstack([state,abs(state[-1,:]-1)])
    for i in range(state.shape[1]):
        count_stance=0;
        count_swing=0;
        number_stance=0
        number_swing=0
        duty_info = {}
                
        for j in range(state.shape[0]-1):
            if state[j,i] ==1:
                count_stance+=1
            else:
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
    for i in range(len(gait_info)):
        beta_singleleg=[]
        for j in range(int(len(gait_info[i]))):
            if (gait_info[i].__contains__(str(j)+'stance') and  gait_info[i].__contains__(str(j)+'swing')):
                beta_singleleg.append(gait_info[i][str(j)+"stance"]/(gait_info[i][str(j)+"stance"] + gait_info[i][str(j) + "swing"]))
            elif (gait_info[i].__contains__(str(j)+'stance') and  (not gait_info[i].__contains__(str(j)+'swing'))):
                beta_singleleg.append(1.0)
            elif ((not gait_info[i].__contains__(str(j)+'stance')) and  gait_info[i].__contains__(str(j)+'swing')):
                beta_singleleg.append(0.0)

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

def neural_preprocessing(data):
    '''
    This is use a recurrent neural network to preprocess the data,
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
    assert(joint_cmd.shape[1]==4)
    assert(actual_grf.shape[1]==4)

    new_expected_grf = np.zeros(joint_cmd.shape)
    new_actual_grf = np.zeros(actual_grf.shape)
    joint_cmd_middle_position=[]

    for idx in range(4):
        joint_cmd_middle_position.append((np.amax(joint_cmd[:,idx])+np.amin(joint_cmd[:,idx]))/2.0)
        new_expected_grf[:,idx] = forceForwardmodel(joint_cmd[:,idx])
        new_actual_grf[:,idx] = neural_preprocessing(actual_grf[:,idx])
    # pre process again
    threshold=0.2
    new_expected_grf = new_expected_grf > threshold
    new_actual_grf = new_actual_grf > threshold
    move_stage = joint_cmd > joint_cmd_middle_position

    new_actual_grf=new_actual_grf.astype(np.int)
    new_expected_grf=new_expected_grf.astype(np.int)
    diff = new_actual_grf-new_expected_grf

    Te1=(diff*move_stage==1)
    Te3=(diff*move_stage==-1)
    Te2=(diff*(~move_stage)==1)
    Te4=(diff*(~move_stage)==-1)

    return [new_expected_grf, new_actual_grf, diff, Te1, Te2, Te3, Te4]




def read_data(freq,start_point,end_point,folder_name):
    #1) Load data
    fileName_CPGs="controlfile_CPGs"
    fileName_commands='controlfile_commands'
    fileName_modules='controlfile_modules'
    fileName_parameters='parameterfile_parameters'
    fileName_joints='sensorfile_joints'

    columnsName_CPGs=['RFO1','RFO2','RHO1','RHO2','LFO1','LFO2','LHO1','LKO2']
    columnsName_GRFs=['RF','RH','LF','LH']
    columnsName_POSEs=['roll','picth','yaw', 'x','y','z','vx','vy','vz']
    columnsName_jointPositions=['p1','p2','p3','p4','p5','p6', 'p7','p8','p9','p10','p11','p12']
    columnsName_jointVelocities=['v1','v2','v3','v4','v5','v6', 'v7','v8','v9','v10','v11','v12']
    columnsName_jointCurrents=['c1','c2','c3','c4','c5','c6', 'c7','c8','c9','c10','c11','c12']
    columnsName_jointVoltages=['vol1','vol2','vol3','vol4','vol5','vol6', 'vol7','vol8','vol9','vol10','vol11','vol12']
    columnsName_modules=['ss']
    columnsName_parameters=['MI']
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
    return cpg_data[start_point:end_point,:], command_data[start_point:end_point,:], module_data[start_point:end_point,:], parameter_data[start_point:end_point,:], grf_data[start_point:end_point,:], pose_data[start_point:end_point,:], position_data[start_point:end_point,:],velocity_data[start_point:end_point,:],current_data[start_point:end_point,:],voltage_data[start_point:end_point,:], time


def plot_slopeWalking_gamma_beta_pitch_displacement(data_file_dic,start_point=90,end_point=1200,freq=60.0,inclinations=['-0.2618','0.0','0.2618']):
    # 1) read data
    #1) load data from file
    data_file = data_file_dic +"slopeWalking.log"
    data_date = pd.read_csv(data_file, sep='\t',header=None, names=['file_name','inclination'], skip_blank_lines=True,dtype=str)
    
    data_date_inclination=data_date.groupby('inclination')
    labels= data_date_inclination.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    displacement={}
    jmc={}

    for name, inclination in data_date_inclination: #name is a inclination names
        gamma[name]=[]  #inclination is the table of the inclination name
        gait_diagram_data[name]=[]
        beta[name]=[]
        pitch[name]=[]
        displacement[name]=[]
        jmc[name]=[]
        for idx in inclination.index:
            folder_name= data_file_dic + inclination['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            gamma[name].append(COG_distribution(grf_data))

            gait_diagram_data_temp, beta_temp=gait(grf_data)
            gait_diagram_data[name].append(gait_diagram_data_temp); beta[name].append(beta_temp)

            pitch[name].append(pose_data)

            displacement[name].append(pose_data[:,3]/np.cos(float(name)))
            jmc[name].append(command_data)

    
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

    figsize=(9.1,7.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(5,len(inclinations))#13
    gs1.update(hspace=0.22,top=0.95,bottom=0.09,left=0.1,right=0.98)
    axs=[]
    for idx in range(len(inclinations)):# how many columns, depends on the inclinations
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))
        axs.append(fig.add_subplot(gs1[4:5,idx]))


    #3.1) plot 
    for idx, inclination in enumerate(inclinations):
        axs[3*idx].plot(time,jmc[inclination][0][:,1], 'r:')
        axs[3*idx].plot(time,jmc[inclination][0][:,2],'r-.')
        axs[3*idx].plot(time,jmc[inclination][0][:,10],'b:')
        axs[3*idx].plot(time,jmc[inclination][0][:,11],'b-.')
        axs[3*idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx].grid(which='both',axis='y',color='k',linestyle=':')
        axs[0].set_ylabel(u'Joint commands')
        axs[3*idx].set_yticks([-1.0,-0.5,0.0,0.5,1.0])
        axs[3*idx].legend(['RF hip','RF knee','LH hip', 'LH knee'],ncol=2)
        axs[3*idx].set_xticklabels([])
        axs[3*idx].set_title('Inclination of the slope:' + str(round(180*float(inclination)/3.1415))+ u'\u00b0')
        axs[3*idx].set(xlim=[min(time),max(time)])


        axs[1].set_ylabel(u'$\gamma$,$p$ [rad],$d$ [m]')
        axs[3*idx+1].plot(time,gamma[inclination][0])
        #axs[3*idx+1].plot(beta[inclination][0])
        axs[3*idx+1].plot(time,pitch[inclination][0])
        axs[3*idx+1].plot(time,displacement[inclination][0]-displacement[inclination][0][0])
        axs[3*idx+1].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+1].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+1].legend([u'$\gamma$',u'$p$',u'$d$'])
        axs[3*idx+1].set_yticks([-0.5,0.0,1.0,2.0,4,6,8])
        axs[3*idx+1].set_xticklabels([])
        axs[3*idx+1].set(xlim=[min(time),max(time)])

        gait_diagram(fig,axs[3*idx+2],gs1,gait_diagram_data[inclination][0])
        axs[2].set_ylabel(u'Gait')
        axs[3*idx+2].set_xlabel(u'Time [s]')
        xticks=np.arange(int(max(time)))
        axs[3*idx+2].set_xticklabels([str(xtick) for xtick in xticks])
        axs[3*idx+2].set_xticks(xticks)
        axs[2].yaxis.set_label_coords(-0.15,.5)

    #figPath=data_file_dic + '../../' + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithVestibularandCOGReflex_singleCurve.svg'
    #plt.savefig(figPath)
    plt.show()


def slopeWalking_gamma_beta_pitch_displacement_statistic(data_file_dic,start_point=60,end_point=900,freq=60.0,inclinations=['0.0']):
    #1) load data from file
    data_file = data_file_dic +"slopeWalking.log"
    data_date = pd.read_csv(data_file, sep='\t',header=None, names=['file_name','inclination'], skip_blank_lines=True,dtype=str)
    
    data_date_inclination=data_date.groupby('inclination')
    labels= data_date_inclination.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    gamma={}
    diff_beta={}
    max_pitch={}
    max_displacement={}

    for name, inclination in data_date_inclination: #name is a inclination names
        gamma[name]=[]  #inclination is the table of the inclination name
        diff_beta[name]=[]
        max_pitch[name]=[]
        max_displacement[name]=[]
        for idx in inclination.index:
            #folder_name= "/home/suntao/workspace/experiment_data/" + inclination['file_name'][idx]
            folder_name= data_file_dic + inclination['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            gamma[name].append(Average_COG_distribution(grf_data[start_point:end_point,:]))

            gait_diagram_data, beta=gait(grf_data[start_point:end_point,:])

            diff_beta[name].append(np.std(AvgerageGaitBeta(beta)))

            max_pitch[name].append(np.mean(pose_data[start_point:end_point,1]))

            max_displacement[name].append(max(pose_data[start_point:end_point,3])/np.cos(float(name)))

    
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

    figsize=(7.1,6.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.15,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))



    ind= np.arange(len(labels))
    width=0.15

    #3.1) plot 
    gamma_mean,gamma_std=[],[]
    diff_beta_mean,diff_beta_std=[],[]
    max_pitch_mean,max_pitch_std=[],[]
    max_displacement_mean,max_displacement_std=[],[]
    for i in labels:
        gamma_mean.append(np.mean(gamma[i]))
        gamma_std.append(np.std(gamma[i]))
        diff_beta_mean.append(np.mean(diff_beta[i]))
        diff_beta_std.append(np.std(diff_beta[i]))
        max_pitch_mean.append(np.mean(max_pitch[i]))
        max_pitch_std.append(np.std(max_pitch[i]))
        max_displacement_mean.append(np.mean(max_displacement[i]))
        max_displacement_std.append(np.std(max_displacement[i]))

    idx=0
    axs[idx].bar(ind-1.5*width,gamma_mean,width,yerr=gamma_std,label=r'COG position reagrding front and hind foot coordination: $\gamma$')
    axs[idx].bar(ind-0.5*width,diff_beta_mean, width, yerr=diff_beta_std,label=r'Variance of the four leg motion duty factor: $std(\beta_{1,2,3,4})$')
    axs[idx].bar(ind+0.5*width,max_pitch_mean, width, yerr=max_pitch_std,label=r'Max pitch angle of robot body [rad]: $p$')
    axs[idx].bar(ind+1.5*width,max_displacement_mean,width,yerr=max_displacement_std,label=r'Walking distance on the floor [m]: $d$')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    axs[idx].set_yticks([-1,0,1,2,3,4,5])
    axs[idx].set_xticklabels([ str(round(180*float(l)/3.1415))+ u'\u00b0' for l in labels])
    axs[idx].legend()
    axs[idx].set_xlabel(r'Inclination of the floor: $\alpha$')


    axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    
    figPath= data_file_dic + '../../' + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithCOGReflexes.svg'
    #plt.savefig(figPath)
    plt.show()


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


def slopeWalking_getTouchData(data_file_dic,start_point=900,end_point=1200,freq=60):

    # 1) read data

    #1) load data from file
    data_file = data_file_dic +"slopeWalking.log"
    data_date = pd.read_csv(data_file, sep='\t',header=None, names=['file_name','inclination'], skip_blank_lines=True,dtype=str)
    
    data_date_inclination=data_date.groupby('inclination')
    labels= data_date_inclination.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    Te={}
    Te1={}
    Te2={}
    Te3={}
    Te4={}

    for name, inclination in data_date_inclination: #name is a inclination names
        Te[name] =[]
        Te1[name]=[]  #inclination is the table of the inclination name
        Te2[name]=[]
        Te3[name]=[]
        Te4[name]=[]
        for idx in inclination.index: # how many time experiments one inclination
            #folder_name= "/home/suntao/workspace/experiment_data/" + inclination['file_name'][idx]
            folder_name= data_file_dic + inclination['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            Te_temp=touch_difference(cpg_data[:,[0,2,4,6]], grf_data)
            Te1[name].append(sum(Te_temp[3])/freq)
            Te2[name].append(sum(Te_temp[4])/freq)
            Te3[name].append(sum(Te_temp[5])/freq)
            Te4[name].append(sum(Te_temp[6])/freq)
            Te[name].append([Te_temp,command_data])

    return Te, Te1, Te2,Te3, Te4, labels

def slopeWalking_touchMomentAnalysis(data_file_dic,start_point=900,end_point=1200,freq=60):

    # 1) get tocuh data
    Te, Te1, Te2,Te3,Te4, labels = slopeWalking_getTouchData(data_file_dic,start_point,end_point,freq)

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

    figsize=(7.1,6.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.15,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))



    ind= np.arange(len(labels))
    width=0.15

    #3.1) plot 
    Te1_mean,Te1_std=[],[]
    Te2_mean,Te2_std=[],[]
    Te3_mean,Te3_std=[],[]
    Te4_mean,Te4_std=[],[]
    for i in labels: # various inclinations
        Te1_mean.append(np.mean(Te1[i]))
        Te1_std.append(np.std(Te1[i]))
        Te2_mean.append(np.mean(Te2[i]))
        Te2_std.append(np.std(Te2[i]))
        Te3_mean.append(np.mean(Te3[i]))
        Te3_std.append(np.std(Te3[i]))
        Te4_mean.append(np.mean(Te4[i]))
        Te4_std.append(np.std(Te4[i]))


    idx=0
    axs[idx].bar(ind-1.5*width,Te1_mean,width,yerr=Te1_std,label=r'Te1')
    axs[idx].bar(ind-0.5*width,Te2_mean,width,yerr=Te2_std,label=r'Te2')
    axs[idx].bar(ind+0.5*width,Te3_mean,width,yerr=Te3_std,label=r'Te3')
    axs[idx].bar(ind+1.5*width,Te4_mean,width,yerr=Te4_std,label=r'Te4')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    axs[idx].set_yticks([-0.5,0,1,2])
    axs[idx].set_xticklabels([ str(round(180*float(l)/3.1415))+ u'\u00b0' for l in labels])
    axs[idx].legend()
    axs[idx].set_xlabel(r'Inclination of the floor: $\alpha$')


    axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    
    figPath=data_file_dic+ '../../' + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithCOGReflexes_TouchDiff.svg'
    #plt.savefig(figPath)
    plt.show()
    

def plot_slopeWalking_comparasion_expected_actual_grf_all_leg(data_file_dic,start_point=600,end_point=1200,freq=60,inclinations=['0.0']):

    Te, Te1, Te2,Te3,Te4, labels = slopeWalking_getTouchData(data_file_dic,start_point,end_point,freq)

    figsize=(9.1,11.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(8,len(inclinations))#13
    gs1.update(hspace=0.24,top=0.95,bottom=0.07,left=0.1,right=0.98)
    axs=[]
    for idx in range(len(inclinations)):# how many columns, depends on the inclinations
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))
        axs.append(fig.add_subplot(gs1[4:6,idx]))
        axs.append(fig.add_subplot(gs1[6:8,idx]))

    for idx,inclination in enumerate(inclinations):
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=0,ax=axs[4*idx])
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=1,ax=axs[4*idx+1])
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=2,ax=axs[4*idx+2])
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=3,ax=axs[4*idx+3])
    plt.show()


def plot_pose_commands(data_file_dic,start_point=10,end_point=400,freq=60,inclinations=['0.0']):
       
    # 1) read data

    data_file = data_file_dic +"slopeWalking.log"
    data_date = pd.read_csv(data_file, sep='\t',header=None, names=['file_name','inclination'], skip_blank_lines=True,dtype=str)
    
    data_date_inclination=data_date.groupby('inclination')
    labels= data_date_inclination.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    pose={}
    commands={}

    for name, inclination in data_date_inclination: #name is a inclination names
        pose[name] =[]
        commands[name]=[]
        for idx in inclination.index: # how many time experiments one inclination
            #folder_name= "/home/suntao/workspace/experiment_data/" + inclination['file_name'][idx]
            folder_name= data_file_dic + inclination['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            pose[name].append(pose_data)
            commands[name].append(command_data)
            print(folder_name)

    #2) plot

    figsize=(9.1,5.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(4,len(inclinations))#13
    gs1.update(hspace=0.24,top=0.95,bottom=0.07,left=0.1,right=0.98)
    axs=[]
    for idx in range(len(inclinations)):# how many columns, depends on the inclinations
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))

    for idx,inclination in enumerate(inclinations):
        axs[2*idx].plot(pose[inclination][0][:,0],'r')
        #axs[idx].plot(pose[inclination][0][:,1],'g')
        axs[2*idx].plot(pose[inclination][0][:,2],'b')
        axs[2*idx].legend(['roll','yaw'])
        axs[2*idx+1].plot(commands[inclination][0][:,1],'r')
        axs[2*idx+1].plot(commands[inclination][0][:,2],'b')
    plt.show()


def plot_adaptiveOffset(data_file_dic,start_point=90,end_point=1200,freq=60.0,inclinations=['0.0']):
    # 1) read data
    #1) load data from file
    data_file = data_file_dic +"slopeWalking.log"
    data_date = pd.read_csv(data_file, sep='\t',header=None, names=['file_name','inclination'], skip_blank_lines=True,dtype=str)
    
    data_date_inclination=data_date.groupby('inclination')
    labels= data_date_inclination.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    CPGs={}
    gamma={}
    jmc={}

    for name, inclination in data_date_inclination: #name is a inclination names
        gamma[name]=[]  #inclination is the table of the inclination name
        jmc[name]=[]
        CPGs[name]=[]
        for idx in inclination.index:
            folder_name= data_file_dic + inclination['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            gamma[name].append(COG_distribution(grf_data))
            CPGs[name].append(cpg_data)
            jmc[name].append(command_data)

    # plot
    figsize=(6.,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,len(inclinations))#13
    gs1.update(hspace=0.22,top=0.95,bottom=0.12,left=0.1,right=0.98)
    axs=[]
    for idx in range(len(inclinations)):# how many columns, depends on the inclinations
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))
        axs.append(fig.add_subplot(gs1[4:6,idx]))

    axs[0].plot(time,CPGs['0'][0][:,0],'r')
    axs[0].plot(time,CPGs['0'][0][:,1],'b')
    axs[0].set_xticklabels([])
    axs[0].legend([r'$o_1$',r'$o_2$'],loc='upper right',ncol=2)
    axs[0].set_yticks([-1.0,0.0,1.0])
    axs[0].set_yticklabels([-1.0,0.0,1.0])
    axs[0].grid(which='both',axis='x',color='k',linestyle=':')
    axs[0].grid(which='both',axis='y',color='k',linestyle=':')
    axs[0].set(xlim=[min(time),max(time)],ylim=[-1.02,1.02])
    axs[0].set_ylabel('CPG')

    axs[1].plot(time,jmc['0'][0][:,1],'r')
    axs[1].plot(time,jmc['0'][0][:,2],'b')
    axs[1].set_xticklabels([])
    axs[1].legend([r'$\theta_1$',r'$\theta_2$'],loc='upper right', ncol=2)
    axs[1].set_yticks([-0.4,-0.2,0,.2])
    axs[1].grid(which='both',axis='x',color='k',linestyle=':')
    axs[1].grid(which='both',axis='y',color='k',linestyle=':')
    axs[1].set(xlim=[min(time),max(time)])
    axs[1].set_ylabel('MNs')


    axs[2].plot(time,gamma['0'][0])
    df=pd.DataFrame(gamma['0'][0],columns=['gamma'])
    average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
    average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
    axs[2].plot(time,average_average_gamma)
    axs[2].plot(time,1.4*np.ones(average_average_gamma.shape),'-.')
    axs[2].legend([r'$\gamma$',r'$\bar{\bar{\gamma}}$',r'$\bar{\bar{\gamma}}^d$'],loc='upper right',ncol=3)
    axs[2].set_xlabel('Time [s]')
    axs[2].grid(which='both',axis='x',color='k',linestyle=':')
    axs[2].grid(which='both',axis='y',color='k',linestyle=':')
    axs[2].set_ylabel('GRFs distribution')
    axs[2].set_yticks([0,1.4,3,5])

    axs[2].set_xticklabels([str(idx) for idx in range(int(time[-1]-time[0]+1))])
    axs[2].set(xlim=[min(time),max(time)])

    figPath=data_file_dic + '../../' + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'adaptive_offset.svg'
    plt.savefig(figPath)
    plt.show()


def Experiment1(data_file_dic,start_point=300,end_point=660,freq=60.0,inclinations=['0']):
    '''
    plot theta1, theta2;  gamma_d, gamma_a; posture, pitch

    '''

    # 1) read data
    #1.1) load file list 
    data_file = data_file_dic +"slopeWalking.log"
    data_date = pd.read_csv(data_file, sep='\t',header=None, names=['file_name','inclination'], skip_blank_lines=True,dtype=str)
    
    data_date_inclination=data_date.groupby('inclination')
    labels= data_date_inclination.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    pose={}
    gamma={}
    jmc={}
    #1.2) read data one by one file
    for name, inclination in data_date_inclination: #name is a inclination names
        gamma[name]=[]  #inclination is the table of the inclination name
        jmc[name]=[]
        pose[name]=[]
        for idx in inclination.index:
            folder_name= data_file_dic + inclination['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            gamma[name].append(COG_distribution(grf_data))
            pose[name].append(pose_data)
            jmc[name].append(command_data)

    # plot
    figsize=(6.,5.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(8,1)#13
    gs1.update(hspace=0.22,top=0.95,bottom=0.11,left=0.11,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))
    axs.append(fig.add_subplot(gs1[2:4,0]))
    axs.append(fig.add_subplot(gs1[4:6,0]))
    axs.append(fig.add_subplot(gs1[6:8,0]))

    legends=[r'$s_1$',r'$s_2$',r'$s_3$',r'$s_4$',r'$s_5$']
    idx=0
    axs[idx].plot(time,jmc['-0.2'][0][:,1],'r')
    axs[idx].plot(time,jmc['-0.1'][0][:,1],'g')
    axs[idx].plot(time,jmc['0.0'][0][:,1],'b')
    axs[idx].plot(time,jmc['0.1'][0][:,1],'k')
    axs[idx].plot(time,jmc['0.2'][0][:,1],'y')
    axs[idx].set_xticklabels([])
    axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    axs[idx].set_yticks([-0.5,0.0,0.4])
    axs[idx].set_yticklabels([-0.5,0.0,0.4])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.55,0.41])
    axs[idx].set_ylabel(r'$\theta_1$')

    idx=1
    axs[idx].plot(time,jmc['-0.2'][0][:,2],'r')
    axs[idx].plot(time,jmc['-0.1'][0][:,2],'g')
    axs[idx].plot(time,jmc['0.0'][0][:,2],'b')
    axs[idx].plot(time,jmc['0.1'][0][:,2],'k')
    axs[idx].plot(time,jmc['0.2'][0][:,2],'y')
    axs[idx].set_xticklabels([])
    axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    axs[idx].set_yticks([-0.4,0.0,0.4])
    axs[idx].set_yticklabels([-0.4,0.0,0.4])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.45,0.45])
    axs[idx].set_ylabel(r'$\theta_2$')

    idx=2
    df=pd.DataFrame(gamma['-0.2'][0],columns=['gamma'])
    average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
    average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
    axs[idx].plot(time,average_average_gamma,'r')

    df=pd.DataFrame(gamma['-0.1'][0],columns=['gamma'])
    average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
    average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
    axs[idx].plot(time,average_average_gamma,'g')

    df=pd.DataFrame(gamma['0.0'][0],columns=['gamma'])
    average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
    average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
    axs[idx].plot(time,average_average_gamma,'b')

    df=pd.DataFrame(gamma['0.1'][0],columns=['gamma'])
    average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
    average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
    axs[idx].plot(time,average_average_gamma,'k')

    df=pd.DataFrame(gamma['0.2'][0],columns=['gamma'])
    average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
    average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
    axs[idx].plot(time,average_average_gamma,'y')
    #axs[idx].plot(time,1.4*np.ones(average_average_gamma.shape),'-.')
    axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_ylabel(r'$\bar{\bar{\gamma}}^a$')
    axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.1,4.1])
    axs[idx].set_xticklabels([])
    axs[idx].set_yticks([0.0,1.44,4])
    axs[idx].set_yticklabels([0.0,1.44,4.0])


    idx=3
    axs[idx].plot(time,pose['-0.2'][0][:,1],'r')
    axs[idx].plot(time,pose['-0.1'][0][:,1],'g')
    axs[idx].plot(time,pose['0.0'][0][:,1],'b')
    axs[idx].plot(time,pose['0.1'][0][:,1],'k')
    axs[idx].plot(time,pose['0.2'][0][:,1],'y')
    axs[idx].set_xticklabels([])
    axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    axs[idx].set_yticks([-0.1,0.0,0.1])
    axs[idx].set_yticklabels([-0.1,0.0,0.1])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.11,0.11])
    axs[idx].set_ylabel('Pitch')

    axs[idx].set_xticklabels([str(idx) for idx in range(int(time[-1]-time[0]+1))])
    axs[idx].set(xlim=[min(time),max(time)])
    axs[idx].set_xlabel('Time [s]')

    figPath=data_file_dic + '../../' + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'adaptive_offset.svg'
    #plt.savefig(figPath)
    plt.show()
    
def Experiment2_1(data_file_dic,start_point=90,end_point=1200,freq=60.0,inclinations=['0.0']):
    # 1) read data
    #1) load data from file
    data_file = data_file_dic +"slopeWalking.log"
    data_date = pd.read_csv(data_file, sep='\t',header=None, names=['file_name','inclination'], skip_blank_lines=True,dtype=str)
    
    data_date_inclination=data_date.groupby('inclination')
    labels= data_date_inclination.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    displacement={}
    jmc={}

    for name, inclination in data_date_inclination: #name is a inclination names
        gamma[name]=[]  #inclination is the table of the inclination name
        gait_diagram_data[name]=[]
        beta[name]=[]
        pitch[name]=[]
        displacement[name]=[]
        jmc[name]=[]
        for idx in inclination.index:
            folder_name= data_file_dic + inclination['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            gamma[name].append(COG_distribution(grf_data))

            gait_diagram_data_temp, beta_temp=gait(grf_data)
            gait_diagram_data[name].append(gait_diagram_data_temp); beta[name].append(beta_temp)

            pitch[name].append(pose_data[:,1])

            displacement[name].append(pose_data[:,3]/np.cos(float(name)))
            jmc[name].append(command_data)

    
    #3) plot
    figsize=(7.6,5.2)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(5,len(inclinations))#13
    gs1.update(hspace=0.12,top=0.95,bottom=0.09,left=0.08,right=0.98)
    axs=[]
    for idx in range(len(inclinations)):# how many columns, depends on the inclinations
        axs.append(fig.add_subplot(gs1[0:1,idx]))
        axs.append(fig.add_subplot(gs1[1:2,idx]))
        axs.append(fig.add_subplot(gs1[2:3,idx]))
        axs.append(fig.add_subplot(gs1[3:4,idx]))
        axs.append(fig.add_subplot(gs1[4:5,idx]))

    #3.1) plot 
    for idx, inclination in enumerate(inclinations):
        axs[3*idx].plot(time,jmc[inclination][0][:,1], 'r:')
        axs[3*idx].plot(time,jmc[inclination][0][:,2],'r-.')
        axs[3*idx].plot(time,jmc[inclination][0][:,10],'b:')
        axs[3*idx].plot(time,jmc[inclination][0][:,11],'b-.')
        axs[3*idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx].grid(which='both',axis='y',color='k',linestyle=':')
        axs[0].set_ylabel(u'Joint commands')
        axs[3*idx].set_yticks([-0.8,-0.4,0.0,0.5])
        axs[3*idx].legend(['RF hip','RF knee','LH hip', 'LH knee'],ncol=4)
        axs[3*idx].set_xticklabels([])
        axs[3*idx].set_title('Inclination of the slope:' + str(round(180*float(inclination)/3.1415))+ u'\u00b0')
        axs[3*idx].set(xlim=[min(time),max(time)])


        axs[1].set_ylabel(r'$\bar{\bar{\gamma}}^a$')
        df=pd.DataFrame(gamma[inclination][0],columns=['gamma'])
        average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
        average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
        axs[3*idx+1].plot(time,average_average_gamma,'r')
        axs[3*idx+1].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+1].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+1].set_yticks([0.0,1.0,2.0])
        axs[3*idx+1].set_xticklabels([])
        axs[3*idx+1].set(xlim=[min(time),max(time)],ylim=[-0.1,2.1])

        axs[2].set_ylabel(u'Pitch [degree]')
        axs[3*idx+2].plot(time,pitch[inclination][0]/3.1415926*180,'r')
        axs[3*idx+2].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+2].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+2].set_yticks([-40,-20,0.0])
        axs[3*idx+2].set_xticklabels([])
        axs[3*idx+2].set(xlim=[min(time),max(time)])

        axs[3].set_ylabel(u'Distance [m]')
        axs[3*idx+3].plot(time,-1.0*displacement[inclination][0]-displacement[inclination][0][0],'r')
        axs[3*idx+3].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+3].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+3].set_yticks([0.0,0.2,0.4])
        axs[3*idx+3].set_xticklabels([])
        axs[3*idx+3].set(xlim=[min(time),max(time)])

        axs[4].set_ylabel(r'Gait')
        gait_diagram(fig,axs[3*idx+4],gs1,gait_diagram_data[inclination][0])
        axs[3*idx+4].set_xlabel(u'Time [s]')
        xticks=np.arange(int(max(time)))
        axs[3*idx+4].set_xticklabels([str(xtick) for xtick in xticks])
        axs[3*idx+4].set_xticks(xticks)
        axs[4].yaxis.set_label_coords(-0.15,.5)

    figPath=data_file_dic + '../../' + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithVestibularandCOGReflex_singleCurve.svg'
    #plt.savefig(figPath)
    plt.show()

def Experiment3(data_file_dic,start_point=90,end_point=1200,freq=60.0,inclinations=['0.0']):
    # 1) read data
    #1) load data from file
    data_file = data_file_dic +"slopeWalking.log"
    data_date = pd.read_csv(data_file, sep='\t',header=None, names=['file_name','inclination'], skip_blank_lines=True,dtype=str)
    
    data_date_inclination=data_date.groupby('inclination')
    labels= data_date_inclination.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    displacement={}
    jmc={}

    for name, inclination in data_date_inclination: #name is a inclination names
        gamma[name]=[]  #inclination is the table of the inclination name
        gait_diagram_data[name]=[]
        beta[name]=[]
        pitch[name]=[]
        displacement[name]=[]
        jmc[name]=[]
        for idx in inclination.index:
            folder_name= data_file_dic + inclination['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            gamma[name].append(COG_distribution(grf_data))

            gait_diagram_data_temp, beta_temp=gait(grf_data)
            gait_diagram_data[name].append(gait_diagram_data_temp); beta[name].append(beta_temp)

            pitch[name].append(pose_data[:,1])

            displacement[name].append(pose_data[:,3]/np.cos(float(name)))
            jmc[name].append(command_data)

    
    #3) plot
    figsize=(7.6,6.2)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,len(inclinations))#13
    gs1.update(hspace=0.12,top=0.95,bottom=0.09,left=0.08,right=0.98)
    axs=[]
    for idx in range(len(inclinations)):# how many columns, depends on the inclinations
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:3,idx]))
        axs.append(fig.add_subplot(gs1[3:4,idx]))
        axs.append(fig.add_subplot(gs1[4:5,idx]))
        axs.append(fig.add_subplot(gs1[5:6,idx]))

    #3.1) plot 
    for idx, inclination in enumerate(inclinations):
        axs[3*idx].plot(time,jmc[inclination][0][:,1], 'r:')
        axs[3*idx].plot(time,jmc[inclination][0][:,2],'r-.')
        axs[3*idx].plot(time,jmc[inclination][0][:,10],'b:')
        axs[3*idx].plot(time,jmc[inclination][0][:,11],'b-.')
        axs[3*idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx].grid(which='both',axis='y',color='k',linestyle=':')
        axs[0].set_ylabel(u'Joint commands')
        axs[3*idx].set_yticks([-0.7,-0.4,0.0,0.4])
        axs[3*idx].legend(['RF hip','RF knee','LH hip', 'LH knee'],ncol=4)
        axs[3*idx].set_xticklabels([])
        axs[3*idx].set_title('Inclination of the slope:' + str(round(180*float(inclination)/3.1415))+ u'\u00b0')
        axs[3*idx].set(xlim=[min(time),max(time)])

        axs[1].set_ylabel(r'$\bar{\bar{\gamma}}^a$')
        df=pd.DataFrame(gamma[inclination][0],columns=['gamma'])
        average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
        average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
        axs[3*idx+1].plot(time,average_average_gamma,'r')
        axs[3*idx+1].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+1].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+1].set_yticks([0.0,3.0,7.0])
        axs[3*idx+1].set_xticklabels([])
        axs[3*idx+1].set(xlim=[min(time),max(time)],ylim=[-0.1,7.1])

        axs[2].set_ylabel(u'Pitch [degree]')
        axs[3*idx+2].plot(time,pitch[inclination][0]/3.1415926*180,'r')
        axs[3*idx+2].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+2].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+2].set_yticks([-30,-15,0,15,30])
        axs[3*idx+2].set_xticklabels([])
        axs[3*idx+2].set(xlim=[min(time),max(time)],ylim=[-32,32])

        axs[3].set_ylabel(u'Distance [m]')
        axs[3*idx+3].plot(time, displacement[inclination][0]-displacement[inclination][0][0],'r')
        axs[3*idx+3].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+3].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+3].set_yticks([0,2,4])
        axs[3*idx+3].set_xticklabels([])
        axs[3*idx+3].set(xlim=[min(time),max(time)],ylim=[0,4.2])

        axs[4].set_ylabel(r'Gait')
        gait_diagram(fig,axs[3*idx+4],gs1,gait_diagram_data[inclination][0])
        axs[3*idx+4].set_xlabel(u'Time [s]')
        xticks=np.arange(0,int(max(time)),4)
        axs[3*idx+4].set_xticklabels([str(xtick) for xtick in xticks])
        axs[3*idx+4].set_xticks(xticks)
        axs[4].yaxis.set_label_coords(-0.15,.5)

    figPath=data_file_dic + '../../' + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithVestibularandCOGReflex_singleCurve.svg'
    #plt.savefig(figPath)
    plt.show()

if __name__=="__main__":
    #slopeWalking_touchMomentAnalysis(data_file_dic)

    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #plot_slopeWalking_gamma_beta_pitch_displacement(data_file_dic,start_point=500,end_point=1500,freq=60.0,inclinations=['0.0','0.174','0.349','0.52','0.61'])
    #plot_slopeWalking_comparasion_expected_actual_grf_all_leg(data_file_dic,start_point=1,end_point=1000,freq=60.0,inclinations=['0'])
    #plot_pose_commands(data_file_dic,start_point=10,end_point=500,freq=60.0,inclinations=['0'])
    
    #plot_adaptiveOffset(data_file_dic,start_point=300,end_point=660,freq=60.0,inclinations=['0']) #Fig. 5


    ''' This is for experiment 1 '''
    #Experiment1(data_file_dic,start_point=180,end_point=720,freq=60.0,inclinations=['-0.1','0','0.1'])
    ''' This is for experiment 2_1 '''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_1/"
    #Experiment2_1(data_file_dic,start_point=120,end_point=2000,freq=60.0,inclinations=['35'])

    ''' This is for experiment 2_2   '''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/Vesti_reflexes/"
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/COG_reflexes/"
    #slopeWalking_gamma_beta_pitch_displacement_statistic(data_file_dic,start_point=1000,end_point=2800,freq=60.0,inclinations=['0.0'])
    ''' This is for experiment 3 '''
    data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment3/"
    Experiment3(data_file_dic,start_point=100,end_point=4700,freq=60.0,inclinations=['0.0'])
    
#   fig1=plt.figure()
#   gs1=gridspec.GridSpec(6,1)#13
#   gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.15,right=0.98)
#   ax=fig1.add_subplot(gs1[0:6,0])
#
#   #grf_diagram(fig1,ax,gs1,grf_data[start_point:end_point,:],time)
#   gait_diagram(fig1,ax,gs1,gait_diagram_data)
#   plt.show()
#

