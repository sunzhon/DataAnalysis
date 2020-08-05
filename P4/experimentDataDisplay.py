#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import gridspec
import os
import gnureadline
import pdb 
plt.rc('font',family='Arial')
import pandas as pd
import re
from brokenaxes import brokenaxes
import time as localtimepkg
import termcolor
import seaborn as sns
sys.path.insert(1,'./../../ESN/pyESN')
from pyESN import ESN

st_r_color=(254/255.0,129/255.0,125/255.0)
st_b_color=(129/255.0,184/255.0,223/255.0)


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
    columnsName_modules=['anc','w1','w2','offset1','offset2']
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
    time = np.linspace(start_point/freq,end_point/freq,end_point-start_point)
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


def lowPassFilter(data,gamma,filterType=3):
    if(filterType==1):
        #filter 1
        filterData=gamma*data+(1.0-gamma)*np.vstack([data[-1,:],data[0:-1,:]])
        return filterData

    if(filterType==2):
        #filter 2
        filterData=[]
        for idx, value in enumerate(data):
            filterData.append(sum(data[0:idx])/(idx+1))
        return np.array(filterData)

    if(filterType==3):
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

def esn_identification(data_file_dic,start_point=1,end_point=1200,freq=60,experiment_classes=['0.0']):
    '''
    This is for using esn to computating GRFs distribution
    '''
    # 1) build model
    rng = np.random.RandomState(42)
    esn = ESN(n_inputs = 4, \
            n_outputs = 1, \
            n_reservoir = 200, \
            spectral_radius = 0.25, \
            sparsity = 0.95, \
            noise = 0.001, \
            input_shift = None, \
            input_scaling = None, \
            teacher_scaling = 1.12, \
            teacher_shift = -0.7, \
            out_activation = np.tanh, \
            inverse_out_activation = np.arctanh, \
            random_state = rng, \
            silent = False) 
  
    #2) import traning data
    data_file = data_file_dic +"ExperimentDataLog.log"
    data_date = pd.read_csv(data_file, sep='\t',header=None, names=['file_name','inclination'], skip_blank_lines=True,dtype=str)
    
    datas_of_experiment_classes=data_date.groupby('inclination')
    labels= datas_of_experiment_classes.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    grf={}

    for name, inclination in datas_of_experiment_classes: #name is a inclination names
        grf[name]=[]  #inclination is the table of the inclination name
        for idx in inclination.index:
            folder_name= data_file_dic + inclination['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            grf[name].append(grf_data)

    #3) training model
    name='0.0'
    train_grfs=grf[name][0][:500,:]
    train_output=0.5*np.ones((len(train_grfs),1))
    
    test_grfs=grf[name][0][600:1000,:]
    test_output=0.5*np.ones((len(test_grfs),1))

    pred_train = esn.fit(train_grfs,train_output)
    print("test error:")
    pred_test = esn.predict(test_grfs)
    plt.plot(pred_test)

    plt.show()
    print(np.sqrt(np.mean((pred_test - test_output)**2)))


def plot_slopeWalking_gamma_beta_pitch_displacement(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['-0.2618','0.0','0.2618']):
    '''
    plot gamma beta pitch and displament of a walking

    @description: plot theta1, theta2;  gamma_d, gamma_a; posture, pitch
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
    #1.1) load file list 
    data_file_log = data_file_dic +"ExperimentDataLog.log"
    data_files = pd.read_csv(data_file_log, sep='\t',header=None, names=['file_name','experiment_classes'], skip_blank_lines=True,dtype=str)
    
    datas_of_experiment_classes=data_files.groupby('experiment_classes')
    labels= datas_of_experiment_classes.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    print(labels)
    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    displacement={}
    jmc={}
    #1.2) read data one by one file
    for class_name, files_name in datas_of_experiment_classes: 
        gamma[class_name]=[]  #inclination is the table of the inclination name
        gait_diagram_data[class_name]=[]
        beta[class_name]=[]
        pitch[class_name]=[]
        displacement[class_name]=[]
        jmc[class_name]=[]

        for idx in files_name.index:
            folder_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            gamma[class_name].append(COG_distribution(grf_data))

            gait_diagram_data_temp, beta_temp=gait(grf_data)
            gait_diagram_data[class_name].append(gait_diagram_data_temp); beta[class_name].append(beta_temp)

            pitch[class_name].append(pose_data)

            displacement[class_name].append(pose_data[:,3]/np.cos(float(class_name)))
            jmc[class_name].append(command_data)

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
    gs1=gridspec.GridSpec(5,len(experiment_classes))#13
    gs1.update(hspace=0.22,top=0.95,bottom=0.09,left=0.1,right=0.98)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))
        axs.append(fig.add_subplot(gs1[4:5,idx]))


    #3.1) plot 
    for idx, inclination in enumerate(experiment_classes):
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

    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath=folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithVestibularandCOGReflex_singleCurve.svg'
    plt.savefig(figPath)
    plt.show()


def slopeWalking_gamma_beta_pitch_displacement_statistic(data_file_dic,start_point=60,end_point=900,freq=60.0,experiment_classes=['0.0']):

    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)
    gamma={}
    diff_beta={}
    max_pitch={}
    max_displacement={}

    for class_name, files_name in datas_of_experiment_classes: 
        gamma[class_name]=[]  #inclination is the table of the inclination name
        diff_beta[class_name]=[]
        max_pitch[class_name]=[]
        max_displacement[class_name]=[]
        for idx in files_name.index:
            folder_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            gamma[class_name].append(Average_COG_distribution(grf_data[start_point:end_point,:]))

            gait_diagram_data, beta=gait(grf_data[start_point:end_point,:])

            diff_beta[class_name].append(np.std(AvgerageGaitBeta(beta)))

            max_pitch[class_name].append(np.mean(pose_data[start_point:end_point,1]))

            max_displacement[class_name].append(max(pose_data[start_point:end_point,3])/np.cos(float(class_name)))

    
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
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithCOGReflexes.svg'
    plt.savefig(figPath)
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

    datas_of_experiment_classes=load_data_log(data_file_dic)
    Te={}
    Te1={}
    Te2={}
    Te3={}
    Te4={}

    for class_name, files_name in datas_of_experiment_classes: #name is a inclination names
        Te[name] =[]
        Te1[name]=[]  #inclination is the table of the inclination name
        Te2[name]=[]
        Te3[name]=[]
        Te4[name]=[]
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            Te_temp=touch_difference(cpg_data[:,[0,2,4,6]], grf_data)
            Te1[class_name].append(sum(Te_temp[3])/freq)
            Te2[class_name].append(sum(Te_temp[4])/freq)
            Te3[class_name].append(sum(Te_temp[5])/freq)
            Te4[class_name].append(sum(Te_temp[6])/freq)
            Te[class_name].append([Te_temp,command_data])

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
    for i in labels: # various experiment_classes
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
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath=folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithCOGReflexes_TouchDiff.svg'
    plt.savefig(figPath)
    plt.show()
    

def plot_comparasion_expected_actual_grf_all_leg(data_file_dic,start_point=600,end_point=1200,freq=60,experiment_classes=['0.0']):
    '''
    compare expected and actual ground reaction force

    '''
    Te, Te1, Te2,Te3,Te4, labels = slopeWalking_getTouchData(data_file_dic,start_point,end_point,freq)

    figsize=(9.1,11.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(8,len(experiment_classes))#13
    gs1.update(hspace=0.24,top=0.95,bottom=0.07,left=0.1,right=0.98)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))
        axs.append(fig.add_subplot(gs1[4:6,idx]))
        axs.append(fig.add_subplot(gs1[6:8,idx]))

    for idx,inclination in enumerate(experiment_classes):
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=0,ax=axs[4*idx])
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=1,ax=axs[4*idx+1])
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=2,ax=axs[4*idx+2])
        plot_comparasion_of_expected_actual_grf(Te[inclination][0],leg_id=3,ax=axs[4*idx+3])
    plt.show()


def plot_actual_grf_all_leg(data_file_dic,start_point=600,end_point=1200,freq=60,experiment_classes=['0.0']):
    '''
    compare all legs' actual ground reaction forces

    '''

    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)
    
    grf={}

    for class_name, files_name in datas_of_experiment_classes: #name is a inclination names
        grf[class_name] =[]
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            grf[class_name].append(grf_data)
            print(folder_name)
    # 2) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(8,len(experiment_classes))#13
    gs1.update(hspace=0.24,top=0.95,bottom=0.07,left=0.1,right=0.99)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:8,idx]))

    for idx,inclination in enumerate(experiment_classes):
        grf_diagram(fig,axs[0],gs1, grf[inclination][idx],time)
    plt.show()

def plot_gait_dutyFactor(data_file_dic,start_point=10,end_point=400,freq=60,experiment_classes=['0.0']):
    ''' Plot gait and its corresponding duty factor '''
    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)

    grf={}

    for class_name, files_name in datas_of_experiment_classes: #name is a inclination names
        grf[class_name] =[]
        for idx in files_name.index: # how many time experiments one inclination
            #folder_name= "/home/suntao/workspace/experiment_data/" + inclination['file_name'][idx]
            folder_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            grf[class_name].append(grf_data)
            print(folder_name)

    #2) plot
    figsize=(6,4)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(4,len(experiment_classes))#13
    gs1.update(hspace=0.2,top=0.95,bottom=0.13,left=0.1,right=0.98)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))

    for idx,inclination in enumerate(experiment_classes):
        gait_diagram_data, duty_factor_data=gait(grf[inclination][0])
        gait_diagram(fig,axs[2*idx],gs1,gait_diagram_data)
        axs[2*idx].set_xticklabels([])
        axs[2*idx+1].plot(duty_factor_data[0],'o',color=(0.01,0.01,0.01))
        axs[2*idx+1].plot(duty_factor_data[1],'o',color=(0.3,0.3,0.25))
        axs[2*idx+1].plot(duty_factor_data[2],'o',color=(0.5,0.5,0.4))
        axs[2*idx+1].plot(duty_factor_data[3],'o',color=(0.7,0.7,0.6))
        # calculate mean and std
        DF=np.array(duty_factor_data)
        axs[2*idx+1].plot(np.mean(DF,axis=0),'s',color=(129/255.0,184/255.0,223/255.0))
        axs[2*idx+1].plot(np.std(DF,axis=0)+0.55,'s',color=(254/255.0,129/255.0,125/255.0))
        axs[2*idx+1].legend(['RF','RH','LF','LH','mean','std'],ncol=6,loc='lower center')
        axs[2*idx+1].set_xlabel('Time [s]')
        axs[2*idx+1].set_yticks([0.4,0.55,0.7])
        axs[2*idx+1].grid(which='both',axis='x',color='k',linestyle=':')
        axs[2*idx+1].grid(which='both',axis='y',color='k',linestyle=':')
        axs[2*idx+1].set_xticklabels([str(i) for i in range( 1+len(np.mean(DF,axis=0)))])
        #axs[2*idx+1].set_xticks[]
    plt.show()

def plot_attitude_accelerate(data_file_dic,start_point=10,end_point=400,freq=60,experiment_classes=['0.0']):
    ''' 
    plot attitude and accelerate

    '''
    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)
    
    pose={}

    for class_name, files_name in datas_of_experiment_classes: #name is a inclination names
        pose[class_name] =[]
        for idx in files_name.index: # how many time experiments one inclination
            #folder_name= "/home/suntao/workspace/experiment_data/" + inclination['file_name'][idx]
            folder_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            pose[class_name].append(pose_data)
            print(folder_name)

    #2) plot
    figsize=(6,4)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(4,len(experiment_classes))#13
    gs1.update(hspace=0.2,top=0.95,bottom=0.13,left=0.1,right=0.98)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))

    for idx,inclination in enumerate(experiment_classes):
        axs[2*idx].plot(time[:-1],(pose[inclination][0][1:,6]-pose[inclination][0][:-1,6])*freq,color='C0')
        axs[2*idx].plot(time[:-1],(pose[inclination][0][1:,7]-pose[inclination][0][:-1,7])*freq,color='C1')
        axs[2*idx].plot(time[:-1],(pose[inclination][0][1:,8]-pose[inclination][0][:-1,8])*freq,color='C2')
        axs[2*idx].legend([r'$a_x$',r'$a_y$',r'$a_z$'],ncol=3,loc='upper right')
        axs[2*idx].set_xticklabels([])
        axs[2*idx].set_ylabel('Accelerate [m/s^2]')
        axs[2*idx].set(ylim=[-3.0,3.0])
        axs[2*idx+1].plot(time[:-1],(pose[inclination][0][1:,0]-pose[inclination][0][:-1,0])*freq*57.29,color='C0')
        axs[2*idx+1].plot(time[:-1],(pose[inclination][0][1:,1]-pose[inclination][0][:-1,1])*freq*57.29,color='C1')
        axs[2*idx+1].plot(time[:-1],(pose[inclination][0][1:,2]-pose[inclination][0][:-1,2])*freq*57.29,color='C2')
        axs[2*idx+1].set_ylabel('Attitude velocity [degree/s]')
        axs[2*idx+1].legend([r'$v_{roll}$',r'$v_{pitch}$',r'$v_{yaw}$'],ncol=3,loc='upper right')
        axs[2*idx+1].set(ylim=[-30.0,30.0])
        axs[2*idx+1].set_xlabel('Time [s]')
    plt.show()


def plot_runningSuccess_statistic(data_file_dic,start_point=10,end_point=400,freq=60,experiment_classes=['0.0']):
    '''
    Statistical of running success
    

    '''
    # 1) read data
    #datas_of_experiment_classes=load_data_log(data_file_dic)
    data_file_dic_COG=data_file_dic+"DFFB_reflexes_V2/"
    datas_of_experiment_classes=load_data_log(data_file_dic_COG)
    
    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels= datas_of_experiment_classes.groups.keys()
    labels=[str(ll) for ll in sorted([float (st) for st in labels])]
    ind= np.arange(len(labels))
    width=0.15

    #3.1) plot 

    idx=0
    axs[idx].bar(ind-0.5*width,[5,5,5,5,5,5,5,5,5],width,label=r'DFFB reflexes')
    axs[idx].bar(ind+0.5*width,[0,0,0,5,5,5,5,0,0], width,label=r'Vestibular reflexes')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    axs[idx].set(ylim=[0,5])
    axs[idx].set_xticklabels([ str(round(180*float(l)/3.1415))+ u'\u00b0' for l in labels])
    axs[idx].legend(loc='center right')
    axs[idx].set_ylabel(r'Success [count]')
    axs[idx].set_xlabel(r'Inclination of the floor: $\epsilon [degree]$')

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'runningSuccess.svg'
    plt.savefig(figPath)
    plt.show()

def plot_stability_statistic(data_file_dic,start_point=10,end_point=400,freq=60,experiment_classes=['0.0']):
    '''
    Stability of statistic

    '''
    # 1) read data
    data_file_dic_COG=data_file_dic+"DFFB_reflexes_V2/"
    datas_of_experiment_classes=load_data_log(data_file_dic_COG)

    pose_COG={}

    for class_name, files_name in datas_of_experiment_classes: #name is a inclination names
        pose_COG[class_name] =[]
        print(class_name)
        for idx in files_name.index: # how many time experiments one inclination
            #folder_name= "/home/suntao/workspace/experiment_data/" + inclination['file_name'][idx]
            folder_name= data_file_dic_COG + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            pose_COG[class_name].append(1.0/np.std(pose_data[:,0],axis=0))
            print(folder_name)
            print(pose_COG[class_name][-1])

    #1.2) read Vesti reflexes
    data_file_dic_Vesti=data_file_dic+"Vesti_reflexes_V2/"
    datas_of_experiment_classes=load_data_log(data_file_dic_Vesti)

    pose_Vesti={}

    for class_name, files_name in datas_of_experiment_classes: #name is a inclination names
        pose_Vesti[class_name] =[]
        print(class_name)
        for idx in files_name.index: # how many time experiments one inclination
            folder_name= data_file_dic_Vesti + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            ''' this part is for the experiemnt 2'''
            '''
            if (class_name=='0.174') or (class_name=='-0.174') or (class_name=='0.0') or (class_name=='0.349'):
                pose_Vesti[class_name].append(1.0/np.std(pose_data[:,0],axis=0))
            else:
                pose_Vesti[class_name].append(0.0)
            '''

            ''' this part is for the experiemnt 1'''
            pose_Vesti[class_name].append(1.0/np.std(pose_data[:,0],axis=0))
            print(folder_name)
            print(pose_Vesti[class_name][-1])

    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels= datas_of_experiment_classes.groups.keys()
    labels=[str(ll) for ll in sorted([float (st) for st in labels])]
    ind= np.arange(len(labels))
    width=0.15

    #3.1) plot 
    angular_COG_mean, angular_COG_std=[],[]
    angular_Vesti_mean, angular_Vesti_std=[],[]
    for i in labels: #inclination
        angular_COG_mean.append(np.mean(pose_COG[i]))
        angular_COG_std.append(np.std(pose_COG[i]))

        angular_Vesti_mean.append(np.mean(pose_Vesti[i]))
        angular_Vesti_std.append(np.std(pose_Vesti[i]))

    idx=0
    axs[idx].bar(ind-0.5*width,angular_COG_mean,width,yerr=angular_COG_std,label=r'DFFB reflexes')
    axs[idx].bar(ind+0.5*width,angular_Vesti_mean, width, yerr=angular_Vesti_std,label=r'Vestibular reflexes')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels([ str(round(180*float(l)/3.1415))+ u'\u00b0' for l in labels])
    axs[idx].legend()
    axs[idx].set_ylabel(r'Stability')
    axs[idx].set_xlabel(r'Inclination of the floor: $\epsilon [degree]$')

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    # save plot
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'stabilityStatistic.svg'
    plt.savefig(figPath)
    plt.show()

def plot_coordination_statistic(data_file_dic,start_point=60,end_point=900,freq=60.0,experiment_classes=['0.0']):
    '''
    @description: this is for experiment two, plot coordination statistic
    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_point, the start point (time) of all the data
    @param: end_point, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_classes, the conditions/cases/experiment_classes of the experimental data
    @return: show and save a data figure.

    '''
    #1) load data from file
    
    #1.1) local COG reflex data
    data_file_dic_COG = data_file_dic + "DFFB_reflexes_V2/"
    datas_of_experiment_classes=load_data_log(data_file_dic_COG)
    
    coordination_COG={}
    stability_COG={}
    displacement_COG={}

    for class_name, files_name in datas_of_experiment_classes: #name is a inclination names
        coordination_COG[class_name]=[]  #inclination is the table of the inclination name
        print(class_name)
        for idx in files_name.index:
            folder_name= data_file_dic_COG + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            gait_diagram_data, beta=gait(grf_data)
            temp_1=min([len(bb) for bb in beta]) #minimum steps of all legs
            beta=np.array([beta[0][:temp_1],beta[1][:temp_1],beta[2][:temp_1],beta[3][0:temp_1]]) # transfer to np array
            coordination_COG[class_name].append(1.0/max(np.std(beta,axis=0)))
            print(coordination_COG[class_name][-1])

    
    #1.2) local vestibular reflex data
    data_file_dic_Vesti = data_file_dic + "Vesti_reflexes_V2/"
    datas_of_experiment_classes=load_data_log(data_file_dic_Vesti)

    coordination_Vesti={}
    stability_Vesti={}
    displacement_Vesti={}

    for class_name, files_name in datas_of_experiment_classes: #name is a inclination names
        coordination_Vesti[class_name]=[]  #inclination is the table of the inclination name
        print(class_name)
        for idx in files_name.index:
            folder_name= data_file_dic_Vesti + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            gait_diagram_data, beta=gait(grf_data)
            temp_1=min([len(bb) for bb in beta]) #minimum steps of all legs
            beta=np.array([beta[0][:temp_1],beta[1][:temp_1],beta[2][:temp_1],beta[3][0:temp_1]]) # transfer to np array
            
            if(beta !=[]):
                coordination_Vesti[class_name].append(1.0/max(np.std(beta, axis=0)))# 
            else:
                coordination_Vesti[class_name].append(0.0)

            print(coordination_Vesti[class_name][-1])

    #3) plot

    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels= datas_of_experiment_classes.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    ind= np.arange(len(labels))
    width=0.15

    #3.1) plot 
    coordinationCOG_mean,coordinationCOG_std=[],[]
    coordinationVesti_mean,coordinationVesti_std=[],[]
    for i in labels:
        coordinationCOG_mean.append(np.mean(coordination_COG[i]))
        coordinationCOG_std.append(np.std(coordination_COG[i]))
        coordinationVesti_mean.append(np.mean(coordination_Vesti[i]))
        coordinationVesti_std.append(np.std(coordination_Vesti[i]))

    idx=0
    axs[idx].bar(ind-0.5*width,coordinationCOG_mean,width,yerr=coordinationCOG_std,label=r'DFFB reflexes')
    axs[idx].bar(ind+0.5*width,coordinationVesti_mean, width, yerr=coordinationVesti_std,label=r'Vestibular reflexes')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels([ str(round(180*float(l)/3.1415))+ u'\u00b0' for l in labels])
    axs[idx].legend()
    axs[idx].set_ylabel(r'Coordination')
    axs[idx].set_xlabel(r'Inclination of the floor: $\epsilon [degree]$')

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coordinationStatistic.svg'
    plt.savefig(figPath)
    plt.show()

def plot_displacement_statistic(data_file_dic,start_point=10,end_point=400,freq=60,experiment_classes=['0.0']):
    '''
    plot displacement statistic

    '''
    # 1) read data
    data_file_dic_COG=data_file_dic+"DFFB_reflexes_V2/"
    datas_of_experiment_classes=load_data_log(data_file_dic_COG)

    pose_COG={}

    for class_name, files_name in datas_of_experiment_classes: #name is a inclination names
        pose_COG[class_name] =[]
        for idx in files_name.index: # how many time experiments one inclination
            #folder_name= "/home/suntao/workspace/experiment_data/" + inclination['file_name'][idx]
            folder_name= data_file_dic_COG + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            pose_COG[class_name].append(np.sqrt(pow((pose_data[-1,3]-pose_data[0,3]),2)+pow(pose_data[-1,5]-pose_data[0,5],2)))
            print(folder_name)

    #1.2) read Vesti reflexes
    data_file_dic_Vesti=data_file_dic+"Vesti_reflexes_V2/"
    datas_of_experiment_classes=load_data_log(data_file_dic_Vesti)

    pose_Vesti={}

    for class_name, files_name in datas_of_experiment_classes: #name is a inclination names
        pose_Vesti[class_name] =[]
        for idx in files_name.index: # how many time experiments one inclination
            #folder_name= "/home/suntao/workspace/experiment_data/" + inclination['file_name'][idx]
            folder_name= data_file_dic_Vesti + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            pose_Vesti[class_name].append(np.sqrt(pow(pose_data[-1,3]-pose_data[0,3],2)+pow(pose_data[-1,5]-pose_data[0,5],2))) #Displacement on slopes 
            print(folder_name)

    #3) plot
    figsize=(5.1,4.1244)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.12,left=0.12,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    labels= datas_of_experiment_classes.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]
    ind= np.arange(len(labels))
    width=0.15

    #3.1) plot 
    disp_COG_mean, disp_COG_std=[],[]
    disp_Vesti_mean, disp_Vesti_std=[],[]
    for i in labels: #inclination
        disp_COG_mean.append(np.mean(pose_COG[i]))
        disp_COG_std.append(np.std(pose_COG[i]))

        disp_Vesti_mean.append(np.mean(pose_Vesti[i]))
        disp_Vesti_std.append(np.std(pose_Vesti[i]))

    idx=0
    axs[idx].bar(ind-0.5*width,disp_COG_mean,width,yerr=disp_COG_std,label=r'DFFB reflexes')
    axs[idx].bar(ind+0.5*width,disp_Vesti_mean, width, yerr=disp_Vesti_std,label=r'Vestibular reflexes')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_xticks(ind)
    #axs[idx].set_yticks([0,0.1,0.2,0.3])
    #axs[idx].set(ylim=[-0.01,0.3])
    axs[idx].set_xticklabels([ str(round(180*float(l)/3.1415))+ u'\u00b0' for l in labels])
    axs[idx].legend()
    axs[idx].set_ylabel(r'Displacement [m]')
    axs[idx].set_xlabel(r'Inclination of the floor: $\epsilon [degree]$')

    #axs[idx].set_title('Quadruped robot Walking on a slope using CPGs-based control with COG reflexes')
    
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'displacementStatistic.svg'
    plt.savefig(figPath)
    plt.show()

def plot_adaptiveOffset(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['0.0']):
    '''
    plot adaptive offsets

    '''
    # 1) read data
    #1) load data from file
    datas_of_experiment_classes=load_data_log(data_file_dic)
    
    CPGs={}
    gamma={}
    jmc={}
    modules={}

    for class_name, files_name in datas_of_experiment_classes: #name is a inclination names
        gamma[class_name]=[]  #inclination is the table of the inclination name
        jmc[class_name]=[]
        CPGs[class_name]=[]
        modules[class_name]=[]
        for idx in files_name.index:
            folder_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            gamma[class_name].append(COG_distribution(grf_data))
            CPGs[class_name].append(cpg_data)
            jmc[class_name].append(command_data)
            modules[class_name].append(module_data)

    # plot
    figsize=(6.1,5.2)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(8,len(experiment_classes))#13
    gs1.update(hspace=0.25,top=0.95,bottom=0.11,left=0.13,right=0.98)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))
        axs.append(fig.add_subplot(gs1[4:6,idx]))
        axs.append(fig.add_subplot(gs1[6:8,idx]))

    inclination='0.0'



    idx=0
    axs[idx].plot(time,CPGs[inclination][0][:,0],'r')
    axs[idx].plot(time,CPGs[inclination][0][:,1],'b')
    axs[idx].set_xticklabels([])
    axs[idx].set_xticks(np.linspace(time[0],time[-1],6))
    axs[idx].legend([r'$o_1$',r'$o_2$'],loc='upper right',ncol=2)
    axs[idx].set_yticks([-1.0,0.0,1.0])
    axs[idx].set_yticklabels([-1.0,0.0,1.0])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)],ylim=[-1.02,1.02])
    axs[idx].set_ylabel('CPG')


    idx=1
    axs[idx].plot(time,modules[inclination][0][:,1]*100.0,'r')
    axs[idx].plot(time,modules[inclination][0][:,2]*100.0,'b')
    axs[idx].set_xticklabels([])
    axs[idx].set_xticks(np.linspace(time[0],time[-1],6))
    axs[idx].legend([r'$w_1$',r'$w_2$'],loc='upper right',ncol=2)
    #axs[idx].set_yticks([0.0,0.007,0.014])
    #axs[idx].set_yticklabels([0.0,0.007,0.014])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    #axs[idx].set(xlim=[min(time),max(time)],ylim=[0.0,0.014])
    axs[idx].set(xlim=[min(time),max(time)])
    axs[idx].set_ylabel('Plastic weights')

    idx=2
    axs[idx].plot(time,jmc[inclination][0][:,1],'r')
    axs[idx].plot(time,jmc[inclination][0][:,2],'b')
    axs[idx].plot(time,modules[inclination][0][:,3],'r:')
    axs[idx].plot(time,modules[inclination][0][:,4],'b:')
    axs[idx].set_xticklabels([])
    axs[idx].set_xticks(np.linspace(time[0],time[-1],6))
    axs[idx].legend([r'$\theta_1$',r'$\beta_1$',r'$\theta_2$',r'$\beta_2$'],loc='upper right', ncol=2)
    axs[idx].set_yticks([-0.4,-0.2,0,.2])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)])
    axs[idx].set_ylabel('MNs')

    idx=3
    axs[idx].plot(time,gamma[inclination][0],color='C0')
    df=pd.DataFrame(gamma[inclination][0],columns=['gamma'])
    average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
    average_average_gamma=average_gamma.rolling(60).mean().fillna(df.iloc[0])
    axs[idx].plot(time,average_average_gamma,color='C1')
    axs[idx].plot(time,1.1*np.ones(average_average_gamma.shape),'-.',color='C2')
    axs[idx].legend([r'$\gamma^a$',r'$\bar{\gamma}^a$',r'$\bar{\gamma}^d$'],loc='upper right',ncol=3)
    axs[idx].set_xlabel('Time [s]')
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_ylabel('GRFs distribution')
    axs[idx].set_yticks([0.0,1.0,2.0,3.0,4.0])
    axs[idx].set_yticklabels([0.0,1.0,2.0,3.0,4.0])
    axs[idx].set_xticks(np.linspace(time[0],time[-1],6))

    idx=3
    axs[idx].set_xticklabels([str(idx) for idx in range(int(round(time[-1]-time[0]+1)))])
    #axs[idx].set_xticks(np.linspace(time[0],time[-1],6),[str(idx) for idx in range(int(round(time[-1]-time[0]+1)))])
    axs[idx].set(xlim=[min(time),max(time)])
    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'adaptive_offset.svg'
    plt.savefig(figPath)
    plt.show()

def Experiment_metrics(data_file_dic,start_point=300,end_point=660,freq=60.0,experiment_classes=['0']):
    '''
    Demonstrating stability, coordination, and displacement. These are the performance metrics of the Quadruped locomotion
    In this sub module, the accelerate, body orientation/ attitude, and GRFs are required.

    '''

    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)

    pose={}
    gamma={}
    jmc={}
    #1.2) read data one by one file
    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        gamma[class_name]=[]  #files_name is the table of the files_name class_name
        jmc[class_name]=[]
        pose[class_name]=[]
        for idx in files_name.index:
            folder_class_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_class_name)
            # 2)  data process
            print(folder_class_name)
            gamma[class_name].append(COG_distribution(grf_data))
            pose[class_name].append(pose_data)
            jmc[class_name].append(command_data)

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


    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'experiment_metric.svg'
    plt.savefig(figPath)

    # show 
    plt.show()

def Experiment1(data_file_dic,start_point=300,end_point=660,freq=60.0,experiment_classes=['0']):
    '''
    @description: plot theta1, theta2;  gamma_d, gamma_a; posture, pitch
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

    pose={}
    gamma={}
    jmc={}
    #1.2) read data one by one file
    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        gamma[class_name]=[]  #files_name is the table of the files_name class_name
        jmc[class_name]=[]
        pose[class_name]=[]
        for idx in files_name.index:
            folder_class_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_class_name)
            # 2)  data process
            print(folder_class_name)
            gamma[class_name].append(COG_distribution(grf_data))
            pose[class_name].append(pose_data)
            jmc[class_name].append(command_data)

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

    legends=[r'$C_1$',r'$C_2$',r'$C_3$',r'$C_4$',r'$C_5$',r'$C_6$',r'$C_7$']
    colorList=['r','g','b','k','y','c','m']
    labels= datas_of_experiment_classes.groups.keys()
    idx=0
    for index, class_name in enumerate(labels): # name is a inclination names
        axs[idx].plot(time,jmc[class_name][0][:,1],color=colorList[index])
    axs[idx].set_xticklabels([])
    axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    axs[idx].set_yticks([-0.5,0.0,0.5])
    axs[idx].set_yticklabels([-0.5,0.0,0.5])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.55,0.51])
    axs[idx].set_ylabel(r'$\theta_1$')

    idx=1
    for index, class_name in enumerate(labels): # name is a inclination names
        axs[idx].plot(time,jmc[class_name][0][:,2],color=colorList[index])
    axs[idx].set_xticklabels([])
    axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    axs[idx].set_yticks([-0.4,0.0,0.4])
    axs[idx].set_yticklabels([-0.4,0.0,0.4])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.5,0.5])
    axs[idx].set_ylabel(r'$\theta_2$')

    idx=2
    for index, class_name in enumerate(labels): # name is a inclination names
        df=pd.DataFrame(gamma[class_name][0],columns=['gamma'])
        average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
        average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
        axs[idx].plot(time,average_average_gamma,color=colorList[index])

    axs[idx].plot(time,1.0*np.ones(average_average_gamma.shape),'-.')
    axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_ylabel(r'$\bar{\gamma}$')
    axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.1,2.1])
    axs[idx].set_xticklabels([])
    axs[idx].set_yticks([0.0,1.0,2.0])
    axs[idx].set_yticklabels([0.0,1.0,2.0])


    idx=3
    for index, class_name in enumerate(labels): # name is a inclination names
        axs[idx].plot(time,pose[class_name][0][:,1],color=colorList[index])
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


    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'experiment_1.svg'
    #plt.savefig(figPath)

    plt.show()
    


def Experiment1_VersionTwo(data_file_dic,start_point=300,end_point=660,freq=60.0,experiment_classes=['0']):
    '''
    @description: plot theta1, theta2;  gamma_d, gamma_a; posture, pitch
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

    pose={}
    gamma={}
    jmc={}
    modules={}
    #1.2) read data one by one file
    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        gamma[class_name]=[]  #files_name is the table of the files_name class_name
        jmc[class_name]=[]
        pose[class_name]=[]
        modules[class_name]=[]
        for idx in files_name.index:
            folder_class_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_class_name)
            # 2)  data process
            print(folder_class_name)
            gamma[class_name].append(COG_distribution(grf_data))
            pose[class_name].append(pose_data)
            jmc[class_name].append(command_data)
            modules[class_name].append(module_data)

    # plot
    figsize=(6.,8.2)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(16,1)#13
    gs1.update(hspace=0.22,top=0.95,bottom=0.11,left=0.11,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))
    axs.append(fig.add_subplot(gs1[2:4,0]))
    axs.append(fig.add_subplot(gs1[4:6,0]))
    axs.append(fig.add_subplot(gs1[6:8,0]))
    axs.append(fig.add_subplot(gs1[8:10,0]))
    axs.append(fig.add_subplot(gs1[10:12,0]))
    axs.append(fig.add_subplot(gs1[12:14,0]))
    axs.append(fig.add_subplot(gs1[14:16,0]))

    legends=[r'$C_1$',r'$C_2$',r'$C_3$',r'$C_4$',r'$C_5$',r'$C_6$',r'$C_7$']
    colorList=['r','g','b','k','y','c','m']
    labels= datas_of_experiment_classes.groups.keys()
    labels=[str(ll) for ll in sorted([ float(ll) for ll in labels])]



    idx=0
    for index, class_name in enumerate(labels): # name is a inclination names
        axs[idx].plot(time,modules[class_name][0][:,1]*100,color=colorList[index], linestyle='solid')
        #axs[idx].plot(time,modules[class_name][0][:,2],color=colorList[index], linestyle='dotted')
    axs[idx].set_xticks(np.linspace(time[0],time[-1],9))
    axs[idx].set_xticklabels([])
    #axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    #axs[idx].set_yticks([-0.5,0.0,0.5])
    #axs[idx].set_yticklabels([-0.5,0.0,0.5])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)])
    axs[idx].set_ylabel(r'$w_1$')

    idx=1
    for index, class_name in enumerate(labels): # name is a inclination names
        #axs[idx].plot(time,modules[class_name][0][:,1]*100,color=colorList[index], linestyle='solid')
        axs[idx].plot(time,modules[class_name][0][:,2]*100,color=colorList[index], linestyle='solid')
    axs[idx].set_xticks(np.linspace(time[0],time[-1],9))
    axs[idx].set_xticklabels([])
    #axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    #axs[idx].set_yticks([-0.5,0.0,0.5])
    #axs[idx].set_yticklabels([-0.5,0.0,0.5])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)])
    axs[idx].set_ylabel(r'$w_2$')

    idx=2
    for index, class_name in enumerate(labels): # name is a inclination names
        axs[idx].plot(time,modules[class_name][0][:,3],color=colorList[index], linestyle='solid')
        #axs[idx].plot(time,modules[class_name][0][:,4],color=colorList[index], linestyle='dotted')
    axs[idx].set_xticks(np.linspace(time[0],time[-1],9))
    axs[idx].set_xticklabels([])
    #axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    #axs[idx].set_yticks([-0.5,0.0,0.5])
    #axs[idx].set_yticklabels([-0.5,0.0,0.5])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)])
    axs[idx].set_ylabel(r'$\beta_1$')


    idx=3
    for index, class_name in enumerate(labels): # name is a inclination names
        #axs[idx].plot(time,modules[class_name][0][:,3],color=colorList[index], linestyle='solid')
        axs[idx].plot(time,modules[class_name][0][:,4],color=colorList[index], linestyle='solid')
    axs[idx].set_xticks(np.linspace(time[0],time[-1],9))
    axs[idx].set_xticklabels([])
    #axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    #axs[idx].set_yticks([-0.5,0.0,0.5])
    #axs[idx].set_yticklabels([-0.5,0.0,0.5])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)])
    axs[idx].set_ylabel(r'$\beta_2$')


    idx=4
    for index, class_name in enumerate(labels): # name is a inclination names
        axs[idx].plot(time,jmc[class_name][0][:,1],color=colorList[index])
    axs[idx].set_xticks(np.linspace(time[0],time[-1],9))
    axs[idx].set_xticklabels([])
    #axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    axs[idx].set_yticks([-0.5,0.0,0.5])
    axs[idx].set_yticklabels([-0.5,0.0,0.5])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.55,0.51])
    axs[idx].set_ylabel(r'$\theta_1$')

    idx=5
    for index, class_name in enumerate(labels): # name is a inclination names
        axs[idx].plot(time,jmc[class_name][0][:,2],color=colorList[index])
    axs[idx].set_xticks(np.linspace(time[0],time[-1],9))
    axs[idx].set_xticklabels([])
    #axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    axs[idx].set_yticks([-0.4,0.0,0.4])
    axs[idx].set_yticklabels([-0.4,0.0,0.4])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.5,0.5])
    axs[idx].set_ylabel(r'$\theta_2$')

    idx=6
    for index, class_name in enumerate(labels): # name is a inclination names
        df=pd.DataFrame(gamma[class_name][0],columns=['gamma'])
        average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
        average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
        axs[idx].plot(time,average_average_gamma,color=colorList[index])

    axs[idx].plot(time,1.1*np.ones(average_average_gamma.shape),'-.')
    #axs[idx].legend(legends,loc='upper right',ncol=len(legends))
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_ylabel(r'$\bar{\gamma}$')
    axs[idx].set(xlim=[min(time),max(time)],ylim=[-0.1,2.1])
    axs[idx].set_xticks(np.linspace(time[0],time[-1],9))
    axs[idx].set_xticklabels([])
    axs[idx].set_yticks([0.0,1.5,3.0])
    axs[idx].set_yticklabels([0.0,1.5,3.0])


    idx=7
    for index, class_name in enumerate(labels): # name is a inclination names
        axs[idx].plot(time,pose[class_name][0][:,1],color=colorList[index])
    axs[idx].set_xticks(np.linspace(time[0],time[-1],9))
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


    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'experiment_1.svg'
    plt.savefig(figPath)

    plt.show()
    

def Experiment1_1(data_file_dic,start_point=300,end_point=660,freq=60.0,experiment_classes=['0'],initial_offsets='0.3'):
    '''
    @Description: This is for experiment one, plot gait diagram, roll and pitch, walking displacement.
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
    pose={}
    grf={}
    #1.2) read data one by one file
    for class_name, files_name in datas_of_experiment_classes: #name is a inclination names
        grf[class_name]=[]  #inclination is the table of the inclination name
        pose[class_name]=[]
        for idx in files_name.index:
            folder_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_name)
            # 2)  data process
            print(folder_name)
            pose[class_name].append(pose_data)
            grf[class_name].append(grf_data)

    # plot
    figsize=(6,4)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)
    gs1.update(hspace=0.26,top=0.95,bottom=0.11,left=0.11,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))
    axs.append(fig.add_subplot(gs1[2:4,0]))
    axs.append(fig.add_subplot(gs1[4:6,0]))

    idx=0
    gait_diagram_data, duty_factor_data=gait(grf[initial_offsets][0])
    gait_diagram(fig,axs[idx],gs1,gait_diagram_data)
    axs[idx].text(-0.1,0.4,'Gait',rotation='vertical')


    idx=1 # roll and pitch
    axs[idx].plot(time,pose[initial_offsets][0][:,0]*57.0,color=st_r_color)
    axs[idx].plot(time,pose[initial_offsets][0][:,1]*57.0,color=st_b_color)
    axs[idx].legend(['roll','pitch'],loc='upper right',ncol=2)
    axs[idx].set_xticklabels([])
    axs[idx].set_yticks([-12,-6,0.0,6,12])
    #axs[idx].set_yticklabels([-0.1,0.0,0.1])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set(xlim=[min(time),max(time)],ylim=[-12,12])
    axs[idx].set_ylabel('Attitude [degree]')

    idx=2 #displacemnet
    displacement = np.sqrt(pow(pose[initial_offsets][0][:,3],2) + pow(pose[initial_offsets][0][:,4], 2)) #Displacement on slopes 
    axs[idx].plot(time,displacement,'r')
    #axs[idx].set_yticks([-0.5,0.0,0.4])
    #axs[idx].set_yticklabels([-0.5,0.0,0.4])
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_yticks([0.0,0.2,0.4,0.6,0.7])
    axs[idx].set(xlim=[min(time),max(time)],ylim=[0.0,0.8])
    axs[idx].set_xticklabels([str(idx) for idx in range(int(time[-1]-time[0]+1))])
    axs[idx].set_xlabel('Time [s]')
    axs[idx].set_ylabel('Displacement [m]')


    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'experiment_1_1.svg'
    #plt.savefig(figPath)

    plt.show()
    
def Experiment2_1(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['0.0'],trail_id=0):
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

    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        gamma[class_name]=[]  #files_name is the table of the files_name class_name
        gait_diagram_data[class_name]=[]
        beta[class_name]=[]
        pose[class_name]=[]
        jmc[class_name]=[]
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

            temp_1=min([len(bb) for bb in beta_temp]) #minimum steps of all legs
            beta_temp2=np.array([beta_temp[0][:temp_1],beta_temp[1][:temp_1],beta_temp[2][:temp_1],beta_temp[3][0:temp_1]]) # transfer to np array
            if(beta_temp2 !=[]):
                print("Coordination:",1.0/max(np.std(beta_temp2, axis=0)))
            else:
                print("Coordination:",0.0)

            print("Stability:",1.0/np.std(pose_data[:,0],axis=0))
            print("Displacemment:",np.sqrt(pow(pose_data[-1,3]-pose_data[0,3],2)+pow(pose_data[-1,5]-pose_data[0,5],2))) #Displacement on slopes 
    #3) plot
    figsize=(6.2,6.)
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
        axs[3*idx].plot(time,jmc[inclination][trail_id][:,1], 'r:')
        axs[3*idx].plot(time,jmc[inclination][trail_id][:,2],'r-.')
        axs[3*idx].plot(time,jmc[inclination][trail_id][:,10],'b:')
        axs[3*idx].plot(time,jmc[inclination][trail_id][:,11],'b-.')
        axs[3*idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx].grid(which='both',axis='y',color='k',linestyle=':')
        axs[0].set_ylabel(u'Joint commands')
        axs[3*idx].set_yticks([-0.8,-0.4,0.0,0.5])
        axs[3*idx].legend(['RF hip','RF knee','LH hip', 'LH knee'],ncol=4)
        axs[3*idx].set_xticklabels([])
        axs[3*idx].set_title('Inclination of the slope:' + str(round(180*float(inclination)/3.1415))+ u'\u00b0')
        axs[3*idx].set(xlim=[min(time),max(time)])


        axs[1].set_ylabel(r'$\bar{\gamma}$')
        print(gamma[inclination][0])
        df=pd.DataFrame(gamma[inclination][trail_id],columns=['gamma'])
        average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
        average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
        axs[3*idx+1].plot(time,average_average_gamma,'r')
        axs[3*idx+1].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+1].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+1].set_yticks([0.0,1.0,2.0])
        axs[3*idx+1].set_xticklabels([])
        axs[3*idx+1].set(xlim=[min(time),max(time)],ylim=[-0.1,2.1])

        axs[2].set_ylabel(u'Atti [deg]')
        axs[3*idx+2].plot(time,pose[inclination][trail_id][:,0]*-57.3,'r')
        axs[3*idx+2].plot(time,pose[inclination][trail_id][:,1]*-57.3,'b')
        axs[3*idx+2].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+2].grid(which='both',axis='y',color='k',linestyle=':')
        #axs[3*idx+2].set_yticks([-40,-20,0.0])
        axs[3*idx+2].legend(['Roll','Pitch'])
        axs[3*idx+2].set_xticklabels([])
        axs[3*idx+2].set(xlim=[min(time),max(time)])

        axs[3].set_ylabel(u'Disp. [m]')
        displacement = np.sqrt(pow(pose[inclination][trail_id][:,3],2) + pow(pose[inclination][trail_id][:,5], 2)) #Displacement on slopes 
        axs[3*idx+3].plot(time,displacement,'r')
        axs[3*idx+3].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+3].grid(which='both',axis='y',color='k',linestyle=':')
        #axs[3*idx+3].set_yticks([0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0])
        axs[3*idx+3].set_xticklabels([])
        axs[3*idx+3].set(xlim=[min(time),max(time)])

        axs[4].set_ylabel(r'Gait')
        gait_diagram(fig,axs[3*idx+4],gs1,gait_diagram_data[inclination][trail_id])
        axs[3*idx+4].set_xlabel(u'Time [s]')
        xticks=np.arange(int(min(time)),int(max(time))+1,2)
        axs[3*idx+4].set_xticklabels([str(xtick) for xtick in xticks])
        axs[3*idx+4].set_xticks(xticks)
        axs[4].yaxis.set_label_coords(-0.1,.5)
        axs[3*idx+4].set(xlim=[min(time),max(time)])

    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'experiment2_1.svg'
    plt.savefig(figPath)

    plt.show()

def Experiment2_1_repeat(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['0.0']):
    ''' 
    This is for experiment two, for the first figure with five trails on inclination

    '''
    #1) load data from file
    datas_of_experiment_classes=load_data_log(data_file_dic)

    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    pose={}
    jmc={}

    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        gamma[class_name]=[]  #files_name is the table of the files_name class_name
        gait_diagram_data[class_name]=[]
        beta[class_name]=[]
        pose[class_name]=[]
        jmc[class_name]=[]
        for idx in files_name.index:
            folder_class_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_class_name)
            # 2)  data process
            print(folder_class_name)
            gamma[class_name].append(COG_distribution(grf_data))

            gait_diagram_data_temp, beta_temp=gait(grf_data)
            gait_diagram_data[class_name].append(gait_diagram_data_temp); beta[class_name].append(beta_temp)

            pose[class_name].append(pose_data)
            jmc[class_name].append(command_data[:,0:3])

    #3) plot
    figsize=(5.5,7.)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(9,len(experiment_classes))#13
    gs1.update(hspace=0.13,top=0.95,bottom=0.1,left=0.11,right=0.98)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))
        axs.append(fig.add_subplot(gs1[4:6,idx]))
        axs.append(fig.add_subplot(gs1[6:8,idx]))
        axs.append(fig.add_subplot(gs1[8:9,idx]))

    #3.1) plot 
    for idx, inclination in enumerate(experiment_classes):
        columns=['j'+str(n) for n in range(1,4)]
        columns.insert(0,'time')
        data=pd.DataFrame(np.insert(jmc[inclination][0],0,values=time,axis=1),columns=columns)
        for number in range(1,len(jmc[inclination])):
            data1=pd.DataFrame(np.insert(jmc[inclination][number],0,values=time,axis=1),columns=columns)
            data=pd.concat([data,data1],axis=0)
            
        sns.lineplot(x='time',y='j2',data=data,ax=axs[3*idx])
        sns.lineplot(x='time',y='j3',data=data,ax=axs[3*idx])
        axs[3*idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx].grid(which='both',axis='y',color='k',linestyle=':')
        axs[0].set_ylabel(u'Joint commands')
        axs[3*idx].set_yticks([-1.0,-0.5,0.0,0.5,1.0])
        axs[3*idx].legend([r'RF $\theta_1$',r'RF $\theta_2$'],ncol=2)
        axs[3*idx].set_xticklabels([])
        axs[3*idx].set_title('Inclination of the slope:' + str(round(180*float(inclination)/3.1415))+ u'\u00b0')
        axs[3*idx].set(xlim=[min(time),max(time)],ylim=[-1.1,1.1])
        
        df=pd.DataFrame(np.insert(gamma[inclination][0],0,values=time,axis=1),columns=['time','gamma'])
        average_gamma=df.rolling(50).mean().fillna(df.iloc[1])
        average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[1])
        for number in range(1,len(gamma[inclination])):
            df=pd.DataFrame(np.insert(gamma[inclination][number],0,values=time,axis=1),columns=['time','gamma'])
            average_gamma=df.rolling(50).mean().fillna(df.iloc[1])
            average_average_gamma1=average_gamma.rolling(50).mean().fillna(df.iloc[1])
            average_average_gamma=pd.concat([average_average_gamma,average_average_gamma1],axis=0)
        
        sns.lineplot(x='time',y= 'gamma', data=average_average_gamma,ax=axs[3*idx+1])
        axs[3*idx+1].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+1].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+1].set_yticks([0.0,1.0,2.0,3.0])
        axs[3*idx+1].set_xticklabels([])
        axs[3*idx+1].set(xlim=[min(time),max(time)])
        axs[3*idx+1].set_ylabel(r'$\bar{\gamma}$')


        atti=pd.DataFrame(np.insert(pose[inclination][0][:,0:3]*-57.3,0,values=time,axis=1),columns=['time','roll','pitch','yaw'])
        for number in range(1,len(pose[inclination])):
            atti1=pd.DataFrame(np.insert(pose[inclination][number][:,0:3]*-57.3,0,values=time,axis=1),columns=['time','roll','pitch','yaw'])
            atti=pd.concat([atti,atti1],axis=0)
        sns.lineplot(x='time',y='roll',data=atti,ax=axs[3*idx+2])
        sns.lineplot(x='time',y='pitch',data=atti,ax=axs[3*idx+2])
        axs[3*idx+2].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+2].grid(which='both',axis='y',color='k',linestyle=':')
        #axs[3*idx+2].set_yticks([-10,0,10,20,30,40])
        axs[3*idx+2].set_yticks([-40,-30,-20,-10,0,10])
        axs[3*idx+2].set_xticklabels([])
        axs[3*idx+2].set(xlim=[min(time),max(time)])
        axs[3*idx+2].legend(['Roll','Pitch'],ncol=2)
        axs[3*idx+2].set_ylabel(u'Atti. [deg]')


        distance=np.sqrt(pow(pose[inclination][0][:,3],2) + pow(pose[inclination][0][:,5], 2)).reshape((-1,1))
        disp = pd.DataFrame(np.insert(distance,0,values=time,axis=1),columns=['time','disp']) #Displacement on slopes 
        for number in range(1,len(pose[inclination])):
            distance=np.sqrt(pow(pose[inclination][number][:,3],2) + pow(pose[inclination][number][:,5], 2)).reshape((-1,1))
            disp1 = pd.DataFrame(np.insert(distance,0,values=time,axis=1),columns=['time','disp']) #Displacement on slopes 
            disp=pd.concat([disp,disp1],axis=0)
        sns.lineplot(x='time',y='disp',data=disp, ax=axs[3*idx+3])
        axs[3*idx+3].grid(which='both',axis='x',color='k', linestyle=':')
        axs[3*idx+3].grid(which='both',axis='y',color='k', linestyle=':')
        axs[3*idx+3].set_yticks([0.0,0.5,1.0])
        axs[3*idx+3].set_xticklabels([])
        axs[3*idx+3].set(xlim=[min(time),max(time)],ylim=[-0.1,1.5])
        axs[3*idx+3].set_ylabel(u'Disp. [m]')

        axs[4].set_ylabel(r'Gait')
        gait_diagram(fig,axs[3*idx+4],gs1,gait_diagram_data[inclination][0])
        axs[3*idx+4].set_xlabel(u'Time [s]')
        xticks=np.arange(int(min(time)),int(max(time))+1,2)
        axs[3*idx+4].set_xticklabels([str(xtick) for xtick in xticks])
        axs[3*idx+4].set_xticks(xticks)
        axs[3*idx+4].yaxis.set_label_coords(-0.1,.5)

    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath=folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithVestibularandCOGReflex_singleCurve.svg'
    plt.savefig(figPath)

    plt.show()

def Experiment3(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['0.0']):
    '''
    Experiment data analysis for expriment three

    '''
    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)

    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    displacement={}
    jmc={}

    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        gamma[class_name]=[]  #files_name is the table of the files_name class_name
        gait_diagram_data[class_name]=[]
        beta[class_name]=[]
        pitch[class_name]=[]
        displacement[class_name]=[]
        jmc[class_name]=[]
        for idx in files_name.index:
            folder_class_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_class_name)
            # 2)  data process
            print(folder_class_name)
            gamma[class_name].append(COG_distribution(grf_data))

            gait_diagram_data_temp, beta_temp=gait(grf_data)
            gait_diagram_data[class_name].append(gait_diagram_data_temp); beta[class_name].append(beta_temp)

            pitch[class_name].append(pose_data[:,1])

            displacement[class_name].append(pose_data[:,3]/np.cos(float(class_name)))
            jmc[class_name].append(command_data)

    
    #3) plot
    figsize=(7.6,6.2)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,len(experiment_classes))#13
    gs1.update(hspace=0.12,top=0.95,bottom=0.09,left=0.08,right=0.98)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:3,idx]))
        axs.append(fig.add_subplot(gs1[3:4,idx]))
        axs.append(fig.add_subplot(gs1[4:5,idx]))
        axs.append(fig.add_subplot(gs1[5:6,idx]))

    #3.1) plot 
    for idx, inclination in enumerate(experiment_classes):
        axs[3*idx].plot(time,jmc[inclination][0][:,1], color=st_r_color,linewidth=0.8)
        axs[3*idx].plot(time,jmc[inclination][0][:,2], color=st_b_color,linewidth=0.8)
        #axs[3*idx].plot(time,jmc[inclination][0][:,10],'b:')
        #axs[3*idx].plot(time,jmc[inclination][0][:,11],'b-.')
        axs[3*idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx].grid(which='both',axis='y',color='k',linestyle=':')
        axs[0].set_ylabel(u'Joint commands')
        axs[3*idx].set_yticks([-0.6,-0.4,0.0,0.4,0.6])
        axs[3*idx].legend(['RF hip','RF knee','LH hip', 'LH knee'],ncol=4)
        axs[3*idx].set_xticklabels([])
        axs[3*idx].set_title('Inclination of the slope:' + str(round(180*float(inclination)/3.1415))+ u'\u00b0')
        axs[3*idx].set(xlim=[min(time),max(time)])

        axs[1].set_ylabel(r'$\bar{\gamma}$')
        df=pd.DataFrame(gamma[inclination][0],columns=['gamma'])
        average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
        average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
        axs[3*idx+1].plot(time,average_average_gamma,'r',linewidth=0.8)
        axs[3*idx+1].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+1].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+1].set_yticks([0.0,1.0, 1.5, 2.0])
        axs[3*idx+1].set_xticklabels([])
        axs[3*idx+1].set(xlim=[min(time),max(time)],ylim=[-0.1,2.1])

        axs[2].set_ylabel(u'Pitch [deg]')
        axs[3*idx+2].plot(time,-1.0*pitch[inclination][0]/3.1415926*180,'r',linewidth=0.8)
        axs[3*idx+2].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+2].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+2].set_yticks([-35,-15,0,15,35])
        axs[3*idx+2].set_xticklabels([])
        axs[3*idx+2].set(xlim=[min(time),max(time)],ylim=[-35,35])

        axs[3].set_ylabel(u'Disp. [m]')
        axs[3*idx+3].plot(time, displacement[inclination][0]-displacement[inclination][0][0],'r',linewidth=0.8)
        axs[3*idx+3].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+3].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+3].set_yticks([0,2,4,6])
        axs[3*idx+3].set_xticklabels([])
        axs[3*idx+3].set(xlim=[min(time),max(time)],ylim=[0,6.2])

        axs[4].set_ylabel(r'Gait')
        gait_diagram(fig,axs[3*idx+4],gs1,gait_diagram_data[inclination][0])
        axs[3*idx+4].set_xlabel(u'Time [s]')
        xticks=np.arange(0,int(max(time)),10)
        axs[3*idx+4].set_xticklabels([str(xtick) for xtick in xticks])
        axs[3*idx+4].set_xticks(xticks)
        axs[4].yaxis.set_label_coords(-0.10,.5)

    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath=folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithVestibularandCOGReflex_singleCurve.svg'
    plt.savefig(figPath)

    plt.show()



def Experiment_Laikago(data_file_dic,start_point=90,end_point=1200,freq=60.0,experiment_classes=['0.0']):
    '''
    Experiment data analysis for Laikago expriment

    '''
    # 1) read data
    datas_of_experiment_classes=load_data_log(data_file_dic)

    gamma={}
    beta={}
    gait_diagram_data={}
    pitch={}
    displacement={}
    jmc={}

    for class_name, files_name in datas_of_experiment_classes: #class_name is a files_name class_names
        gamma[class_name]=[]  #files_name is the table of the files_name class_name
        gait_diagram_data[class_name]=[]
        beta[class_name]=[]
        pitch[class_name]=[]
        displacement[class_name]=[]
        jmc[class_name]=[]
        for idx in files_name.index:
            folder_class_name= data_file_dic + files_name['file_name'][idx]
            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_class_name)
            # 2)  data process
            print(folder_class_name)
            gamma[class_name].append(COG_distribution(grf_data))

            gait_diagram_data_temp, beta_temp=gait(grf_data)
            gait_diagram_data[class_name].append(gait_diagram_data_temp); beta[class_name].append(beta_temp)

            pitch[class_name].append(pose_data[:,1])

            displacement[class_name].append(pose_data[:,3]/np.cos(float(class_name)))
            jmc[class_name].append(command_data)

    
    #3) plot
    figsize=(7.6,6.2)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,len(experiment_classes))#13
    gs1.update(hspace=0.12,top=0.95,bottom=0.09,left=0.08,right=0.98)
    axs=[]
    for idx in range(len(experiment_classes)):# how many columns, depends on the experiment_classes
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:3,idx]))
        axs.append(fig.add_subplot(gs1[3:4,idx]))
        axs.append(fig.add_subplot(gs1[4:5,idx]))
        axs.append(fig.add_subplot(gs1[5:6,idx]))

    #3.1) plot 
    for idx, inclination in enumerate(experiment_classes):
        axs[3*idx].plot(time,jmc[inclination][0][:,1], color=st_r_color,linewidth=0.8)
        axs[3*idx].plot(time,jmc[inclination][0][:,2], color=st_b_color,linewidth=0.8)
        #axs[3*idx].plot(time,jmc[inclination][0][:,10],'b:')
        #axs[3*idx].plot(time,jmc[inclination][0][:,11],'b-.')
        axs[3*idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx].grid(which='both',axis='y',color='k',linestyle=':')
        axs[0].set_ylabel(u'Joint commands')
        axs[3*idx].set_yticks([-0.9,-0.6,-0.3,0.0,0.2])
        axs[3*idx].legend(['RF hip','RF knee','LH hip', 'LH knee'],ncol=4)
        axs[3*idx].set_xticklabels([])
        axs[3*idx].set_title('Inclination of the slope:' + str(round(180*float(inclination)/3.1415))+ u'\u00b0')
        axs[3*idx].set(xlim=[min(time),max(time)])

#        axs[1].set_ylabel(r'$\bar{\gamma}$')
#        df=pd.DataFrame(gamma[inclination][0],columns=['gamma'])
#        average_gamma=df.rolling(50).mean().fillna(df.iloc[0])
#        average_average_gamma=average_gamma.rolling(50).mean().fillna(df.iloc[0])
#        axs[3*idx+1].plot(time,average_average_gamma,'r',linewidth=0.8)
#        axs[3*idx+1].grid(which='both',axis='x',color='k',linestyle=':')
#        axs[3*idx+1].grid(which='both',axis='y',color='k',linestyle=':')
#        axs[3*idx+1].set_yticks([0.0,1.0, 1.5, 2.0])
#        axs[3*idx+1].set_xticklabels([])
#        axs[3*idx+1].set(xlim=[min(time),max(time)],ylim=[-0.1,2.1])

        axs[2].set_ylabel(u'Pitch [deg]')
        axs[3*idx+2].plot(time,-1.0*pitch[inclination][0]/3.1415926*180,'r',linewidth=0.8)
        axs[3*idx+2].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+2].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+2].set_yticks([-10,0,15,30,45,60])
        axs[3*idx+2].set_xticklabels([])
        axs[3*idx+2].set(xlim=[min(time),max(time)],ylim=[-10,60])

        axs[3].set_ylabel(u'Disp. [m]')
        axs[3*idx+3].plot(time, displacement[inclination][0]-displacement[inclination][0][0],'r',linewidth=0.8)
        axs[3*idx+3].grid(which='both',axis='x',color='k',linestyle=':')
        axs[3*idx+3].grid(which='both',axis='y',color='k',linestyle=':')
        axs[3*idx+3].set_yticks([0,0.5,1.0,1.5,2.0])
        axs[3*idx+3].set_xticklabels([])
        axs[3*idx+3].set(xlim=[min(time),max(time)],ylim=[0,2.2])

        axs[4].set_ylabel(r'Gait')
        gait_diagram(fig,axs[3*idx+4],gs1,gait_diagram_data[inclination][0])
        axs[3*idx+4].set_xlabel(u'Time [s]')
        xticks=np.arange(0,int(max(time)),10)
        axs[3*idx+4].set_xticklabels([str(xtick) for xtick in xticks])
        axs[3*idx+4].set_xticks(xticks)
        axs[4].yaxis.set_label_coords(-0.10,.5)

    # save figure
    folder_fig = data_file_dic + 'data_visulization/'
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath=folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + 'coupledCPGsWithVestibularandCOGReflex_singleCurve.svg'
    plt.savefig(figPath)

    plt.show()



if __name__=="__main__":
    ''' touch moment analysis '''
    #slopeWalking_touchMomentAnalysis(data_file_dic)
    ''' all single parameters '''
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #plot_slopeWalking_gamma_beta_pitch_displacement(data_file_dic,start_point=500,end_point=1500,freq=60.0,experiment_classes=['0.0','0.174','0.349','0.52','0.61'])
    ''' expected and actuall grf comparison'''
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #plot_comparasion_expected_actual_grf_all_leg(data_file_dic,start_point=1,end_point=1000,freq=60.0,experiment_classes=['0.0'])

    ''' actuall grfs'''
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #plot_actual_grf_all_leg(data_file_dic,start_point=100,end_point=3800,freq=60.0,experiment_classes=['-0.2'])

    ''' esn identification'''
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #esn_identification(data_file_dic,start_point =400,end_point=1500,freq=60.0,experiment_classes=['0.0'])

    ''' Plot gait diagram and duty factor '''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/COG_reflexes/" # Experiment metric demo
    #plot_gait_dutyFactor(data_file_dic,start_point=1150,end_point=1380,freq=60.0,experiment_classes=['0.0'])

    ''' Plot body attitude angle and accelerate '''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/COG_reflexes/"
    #plot_attitude_accelerate(data_file_dic,start_point=1200,end_point=1380,freq=60.0,experiment_classes=['0.0'])

    ''' This is for ploting CPG offset and GRFs distribution in method section   '''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/COG_reflexes/"
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/Method_Fig6/Version Two/WithDIL/"
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #plot_adaptiveOffset(data_file_dic,start_point=340,end_point=640+120,freq=60.0,experiment_classes=['0.0']) #Fig. 5 In method section

    ''' This is for experiment indexes demonstration '''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/"
    #Experiment_indexes(data_file_dic,start_point=180,end_point=720,freq=60.0,experiment_classes=['0'])

    ''' This is for experiemnt 1-1 a particular case (C1 or C5)'''
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/COG_reflexes/"
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/DFFB_reflexes_V2/"
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/Vesti_reflexes_V2/"
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #Experiment1_1(data_file_dic,start_point=180,end_point=840,freq=60.0,experiment_classes=['0.0'],initial_offsets='0.3')

    ''' This is for experiment 1  for all case  to plot joint commands'''
    #Experiment1(data_file_dic,start_point=180,end_point=720,freq=60.0,experiment_classes=['0'])
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/Vesti_reflexes_V2/"
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #Experiment1(data_file_dic,start_point=220,end_point=720,freq=60.0,experiment_classes=['0'])
    data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/experiment1_1/DFFB_reflexes_V2/VersionTwo/"
    Experiment1_VersionTwo(data_file_dic,start_point=220+240,end_point=720+240+60,freq=60.0,experiment_classes=['0'])

    ''' This is for experiment 2_1 one trail for a inclination'''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/No_reflexes/"
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/Vesti_reflexes/"
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/COG_reflexes/"
    data_file_dic= "/home/suntao/workspace/experiment_data/"
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/DFFB_reflexes_V2/"
    #Experiment2_1(data_file_dic,start_point=460,end_point=1020, freq=60.0, experiment_classes=['-0.3'],trail_id=0)#1440-2160

    ''' This is for experiment 2_1 five trail for a inclination'''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/Vesti_reflexes_V2/"
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/DFFB_reflexes_V2/"
    #Experiment2_1_repeat(data_file_dic,start_point=180,end_point=2100,freq=60.0,experiment_classes=['-0.61'])

    ''' This is for experiment 2_2   '''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/Vesti_reflexes/"
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/COG_reflexes/"
    #slopeWalking_gamma_beta_pitch_displacement_statistic(data_file_dic,start_point=1000,end_point=2800,freq=60.0,experiment_classes=['0.0'])

    ''' This is comapre the successful counts of two reflexes  in Experiment two'''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/"
    #plot_runningSuccess_statistic(data_file_dic,start_point=1440,end_point=2160,freq=60,experiment_classes=['0.0'])

    ''' This is comapre the coordination of two reflexes  in Experiment two'''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/"
    #plot_coordination_statistic(data_file_dic,start_point=1440,end_point=2160,freq=60.0,experiment_classes=['0.0'])

    ''' This is comapre the stability of two reflexes  in Experiment two'''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/"
    #plot_stability_statistic(data_file_dic,start_point=1440,end_point=2160,freq=60,experiment_classes=['0.0'])

    ''' This is comapre the displacement of two reflexes  in Experiment two'''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment2/experiment2_2/"
    #plot_displacement_statistic(data_file_dic,start_point=1440,end_point=2160,freq=60,experiment_classes=['0.0'])
    
    '''************************************************************'''
    '''-------The following codes is for Experiment1_2-------------'''
    '''************************************************************'''
    data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/experiment1_2/"
    #plot_coordination_statistic(data_file_dic,start_point=960,end_point=1560,freq=60.0,experiment_classes=['0.0'])


    data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/experiment1_2/"
    #plot_stability_statistic(data_file_dic,start_point=960,end_point=1560,freq=60,experiment_classes=['0.0'])


    data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/experiment1_2/"
    #plot_displacement_statistic(data_file_dic,start_point=960,end_point=1560,freq=60,experiment_classes=['0.0'])


    data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment1/experiment1_2/Vesti_reflexes_V2/"
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #Experiment2_1(data_file_dic, start_point=960, end_point=1560, freq=60.0, experiment_classes = ['0.3'], trail_id=3)#1440-2160


    '''************************************************************'''
    '''-------The following codes is for Experiment 3-------------'''
    '''************************************************************'''
    ''' This is for experiment 3 '''

    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/experiment3/VersionV2/"
    #data_file_dic= "/home/suntao/workspace/experiment_data/"
    #Experiment3(data_file_dic, start_point=330, end_point=12460, freq=60.0, experiment_classes=['0.0']) #



    '''************************************************************'''
    '''-------The following codes is for Laikago-------------'''
    '''************************************************************'''
    ''' This is for Laikago '''
    #data_file_dic= "/media/suntao/DATA/Research/P4_workspace/Figures/experiment_data/Laikago/50Slope/"
    #Experiment_Laikago(data_file_dic, start_point=120, end_point=3000, freq=60.0, experiment_classes=['-0.2']) #
