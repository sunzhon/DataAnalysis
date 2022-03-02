#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

# 使用Savitzky-Golay 滤波器后得到平滑图线 ,是基于局域多项式最小二乘法拟合的滤波方法
from scipy.signal import savgol_filter

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import data_manager


Mass=2.5 # Kg
Gravity=9.8 # On earth


'''
This module was developed to calculate robot locomotion metrics, which are common in evelation of 
robot locotion.

Author: suntao
Email: suntao.hn@gmail.com
Created date: 17-12-2021
'''



class neuralprocessing:
    '''
    Neral processing unit for filter GRFs signals
    '''
    w_i= 20.0
    w_r= 7.2
    bias= -6.0

    def __init__(self):
        self.input=0.0
        self.output=0.0
        self.output_old=0.0

    def step(self, input_signal):
        self.input = input_signal 
        self.output_old = self.output;
        activity = self.w_i*self.input+self.w_r*self.output_old+self.bias
        self.output = 1.0/(1.0+math.exp(-activity))
        return self.output

def NP(data):
    neural_process= neuralprocessing()

    filter_output=[]
    for idx in range(len(data)):
        filter_output.append(neural_process.step(data[idx]))

    return np.array(filter_output)

def test_neuralprocessing():
    '''
    It is a test for the class of the neuralprocessing  and its function implementation
    '''

    np = neuralprocessing()
    filtered=[]
    source=[]
    for idx in range(100):
        source.append(0.15 * math.sin(idx/20*3.14))
        filtered.append(np.step(source[-1]))

    
    filtered=NP(source)
    plt.plot(source,'g')
    plt.plot(filtered,'r')
    plt.show()


def calculate_energy_cost(U,I,Fre):
    '''
    U is also means joint toruqe
    I is also means joint velocity
    '''
    if ((type(U) is np.ndarray) and (type(I) is np.ndarray)):
        E=sum(sum(np.fabs(U)*np.fabs(I)*1/Fre))
    else:
        print("input data type is wrong, please use numpy array")
    return E

def calculate_COT(U,I,Fre,D):
    if(D>0.0): # the robot should walking a distance
        return calculate_energy_cost(U,I,Fre)/(Mass*Gravity*D)
    else:
        return 0.0;

def calculate_phase_diff(CPGs_output,time):
    '''
    Calculating phase difference bewteen  C12, C13, C14
    @param: CPGs_output, numpy darray n*8
    @return phi, DataFrame N*4, [time, phi_12, phi_13, phi_14]
    NOTE: This algorithm assume that the center of the orbit is in origin (0, 0)

    '''
    # Arrange CPG data
    phi=pd.DataFrame(columns=['time','phi_12','phi_13','phi_14'])
    C1=CPGs_output[:,0:2]
    C2=CPGs_output[:,2:4]
    C3=CPGs_output[:,4:6]
    C4=CPGs_output[:,6:8]
        
    # Checking wheather the center of the orbit in in orgin
    #start=500;end=800 # 选取一段做评估, 建立极限环的圆心和半径
    start=0;end=140 # 选取一段做评估, 建立极限环的圆心和半径
    C1_center=np.sum(C1[start:end,:], axis=0) # 轨迹圆心坐标
    C1_center_norm=np.linalg.norm(C1_center) #轨迹圆心到坐标原点的距离

    C2_center=np.sum(C2[start:end,:], axis=0) # 轨迹圆心坐标
    C2_center_norm=np.linalg.norm(C2_center) #轨迹圆心到坐标原点的距离

    C3_center=np.sum(C3[start:end,:], axis=0) # 轨迹圆心坐标
    C3_center_norm=np.linalg.norm(C3_center) #轨迹圆心到坐标原点的距离

    C4_center=np.sum(C4[start:end,:], axis=0) # 轨迹圆心坐标
    C4_center_norm=np.linalg.norm(C4_center) #轨迹圆心到坐标原点的距离

    threshold_dis=98 # CPG 极限环 圆心到坐标原点的距离的阈值
    if (C1_center_norm < threshold_dis) and (C2_center_norm < threshold_dis) and (C3_center_norm < threshold_dis) and (C4_center_norm < threshold_dis):
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
    else:
        warnings.warn('The CPG dynamic property changes, thus phi is set to 0')
        phi_12=np.zeros(len(C2))
        phi_13=np.zeros(len(C3))
        phi_14=np.zeros(len(C4))
        data=np.concatenate((time.reshape((-1,1)),phi_12.reshape((-1,1)),phi_13.reshape((-1,1)),phi_14.reshape((-1,1))),axis=1)
        data=pd.DataFrame(data,columns=['time','phi_12','phi_13','phi_14'])
        phi=pd.concat([phi,data])

    return phi

def calculate_phase_diff_std(phase_diff,method_option=1):
    '''
    There are two methods

    M1: Calculation the standard derivation of the phase diffs

    M2: Calculation the distance of the phase diff state variables to the desired state (3.14,3.14,0)
    '''
    if method_option==1:
        filter_width=50# the roll_out width
        phase_diff_std=[]
        phase_diff['phi_12']=savgol_filter(phase_diff['phi_12'],91,2,mode='nearest')
        phase_diff['phi_13']=savgol_filter(phase_diff['phi_13'],91,2,mode='nearest')
        phase_diff['phi_14']=savgol_filter(phase_diff['phi_14'],91,2,mode='nearest')
        for idx in range(phase_diff.shape[0]): # calculate the desired state (in a moving window)
            if idx>=filter_width:
                temp= filter_width 
            else:
                temp=idx
                
            std_phi=np.std(phase_diff.loc[idx-temp:idx]) # standard derivation of the phi in a window (filter_width)
            phase_diff_std.append(sum(std_phi[1:])) # the sum of three phis, phi_12, phi_13, phi_14

        return np.array(phase_diff_std)

    if method_option==2:
        desired_point=np.array([3.14, 3.14, 0])
        distances=np.sqrt(np.sum((phase_diff[['phi_12','phi_13','phi_14']]-desired_point)**2,axis=1))

        return savgol_filter(distances,91,2,mode="nearest")


def calculate_touch_idx_phaseConvergence_idx(time,grf_data,cpg_data,method_option=2):
    '''
    There are two methods, the first one is based on phi standard deviation, the second is based on distance between PHI and (3.14, 3.14, 0)
    Claculate phase convergnece idx and touch idx
    '''
    if method_option==1: # using the deviation of the phi sum 
        grf_stance = grf_data > Mass*Gravity/5.0# GRF is bigger than 7, we see the robot starts to interact weith the ground
        grf_stance_index=np.where(grf_stance.sum(axis=1)>=2)# Find the robot drop on ground moment if has two feet on ground at least
        if(grf_stance_index[0].size!=0):#机器人落地了
            touch_moment_idx= grf_stance_index[0][0]# 落地时刻
            phase_diff=calculate_phase_diff(cpg_data,time)
            phi_stability=calculate_phase_diff_std(phase_diff,method_option=1) # 相位差的标准差
            phi_stability_threshold=0.7# empirically set 0.7
            for idx, value in enumerate(phi_stability): #serach the idx of the convergence moment/time
                if idx>=len(phi_stability)-1: # Not converge happen
                    #convergence_idx=len(phi_stability) # 仿真时间
                    convergence_idx=-1 #-1
                    print("CPG phase do not converge", convergence_idx)
                    break
                    # meet convergence condition, "max(phi_stability) >1.0 is to avoid the the CPG oscillatory disapper 
                if (value > phi_stability_threshold) and (phi_stability[idx+1] <= phi_stability_threshold) and (max(phi_stability) > 0.8):
                    convergence_idx=idx-touch_moment_idx # start from the touch moment
                    break
        else:#机器人没有放在地面
            convergenTime=0
            warnings.warn('The robot may be not dropped on the ground!')

        return touch_moment_idx, convergence_idx
    if method_option==2:# using the distance in 3D phase space
        grf_stance = grf_data > Mass*Gravity/5.0# GRF is bigger than 7, we see the robot starts to interact weith the ground
        grf_stance_index=np.where(grf_stance.sum(axis=1)>=2)# Find the robot drop on ground moment if has two feet on ground at least
        if(grf_stance_index[0].size!=0):#机器人落地了 robot on the ground
            touch_moment_idx= grf_stance_index[0][0]# 落地时刻 the touch momnet $n_0$
            phase_diff=calculate_phase_diff(cpg_data,time)
            phi_distances=calculate_phase_diff_std(phase_diff,method_option=2) # distance to the desired state (pi, pi, 0), start from touch moment
            phi_distances_threshold=2.8# empirically set 1.4
            for idx, value in enumerate(phi_distances): #serach the idx of the convergence moment/time
                if idx>=len(phi_distances)-1: # Not converge happen during walking period
                    #convergence_idx=len(phi_distances)
                    convergence_idx=-1
                    warnings.warn("CPG phase do not converge, the convergence index is {}".format(convergence_idx))
                    break
                    # meet convergence condition, "max(phi_stability) >1.0 is to avoid the the CPG oscillatory disapper 
                if (value > phi_distances_threshold) and (phi_distances[idx+1]<=phi_distances_threshold):# the state variable converge to the desired fixed point (3.14, 3.14, 0)
                    convergence_idx=idx-touch_moment_idx # start from touch moment
                    break
        else:#机器人没有放在地面
            convergenTime=0
            warnings.warn('The robot may be not dropped on the ground!')
    

        #print("touch_time: {}, convergence_time: {}".format(1.0*touch_moment_idx,1.0*convergence_idx))
        
        #NOTE: the touch_moment _idx is based on (counted from) the start_time that is specified at the main function.
        return touch_moment_idx, convergence_idx


def calculate_phase_convergence_time(time,grf_data, cpg_data,freq,method_option=2):
    '''
    Claculate phase convergnece time
    '''
    touch_idx,convergence_idx=calculate_touch_idx_phaseConvergence_idx(time,grf_data,cpg_data,method_option=method_option)
    return convergence_idx/freq

def calculate_motion_coordination(duty_factor):
    '''
    Coordination metric represents the derivation of the four legs' duty factors

    @params: duty_factor (stance ratio) (M*leg_num) numpy , M steps of four legs.
    
    @return: the reciprocal of the max standard derivation of the duty factor
    '''
    #- metrics for evaluation the robot locomotion
    if(duty_factor.size==0):
        coordination = 0
    else:
        coordination = 1.0/max(np.std(duty_factor, axis=0))


    return coordination



def calculate_phase_diff_stability(grf_data,cpg_data,time):
    touch_idx, convergence_idx, phi_std =touch_convergence_moment_identification(grf_data,cpg_data,time)
    formed_phase_std=np.mean(phi_std[convergence_idx:]) # consider the phase diff standard derivation after the self-organzied locomotion formed or interlimb formed
    phase_stability=1.0/formed_phase_std 
    return phase_stability

def calculate_displacement(pose_data):
    '''
    Displacemnt on level ground
    '''
    d=np.sqrt(pow(pose_data[-1,3]-pose_data[0,3],2)+pow(pose_data[-1,4]-pose_data[0,4],2)) #Displacement
    return d

def calculate_body_balance(pose_data):
    '''
    Try to find a better metrix for describle locomotion performance, balance

    @params: pose_data is N*M, M is the dimention of the robot trunk orientation, including: roll, pitch, yaw, x,y,z, vx, vy,vz

    '''
    
    balance = 1.0/np.std(pose_data[:,0],axis=0) + 1.0/np.std(pose_data[:,1],axis=0) #+ 1.0/np.std(pose_data[:,2]) +1.0/np.std(pose_data[:,5])
    return balance




def COG_distribution(grf_data):
    '''
    Calculate the Center of gravity of the robot locomotion
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
        sum_f=sum_f+0.00001 # in case the Divisor is zero
        gamma=np.true_divide(sum_f[:,0],sum_f[:,1])
    # set the gamma to zero when robot is fixed in the air
    temp=pd.DataFrame(np.column_stack([sum_f,gamma]),columns=['front','hind','gamma'])
    temp.loc[temp['hind']<Mass*Gravity/5.0,'gamma']=0.0
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


def calculate_ZMP_stability(grf_data,pose_data):
    ''' simple ZMP amplitude changes '''
    return 1.0/(Average_COG_distribution(grf_data)-1.1)


def calculate_body_stability(grf_data,pose_data):
    ''' body movement along with lateral direction '''
    return pose_data[0];


def calculate_distance(pose_data):
    '''
    The distance from the original point to the end point

    '''
    distance=0
    for step_index in range(pose_data.shape[0]-1):
        distance+=np.sqrt(pow(pose_data[step_index+1,3]-pose_data[step_index,3],2)+pow(pose_data[step_index+1,4]-pose_data[step_index,4],2))
    return distance


def calculate_joint_velocity(position_data, freq):
    velocity_data=(position_data[1:,:]-position_data[0:-1,:])*freq
    initial_velocity=[0,0,0, 0,0,0, 0,0,0, 0,0,0]
    velocity_data=np.vstack([initial_velocity,velocity_data])
    return velocity_data


def calculate_gait(data,stance_threshold_coefficient=0.1):
    ''' 
    Calculating the gait information including touch gaitphases and duty factor
    
    @params: data is N*4, 4 legs's GRF

    @return: gaitphase (N*leg_num) numpy array indicates the stance phase with 1 or swing phase with 0
             stanceratio (M*leg_num) numpy array indicates duty factors, where M indicates the step number 
    
    '''
    # binary the GRF value 
    threshold = stance_threshold_coefficient*max(data[:,0])
    gaitphase=np.zeros(data.shape,int)
    for i in range(data.shape[1]): #legs
        for j in range(data.shape[0]):# step count
            if data[j,i] < threshold:
                gaitphase[j,i] = 0
            else:
                gaitphase[j,i]=1
    
    # get gait info, count the touch and lift number steps
    gait_info=[]
    stanceratio=[]
    gaitphase=np.vstack([gaitphase,abs(gaitphase[-1,:]-1)])# repeat the last step count gaitphase, since the for use an additional row [j+1, ]
    for i in range(gaitphase.shape[1]): #each leg
        count_stance=0;
        count_swing=0;
        number_stance=0
        number_swing=0
        duty_info = {}
                
        for j in range(gaitphase.shape[0]-1): #every count
            if gaitphase[j,i] ==1:# stance 
                count_stance+=1
            else: # swing
                count_swing+=1

            if (gaitphase[j,i]==0) and (gaitphase[j+1,i]==1): # come into swing phase
                duty_info[str(number_swing)+ "swing"]= count_swing # swing phase number
                count_swing=0 # swing number of a swing phase
                number_swing+=1
            if (gaitphase[j,i]==1) and (gaitphase[j+1,i]==0): # come into stance phase
                duty_info[str(number_stance) + "stance"]= count_stance
                count_stance=0
                number_stance+=1
            
        gait_info.append(duty_info)
    # calculate the duty factors of all legs
    for i in range(len(gait_info)): # each leg
        stanceratio_singleleg=[]
        for j in range(len(gait_info[i])): # each step, ignore the first stance or swing
            if (gait_info[i].__contains__(str(j)+'stance') and gait_info[i].__contains__(str(j)+'swing')):
                stanceratio_singleleg.append(gait_info[i][str(j)+"stance"]/(gait_info[i][str(j)+"stance"] + gait_info[i][str(j) + "swing"]))
            elif (gait_info[i].__contains__(str(j)+'stance') and  (not gait_info[i].__contains__(str(j)+'swing'))): 
                stanceratio_singleleg.append(1.0)
            elif ((not gait_info[i].__contains__(str(j)+'stance')) and  gait_info[i].__contains__(str(j)+'swing')):
                stanceratio_singleleg.append(0.0)
        
        # if the step more than theree, then remove the max and min stanceratio of each legs during the whole locomotion
        if(len(stanceratio_singleleg)>3):
            stanceratio_singleleg.remove(max(stanceratio_singleleg))
        if(len(stanceratio_singleleg)>3):
            stanceratio_singleleg.remove(min(stanceratio_singleleg))
        stanceratio.append(stanceratio_singleleg)
    


    # transfer 2D list into 2D numpy array
    min_step_num=min([len(bb) for bb in stanceratio]) #minimum steps of all legs
    stanceratio=np.array([stanceratio[0][:min_step_num],stanceratio[1][:min_step_num],stanceratio[2][:min_step_num],stanceratio[3][0:min_step_num]]) # transfer four legs' duty factors
    stanceratio=np.array(stanceratio).T # stride number * leg number


    return gaitphase, stanceratio # gaitphase indicates the swing of stance, stanceratio indicates the duty factors


def metrics_calculatiions(data_file_dic,start_time=5,end_time=30,freq=60.0,experiment_categories=['0.0'],control_methods=['apnc'],trial_ids=[0],**kwargs):
    ''' 
    @Description:
    This is for calculate all metrics for many trials under different experiment category (experiment variables, such as robot situations, different environments) and control_methods (different controller)

    This function was developed to calcualte some metrics for evaluating robot locomotion performances. The enrolled metrics have:
    1) robot walking dispacement, 2) robot walking distance, 3) robot walking balance, 4) leg movemente coordination, 
    5) phase convergence time from initial status to stable status, 6) duty factors of four legs, 
    6) robot energy cost, 7) robot COT

    @param: data_file_dic, the folder of the data files, this path includes a log file which list all folders of the experiment data for display
    @param: start_time, the start point (time) of all the data, units: seconds
    @param: end_time, the end point (time) of all the data
    @param: freq, the sample frequency of the data 
    @param: experiment_categories, the conditions/cases/experiment_categories of the experimental data
    @param: control_methods which indicate the control method used in the experiment
    @param: trial_ids, it indicates which experiments (trials) among a inclination/situation/case experiments 
    @param: kwargs is unnamed parameters, it can be "investigation", it indicates which experiment in the paper, Experiment I, experiment II, ...
    @return: show and save a data figure.
    '''
    # 1) read data
    titles_files_categories=data_manager.load_data_log(data_file_dic)# a log file to save all data files and their categories and controlm methods
    #- variables to store robot executing information including movement, controller, and environments
    gamma={}
    duty_factor={}
    gait_diagram_data={}
    pitch={}
    pose={}
    jmc={}
    jmp={}
    jmv={}
    jmf={}
    grf={}
    cpg={}
    phi={}# cpgs phase difference
    modules={}
    rosparameter={}
    metrics={}
    experiment_data={}
    for category, category_group in titles_files_categories: #category is name of catogory_group
        if category in experiment_categories:
            gamma[category]=[]  #category_group is the table of the category
            gait_diagram_data[category]=[]
            duty_factor[category]=[]
            pose[category]=[]
            jmc[category]=[]
            jmp[category]=[]
            jmv[category]=[]
            jmf[category]=[]
            grf[category]=[]
            cpg[category]=[]
            phi[category]=[]
            modules[category]=[]
            rosparameter[category]=[]
            metrics[category]={}
            experiment_data[category]={}
            for control_method, category_control_group in category_group.groupby('titles'): # titles means control methods
                if(control_method in control_methods): # which control methoid is to be display
                    metrics[category][control_method]=[]
                    experiment_data[category][control_method]=[]
                    for idx,trial_folder_name in enumerate(category_control_group['data_files']): # trials for display in below
                        try:
                            if (idx in trial_ids):# which one is to load
                                if('trial_folder_names' in kwargs): # check  unnamed kwargs
                                    if(trial_folder_name not in kwargs['trial_folder_names']):# if specified trial_folder_name in kwargs, then use the trial_folder_name
                                        continue
                                print("The experiment category is: ", category, ", control method is: ", control_method, ", trial folder name is:", trial_folder_name)
                                folder_category= os.path.join(data_file_dic,trial_folder_name)
                                print(folder_category)
                                if('investigation' in kwargs): # check  unnamed kwargs
                                    if kwargs['investigation']=='update_frequency':
                                        freq=int(category) # the category is the frequency
                                cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = data_manager.load_a_trial_data(freq,start_time,end_time,folder_category)

                                # 2)  data process
                                rosparameter[category].append(parameter_data)
                                cpg[category].append(cpg_data)
                                jmc[category].append(command_data)
                                pose[category].append(pose_data)
                                jmp[category].append(position_data)
                                velocity_data = calculate_joint_velocity(position_data,freq)
                                jmv[category].append(velocity_data)
                                jmf[category].append(current_data)
                                grf[category].append(grf_data)
                                modules[category].append(module_data)
                                #----Metrics
                                metric_phase_diff=calculate_phase_diff(cpg_data,time)
                                metric_phase_diff_std=calculate_phase_diff_std(metric_phase_diff,2)
                                gait_diagram_data_temp, duty_factor_data = calculate_gait(grf_data)
                                gait_diagram_data[category].append(gait_diagram_data_temp); 
                                duty_factor[category].append(duty_factor_data)
                                metric_coordination = calculate_motion_coordination(duty_factor_data)
                                metric_convergence_time = calculate_phase_convergence_time(time,grf_data,cpg_data,freq)
                                metric_stability = calculate_ZMP_stability(pose_data,grf_data)
                                metric_balance = calculate_body_balance(pose_data)
                                metric_displacement = calculate_displacement(pose_data)
                                metric_distance = calculate_distance(pose_data)
                                metric_energy_cost = calculate_energy_cost(velocity_data,current_data,freq)
                                metric_COT = calculate_COT(velocity_data,current_data,freq,metric_displacement)
                                metrics[category][control_method].append({'phase_diff':metric_phase_diff, 'phase_diff_std':metric_phase_diff_std, 'coordination': metric_coordination, 'phase_convergence_time':metric_convergence_time,'stability': metric_stability, 'balance': metric_balance, 'displacement': metric_displacement, 'distance': metric_distance, 'energy_cost':metric_energy_cost, 'COT': metric_COT,'trial_folder_name':trial_folder_name})
                                experiment_data[category][control_method].append({'cpg': cpg_data, 'grf': grf_data, 'jmp': position_data, 'pose': pose_data, 'jmf':current_data,'jmc': command_data,'jmv':velocity_data,'time':time,'gait_diagram_data':gait_diagram_data_temp,'rosparameter':parameter_data,'modules':module_data})

                                print("METRICS DISPLAY AS FOLLOW:")
                                print("Coordination:{:.2f}".format(metric_coordination))
                                print("Convergence time:{:.2f}".format(metric_convergence_time))
                                print("Stability:{:.2f}".format(metric_stability))
                                print("Balance:{:.2f}".format(metric_balance))
                                print("Displacemment:{:.2f}".format(metric_displacement)) #Displacement
                                print("Distance:{:.2f}".format(metric_distance)) #Distance 
                                print("Energy cost:{:.2f}".format(metric_energy_cost)) # enery cost
                                print("COT:{:.2f}".format(metric_COT))

                        except IndexError:
                            print("category 类别数目没有trial_ids 列出的多, 请添加trials")

    return experiment_data, metrics # dict






if __name__=="__main__":
    #print(globals())
    print(__file__)
    print(__package__)
