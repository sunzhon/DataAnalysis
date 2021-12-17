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

# 使用Savitzky-Golay 滤波器后得到平滑图线
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
    start=500;end=800 # 选取一段做评估, 建立极限环的圆心和半径
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

def calculate_phase_diff_std(cpg_data,time,method_option=1):
    '''
    There are two methods

    M1: Calculation the standard derivation of the phase diffs

    M2: Calculation the distance of the phase diff state variable to the (3.14,3.14,0)
    '''
    if method_option==1:
        phi = calculate_phase_diff(cpg_data,time)
        filter_width=50# the roll_out width
        phi_stability=[]
        phi['phi_12']=savgol_filter(phi['phi_12'],91,2,mode='nearest')
        phi['phi_13']=savgol_filter(phi['phi_13'],91,2,mode='nearest')
        phi['phi_14']=savgol_filter(phi['phi_14'],91,2,mode='nearest')
        for idx in range(phi.shape[0]):
            if idx>=filter_width:
                temp= filter_width 
            else:
                temp=idx
                
            std_phi=np.std(phi.loc[idx-temp:idx]) # standard derivation of the phi
            phi_stability.append(sum(std_phi[1:])) # the sum of three phis, phi_12, phi_13, phi_14

        return np.array(phi_stability)

    if method_option==2:
        phi = calculate_phase_diff(cpg_data,time)
        #phi['phi_12']=savgol_filter(phi['phi_12'],91,2,mode='nearest')
        #phi['phi_13']=savgol_filter(phi['phi_13'],91,2,mode='nearest')
        #phi['phi_14']=savgol_filter(phi['phi_14'],91,2,mode='nearest')
        desired_point=np.array([3.14, 3.14, 0])
        distances=np.sqrt(np.sum((phi[['phi_12','phi_13','phi_14']]-desired_point)**2,axis=1))

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
            phi_stability=calculate_phase_diff_std(cpg_data[touch_moment_idx:,:],time[touch_moment_idx:],method_option=1) # 相位差的标准差
            phi_stability_threshold=0.7# empirically set 0.7
            for idx, value in enumerate(phi_stability): #serach the idx of the convergence moment/time
                if idx>=len(phi_stability)-1: # Not converge happen
                    #convergence_idx=len(phi_stability) # 仿真时间
                    convergence_idx=-1 #-1
                    print("CPG phase do not converge", convergence_idx)
                    break
                    # meet convergence condition, "max(phi_stability) >1.0 is to avoid the the CPG oscillatory disapper 
                if (value > phi_stability_threshold) and (phi_stability[idx+1] <= phi_stability_threshold) and (max(phi_stability) > 0.8):
                    convergence_idx=idx
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
            phi_distances=calculate_phase_diff_std(cpg_data[touch_moment_idx:,:],time[touch_moment_idx:],method_option=2) # 相位差的标准差, start from touch moment
            #phi_distances=calculate_phase_diff_std(cpg_data,time,method_option=2) # 相位差的标准差, start from start_time
            phi_distances_threshold=1.4# empirically set 1.4
            for idx, value in enumerate(phi_distances): #serach the idx of the convergence moment/time
                if idx>=len(phi_distances)-1: # Not converge happen
                    #convergence_idx=len(phi_distances)
                    convergence_idx=-1
                    print("CPG phase do not converge", convergence_idx)
                    break
                    # meet convergence condition, "max(phi_stability) >1.0 is to avoid the the CPG oscillatory disapper 
                if (value > phi_distances_threshold) and (phi_distances[idx+1]<=phi_distances_threshold):# the state variable converge to the desired fixed point (3.14, 3.14, 0)
                    convergence_idx=idx # start from touch moment
                    #convergence_idx=idx-touch_moment_idx # start from start_time
                    break
        else:#机器人没有放在地面
            convergenTime=0
            warnings.warn('The robot may be not dropped on the ground!')
    

        #print("touch_time: {}, convergence_time: {}".format(1.0*touch_moment_idx,1.0*convergence_idx))
        
        #NOTE: the touch_moment _idx is based on (counted from) the start_time that is specified at the main function.
        return touch_moment_idx, convergence_idx


def calculate_phase_convergence_time(time,grf_data, cpg_data,freq):
    '''
    Claculate phase convergnece time
    '''
    touch_idx,convergence_idx=calculate_touch_idx_phaseConvergence_idx(time,grf_data,cpg_data)
    return convergence_idx/freq





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
    Try to find a better metrix for descibel locomotion performance

    '''

    stability= 1.0/np.std(pose_data[:,0],axis=0) + 1.0/np.std(pose_data[:,1],axis=0) #+ 1.0/np.std(pose_data[:,2]) +1.0/np.std(pose_data[:,5])
    return stability




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


def calculate_stability(grf_data,pose_data):
    ''' simple ZMP amplitude changes '''
    return 1.0/(Average_COG_distribution(grf_data)-1.1)



def calculate_distance(pose_data):
    distance=0
    for step_index in range(pose_data.shape[0]-1):
        distance+=np.sqrt(pow(pose_data[step_index+1,3]-pose_data[step_index,3],2)+pow(pose_data[step_index+1,4]-pose_data[step_index,4],2))
    return distance


def calculate_joint_velocity(position_data, freq):
    velocity_data=(position_data[1:,:]-position_data[0:-1,:])*freq
    initial_velocity=[0,0,0, 0,0,0, 0,0,0, 0,0,0]
    velocity_data=np.vstack([initial_velocity,velocity_data])
    return velocity_data


def calculate_gait(data):
    ''' Calculating the gait information including touch states and duty factor'''
    # binary the GRF value 
    threshold = 0.1*max(data[:,0])
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
        for j in range(len(gait_info[i])): # each step, ignore the first stance or swing
            if (gait_info[i].__contains__(str(j)+'stance') and  gait_info[i].__contains__(str(j)+'swing')):
                beta_singleleg.append(gait_info[i][str(j)+"stance"]/(gait_info[i][str(j)+"stance"] + gait_info[i][str(j) + "swing"]))
            elif (gait_info[i].__contains__(str(j)+'stance') and  (not gait_info[i].__contains__(str(j)+'swing'))):
                beta_singleleg.append(1.0)
            elif ((not gait_info[i].__contains__(str(j)+'stance')) and  gait_info[i].__contains__(str(j)+'swing')):
                beta_singleleg.append(0.0)
        
        # if the step more than theree, then remove the max and min beta of each legs during the whole locomotion
        if(len(beta_singleleg)>3):
            beta_singleleg.remove(max(beta_singleleg))
        if(len(beta_singleleg)>3):
            beta_singleleg.remove(min(beta_singleleg))
        beta.append(beta_singleleg)

    return state, beta


def metrics_calculatiions(data_file_dic,start_time=5,end_time=30,freq=60.0,experiment_categories=['0.0'],control_methods=['apnc'],trial_ids=[0],**args):
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
    @param: args is unnamed parameters, it can be "investigation", it indicates which experiment in the paper, Experiment I, experiment II, ...
    @return: show and save a data figure.
    '''
    # 1) read data
    titles_files_categories=data_manager.load_data_log(data_file_dic)# a log file to save all data files and their categories and controlm methods
    #- variables to store robot executing information including movement, controller, and environments
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
    rosparameter={}
    metrics={}
    for category, files_name in titles_files_categories: #category is a files_name categorys
        if category in experiment_categories:
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
            rosparameter[category]=[]
            metrics[category]={}
            #control_method=files_name['titles'].iat[0]
            for control_method, file_name in files_name.groupby('titles'): #control methods
                if(control_method in control_methods): # which control methoid is to be display
                    print("The experiment category: ", category, "control method is: ", control_method)
                    metrics[category][control_method]=[]
                    for idx in file_name.index: # trials for display  in below
                        if idx in np.array(file_name.index)[trial_ids]:# which one is to load
                            folder_category= os.path.join(data_file_dic,file_name['data_files'][idx])
                            print(folder_category)
                            if('investigation' in args): # check  unnamed args
                                if args['investigation']=='update_frequency':
                                    freq=int(category) # the category is the frequency
                            cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = data_manager.read_data(freq,start_time,end_time,folder_category)

                            # 2)  data process
                            rosparameter[category].append(parameter_data)
                            cpg[category].append(cpg_data)
                            jmc[category].append(command_data)
                            pose[category].append(pose_data)
                            jmp[category].append(position_data)
                            velocity_data=calculate_joint_velocity(position_data,freq)
                            jmv[category].append(velocity_data)
                            jmf[category].append(current_data)
                            grf[category].append(grf_data)
                            gait_diagram_data_temp, beta_temp = calculate_gait(grf_data)
                            gait_diagram_data[category].append(gait_diagram_data_temp); beta[category].append(beta_temp)
                            noise[category].append(module_data)
                            temp_1=min([len(bb) for bb in beta_temp]) #minimum steps of all legs
                            beta_temp2=np.array([beta_temp[0][:temp_1],beta_temp[1][:temp_1],beta_temp[2][:temp_1],beta_temp[3][0:temp_1]]) # transfer to np array
                            #- metrics for evaluation the robot locomotion
                            if(beta_temp2.size==0):
                                metric_coordination=0
                            else:
                                metric_coordination=1.0/max(np.std(beta_temp2, axis=0))
                            metric_convergence_time= calculate_phase_convergence_time(time,grf_data,cpg_data,freq)
                            metric_stability= calculate_stability(pose_data,grf_data)
                            metric_balance= calculate_body_balance(pose_data)
                            metric_displacement= calculate_displacement(pose_data)
                            metric_distance= calculate_distance(pose_data)
                            metric_energy_cost= calculate_energy_cost(velocity_data,current_data,freq)
                            metric_COT= calculate_COT(velocity_data,current_data,freq,metric_displacement)
                            metrics[category][control_method].append({'coordination': metric_coordination, 'stability': metric_stability, 'balance': metric_balance, 'displacement': metric_displacement,'distance': metric_distance,'energy_cost':metric_energy_cost,'COT': metric_COT})

                            print("METRICS DISPLAY AS FOLLOW:")
                            print("Coordination:{:.2f} with shape: {}".format(metric_coordination, beta_temp2.shape))
                            print("Convergence time:{:.2f}".format(metric_convergence_time))
                            print("Stability:{:.2f}".format(metric_stability))
                            print("Balance:{:.2f}".format(metric_balance))
                            print("Displacemment:{:.2f}".format(metric_displacement)) #Displacement
                            print("Distance:{:.2f}".format(metric_distance)) #Distance 
                            print("Energy cost:{:.2f}".format(metric_energy_cost)) # enery cost
                            print("COT:{:.2f}".format(metric_COT))

    return metrics # dict


if __name__=="__main__":
    #print(globals())
    print(__file__)
    print(__package__)
