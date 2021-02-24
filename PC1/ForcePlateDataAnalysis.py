import sqlite3
from sqlite3 import Error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb 
from cycler import cycler
import matplotlib as mpl
from matplotlib import gridspec
import os
import gnureadline
import termcolor

#Load data of robot

def load_data_log(data_file_dic):
    '''
    Load data log that stores data file names,
    Group data by experiment_categories/categories and output the categories (experiemnt classes)

    '''
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    #1.1) load file list 
    data_file_log = data_file_dic +"ExperimentDataLog.csv"
    data_files = pd.read_csv(data_file_log, sep='\t',header=None, names=['titles', 'data_files','categories'], skip_blank_lines=True,dtype=str)

    data_files_categories=data_files.groupby('categories')
    keys = data_files_categories.groups.keys()
    categories=[]
    for ll in keys:
        if is_number(ll):
            categories.append(ll)
    
    temp_dic={}
    for idx, value in enumerate(categories):
        temp_dic[str(float(categories[idx]))]=value

    temp_dic_keys =[str(ll) for ll in sorted([ float(ll) for ll in temp_dic.keys()])]

    for idx,value in enumerate(temp_dic_keys):
        categories[idx]=temp_dic[value]

    print(categories)
    return data_files_categories

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
    columnsName_parameters=['CPGtype','CPGMi','CPGPGain', 'CPGPThreshold', 'PCPGBeta', \
                            'RF_PSN','RF_VRN_Hip','RF_VRN_Knee','RF_MN1','RF_MN2','RF_MN3',\
                            'RH_PSN','RH_VRN_Hip','RH_VRN_Knee','RH_MN1','RH_MN2','RH_MN3',\
                            'LF_PSN','LF_VRN_Hip','LF_VRN_Knee','LF_MN1','LF_MN2','LF_MN3',\
                            'LH_PSN','LH_VRN_Hip','LH_VRN_Knee','LH_MN1','LH_MN2','LH_MN3'
                           ]
    columnsName_commands=['c1','c2','c3','c4','c5','c6', 'c7','c8','c9','c10','c11','c12']


    columnsName_joints = columnsName_jointPositions + columnsName_jointVelocities + columnsName_jointCurrents + columnsName_jointVoltages + columnsName_POSEs + columnsName_GRFs
    
    #CPG
    cpg_data=loadData(fileName_CPGs,columnsName_CPGs,folder_name)    
    cpg_data=cpg_data.values

    #commands
    command_data=loadData(fileName_commands,columnsName_commands,folder_name)    
    command_data=command_data.values

    #ANC stability value
    module_data=loadData(fileName_modules,columnsName_modules,folder_name)    
    module_data=module_data.values

    #parameter
    parameter_data=loadData(fileName_parameters,columnsName_parameters,folder_name)    
    parameter_data=parameter_data.values

    #joint sensory data
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



def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
 
    return conn

def select_all_data(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM DATA")
 
    rows = cur.fetchall()
    return rows
    #for row in rows:
    #    print(row)
 
 
def select_data_by_priority(conn, step):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM DATA WHERE ID=?", (step,))
 
    rows = cur.fetchall()
    return rows
    
    #for row in rows:
    #    print(row)
def read_force_data(filename,freq=200):
    #freq=200# Hz sample

    conn=create_connection(filename)
    with conn:
        datas=select_all_data(conn)
        pd_datas=pd.DataFrame(datas)
        
    Fx,Fy,Fz=[],[],[]
    for idx in range(3,82,4):
        Fx.append(pd_datas.iloc[:,idx])
        Fy.append(pd_datas.iloc[:,idx+1])
        Fz.append(-1.0*pd_datas.iloc[:,idx+2])
    '''
    Sum for all Fx, Fy, and Fz
    Fx_sum=Fx[0];Fy_sum=Fy[0];Fz_sum=Fz[0]
    for idx in range(1,len(Fx)):
        Fx_sum+=Fx[idx]
        Fy_sum+=Fy[idx]
        Fz_sum+=Fz[idx]
    '''
    sample_time=np.linspace(0,len(Fx[0]),len(Fx[0]))/freq
    return [sample_time,Fx,Fy,Fz]

def plot_current(trial_id=1,visual_start_t=0,visual_end_t=20):
    start_point=0;end_point=10000;freq=60;
    robot_data=["1120111121","1120111459","1120111653","1120111759","1120111849","1120111943","1120112028","1120112122","1120112208","1120112346","1120112450","1120112548","1120112620","1120112700","1120112741"
               ,"1120125332","1120125422","1120125457","1120125539","1120125626","1120125701","1120125744","1120125821","1120125854"]

    folder_category=r"/media/suntao/DATA/Research/PC1_workspace/Experimental_data/Robot_data/"+robot_data[trial_id]
    print("robot_data_folder",folder_category)
    cpg_data, command_data, module_data, parameter_data, grf_data, pose_data, position_data, velocity_data, current_data,voltage_data, time = read_data(freq,start_point,end_point,folder_category)

    # Plot current of joints
    figsize=(16,8)
    fig2=plt.figure(figsize=figsize)
    plot_num=4
    current_start_time=int(visual_start_t*freq);
    if(visual_end_t*freq>len(current_data)):
        current_end_time=len(current_data)-1;
    else:
        current_end_time=int(visual_end_t*freq)

    temp=abs(current_data[current_start_time:current_end_time,0])+abs(current_data[current_start_time:current_end_time,3])+abs(current_data[current_start_time:current_end_time,6])+abs(current_data[current_start_time:current_end_time,9])
    temp=150*(temp>40)
    start_board=min(np.where(temp==150)[0])+current_start_time-1
    end_board=max(np.where(temp==150)[0])+current_start_time-1
    for plot_idx in range(plot_num):
        plt.subplot(plot_num,1,plot_idx+1)
        plt.plot(time[current_start_time:current_end_time], current_data[current_start_time:current_end_time,3*plot_idx])
        #plt.plot(time[current_start_time:current_end_time], current_data[current_start_time:current_end_time,3*plot_idx+1])
        #plt.plot(time[current_start_time:current_end_time], current_data[current_start_time:current_end_time,3*plot_idx+2])
        plt.plot(time[current_start_time:current_end_time], temp)
        plt.vlines(start_board/freq, 0, 200, colors = "k", linestyles = "dashed")
        plt.vlines(end_board/freq, 0, 200, colors = "k", linestyles = "dashed")
        plt.grid()
        plt.ylabel("Current [mA]")
        plt.legend(["Joint ID: "+str(3*plot_idx),"Joint ID: "+str(3*plot_idx+1),"Joint ID: "+str(3*plot_idx+2)])
    plt.xlabel("Time [s]")
    plt.show()

    return [current_data,start_board,end_board]


def process_force_data(trial_id=0,visual_start_t=58.5,visual_end_t=66.5):

    # 1) read data
    force_plate_data=["1th_11_09","2nd_11_11","3rd_11_12","4th_11_13","5th_11_14","6th_11_15","7th_11_16","8th_11_17","9th_11_17","10th_11_18","11th_11_19","12th_11_20","13th_11_21","14th_11_21",
                      "15th_11_22","16th_11_23","17th_12_48","18th_12_49","19th_12_49","20th_12_50","21th_12_51","22th_12_51","23th_12_52","24th_12_53","25th_12_53"]

    filename="./"+force_plate_data[trial_id]+r".db"
    print("force plate data", filename)
    [Time,Fx,Fy,Fz]=read_force_data(filename)
    freq_plate=200;# force palet system sample frequency
    start_time=int(visual_start_t*freq_plate);end_time=int(visual_end_t*freq_plate)
    maximal_board_list=[]

    #plot force plate
    figsize=(16,8)
    fig=plt.figure(figsize=figsize)
    plt_num=5
    for plot_idx in range(plt_num):
        plate_unit=2*plot_idx
        plt.subplot(plt_num,1,plot_idx+1)
        if(plate_unit==6):
            Fz[6][start_time:end_time]=-1.0*Fz[6][start_time:end_time]# This plate unit has opposite Z direction with other units
        temp=Fz[plate_unit][start_time:end_time]+Fz[plate_unit+1][start_time:end_time]       
        temp=np.array(temp)>1
        #pdb.set_trace()
        index_board=np.where(temp==1)
        if index_board==None:
            print("The sum of J1 currents are all less than 1 mA, Try to adjust the visual time range")
        minimal_board=min(index_board[0])+start_time
        maximal_board=max(index_board[0])+1+start_time
        if plate_unit==0: # start_board is the minimal board of the first two plates
            maximal_board_list.append(minimal_board)
        maximal_board_list.append(maximal_board)
        plt.plot(Time[start_time:end_time], Fz[plate_unit][start_time:end_time])
        plt.plot(Time[start_time:end_time], Fz[plate_unit+1][start_time:end_time])
        plt.vlines(minimal_board/freq_plate, 0, 20, colors = "k", linestyles = "dashed")
        plt.vlines(maximal_board/freq_plate, 0, 20, colors = "k", linestyles = "dashed")
        plt.grid()
        plt.ylabel("Force [N]")
        plt.legend([str(plate_unit),str(plate_unit+1)])
        plt.xlabel("Time [s]")
        #plt.show()

    #Arrange the force plate data to four legs
    leg_num=4
    GRFs=np.zeros((len(Fz[0]),leg_num))
    for board_idx in range(len(maximal_board_list)-2):#The first and last maxinaml board is the start and end board, here iterate the ranges, and just calculate the repeated time range data 
        plate_unit=2*board_idx; leg_idx=1
        GRFs[maximal_board_list[board_idx]:maximal_board_list[board_idx+1],leg_idx]=Fz[plate_unit][maximal_board_list[board_idx]:maximal_board_list[board_idx+1]]
        plate_unit=2*board_idx+1;leg_idx=3
        GRFs[maximal_board_list[board_idx]:maximal_board_list[board_idx+1],leg_idx]=Fz[plate_unit][maximal_board_list[board_idx]:maximal_board_list[board_idx+1]]
        plate_unit=2*board_idx+2;leg_idx=0
        GRFs[maximal_board_list[board_idx]:maximal_board_list[board_idx+1],leg_idx]=Fz[plate_unit][maximal_board_list[board_idx]:maximal_board_list[board_idx+1]]
        plate_unit=2*board_idx+3;leg_idx=2
        GRFs[maximal_board_list[board_idx]:maximal_board_list[board_idx+1],leg_idx]=Fz[plate_unit][maximal_board_list[board_idx]:maximal_board_list[board_idx+1]]
    if True:
        #Plot force data of the four legs
        figsize=(14,6)
        fig2=plt.figure(figsize=figsize)
        plot_num=4
        for plot_idx in range(plot_num):
            plt.subplot(plot_num,1,plot_idx+1)
            plt.plot(Time[start_time:end_time], GRFs[start_time:end_time,plot_idx])
            plt.vlines(maximal_board_list[0]/freq_plate, 0, 20, colors = "k", linestyles = "dashed")
            plt.vlines(maximal_board_list[-2]/freq_plate, 0, 20, colors = "k", linestyles = "dashed")
            plt.grid()
            plt.ylabel("Force [N]")
            plt.legend(["Leg ID: "+str(plot_idx)])
        plt.xlabel("Time [s]")
    
    
    plt.show()

    return [GRFs, maximal_board_list[0], maximal_board_list[-2]]



if __name__=='__main__':


    trial_id=3
    visual_start_t_current=5;visual_end_t_current=20;
    visual_start_t_grf=22;visual_end_t_grf=31;

    trial_id=6
    visual_start_t_current=5;visual_end_t_current=30;
    visual_start_t_grf=12;visual_end_t_grf=20;

    trial_id=9
    visual_start_t_current=8.8;visual_end_t_current=30;
    visual_start_t_grf=19;visual_end_t_grf=25;



    trial_id=10
    visual_start_t_current=7.5;visual_end_t_current=18.5;
    visual_start_t_grf=17.5;visual_end_t_grf=23.5;

    trial_id=11
    visual_start_t_current=7.1;visual_end_t_current=28;
    visual_start_t_grf=13;visual_end_t_grf=22;


    trial_id=13
    visual_start_t_current=7.4;visual_end_t_current=45;
    visual_start_t_grf=10;visual_end_t_grf=19;


    trial_id=14
    visual_start_t_current=6.5;visual_end_t_current=45;
    visual_start_t_grf=9;visual_end_t_grf=15.4;
    

    trial_id=15
    visual_start_t_current=8.5;visual_end_t_current=45;
    visual_start_t_grf=10;visual_end_t_grf=20;


    trial_id=17
    visual_start_t_current=6;visual_end_t_current=15;
    visual_start_t_grf=40;visual_end_t_grf=48;


    trial_id=19
    visual_start_t_current=6;visual_end_t_current=15;
    visual_start_t_grf=0;visual_end_t_grf=48;

    [current_data,c_start,c_end]=plot_current(trial_id=trial_id-1,visual_start_t=visual_start_t_current,visual_end_t=visual_end_t_current)
    [grf_data,g_start,g_end]=process_force_data(trial_id=trial_id-1,visual_start_t=visual_start_t_grf,visual_end_t=visual_end_t_grf)
    
    
    #align the time of current and grf
    c_end=c_start+int((g_end-g_start)/200*60)
    new_current_data=current_data[c_start:c_end,:]
    new_grf_data=grf_data[g_start:g_end,:]
    num_sample=c_end-c_start;
    new_current_time=np.linspace(0,num_sample/60,num_sample)
    new_grf_time=np.linspace(0,num_sample/60,g_end-g_start)
    new_new_grf_data=np.zeros((len(new_current_data),4))
    new_new_grf_data[:,0]=np.interp(new_current_time,new_grf_time,new_grf_data[:,0])
    new_new_grf_data[:,1]=np.interp(new_current_time,new_grf_time,new_grf_data[:,1])
    new_new_grf_data[:,2]=np.interp(new_current_time,new_grf_time,new_grf_data[:,2])
    new_new_grf_data[:,3]=np.interp(new_current_time,new_grf_time,new_grf_data[:,3])

    new_data= np.column_stack((new_current_data,new_new_grf_data))
    columns_name=['c1','c2','c3','c4','c5','c6', 'c7','c8','c9','c10','c11','c12','RF','RH','LF','LH']
    fine_data=pd.DataFrame(data=new_data,columns=columns_name)
    
    plt.plot(np.linspace(0,len(fine_data['c1'])/60,len(fine_data['c1'])),fine_data['c1'])
    plt.plot(np.linspace(0,len(fine_data['c1'])/60,len(fine_data['c1'])),fine_data['RF'])
    plt.show()

    fine_data.to_csv("Trial_"+str(trial_id)+".csv")
