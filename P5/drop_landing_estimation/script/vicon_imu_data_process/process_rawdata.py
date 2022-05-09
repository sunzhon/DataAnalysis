#Python

"""
Description:
    This is an module to process data , it is a base libe to implement ann to predict knee joint values in drop landing experiments

Author: Sun Tao
Email: suntao.hn@gmail.com
Date: 2021-07-01

"""
import pandas as pd
import numpy as np
import os
import h5py
import re


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from matplotlib import gridspec

import copy
import matplotlib.pyplot as plt
import time as localtimepkg
import seaborn as sns
import math
import inspect
import yaml
import pdb
import warnings
import termcolor
import matplotlib._color_data as mcd

import datetime


from scipy.stats import normaltest 
from scipy.stats import kstest
from scipy.stats import shapiro
from scipy.stats import levene
from scipy import stats

from statannotations.Annotator import Annotator

if __name__=='__main__':
    from const import FEATURES_FIELDS, LABELS_FIELDS, V3D_LABELS_FIELDS, DATA_PATH, TRAIN_USED_TRIALS, TRAIN_USED_TRIALS_SINGLE_LEG, TRIALS, DATA_VISULIZATION_PATH, DROPLANDING_PERIOD, RESULTS_PATH,IMU_SENSOR_LIST, IMU_RAW_FIELDS, ACC_GYRO_FIELDS, SYN_DROPLANDING_PERIOD
else:
    from vicon_imu_data_process.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH, TRAIN_USED_TRIALS, TRAIN_USED_TRIALS_SINGLE_LEG, TRIALS, DATA_VISULIZATION_PATH, DROPLANDING_PERIOD,V3D_LABELS_FIELDS,RESULTS_PATH, IMU_SENSOR_LIST,IMU_RAW_FIELDS, ACC_GYRO_FIELDS,SYN_DROPLANDING_PERIOD


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


def read_subject_trials(subject_id_name: str, trials: int, data_fields: list, raw_datasets_path=None, **kwargs):
    """
    @Description:
    To read raw data of a subject from a h5 file
    @Parameters:
    trials: sepcify the data of trial 
    data_fields: the names of columns. data type is string, the sequence of the field in data_fields determines the value sequences
    kwargs['assign'], which  concate the data in three dimensions, the first dimension is trial numbers
    
    """
    assert(isinstance(data_fields,list))
    assert(isinstance(trials,list))

    #1) read h5 data file
    with h5py.File(raw_datasets_path, 'r') as fd:
        #i) get all subject names, e.g., [P_01_suntao, ..., P_09_libang, P_24_XX]
        subject_ids_names = list(fd.keys())

        #ii) get all data feilds: the features and labels
        all_data_fields = fd[subject_ids_names[0]].attrs.get('columns')

        trials_data={}

        #iii) check the specified subject name
        if(subject_id_name not in subject_ids_names):
            print("This subject:{} is not in datasets, see line 91 in process_rawdata.py".format(subject_name))
            exit()

        #iv) get each trial data with specified columns (data fields) of a subject (subject_id_name)
        for trial in trials: 
            try:
                #a) get all column data of a trial of a subject into a dataframe
                data = fd[subject_id_name][trial]
                columns = fd[subject_id_name].attrs['columns']
                temp_pd_data = pd.DataFrame(data=np.array(data), columns=columns)
                #b) retrive specified columns by parameter: data_fields
                trials_data[trial] = temp_pd_data[data_fields].values
            except Exception as e:
                print(e," Error in line 101 of process_raw_data.py, no trial in : {}".format(trial))

    return trials_data # a dictory with trials' data in form of numpy array




'''
suntao drop landing experiment data
'''

def read_rawdata(trials: int,col_names: list,raw_datasets_path=None,**args)-> np.ndarray:
    """
    @Description:
    To read raw data of a subject from h5 file and normalize the features and labels.
    @Parameters:
    trials: sepcify the data of trial 
    col_names: the names of columns. data type is string

    args['assign'], which  concate the data in three dimensions, the first dimension is trial numbers
    
    """
    assert(type(col_names)==list)
    assert(isinstance(data_range,list))
    #1) read h5 data file
    with h5py.File(raw_datasets_path, 'r') as fd:
        # get all subject names, e.g., P_09_libang
        subject_ids_names=list(fd.keys())

        # get all data feilds: the features and labels
        all_data_fields=fd[subject_ids_names[0]].attrs.get('columns')

        dataset_of_trials=[]

        #- specified subject name
        subject_name=args['subject_name']
        if(subject_name not in subject_ids_names):
            print("This subject:{subject_name} is not in datasets, see line 91 in process_rawdata.py".format(subject_name))
            exit()

        #-- get each trial data with specified columns of a subject (subject_name)
        for trial in trials: 
            try:
                #- get all column data of a trial of a subject into a dataframe
                data = fd[subject_name][trial]
                columns=fd[subject_name].attrs['columns']
                temp_pd_data=pd.DataFrame(data=np.array(data),columns=columns)
            except Exception as e:
                print(e," Error in line 101 of process_raw_data.py, no trial in {}".format(subject_name))

            #retrive specified columns by parameter: col_names
            temp_necessary_data=temp_pd_data[col_names].values

            dataset_of_trials.append(temp_necessary_data)

        # extract the drop landing trials, the output is a numpy matrix with three dimensions, the first dimension is trial times
        try:
            if('assign_trials' in args.keys()):
                all_datasets_np=np.array(dataset_of_trials)
            else:
                if(sum([len(trial) for trial in dataset_of_trials])%len(dataset_of_trials[0])==0):# checking each trial data has same dimensions (row and column number)
                    all_datasets_np=np.concatenate(dataset_of_trials,axis=0) # concate into 2 dimension along with row
                else:
                    warnings.warm(termcolor.colored("Trials have different time step numbers, please use a small DROPLANDING_PERIOD"))
        except Exception as e:
            print(e,"Trials have different counts")

        # return 3D shape numpy dataset
        return all_datasets_np



'''
read bingfei drop landing experiment data
'''

def read_bingfei_experiment_data(data_range: int,col_names: list,raw_datasets_path=None,**args)-> np.ndarray:
    """
    @Description:
    To read raw data of a subject from h5 file and normalize the features and labels.
    @Parameters:
    data_range: sepcify the data ranges, it can be a trial, a range, or a tupe (start_index, end_index).
    col_names: the names of columns. data type is string

    args['assign'], which  concate the data in three dimensions, the first dimension is trial numbers
    
    """
    assert(type(col_names)==list)
    #1) read h5 data file
    with h5py.File(raw_datasets_path, 'r') as fd:
        col_idxs=[]
        for col_name in col_names:
            col_idxs.append(np.argwhere(columns==col_name)[0][0])
        
        # 2) length of every subject's datasets
        data_len_list=[]
        for idx in range(len(fd.keys())):
            key="sub_"+str(idx)
            #print(key)
            data_len_list.append(len(fd[key]))
        # 3) sum of subjects' datasets
        data_len_list_sum=[]
        sum_num=0
        for num in data_len_list:
            sum_num+=num
            data_len_list_sum.append(sum_num)
        data_len_list_sum=np.array(data_len_list_sum)


        #-- return data of an row
        if(type(data_range)==int):
            # 4) subject idx and internal data_range
            subject_trials=np.argwhere(data_len_list_sum > data_range)[0,0]
            if(subject_trials>0):
                data_range=data_range-data_len_list_sum[subject_trials-1]
            return fd['sub_'+str(subject_trials)][data_range,col_idxs]
        
        #-- return data with a np.array(list), the list contains each subject's data
        if((isinstance(data_range,list) and re.search('sub_',data_range[0])) or isinstance(data_range,range)): #-- return data of multiple rows
            # 5) load h5 file data into a dic: all_datasets
            all_datasets={subject: subject_data[:] for subject, subject_data in fd.items()}
            # 6) return datasets of multiple rows
            return_dataset=[]
            for row_i in data_range:
                subject_trials=np.argwhere(data_len_list_sum > row_i)[0,0]
                if(subject_trials>0):
                    row_i=row_i-data_len_list_sum[subject_trials-1]
                return_dataset.append(all_datasets['sub_'+str(subject_trials)][row_i,col_idxs])
            return np.array(return_dataset)

        # -- return data with ....
        if(isinstance(data_range,str)): # return data indexed by subject id
            subject_trials=data_range
            assert(subject_trials in ['sub_'+str(ii) for ii in range(15)])
            # 5) load h5 file data into a dic: all_datasets
            all_datasets={subject: subject_data[:] for subject, subject_data in fd.items()}
            # 6) return datasets of multiple rows
            return all_datasets[subject_trials][:,col_idxs]


def normalization_parameters(row_idx,col_names,datarange="all_subject", norm_type="mean_std", raw_datasets_path="./datasets_files/raw_datasets.hdf5"):
    """
    Calculate the mean and std of the dataset for calculating normalization 
    """

    with h5py.File(raw_datasets_path, 'r') as fd:
        keys=list(fd.keys())# the keys/columns name of the h5 datafile 
        columns=fd[keys[0]].attrs.get('columns')
        col_idxs=[]
        # read the needed columns index 
        for col_name in col_names:
            col_idxs.append(np.argwhere(columns==col_name)[0][0])
    
        # cal data length of every subject's data
        data_len_list=[]
        subject_num=len(fd.keys())
        for idx in range(subject_num):
            key="sub_"+str(idx)
            data_len_list.append(len(fd[key]))
        
        # sum the data length 
        data_len_list_sum=[]
        sum_num=0
        for num in data_len_list:
            sum_num+=num
            data_len_list_sum.append(sum_num)
        data_len_list_sum=np.array(data_len_list_sum)
    
        # calculate the subject index and update row index
        subject_trials=np.argwhere(data_len_list_sum > row_idx)[0,0]
        if(subject_trials>0):
            row_idx=row_idx-data_len_list_sum[subject_trials-1]
        

        # --- read datasets
        if(datarange=='one_subject'):
            # read datasets from the h5 file with respect to a specific subect
            for idx, col_idx in enumerate(col_idxs):
                if(idx==0):
                    numpy_datasets=fd['sub_'+str(subject_trials)][:,col_idx]
                else:# stack along with columns
                    numpy_datasets=np.column_stack((numpy_datasets,fd['sub_'+str(subject_trials)][:,col_idx]))
        

        if(datarange=='all_subject'):
            # load h5 file data into a dic: all_datasets
            all_datasets={subject: subject_data[:] for subject, subject_data in fd.items()}
            # read datasets from the h5 file with respect all subects
            for subject_trials in range(subject_num):
                if(subject_trials==0):
                    numpy_datasets=all_datasets['sub_'+str(subject_trials)][:,col_idxs]
                else:# stack along with columns
                    numpy_datasets=np.row_stack((numpy_datasets,all_datasets['sub_'+str(subject_trials)][:,col_idxs]))

        assert(norm_type in ['mean_std','max_min'])
        if(norm_type=="mean_std"):
            mean=np.mean(numpy_datasets,axis=0,keepdims=True)
            std=np.std(numpy_datasets,axis=0,keepdims=True)
            data_mean=pd.DataFrame(data=mean,columns=col_names)
            data_std=pd.DataFrame(data=std,columns=col_names)
            return data_mean, data_std    
        
        if(norm_type=="max_min"):
            max_value=np.max(numpy_datasets,axis=0,keepdims=True)
            min_value=np.min(numpy_datasets,axis=0,keepdims=True)
            data_max=pd.DataFrame(data=max_value,columns=col_names)
            data_min=pd.DataFrame(data=min_value,columns=col_names)
            return data_max, data_min    
        


def create_training_files(model_object=None, hyperparams={'lr':0}, base_folder=os.path.join(RESULTS_PATH,'training_testing/')):
    '''
    Create folder and sub folder for training, as well as model source code and super parameters

    '''

    # create top folder based on date
    date_base_folder = base_folder + str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime()))
    if(os.path.exists(date_base_folder)==False):
        os.makedirs(date_base_folder)

    # create training sub folder
    training_folder=date_base_folder+"/training_"+ str(localtimepkg.strftime("%H%M%S", localtimepkg.localtime()))
    if(os.path.exists(training_folder)==False):
        os.makedirs(training_folder)

    # create train process sub folder
    training_process_folder=training_folder+"/train_process"
    if(os.path.exists(training_process_folder)==False):
        os.makedirs(training_process_folder)

    # create sub folder for loss plots
    training_process_folder_lossplots=training_process_folder+"/lossplots/"
    os.makedirs(training_process_folder_lossplots)

    # create train results sub folder
    training_results_folder=training_folder+"/train_results"
    if(os.path.exists(training_results_folder)==False):
        os.makedirs(training_results_folder)

    # save model source to model.py
    if(model_object!=None):
        model_file=training_folder+"/model_source.py"
        model_class=globals()[type(model_object).__name__]
        source = inspect.getsource(model_class)
        with open(model_file, 'w') as fd:
            fd.write(source)

    # save hyper parameters to hyperparam.yaml
    hyperparams_file = training_folder + "/hyperparams.yaml"
    with open(hyperparams_file, 'w') as fd:
        yaml.dump(hyperparams,fd)

    return training_folder


def save_training_results(training_folder, model, loss):
    """
    @Description: save model parameters: including paramters and loss values
    @Args:
    based_path_folder, model, loss, iteration
    @Output: valid
    """
    # save model parameters
    model_parameters_file=training_folder+"/train_results/"+"model_parameters"+u".pk1"
    if(os.path.exists(model_parameters_file)):
        os.remove(model_parameters_file)
    torch.save(model.state_dict(),model_parameters_file)

    # save trained model
    model_file=training_folder+"/train_results/"+"model"+u".pth"
    if(os.path.exists(model_file)):
        os.remove(model_file)
    torch.save(model,model_file)

    # save loss values
    loss_values_file=training_folder+"/train_results/"+"loss_values"+u".csv"
    if(os.path.exists(loss_values_file)):
        os.remove(loss_values_file)
    pd_loss=pd.DataFrame(data={'train_loss':loss[:,0], 'eval_loss':loss[:,1]})
    pd_loss.to_csv(loss_values_file)

def save_training_process(training_folder, loss):
    """
    @Description: save model parameters: including paramters and loss values
    @Args:
    based_path_folder, model, loss, epochs
    @Output: valid
    """
    # save model parameters
    lossplots=training_folder+"/train_process/lossplots/epochs_"+str(loss.shape[0])+".png"
    epochs=range(loss.shape[0])
    plt.plot(epochs,loss[:,0],'-', lw='1',color='r')
    plt.plot(epochs,loss[:,1],'-', lw='1',color='b')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(['train loss','evaluation loss'])
    plt.savefig(lossplots)



def create_testing_files(training_folder, base_folder=os.path.join(RESULTS_PATH,'training_testing/')):

    # create top folder based on date
    date_base_folder=base_folder+str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime()))
    if(os.path.exists(date_base_folder)==False):
        os.makedirs(date_base_folder)

    # create testing sub folder
    training_id = re.search(r"\d+$",training_folder).group()
    testing_folder = date_base_folder+"/test_" + training_id
    if(os.path.exists(testing_folder)==False):
        os.makedirs(testing_folder)

    # ceate testing sub folder for each test
    test_id = len(os.listdir(testing_folder))+1
    each_testing_folder = testing_folder+"/test_"+str(test_id)
    if(os.path.exists(each_testing_folder)==False):
        os.makedirs(each_testing_folder)

    return each_testing_folder
    

    

def load_test_datasets(test_datasets_range:list,norm_type='max_min',raw_datasets_path="./datasets_files/raw_datasets.hdf5")->zip:
    '''
    Description:
        load datasets for testing model, the datasets have been normalizated.
        The output is a zip object
    '''

    assert(type(test_datasets_range)==list)
    # Load test datasets ranges
    row_idx_start = test_datasets_range[0]
    row_idx_end =   test_datasets_range[1]
    # Read raw datasets
    features=read_rawdata(range(test_datasets_range[0],test_datasets_range[1]),features_names,raw_datasets_path)
    labels=read_rawdata(range(test_datasets_range[0],test_datasets_range[1]),labels_names,raw_datasets_path)
    # The mean and std for normalization and non-normalization
    assert(norm_type in ['mean_std','max_min'])
    if(norm_type=='max_min'):
        data_max,data_min=normalization_parameters(row_idx_start, features_names, datarange="all_subject",norm_type="max_min")
        features_norm=(features-data_min.values)/(data_max.values-data_min.values)
        data_max,data_min=normalization_parameters(row_idx_start, labels_names, datarange="all_subject",norm_type="max_min")
        labels_norm=(labels-data_min.values)/(data_max.values-data_min.values)

    if(norm_type=='mean_std'):
        data_mean, data_std = normalization_parameters(row_idx_start, features_names, datarange="all_subject",norm_type="mean_std")
        features_norm=(features-data_mean.values)/data_std.values
        data_mean, data_std = normalization_parameters(row_idx_start, labels_names, datarange="all_subject",norm_type="mean_std")
        labels_norm=(labels-data_mean.values)/data_std.values

    return zip(features_norm, labels_norm)


def inverse_norm(norm_datasets:np.ndarray, col_names:list, norm_type:str)->np.ndarray:
    '''
    Description: Inverse  the normalizated datasets
    col_names is the column name of the norm_datasets

    '''
    assert(norm_type in ['mean_std','max_min', 'None'])
    if(norm_type=='max_min'):
        data_max,data_min=normalization_parameters(0, col_names, datarange="all_subject",norm_type="max_min")
        datasets=norm_datasets*(data_max.values-data_min.values)+data_min.values
    
    if(norm_type=='mean_std'):
        data_mean, data_std = normalization_parameters(0, col_names, datarange="all_subject",norm_type="mean_std")
        datasets=norm_datasets*data_std.values+data_mean.values
    if(norm_type=='None'):
        datasets=copy.deepcopy(norm_datasets)
    return datasets






def norm_datasets(datasets:np.ndarray,col_names:str,norm_type='mean_std')->np.ndarray:
    '''
    Normalize datasets
    '''
    assert(norm_type in ['mean_std','max_min']), "Incorrect norm type"
    if(norm_type=='max_min'):
        data_max,data_min=normalization_parameters(0,col_names,datarange="all_subject",norm_type="max_min")
        datasets_norm=(datasets-data_min.values)/(data_max.values-data_min.values)
    if(norm_type=='mean_std'):
        data_mean,data_std=normalization_parameters(0,col_names,datarange="all_subject",norm_type="mean_std")
        datasets_norm=(datasets-data_mean.values)/data_std.values

    return datasets_norm
    #datasets=read_rawdata(range(3951),col_names,raw_datasets_path)
    #pd_datasets=pd.DataFrame(data=datasets_norm,columns=col_names)



def plot_rawdataset_curves(datasets, col_names=None, figwidth=9.5,figheight=3,figtitle="Figure",show=False,**args):
    '''
    Params: datasets: a two dimension numpy: experiment trial

    '''
    #0) read datasets
    if(isinstance(datasets,np.ndarray)):
        pd_datasets = pd.DataFrame(data=datasets,columns=col_names)
    if(isinstance(datasets,pd.DataFrame)):
        pd_datasets = copy.deepcopy(datasets)

    dim = pd_datasets.shape
    # add time column
    pd_datasets['time'] = np.linspace(0,dim[0]/100.0,dim[0])
    max_time_tick = max(pd_datasets['time'].values)
    pd_datasets = pd_datasets.melt(id_vars=['time'],var_name='cols',value_name='vals')

    #2) plots
    # plot configuration
    subplot_left=0.08; subplot_right=0.95; subplot_top=0.9;subplot_bottom=0.1; hspace=0.12; wspace=0.12

    # plot
    g=sns.FacetGrid(pd_datasets,col='cols',col_wrap=3,height=2,sharey=False)
    g.map_dataframe(sns.lineplot,'time','vals')
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom,hspace=hspace,wspace=wspace)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    g.set_titles(col_template=figtitle+" {col_name}")
    g.set(xticks=np.linspace(0,max_time_tick,int(max_time_tick*10+1)))# xticks
    # add grid on every axs
    [ax.grid(which='both',axis='both',color='k',linestyle=':') for ax in g.axes]

    # save file
    # create folder
    datasets_visulization_folder = os.path.join(DATA_VISULIZATION_PATH, str(localtimepkg.strftime("%Y-%m-%d_%H")))
    if(not os.path.exists(datasets_visulization_folder)):
        os.makedirs(datasets_visulization_folder)

    # create file
    datasets_visulization_path=os.path.join(datasets_visulization_folder,figtitle+'_'+str(localtimepkg.strftime("%M_%S", localtimepkg.localtime()))+"_rawdata_curves.svg")
    g.savefig(datasets_visulization_path)
    if(show):
        plt.show()

    plt.close("all")



def extract_subject_drop_landing_data(sub_idx: int)->np.ndarray:
    '''
    # Extract drop landing period data
    # The output is a three dimentional array, the first dimension is drop landing times
    # The second dimension is time/rows 
    # The third dimension is the features and labels including XX
    '''
    end=fnn_model_v3.all_datasets_ranges['sub_'+str(sub_idx)]
    start=fnn_model_v3.all_datasets_ranges['sub_'+str(sub_idx-1)]
    sub_data=fnn_model_v3.read_rawdata(range(start,end), columns_names, raw_dataset_path)
    #start=800;
    #end=start+1000
    start=800;
    end=2000
    right_flexion=sub_data[start:end,columns_names.index('R_FE')]
    left_flexion=sub_data[start:end,columns_names.index('L_FE')]
    fig=plt.figure(figsize=(16,10))
    fig.add_subplot(4,1,1)
    plt.plot(right_flexion,'r')
    plt.plot(left_flexion,'g')
    plt.legend(['right knee joint','left knee joint'])
    plt.ylabel('Flexion/Extension [deg]')
    #plt.xticks(range(0,1000,50))
    plt.grid()
    fig.add_subplot(4,1,2)
    multipy_flexion=(right_flexion*left_flexion)#/(right_flexion-left_flexion)
    plt.plot(multipy_flexion)
    #plt.xticks(range(0,1000,50))
    #plt.ylim([0,1000])
    
    threshold=300
    step_signals=(multipy_flexion>threshold).astype(np.float32)
    plt.grid()
    
    delta_vstep_signals=step_signals[1:]-step_signals[:-1]
    
    fig.add_subplot(4,1,3)
    plt.plot(step_signals)
    fig.add_subplot(4,1,4)
    plt.plot(delta_vstep_signals,'r')
    #plt.xticks(range(0,1000,50))
    plt.grid()
    start_drop=np.argwhere(delta_vstep_signals.astype(np.float32)>0.5)
    end_drop=np.argwhere(delta_vstep_signals.astype(np.float32)<-0.5)
    
    start_drop=np.insert(start_drop,0,0,axis=0)
    start_drop_distance=start_drop[1:,:]-start_drop[:-1,:]
    
    for idx in start_drop:
        plt.text(idx-100, 1.5,idx[0],fontsize='small',rotation='vertical')
        
    for idx in end_drop:
        plt.text(idx-100, -1.5,idx[0],fontsize='small',rotation='vertical')
    plt.ylim(-2,2)
    #plt.xlim(900,5000)
    plt.savefig(os.path.joint(RESULTS_PATH,'models_parameters_results/split_droplanding.svg'))
    return (start_drop,end_drop)



def calculate_statistic_variable(data: list, col_names:list, displayed_variables, subjects: list, trial_categories, statistic_methods='max',statistic_value_name='KAM'):
    # retrieve variables of subjects' trials for display from "data"
    variables={}
    for subject in subjects:# subjects
        variables[subject]={}
        one_subject_data=data[subject]
        for trial_idx, trial in enumerate(trial_categories):# trial types/categories
            variables[subject][trial]=[]
            for idx in range(5*trial_idx,5*(trial_idx+1)):# trial numbers
                pd_temp_data= pd.DataFrame(data=one_subject_data[idx,:,:],columns=col_names)
                displayed_values=pd_temp_data[displayed_variables]
                ## get the max value of displayed variables ###
                if(statistic_methods=='max'):
                    variables[subject][trial].append(displayed_values.max().values.max())

                print("The trial:{} of subject:{} in session:{} has values: {}".format(idx, subject,trial,variables[subject][trial]))


    # transfer variables dict into pandas
    pd_variables=pd.DataFrame(variables).T
    for col in pd_variables.columns.values:
        pd_variables=pd_variables.explode(col)

    pd_variables=pd_variables.reset_index().rename(columns={'index':'subject names'}).melt(id_vars='subject names',var_name='trial categories', value_name=statistic_value_name)

    return pd_variables
    




def plot_statistic_variables(pd_variables,x='trial categories',y='PKAM',hue=None,col=None,col_wrap=None,kind='bar'):

    #1) plot configuration
    figwidth=12.3;figheight=12.4
    subplot_left=0.08; subplot_right=0.97; subplot_top=0.95;subplot_bottom=0.06

    # x axis labels
    FPA=[ str(ll) for ll in trial_categories]
    print(FPA)
    ind= np.arange(len(trial_categories))
    
    sns.set_theme(style='whitegrid')

    #2) plot
    g=sns.catplot(data=pd_variables, x=x,y=y,col=col,col_wrap=col_wrap,sharey=False,kind=kind)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    
    # add grid on every axs
    [ax.grid(which='both',axis='x',color='k',linestyle=':') for ax in g.axes]

    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+".svg")
    plt.savefig(datasets_visulization_path)
    plt.show()



def plot_statistic_value_under_fpa(data: list, col_names:list, displayed_variables, subjects: list, trial_categories):
    '''
    Description: Plot peak values of various biomechanic variables for different trials, subjects under various foot progression angles (FPA)
    Parameters: data, a numpy array with three dimensions

    '''

    biomechanic_variables={}
    static_calibration_value={}# the variable values in static phase in baseline trial
    for sub_idx, subject in enumerate(subjects):# subjects
        biomechanic_variables[subject]={}
        one_subject_data=data[subject]
        static_calibration_value[subject]={}# the variable values in static phase in baseline trial
        for cat_idx, category in enumerate(trial_categories):# trial types
            biomechanic_variables[subject][category]=[]
            static_calibration_value[subject][category]=[]# the variable values in static phase in baseline trial
            for idx in range(5*cat_idx,5*(cat_idx+1)):# trials number
                pd_temp_data= pd.DataFrame(data=one_subject_data[idx,:,:],columns=col_names)
                peak_temp = {}
                static_temp = {}
                touch_moment_index=int(DROPLANDING_PERIOD/4) #- check wearable_toolkit, line 173, which define the formula of tocuh_moment 
                for display in displayed_variables: # biomechanic variables 
                    #-- calculate peak values
                    if(re.search('FPA',display)): #FPA
                        peak_temp['TOUCH_'+display]=pd_temp_data[display][touch_moment_index]
                        #-- get static value
                        static_temp['TOUCH_'+display]=pd_temp_data[display].iloc[-1]
                    elif(re.search('PELVIS',display)):
                        peak_temp['TOUCH_'+display]=pd_temp_data[display][touch_moment_index]
                        #-- get static value
                        static_temp['TOUCH_'+display]=pd_temp_data[display].iloc[-1]
                    elif(re.search('THORAX',display)):
                        peak_temp['TOUCH_'+display]=pd_temp_data[display][touch_moment_index]
                        #-- get static value
                        static_temp['TOUCH_'+display]=pd_temp_data[display].iloc[-1]
                    else:#find the peak value at the short touch period
                        border= [max(pd_temp_data[display][touch_moment_index:]), min(pd_temp_data[display][touch_moment_index:])] # find the max and min (nagetive) 
                        peak_temp['PEAK_'+display]=max(border) if abs(max(border))>=abs(min(border)) else min(border)
                        #-- get static value
                        static_temp['PEAK_'+display]=pd_temp_data[display].iloc[-1]
                # --------------------------- #
                peak_temp['trial_type']=category
                # The avarage of left and right leg
                peak_temp['TOUCH_LR_FPA_Z']=round((abs(peak_temp['TOUCH_L_FPA_Z'])+abs(peak_temp['TOUCH_R_FPA_Z']))/2.0,2)
                peak_temp['PEAK_LR_KNEE_MOMENT_X']=max([peak_temp['PEAK_L_KNEE_MOMENT_X'],peak_temp['PEAK_R_KNEE_MOMENT_X']])
                peak_temp['subjects']=subject
                biomechanic_variables[subject][category].append(peak_temp)
                static_calibration_value[subject][category].append(static_temp)# the variable values in static phase in baseline trial
                print("The trial:{} of subject:{} in session:{} has max value: {}".format(idx, subject,category,biomechanic_variables[subject][category][-1]))
    
    # 2)  Tansfer the dict data into pandas dataframe
    pd_biomechanic_variables_list=[]
    for idx,subject in enumerate(subjects):
        for category in trial_categories:
            temp=pd.DataFrame(data=biomechanic_variables[subject][category])
            pd_biomechanic_variables_list.append(temp)
    
    pd_biomechanic_variables=pd.concat(pd_biomechanic_variables_list)
    pd_biomechanic_variables.reset_index(inplace=True)# index from 0 to few hundreds


    # 2.1  Tansfer the static dict data into pandas dataframe
    pd_static_value_list=[]
    for idx,subject in enumerate(subjects):
        for category in trial_categories:
            temp=pd.DataFrame(data=static_calibration_value[subject][category])
            pd_static_value_list.append(temp)
    
    pd_static=pd.concat(pd_static_value_list)
    pd_static.reset_index(inplace=True)# index from 0 to few hundreds

    # retrive the varibale of joint angles
    need_calibration_variables = [match for match in pd_biomechanic_variables.columns.tolist() if "ANGLE" in match]

    #-- add TOUCH_LR_FPA - average baseline fpa, self-selected
    baseline_trial_data_mean=pd_biomechanic_variables[pd_biomechanic_variables['trial_type']=='baseline'].groupby('subjects').mean()


    #-- reset the FPA class 
    FPA_categories=['F1','F2','F3','F4']

    # calibrate the values by substract static calibration value
    pd_temp2=[]
    for subject in subjects:
        pd_biomechanic_variables.loc[pd_biomechanic_variables['subjects']==subject,'TOUCH_LR_FPA_Z'] = pd_biomechanic_variables.loc[pd_biomechanic_variables['subjects']==subject,'TOUCH_LR_FPA_Z']-baseline_trial_data_mean['TOUCH_LR_FPA_Z'][subject] # ALL FPA subtract mean FPA
        for col in need_calibration_variables:
            pd_biomechanic_variables.loc[pd_biomechanic_variables['subjects']==subject,col]=pd_biomechanic_variables.loc[pd_biomechanic_variables['subjects']==subject,col]-baseline_trial_data_mean[col][subject] #Angles subtract  static of the baseline trial

        pd_temp=pd_biomechanic_variables.loc[pd_biomechanic_variables['subjects']==subject].sort_values(by='TOUCH_LR_FPA_Z') # sorted by FPA

        each_trial_num= pd_biomechanic_variables[pd_biomechanic_variables['subjects']==subject].shape[0]
        re_trial_type=[int(each_trial_num/len(FPA_categories))*[temp] for temp in FPA_categories]
        re_trial_type=np.array(re_trial_type).reshape(-1,) #add a new column named 'retrial_type'
        while(len(re_trial_type)<each_trial_num):#make sure the reset trial FPA number is same with the original trial FPA number
                re_trial_type=np.append(re_trial_type,FPA_categories[-1])

        pd_temp['re_trial_type']=re_trial_type
        pd_temp2.append(pd_temp)
    
    pd_biomechanic_variables=pd.concat(pd_temp2)
    

    #1.1) Statistically analysis

    # i) norm distribution test for each subject's each FPA class ('re_trial_type')
    x='re_trial_type';y='PEAK_L_KNEE_ANGLE_X'
    var='PEAK_L_KNEE_ANGLE_X'
    variables=['PEAK_L_KNEE_ANGLE_X','PEAK_L_KNEE_ANGLE_Y','PEAK_L_KNEE_ANGLE_Z', 'PEAK_L_KNEE_MOMENT_X','PEAK_L_KNEE_MOMENT_Y','PEAK_L_KNEE_MOMENT_Z','TOUCH_L_FPA_X','TOUCH_PELVIS_ANGLE_X']
    for var in variables:
        norm_distr=pd_biomechanic_variables.groupby(['subjects','re_trial_type']).apply(lambda x: kstest(x[var],'norm',(x[var].mean(),x[var].std()))[1])
        print('variable: ',var, 'of :',norm_distr[norm_distr<0.05].index.to_list())



    # ii) homogeneous analysis of variance, to test the samples of different group from populations with equal variances
    subjects_data=pd_biomechanic_variables.groupby('subjects')
    subjects_name=[nn for nn in subjects_data.groups.keys()]


    for var in variables:
        for sub_idx_1 in range(len(subjects_name)):
            for sub_idx_2 in range(len(subjects_name)):
                for FPA in FPA_categories:
                    sub_idx_1=0
                    x=subjects_data.get_group(subjects_name[sub_idx_1]).groupby('re_trial_type').get_group(FPA)[var]
                    y=subjects_data.get_group(subjects_name[sub_idx_2]).groupby('re_trial_type').get_group(FPA)[var]
                    lv_test=levene(x,y)# 不同对象的同一变量 分组之间 方差齐次
                    if(lv_test.pvalue<0.05):
                        my_equal_var=False
                        print('Not homogenous',lv_test.pvalue,', variable is ',var, ', FPA is ', FPA, ' between subjects: ', subjects_name[sub_idx_1],' and ', subjects_name[sub_idx_2])
                    else:# 方差其次
                        my_equal_var=True
                
                    # iii) Differences analysis, T test
                    ci=0.05
                    t_test_p=stats.ttest_ind(x,y,equal_var=my_equal_var).pvalue
                    if t_test_p < ci:
                        pass
                        print('Significant difference',t_test_p,', variable is ',var, ', FPA is ', FPA, ' between subjects: ', subjects_name[sub_idx_1],' and ', subjects_name[sub_idx_2])



    # save peak metrics
    peak_metrics_data_path=os.path.join(RESULTS_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_peak_metrics.csv")
    pd_biomechanic_variables.to_csv(peak_metrics_data_path)
    
    #2) plot
    # paraemter setup for plot
    test_method = 'Mann-Whitney'
    figwidth=12.3;figheight=12.4
    subplot_left=0.08; subplot_right=0.97; subplot_top=0.95;subplot_bottom=0.06; hspace=0.12; wspace=0.12
    xticklabels=FPA_categories
    sns.set_theme(style='whitegrid')
    pairs=[('FPA_01','FPA_02'),('FPA_02','FPA_03'),('FPA_03','FPA_04'),('FPA_04','FPA_05'),('FPA_05','FPA_06')]
    plot_kind='box'

    
    #每个实验对象的相关系数和拟合的R^2统计
    fitting_R_square={}
    correlation_value={}

    # 计算类别 箱线图和差异性假设检验     **不同FPA 条件下的对比**
    x='re_trial_type';y='PEAK_L_KNEE_ANGLE_X'
    g=sns.catplot(x=x,y=y,height=4,data=pd_biomechanic_variables,kind=plot_kind,col='subjects',col_wrap=4,sharey=False,showfliers=False);
    #annotator=Annotator(g.axes[0],pairs=pairs,data=pd_biomechanic_variables,x=x,y=y)
    #annotator.configure(test=test_method, text_format='star', loc='outside')
    #annotator.apply_and_annotate()
    g.set_axis_labels("FPA", r"Knee flexion angle [deg]")
    g.set_xticklabels(xticklabels)
    plt.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom,hspace=hspace,wspace=wspace)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_KFA.svg")
    plt.savefig(datasets_visulization_path)
   
    x='re_trial_type';y='PEAK_L_KNEE_ANGLE_Y'
    #x='TOUCH_LR_FPA_Z'
    ##g = sns.FacetGrid(data=pd_biomechanic_variables, height=4,col='subjects',  col_wrap=4,sharey=False,sharex=False)
    ##g.map(sns.scatterplot,x, y)
    g=sns.catplot(x=x,y=y,height=4,data=pd_biomechanic_variables,kind=plot_kind,col='subjects',col_wrap=4,sharey=False,showfliers=False);
    #annotator=Annotator(g.axes[0],pairs=pairs,data=pd_biomechanic_variables,x=x,y=y)
    #annotator.configure(test=test_method, text_format='star', loc='outside')
    #annotator.apply_and_annotate()
    g.set_axis_labels("FPA", r"Knee abduction angle [deg]")
    g.set_xticklabels(xticklabels)
    plt.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_KAA.svg")
    plt.savefig(datasets_visulization_path)

    
    x='re_trial_type';y='PEAK_L_KNEE_ANGLE_Z'
    g=sns.catplot(x=x,y=y,height=4,data=pd_biomechanic_variables,kind=plot_kind,col='subjects',col_wrap=4,sharey=False,showfliers=False);
    #annotator=Annotator(g.axes[0],pairs=pairs,data=pd_biomechanic_variables,x=x,y=y)
    #annotator.configure(test=test_method, text_format='star', loc='outside')
    #annotator.apply_and_annotate()
    g.set_axis_labels("FPA", r"Knee ratation angle [deg]")
    g.set_xticklabels(xticklabels)
    plt.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_KRA.svg")
    plt.savefig(datasets_visulization_path)

    x='re_trial_type';y='PEAK_L_KNEE_MOMENT_Y'
    g=sns.catplot(x=x,y=y,height=4,data=pd_biomechanic_variables,kind=plot_kind, col='subjects',col_wrap=4,sharey=False,showfliers=False);
    #annotator=Annotator(g.ax,pairs=pairs,data=pd_biomechanic_variables,x=x,y=y)
    #annotator.configure(test=test_method, text_format='star', loc='outside')
    #annotator.apply_and_annotate()
    g.set_axis_labels("FPA", r"Knee adbuction moment [BW $\cdot$ BH]")
    g.set_xticklabels(xticklabels)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_KAM.svg")
    plt.savefig(datasets_visulization_path)

    x='re_trial_type';y='PEAK_L_KNEE_MOMENT_Z'
    g=sns.catplot(x=x,y=y,height=4,data=pd_biomechanic_variables,kind=plot_kind, col='subjects',col_wrap=4,sharey=False,showfliers=False);
    #annotator=Annotator(g.ax,pairs=pairs,data=pd_biomechanic_variables,x=x,y=y)
    #annotator.configure(test=test_method, text_format='star', loc='outside')
    #annotator.apply_and_annotate()
    g.set_axis_labels("FPA", r"KRM [BW $\cdot$ BH]")
    g.set_xticklabels(xticklabels)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_KRM.svg")
    plt.savefig(datasets_visulization_path)

    x='re_trial_type';y='TOUCH_L_FPA_X'
    g=sns.catplot(x=x,y=y,height=4,data=pd_biomechanic_variables,kind=plot_kind, col='subjects',col_wrap=4,sharey=False,showfliers=False);
    #annotator=Annotator(g.ax,pairs=pairs,data=pd_biomechanic_variables,x=x,y=y)
    #annotator.configure(test=test_method, text_format='star', loc='outside')
    #annotator.apply_and_annotate()
    g.set_axis_labels("FPA", r"Foot heading angle [deg]")
    g.set_xticklabels(xticklabels)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_FPA_X.svg")
    plt.savefig(datasets_visulization_path)

    x='re_trial_type';y='TOUCH_PELVIS_ANGLE_X'
    g=sns.catplot(x=x,y=y,height=4,data=pd_biomechanic_variables,kind=plot_kind, col='subjects', col_wrap=4, sharey=False,showfliers=False);
    #annotator=Annotator(g.ax,pairs=pairs,data=pd_biomechanic_variables,x=x,y=y)
    #annotator.configure(test=test_method, text_format='star', loc='outside')
    #annotator.apply_and_annotate()
    g.set_axis_labels("FPA", r"Trunk pitch angle [deg]")
    g.set_xticklabels(xticklabels)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_Pelvis_X.svg")
    plt.savefig(datasets_visulization_path)


    ## 高阶回归分析和检验    **不同FPA和其他变量的因果关系，cause-effect关系**

    ## 数据中不要包含self-selected trials (baseline), 因为baseline 的FPA和F_03的很接近，包含baseline 会导致 采样集中到F_03附近， 导致用于拟合的数据分布不均匀
    pd_biomechanic_variables=pd_biomechanic_variables[pd_biomechanic_variables['trial_type']!='baseline']
    #-- preprocess data

    reg_order=1# regression model order, if bigger than 1, then use polyfit

    # Each-subjects
    id_vars= pd_biomechanic_variables.columns.tolist()
    id_vars.remove('PEAK_L_KNEE_ANGLE_Y')
    id_vars.remove('PEAK_L_KNEE_ANGLE_Z')
    melt_pd_biomechanic_variables=pd_biomechanic_variables.melt(id_vars=id_vars,var_name='cols',value_name='vals')
    x='TOUCH_LR_FPA_Z';y='vals'
    g=sns.lmplot(x=x,y=y,order=reg_order,height=4,hue='cols',data=melt_pd_biomechanic_variables,legend=False,col='subjects',col_wrap=4,sharey=False,sharex=False,x_ci='sd');
    g.set_axis_labels("FPA [deg]", r"Knee joint angles [deg]")
    #g.ax.set_xticks([-0.5,0,0.5,1.0,1.5,2.0,2.5])
    #g.set_xticklabels(['-0.5','0','0.5','1.0','1.5','2.0','2.5'])
    plt.legend(title='', loc='best', labels=['Peak abduction angle', 'Peak internal rotation angle'])
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_KARA.svg")
    plt.savefig(datasets_visulization_path)



    id_vars= pd_biomechanic_variables.columns.tolist()
    id_vars.remove('PEAK_L_KNEE_ANGLE_X')
    melt_pd_biomechanic_variables=pd_biomechanic_variables.melt(id_vars=id_vars,var_name='cols',value_name='vals')
    x='TOUCH_LR_FPA_Z';y='vals'
    g=sns.lmplot(x=x,y=y,order=reg_order,height=4,hue='cols',data=melt_pd_biomechanic_variables,legend=False,col='subjects',col_wrap=4,sharey=False,sharex=False);
    g.set_axis_labels("FPA [deg]", r"Knee flexion angle [deg]")
    #g.ax.set_xticks([-0.5,0,0.5,1.0,1.5,2.0,2.5])
    #g.set_xticklabels(['-0.5','0','0.5','1.0','1.5','2.0','2.5'])
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    #g.ax.legend(title='', loc='upper left', labels=['Peak abduction moment', 'Peak internal rotation moment'])
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_KFA.svg")
    plt.savefig(datasets_visulization_path)

    #-- Knee Moment
    id_vars= pd_biomechanic_variables.columns.tolist()
    id_vars.remove('PEAK_L_KNEE_MOMENT_X')
    melt_pd_biomechanic_variables=pd_biomechanic_variables.melt(id_vars=id_vars,var_name='cols',value_name='vals')
    x='TOUCH_LR_FPA_Z';y='vals'
    g=sns.lmplot(x=x,y=y,order=reg_order,height=4,hue='cols',data=melt_pd_biomechanic_variables,legend=False,col='subjects',col_wrap=4,sharex=False,sharey=False);
    g.set_axis_labels("FPA [deg]", r"Knee flexion moments [BM$\cdot$BH]")
    #g.ax.set_xticks([-0.5,0,0.5,1.0,1.5,2.0,2.5])
    #g.set_xticklabels(['-0.5','0','0.5','1.0','1.5','2.0','2.5'])
    #g.ax.legend(title='', loc='upper left', labels=['Peak abduction moment', 'Peak internal rotation moment'])
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_KFM.svg")
    plt.savefig(datasets_visulization_path)


    id_vars= pd_biomechanic_variables.columns.tolist()
    id_vars.remove('PEAK_L_KNEE_MOMENT_Y')
    id_vars.remove('PEAK_L_KNEE_MOMENT_Z')
    melt_pd_biomechanic_variables=pd_biomechanic_variables.melt(id_vars=id_vars,var_name='cols',value_name='vals')
    x='TOUCH_LR_FPA_Z';y='vals'
    g=sns.lmplot(x=x,y=y,order=reg_order,height=4,hue='cols',data=melt_pd_biomechanic_variables,legend=False,col='subjects', col_wrap=4, sharey=False,sharex=False);
    g.set_axis_labels("FPA [deg]", r"Knee joint moments [BM$\cdot$BH]")
    #g.ax.set_xticks([-0.5,0,0.5,1.0,1.5,2.0,2.5])
    #g.set_xticklabels(['-0.5','0','0.5','1.0','1.5','2.0','2.5'])
    #g.ax.legend(title='', loc='upper left', labels=['Peak abduction moment', 'Peak internal rotation moment'])
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_KARM.svg")
    plt.savefig(datasets_visulization_path)

    #pdb.set_trace()
    # All-subject together 回归分析
    #- plot_setup

    subplot_left=0.12; subplot_right=0.9; subplot_top=0.9;subplot_bottom=0.12
    figwidth=5;figheight=5

    id_vars= pd_biomechanic_variables.columns.tolist()
    id_vars.remove('PEAK_L_KNEE_ANGLE_Y')
    id_vars.remove('PEAK_L_KNEE_ANGLE_Z')
    melt_pd_biomechanic_variables=pd_biomechanic_variables.melt(id_vars=id_vars,var_name='cols',value_name='vals')
    x='TOUCH_LR_FPA_Z';y='vals'
    g=sns.lmplot(x=x,y=y,order=reg_order,height=4,hue='cols',data=melt_pd_biomechanic_variables,legend=False);
    g.set_axis_labels("FPA [deg]", r"Knee joint angles [deg]")
    #g.ax.set_xticks([-0.5,0,0.5,1.0,1.5,2.0,2.5])
    #g.set_xticklabels(['-0.5','0','0.5','1.0','1.5','2.0','2.5'])
    plt.legend(title='', loc='best', labels=['Peak abduction angle', 'Peak internal rotation angle'])
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_all_KARA.svg")
    plt.savefig(datasets_visulization_path)


    id_vars= pd_biomechanic_variables.columns.tolist()
    id_vars.remove('PEAK_L_KNEE_MOMENT_Y')
    id_vars.remove('PEAK_L_KNEE_MOMENT_Z')
    melt_pd_biomechanic_variables=pd_biomechanic_variables.melt(id_vars=id_vars,var_name='cols',value_name='vals')
    x='TOUCH_LR_FPA_Z';y='vals'
    g=sns.lmplot(x=x,y=y,order=reg_order,height=4,hue='cols',data=melt_pd_biomechanic_variables,legend=False);
    g.set_axis_labels("FPA [deg]", r"Knee joint moments [BM$\cdot$BH]")
    #g.ax.set_xticks([-0.5,0,0.5,1.0,1.5,2.0,2.5])
    #g.set_xticklabels(['-0.5','0','0.5','1.0','1.5','2.0','2.5'])
    #g.ax.legend(title='', loc='upper left', labels=['Peak abduction moment', 'Peak internal rotation moment'])
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_all_KARM.svg")
    plt.savefig(datasets_visulization_path)


    x='TOUCH_LR_FPA_Z';y='PEAK_L_KNEE_ANGLE_Y'
    g=sns.lmplot(x=x,y=y,order=reg_order,height=4,data=pd_biomechanic_variables);
    g.set_axis_labels("FPA [deg]", r"Knee abduction angle [deg]")
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_all_KAA.svg")
    plt.savefig(datasets_visulization_path)

    x='TOUCH_LR_FPA_Z'; y='PEAK_L_KNEE_ANGLE_Z'
    g=sns.lmplot(x=x,y=y,order=reg_order,height=4,data=pd_biomechanic_variables);
    g.set_axis_labels("FPA [deg]", r"Knee ratation angle [deg]")
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_all_KRA.svg")
    plt.savefig(datasets_visulization_path)

    x='TOUCH_LR_FPA_Z';y='PEAK_L_KNEE_MOMENT_Y'
    g=sns.lmplot(x=x,y=y,order=reg_order,height=4,data=pd_biomechanic_variables);
    g.set_axis_labels("FPA [deg]", r"Knee adbuction moment [BW $\cdot$ BH]")
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_all_KAM.svg")
    plt.savefig(datasets_visulization_path)

    x='TOUCH_LR_FPA_Z';y='PEAK_L_KNEE_MOMENT_Z'
    g=sns.lmplot(x=x,y=y,order=reg_order,height=4,data=pd_biomechanic_variables);
    g.set_axis_labels("FPA [deg]", r"KRM [BW $\cdot$ BH]")
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_all_KRM.svg")
    plt.savefig(datasets_visulization_path)

    x='TOUCH_LR_FPA_Z';y='TOUCH_PELVIS_ANGLE_X'
    g=sns.lmplot(x=x,y=y,order=reg_order,height=4,data=pd_biomechanic_variables);
    g.set_axis_labels("FPA [deg]", r"TFA [deg]")
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_all_TFA.svg")
    plt.savefig(datasets_visulization_path)


    ### 计算相关系数矩阵，热力图
    #每个实验对象的热力图
    figwidth=12.3;figheight=12.4
    subplot_left=0.08; subplot_right=0.97; subplot_top=0.95;subplot_bottom=0.06
    columns=['TOUCH_LR_FPA_Z','TOUCH_L_FPA_X', 'TOUCH_PELVIS_ANGLE_X', 'PEAK_L_KNEE_ANGLE_Z','PEAK_L_KNEE_ANGLE_Y','PEAK_L_KNEE_MOMENT_Z','PEAK_L_KNEE_MOMENT_Y','PEAK_L_KNEE_ANGLE_X','PEAK_L_KNEE_MOMENT_X']
    cormat=round(pd_biomechanic_variables.groupby('subjects')[columns].corr(method='spearman'),2)
    cormat=abs(cormat)# 取绝对值
    def draw_heatmap(*args,**kwargs):
        data=kwargs.pop('data')
        #pdb.set_trace()
        dd=data.set_index('level_1').drop('subjects',axis=1)
        mask=np.zeros_like(dd)
        mask[np.triu_indices_from(mask)] = True
        mask[:,3:]=True
        sns.heatmap(dd,mask=mask,annot=kwargs['annot'],square=kwargs['square'],cmap='YlGnBu')
    g = sns.FacetGrid(data=cormat.reset_index(), height=4,col='subjects',  col_wrap=4,sharey=False,sharex=False)
    g.map_dataframe(draw_heatmap,annot=False,square=True)

    g.set_xticklabels(['FPA','FFA','TFA','KRA','KAA','KRM','KAM','KFA','KFM'])
    g.set_yticklabels(['FPA','FFA','TFA','KRA','KAA','KRM','KAM','KFA','KFM'])
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom,hspace=0.2,wspace=0.2)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    # adjust the titles of the subplots
    for ax in g.axes:
        ax.title.set_position([.5, 0.5])
        ax.yaxis.labelpad = 25

    g.fig.tight_layout()
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_each_heatmap.svg")
    plt.savefig(datasets_visulization_path)

    #-- save correlation parameter values
    #pdb.set_trace()

    #所有实验对象在一起的热力图
    figsize=(5.5*1.2,4*1.2)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    #cormat=round(pd_biomechanic_variables[['TOUCH_LR_FPA_Z','TOUCH_PELVIS_ANGLE_X','TOUCH_L_FPA_X']].corr(),2)
    columns=['TOUCH_LR_FPA_Z','TOUCH_L_FPA_X', 'TOUCH_PELVIS_ANGLE_X', 'PEAK_L_KNEE_ANGLE_Z','PEAK_L_KNEE_ANGLE_Y','PEAK_L_KNEE_MOMENT_Z','PEAK_L_KNEE_MOMENT_Y','PEAK_L_KNEE_ANGLE_X','PEAK_L_KNEE_MOMENT_X']
    cormat=round(pd_biomechanic_variables[columns].corr(method='spearman'),2)
    cormat=abs(cormat)# 取绝对值
    mask=np.zeros_like(cormat)
    mask[np.triu_indices_from(mask)] = True # mask 上三角
    mask[:,3:]=True #mask 第三列之后的列
    f=sns.heatmap(cormat,mask=mask,annot=True,square=True,cmap='YlGnBu')

    f.set_xticklabels(['FPA','FFA','TFA','KRA','KAA','KRM','KAM','KFA','KFM'])
    f.set_yticklabels(['FPA','FFA','TFA','KRA','KAA','KRM','KAM','KFA','KFM'])
    fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom+0.1)
    #fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+"_all_heatmap.svg")
    plt.savefig(datasets_visulization_path)



'''

Select valid subjects and their trials,

inputs are subjects, can be subject ids e.g., [P_01], or subject_ids_names, e.g. [P_01_suntao]

'''
def set_subjects_trials(subject_ids=None, selected=True, landing_manner='double_legs'):
    
    valid_subjects_trials = {}

    if (landing_manner=='double_legs'):
        if(subject_ids==None):
            subject_ids=['P_08', 'P_10', 'P_11', 'P_13', 'P_14', 'P_15','P_16','P_17','P_18','P_19','P_20','P_21','P_22','P_23', 'P_24']
         
        # get subject_ids_names based on subject_ids
        subject_ids_names = get_subject_ids_names(subject_ids)
        # set trials of subjects'
        for subject_id_name in subject_ids_names:
            valid_subjects_trials[subject_id_name] = copy.deepcopy(TRAIN_USED_TRIALS) # TRIALS is "01", "02", ... '30'


        # load_rawdata.py did not load P_09_plibang 09
        if('P_09' in subject_ids) or ('P_09_libang' in subject_ids):
            valid_subjects_trials['P_09_libang'].remove('09')

        if(selected==True):
            # particular subjects with some trials not useful
            # load_rawdata.py did not load P_09_plibang 09
            if('P_09' in subject_ids) or ('P_09_libang' in subject_ids):
                valid_subjects_trials['P_09_libang']=['01','02','03','04','06']

            if('P_11' in subject_ids) or ('P_11_liuchunyu' in subject_ids):
                valid_subjects_trials['P_11_liuchunyu'].remove('18')
                valid_subjects_trials['P_11_liuchunyu'].remove('20')

            if('P_12' in subject_ids) or ('P_12_fuzijun' in subject_ids):
                valid_subjects_trials['P_12_fuzijun'].remove('04')

            if('P_14' in subject_ids) or ('P_14_hunan' in subject_ids):
                valid_subjects_trials['P_14_hunan'].remove('05')
                valid_subjects_trials['P_14_hunan'].remove('06') # CHEST, Accel_Y
                valid_subjects_trials['P_14_hunan'].remove('27') # CHESK, Accel_Y

            if('P_16' in subject_ids) or ('P_16_zhangjinduo' in subject_ids):
                valid_subjects_trials['P_16_zhangjinduo'] = ['01','03','04','06','07','08','09','11','12','13','14','16','18','19','21','22','23','25','26','28','29','30']

            if('P_19' in subject_ids) or ('P_19_xiongyihui' in subject_ids):
                valid_subjects_trials['P_19_xiongyihui'].remove('01') # L_FOOT_Z Accel Z is wrong
                valid_subjects_trials['P_19_xiongyihui'].remove('12') # L_FOOT_Z Accel Z is wrong
                valid_subjects_trials['P_19_xiongyihui'].remove('22') # L_FOOT_Z Accel Z is wrong, CHEST Accel_Y


    elif (landing_manner=='single_leg_R'):
        if(subject_ids==None):
            subject_ids=['P_08', 'P_10', 'P_13', 'P_15','P_16','P_19','P_20','P_21','P_22', 'P_24']

        # get subject_ids_names based on subject_ids
        subject_ids_names = get_subject_ids_names(subject_ids)
        # set trials of subjects'
        for subject_id_name in subject_ids_names:
            valid_subjects_trials[subject_id_name] = copy.deepcopy(TRAIN_USED_TRIALS_SINGLE_LEG) # TRIALS is "31", "32", ... '40'

    elif (landing_manner=='single_leg_L'):
        if(subject_ids==None):
            subject_ids=['P_09', 'P_11', 'P_12', 'P_14','P_17','P_18','P_23']

        # get subject_ids_names based on subject_ids
        subject_ids_names = get_subject_ids_names(subject_ids)
        # set trials of subjects'
        for subject_id_name in subject_ids_names:
            valid_subjects_trials[subject_id_name] = copy.deepcopy(TRAIN_USED_TRIALS_SINGLE_LEG) # TRIALS is "31", "32", ... '40'

    else:
        print("WRONG DROP LANDING MANNER!")
        exit()

    return valid_subjects_trials

'''
        subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0)
        subject_ids_names =[ss for ss in subject_infos.index]
        for subject_id in subject_ids:
            for subject_id_name in subject_ids_names:
                if(re.search(subject_id, subject_id_name)!=None): # checking the sub_idx_id is in subject_infos
                    valid_subjects_trials[subject_id_name] = copy.deepcopy(TRAIN_USED_TRIALS_SINGLE_LEG) # TRIALS is "31", "32", ... '40'
                    break;
'''


'''
----------------------          START  -------------------------------
----------------------    Dataset process  -------------------------------

Normalize all subject data for model training
# This is for process suntao experimental datasets

'''


def load_normalize_data(hyperparams, scaler='standard', **kwargs):

    subjects_trials = hyperparams['subjects_trials']
    
    #0) load raw datasets
    subjects_trials_data = {}
    subject_ids_names = []
    assert(isinstance(subjects_trials, dict))
    # extract the trials as the first dimension 
    for subject_id_name, trials in subjects_trials.items():# subjects_trials-> {sub_name:trials, sub_name:trials,...}
        subject_ids_names.append(subject_id_name)
        # this output is list contain many three dimension array
        subject_trials_data = read_subject_trials(
                                                 subject_id_name, 
                                                 trials, 
                                                 hyperparams['columns_names'], 
                                                 hyperparams['raw_dataset_path']
                                                 )

        subjects_trials_data[subject_id_name] = {}
        for trial in trials:
            try:
                subjects_trials_data[subject_id_name][trial] = subject_trials_data[trial]
            except Exception as e:
                print(e)
                pdb.set_trace()

    #1) synchronize (scaled) features and labels based on the event: touch index
    if('syn_features_labels' in kwargs.keys()):   
        if(kwargs['syn_features_labels']==True):
            syn_subjects_trials_data = syn_features_lables_events(subjects_trials_data, hyperparams)
            subjects_trials_data = copy.deepcopy(syn_subjects_trials_data)

    #2) concate them into a numpy for scale
    np_subjects_trials_data = np.concatenate([subjects_trials_data[subject_id_name][trial] for subject_id_name in subject_ids_names for trial in subjects_trials[subject_id_name] ],axis=0)
    
    #3) normalizate data
    #i) normalization method
    if scaler=='standard':
        scaler=StandardScaler()
    if scaler=='minmax':
        scaler=MinMaxScaler()
    if scaler=='robust':
        scaler=RobustScaler()

    try:
        scaler.fit(np_subjects_trials_data)
        scaled_np_subjects_trials_data = scaler.transform(np_subjects_trials_data.astype(np.float32))
    except Exception as e:
        print(e)
        pdb.set_trace()

    #4) trasnfer scaled subjects trials data into a dictory of subjects and trials
    scaled_subjects_trials_data={}
    idx = 0 # trial index
    for subject_id_name in subject_ids_names:
        scaled_subjects_trials_data[subject_id_name] = {}
        for trial in subjects_trials[subject_id_name]:
            a_trial_data_row_num = subjects_trials_data[subject_id_name][trial].shape[0] # data shape of a trial
            scaled_subjects_trials_data[subject_id_name][trial] = scaled_np_subjects_trials_data[idx*a_trial_data_row_num:(idx+1)*a_trial_data_row_num,:]
            idx = idx + 1
    
    return subjects_trials_data, scaled_subjects_trials_data, scaler




'''
synchronize features and lables data using their own touch moment event

The touch moment event of IMU can be recongnized by its maximum Acc value of left foot

The touch moment event of Force plate is 1/4* DROPLANDING_PERIOD

'''

def syn_features_lables_events(subjects_trials_data, hyperparams):
    
    assert(SYN_DROPLANDING_PERIOD < DROPLANDING_PERIOD)

    syn_subjects_trials_data={}
    for subject, trials in subjects_trials_data.items():
        syn_subjects_trials_data[subject]={}
        for trial, data in trials.items():
            # transfer the data into pandas
            pd_trial_data = pd.DataFrame(data=data, columns=hyperparams['columns_names'])
            pd_features_data = pd_trial_data[hyperparams['features_names']]
            pd_labels_data = pd_trial_data[hyperparams['labels_names']]

            # imu sensor list, how many imu are used 
            imu_sensor_list=[]
            for imu in IMU_SENSOR_LIST:
                if(imu in ''.join(hyperparams['features_names'])):
                    imu_sensor_list.append(imu)

            # calculate norm of each acceleteor and their peak value index
            peak_accelvalue_index = 0
            croped_features_data = []
            prefix_frames, suffix_frames = 10, SYN_DROPLANDING_PERIOD-10-1
            # crop features data (IMU sensor data)
            for sensor in imu_sensor_list:
                # accelemeter fields
                target_leg = hyperparams['target_leg']# left(L), right(R) or double
                landing_manner = hyperparams['landing_manner'] # single or double legs
                if target_leg + '_FOOT_Accel_Z' in pd_features_data.columns:
                    sensor_accel_fields = ['L_FOOT' + '_Accel_Z']
                elif target_leg + '_SHANK' in imu_sensor_list:
                    sensor_accel_fields = ['L_SHANK' + '_Accel_Y']
                elif target_leg + '_THIGH' in imu_sensor_list:
                    sensor_accel_fields = ['L_THIGH' + '_Accel_Y']
                elif 'WAIST' in imu_sensor_list:
                    sensor_accel_fields = ['WAIST' + '_Accel_Y']
                elif 'CHEST' in imu_sensor_list:
                    sensor_accel_fields = ['CHEST' + '_Accel_Y']
                else:
                    sensor_accel_fields = [sensor+'_Accel_X', sensor+'_Accel_Y', sensor+'_Accel_Z']


                # imu fields of a sensor
                if('Mag' in ''.join(hyperparams['features_names'])):# include magnetometer
                    sensor_fields = [sensor+ "_" + imu_field for imu_field in IMU_RAW_FIELDS]
                else:
                    sensor_fields = [sensor+ "_" + imu_field for imu_field in ACC_GYRO_FIELDS]

                # norm of accelemeter fields
                sensor_norm = np.linalg.norm(pd_features_data[sensor_accel_fields].values, axis=1)

                # index of the peak norm 
                #peak_accelvalue_index = np.argmax(sensor_norm, axis=0)
                peak_accelvalue_index = np.argmax(pd_features_data[sensor_accel_fields].values,axis=0)[0]

                # check whether the index of the peak value is proper
                if((peak_accelvalue_index < prefix_frames) or ((peak_accelvalue_index+suffix_frames) >= DROPLANDING_PERIOD)):
                    print("peak value index: {}  is out boardary in {} of {}".format(peak_accelvalue_index, subject,trial))
                    pdb.set_trace()

                # crop features data (imu data) based on the accelemeter norm
                try:
                    a_sensor_features = pd_features_data.loc[peak_accelvalue_index - prefix_frames : peak_accelvalue_index + suffix_frames, sensor_fields]
                except Exception as e:
                    print(e)
                    pdb.set_trace()
                
                # remove index of the dataframe
                a_sensor_features = a_sensor_features.reset_index(drop=True) # remove the old index and reset the index from 0 to end
                croped_features_data.append(a_sensor_features)

            # crop labels data
            croped_labels_data = pd_labels_data.loc[int(DROPLANDING_PERIOD/4)-prefix_frames:int(DROPLANDING_PERIOD/4)+suffix_frames,:]
            croped_labels_data = croped_labels_data.reset_index(drop=True) # remove the old index and reset the index from 0 to the end

            # merge croped features (IMU sensor data) and labels data
            croped_trial_data = [temp  for temp in croped_features_data]
            croped_trial_data.append(croped_labels_data)
            pd_syn_trial_data = pd.concat(croped_trial_data, axis=1)
            
            # checking whether there are NULL in datasets, if so, the index is out of boundary
            if(pd_syn_trial_data.isnull().values.any()):
                print("There are some NaN in datasets, please check syn function in line 1335")
                pdb.set_trace()
                exit()

            syn_trial_data = pd_syn_trial_data[hyperparams['columns_names']].values
            syn_subjects_trials_data[subject][trial]=syn_trial_data

    return syn_subjects_trials_data






'''
    if(row_mode!=MULTI_NAME_TRIAL_MODE):
        # Get the initial stage data of every subjects
        init_stage_sub_ranges={keys:range(all_datasets_ranges['sub_'+str(int(keys[4:])-1)],all_datasets_ranges['sub_'+str(int(keys[4:])-1)]+300)
                      for keys in ['sub_'+str(idx) for idx in range(14)]}
        init_stage_sub_data={keys:read_rawdata(values,hyperparams['columns_names'],hyperparams['raw_dataset_path']) for keys, values in init_stage_sub_ranges.items()}
        # Normalization of init-stage data
        scaled_init_stage_sub_data={keys:scaler.transform(values).mean(axis=0,keepdims=True) 
                                    for keys, values in init_stage_sub_data.items()}
        #empirically_orientation=[-2.32, 2.77,-3.5,  -8.65, 4.0, 3.01, -0.22, -2.11, 5.29, -9.47, -2.85, -1.33]
'''


def get_subject_ids_names(selected_subject_ids):
    '''
    Get subjects' ids and names ([P_01_suntao, P_02_liyan]) based on listed subjects_ids [P_01, P_02] and subject_info.csv

    '''

    # load subject info from a file
    subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0, header=0)
    subject_ids_names = [ss for ss in subject_infos.index]


    selected_subject_ids_names=[]
    for selected_subject_id in selected_subject_ids:
        #i) check whether the selected subject is in subject_infos.csv
        for subject_id_name in subject_ids_names:
            if(re.search(selected_subject_id,subject_id_name)!=None):
                selected_subject_id_name = subject_id_name
                break

        print(selected_subject_id_name)
        selected_subject_ids_names.append(selected_subject_id_name)

    return selected_subject_ids_names




if __name__=='__main__':

    #-- define hyper parameters
    multi_subject_data={}
    hyperparams={}
    hyperparams['raw_dataset_path']= os.path.join(DATA_PATH,'features_labels_rawdatasets.hdf5')

    #-- select subjects 
    selected_subject_ids=['P_08','P_09','P_10', 'P_11', 'P_12', 'P_13', 'P_14', 'P_15','P_16','P_17','P_18','P_19','P_20','P_21','P_22','P_23', 'P_24']

    #-- subject info
    subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0, header=0)
    subject_ids_names = [ss for ss in subject_infos.index]

    # select subjects
    selected_subject_ids_names = get_subject_ids_names(selected_subject_ids)

    #for selected_subject_id_name in selected_subject_ids_names:
    #i) define hyperparams values: subject, columns_names
    hyperparams['subjects_trials'] = set_subjects_trials(selected=True, landing_manner='single_leg_R')
    
    #ii) set data fields in 'columns_names'
    labels_fields = ['L_GRF_X','L_GRF_Y','L_GRF_Z','R_GRF_X','R_GRF_Y','R_GRF_Z']
    hyperparams['features_names'] = FEATURES_FIELDS
    hyperparams['labels_names'] = labels_fields
    hyperparams['columns_names'] = hyperparams['features_names'] + hyperparams['labels_names']

    #iii) load multiple subject data, the output subjects trials data columns with indicated sequences
    subjects_trials_data, scaled_subjects_trials_data, scaler = load_normalize_data(hyperparams=hyperparams, scaler='standard', syn_features_labels=False)

    print(selected_subject_ids_names)
    #-- subject height
    subject_heights = [float(subject_infos['body height'][sub_name]) for sub_name in selected_subject_ids_names]
    #-- subject mass
    subject_masses = [float(subject_infos['body weight'][sub_name]) for sub_name in selected_subject_ids_names]

    #---------------------------------PLOT 1 ----------------------------#
    #data=pd.DataFrame(data=subjects_trials_data['P_11_liuchunyu']['31'],columns=hyperparams['columns_names'])
    #displayed_data=data[['R_FOOT_Accel_X','R_FOOT_Accel_Y','R_FOOT_Accel_Z','R_GRF_X', 'R_GRF_Y', 'R_GRF_Z']]
    #displayed_data=data[['L_FOOT_Accel_X','L_FOOT_Accel_Y','L_FOOT_Accel_Z','L_GRF_X','L_GRF_Y','L_GRF_Z']]
    #displayed_data=data[['L_GRF_X','L_GRF_Y','L_GRF_Z','R_GRF_X', 'R_GRF_Y', 'R_GRF_Z']]
    #plot_rawdataset_curves(displayed_data,figheight=6,figtitle='P_11 trial 01',show=True)

    #pdb.set_trace()
    #---------------------------------PLOT 2 ----------------------------#
    # checking dataset of each trials of each subjects
    for subject, trials in hyperparams['subjects_trials'].items():
        for trial in trials:
            data = pd.DataFrame(data=subjects_trials_data[subject][trial], columns=hyperparams['columns_names'])
            #displayed_data=data[['L_GRF_X','L_GRF_Y','L_GRF_Z','R_GRF_X', 'R_GRF_Y', 'R_GRF_Z']]
            displayed_data=data[['R_FOOT_Accel_X','R_FOOT_Accel_Y','R_FOOT_Accel_Z','R_GRF_X', 'R_GRF_Y', 'R_GRF_Z']]
            plot_rawdataset_curves(displayed_data, figheight=6, figtitle=subject+"_"+trial)

    pdb.set_trace()
    #---------------------------------PLOT 3 ----------------------------#
    #-- plot statistic knee moment under various fpa
    trial_categories = ['fpa_01','fpa_02','fpa_03','fpa_04','fap_05']
    displayed_variables = ['L_GRF_Z']
    #pd_statistic_variables = calculate_statistic_variable(multi_subject_data,hyperparams['columns_names'],displayed_variables,selected_subject_ids,trial_categories)
    #plot_statistic_variables(pd_statistic_variables,x='trial categories',y='KAM',col='subject names',col_wrap=len(selected_subject_ids)//4)


    #---------------------------------PLOT 4 ----------------------------#
    #-- display statistic peak value under various fpa
    trial_categories = ['baseline','fpa_01','fpa_02','fpa_03','fpa_04','fap_05']
    display_bio_variables = ['L_FPA_Z','R_FPA_Z',
                            'L_GRF_X', 'R_GRF_X', 'L_GRF_Y', 'R_GRF_Y', 'L_GRF_Z', 'R_GRF_Z',
                            'L_ANKLE_ANGLE_X','R_ANKLE_ANGLE_X', 'L_ANKLE_ANGLE_Y','R_ANKLE_ANGLE_Y', 'L_ANKLE_ANGLE_Z','R_ANKLE_ANGLE_Z',
                            'L_ANKLE_MOMENT_X','R_ANKLE_MOMENT_X', 'L_ANKLE_MOMENT_Y','R_ANKLE_MOMENT_Y', 'L_ANKLE_MOMENT_Z','R_ANKLE_MOMENT_Z',
                            'L_KNEE_ANGLE_X','R_KNEE_ANGLE_X', 'L_KNEE_ANGLE_Y','R_KNEE_ANGLE_Y', 'L_KNEE_ANGLE_Z','R_KNEE_ANGLE_Z',
                            'L_KNEE_MOMENT_X','R_KNEE_MOMENT_X', 'L_KNEE_MOMENT_Y','R_KNEE_MOMENT_Y', 'L_KNEE_MOMENT_Z','R_KNEE_MOMENT_Z',
                           ]
    display_bio_variables = ['L_FPA_Z','R_FPA_Z', 'L_FPA_X','R_FPA_X',  'L_KNEE_MOMENT_X','R_KNEE_MOMENT_X','R_KNEE_MOMENT_Y',  'R_KNEE_MOMENT_Z','L_KNEE_MOMENT_Z',  'L_KNEE_MOMENT_Y','L_KNEE_ANGLE_Y','R_KNEE_ANGLE_Y',  'L_KNEE_ANGLE_X','R_KNEE_ANGLE_X',   'L_KNEE_ANGLE_Z','R_KNEE_ANGLE_Z' ,'PELVIS_ANGLE_X','THORAX_ANGLE_X']
    #plot_statistic_value_under_fpa(multi_subject_data, hyperparams['columns_names'], display_bio_variables, selected_subject_ids, trial_categories)

    

