#Python

"""
Description:
    This is an module to process data , it is a base libe to implement ann to predict knee joint values in drop landing experiments

Author: Sun Tao
Email: suntao.hn@gmail.com
Date: 2021-07-01

"""
import numpy
import pandas as pd
import os
import h5py
import re

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from matplotlib import gridspec

import copy
import matplotlib.pyplot as plt
import time as localtimepkg
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import inspect
import yaml
import pdb
import re
import warnings
import termcolor

import datetime


from scipy.stats import normaltest 
from scipy.stats import kstest
from scipy.stats import shapiro
from scipy.stats import levene
from scipy import stats

from statannotations.Annotator import Annotator

if __name__=='__main__':
    from const import FEATURES_FIELDS, LABELS_FIELDS, V3D_LABELS_FIELDS, DATA_PATH, TRIALS, DATA_VISULIZATION_PATH, DROPLANDING_PERIOD, EXPERIMENT_RESULTS_PATH
else:
    from vicon_imu_data_process.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH, TRIALS, DATA_VISULIZATION_PATH, DROPLANDING_PERIOD,V3D_LABELS_FIELDS,EXPERIMENT_RESULTS_PATH


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

def read_rawdata(row_idx: int,col_names: list,raw_datasets_path=None,**args)-> numpy.ndarray:
    """
    @Description:
    To read the data from h5 file and normalize the features and labels.
    @Parameters:
    Row_idx: the index of row. data type is int
    Col_names: the names of columns. data type is string

    args['assign'], which  concate the data in three dimensions, the firsy dimension is trial numbers
    
    """
    assert(type(col_names)==list)
    #--  read h5 data file
    with h5py.File(raw_datasets_path, 'r') as fd:
        # all subject names
        subject_names=list(fd.keys())
        # the data_filed (coloms) of the features and labels
        all_data_fields=fd[subject_names[0]].attrs.get('columns')
        col_idxs=[]

        #-- suntao drop landing experiment data
        if(isinstance(row_idx,list) and isinstance(row_idx[0],str)):
            all_datasets_list=[]
            trials=row_idx
            #- specified subject name
            subject_name=args['subject_name']
            if(subject_name not in subject_names):
                print("This subject:{subject_name} is not in datasets".format(subject_name))
                exit()

            #-- get each trial data with specified columns
            for trial in trials:
                try:
                    #- get all column data of a trial of a subject into a dataframe
                    temp_pd_data=pd.DataFrame(data=np.array(fd[subject_name][trial]),columns=fd[subject_name].attrs['columns'])
                except Exception as e:
                    print(e)
                #-- read the specified columns by parameter: col_names
                temp_necessary_data=temp_pd_data[col_names].values
                all_datasets_list.append(temp_necessary_data)

            # extract the drop landing trials, the output is a numpy matrix with three dimensions, the first dimension is trial times
            try:
                if('assign_trials' in args.keys()):
                    all_datasets_np=np.array(all_datasets_list)
                else:
                    if(sum([len(trial) for trial in all_datasets_list])%DROPLANDING_PERIOD==0):
                        all_datasets_np=np.concatenate(all_datasets_list,axis=0)
                    else:
                        warnings.warm(termcolor.colored("Trials have different time step numbers, please use a small DROPLANDING_PERIOD"))
            except Exception as e:
                print(e,"Trials have different counts")
            return all_datasets_np

        # bingfei drop landing experiment data
        else:
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
        if(type(row_idx)==int):
            # 4) subject idx and internal row_idx
            sub_idx=np.argwhere(data_len_list_sum > row_idx)[0,0]
            if(sub_idx>0):
                row_idx=row_idx-data_len_list_sum[sub_idx-1]
            return fd['sub_'+str(sub_idx)][row_idx,col_idxs]
        
        #-- return data with a np.array(list), the list contains each subject's data
        if((isinstance(row_idx,list) and re.search('sub_',row_idx[0])) or isinstance(row_idx,range)): #-- return data of multiple rows
            # 5) load h5 file data into a dic: all_datasets
            all_datasets={subject: subject_data[:] for subject, subject_data in fd.items()}
            # 6) return datasets of multiple rows
            return_dataset=[]
            for row_i in row_idx:
                sub_idx=np.argwhere(data_len_list_sum > row_i)[0,0]
                if(sub_idx>0):
                    row_i=row_i-data_len_list_sum[sub_idx-1]
                return_dataset.append(all_datasets['sub_'+str(sub_idx)][row_i,col_idxs])
            return np.array(return_dataset)

        # -- return data with ....
        if(isinstance(row_idx,str)): # return data indexed by subject id
            sub_idx=row_idx
            assert(sub_idx in ['sub_'+str(ii) for ii in range(15)])
            # 5) load h5 file data into a dic: all_datasets
            all_datasets={subject: subject_data[:] for subject, subject_data in fd.items()}
            # 6) return datasets of multiple rows
            return all_datasets[sub_idx][:,col_idxs]


        
            
def load_normalize_data(hyperparams,scaler=None,**args):

    sub_idx=hyperparams['sub_idx']
    
    [SINGLE_MODE, MULTI_NUM_MODE, MULTI_NAME_MODE, MULTI_NAME_TRIAL_MODE]=range(4)
    #**** Single subject test
    if(isinstance(sub_idx,int)):
        row_mode=SINGLE_MODE
        start=all_datasets_ranges['sub_'+str(sub_idx-1)]
        end=all_datasets_ranges['sub_'+str(sub_idx)]
        series=read_rawdata(range(start,end),hyperparams['columns_names'],hyperparams['raw_dataset_path'])
        print("Raw data of subject {:}, rows from {:} to {:}".format(sub_idx,start,end))
    
    #**** Multiple subject data indexed by numbers
    if(isinstance(sub_idx,list) and isinstance(sub_idx[0],int)):
        row_mode=MULTI_NUM_MODE
        start_sub_num=int(sub_idx[0])
        end_sub_num=int(sub_idx[-1])
        start=all_datasets_ranges['sub_'+str(start_sub_num-1)]
        end=all_datasets_ranges['sub_'+str(end_sub_num)]
        series=read_rawdata(range(start,end),hyperparams['columns_names'],hyperparams['raw_dataset_path'])
        print("Raw data of subject {:}, rows from {:} to {:}".format(sub_idx,start,end))
    
    #**** Multiple subject data indexed by "sub_num"
    if(isinstance(sub_idx,list) and isinstance(sub_idx[0],str)):
        row_mode=MULTI_NAME_MODE
        series_temp=[]
        for idx in sub_idx:
            assert(isinstance(idx,str))
            series_temp.append(read_rawdata(idx,hyperparams['columns_names'],hyperparams['raw_dataset_path']))
        series=np.concatenate(series_temp,axis=0)
        print("Raw data of subject {:}".format(sub_idx))
        
    # This is for process suntao experimental datasets
    #**** Multiple subject and trial data indexed by "subs, trials, trial"
    if(isinstance(sub_idx,dict)):
        row_mode=MULTI_NAME_TRIAL_MODE
        series_temp=[]
        #-- extract the trials as the first dimension 
        if('assign_trials' in args.keys()):
            for subject_name,trials in sub_idx.items():# sub_idx-> {sub_name:trials, sub_name:trials,...}
                assert(isinstance(trials,list))
                # this output is list contain many three dimension array
                series_temp.append(read_rawdata(trials,hyperparams['columns_names'],hyperparams['raw_dataset_path'],subject_name=subject_name, assign_trials=True))

            series=np.concatenate(series_temp,axis=0)
            #print("Raw data of subject {:}".format(sub_idx))
        else: # - not extract drop landing period
            for subject_name,trials in sub_idx.items():
                assert(isinstance(trials,list))
                series_temp.append(read_rawdata(trials,hyperparams['columns_names'],hyperparams['raw_dataset_path'],subject_name=subject_name))
            series=np.concatenate(series_temp,axis=0)
            #print("Raw data of subject {:}".format(sub_idx))

    

    # load dataset
    #print('Loaded dataset shape:',series.shape)

    #Normalization data
    if (scaler==None) or (scaler=='standard'):
        scaler=StandardScaler()
    if scaler=='minmax':
        scaler=MinMaxScaler()
    if scaler=='robust':
        scaler=RobustScaler()

    dim=series.shape
    reshape_series=series.reshape(-1,dim[-1])
    try:
        scaler.fit(reshape_series)
        scaled_series=scaler.transform(reshape_series.astype(np.float32))
    except Exception as e:
        print(e)
        pdb.set_trace()

    # NOTE: 是否需要三维显示
    # three dimension
    scaled_series=scaled_series.reshape(dim)


    if(row_mode!=MULTI_NAME_TRIAL_MODE):
        # Get the initial stage data of every subjects
        init_stage_sub_ranges={keys:range(all_datasets_ranges['sub_'+str(int(keys[4:])-1)],all_datasets_ranges['sub_'+str(int(keys[4:])-1)]+300)
                      for keys in ['sub_'+str(idx) for idx in range(14)]}
        init_stage_sub_data={keys:read_rawdata(values,hyperparams['columns_names'],hyperparams['raw_dataset_path']) for keys, values in init_stage_sub_ranges.items()}
        # Normalization of init-stage data
        scaled_init_stage_sub_data={keys:scaler.transform(values).mean(axis=0,keepdims=True) 
                                    for keys, values in init_stage_sub_data.items()}
        #empirically_orientation=[-2.32, 2.77,-3.5,  -8.65, 4.0, 3.01, -0.22, -2.11, 5.29, -9.47, -2.85, -1.33]
    return series, scaled_series, scaler

    

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
        sub_idx=np.argwhere(data_len_list_sum > row_idx)[0,0]
        if(sub_idx>0):
            row_idx=row_idx-data_len_list_sum[sub_idx-1]
        

        # --- read datasets
        if(datarange=='one_subject'):
            # read datasets from the h5 file with respect to a specific subect
            for idx, col_idx in enumerate(col_idxs):
                if(idx==0):
                    numpy_datasets=fd['sub_'+str(sub_idx)][:,col_idx]
                else:# stack along with columns
                    numpy_datasets=np.column_stack((numpy_datasets,fd['sub_'+str(sub_idx)][:,col_idx]))
        

        if(datarange=='all_subject'):
            # load h5 file data into a dic: all_datasets
            all_datasets={subject: subject_data[:] for subject, subject_data in fd.items()}
            # read datasets from the h5 file with respect all subects
            for sub_idx in range(subject_num):
                if(sub_idx==0):
                    numpy_datasets=all_datasets['sub_'+str(sub_idx)][:,col_idxs]
                else:# stack along with columns
                    numpy_datasets=np.row_stack((numpy_datasets,all_datasets['sub_'+str(sub_idx)][:,col_idxs]))

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
        


def create_training_files(model_object=None, hyperparams={'lr':0},base_folder=os.path.join(EXPERIMENT_RESULTS_PATH,'models_parameters_results/')):
    '''
    Create folder and sub folder for training, as well as model source code and super parameters

    '''

    # Create top folder based on date
    date_base_folder=base_folder+str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime()))
    if(os.path.exists(date_base_folder)==False):
        os.makedirs(date_base_folder)
    # Create training sub folder
    training_folder=date_base_folder+"/training_"+ str(localtimepkg.strftime("%H%M%S", localtimepkg.localtime()))
    if(os.path.exists(training_folder)==False):
        os.makedirs(training_folder)

    # Create train process sub folder
    training_process_folder=training_folder+"/train_process"
    if(os.path.exists(training_process_folder)==False):
        os.makedirs(training_process_folder)
    # sub folder for loss plots
    training_process_folder_lossplots=training_process_folder+"/lossplots/"
    os.makedirs(training_process_folder_lossplots)

    # Create train results sub folder
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
    hyperparams_file=training_folder+"/hyperparams.yaml"
    with open(hyperparams_file,'w') as fd:
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

    # save model
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



def create_testing_files(training_folder, base_folder=os.path.join(EXPERIMENT_RESULTS_PATH,'models_parameters_results/')):

    # Create top folder based on date
    date_base_folder=base_folder+str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime()))
    if(os.path.exists(date_base_folder)==False):
        os.makedirs(date_base_folder)
    # Create testing sub folder
    training_id=re.search(r"\d+$",training_folder).group()
    testing_folder=date_base_folder+"/test_"+training_id
    if(os.path.exists(testing_folder)==False):
        os.makedirs(testing_folder)

    #Ceate testing sub folder for each test
    test_id=len(os.listdir(testing_folder))+1
    each_testing_folder=testing_folder+"/test_"+str(test_id)
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



def display_rawdatase(datasets_ranges,col_names,norm_type='mean_std',**args):
    '''
    Params: datasets_ranges: a two dimension numpy or a range 

    '''
    #0) read datasets
    print(isinstance(datasets_ranges,np.ndarray))

    #-- input datasets
    if(isinstance(datasets_ranges, np.ndarray)):# load dataset from a numpy array
        datasets=copy.deepcopy(datasets_ranges)
    else:#-- load dataset from a h5 file 
        if('raw_datasets_path' in args.keys()):
            raw_datasets_path=args['raw_datasets_path']
        else:
            raw_datasets_path="./datasets_files/raw_datasets.hdf5"
        datasets=read_rawdata(range(datasets_ranges[0],datasets_ranges[1]),col_names,raw_datasets_path)
    
    #1) data process
    if datasets.ndim>=3:# 如果是三维，说明有多个trials, 将他们按行合并
        datasets=datasets.reshape(-1,datasets.shape[-1])
    #-- normalize datasets if speified
    if(norm_type!=None):
        datasets_norm=norm_datasets(datasets,col_names,norm_type)
        pd_datasets=pd.DataFrame(data=datasets_norm,columns=col_names)
        print('plot normalized raw datasets')
    else:
        pd_datasets=pd.DataFrame(data=datasets,columns=col_names)
        print('plot raw datasets without normalization')

    #2) plots
    figsize=(14,16)
    fig=plt.figure(figsize=figsize)
    plot_method='seaborn'
    if(plot_method=='matplotlib'):#绘制的方法
        display_rows=args['display_rows']#绘制的行数
        display_cols=args['display_cols']#绘制的列数
        gs1=gridspec.GridSpec(2*len(display_rows),len(display_cols))#13
        gs1.update(hspace=0.1,wspace=0.15,top=0.95,bottom=0.05,left=0.04,right=0.98)
        axs=[]
        for plot_col in range(2):
            for plot_row in range(len(display_rows)):
                axs.append(fig.add_subplot(gs1[2*plot_row:2*plot_row+2,plot_col]))

        print(pd_datasets.shape)
        axs=np.array(axs).reshape(2,-1).T
        freq=100.0;
        Time=np.linspace(0,pd_datasets.shape[0]/freq,num=pd_datasets.shape[0])
        for plot_col in range(2):# Left and ight
            axs[0,plot_col].set_title(args['plot_title'])
            if plot_col==0:
                prefix="L_"
            else:
                prefix="R_"
            for plot_idx, plot_row in enumerate(display_rows):
                axs[plot_idx,plot_col].plot(Time,pd_datasets[prefix+plot_row])
                axs[plot_idx,plot_col].legend([prefix+plot_row])

                #axs[plot_idx,plot_col].plot(Time,pd_datasets[col_names[3*plot_idx+plot_col*len(FEATURES_FIELDS)+1]])
                #axs[plot_idx,plot_col].plot(Time,pd_datasets[col_names[3*plot_idx+plot_col*len(FEATURES_FIELDS)+2]])
                #axs[plot_idx,plot_col].plot(Time,pd_datasets[col_names[3*plot_idx+plot_col*len(FEATURES_FIELDS)+3]])
                #axs[plot_idx,plot_col].legend(col_names[3*plot_idx+plot_col*len(FEATURES_FIELDS)+1:3*plot_idx+plot_col*len(FEATURES_FIELDS)+4],ncol=3)
                axs[plot_idx,plot_col].grid(which='both',axis='x',color='k',linestyle=':')
                axs[plot_idx,plot_col].grid(which='both',axis='y',color='k',linestyle=':')
                #axs[plot_idx].set_ylabel()
                axs[plot_idx,plot_col].set_xticklabels([])
            # plot targets
            #for plot_idx in range(6,9):
            #    axs[plot_idx,plot_col].plot(Time,pd_datasets[col_names[plot_idx-(12-3*plot_idx)]])
            #    axs[plot_idx,plot_col].legend([col_names[plot_idx-(12-plot_col*3)]])
            #    axs[plot_idx,plot_col].grid(which='both',axis='x',color='k',linestyle=':')
            #    axs[plot_idx,plot_col].grid(which='both',axis='y',color='k',linestyle=':')
            #    axs[plot_idx,plot_col].set_xticklabels(labels=[])
            
            xticks=axs[plot_idx,plot_col].get_xticks()
            #axs[plot_idx,plot_col].set_ylim((-10,120))
            #axs[plot_idx,plot_col].set_xticklabels([str(round(tt,1)) for tt in xticks])
            axs[plot_idx,plot_col].set_xlabel("Time [s]")

    if(plot_method=='seaborn'):
        pd_datasets['time']=np.linspace(0,DROPLANDING_PERIOD/100,DROPLANDING_PERIOD)
        reshape_pd_datasets=pd_datasets.melt(id_vars=['time'],var_name='cols',value_name='vals')
        #sns.lineplot(data=reshape_pd_datasets,x='time',y='vals',hue='cols')
        g=sns.FacetGrid(reshape_pd_datasets,col='cols',col_wrap=4,height=2)
        g.map_dataframe(sns.lineplot,'time','vals')


    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+".svg")
    plt.savefig(datasets_visulization_path)
    plt.show()
    #data_mean, data_std = normalization_parameters(200,features_names)    
    #print(data_std)



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
    plt.savefig(os.path.joint(EXPERIMENT_RESULTS_PATH,'models_parameters_results/split_droplanding.svg'))
    return (start_drop,end_drop)



#sub_idx=2
#start_drop, end_drop=extract_subject_drop_landing_data(sub_idx)

def drop_landing_range():

    # subject 0
    Up_sub0  = [1012, 1804,2594,3419,4157,4933,5695,6460]
    Down_sub0= [1173,1974,2755,3565,4306,5100,5863,6625]

    #subject 1
    Up_sub1   =[1268,2148,2970,3731,4453,5311,6312,6903]
    Down_sub1 =[1374,2355,3130,3904,4628,5473,6476,7090]







def plot_statistic_kneemoment_under_fpa(data: list, col_names:list, display_name, subjects: list, categories,plot_type='catbox'):

    phi={}
    for cat_idx, category in enumerate(categories):# trial types
        phi[category]={}
        for sub_idx, subject in enumerate(subjects):# subjects
            phi[category][subject]=[]
            one_subject_data=data[sub_idx]
            for idx in range(5*cat_idx,5*(cat_idx+1)):# trials
                pd_temp_data= pd.DataFrame(data=one_subject_data[idx,:,:],columns=col_names)
                temp_left=pd_temp_data[display_name[0]]
                temp_right=pd_temp_data[display_name[1]]
                phi[category][subject].append(max([max(temp_right),max(temp_left)]))
                print("The trial:{} of subject:{} in session:{} has max value: {}".format(idx, subject,category,phi[category][subject][-1]))

    
    #2) plot
    figsize=(8,4)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.16,left=0.12,right=0.89)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    FPA=[ str(ll) for ll in categories]
    print(FPA)
    ind= np.arange(len(categories))


    #3.1) plot 
    phi_values=[]
    pd_phi_values_list=[]
    subject_names=list(phi[categories[0]].keys())
    for idx,subject_name in enumerate(subject_names):
        phi_values.append([])
        for category in categories:
            phi_values[idx].append(phi[category][subject_name])
            temp=pd.DataFrame(data=phi[category][subject_name],columns=["values"])
            temp.insert(1,'categories',category)
            temp.insert(2,'subject_names',subject_name)
            pd_phi_values_list.append(temp)
    
    pd_phi_values=pd.concat(pd_phi_values_list)


    if(plot_type=='catbox'):
        idx=0
        boxwidth=0.05
        box=[]
        for box_idx in range(len(subject_names)):
            box.append(axs[idx].boxplot(phi_values[box_idx],widths=boxwidth, positions=ind+(box_idx-int(len(subject_names)/2))*boxwidth ,vert=True,patch_artist=True,meanline=True,showmeans=True,showfliers=False)) 
           # fill with colors
        colors = ['lightblue', 'lightgreen','wheat']
        import matplotlib._color_data as mcd
        overlap = {name for name in mcd.CSS4_COLORS if "xkcd:" + name in mcd.XKCD_COLORS}
        colors=[mcd.CSS4_COLORS[color_name] for color_name in overlap]

        for bplot, color in zip(box,colors[0:len(box)]):
            for patch in bplot['boxes']:
                patch.set_facecolor(color)

        axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
        axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
        #axs[idx].set_yticks([0,0.1,0.2,0.3])
        #axs[idx].set(ylim=[-0.01,0.3])
        legend_names=[name for name in subject_names]
        axs[idx].legend([bx['boxes'][0] for bx in box],legend_names[0:len(box)],ncol=4)
        axs[idx].set_xticks(ind)
        axs[idx].set_xticklabels(FPA)
        axs[idx].set_ylabel(r'Knee Moment [weight*NM]')
        axs[idx].set_xlabel(r'FPA')
    else:
        idx=0
        sns.set_theme(style='whitegrid')
        pdb.set_trace()
        g=sns.catplot(x='categories',y='values',hue='subject_names',data=pd_phi_values,kind='point')
        #g=sns.catplot(x='categories',y='values',data=pd_phi_values,kind='point')


        
        g=sns.FacetGrid(pd_phi_values,col='subject_names',col_wrap=4,sharex=False,sharey=False)
        g.map(sgs.catplot,'categories','values')

        g.ax.grid(which='both',axis='x',color='k',linestyle=':')
        g.ax.grid(which='both',axis='y',color='k',linestyle=':')
        #axs[idx].set_yticks([0,0.1,0.2,0.3])
        #axs[idx].set(ylim=[-0.01,0.3])
        legend_names=[name for name in subject_names]
        #axs[idx].legend([bx['boxes'][0] for bx in box],legend_names[0:len(box)],ncol=4)
        g.ax.set_xticks(ind)
        g.ax.set_xticklabels(FPA)
        g.ax.set_ylabel(r'Peak knee moment [BW.BH], display_name')
        g.ax.set_xlabel(r'FPA')




    datasets_visulization_path=os.path.join(DATA_VISULIZATION_PATH,str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime()))+".svg")
    plt.savefig(datasets_visulization_path)
    plt.show()



def plot_statistic_value_under_fpa(data: list, col_names:list, display_name, subjects: list, categories,plot_type='catbox'):
    '''
    Description: Plot peak values of various biomechanic variables for different trials, subjects under various foot progression angles (FPA)
    Parameters: data, a numpy array with three dimensions

    '''

    biomechanic_variables={}
    static_calibration_value={}# the variable values in static phase in baseline trial
    for sub_idx, subject in enumerate(subjects):# subjects
        biomechanic_variables[subject]={}
        static_calibration_value[subject]={}# the variable values in static phase in baseline trial
        for cat_idx, category in enumerate(categories):# trial types
            biomechanic_variables[subject][category]=[]
            static_calibration_value[subject][category]=[]# the variable values in static phase in baseline trial
            one_subject_data=data[sub_idx]
            for idx in range(5*cat_idx,5*(cat_idx+1)):# trials number
                pd_temp_data= pd.DataFrame(data=one_subject_data[idx,:,:],columns=col_names)
                peak_temp = {}
                static_temp = {}
                touch_moment_index=int(DROPLANDING_PERIOD/4) #- check wearable_toolkit, line 173, which define the formula of tocuh_moment 
                for display in display_name: # biomechanic variables 
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
        for category in categories:
            temp=pd.DataFrame(data=biomechanic_variables[subject][category])
            pd_biomechanic_variables_list.append(temp)
    
    pd_biomechanic_variables=pd.concat(pd_biomechanic_variables_list)
    pd_biomechanic_variables.reset_index(inplace=True)# index from 0 to few hundreds



    # 2.1  Tansfer the static dict data into pandas dataframe
    pd_static_value_list=[]
    for idx,subject in enumerate(subjects):
        for category in categories:
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



        pdb.set_trace()
    
    #2) plot
    
    # paraemter setup for plot
    test_method = 'Mann-Whitney'
    figwidth=12.3;figheight=12.4
    subplot_left=0.08; subplot_right=0.97; subplot_top=0.95;subplot_bottom=0.06
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
    plt.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
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





#- set hyperparaams: sub_idx. subject_name: trial
def setHyperparams_subject(hyperparams,subject_list=None):
    if(subject_list==None):
        subjects_list=['P_08','P_09','P_10', 'P_11', 'P_13', 'P_14', 'P_15','P_16','P_17','P_18','P_19','P_20','P_21','P_22','P_23', 'P_24']

    subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0)
    subject_names=[ss for ss in subject_infos.index]
    for subject_idx in subjects_list:
        for subject_name in subject_names:
            if(re.search(subject_idx,subject_name)!=None): # checking the sub_idx_id is in subject_infos
                subject_idx_name=subject_name
                break
        hyperparams['sub_idx'][subject_idx_name]=TRIALS

    return hyperparams


'''
# basic parameters
all_datasets_len={'sub_0':6951, 'sub_1':7439, 'sub_2': 7686, 'sub_3': 8678, 'sub_4':6180, 'sub_5': 6671,
                  'sub_6': 7600, 'sub_7': 5583, 'sub_8': 6032, 'sub_9': 6508, 'sub_10': 6348, 'sub_11': 7010, 'sub_12': 8049, 'sub_13': 6248}
all_datasets_ranges={'sub_-1':0,'sub_0': 6951, 'sub_1': 14390, 'sub_2': 22076, 'sub_3': 30754, 'sub_4': 36934, 'sub_5': 43605,
                     'sub_6': 51205, 'sub_7': 56788, 'sub_8': 62820, 'sub_9': 69328, 'sub_10': 75676, 'sub_11':82686, 'sub_12': 90735, 'sub_13': 96983}


#'device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
hyperparams={
        'norm_type': "mean_std",
        'batch_size': 64,
        'epochs': 120,
        'window_size': 10,
        'cost_threashold': 0.001,
        'learning_rate': 0.015,
        #'device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
}

'''


if __name__=='__main__':
    multi_subject_data=[]

    #-- list subject names
    subjects_list=['P_08','P_09','P_10', 'P_11', 'P_13', 'P_14', 'P_15','P_16','P_17','P_18','P_19','P_20','P_21','P_22','P_23', 'P_24']
    subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0)
    subject_names_column=[ss for ss in subject_infos.index]

    #-- define subject names and load subject dataset
    subject_names=[]
    for subject_idx in subjects_list:
        for subject_name in subject_names_column:
            if(re.search(subject_idx,subject_name)!=None):
                subject_idx_name=subject_name
                break
        print(subject_idx_name)
        subject_names.append(subject_idx_name)
        #-- define hyperparams values: subject, columns_names
        hyperparams['sub_idx']={subject_idx_name:TRIALS}
        #hyperparams['columns_names']=['L_KNEE_MOMENT_X','L_KNEE_MOMENT_Y','R_KNEE_MOMENT_X','R_KNEE_MOMENT_Y','L_FPA_Z','R_FPA_Z']
        hyperparams['columns_names']=['L_FPA_Z','R_FPA_Z','L_FPA_X','R_FPA_X',
                            'L_GRF_X', 'R_GRF_X', 'L_GRF_Y', 'R_GRF_Y', 'L_GRF_Z', 'R_GRF_Z',
                            'L_ANKLE_ANGLE_X','R_ANKLE_ANGLE_X', 'L_ANKLE_ANGLE_Y','R_ANKLE_ANGLE_Y', 'L_ANKLE_ANGLE_Z','R_ANKLE_ANGLE_Z',
                            'L_ANKLE_MOMENT_X','R_ANKLE_MOMENT_X', 'L_ANKLE_MOMENT_Y','R_ANKLE_MOMENT_Y', 'L_ANKLE_MOMENT_Z','R_ANKLE_MOMENT_Z',
                            'L_KNEE_ANGLE_X','R_KNEE_ANGLE_X', 'L_KNEE_ANGLE_Y','R_KNEE_ANGLE_Y', 'L_KNEE_ANGLE_Z','R_KNEE_ANGLE_Z',
                            'L_KNEE_MOMENT_X','R_KNEE_MOMENT_X', 'L_KNEE_MOMENT_Y','R_KNEE_MOMENT_Y', 'L_KNEE_MOMENT_Z','R_KNEE_MOMENT_Z',
                            'PELVIS_ANGLE_X','THORAX_ANGLE_X'
                           ]
        #-- load multiple subject data, the output series columns with indicated sequences
        series, scaled_series,scaler=load_normalize_data(sub_idx={subject_idx_name:TRIALS},scaler='minmax',hyperparams=hyperparams,assign_trials=True)
        multi_subject_data.append(series)
    
    #-- subject height
    subject_heights=[float(subject_infos['Unnamed: 3'][sub_name]) for sub_name in subject_names]
    #-- subject mass
    subject_masses=[float(subject_infos['Unnamed: 2'][sub_name]) for sub_name in subject_names]


    #-- plot statistic knee moment under various fpa
    categories=['fpa_01','fpa_02','fpa_03','fpa_04','fap_05']
    display_names=['L_KNEE_MOMENT_Y','R_KNEE_MOMENT_Y']
    #plot_statistic_kneemoment_under_fpa(multi_subject_data,hyperparams['columns_names'],display_names,subjects_list,categories,plot_type="s")



    #-- display time-based curves of the dataset
    #display_rawdatase(series[0,:,:], hyperparams['columns_names'], norm_type=None)


    #-- display statistic peak value under various fpa
    categories = ['baseline','fpa_01','fpa_02','fpa_03','fpa_04','fap_05']
    display_bio_variables = ['L_FPA_Z','R_FPA_Z',
                            'L_GRF_X', 'R_GRF_X', 'L_GRF_Y', 'R_GRF_Y', 'L_GRF_Z', 'R_GRF_Z',
                            'L_ANKLE_ANGLE_X','R_ANKLE_ANGLE_X', 'L_ANKLE_ANGLE_Y','R_ANKLE_ANGLE_Y', 'L_ANKLE_ANGLE_Z','R_ANKLE_ANGLE_Z',
                            'L_ANKLE_MOMENT_X','R_ANKLE_MOMENT_X', 'L_ANKLE_MOMENT_Y','R_ANKLE_MOMENT_Y', 'L_ANKLE_MOMENT_Z','R_ANKLE_MOMENT_Z',
                            'L_KNEE_ANGLE_X','R_KNEE_ANGLE_X', 'L_KNEE_ANGLE_Y','R_KNEE_ANGLE_Y', 'L_KNEE_ANGLE_Z','R_KNEE_ANGLE_Z',
                            'L_KNEE_MOMENT_X','R_KNEE_MOMENT_X', 'L_KNEE_MOMENT_Y','R_KNEE_MOMENT_Y', 'L_KNEE_MOMENT_Z','R_KNEE_MOMENT_Z',
                           ]
    display_bio_variables = ['L_FPA_Z','R_FPA_Z', 'L_FPA_X','R_FPA_X',  'L_KNEE_MOMENT_X','R_KNEE_MOMENT_X','R_KNEE_MOMENT_Y',  'R_KNEE_MOMENT_Z','L_KNEE_MOMENT_Z',  'L_KNEE_MOMENT_Y','L_KNEE_ANGLE_Y','R_KNEE_ANGLE_Y',  'L_KNEE_ANGLE_X','R_KNEE_ANGLE_X',   'L_KNEE_ANGLE_Z','R_KNEE_ANGLE_Z' ,'PELVIS_ANGLE_X','THORAX_ANGLE_X']


    plot_statistic_value_under_fpa(multi_subject_data, hyperparams['columns_names'], display_bio_variables, subjects_list, categories, plot_type="s")

    #dp_lib.display_rawdatase(scaled_series[6000:6250,:], columns_names, norm_type=None, raw_datasets_path=raw_dataset_path,plot_title='sub_'+str(sub_idx))
    #dp_lib.display_rawdatase(scaler.inverse_transform(scaled_series[6000:6250,:]), columns_names, norm_type=None, raw_datasets_path=raw_dataset_path)
    #dp_lib.display_rawdatase(datasets_ranges, columns_names, norm_type='mean_std', raw_datasets_path=raw_dataset_path,plot_title='raw data')
    #display_rawdatase(datasets_ranges, columns_names, norm_type='mean_std', raw_datasets_path=raw_dataset_path,plot_title='raw data')
    

