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

if __name__=='__main__':
    from const import FEATURES_FIELDS, LABELS_FIELDS, V3D_LABELS_FIELDS, DATA_PATH, TRIALS, DATA_VISULIZATION_PATH, DROPLANDING_PERIOD, EXPERIMENT_RESULTS_PATH
else:
    from package_lib.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH, TRIALS, DATA_VISULIZATION_PATH, DROPLANDING_PERIOD,V3D_LABELS_FIELDS,EXPERIMENT_RESULTS_PATH


from sklearn.preprocessing import StandardScaler

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
    with h5py.File(raw_datasets_path, 'r') as fd:
        # 1) The coloms of the features and labels
        subject_names=list(fd.keys())
        all_data_fields=fd[subject_names[0]].attrs.get('columns')
        col_idxs=[]

        # suntao drop landing experiment data
        if(isinstance(row_idx,list) and isinstance(row_idx[0],str)):
            all_datasets_list=[]
            trials=row_idx
            subject_name=args['subject_name']
            if(subject_name not in subject_names):
                print("This subject:{subject_name} is not in datasets".format(subject_name))
                exit()
            feature_data_fields=[hyperparams['features_names']]
            for trial in trials:
                try:
                    temp_pd_data=pd.DataFrame(data=np.array(fd[subject_name][trial]),columns=fd[subject_name].attrs['columns'])
                except Exception as e:
                    print(e)
                #-- read the specified columns
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
            for key,value in sub_idx.items():# sub_idx-> {sub_name:trials, sub_name:trials,...}
                subject_name=key
                assert(isinstance(value,list))
                # this output is list contain many three dimension array
                series_temp.append(read_rawdata(value,hyperparams['columns_names'],hyperparams['raw_dataset_path'],subject_name=subject_name, assign_trials=True))

            series=np.concatenate(series_temp,axis=0)
            print("Raw data of subject {:}".format(sub_idx))
        else: # - not extract drop landing period
            for key,value in sub_idx.items():
                subject_name=key
                assert(isinstance(value,list))
                series_temp.append(read_rawdata(value,hyperparams['columns_names'],hyperparams['raw_dataset_path'],subject_name=subject_name))
            series=np.concatenate(series_temp,axis=0)
            print("Raw data of subject {:}".format(sub_idx))

    

    # load dataset
    print('Loaded dataset shape:',series.shape)

    #Normalization data
    if scaler==None:
        scaler=StandardScaler()
    

    dim=series.shape
    reshape_series=series.reshape(-1,dim[-1])
    try:
        scaler.fit(reshape_series)
        scaled_series=scaler.transform(reshape_series.astype(np.float32),copy=True)
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






def plot_test_results(features,labels,predictions,features_names,labels_names,fig_save_folder=None,**args):
    """
    Plot the test results

    """
    print("Plot the test results")
    
    #------------Process the test results
    # features
    #features=np.squeeze(np.array(features))
    # lablels
    #labels=np.squeeze(np.array(labels))
    # predictions
    #predictions=np.squeeze(np.array(predictions))
    
    # Pandas DataFrame of the above datasets
    pd_features = pd.DataFrame(features,columns=features_names)
    pd_labels = pd.DataFrame(labels,columns=labels_names)
    pd_predictions = pd.DataFrame(data=predictions,columns=labels_names)

    freq=100.0;
    Time=np.linspace(0,labels.shape[0]/freq,num=labels.shape[0])


    # ---  Plots ------
    num_pred=predictions.shape[1]
    if(num_pred>3):
        subplots_rows=num_pred//2
    else:
        subplots_rows=num_pred

    figsize=(12,3*subplots_rows)
    fig=plt.figure(figsize=figsize)
    axs=fig.subplots(subplots_rows,2).reshape(-1,2)
    # -- plot labels and predictions
    for plot_idx, label_name in enumerate(labels_names):
        axs[plot_idx%3,plot_idx//3].plot(Time,pd_labels[label_name],'g')
        axs[plot_idx%3,plot_idx//3].plot(Time,pd_predictions[label_name],'r')
        axs[plot_idx%3,plot_idx//3].legend(['Measured value','Estimated value'])
        axs[plot_idx%3,plot_idx//3].set_ylabel(label_name+" [*]")
        axs[plot_idx%3,plot_idx//3].grid(which='both',axis='x',color='k',linestyle=':')
        axs[plot_idx%3,plot_idx//3].grid(which='both',axis='y',color='k',linestyle=':')
    axs[plot_idx%3,plot_idx//3].set_xlabel("Time [s]")
    # save figure
    if(fig_save_folder!=None):
        folder_fig = fig_save_folder+"/"
    else:
        folder_fig="./"
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    # whether define the figsave_file
    if('prediction_file' in args.keys()):
        figPath=args['prediction_file']
    else:
        figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + '_test_prediction.svg'
    plt.savefig(figPath)


    # statisrical estimation errors
    pred_error=predictions-labels
    pred_mean=np.mean(pred_error,axis=0)
    pred_std=np.std(pred_error,axis=0)
    pred_rmse=np.sqrt(np.sum(np.power(pred_error,2),axis=0)/pred_error.shape[0])
    print("mean: {:.2f}, std: {:.2f}, rsme: {:.2f} of the errors between estimation and ground truth", pred_mean, pred_std, pred_rmse)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


    #plot absolute error and noramlized error (error-percentage)
    # error= labels-prediction, along the time, each column indicates a labels
    error=np.subtract(pd_labels.values,pd_predictions.values)
    AE = np.abs(error)
    figsize=(12,3*subplots_rows)
    fig=plt.figure(figsize=figsize)
    axs=fig.subplots(subplots_rows,2).reshape(-1,2)
    #plot_idx=0
    #axs[plot_idx].
    colors=['r','g','b','y']*5
    for plot_idx in range(AE.shape[1]):
        axs[plot_idx%3,plot_idx//3].plot(Time,AE[:,plot_idx],colors[plot_idx])
        axs[plot_idx%3,plot_idx//3].plot(Time,AE[:,plot_idx]/np.abs(pd_labels.values[:,plot_idx]),colors[plot_idx+1])
        axs[plot_idx%3,plot_idx//3].legend(['Absolute '+labels_names[plot_idx], 'Relative ' +labels_names[plot_idx]  ] )
        axs[plot_idx%3,plot_idx//3].set_ylabel("Absolute and relative error [*]")
        axs[plot_idx%3,plot_idx//3].grid(which='both',axis='x',color='k',linestyle=':')
        axs[plot_idx%3,plot_idx//3].grid(which='both',axis='y',color='k',linestyle=':')
        # place a text box in upper left in axes coords
        #textstr = '\n'.join((r'$mean \pm std=%.2f \pm %.2f$' % (pred_mean[plot_idx], pred_std[plot_idx],),r'$RMSE=%.2f$' % (pred_rmse[plot_idx], )))
        #axs[plot_idx%3,plot_idx//3].text(0.05, 0.95, textstr, transform=axs[plot_idx%3,plot_idx//3].transAxes, fontsize=14,verticalalignment='top', bbox=props)
    
    axs[plot_idx%3,plot_idx//3].set_xlabel('Time [s]')
    # save figure
    if(fig_save_folder!=None):
        folder_fig = fig_save_folder+"/"
    else:
        folder_fig="./"
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    # figure save file
    if('prediction_error_file' in args.keys()):
        figPath=args['prediction_error_file']
    else:
        figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + '_test_mes.svg'
    plt.savefig(figPath)




    plt.show()
    """
    print("Actual labels:\n",pd_labels.head())
    print("Predictions:\n",pd_predictions.head())
    print("Error:\n",R_IE_error.head())

    """

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

    # input datasets
    if(isinstance(datasets_ranges, np.ndarray)):# load dataset
        datasets=copy.deepcopy(datasets_ranges)
    else:# load data from path
        if('raw_datasets_path' in args.keys()):
            raw_datasets_path=args['raw_datasets_path']
        else:
            raw_datasets_path="./datasets_files/raw_datasets.hdf5"
        datasets=read_rawdata(range(datasets_ranges[0],datasets_ranges[1]),col_names,raw_datasets_path)
    
    #1) data process
    if datasets.ndim>=3:# 如果是三维，说明有多个trials, 将他们按行合并
        datasets=datasets.reshape(-1,datasets.shape[-1])
    # normalize datasets
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
    display_rows=args['display_rows']#绘制的行数
    display_cols=args['display_cols']

    plot_method='seaborn'
    if(plot_method=='matplotlib'):
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
        g=sns.FacetGrid(pd_datasets)
        g.map_dataframe(x='',y='',)
    #plt.show()
    #reshape_pd_datasets=pd_datasets.melt('Time_vicon', var_name='cols',value_name='vals')
    #reshape_pd_datasets.head()
    #sns.lineplot(data=reshape_pd_datasets,x='Time_vicon', y='vals',hue='cols')
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







def plot_statistic_kneemoment_under_fpa(data: list, col_names:list, display_name, subjects: list, subject_heights,plot_type='catbox'):

    phi={}
    experiment_categories=['baseline','fpa_01','fpa_02','fpa_03','fpa_04','fap_05','single']
    for cat_idx, category in enumerate(experiment_categories):# trial types
        phi[category]={}
        for sub_idx, subject in enumerate(subjects):# subjects
            phi[category][subject]=[]
            one_subject_data=data[sub_idx]
            for idx in range(5*cat_idx,5*(cat_idx+1)):# trials
                pd_temp_data= pd.DataFrame(data=one_subject_data[idx,:,:],columns=col_names)
                temp_left=pd_temp_data[display_name[0]]
                temp_right=pd_temp_data[display_name[1]]
                phi[category][subject].append(max([max(temp_right),max(temp_left)])/subject_heights[sub_idx])
                print("The trial:{} of subject:{} in session:{} has max value: {}".format(idx, subject,category,phi[category][subject][-1]))

    
    pdb.set_trace()
    #2) plot
    figsize=(8,4)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(6,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.16,left=0.12,right=0.89)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:6,0]))

    FPA=[ str(ll) for ll in experiment_categories]
    print(FPA)
    ind= np.arange(len(experiment_categories))


    #3.1) plot 
    phi_values=[]
    pd_phi_values_list=[]
    subject_names=list(phi[experiment_categories[0]].keys())
    for idx,subject_name in enumerate(subject_names):
        phi_values.append([])
        for category in experiment_categories:
            phi_values[idx].append(phi[category][subject_name])
            temp=pd.DataFrame(data=phi[category][subject_name],columns=["values"])
            temp.insert(1,'experiment_categories',category)
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
        g=sns.catplot(x='experiment_categories',y='values',hue='subject_names',data=pd_phi_values,kind='point')
        #g=sns.catplot(x='experiment_categories',y='values',data=pd_phi_values,kind='point')

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


#- set hyperparaams: sub_idx. subject_name: trial
def setHyperparams_subject(hyperparaams):
    subjects_list=['P_08','P_09','P_10', 'P_11', 'P_13', 'P_14', 'P_15','P_16','P_17','P_18','P_19','P_20','P_21','P_22','P_23', 'P_24']
    subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0)
    subject_names=[ss for ss in subject_infos.index]
    for subject_idx in subjects_list:
        for subject_name in subject_names:
            if(re.search(subject_idx,subject_name)!=None):
                subject_idx_name=subject_name
                break
        print(subject_idx_name)
        hyperparaams['sub_idx'][subject_idx_name]=TRIALS



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

# h5 dataset path
raw_dataset_path=os.path.join(DATA_PATH,'features_labels_rawdatasets.hdf5')
hyperparams['raw_dataset_path']=raw_dataset_path
hyperparams['sub_idx']={}
hyperparams['features_names']=FEATURES_FIELDS;
hyperparams['labels_names']=LABELS_FIELDS
hyperparams['features_num']=len(FEATURES_FIELDS)
hyperparams['labels_num']=len(LABELS_FIELDS)
hyperparams['columns_names']=hyperparams['features_names']+hyperparams['labels_names']
setHyperparams_subject(hyperparams)




if __name__=='__main__':
    multi_subject_data=[]

    subjects_list=['P_08','P_09','P_10', 'P_11', 'P_13', 'P_14', 'P_15','P_16','P_17','P_18','P_19','P_20','P_21','P_22','P_23', 'P_24']
    subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0)
    subject_names_column=[ss for ss in subject_infos.index]

    
    subject_names=[]
    for subject_idx in subjects_list:
        for subject_name in subject_names_column:
            if(re.search(subject_idx,subject_name)!=None):
                subject_idx_name=subject_name
                break
        print(subject_idx_name)
        subject_names.append(subject_idx_name)
        hyperparams['sub_idx']={subject_idx_name:TRIALS}
        hyperparams['columns_names']=['L_KneeMoment_X','L_KneeMoment_Y','R_KneeMoment_X','R_KneeMoment_Y']
        series, scaled_series,scaler=load_normalize_data(sub_idx={subject_idx_name:TRIALS},hyperparams=hyperparams,assign_trials=True)
        multi_subject_data.append(series)

    subject_heights=[float(subject_infos['Unnamed: 3'][sub_name]) for sub_name in subject_names]
    plot_statistic_kneemoment_under_fpa(multi_subject_data,hyperparams['columns_names'],['L_KneeMoment_Y','R_KneeMoment_Y'],subjects_list,subject_heights,  plot_type="s")

    # display datasets
    #display_rows=['KneeAngle_X', 'KneeAngle_Y','KneeAngle_Z','KneeForce_X','KneeForce_Y', 'KneeForce_Z', 'KneeMoment_X','KneeMoment_Y','KneeMoment_Z','Force_X','Force_Y','Force_Z']
    #display_rows=['KneeMoment_X', 'KneeMoment_Y','KneeMoment_Z','Cop_X','Cop_Y','Cop_Z']
    #display_rows=display_rows
    #display_rows=['Up_Acc_X', 'Up_Acc_Y','Up_Acc_Z','Up_Gyr_X','Up_Gyr_Y','Up_Gyr_Z','Lower_Acc_X','Lower_Acc_Y','Lower_Acc_Z', 'Lower_Gyr_X','Lower_Gyr_Y','Lower_Gyr_Z']
    #display_rows=['SHANK_Accel_X']
    display_rows=['KneeMoment_X','KneeMoment_Y']
    #datasets_ranges=(dp_lib.all_datasets_ranges['sub_'+str(sub_idx-1)],dp_lib.all_datasets_ranges['sub_'+str(sub_idx)])
    #display_rawdatase(series[0,0:150,:], columns_names, norm_type=None, raw_datasets_path=raw_dataset_path,plot_title=sub_name, display_rows=display_rows,display_cols=[''])
    #dp_lib.display_rawdatase(scaled_series[6000:6250,:], columns_names, norm_type=None, raw_datasets_path=raw_dataset_path,plot_title='sub_'+str(sub_idx))
    #dp_lib.display_rawdatase(scaler.inverse_transform(scaled_series[6000:6250,:]), columns_names, norm_type=None, raw_datasets_path=raw_dataset_path)
    #dp_lib.display_rawdatase(datasets_ranges, columns_names, norm_type='mean_std', raw_datasets_path=raw_dataset_path,plot_title='raw data')
    #display_rawdatase(datasets_ranges, columns_names, norm_type='mean_std', raw_datasets_path=raw_dataset_path,plot_title='raw data')
    

