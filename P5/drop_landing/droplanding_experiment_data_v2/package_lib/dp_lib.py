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

import datetime

def read_rawdata(row_idx: int,col_names: list,raw_datasets_path="./datasets_files/raw_datasets.hdf5")-> numpy.ndarray:
    """
    @Description:
    To read the data from h5 file and normalize the features and labels.
    @Parameters:
    Row_idx: the index of row. data type is int
    Col_names: the names of columns. data type is string
    
    """
    assert(type(col_names)==list)
    with h5py.File(raw_datasets_path, 'r') as fd:
        # 1) The coloms of the features and labels
        keys=list(fd.keys())
        columns=fd[keys[0]].attrs.get('columns')
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
        if(type(row_idx)==int):
            # 4) subject idx and internal row_idx
            sub_idx=np.argwhere(data_len_list_sum > row_idx)[0,0]
            if(sub_idx>0):
                row_idx=row_idx-data_len_list_sum[sub_idx-1]
            return fd['sub_'+str(sub_idx)][row_idx,col_idxs]

        if(isinstance(row_idx,list) or isinstance(row_idx,range)): #-- return data of multiple rows
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
        if(isinstance(row_idx,str)): # return data indexed by subject id
            sub_idx=row_idx
            assert(sub_idx in ['sub_'+str(ii) for ii in range(15)])
            # 5) load h5 file data into a dic: all_datasets
            all_datasets={subject: subject_data[:] for subject, subject_data in fd.items()}
            # 6) return datasets of multiple rows
            return all_datasets[sub_idx][:,col_idxs]
    
    

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
        


def create_training_files(model_object=None, hyperparams={'lr':0},base_folder='./models_parameters_results/'):
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
        pdb.set_trace()
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



def create_testing_files(training_folder, base_folder='./models_parameters_results/'):

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
    figsize=(12,3*subplots_rows)
    fig=plt.figure(figsize=figsize)
    axs=fig.subplots(subplots_rows,2).reshape(-1,2)
    # plot labels and predictions
    for plot_idx, label_name in enumerate(labels_names):
        axs[plot_idx%3,plot_idx//3].plot(Time,pd_labels[label_name],'g')
        axs[plot_idx%3,plot_idx//3].plot(Time,pd_predictions[label_name],'r')
        axs[plot_idx%3,plot_idx//3].legend(['Measured values','Estimated value'])
        axs[plot_idx%3,plot_idx//3].set_ylabel(label_name+" [deg]")
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
    print("mean: {:.2f}, std: {:.2f}, rsme: {:.2f}", pred_mean, pred_std, pred_rmse)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


    #plot absolute error
    # error= labels-prediction, along the time, each column indicates a labels
    error=np.subtract(pd_labels.values,pd_predictions.values)
    AE = np.abs(error)
    figsize=(12,3*subplots_rows)
    fig=plt.figure(figsize=figsize)
    axs=fig.subplots(subplots_rows,2).reshape(-1,2)
    #plot_idx=0
    #axs[plot_idx].
    colors=['b','b']*5
    for plot_idx in range(AE.shape[1]):
        axs[plot_idx%3,plot_idx//3].plot(Time,AE[:,plot_idx],colors[plot_idx])
        axs[plot_idx%3,plot_idx//3].legend([labels_names[plot_idx]])
        axs[plot_idx%3,plot_idx//3].set_ylabel("Absolute error [deg]")
        axs[plot_idx%3,plot_idx//3].grid(which='both',axis='x',color='k',linestyle=':')
        axs[plot_idx%3,plot_idx//3].grid(which='both',axis='y',color='k',linestyle=':')
        # place a text box in upper left in axes coords
        textstr = '\n'.join((r'$mean \pm std=%.2f \pm %.2f$' % (pred_mean[plot_idx], pred_std[plot_idx],),r'$RMSE=%.2f$' % (pred_rmse[plot_idx], )))
        axs[plot_idx%3,plot_idx//3].text(0.05, 0.95, textstr, transform=axs[plot_idx%3,plot_idx//3].transAxes, fontsize=14,verticalalignment='top', bbox=props)
    
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
    # read datasets
    print(isinstance(datasets_ranges,np.ndarray))
    if(isinstance(datasets_ranges, np.ndarray)):
        datasets=copy.deepcopy(datasets_ranges)
    else:
        if('raw_datasets_path' in args.keys()):
            raw_datasets_path=args['raw_datasets_path']
        else:
            raw_datasets_path="./datasets_files/raw_datasets.hdf5"
        datasets=read_rawdata(range(datasets_ranges[0],datasets_ranges[1]),col_names,raw_datasets_path)
    # normalize datasets
    if(norm_type!=None):
        datasets_norm=norm_datasets(datasets,col_names,norm_type)
        pd_datasets=pd.DataFrame(data=datasets_norm,columns=col_names)
        print('plot normalized raw datasets')
    else:
        pd_datasets=pd.DataFrame(data=datasets,columns=col_names)
        print('plot raw datasets without normalization')

    # plots
    figsize=(14,16)
    fig=plt.figure(figsize=figsize)

    gs1=gridspec.GridSpec(10,2)#13
    gs1.update(hspace=0.1,wspace=0.1,top=0.95,bottom=0.01,left=0.04,right=0.98)
    axs=[]
    for plot_col in range(2):
        axs.append(fig.add_subplot(gs1[0:1,plot_col]))
        axs.append(fig.add_subplot(gs1[1:2,plot_col]))
        axs.append(fig.add_subplot(gs1[2:3,plot_col]))
        axs.append(fig.add_subplot(gs1[3:4,plot_col]))
        axs.append(fig.add_subplot(gs1[4:5,plot_col]))
        axs.append(fig.add_subplot(gs1[5:6,plot_col]))
        axs.append(fig.add_subplot(gs1[6:7,plot_col]))
        axs.append(fig.add_subplot(gs1[7:8,plot_col]))
        axs.append(fig.add_subplot(gs1[8:9,plot_col]))

    
    axs=np.array(axs).reshape(2,-1).T
    freq=100.0;
    Time=np.linspace(0,pd_datasets.shape[0]/freq,num=pd_datasets.shape[0])
    for plot_col in range(2):
        axs[0,plot_col].set_title(args['plot_title'])
        #plot features
        for plot_idx in range(6):
            axs[plot_idx,plot_col].plot(Time,pd_datasets[col_names[3*plot_idx+plot_col*18+1]])
            axs[plot_idx,plot_col].plot(Time,pd_datasets[col_names[3*plot_idx+plot_col*18+2]])
            axs[plot_idx,plot_col].plot(Time,pd_datasets[col_names[3*plot_idx+plot_col*18+3]])
            axs[plot_idx,plot_col].legend(col_names[3*plot_idx+plot_col*18+1:3*plot_idx+plot_col*18+4],ncol=3)
            axs[plot_idx,plot_col].grid(which='both',axis='x',color='k',linestyle=':')
            axs[plot_idx,plot_col].grid(which='both',axis='y',color='k',linestyle=':')
            #axs[plot_idx].set_ylabel()
            axs[plot_idx,plot_col].set_xticklabels([])
        # plot targets
        for plot_idx in range(6,9):
            axs[plot_idx,plot_col].plot(Time,pd_datasets[col_names[plot_idx-(12-plot_col*3)]])
            axs[plot_idx,plot_col].legend([col_names[plot_idx-(12-plot_col*3)]])
            axs[plot_idx,plot_col].grid(which='both',axis='x',color='k',linestyle=':')
            axs[plot_idx,plot_col].grid(which='both',axis='y',color='k',linestyle=':')
            axs[plot_idx,plot_col].set_xticklabels(labels=[])
        xticks=axs[plot_idx,plot_col].get_xticks()
        #axs[plot_idx,plot_col].set_ylim((-10,120))
        axs[plot_idx,plot_col].set_xticklabels([str(tt) for tt in xticks])
        axs[plot_idx,plot_col].set_xlabel("Time [s]")
    #plt.show()
    #pdb.set_trace()
    #reshape_pd_datasets=pd_datasets.melt('Time_vicon', var_name='cols',value_name='vals')
    #reshape_pd_datasets.head()
    #sns.lineplot(data=reshape_pd_datasets,x='Time_vicon', y='vals',hue='cols')
    datasets_visulization_path="./datasets_files/datasets_visulization/"+str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime()))+".svg"
    plt.savefig(datasets_visulization_path)
    plt.show()
    #data_mean, data_std = normalization_parameters(200,features_names)    
    #print(data_std)



def extract_subject_drop_landing_data(sub_idx: int)->np.ndarray:
    '''
    # Extract drop landing period data
    # The output is a three dimentional array, the first dimension is drop landing times
    # The second dimension is time
    # The third dimension is the features including 42
    '''
    end=fnn_model_v3.all_datasets_ranges['sub_'+str(sub_idx)]
    start=fnn_model_v3.all_datasets_ranges['sub_'+str(sub_idx-1)]
    sub_data=fnn_model_v3.read_rawdata(range(start,end),columns_names,raw_dataset_path)
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
    plt.savefig('./models_parameters_results/split_droplanding.svg')
    return (start_drop,end_drop)



#sub_idx=2
#start_drop, end_drop=extract_subject_drop_landing_data(sub_idx)

def drop_landing_period():
    # subject 0
    Up_sub0  = [1012, 1804,2594,3419,4157,4933,5695,6460]
    Down_sub0= [1173,1974,2755,3565,4306,5100,5863,6625]

    #subject 1
    Up_sub1   =[1268,2148,2970,3731,4453,5311,6312,6903]
    Down_sub1 =[1374,2355,3130,3904,4628,5473,6476,7090]







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
        'raw_datasets_path': "./datasets_files/raw_datasets.hdf5",
        'labels_names': [ 'L_IE', 'L_AA', 'L_FE','R_IE', 'R_AA', 'R_FE' ],
        'features_names': ['L_Up_Acc_X', 'L_Up_Acc_Y', 'L_Up_Acc_Z', 'L_Up_Gyr_X', 'L_Up_Gyr_Y','L_Up_Gyr_Z', 'L_Up_Mag_X', 'L_Up_Mag_Y','L_Up_Mag_Z',
                           'L_Lower_Acc_X', 'L_Lower_Acc_Y', 'L_Lower_Acc_Z', 'L_Lower_Gyr_X', 'L_Lower_Gyr_Y','L_Lower_Gyr_Z', 'L_Lower_Mag_X', 'L_Lower_Mag_Y','L_Lower_Mag_Z',
                           'R_Up_Acc_X', 'R_Up_Acc_Y', 'R_Up_Acc_Z', 'R_Up_Gyr_X', 'R_Up_Gyr_Y','R_Up_Gyr_Z', 'R_Up_Mag_X', 'R_Up_Mag_Y','R_Up_Mag_Z',
                           'R_Lower_Acc_X', 'R_Lower_Acc_Y', 'R_Lower_Acc_Z', 'R_Lower_Gyr_X', 'R_Lower_Gyr_Y','R_Lower_Gyr_Z', 'R_Lower_Mag_X', 'R_Lower_Mag_Y','R_Lower_Mag_Z']
}

