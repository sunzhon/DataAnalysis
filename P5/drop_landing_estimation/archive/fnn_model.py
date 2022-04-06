#Python

"""
Description:
    This is an module to implement ann to predict knee joint values in drop landing experiments 
    using pytorch 

Author: Sun Tao
Email: suntao.hn@gmail.com
Date: 2021-07-01

"""
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
import numpy
import pandas as pd
import os
import h5py
import re
import pandas as pd
import numpy as np

import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


from matplotlib import gridspec
import matplotlib.pyplot as plt
import time as localtimepkg
import seaborn as sns
import math
import inspect
import pdb
import datetime
import copy

from dp_lib import *




class DroplandingDataset(torch.utils.data.Dataset):
    """
    Dataset class
    Note: the key of fd is with a format "sub_*", * indicate a number, while the numbers are not consistency
    """
    def __init__(self,datafile,features_names,labels_names,train=True,norm_type='mean_std',preprocess_filer=False):
        with h5py.File(datafile,'r') as fd:
            keys=fd.keys()
            ## features and labels
            keys=list(fd.keys())

            #print(f.attrs["columns"]
            columns=fd[keys[0]].attrs.get('columns')
            # features and labels idx
            self.features_idx=[columns.tolist().index(f_name) for f_name in features_names]
            self.labels_idx=[columns.tolist().index(l_name) for l_name in labels_names]
                     
            self.all_datasets={subject: subject_data[:] for subject, subject_data in fd.items()}
            #Normalization
            if(norm_type=='mean_std'):
                data_mean, data_std=normalization_parameters(0,columns,datarange="all_subject",norm_type="mean_std")
                self.all_datasets={subject: (subject_data[:]-data_mean.values)/data_std.values for subject, subject_data in self.all_datasets.items()}
            if(norm_type=='max_min'):
                data_max, data_min=normalization_parameters(0,columns,datarange="all_subject",norm_type="max_min")
                self.all_datasets={subject: (subject_data[:]-data_min.values)/(data_max.values-data_min.values) for subject, subject_data in self.all_datasets.items()}
            # sorted directory. sub_0, sub_1, sub_2, sub_3.....
            sorted_keys=sorted(self.all_datasets.keys(),key=lambda key: int(key[4:]))
            if(preprocess_filer):
                temp_all_datasets=copy.deepcopy(self.all_datasets)
                window_size=5
                for sub_idx in sorted_keys:
                    for idx in range(temp_all_datasets[sub_idx].shape[0]-window_size):
                        temp_all_datasets[sub_idx][idx,:]=np.mean(self.all_datasets[sub_idx][idx*window_size:(idx+1)*window_size,:],axis=0)
                self.all_datasets=temp_all_datasets



            # put every subject data length in a list
            subject_data_length=[self.all_datasets[subject].shape[0] for subject in sorted_keys]
            # sum of data length, with an increase way to save the length in an array
            sum_subject_data_length=np.array([sum(subject_data_length[:index+1]) for index in range(len(subject_data_length))])
            # class attrs
            self.data_len=sum_subject_data_length[-1]
            self.data_len_list_sum=sum_subject_data_length
            
            
    def __len__(self):
        return self.data_len


    def __getitem__(self,row_idx):       
        #The index of the subjects
        sub_idx=np.argwhere(self.data_len_list_sum>row_idx)[0,0]
        if(sub_idx>0):
            row_idx=row_idx-self.data_len_list_sum[sub_idx-1]
        
        #Features and labels
        features = self.all_datasets['sub_' + str(sub_idx)][row_idx, self.features_idx]
        labels = self.all_datasets['sub_' + str(sub_idx)][row_idx, self.labels_idx]    
        
        return (torch.from_numpy(features).to(torch.float32), torch.from_numpy(labels).to(torch.float32))




import torch.nn as nn
import torch.nn.functional as F

class MyFNN_ModelV2(nn.Module):
    """
    FNN model
    """
    def __init__(self, num_features, num_labels):
        super(MyFNN_ModelV2,self).__init__()
        self.fnnModel=nn.Sequential(
            nn.Linear(num_features,500),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(500,400),
            nn.Tanh(),
            #-----------------#
            
            # #nn.Dropout(),
            # nn.Linear(400,400),
            # nn.Tanh(),
            # #nn.Dropout(),
            # nn.Linear(400,400),
            # nn.Tanh(),
            # nn.Linear(400,400),
            # nn.Tanh(),
            # nn.Linear(400,400),
            # nn.Tanh(),
            
            #----------------#
            nn.Linear(400,400),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(400,num_labels)
        )
        
    def forward(self,x):# batch_size, sequence, input_size=features_dim
        y=self.fnnModel(x)
        return y



class MyLSTM_MoldeV1(nn.Module):
    """
    LSTM
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size,seed=0,device='cpu'):
        # output size=3, represents 3 knee values of a leg,
        # input_size =2, represents 2 IMU of a leg
        # hidden_size = 3, here same with output size
        super(MyLSTM_MoldeV1,self).__init__()
        torch.manual_seed(seed)
        self.lstm_num_layers=num_layers
        self.lstm_hidden_size=hidden_size
        self.lstm=nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,bidirectional=True)
        self.fc=nn.Linear(hidden_size*2,100)
        self.fc1=nn.Linear(100,output_size)
        self.device=device

    def forward(self, inputs):
        # resize the inputs to N, L, H_in
        batch_size=inputs.shape[0]
        inputs.unsqueeze_(1).resize_(batch_size,2,num_features//2).transpose_(1,2)
        # input shape =(batch_size,sequence length,input_size)
        h0=torch.randn(2*self.lstm_num_layers, inputs.shape[0], self.lstm_hidden_size).to(self.device)# shape=(D*num_layers,batch_size,hidden_size)
        c0=torch.randn(2*self.lstm_num_layers, inputs.shape[0], self.lstm_hidden_size).to(self.device)#
        outputs, (hn, cn) = self.lstm(inputs, (h0, c0))
        #outputs shape=N*L*D*H_o
        outs=self.fc(outputs[:,-1,:].squeeze(dim=1))
        outs=self.fc1(outs)
        return outs



class MyLSTM_MoldeV2(nn.Module):
    """
    LSTM V2
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size,seed=0,device='cpu'):
        # output size=3, represents 3 knee values of a leg,
        # input_size =2, represents 2 IMU of a leg
        # hidden_size = 3, here same with output size
        super(MyLSTM_MoldeV1,self).__init__()
        torch.manual_seed(seed)
        self.lstm_num_layers=num_layers
        self.lstm_hidden_size=hidden_size
        self.lstm=nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,bidirectional=True)
        self.fc=nn.Linear(hidden_size*2,100)
        self.fc1=nn.Linear(100,output_size)
        self.device=device

    def forward(self, inputs):
        # resize the inputs to N, L, H_in
        batch_size=inputs.shape[0]
        inputs.unsqueeze_(1).resize_(batch_size,2,num_features//2).transpose_(1,2)
        # input shape =(batch_size,sequence length,input_size)
        h0=torch.randn(2*self.lstm_num_layers, inputs.shape[0], self.lstm_hidden_size).to(self.device)# shape=(D*num_layers,batch_size,hidden_size)
        c0=torch.randn(2*self.lstm_num_layers, inputs.shape[0], self.lstm_hidden_size).to(self.device)#
        outputs, (hn, cn) = self.lstm(inputs, (h0, c0))
        #outputs shape=N*L*D*H_o
        outs=self.fc(outputs[:,-1,:].squeeze(dim=1))
        outs=self.fc1(outs)
        return outs



    
def train_model(model,hyperparams, train_loader,eval_loader):
    """
    Description: train a model 
    Args:
        model is a deep neural network constructed based on torch
        hyperparams is a dictory storing the super parameters for this training process
        train_loader is an iterator which store the ready datasets for training
        eval_loader is an iterator which store the ready datasets for cross-validation. It will be
        applied in the future.
    """
    # 

    # super parameters
    epochs=hyperparams['epochs']
    cost_threashold=hyperparams['cost_threashold']
    features_names=hyperparams['features_names']
    labels_names=hyperparams['labels_names']
    batch_size=hyperparams['batch_size']
    learning_rate=hyperparams['learning_rate']
    device=hyperparams['device']
    num_features=len(features_names)
    num_labels=len(labels_names)
    print("#----------------Train process---------------------#")
    print("#                                                  #")
    print("epochs: {}\n, batch size: {}\n, learning rate: {}\n, num_features: {}\n, num_labels: {}".format(epochs, batch_size, learning_rate, num_features, num_labels))
    print("#                                                  #")
    print("#--------------------------------------------------#")
    
    #---folder to save model training process and reults
    training_folder = create_training_files(hyperparams=hyperparams)
    # -- Model to devices: gpu, do this before constructing optimizer
    model = model.to(device)

    # ---Loss ---
    criterion=nn.MSELoss()
    # ---Optimizer ---
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, last_epoch=-1)

    #optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
    
    # -- train time
    starttime = datetime.datetime.now()

    #--- Decalre variables
    loss_array = np.zeros((epochs,2)) # 保存loss, train and eval
    accuracy_list = [] # 保存accuracy
    iteration_list = [] # 保存循环次数
    outputs_list=[]
    itera_step=int(round(0.1*len(train_loader)))
    lossplot_x_axis=range(len(train_loader)*epochs//itera_step)
    
    #---- Training and evaluating loop
    for epoch in range(epochs):
        # Train loop
        train_loss_epoch=0
        for iteration, (features, labels) in enumerate(train_loader):
            model.train() # 声明训练
            features=features.to(device)
            labels = labels.to(device)
            # optimizer 梯度清零（否则会不断累加）
            optimizer.zero_grad()
            # 前向传播
            outputs = model(features)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            train_loss_epoch+=loss.cpu().detach().numpy()
        scheduler.step() #Update learning rate
        average_train_loss_epoch=train_loss_epoch/(iteration+1)
        loss_array[epoch,0] = average_train_loss_epoch# save the average train loss of current epoch

        # Evaluating loop
        eval_loss_epoch=0
        for iteration, (features,labels) in enumerate(eval_loader):
            with torch.no_grad():
                model.eval()
                features=features.to(device)
                labels=labels.to(device)
                outputs=model(features)
                loss = criterion(outputs, labels)
                eval_loss_epoch+=loss.cpu().detach().numpy()
        average_eval_loss_epoch=eval_loss_epoch/(iteration+1)
        loss_array[epoch,1] = average_eval_loss_epoch# save the average eval loss of current epoch

        # Display a epoch train and eval info
        print("epoch: {}, train loss: {:.5f}, eval loss: {:.5f}".format(epoch, average_train_loss_epoch, average_eval_loss_epoch))
        #if(epoch%int(0.2*epochs)==0):
        # Early stop when losses are less than a threshold

        if(train_loss_epoch*eval_loss_epoch<cost_threashold):
            # Save training results
            save_training_process(training_folder,loss_array)
            save_training_results(training_folder,model,loss_array)
            return training_folder

    
    # --- Training time
    endtime = datetime.datetime.now()
    during_time=(endtime - starttime).seconds
    print("Running time: {}".format(during_time))

    # -----Save training process (loss) and results (model and its params)
    save_training_process(training_folder,loss_array)
    save_training_results(training_folder,model,loss_array)
    return training_folder



'''
Split datasets into train, valid and test

'''
def split_dataset(scaled_series,sub_idx):
    #suntao experiment data
    if(isinstance(sub_idx,dict)):# split multiple subject data (dict), using leave-one-out cross-validtion
        train_split=scaled_series.shape[0]-2*DROPLANDING_PERIOD
        valid_split=scaled_series.shape[0]-DROPLANDING_PERIOD
    
    xy_train = scaled_series[:train_split,:]
    xy_valid = scaled_series[train_split:valid_split,:]
    xy_test = scaled_series[valid_split:,:]
    
    
    # Sensor segment calibration transfer process
    Init_stage_calibration=False
    if(Init_stage_calibration):
        np.random.seed(101)
        transfer_weight=np.random.randn(features_num,features_num)
        transfer_temp=np.dot(scaled_init_stage_sub_data['sub_'+str(sub_idx)][:,:features_num],transfer_weight)
        xy_train[:,:features_num]=xy_train[:,:features_num]+np.tanh(transfer_temp)
        xy_valid[:,:features_num]=xy_valid[:,:features_num]+np.tanh(transfer_temp)
        xy_test[:,:features_num]=xy_test[:,:features_num]+np.tanh(transfer_temp)
        # init info fom calibration stage 
        xy_train_init=scaled_init_stage_sub_data['sub_'+str(sub_idx)]*np.ones(xy_train.shape)
        xy_valid_init=scaled_init_stage_sub_data['sub_'+str(sub_idx)]*np.ones(xy_valid.shape)
        xy_test_init=scaled_init_stage_sub_data['sub_'+str(sub_idx)]*np.ones(xy_test.shape)
    
    print("Subject {:} dataset".format(sub_idx))
    print("xy_train shape:",xy_train.shape)
    print("xy valid shape:",xy_valid.shape)
    print("xy_test shape:",xy_test.shape)
    
    return xy_train, xy_valid,xy_test
    
    




def test_model(training_folder:str,test_loader,load_model_manner='whole_model',**args)->np.ndarray:
    """
    Description: Test model
    Args:
        training_folder save the model, paramters, dataset culumns that are used to test,

        test_datasets_range specify the row range of the datasets fro testing
    """
    print("#------------ Test process------------#")
    print("#                                     #")
    print("#-------------------------------------#")
    #---------Load super parameters of the model
    hyperparams_file=training_folder+"/hyperparams.yaml"
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    
    raw_datasets_path=hyperparams['raw_datasets_path']
    train_device=hyperparams['device']
    features_names=hyperparams['features_names']
    labels_names=hyperparams['labels_names']
    batch_size=hyperparams['batch_size']
    norm_type=hyperparams['norm_type']
    num_features=len(features_names)
    num_labels=len(labels_names)
    test_device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #-------------Create folder to save test results --------
    testing_results_folder = create_testing_files(training_folder)
    # ---------------Load model----------------------
    assert(load_model_manner in ['parameters_model_separate','whole_model']), 'incorrect model loading manner'
    #load model and its trained parameters separatively
    if(load_model_manner=='parameters_model_separate'):
        #create a model
        trained_model=MyFNN_ModelV2(num_features,num_labels)
        #load trained model parameters
        model_parameters_file=training_folder+"/train_results/model_parameters.pk1"
        if(test_device==train_device):
            model_state_dic=torch.load(model_parameters_file)
        else:
            model_state_dic=torch.load(model_parameters_file,map_location='cpu')

        trained_model.load_state_dict(model_state_dic)
    # load whole model
    if(load_model_manner=='whole_model'):
        model_file=training_folder+"/train_results/model.pth"
        trained_model=torch.load(model_file)
    
    trained_model.to(test_device)
    #--- Declare variables for features, labels, and predictions
    predictions_list=[]
    labels_list=[]
    features_list=[]

    #------------test loop---
    for index, (feature, label) in enumerate(test_loader):
        with torch.no_grad():
            features_list.append(feature.numpy())
            labels_list.append(label.numpy())
            trained_model.eval()
            feature=feature.to(device)
            prediction=trained_model(feature)
            predictions_list.append(prediction.cpu().numpy())
            
    # Inverse normalization of the model prediction, features and labels (autucal)
    features=inverse_norm(np.row_stack(features_list), features_names, norm_type=norm_type)# np.ndarray
    labels=inverse_norm(np.row_stack(labels_list), labels_names, norm_type)
    predictions=inverse_norm(np.row_stack(predictions_list), labels_names, norm_type=norm_type)
    if('display_plot' in args.keys()):
        plot_test_results(features,labels,predictions,features_names,labels_names,fig_save_folder=testing_results_folder)
    return (features,labels,predictions)








# basic parameters
all_datasets_len={'sub_0':6951, 'sub_1':7439, 'sub_2': 7686, 'sub_3': 8678, 'sub_4':6180, 'sub_5': 6671,
                  'sub_6': 7600, 'sub_7': 5583, 'sub_8': 6032, 'sub_9': 6508, 'sub_10': 6348, 'sub_11': 7010, 'sub_12': 8049, 'sub_13': 6248}
all_datasets_ranges={'sub_-1':0,'sub_0': 6951, 'sub_1': 14390, 'sub_2': 22076, 'sub_3': 30754, 'sub_4': 36934, 'sub_5': 43605,
                     'sub_6': 51205, 'sub_7': 56788, 'sub_8': 62820, 'sub_9': 69328, 'sub_10': 75676, 'sub_11':82686, 'sub_12': 90735, 'sub_13': 96983}


hyperparams={
        'norm_type': "mean_std",
        'batch_size': 64,
        'epochs': 2,
        'window_size': 10,
        'cost_threashold': 0.001,
        'learning_rate': 0.015,
        'device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        'raw_datasets_path': "./datasets_files/raw_datasets.hdf5",
        'labels_names': [ 'L_IE', 'L_AA', 'L_FE','R_IE', 'R_AA', 'R_FE' ],
        'features_names': ['L_Up_Acc_X', 'L_Up_Acc_Y', 'L_Up_Acc_Z', 'L_Up_Gyr_X', 'L_Up_Gyr_Y','L_Up_Gyr_Z', 'L_Up_Mag_X', 'L_Up_Mag_Y','L_Up_Mag_Z',
                           'L_Lower_Acc_X', 'L_Lower_Acc_Y', 'L_Lower_Acc_Z', 'L_Lower_Gyr_X', 'L_Lower_Gyr_Y','L_Lower_Gyr_Z', 'L_Lower_Mag_X', 'L_Lower_Mag_Y','L_Lower_Mag_Z',
                           'R_Up_Acc_X', 'R_Up_Acc_Y', 'R_Up_Acc_Z', 'R_Up_Gyr_X', 'R_Up_Gyr_Y','R_Up_Gyr_Z', 'R_Up_Mag_X', 'R_Up_Mag_Y','R_Up_Mag_Z',
                           'R_Lower_Acc_X', 'R_Lower_Acc_Y', 'R_Lower_Acc_Z', 'R_Lower_Gyr_X', 'R_Lower_Gyr_Y','R_Lower_Gyr_Z', 'R_Lower_Mag_X', 'R_Lower_Mag_Y','R_Lower_Mag_Z']
}


#---------------------------Main function----------------------------#

if __name__=="__main__":
    # ---------------- 超参数 -----------------------#
    '''
    features_names=['L_Up_Quat_q0', 'L_Up_Quat_q1', 'L_Up_Quat_q2', 'L_Up_Quat_q3',
    'L_Lower_Quat_q0', 'L_Lower_Quat_q1', 'L_Lower_Quat_q2', 'L_Lower_Quat_q3',
    'R_Up_Quat_q0', 'R_Up_Quat_q1', 'R_Up_Quat_q2', 'R_Up_Quat_q3',
    'R_Lower_Quat_q0', 'R_Lower_Quat_q1', 'R_Lower_Quat_q2', 'R_Lower_Quat_q3']
    '''



    '''

        'labels_names': [ 'R_IE', 'R_AA', 'R_FE' ],
        'features_names': ['R_Up_Acc_X', 'R_Up_Acc_Y', 'R_Up_Acc_Z', 'R_Up_Gyr_X', 'R_Up_Gyr_Y','R_Up_Gyr_Z', 'R_Up_Mag_X', 'R_Up_Mag_Y','R_Up_Mag_Z','R_Lower_Acc_X', 'R_Lower_Acc_Y', 'R_Lower_Acc_Z', 'R_Lower_Gyr_X', 'R_Lower_Gyr_Y','R_Lower_Gyr_Z', 'R_Lower_Mag_X', 'R_Lower_Mag_Y','R_Lower_Mag_Z']


        'labels_names': [ 'L_IE','L_AA','L_FE'],
        'features_names': ['L_Up_Acc_X', 'L_Up_Acc_Y', 'L_Up_Acc_Z', 'L_Up_Gyr_X', 'L_Up_Gyr_Y','L_Up_Gyr_Z', 'L_Up_Mag_X', 'L_Up_Mag_Y','L_Up_Mag_Z',
                           'L_Lower_Acc_X', 'L_Lower_Acc_Y', 'L_Lower_Acc_Z', 'L_Lower_Gyr_X', 'L_Lower_Gyr_Y','L_Lower_Gyr_Z', 'L_Lower_Mag_X', 'L_Lower_Mag_Y','L_Lower_Mag_Z']


    'labels_names': [ 'L_IE', 'L_AA', 'L_FE','R_IE', 'R_AA', 'R_FE' ],
    'features_names': ['L_Up_Acc_X', 'L_Up_Acc_Y', 'L_Up_Acc_Z', 'L_Up_Gyr_X', 'L_Up_Gyr_Y','L_Up_Gyr_Z', 'L_Up_Mag_X', 'L_Up_Mag_Y','L_Up_Mag_Z',
    'L_Lower_Acc_X', 'L_Lower_Acc_Y', 'L_Lower_Acc_Z', 'L_Lower_Gyr_X', 'L_Lower_Gyr_Y','L_Lower_Gyr_Z', 'L_Lower_Mag_X', 'L_Lower_Mag_Y','L_Lower_Mag_Z',
    'R_Up_Acc_X', 'R_Up_Acc_Y', 'R_Up_Acc_Z', 'R_Up_Gyr_X', 'R_Up_Gyr_Y','R_Up_Gyr_Z', 'R_Up_Mag_X', 'R_Up_Mag_Y','R_Up_Mag_Z',
    'R_Lower_Acc_X', 'R_Lower_Acc_Y', 'R_Lower_Acc_Z', 'R_Lower_Gyr_X', 'R_Lower_Gyr_Y','R_Lower_Gyr_Z', 'R_Lower_Mag_X', 'R_Lower_Mag_Y','R_Lower_Mag_Z']
    '''

    raw_datasets_path=hyperparams['raw_datasets_path']
    features_names=hyperparams['features_names']
    labels_names=hyperparams['labels_names']
    batch_size=hyperparams['batch_size']
    norm_type=hyperparams['norm_type']
    num_features=len(features_names)
    num_labels=len(labels_names)
    device=hyperparams['device']

    #---------------------加载数据------------------#
    if(True):
        '''
        datasets=DroplandingDataset(raw_datasets_path,features_names,labels_names,norm_type=norm_type)
        train_sets, eval_sets, test_sets=torch.utils.data.random_split(datasets,[5000,1000,86983+4000])
        train_loader = torch.utils.data.DataLoader(dataset=train_sets, batch_size=batch_size, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(dataset=eval_sets, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_sets, batch_size=batch_size, shuffle=False)
        '''
        datasets=DroplandingDataset(raw_datasets_path,features_names,labels_names,norm_type=norm_type)
        '''
        data_t=[]
        for dd in datasets:
            data_t.append(dd)
        print(len(data_t))
        X=np.row_stack([tt[1].numpy() for tt in data_t])
        print(X.shape)
        plt.plot(np.linspace(0,2.5,num=250),X[6000:6250])
        plt.show()
        '''
        '''
        indices_train, indices_eval, indices_test=range(5000),range(5000,6000),range(6000,6250)
        train_sampler=torch.utils.data.SubsetRandomSampler(indices_train)
        train_loader = torch.utils.data.DataLoader(datasets,sampler=train_sampler, batch_size=batch_size, shuffle=False)
        eval_sampler=torch.utils.data.SubsetRandomSampler(indices_eval)
        eval_loader = torch.utils.data.DataLoader(datasets,sampler=eval_sampler, batch_size=batch_size, shuffle=False)
        test_sampler=torch.utils.data.SubsetRandomSampler(indices_test)
        test_loader = torch.utils.data.DataLoader(datasets,sampler=test_sampler, batch_size=batch_size, shuffle=False)
        
        indices_train, indices_eval, indices_test=range(5000),range(5000,6000),range(6000,6250)
        train_loader=torch.utils.data.Subset(datasets,indices_train)
        eval_loader=torch.utils.data.Subset(datasets,indices_eval)
        test_loader=torch.utils.data.Subset(datasets,indices_test)
        '''

        indices_train, indices_eval, indices_test=range(5000),range(5000,6000),range(6000,6250)
        train_sets=torch.utils.data.Subset(datasets,indices_train)
        eval_sets=torch.utils.data.Subset(datasets,indices_eval)
        test_sets=torch.utils.data.Subset(datasets,indices_test)
        train_loader = torch.utils.data.DataLoader(dataset=train_sets, batch_size=batch_size, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(dataset=eval_sets, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_sets, batch_size=batch_size, shuffle=False)


    # ---------------训练模型 --------------------#
    if(True):
        #model=MyFNN_ModelV2(num_features,num_labels)
        model=MyLSTM_MoldeV1(input_size=2, hidden_size=num_labels, num_layers=2, output_size=num_labels,device=device)# input_size=number of IMU
        training_folder=train_model(model,hyperparams,train_loader,eval_loader)

    # ---------------测试模型  -------------------- #
    if(True):
        #training_folder="./models_parameters_results/2021-07-16/training_140602"
        features,labels,predictions= test_model(training_folder,test_loader,display_plot=True)

    # --------------评价指标 ----------------------------#
    # ------------ Raw data display----------------------#
    if(True):
        col_names=['Time_vicon','L_Up_Acc_X', 'L_Up_Acc_Y', 'L_Up_Acc_Z', 'L_Up_Gyr_X', 'L_Up_Gyr_Y','L_Up_Gyr_Z', 'L_Up_Mag_X', 'L_Up_Mag_Y','L_Up_Mag_Z',
               'L_Lower_Acc_X', 'L_Lower_Acc_Y', 'L_Lower_Acc_Z', 'L_Lower_Gyr_X', 'L_Lower_Gyr_Y','L_Lower_Gyr_Z', 'L_Lower_Mag_X', 'L_Lower_Mag_Y','L_Lower_Mag_Z',
               'R_Up_Acc_X', 'R_Up_Acc_Y', 'R_Up_Acc_Z', 'R_Up_Gyr_X', 'R_Up_Gyr_Y','R_Up_Gyr_Z', 'R_Up_Mag_X', 'R_Up_Mag_Y','R_Up_Mag_Z',
               'R_Lower_Acc_X', 'R_Lower_Acc_Y', 'R_Lower_Acc_Z', 'R_Lower_Gyr_X', 'R_Lower_Gyr_Y','R_Lower_Gyr_Z', 'R_Lower_Mag_X', 'R_Lower_Mag_Y','R_Lower_Mag_Z',
               'L_IE', 'L_AA', 'L_FE','R_IE', 'R_AA', 'R_FE' 
              ]

        datasets_ranges=(0,6951)
        sub_idx=0
        datasets_ranges=(all_datasets_ranges['sub_'+str(sub_idx-1)],all_datasets_ranges['sub_'+str(sub_idx)])
        display_rawdatase(datasets_ranges, col_names, norm_type=None, raw_datasets_path=raw_datasets_path,plot_title='Subject '+str(sub_idx))

