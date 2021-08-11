#Python

"""
import package
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import torch
import h5py
import numpy




def read_rawdata(row_idx: int,col_names: list)-> numpy.ndarray:
    """
    @Description:
    To read the data from h5 file and normalize the features and labels.
    @Parameters:
    Row_idx: the index of row. data type is int
    Col_names: the names of columns. data type is string
    
    """
    with h5py.File('raw_datasets.hdf5', 'r') as fd:
        ## The coloms of the features and labels
        keys=list(fd.keys())
        columns=fd[keys[0]].attrs.get('columns')
        col_idxs=[]
        for col_name in col_names:
            col_idxs.append(np.argwhere(columns==col_name)[0][0])
        
        data_len_list=[]
        for idx in range(len(fd.keys())):
            key="sub_"+str(idx)
            #print(key)
            data_len_list.append(len(fd[key]))
        
        data_len_list_sum=[]
        sum_num=0
        for num in data_len_list:
            sum_num+=num
            data_len_list_sum.append(sum_num)
        
        data_len_list_sum=np.array(data_len_list_sum)
        
        sub_idx=np.argwhere(data_len_list_sum > row_idx)[0,0]
        if(sub_idx>0):
            row_idx=row_idx-data_len_list_sum[sub_idx-1]
            
        return fd['sub_'+str(sub_idx)][row_idx,col_idxs]
    
    

def normalization_parameters(row_idx,col_names):
    with h5py.File('raw_datasets.hdf5', 'r') as fd:
        keys=list(fd.keys())# the keys/columns name of the datafile 
        columns=fd[keys[0]].attrs.get('columns')
        col_idxs=[]
        for col_name in col_names:
            col_idxs.append(np.argwhere(columns==col_name)[0][0])
    
        data_len_list=[]
        for idx in range(len(fd.keys())):
            key="sub_"+str(idx)
            data_len_list.append(len(fd[key]))
    
        data_len_list_sum=[]
        sum_num=0
        for num in data_len_list:
            sum_num+=num
            data_len_list_sum.append(sum_num)
    
        data_len_list_sum=np.array(data_len_list_sum)
    
        sub_idx=np.argwhere(data_len_list_sum > row_idx)[0,0]
        if(sub_idx>0):
            row_idx=row_idx-data_len_list_sum[sub_idx-1]
            sub_idx=np.argwhere(data_len_list_sum > row_idx)[0,0]
            if(sub_idx>0):
                row_idx=row_idx-data_len_list_sum[sub_idx-1]
        
        mean=np.mean(fd['sub_'+str(sub_idx)][:,col_idxs],axis=0,keepdims=True)
        std=np.std(fd['sub_'+str(sub_idx)][:,col_idxs],axis=0,keepdims=True)
        data_mean=pd.DataFrame(data=mean,columns=col_names)
        data_std=pd.DataFrame(data=std,columns=col_names)
        return data_mean, data_std    
        

#features_names=['L_Up_Quat_q0', 'L_Up_Quat_q1', 'L_Up_Quat_q2', 'L_Up_Quat_q3','L_Lower_Quat_q0', 'L_Lower_Quat_q1', 'L_Lower_Quat_q2', 'L_Lower_Quat_q3']
#print(read_rawdata(20000,['R_IE']))
#data_mean, data_std = normalization_parameters(200,features_names)    
#print(data_std)

## Dataset class

class DroplandingDataset(torch.utils.data.Dataset):
    def __init__(self,datafile,features_names,labels_names,train=True):
        with h5py.File(datafile,'r') as fd:
            keys=fd.keys()
            ## features and labels
            keys=list(fd.keys())
            #print(f.attrs["columns"]
            columns=fd[keys[0]].attrs.get('columns')
            '''
            features_names=['L_Up_Quat_q0', 'L_Up_Quat_q1', 'L_Up_Quat_q2', 'L_Up_Quat_q3',
                        'L_Lower_Quat_q0', 'L_Lower_Quat_q1', 'L_Lower_Quat_q2', 'L_Lower_Quat_q3']
            labels_names=['R_IE']
            '''
            
            # 16 features and 6 lables/targets
            # features and labels idx
            features_idx=[]
            labels_idx=[]
            for f_name in features_names:
                features_idx.append(np.argwhere(columns==f_name)[0,0])
            for l_name in labels_names:
                labels_idx.append(np.argwhere(columns==l_name)[0,0])
                
            #row_length and row_idx
            data_len_list=[]
            self.all_datasets={}
            for idx in range(len(fd.keys())):
                key="sub_"+str(idx)
                data_len_list.append(len(fd[key]))
                temp_data=np.array(fd[key])
                #Normalization (正则化)
                temp_mean = np.mean(temp_data,axis=0,keepdims=True)
                temp_std = np.std(temp_data,axis=0,keepdims=True)
                temp_data=(temp_data-temp_mean)/temp_std
                self.all_datasets[key]=temp_data
                
            
            #summary data length
            data_len_list_sum=[]
            sum_num=0
            for num in data_len_list:
                sum_num+=num
                data_len_list_sum.append(sum_num)
        
            data_len_list_sum=np.array(data_len_list_sum)
            
            #class attrs
            self.data_len=sum_num
            self.features_idx=features_idx
            self.labels_idx=labels_idx
            self.data_len_list_sum=data_len_list_sum
        
        
    def __len__(self):
        #print(self.data_len)
        return self.data_len

    def __getitem__(self,row_idx):       
        #The index of the subjects 
        sub_idx=np.argwhere(self.data_len_list_sum>row_idx)[0,0]
        if(sub_idx>0):
            row_idx=row_idx-self.data_len_list_sum[sub_idx-1]
        
        #Features and labels
        features = self.all_datasets['sub_'+str(sub_idx)][row_idx,self.features_idx]
        labels = self.all_datasets['sub_'+str(sub_idx)][row_idx,self.labels_idx]    
        
        #print("feature type:{} and shape:{}".format(type(features),features.shape))
        return (torch.from_numpy(features).to(torch.float32),torch.from_numpy(labels).to(torch.float32))




import torch.nn as nn
import torch.nn.functional as F

class MyFNN_ModelV2(nn.Module):
    def __init__(self,num_features, num_labels):
        super(MyFNN_ModelV2,self).__init__()
        self.fnnModel=nn.Sequential(
            nn.Linear(num_features,200),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(200,100),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(100,100),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(100,50),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(50,num_labels)
        )
        
    def forward(self,x):# batch_size, sequence, input_size=features_dim
        y=self.fnnModel(x)
        return y


# Declare super parameters
# Two legs's knee angles, right: interal rotation, abduction, flexion as well as left size: ...
import matplotlib.pyplot as plt
import time as localtimepkg
import os
batch_size=100
epochs=1200
learning_rate=0.0001

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

h5format_dataset="raw_datasets.hdf5"

labels_names=['R_IE', 'R_AA', 'R_FE', 'L_IE', 'L_AA', 'L_FE']
#labels_names=['R_FE']
features_names=['L_Up_Acc_X', 'L_Up_Acc_Y', 'L_Up_Acc_Z', 'L_Up_Gyr_X', 'L_Up_Gyr_Y','L_Up_Gyr_Z',
'L_Lower_Acc_X', 'L_Lower_Acc_Y', 'L_Lower_Acc_Z', 'L_Lower_Gyr_X', 'L_Lower_Gyr_Y','L_Lower_Gyr_Z',
'R_Up_Acc_X', 'R_Up_Acc_Y', 'R_Up_Acc_Z', 'R_Up_Gyr_X', 'R_Up_Gyr_Y','R_Up_Gyr_Z',
'R_Lower_Acc_X', 'R_Lower_Acc_Y', 'R_Lower_Acc_Z', 'R_Lower_Gyr_X', 'R_Lower_Gyr_Y','R_Lower_Gyr_Z']

num_features=len(features_names)
num_labels=len(labels_names)

print("num_features: {}, num_labels: {}".format(num_features,num_labels))


## Declare
import os
import matplotlib.pyplot as plt
import time as localtimepkg
import datetime


def save_model_parameters(based_path_folder, model, loss_list: list, iteration):
  """
  @Description: save model parameters: including paramters and loss values
  @Args:
    based_path_folder, model, loss_list, iteration
  @Output: valid
  """
  # Create based folder to save trained model parameters and loss values
  save_based_path_folder=based_path_folder+str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime()))
  if(os.path.exists(save_based_path_folder)==False):
    os.makedirs(save_based_path_folder)
  # create subfolder for model parameters
  save_based_path_folder_model_parameters=based_path_folder+str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime()))+"/model_parameters/"
  if(os.path.exists(save_based_path_folder_model_parameters)==False):
    os.makedirs(save_based_path_folder_model_parameters)
  #creat subfolder for loss values
  save_based_path_folder_loss_values=based_path_folder+str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime()))+"/loss_values/"
  if(os.path.exists(save_based_path_folder_loss_values)==False):
    os.makedirs(save_based_path_folder_loss_values)

  # save model parameters
  model_parameters_file=save_based_path_folder_model_parameters+"fnn_model_parameters_loss_"+str(round(float(loss_list[-1]),2))+"_iter_"+str(iteration)+u".pk1"
  if(os.path.exists(model_parameters_file)):
    os.remove(model_parameters_file)
  torch.save(model.state_dict(),model_parameters_file)
  # save loss values
  loss_values_file=save_based_path_folder_loss_values+"loss_value"+"_iter_"+str(iteration)+u".csv"
  if(os.path.exists(loss_values_file)):
    os.remove(loss_values_file)
  pd_loss=pd.DataFrame(data={'loss_values':loss_list})
  pd_loss.to_csv(loss_values_file)



def train(model,train_loader,test_loader):

  if torch.cuda.is_available():
      model = model.cuda()
  criterion=nn.MSELoss()
  optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
  #optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

  iter=0
  loss_list = [] # 保存loss
  accuracy_list = [] # 保存accuracy
  iteration_list = [] # 保存循环次数
  outputs_list=[]

  for epoch in range(epochs):
      for i, (features, labels) in enumerate(train_loader):
          model.train() # 声明训练
          features=features.to(device)
          labels = labels.to(device)
          # 梯度清零（否则会不断累加）
          optimizer.zero_grad()
          # 前向传播
          #print(" features shape:{}, labels shape:{} ".format(features.shape, labels.shape))
          outputs = model(features)
          #print("outputs shape:{}, labels shape:{} ".format(outputs.shape, labels.shape))
          # 计算损失
          loss = criterion(outputs, labels)
          # 反向传播
          loss.backward()
          # 更新参数
          optimizer.step()
          # 计数器自动加1
          iter+=1
          iter_step=500
          lossplot_x_axis=range(len(train_loader)*epochs//iter_step)
          if(iter%iter_step==0):
              #plt.gca().cla()
              loss_list.append(loss.cpu().detach().numpy())
              iteration_list.append(iter)
              print("epoch: {}, loss: {}".format(epoch,loss.cpu().detach()))
              #Visualization of trainning
              #plt.cla()
              # 无误差真值曲线
              #plt.scatter(features.cpu().numpy()[-1,1], labels.cpu().numpy()[-1,1], c='blue', lw='3')
              # 有误差散点
              #plt.scatter(x_data.numpy(), y_data.numpy(), c='orange')
              # 实时预测的曲线
              if(epoch%int(0.2*epochs)==0):
                saved_figure_name=save_based_path_folder_plots+str(epoch)+str(iter)+".png"
                plt.plot(iteration_list,loss_list, 'ro', lw='2')
                plt.title("epoch: "+str(epoch))
                plt.savefig(saved_figure_name)
                plt.clf()
              #plt.text(-0.5, -65, 'Time=%d Loss=%.4f' % (i, loss.cpu().data.numpy()), fontdict={'size': 15, 'color': 'red'})
              #plt.pause(0.1)
              if(loss_list[-1]<0.06):
                save_model_parameters(based_path_folder,model,loss_list,iter)
                return 0

  save_model_parameters(based_path_folder,model,loss_list,iter)
  return -1

#---------------------------Functions-----------------


datasets=DroplandingDataset(h5format_dataset,features_names,labels_names)
train_sets, test_sets, val_sets=torch.utils.data.random_split(datasets,[60000,10000,26983])
  # 创建数据集的可迭代对象，并且分批、打乱数据集
train_loader = torch.utils.data.DataLoader(dataset=train_sets, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_sets, batch_size=batch_size, shuffle=True)
  #print("train_loader:", next(iter(train_loader)).shape)
model=MyFNN_ModelV2(num_features,num_labels)

# save plots
based_path_folder="./trained_model_parameters/"
save_based_path_folder_plots=based_path_folder+str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime()))+"/plots/"
if(os.path.exists(save_based_path_folder_plots)==False):
  os.makedirs(save_based_path_folder_plots)


starttime = datetime.datetime.now()
train(model,train_loader,test_loader)
endtime = datetime.datetime.now()
during_time=(endtime - starttime).seconds
print("Running time: {}".format(during_time))



