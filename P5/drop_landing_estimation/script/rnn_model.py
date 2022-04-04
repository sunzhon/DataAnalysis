#!/usr/bin/env python
# coding: utf-8
'''
 Import necessary packages

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import pandas as pd
import yaml
import h5py
print("tensorflow version:",tf.__version__)
import vicon_imu_data_process.process_rawdata as pro_rd
import estimation_assessment.estimation_scores as es_as

import seaborn as sns
import copy
import re
import json

from vicon_imu_data_process.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH, TRIALS
from vicon_imu_data_process.const import DROPLANDING_PERIOD, RESULTS_PATH
from vicon_imu_data_process import const


from sklearn.preprocessing import StandardScaler
from vicon_imu_data_process.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH, TRIALS


from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import time as localtimepkg



#subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0)

cpus=tf.config.list_logical_devices(device_type='CPU')
gpus=tf.config.list_logical_devices(device_type='GPU')
print(cpus,gpus)

'''
Set hyper parameters

'''
def initParameters(labels_names=None,features_names=None):
    # hyper parameters
    hyperparams={}
    
    # specify labels and features names
    if(labels_names==None):
        labels_names=LABELS_FIELDS
    else:
        labels_names=labels_names

    if(features_names==None):
        features_names=FEATURES_FIELDS
    else:
        features_names=features_names

    # specify other paramters
    columns_names=features_names+labels_names
    hyperparams['features_num']=len(features_names)
    hyperparams['labels_num']=len(labels_names)
    hyperparams['features_names']=features_names;
    hyperparams['labels_names']=labels_names
    hyperparams['learning_rate']=8e-2
    hyperparams['batch_size']=20
    hyperparams['window_size']=DROPLANDING_PERIOD
    hyperparams['shift_step']=DROPLANDING_PERIOD
    hyperparams['epochs']=30
    hyperparams['columns_names']=columns_names
    hyperparams['raw_dataset_path']= os.path.join(DATA_PATH,'features_labels_rawdatasets.hdf5')
    hyperparams['subjects_trials']={}
    subjects_list=['P_08','P_10', 'P_11', 'P_13', 'P_14', 'P_15','P_16','P_17','P_18','P_19','P_20','P_21','P_22','P_23', 'P_24']
    hyperparams=pro_rd.setHyperparams_subject(hyperparams,subjects_list)
    
    return hyperparams



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
    
    

'''
Packing data into windows 


'''

def windowed_dataset(series, hyperparams,shuffle_buffer):
    window_size=hyperparams['window_size']
    batch_size=hyperparams['batch_size']
    shift_step=hyperparams['shift_step']
    labels_num=hyperparams['labels_num']
    
    #series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=shift_step, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:,:-labels_num], w[:,-labels_num:]))
    ds = ds.batch(batch_size).prefetch(1)
    #print(list(ds.as_numpy_iterator())[0])
    return ds

# model prediction
def model_forecast(model, series, hyperparams):
    window_size=int(hyperparams['window_size'])
    batch_size=int(hyperparams['batch_size'])
    shift_step=int(hyperparams['shift_step'])
    labels_num=int(hyperparams['labels_num'])

    ds = tf.data.Dataset.from_tensor_slices(series[:,:-labels_num])
    ds = ds.window(window_size, shift=shift_step, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    #print(list(ds.as_numpy_iterator())[0])
    # model_prediction
    forecast = model.predict(ds)
    return forecast


'''
Model_V1 definition
'''
def model_v1(hyperparams):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(filters=60, kernel_size=6,
                          strides=1, padding="causal",
                          activation="selu",
                          input_shape=[None, hyperparams['features_num']]),
      #tf.keras.layers.LSTM(60, return_sequences=True),
      tf.keras.layers.LSTM(60, return_sequences=True),
      #tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(60, activation="relu"),
      tf.keras.layers.Dense(30, activation="relu"),
      tf.keras.layers.Dense(hyperparams['labels_num'])
      #tf.keras.layers.Lambda(lambda x: x *180)
    ])
    return model

'''
Define callback class

'''
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epcoh, logs={}):
        if(logs.get('loss')<0.003):
            print('\nLoss is low so cancelling training!')
            self.model.stop_training = True


    
'''

Model_V2 definition

'''
def model_v2():
    print(tf.__version__)
    ####定义一个方便构造常规 Sequential() 网络的函数
    def DNN_A_Graph(x,n_input=36,n_output=6,name='Transfer_graph'):
        tf.random.set_seed(50)
        np.random.seed(50)
        he = tf.initializers.he_normal()
        elu = tf.nn.elu
        x=Dense(n_input, kernel_initializer=he, activation=elu,name=name+'_1')(x)
        x=Dense(n_output,kernel_initializer=he, activation=elu,name=name+'_2')(x)
        return(x)
    
    def DNN_B_Graph(x,n_input=36,n_output=6,name='Main_graph'):
        ##
        x=tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=[window_size, features_num],name='Conv1D')(x)
        x=tf.keras.layers.LSTM(60, return_sequences=True,name='lstm_1')(x)
        x=tf.keras.layers.LSTM(60, return_sequences=True,name='lstm_2')(x)
        x=tf.keras.layers.Flatten()(x)
        x=tf.keras.layers.Dense(60, activation="relu")(x)
        x=tf.keras.layers.Dense(30)(x)
        x=tf.keras.layers.Dense(6)(x)
        return(x)
        
        
    #### 构造并联网络图
    ##需要并联的两个网络的输入
    input_a=tf.keras.layers.Input(shape=[features_num],name='Input_A')
    input_b=tf.keras.layers.Input(shape=[hyperparams['window_size'],features_num],name='Input_B')
    window_size=hyperparams['window_size']
    ##构造两个需要并联的子网络结构
    dnn_a=DNN_A_Graph(input_a,n_input=features_num,n_output=labels_num,name="DNN_A")
    dnn_b=DNN_B_Graph(input_b,n_input=features_num,n_output=labels_num,name="DNN_B")
    ##concat操作
    concat=tf.keras.layers.concatenate([dnn_a,dnn_b],axis=-1,name="Concat_Layer")
    ##在concat基础上继续添加一些层
    output=Dense(labels_num,name="Output_Layer")(concat)
    ##这一步很关键：这一步相当于把输入和输出对应起来，形成系统认识的一个完整的图。
    model_v2=Model(inputs=[input_a,input_b],outputs=[output])
    model_v2.get_layer('DNN_A_1').trainable=False
    model_v2.get_layer('DNN_A_2').trainable=False
    
    ##网络的其他组件
    optimizer=tf.keras.optimizers.Adam()
    loss_fn=tf.keras.losses.mean_squared_error
    model_v2.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=['accuracy'],
                  )
    
    model_v2.summary()
    #### 训练和测试：这里的x1,x2,对应前述的input_a和input_b
    #model.fit(x=[x1,x2],y=y,epochs=10,batch_size=500,verbose=2)
    
    #model.evaluate(x=[x1_test,x2_test],y=y_test,verbose=2)
    return model_v2



'''
Model training

'''
def train_model(model,hyperparams,train_set,valid_set,training_mode='Integrative_way'):
    # specify a training session
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    
    # Instance callback
    callbacks=myCallback()
    
    # Crerate train results folder
    training_folder=pro_rd.create_training_files(hyperparams=hyperparams)
    
    # register tensorboard writer
    sensorboard_file=os.path.join(training_folder,'tensorboard')
    if(os.path.exists(sensorboard_file)==False):
        os.makedirs(sensorboard_file)
    summary_writer=tf.summary.create_file_writer(sensorboard_file)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=hyperparams['learning_rate'], momentum=0.9)
    
    """ Integrated mode   """
    if training_mode=='Integrative_way':
        model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=optimizer,
                      metrics=["mae"])
        history = model.fit(train_set,epochs=hyperparams['epochs'],validation_data=valid_set,callbacks=[callbacks])
        history_dict=history.history
    """ Specified mode   """
    if training_mode=='Manual_way':
        tf.summary.trace_on(profiler=True) # 开启trace
        for batch_idx, (X,y_true) in enumerate(train_set): 
            with tf.GradientTape() as tape:
                y_pred=model(X)
                loss=tf.reduce_mean(tf.square(y_pred-y_true))
                # summary writer
                with summary_writer.as_default():
                    tf.summary.scalar('loss',loss,step=batch_idx)
            # calculate grads
            grads=tape.gradient(loss, model.variables)
            # update params
            optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))
        # summary trace
        history_dict={"loss":'none'}
        with summary_writer.as_default():
            tf.summary.trace_export(name='model_trace',step=0,profiler_outdir=sensorboard_file)
    
    
    # Save trained model, its parameters, and training history 
    save_trained_model(model,history_dict,training_folder)
    return model, history_dict, training_folder



'''
 Save trained model


'''
def save_trained_model(trained_model,history_dict,training_folder,**args):

    # Load hyperparameters 
    hyperparams_file=training_folder+"/hyperparams.yaml"
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()

    # use the subjects' index as the saved file' name
    train_sub_idx=hyperparams['train_sub_idx']
    train_sub_idx_str=train_sub_idx[0]+train_sub_idx[-1]


    # Save weights and models
    
    # checkpoints
    checkpoint_folder=os.path.join(train_sub_idx_str,'checkpoints')
    if(os.path.exists(checkpoint_folder)==False):
        os.makedirs(checkpoint_folder)
    checkpoint_name='my_checkpoint_'+train_sub_idx_str+'.ckpt'
    checkpoint=tf.train.Checkpoint(myAwesomeModel=trained_model)
    checkpoint_manager=tf.train.CheckpointManager(checkpoint,directory=checkpoint_folder,
                                                  checkpoint_name=checkpoint_name,max_to_keep=20)
    checkpoint_manager.save()
    
        
    #saved_model
    saved_model_folder=os.path.join(training_folder,'trained_model')
    if(os.path.exists(saved_model_folder)==False):
        os.makedirs(saved_model_folder)
    saved_model_file=saved_model_folder+'/my_model_'+train_sub_idx_str+'.h5'
    trained_model.save(saved_model_file)
    
    # save history
    # Get the dictionary containing each metric and the loss for each epoch
    history_folder = os.path.join(training_folder,'train_process')
    history_file = history_folder +'/my_history_'+train_sub_idx_str
    # Save it under the form of a json file
    with open(history_file,'w') as fd:
        json.dump(history_dict, fd)
    # load history
    #history_dict = json.load(open(history_path, 'r'))
    
    
    
'''
Training model_v2

'''
def train_model_v2():
    # train model_v2
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    callbacks=myCallback()
    history = model_v2.fit([xy_train_init,train_set],epochs=150),#validation_data=[valid_set_init,valid_set],callbacks=[callbacks])
    return history


'''
Plot the history metrics in training process

'''
def plot_history(history_dict):
    print(history_dict.keys())
    plt.plot(history_dict['loss'],'r')
    plt.plot(history_dict['val_loss'],'g')
    plt.plot(history_dict['mae'])
    plt.legend(['train loss', 'valid loss','mae'])
    
    #plt.axis([0,150, 0.0,0.035])
    print('max train MAE: {:.4f} and max val MAE: {:.4f}'.format(max(history_dict['mae']),max(history_dict['val_mae'])))



'''
Testing model
'''
def test_model(training_folder, xy_test,scaler,**args):
    
    #1) Crerate test results folder
    testing_folder=pro_rd.create_testing_files(training_folder)
    
    #2) Load hyperparameters, Note the values in hyperparams become string type
    hyperparams_file=os.path.join(training_folder,"hyperparams.yaml")
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()

    
    #3) use subjects idx of trainning as file names
    train_sub_idx=hyperparams['train_sub_idx']
    train_sub_idx_str=train_sub_idx[0]+train_sub_idx[-1]

    #4) load model
    saved_model_file=os.path.join(training_folder,'trained_model','my_model_'+train_sub_idx_str+'.h5')
    #print(saved_model_file)
    trained_model=tf.keras.models.load_model(saved_model_file)
    
    #5) test data
    model_output = model_forecast(trained_model, xy_test, hyperparams)
    model_prediction=np.row_stack([model_output[:-1,0,:],model_output[-1,0:,:]])
    
    
    #print("Test dataset shape:",xy_test.shape)
    #print("Model output shape",model_output.shape)
    #print("Model prediction shape",model_prediction.shape)
    
    #6) Reshape and inverse normalization
    prediction_xy_test=copy.deepcopy(xy_test) # deep copy of test data
    prediction_xy_test[:,-int(hyperparams['labels_num']):]=model_prediction # using same shape with all datasets
    predictions = scaler.inverse_transform(prediction_xy_test)[:,-int(hyperparams['labels_num']):] # inversed norm predition
    labels  = scaler.inverse_transform(xy_test)[:,-int(hyperparams['labels_num']):]
    features= scaler.inverse_transform(xy_test)[:,:-int(hyperparams['labels_num'])]
    
    save_test_result(features,labels,predictions,testing_folder)
    
    return features,labels,predictions,testing_folder

'''
save test results
'''
def save_test_result(features,labels,predictions,testing_folder):
    saved_test_results_file=testing_folder+"/test_results"+'.h5'
    with h5py.File(saved_test_results_file,'w') as fd:
        fd.create_dataset('features',data=features)
        fd.create_dataset('labels',data=labels)
        fd.create_dataset('predictions',data=predictions)





'''
Plot the estimation results

'''
def plot_prediction(features,labels,predictions,testing_folder):
    
    #1) evaluate using two metrics, mae and mse
    mae=tf.keras.metrics.mean_absolute_error(labels, predictions).numpy()
    mse=tf.keras.metrics.mean_squared_error(labels, predictions).numpy()
    print('Mean absolute error: {:.3f}, mean root squard error:{:.3f} in a period'.format(np.mean(mae),np.mean(np.sqrt(mse))))
    
    #2) preparation

    #i) load hyperparameters, Note: the values in hyperparams become string type
    if(re.search('test_\d',testing_folder)!=None):
        testing_folder=os.path.dirname(testing_folder) # 父目录

    training_folder=testing_folder+"/../training_"+re.search("\d+$",testing_folder).group()
    hyperparams_file=training_folder+"/hyperparams.yaml"
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    
    #ii) read features and label names from hyper parameters    
    features_names=hyperparams['features_names']
    labels_names=hyperparams['labels_names']
    
    
    #iii) create file name to save plot results
    test_sub_idx=hyperparams['test_sub_idx']
    test_sub_idx_str=''
    for ii in test_sub_idx:
        test_sub_idx_str+='_'+str(ii)
        
    prediction_file=testing_folder+'/sub'+test_sub_idx_str+'_estimation.svg'
    prediction_error_file=testing_folder+'/sub'+test_sub_idx_str+'_estimation_error.svg'
    
    
    #iv) plot the estimation results and errors
    es_as.plot_estimation_comparison(labels, predictions, labels_names,testing_folder, 
                                  prediction_file=prediction_file,prediction_error_file=prediction_error_file)

    
    
def plot_history(history_dict):
    print(history_dict.keys())
    plt.plot(history_dict['loss'],'r')
    plt.plot(history_dict['val_loss'],'g')
    plt.plot(history_dict['mae'])
    plt.legend(['train loss', 'valid loss','mae'])
    
    #plt.axis([0,150, 0.0,0.035])
    print('max train MAE: {:.4f} and max val MAE: {:.4f}'.format(max(history_dict['mae']),max(history_dict['val_mae'])))


    
def plot_prediction_statistic(features, labels, predictions,testing_folder):
    '''
    This function calculate the error between predicted and ground truth, and plot them for comparison
    '''
    
    if(re.search('test_\d',testing_folder)!=None):
        testing_folder=os.path.dirname(testing_folder) # 父目录

    # Load hyperparameters, Note the values in hyperparams become string type
    training_folder=testing_folder+"/../training"+re.search(r"_\d+$",testing_folder).group()
    hyperparams_file=training_folder+"/hyperparams.yaml"
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    
    # hyper parameters    
    features_names=hyperparams['features_names']
    labels_names=hyperparams['labels_names']
    
    
    # test subject idx, which one is for testing
    test_sub_idx=hyperparams['test_sub_idx']
    test_sub_idx_str=''
    for ii in test_sub_idx:
        test_sub_idx_str+='_'+str(ii)

    pd_error, pd_NRMSE = estimation_accuracy(predictions,labels,labels_names)

    plot_estimation_accuracy(pd_error, pd_NRMSE)
    
    
def estimation_accuracy(estimation, actual, labels_names):
    # Plot the statistical results of the estimation results and errors
    error=abs(estimation-actual)
    pd_error=pd.DataFrame(data=error,columns=labels_names)
    NRMSE=100.0*np.sqrt(pd_error.apply(lambda x: x**2).mean(axis=0).to_frame().transpose())/(actual.max(axis=0)-actual.min(axis=0))
    #*np.ones(pd_error.shape)*100
    pd_NRMSE=pd.DataFrame(data=NRMSE, columns = [col for col in list(pd_error.columns)])

    return pd_error, pd_NRMSE 


def plot_estimation_accuracy(pd_error, pd_NRMSE):
    # create experiment results folder
    # MAE
    fig=plt.figure(figsize=(10,2))
    style = ['darkgrid', 'dark', 'white', 'whitegrid', 'ticks']
    sns.set_style(style[4],{'grid.color':'k'})
    sns.catplot(data=pd_error,kind='bar', palette="Set3").set(ylabel='Absolute error [deg]')
    #plt.text(2.3,1.05, r"$\theta_{ae}(t)=abs(\hat{\theta}(t)-\theta)(t)$",horizontalalignment='center', fontsize=20)
    test_sub_idx=hyperparams['test_sub_idx']
    savefig_file=testing_folder+'/sub'+str(test_sub_idx_str)+'_mae.svg'
    plt.savefig(savefig_file)
    
    # NRMSE
    fig=plt.figure(figsize=(10,3))
    sns.catplot(data=pd_NRMSE,kind='bar', palette="Set3").set(ylabel='NRMSE [%]')
    #plt.text(2.3, 2.6, r"$NRMSE=\frac{\sqrt{\sum_{t=0}^{T}{\theta^2_{ae}(t)}/T}}{\theta_{max}-\theta_{min}} \times 100\%$",horizontalalignment='center', fontsize=20)
    savefig_file=testing_folder+'/sub'+str(test_sub_idx_str)+'_nrmse.svg'
    plt.savefig(savefig_file)
    

    


'''
Normalize all subject data

'''
def normalize_subjects_data(hyperparams):
    sub_idxs=hyperparams['subjects_trials']
    if(isinstance(sub_idxs,list)):
        hyperparams['subjects_trials'] = ['sub_'+str(ii) for ii in sub_idxs]
        xy_data, scaled_xy_data, scaler = pro_rd.load_normalize_data(hyperparams)
        subject_data_len=pro_rd.all_datasets_len
    
        norm_trials_data={}
        start,end=0,0
        for ii, sub_idx in enumerate(sub_idxs):
            sub_idx_str='trial_'+str(sub_idx)
            if(ii==0):
                start=0
            else:
                start=end
            end+=subject_data_len[sub_idx_str]
            
            norm_trials_data[sub_idx_str]=xy_data[start:end,:]
        return norm_trials_data, scaler
    
    # suntao experimental data
    if(isinstance(sub_idxs,dict)):
        xy_data, scaled_xy_data, scaler = pro_rd.load_normalize_data(hyperparams,assign_trials=True)
        norm_trials_data={}
        for idx in range(scaled_xy_data.shape[0]):# trasnfer list of trials to a dictory of trials
            norm_trials_data['trial_'+str(idx)]=scaled_xy_data[idx,:,:]
        return norm_trials_data, scaler



'''
Main rountine for developing ANN model for biomechanic variable estimations

'''
def train_test_loops(hyperparams=None):
    #1) Set hyper parameters
    if(hyperparams==None):
        hyperparams=initParameters()
    else:
        hyperparams=hyperparams
    
    #2) Create a list of training and testing files
    train_test_folder= os.path.join(RESULTS_PATH,"models_parameters_results/"+str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime())))
    if(os.path.exists(train_test_folder)==False):
        os.makedirs(train_test_folder)    
    train_test_folders_log=os.path.join(train_test_folder,"train_test_folders.log")
    if(os.path.exists(train_test_folders_log)):
        os.remove(train_test_folders_log)
    dict_log={'training_folder':[],'testing_folder':[]}
    
    #3) Load and normalize datasets for training and testing
    norm_trials_data,scaler=normalize_subjects_data(hyperparams)


    #4) leave-one-out cross-validation
    loo = LeaveOneOut()
    loop_times=0
    for train_sub_index, test_sub_index in loo.split(list(hyperparams['subjects_trials'].keys())):
        loop_times=loop_times+1

        #i) decide train and test subject dataset 
        print("train subject set:", train_sub_index, "test subject set:", test_sub_index)
        hyperparams['train_sub_idx'] = [str(idx) for idx in train_sub_index]
        hyperparams['test_sub_idx'] = [str(idx) for idx in test_sub_index]

        # tran and test data trails index
        train_trial_index  = [str(sub_idx*len(TRIALS)+trial_idx) for sub_idx in train_sub_index for trial_idx in range(len(TRIALS))]
        test_trial_index  = [str(sub_idx*len(TRIALS)+trial_idx) for sub_idx in test_sub_index for trial_idx in range(len(TRIALS))]

        xy_train=[norm_trials_data['trial_'+ ii] for ii in train_trial_index]
        xy_valid=[norm_trials_data['trial_'+ test_trial_index[0]]]
        xy_test=[norm_trials_data['trial_'+ test_trial_index[1]]]
        
        xy_train=np.concatenate(xy_train,axis=0)
        xy_valid=np.concatenate(xy_valid,axis=0)
        xy_test=np.concatenate(xy_test,axis=0)
        
        #ii) load train and test dataset
        train_set = windowed_dataset(xy_train, hyperparams,   shuffle_buffer=1000)
        valid_set = windowed_dataset(xy_valid, hyperparams,   shuffle_buffer=1000)
        print("Train set shape",xy_train.shape)
        print("Valid set shape",xy_valid.shape)
        print("Test set shape",xy_test.shape)
        #print("X Shape for a iteration train",list(train_set.as_numpy_iterator())[0][0].shape)
        #print("Y Shape for a iteration train",list(train_set.as_numpy_iterator())[0][1].shape)

        #iii) declare model
        model=model_v1(hyperparams)

        #iv) train model
        trained_model,history_dict,training_folder=train_model(model,hyperparams,train_set,valid_set)
        
        #v) test model
        features, labels, predictions, testing_folder = test_model(training_folder,xy_test,scaler)
        dict_log['training_folder'].append(training_folder)
        dict_log['testing_folder'].append(testing_folder)
        print("scores (r2, rmse, mae, r_rmse):", es_as.get_scores(labels,predictions))

        #vi) Plot estimation results

        #plot_prediction(features,labels,predictions,testing_folder)
        #plot_prediction_statistic(features, labels, predictions,testing_folder)
        if loop_times>4: # only repeat 4 times
            break;# only run a leave-one-out a time
    
    
    #5) save train and test folder path
    with open(train_test_folders_log,'w') as fd:
        yaml.dump(dict_log,fd)

    return dict_log, xy_test, scaler



def deploy_sensor_combination():

    # sensor placement configurations
    list_testing_folders={}
    sensor_placement_combination_dict = {
                                   'F': ['L_FOOT'],
                                   'S': ['L_SHANK'],
                                   'T': ['L_THIGH'],
                                   'W': ['WAIST'],
                                   'C': ['CHEST'],
                                   'FS': ['L_FOOT','L_SHANK'],
                                   'FT': ['L_FOOT','L_THIGH'],
                                   'FW': ['L_FOOT','WAIST'],
                                   'FC': ['L_FOOT','CHEST'],
                                   'FST': ['L_FOOT','L_SHANK','L_THIGH'], 
                                   'FTW': ['L_FOOT','L_THIGH','WAIST'], 
                                   'FWC': ['L_FOOT','WAIST','CHEST'], 
                                   'FSTW': ['L_FOOT','L_SHANK','L_THIGH','WAIST'], 
                                   'FSTC': ['L_FOOT','L_SHANK','L_THIGH','CHEST'], 
                                   'FSTWC': ['L_FOOT','L_SHANK','L_THIGH','WAIST', 'CHEST']
                                  }
    #sensor_placement_combination_dict = {'F': ['L_FOOT']}
    #model_size_dict = {'layer_size': 3}
    # train and test model
    for config, sensor_list in sensor_placement_combination_dict.items():
        print("Sensor is:",sensor_list)
        features_fields = const.extract_imu_fields(sensor_list, const.IMU_RAW_FIELDS)
        hyperparams=initParameters(labels_names=LABELS_FIELDS, features_names=features_fields)
        print('features are:', hyperparams['features_names'])
        dict_log, xy_test, scaler =  train_test_loops(hyperparams)# model traning
        list_testing_folders[config]=dict_log

    # save testing folders
    overall_metrics_folder= os.path.join(RESULTS_PATH,"overall_metrics_results",str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime())), str(localtimepkg.strftime("%H%M%S", localtimepkg.localtime())))
    if(not os.path.exists(overall_metrics_folder)):
        os.makedirs(overall_metrics_folder)

    overall_metrics_file= os.path.join(overall_metrics_folder,"testing_result_folders.txt")
    if(os.path.exists(overall_metrics_file)==False):
        with open(overall_metrics_file,'a') as f:
            for config, dict_log in list_testing_folders.items():
                for testing_folder in dict_log["testing_folder"]: # in a loops which has many train and test loop 
                    f.write(config+'\t'+testing_folder+'\n')

    return overall_metrics_file



def display_testing_results(overall_metrics_file):
    # open testing folder
    assessment=[]
    for line in open(overall_metrics_file,'r'):
        #0) tesing results folder
        line=line.strip('\n')
        [sensor_config, testing_folder]=line.split('\t')
        print(testing_folder)

        #1) load testing results
        testing_results=os.path.join(testing_folder,'test_results.h5')
        with h5py.File(testing_results,'r') as fd:
            features=fd['features'][:,:]
            predictions=fd['predictions'][:,:]
            labels=fd['labels'][:,:]
        
        #2) estimation results
        #i) plot curves
        plot_prediction(features,labels,predictions,testing_folder)
        #ii) collect results
        temp=list(es_as.get_scores(labels,predictions));temp.insert(0,sensor_config)
        assessment.append(temp)

    # statistically results
    columns=['Sensor configurations','r2','mae','rmse','r_rmse']
    pd_assessment=pd.DataFrame(assessment,columns=columns)
    pd_assessment.to_csv(re.search("[\s\S]+(\d)+",overall_metrics_file).group()+ "_metrics.csv")
    pd_assessment=pd_assessment.melt(id_vars=['Sensor configurations'],var_name='Metrics',value_name='Value')


    # plot statistical results
    figwidth=13;figheight=10
    subplot_left=0.06; subplot_right=0.97; subplot_top=0.95;subplot_bottom=0.06
    g=sns.catplot(data=pd_assessment,x='Sensor configurations',y='Value',col='Metrics',col_wrap=2,kind='bar',height=3, aspect=0.8,sharey=False)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom,hspace=0.1, wspace=0.1)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    [ax.yaxis.grid(True) for ax in g.axes]
    plt.show()
    file=re.search("[\s\S]+(\d)+",overall_metrics_file).group()
    g.savefig(re.search("[\s\S]+(\d)+",overall_metrics_file).group()+ "_metrics.svg")


        
if __name__=='__main__':
    #overall_metrics_file=deploy_sensor_combination()
    overall_metrics_file= os.path.join(RESULTS_PATH,"overall_metrics_results/2022-04-03/220718/testing_result_folders.txt")
    display_testing_results(overall_metrics_file)
    
