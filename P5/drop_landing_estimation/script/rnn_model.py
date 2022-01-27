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
# load datasets in a numpy 
import package_lib.dp_process_rawdata as dp_lib

import seaborn as sns
import copy
import re

from package_lib.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH, TRIALS
from package_lib.const import DROPLANDING_PERIOD, EXPERIMENT_RESULTS_PATH


from sklearn.preprocessing import StandardScaler
from package_lib.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH, TRIALS

subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0)

cpus=tf.config.list_logical_devices(device_type='CPU')
gpus=tf.config.list_logical_devices(device_type='GPU')

print(cpus,gpus)

'''
Set hyper parameters

'''
def initParameters():
    # hyper parameters
    hyperparams=dp_lib.hyperparams
    
    labels_names=LABELS_FIELDS
    features_names=FEATURES_FIELDS
    
    columns_names=features_names+labels_names
    hyperparams['features_num']=len(features_names)
    hyperparams['labels_num']=len(labels_names)
    hyperparams['features_names']=features_names;
    hyperparams['labels_names']=labels_names
    hyperparams['learning_rate']=10e-2
    hyperparams['batch_size']=4
    hyperparams['window_size']=DROPLANDING_PERIOD
    hyperparams['shift_step']=DROPLANDING_PERIOD
    hyperparams['epochs']=10
    hyperparams['columns_names']=columns_names
    hyperparams['raw_dataset_path']= os.path.join(DATA_PATH,'features_labels_rawdatasets.hdf5')
    
    return hyperparams



'''
Split datasets into train, valid and test

'''
def split_dataset(scaled_series,sub_idx):
    #1) Split raw dataset into train, valid and test dataset
    if(isinstance(sub_idx,int)):# split single subject data 
        all_train_split={'sub_0':5600,'sub_1':6000,'sub_2':6400,'sub_3':7500,
                         'sub_4':4900,'sub_5':6000,'sub_6':4500,'sub_7':4500,
                         'sub_8':4900,'sub_9':5100,'sub_10':5000,'sub_11':5700,'sub_12':6400,'sub_13':4000}
        all_valid_split={'sub_0':6500,'sub_1':6800,'sub_2':7000,'sub_3':8100,
                         'sub_4':5600,'sub_5':6900,'sub_6':5100,'sub_7':5000,
                         'sub_8':5600,'sub_9':5900,'sub_10':5800,'sub_11':6200,'sub_12':7200, 'sub_13':4900}
        train_split = all_train_split['sub_'+str(sub_idx)]
        valid_split= all_valid_split['sub_'+str(sub_idx)]
    if(isinstance(sub_idx,list)):# split multiple subject data, using leave-one-out cross-validtion
        train_split=scaled_series.shape[0]-3000
        valid_split=scaled_series.shape[0]-2000
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
    ds=ds.batch(batch_size).prefetch(1)
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
      tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=[None, hyperparams['features_num']]),
      tf.keras.layers.LSTM(60, return_sequences=True),
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
    # train model_v1
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    
    # Instance callback
    callbacks=myCallback()
    
    # Crerate train results folder
    training_folder=dp_lib.create_training_files(hyperparams=hyperparams)
    
    # register tensorboard writer
    sensorboard_file=training_folder+'/tensorboard/'
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
    
    
    # Save trained model and its parameters, history 
    save_trainedModel(model,history_dict,training_folder)
    return model, history_dict, training_folder



'''
 Save trained model


'''
def save_trainedModel(trained_model,history_dict,training_folder,**args):
    # Load hyperparameters 
    hyperparams_file=training_folder+"/hyperparams.yaml"
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    # sub_idx of the subjects for training 
    train_sub_idx=hyperparams['train_sub_idx']
    train_sub_idx_str=''
    if(len(train_sub_idx)>10):# if subject has too much, then just use its first and last sub to name
        train_sub_idx_str=train_sub_idx[0]+train_sub_idx[-1]
    else:
        for ii in train_sub_idx:
            train_sub_idx_str+='_'+str(ii)

    # Save weights and models
    
    # checkpoints
    checkpoint_folder=training_folder+'/checkpoints/'
    if(os.path.exists(checkpoint_folder)==False):
        os.makedirs(checkpoint_folder)
    checkpoint_name='my_checkpoint_sub'+train_sub_idx_str+'.ckpt'
    checkpoint=tf.train.Checkpoint(myAwesomeModel=trained_model)
    checkpoint_manager=tf.train.CheckpointManager(checkpoint,directory=checkpoint_folder,
                                                  checkpoint_name=checkpoint_name,max_to_keep=20)
    checkpoint_manager.save()
    
        
    #saved_model
    saved_model_folder=training_folder+'/saved_model/'
    if(os.path.exists(saved_model_folder)==False):
        os.makedirs(saved_model_folder)
    saved_model_file=saved_model_folder+'my_model_sub'+train_sub_idx_str+'.h5'
    trained_model.save(saved_model_file)
    
    # save history
    import json
    # Get the dictionary containing each metric and the loss for each epoch
    history_path= training_folder+'/train_process/my_history_sub'+train_sub_idx_str
    # Save it under the form of a json file
    with open(history_path,'w') as fd:
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
    testing_folder=dp_lib.create_testing_files(training_folder)
    
    #2) Load hyperparameters, Note the values in hyperparams become string type
    hyperparams_file=training_folder+"/hyperparams.yaml"
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()

    
    #3) sub_idx of the subjects for training 
    train_sub_idx=hyperparams['train_sub_idx']
    train_sub_idx_str=''
    if(len(train_sub_idx)>10):# if subject has too much, then just use its first and last sub to name
        train_sub_idx_str=train_sub_idx[0]+train_sub_idx[-1]
    else:
        for ii in train_sub_idx:
            train_sub_idx_str+='_'+str(ii)

    #4) Load model
    saved_model_file=training_folder+'/saved_model/my_model_sub'+train_sub_idx_str+'.h5'
    #saved_model_file=training_folder+'/saved_model/my_model_sub_'+'0123456789a'+'.h5'
    #print(saved_model_file)
    trained_model=tf.keras.models.load_model(saved_model_file)
    
    #5) Test data
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
    
    save_testResult(features,labels,predictions,testing_folder)
    
    return features,labels,predictions,testing_folder

'''
save test results
'''
def save_testResult(features,labels,predictions,testing_folder):
    saved_test_results_file=testing_folder+"/test_results"+'.h5'
    with h5py.File(saved_test_results_file,'w') as fd:
        fd.create_dataset('features',data=features)
        fd.create_dataset('labels',data=labels)
        fd.create_dataset('predictions',data=predictions)



#testing_folder='./model/test_2020'
#training_folder=testing_folder+"/../training_"+re.search(r"\d+$",testing_folder).group()
#print(training_folder)


'''
Plot the estimation results

'''
def plot_prediction(features,labels,predictions,testing_folder):
    
    #1) evaluate using two metrics, mae and mse
    mae=tf.keras.metrics.mean_absolute_error(labels, predictions).numpy()
    mse=tf.keras.metrics.mean_squared_error(labels, predictions).numpy()
    print('MAE: {:.3f}, RMSE:{:.3f} in a period'.format(np.mean(mae),np.mean(np.sqrt(mse))))
    
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
    dp_lib.plot_test_results(features, labels, predictions, features_names, labels_names,testing_folder,
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
    
    
    # Plot the statistical results of the estimation results and errors
    error=abs(predictions-labels)
    pd_error=pd.DataFrame(data=error,columns=labels_names)
    NRMSE=100.0*np.sqrt(pd_error.apply(lambda x: x**2).mean(axis=0).to_frame().transpose())/(labels.max(axis=0)-labels.min(axis=0))
    #*np.ones(pd_error.shape)*100
    pd_NRMSE=pd.DataFrame(data=NRMSE, columns = [col for col in list(pd_error.columns)])
    
    
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
    

    


import numpy as np
from sklearn.model_selection import LeaveOneOut
import time as localtimepkg




'''
Normalize all subject data

'''
def normalize_subjects_data(hyperparams):
    sub_idxs=hyperparams['sub_idx']
    if(isinstance(sub_idxs,list)):
        hyperparams['sub_idx'] = ['sub_'+str(ii) for ii in sub_idxs]
        xy_data, scaled_xy_data, scaler = dp_lib.load_normalize_data(hyperparams)
        subject_data_len=dp_lib.all_datasets_len
    
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
        xy_data, scaled_xy_data, scaler = dp_lib.load_normalize_data(hyperparams,assign_trials=True)
        norm_trials_data={}
        for idx in range(scaled_xy_data.shape[0]):
            norm_trials_data['trial_'+str(idx)]=scaled_xy_data[idx,:,:]
        return norm_trials_data, scaler



'''
Main rountine for developing ANN model for biomechanic variable estimations

'''
def main():
    #1) Setup hyper parameters
    hyperparams=initParameters()
    
    #2) Create a list of training and testing files
    train_test_folder= os.path.join(EXPERIMENT_RESULTS_PATH,"models_parameters_results/"+str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime())))
    if(os.path.exists(train_test_folder)==False):
        os.makedirs(train_test_folder)    
    train_test_folder_log=train_test_folder+"/train_test_folder.log"
    if(os.path.exists(train_test_folder_log)):
        os.remove(train_test_folder_log)
    log_dict={'training_folder':[],'testing_folder':[]}
    
    #3) Load and normalize datasets for training and testing
    norm_trials_data,scaler=normalize_subjects_data(hyperparams)


    #4) leave-one-out cross-validation
    loo = LeaveOneOut()
    applied_dataset_trials = range(len(norm_trials_data.keys()))
    for train_index, test_index in loo.split(applied_dataset_trials):
        #i) decide train and test subject dataset 
        print("train set:", train_index, "test set:", test_index)
        hyperparams['train_sub_idx']=[str(ii) for ii in  train_index] # the values of params should be str or int types
        hyperparams['test_sub_idx']=[str(ii) for ii in test_index]
        xy_train=[norm_trials_data['trial_'+str(ii)] for ii in train_index]
        xy_test=[norm_trials_data['trial_'+str(ii)] for ii in test_index]
        
        xy_train=np.concatenate(xy_train,axis=0)
        xy_test=np.concatenate(xy_test,axis=0)
        xy_valid=xy_test
        
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
        log_dict['training_folder'].append(training_folder)
        log_dict['testing_folder'].append(testing_folder)
         
        #vi) Plot estimation results
        #plot_prediction(features,labels,predictions,testing_folder)
        #plot_prediction_statistic(features, labels, predictions,testing_folder)
        break;# only run a leave-one-out a time
    
    
    #5) save train and test folder path
    with open(train_test_folder_log,'w') as fd:
        yaml.dump(log_dict,fd)

    return training_folder, testing_folder, xy_test, scaler



if __name__=='__main__':

    #0) Train and test model or testing existing model
    if(True):#retrain model
        training_folder, testing_folder, xy_test, scaler =  main()
    else:# plot existing model
        hyperparams=initParameters()
        if(not testing_folder in locals().keys()):
            testing_folder = os.path.join(EXPERIMENT_RESULTS_PATH,'models_parameters_results/2022-01-24/test_115856/test_1')
            training_folder = os.path.join(EXPERIMENT_RESULTS_PATH,'models_parameters_results/2022-01-24/training_115856/')

    #1) load testing results
    testing_results=os.path.join(testing_folder,'test_results.h5')
    #print(testing_results)
    with h5py.File(testing_results,'r') as fd:
        features=fd['features'][:,:]
        predictions=fd['predictions'][:,:]
        labels=fd['labels'][:,:]
    
    #2) visulize estimation results
    #i) plot curves
    plot_prediction(features,labels,predictions,testing_folder)
    #ii) statistical estimation results
    plot_prediction_statistic(features, labels, predictions,testing_folder)
    
    

