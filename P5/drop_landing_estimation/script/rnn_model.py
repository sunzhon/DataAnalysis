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

from vicon_imu_data_process.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH, TRAIN_USED_TRIALS
from vicon_imu_data_process.const import DROPLANDING_PERIOD, RESULTS_PATH
from vicon_imu_data_process import const


from sklearn.preprocessing import StandardScaler


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




def initParameters(labels_names=None,features_names=None,subject_ids=None):
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

    if(subject_ids==None):
        subject_ids=['P_08','P_10', 'P_13', 'P_14', 'P_15','P_16','P_17','P_18','P_19','P_20','P_21','P_22','P_23']

    # specify other paramters
    columns_names=features_names+labels_names
    hyperparams['features_num']=len(features_names)
    hyperparams['labels_num']=len(labels_names)
    hyperparams['features_names']=features_names
    hyperparams['labels_names']=labels_names
    hyperparams['learning_rate']=10e-2
    hyperparams['batch_size']=20
    hyperparams['window_size']=DROPLANDING_PERIOD
    hyperparams['shift_step']=DROPLANDING_PERIOD
    hyperparams['epochs']=10
    hyperparams['columns_names']=columns_names
    hyperparams['raw_dataset_path']= os.path.join(DATA_PATH,'features_labels_rawdatasets.hdf5')
    hyperparams['subjects_trials']=pro_rd.select_valid_subjects_trials(subject_ids)
    
    return hyperparams


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
def save_trained_model(trained_model,history_dict,training_folder,**kwargs):

    # load hyperparameters 
    hyperparams_file=training_folder+"/hyperparams.yaml"
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    else:
        print("Not Found hyper params file at {}".format(hyperparams_file))
        exit()


    #-----save trained model and parameters----------#
    
    # save checkpoints
    checkpoint_folder=os.path.join(training_folder,'checkpoints')
    if(os.path.exists(checkpoint_folder)==False):
        os.makedirs(checkpoint_folder)
    checkpoint_name='my_checkpoint.ckpt'
    checkpoint=tf.train.Checkpoint(myAwesomeModel=trained_model)
    checkpoint_manager=tf.train.CheckpointManager(checkpoint,directory=checkpoint_folder,
                                                  checkpoint_name=checkpoint_name,max_to_keep=20)
    checkpoint_manager.save()
    
        
    #save trained model
    saved_model_folder=os.path.join(training_folder,'trained_model')
    if(os.path.exists(saved_model_folder)==False):
        os.makedirs(saved_model_folder)
    saved_model_file=saved_model_folder+'/my_model.h5'
    trained_model.save(saved_model_file)
    
    # save training history
    # Get the dictionary containing each metric and the loss for each epoch
    history_folder = os.path.join(training_folder,'train_process')
    history_file = history_folder +'/my_history'
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
def test_model(training_folder, xy_test, scaler, **kwargs):
    
    #1) create test results folder
    testing_folder=pro_rd.create_testing_files(training_folder)
    
    #2) load hyperparameters, note that the values in hyperparams become string type
    hyperparams_file=os.path.join(training_folder,"hyperparams.yaml")
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    else:
        print("Not Found hyper params file at {}".format(hyperparams_file))
        exit()

    #4) load trained model
    trained_model_file=os.path.join(training_folder,'trained_model','my_model.h5')
    trained_model=tf.keras.models.load_model(trained_model_file)
    
    #5) test data
    model_output = model_forecast(trained_model, xy_test, hyperparams)
    model_prediction=np.row_stack([model_output[:-1,0,:],model_output[-1,0:,:]])
    
    #6) reshape and inverse normalization
    prediction_xy_test = copy.deepcopy(xy_test) # deep copy of test data
    prediction_xy_test[:,-int(hyperparams['labels_num']):]=model_prediction # using same shape with all datasets
    predictions = scaler.inverse_transform(prediction_xy_test)[:,-int(hyperparams['labels_num']):] # inversed norm predition
    labels  = scaler.inverse_transform(xy_test)[:,-int(hyperparams['labels_num']):]
    features= scaler.inverse_transform(xy_test)[:,:-int(hyperparams['labels_num'])]
    
    # save params in testing
    hyperparams_file = os.path.join(testing_folder,"hyperparams.yaml")
    with open(hyperparams_file,'w') as fd:
        yaml.dump(hyperparams,fd)
    
    # save testing results
    save_test_result(features, hyperparams['features_names'], labels, hyperparams['labels_names'], predictions, testing_folder)
    save_test_metrics(features, hyperparams['features_names'], labels, hyperparams['labels_names'], predictions, testing_folder)

    return features,labels,predictions,testing_folder

'''
save test results
'''
def save_test_result(features, features_names, labels, labels_names, predictions, testing_folder):
    saved_test_results_file = os.path.join(testing_folder, "test_results.h5")
    with h5py.File(saved_test_results_file,'w') as fd:
        fd.create_dataset('features',data=features)
        fd.create_dataset('labels',data=labels)
        fd.create_dataset('predictions',data=predictions)
        fd['features'].attrs['features_names'] = features_names
        fd['labels'].attrs['labels_names'] = labels_names


def save_test_metrics(features, features_names, labels, labels_names, predictions, testing_folder):
    metrics_file = os.path.join(testing_folder, "test_metrics.csv")

    metrics = es_as.get_estimation_metrics(labels, predictions, labels_names)

    metrics.to_csv(metrics_file)
    


'''
Plot the estimation results

'''
def plot_prediction(features,labels,predictions,testing_folder):
    
    #1) evaluate using two metrics, mae and mse
    mae=tf.keras.metrics.mean_absolute_error(labels, predictions).numpy()
    mse=tf.keras.metrics.mean_squared_error(labels, predictions).numpy()
    print('Mean absolute error: {:.3f}, mean root squard error:{:.3f} in a period'.format(np.mean(mae),np.mean(np.sqrt(mse))))
    
    #2) preparation

    #i) load hyper parameters
    hyperparams_file = os.path.join(testing_folder,"hyperparams.yaml")
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    
    #ii) read features and label names from hyper parameters    
    features_names=hyperparams['features_names']
    labels_names=hyperparams['labels_names']
    
    
    #iii) create file name to save plot results
    test_subject_ids = hyperparams['test_subject_ids']
    prediction_file = os.path.join(testing_folder, test_subject_ids[0] + '_estimation.svg')
    prediction_error_file = os.path.join(testing_folder, test_subject_ids[0] + '_estimation_error.svg')
    
    
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
    
    # load hyperparameters, Note the values in hyperparams become string type
    hyperparams_file = os.path.join(testing_folder,"hyperparams.yaml")
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    
    # hyper parameters    
    features_names=hyperparams['features_names']
    labels_names=hyperparams['labels_names']
    
    
    # test subject idx, which one is for testing
    test_subject_ids=hyperparams['test_subject_ids']
    test_subject_ids_str=''
    for ii in test_subject_ids:
        test_subject_ids_str+='_'+str(ii)

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
    test_subject_ids=hyperparams['test_subject_ids']
    savefig_file=testing_folder+'/sub'+str(test_subject_ids_str)+'_mae.svg'
    plt.savefig(savefig_file)
    
    # NRMSE
    fig=plt.figure(figsize=(10,3))
    sns.catplot(data=pd_NRMSE,kind='bar', palette="Set3").set(ylabel='NRMSE [%]')
    #plt.text(2.3, 2.6, r"$NRMSE=\frac{\sqrt{\sum_{t=0}^{T}{\theta^2_{ae}(t)}/T}}{\theta_{max}-\theta_{min}} \times 100\%$",horizontalalignment='center', fontsize=20)
    savefig_file=testing_folder+'/sub'+str(test_subject_ids_str)+'_nrmse.svg'
    plt.savefig(savefig_file)
    

    


'''
Normalize all subject data

'''
def normalize_subjects_data(hyperparams):
    subjects_trials=hyperparams['subjects_trials']
    
    assert(subjects_trials,dict)

    # load and normalize dataset, scaled_xy_data is a three dimension matrics, the first dimensioin is trials
    xy_data, scaled_xy_data, scaler = pro_rd.load_normalize_data(hyperparams,assign_trials=True)
    scaled_subjects_trials={}

    # trasnfer list of trials to a dictory of subjects
    idx=0
    for subject_id_name, trials in subjects_trials.items():
        scaled_subjects_trials[subject_id_name] = {}
        for trial in trials:
            scaled_subjects_trials[subject_id_name][trial]=scaled_xy_data[idx,:,:]
            idx=idx+1
    
    return scaled_subjects_trials, scaler



'''
Main rountine for developing ANN model for biomechanic variable estimations

'''
def train_test_loops(hyperparams=None):
    #1) set hyper parameters
    if(hyperparams==None):
        hyperparams=initParameters()
    else:
        hyperparams=hyperparams
    
    #2) create a list of training and testing files
    train_test_folder= os.path.join(RESULTS_PATH,"models_parameters_results/"+str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime())))
    if(os.path.exists(train_test_folder)==False):
        os.makedirs(train_test_folder)    
    train_test_folders_log=os.path.join(train_test_folder,"train_test_folders.log")
    if(os.path.exists(train_test_folders_log)):
        os.remove(train_test_folders_log)
    dict_log={'training_folder':[],'testing_folder':[]}
    
    #3) Load and normalize datasets for training and testing
    norm_subjects_trials_data,scaler=normalize_subjects_data(hyperparams)


    #4) leave-one-out cross-validation
    loo = LeaveOneOut()
    loop_times = 0
    subjects_trials = hyperparams['subjects_trials']
    subjects=list(subjects_trials.keys())
    for train_subject_ids, test_subject_ids in loo.split(subjects):
        loop_times=loop_times+1

        #i) subjects for train and test
        train_subjects=[subjects[subject_id] for subject_id in train_subject_ids]
        test_subjects=[subjects[subject_id] for subject_id in test_subject_ids]

        #i) decide train and test subject dataset 
        print("train subject set:", train_subject_ids, "test subject set:", test_subject_ids)
        hyperparams['train_subject_ids'] = train_subjects
        hyperparams['test_subject_ids'] = test_subjects

        # data from train and test subjects and their trials
        xy_train = [norm_subjects_trials_data[subject_id_name][trial] for subject_id_name in train_subjects for trial in subjects_trials[subject_id_name]]
        xy_valid = [norm_subjects_trials_data[subject_id_name][trial] for subject_id_name in test_subjects for trial in subjects_trials[subject_id_name]]
        xy_test=[xy_valid[1]]
        
        xy_train=np.concatenate(xy_train,axis=0)
        xy_valid=np.concatenate(xy_valid,axis=0)
        xy_test=np.concatenate(xy_test,axis=0)
        
        #ii) load train and test dataset
        train_set = windowed_dataset(xy_train, hyperparams,   shuffle_buffer=1000)
        valid_set = windowed_dataset(xy_valid, hyperparams,   shuffle_buffer=1000)
        print("Train set shape",xy_train.shape)
        print("Valid set shape",xy_valid.shape)
        print("Test set shape",xy_test.shape)


        #iii) declare model
        model=model_v1(hyperparams)

        #iv) train model
        trained_model,history_dict,training_folder = train_model(model,hyperparams,train_set,valid_set)
        
        #v) test model
        features, labels, predictions, testing_folder = test_model(training_folder,xy_test,scaler)
        dict_log['training_folder'].append(training_folder)
        dict_log['testing_folder'].append(testing_folder)
        es_as.get_estimation_metrics(labels, predictions, hyperparams['labels_names'])

        #vi) Plot estimation results
        #plot_prediction(features,labels,predictions,testing_folder)
        #plot_prediction_statistic(features, labels, predictions,testing_folder)
        
        #if loop_times > 0: # only repeat 4 times
        #   break;# only run a leave-one-out a time
    
    
    #5) save train and test folder path
    with open(train_test_folders_log,'w') as fd:
        yaml.dump(dict_log,fd)

    return dict_log, xy_test, scaler



def deploy_sensor_combination():

    #1) sensor placement configurations
    list_testing_folders={}
    sensor_placement_combination_dict = {
                                   'F': ['L_FOOT'],
                                  # 'S': ['L_SHANK'],
                                  # 'T': ['L_THIGH'],
                                  # 'W': ['WAIST'],
                                  # 'C': ['CHEST'],
                                  # 'FS': ['L_FOOT','L_SHANK'],
                                  # 'FT': ['L_FOOT','L_THIGH'],
                                  # 'FW': ['L_FOOT','WAIST'],
                                  # 'FC': ['L_FOOT','CHEST'],
                                  # 'FST': ['L_FOOT','L_SHANK','L_THIGH'], 
                                  # 'FTW': ['L_FOOT','L_THIGH','WAIST'], 
                                  # 'FWC': ['L_FOOT','WAIST','CHEST'], 
                                  # 'FSTW': ['L_FOOT','L_SHANK','L_THIGH','WAIST'], 
                                  # 'FSTC': ['L_FOOT','L_SHANK','L_THIGH','CHEST'], 
                                   'FSTWC': ['L_FOOT','L_SHANK','L_THIGH','WAIST', 'CHEST']
                                  }
    #sensor_placement_combination_dict = {'F': ['L_FOOT']}
    #model_size_dict = {'layer_size': 3}

    #2) train and test model
    for config, sensor_list in sensor_placement_combination_dict.items():
        print("Sensor is:",sensor_list)
        features_fields = const.extract_imu_fields(sensor_list, const.ACC_GYRO_FIELDS)
        hyperparams=initParameters(labels_names=LABELS_FIELDS, features_names=features_fields)
        print('features are:', hyperparams['features_names'])
        dict_log, xy_test, scaler =  train_test_loops(hyperparams)# model traning
        list_testing_folders[config]=dict_log

    #3) save testing folders
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


'''


'''
def display_testing_results(overall_metrics_file):
    # open testing folder
    assessment=[]
    for line in open(overall_metrics_file,'r'):
        #0) tesing results folder
        line=line.strip('\n')
        [sensor_config, testing_folder]=line.split('\t')
        print(testing_folder)
        test_id=re.search("test_([0-9])+",testing_folder).group(0)

        #1) load testing results
        testing_results = os.path.join(testing_folder,'test_results.h5')
        with h5py.File(testing_results,'r') as fd:
            features=fd['features'][:,:]
            predictions=fd['predictions'][:,:]
            labels=fd['labels'][:,:]
            labels_names=fd['labels'].attrs['labels_names']

        #2) estimation results
        #i) plot curves
        plot_prediction(features,labels,predictions,testing_folder)
        #ii) collect metric results
        metrics = es_as.get_estimation_metrics(labels, predictions, labels_names)

        metrics['Sensor configurations'] = sensor_config
        metrics['Test ID'] = test_id
        assessment.append(metrics)

    # concate to a pandas dataframe
    pd_assessment=pd.concat(assessment, axis=0)

    # save pandas DataFrame
    overall_metrics_folder=re.search("[\s\S]+(\d)+",overall_metrics_file).group()
    pd_assessment.to_csv(os.path.join(overall_metrics_folder,"metrics.csv"))
    
    # plot statistical results
    figwidth=13;figheight=10
    subplot_left=0.06; subplot_right=0.97; subplot_top=0.95;subplot_bottom=0.06
    g=sns.catplot(data=pd_assessment,x='Sensor configurations',y='scores',col='metrics',col_wrap=2,kind='bar',hue='fields',height=3, aspect=0.8,sharey=False)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom,hspace=0.1, wspace=0.1)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    [ax.yaxis.grid(True) for ax in g.axes]
    g.savefig(overall_metrics_folder+ "/metrics.svg")

def check_model_test(training_folder):

    #1) load and normalize datasets for training and testing
    # load hyperparameters 
    hyperparams_file=training_folder+"/hyperparams.yaml"
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    else:
        print("Not Found hyper params file at {}".format(hyperparams_file))
        exit()

    #2) load dataset
    norm_subjects_trials_data,scaler = normalize_subjects_data(hyperparams)


    #2) subject and trials for testing
    subject_id_name = 'P_24_liziqing'
    trial='03'
    xy_test = norm_subjects_trials_data[subject_id_name][trial]

    # testing model
    features, labels, predictions, testing_folder = test_model(training_folder,xy_test,scaler)
    
    # plot testing results
    plot_prediction(features,labels,predictions,testing_folder)

        
if __name__=='__main__':
    #overall_metrics_file=deploy_sensor_combination()
    overall_metrics_file= os.path.join(RESULTS_PATH,"overall_metrics_results/2022-04-06/220449/testing_result_folders.txt")
    #display_testing_results(overall_metrics_file)
    
    training_folder = "/media/sun/My Passport/DropLanding_workspace/suntao/Results/Experimental_Results/models_parameters_results/2022-04-06/training_224518"
    check_model_test(training_folder)
