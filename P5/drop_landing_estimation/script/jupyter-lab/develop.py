#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("./../")




## %load ./../rnn_model.py
#!/usr/bin/env python
'''
 Import necessary packages

'''
import tensorflow as tf
# set hardware config
#tf.debugging.set_log_device_placement(True)

cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

# set gpu memory grouth automatically
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

if(gpus!=[]):
    # set virtal gpu/ logical gpu, create four logical gpu from a physical gpu (gpus[0])
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072),
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072),
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072),
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)
        ]
        )

logical_cpus = tf.config.experimental.list_logical_devices(device_type='CPU')
logical_gpus = tf.config.experimental.list_logical_devices(device_type='GPU')
print('physical cpus and gpus: ',cpus, gpus)
print('physical cpus number: ', len(cpus))
print('physical cpgs number: ', len(gpus))
print('logical cpus and gpus: ',logical_cpus, logical_gpus)
print('logical cpgs number: ', len(logical_gpus))



import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import pandas as pd
import yaml
import h5py
print("tensorflow version:",tf.__version__)
import vicon_imu_data_process.process_rawdata as pro_rd
import estimation_assessment.scores as es_as
import estimation_assessment.visualization as es_vl

import seaborn as sns
import copy
import re
import json

from vicon_imu_data_process.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH, TRAIN_USED_TRIALS
from vicon_imu_data_process.const import DROPLANDING_PERIOD, RESULTS_PATH
from vicon_imu_data_process import const
from vicon_imu_data_process.dataset import *


from estimation_models.rnn_models import *
from estimation_study import *


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import time as localtimepkg


# In[ ]:


'''
This function investigate the estimation metrics by 
testing different sensor configurations and model LSTM layer size

'''

def integrative_investigation(investigation_variables, prefix_name=''):

    #1) paraser investigation variables
    sensor_configurations = investigation_variables['sensor_configurations']
    lstm_units  = investigation_variables['lstm_units']

    if('syn_features_labels' in investigation_variables.keys()):
        syn_features_labels = investigation_variables['syn_features_labels']
    else:
        syn_features_labels = [True] # default value is true, to synchorinize features and labels using event

    estimated_variables = investigation_variables['estimated_variables']

    landing_manners = investigation_variables['landing_manners']



    #2) train and test model
    combination_investigation_info = []
    
    #i) sensor configurations
    for sensor_configuration_name, sensor_list in sensor_configurations.items():
        # features fields based on sensors
        features_fields = const.extract_imu_fields(sensor_list, const.ACC_GYRO_FIELDS)
        
        #ii) init hyper params with different labels
        for labels_fields in estimated_variables:
            hyperparams = initParameters(labels_names=labels_fields, features_names=features_fields)
        
            #iii) model size configuations
            for lstm_unit in lstm_units:
                hyperparams['lstm_units'] = lstm_unit
                hyperparams['sensor_configurations'] = sensor_configuration_name
            
                #iv) synchronization state
                for syn_state in syn_features_labels:
                    
                    #i) landing manners: single-leg or double-leg drop landing
                    for landing_manner in landing_manners:
                        if landing_manner == 'single_leg_R':
                            hyperparams['target_leg']='R'
                        else:
                            hyperparams['target_leg'] = 'L'# left(L), right(R) or double
                        hyperparams['landing_manner'] = landing_manner # single or double legs
            
                        # train and test model
                        print("#**************************************************************************#")
                        print("Sensor configuration: {}; LSTM size: {}".format(sensor_configuration_name, lstm_unit))
            
                        # do training and testing
                        training_testing_folders, xy_test, scaler =  train_test_loops(
                            hyperparams, fold_number=1, test_multil_trials=False, 
                            syn_features_labels=syn_state, landing_manner=landing_manner)# model traning
            
                        # list testing folders 
                        a_single_config = [sensor_configuration_name, str(lstm_unit), str(syn_state), landing_manner, labels_fields, training_testing_folders]
                        a_single_config_columns = '\t'.join(['Sensor configurations',
                                                    'LSTM units',
                                                    'syn_features_labels',
                                                    'landing_manners',
                                                    'estimated_variables',
                                                    'training_testing_folders']) + '\n'
                        combination_investigation_info.append(a_single_config)

    #3) create folders to save testing folders
    combination_investigation_testing_folders = os.path.join(RESULTS_PATH,"investigation",
                                         str(localtimepkg.strftime("%Y-%m-%d",localtimepkg.localtime())),
                                         str(localtimepkg.strftime("%H%M%S", localtimepkg.localtime())))
    if(not os.path.exists(combination_investigation_testing_folders)):
        os.makedirs(combination_investigation_testing_folders)
    
    #4) save testing folders
    combination_testing_folders = os.path.join(combination_investigation_testing_folders, prefix_name + "testing_result_folders.txt")
    if(os.path.exists(combination_testing_folders)==False):
        with open(combination_testing_folders,'a') as f:
            f.write(a_single_config_columns)
            for single_investigation_info in combination_investigation_info:
                train_test_results = single_investigation_info[-1] # the last elements is a list
                for idx, testing_folder in enumerate(train_test_results["testing_folder"]): # in a loops which has many train and test loop
                    # a single investigation config info and its results
                    single_investigation_info_results = single_investigation_info[:-1] + [testing_folder]
                    #print('single investigation info and results:', single_investigation_info_results)
                    # transfer into strings with '\t' seperator
                    str_single_investigation_info_results = '\t'.join([str(i) for i in single_investigation_info_results])
                    # save config and its results
                    f.write(str_single_investigation_info_results +'\n')
                                                    
    print("INESTIGATION DONE!")
    return combination_testing_folders


# ## Perform investigation by training model



#1) The variables that are needed to be investigate
investigation_variables={
    "sensor_configurations":
                            {
                               'F': ['L_FOOT'],
                               'S': ['L_SHANK'],
                               'T': ['L_THIGH'],
                               'W': ['WAIST'],
                               'C': ['CHEST'],
                                
                               'FS': ['L_FOOT','L_SHANK'],
                               'FT': ['L_FOOT','L_THIGH'],
                               'FW': ['L_FOOT','WAIST'],
                               'FC': ['L_FOOT','CHEST'],
                               'ST': ['L_SHANK','L_THIGH'],
                               'SW': ['L_SHANK','WAIST'],
                               'SC': ['L_SHANK','CHEST'],
                               'TW': ['L_THIGH','WAIST'], 
                               'TC': ['L_THIGH', 'CHEST'],
                               'WC': ['WAIST', 'CHEST'],
                               
                                
                               'FST': ['L_FOOT','L_SHANK','L_THIGH'], 
                               'FSW': ['L_FOOT','L_SHANK','WAIST'],
                               'FSC': ['L_FOOT','L_SHANK','CHEST'],
                                
                               'FTW': ['L_FOOT','L_THIGH','WAIST'],
                               'FTC': ['L_FOOT','L_THIGH','CHEST'],
                               
                               'FWC': ['L_FOOT','WAIST', 'CHEST'],
                                
                               'STW': ['L_SHANK','L_THIGH','WAIST' ],
                               'STC': ['L_SHANK','L_THIGH','CHEST' ],
                               'SWC': ['L_SHANK','WAIST','CHEST' ],
                               'TWC': ['L_THIGH','WAIST', 'CHEST'],
                                
                               'FSTW': ['L_FOOT','L_SHANK','L_THIGH','WAIST'], 
                               'FSTC': ['L_FOOT','L_SHANK','L_THIGH','CHEST'], 
                               'FSWC': ['L_FOOT','L_SHANK','WAIST', 'CHEST'],
                               'FTWC': ['L_FOOT','L_THIGH','WAIST', 'CHEST'],
                               'STWC': ['L_SHANK','L_THIGH','WAIST', 'CHEST'],
                                
                               'FSTWC': ['L_FOOT','L_SHANK','L_THIGH','WAIST', 'CHEST']
                              },
    
    "sensor_configurations": {'FSTWC': ['L_FOOT','L_SHANK','L_THIGH','WAIST', 'CHEST']},
    "syn_features_labels": [False],
    #'estimated_variables': [['L_KAM_X'], ['L_GRF_Z']],
    "estimated_variables": [['L_KNEE_MOMENT_X']],
    #"landing_manners": ['single_leg_R', 'double_legs']
    "landing_manners": ['single_leg_R'],

    #"lstm_units": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    #"lstm_units": [15, 20, 25, 30, 35]
    #"lstm_units": [5, 10]
    "lstm_units": [35]


}


# investigate model
combination_investigation_results = integrative_investigation(investigation_variables,'off_on_synchronization')

print(combination_investigation_results)


# ## exit machine and save environment

#os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node -save -name suntao_env")

