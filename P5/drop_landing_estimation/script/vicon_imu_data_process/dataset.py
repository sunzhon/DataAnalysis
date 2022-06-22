#Python

"""
Description:
    This module transfer normalized raw data into dataset for training and testing
        
Author: Sun Tao
Email: suntao.hn@gmail.com
Date: 2022-04-09

"""
import pandas as pd
import os
import h5py
import re

import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from matplotlib import gridspec

import yaml
import pdb
import re
import warnings
import termcolor
import matplotlib._color_data as mcd

import datetime


if __name__ == "__main__":
    import wearable_toolkit as wearable_toolkit
    import wearable_math as wearable_math
    from const import SEGMENT_DEFINITIONS, SUBJECTS, STATIC_TRIALS, DYNAMIC_TRIALS,TRIALS, SESSIONS, DATA_PATH, \
            SUBJECT_HEIGHT, SUBJECT_WEIGHT, SUBJECT_ID, TRIAL_ID, XSEN_IMU_ID, IMU_DATA_FIELDS, FORCE_DATA_FIELDS,\
            KNEE_DATA_FIELDS, WRONG_TRIALS
else:
    from vicon_imu_data_process.process_rawdata import load_normalize_data
    import vicon_imu_data_process.wearable_toolkit as wearable_toolkit
    import vicon_imu_data_process.wearable_math as wearable_math
    from vicon_imu_data_process.const import SEGMENT_DEFINITIONS, SUBJECTS, STATIC_TRIALS, DYNAMIC_TRIALS,TRIALS, SESSIONS, DATA_PATH, \
            SUBJECT_HEIGHT, SUBJECT_WEIGHT, SUBJECT_ID, TRIAL_ID, XSEN_IMU_ID, IMU_DATA_FIELDS, FORCE_DATA_FIELDS,\
            KNEE_DATA_FIELDS, WRONG_TRIALS



'''
Packing data into windows 


'''

def windowed_dataset(series, hyperparams,shuffle_buffer=1000):
    window_size = hyperparams['window_size']
    batch_size = hyperparams['batch_size']
    shift_step = hyperparams['shift_step']
    labels_num = hyperparams['labels_num']
    
    #series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    # window size defines the time sequence of the input
    ds = ds.window(window_size, shift=shift_step, stride=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    #ds = ds.shuffle(shuffle_buffer) # when using LSTM (RNN), we cannot use shuffter, since it changes initial states due RNN having memory
    ds = ds.map(lambda w: (w[:,:-labels_num], w[:,-labels_num:]))
    ds = ds.batch(batch_size).prefetch(1)
    #print(list(ds.as_numpy_iterator())[0])

    # The ds has shape (batch_num, window_batch_num, window_size, festures_num)
    return ds



'''
Split datasets of subjects_trials_data into train, valid and test set

'''

def split_dataset(norm_subjects_trials_data, train_subject_indices, test_subject_indices, hyperparams, model_type='tf_keras', test_multi_trials=False):
    subjects_trials = hyperparams['subjects_trials']
    subject_ids_names = list(subjects_trials.keys())

    #i) subjects for train and test
    train_subject_ids_names = [subject_ids_names[subject_idx] for subject_idx in train_subject_indices]
    test_subject_ids_names = [subject_ids_names[subject_idx] for subject_idx in test_subject_indices]

    #ii) decide train and test subject dataset 
    print("Train subject set:", train_subject_ids_names, "Test subject set:", test_subject_ids_names)
    hyperparams['train_subject_ids_names'] = train_subject_ids_names
    hyperparams['test_subject_ids_names'] = test_subject_ids_names

    #iii) data from train and test subjects and their trials
    xy_train = [norm_subjects_trials_data[subject_id_name][trial] for subject_id_name in train_subject_ids_names for trial in subjects_trials[subject_id_name]]
    xy_valid = [norm_subjects_trials_data[subject_id_name][trial] for subject_id_name in test_subject_ids_names for trial in subjects_trials[subject_id_name]]

    #iV) test dataset
    if(test_multi_trials):
        xy_test = xy_valid
    else:
        xy_test = xy_valid[0]
        
    if(model_type=='tf_keras'):
        #iv) concate data of several trials into a numpy arrary
        xy_train = np.concatenate(xy_train, axis=0)
        xy_valid = np.concatenate(xy_valid, axis=0)
        print("Train set shape", xy_train.shape)
        print("Valid set shape", xy_valid.shape)
        
        #v) window train and test dataset
        train_set = windowed_dataset(xy_train, hyperparams)
        valid_set = windowed_dataset(xy_valid, hyperparams)

        return train_set, valid_set, xy_test

    elif(model_type=='sklearn'):
        return xy_train, xy_valid, xy_test

    else:
        print("MODEL TYPE IS WRONG")
        exit()
        


