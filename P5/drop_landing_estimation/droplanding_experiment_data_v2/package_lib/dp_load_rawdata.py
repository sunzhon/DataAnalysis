"""
##
#
# Author: suntao
# Description: This module is to get drop landing experiment raw data into h5 format from equipment outputs incluing vicon and xsens
#
#
##
"""
import pandas as pd
import os
import numpy as np

import h5py
import csv
import numpy as np
import math
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
import re
import termcolor

import pdb
if __name__ == "__main__":
    import wearable_toolkit as wearable_toolkit
    import wearable_math as wearable_math
    from const import SEGMENT_DEFINITIONS, SUBJECTS, STATIC_TRIALS, DYNAMIC_TRIALS,TRIALS, SESSIONS, DATA_PATH, \
            SUBJECT_HEIGHT, SUBJECT_WEIGHT, SUBJECT_ID, TRIAL_ID, XSEN_IMU_ID, IMU_DATA_FIELDS, FORCE_DATA_FIELDS,\
            KNEE_DATA_FIELDS
else:
    import package_lib.wearable_toolkit as wearable_toolkit
    import package_lib.wearable_math as wearable_math
    from package_lib.const import SEGMENT_DEFINITIONS, SUBJECTS, STATIC_TRIALS, DYNAMIC_TRIALS,TRIALS, SESSIONS, DATA_PATH, \
            SUBJECT_HEIGHT, SUBJECT_WEIGHT, SUBJECT_ID, TRIAL_ID, XSEN_IMU_ID, IMU_DATA_FIELDS, FORCE_DATA_FIELDS,\
            KNEE_DATA_FIELDS

subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0)

 
class XsenReader():
    def __init__(self,subject_info,session):
        self.subject_name=subject_info.name
        self.xsen_data={}
        self.folder_path=os.path.join(DATA_PATH,self.subject_name,session)
        self.session_trial_exists=False
        for trial in TRIALS:
            ## Xsen file has no trial_type varaible, just trial_number
            #for trial_type in DYNAMIC_TRIALS:
            xsen_data_path = os.path.join(self.folder_path)
            if os.path.exists(xsen_data_path):
                print(xsen_data_path,trial)
                self.xsen_data[trial]=wearable_toolkit.XsenTxtReader(self.folder_path,self.subject_name,trial)
                self.session_trial_exists=(self.session_trial_exists or True)
            else:
                self.session_trial_exists=(self.session_trial_exists or False)


    def get_data_to_h5(self):
        # remove the exist h5 file
        h5format_dataset=os.path.join(self.folder_path,'features_rawdataset.hdf5')
        if os.path.exists(h5format_dataset):
            try:
                os.remove(h5format_dataset)
            except IOError:
                print("cannot remove h5 file")
    
        # save the h5 file
        with h5py.File(h5format_dataset, "w") as f:
            f.attrs['columns']=list(self.xsen_data['01'].data_frame.columns)
            subject_h5dataset=f.create_group(self.subject_name)
            for trial in TRIALS:
                subject_h5dataset.create_dataset(trial,data=self.xsen_data[trial].data_frame)



'''
Update: 2021/09/25

Read V3d data

AUthor: suntao

'''
class V3DReader():

    def __init__(self,subject_info,session):
        self.subject_name=subject_info.name
        v3d_calibrate_data_path = os.path.join(DATA_PATH, self.subject_name, session, self.subject_name+'static' + '.csv')
        self.v3d_data={}
        self.subject_name=subject_info.name
        self.folder_path=os.path.join(DATA_PATH,self.subject_name,session)
        self.session_trial_exists=False
        for trial in TRIALS:
            for trial_type in DYNAMIC_TRIALS:
                v3d_data_path = os.path.join(self.folder_path, self.subject_name + ' ' + trial_type +' '+ trial + '.csv')
                if os.path.exists(v3d_data_path):
                    print(v3d_data_path)
                    self.v3d_data[trial]=wearable_toolkit.Visual3dCsvReader(v3d_data_path,self.subject_name,trial)
                    self.session_trial_exists=(self.session_trial_exists or True)
                else:
                    self.session_trial_exists=(self.session_trial_exists or False)
            
        
    def get_data_to_h5(self):
        
        # remove the exist h5 file
        h5format_dataset=os.path.join(self.folder_path,'labels_rawdataset.hdf5')
        if os.path.exists(h5format_dataset):
            try:
                os.remove(h5format_dataset)
            except IOError:
                print("cannot remove h5 file")
    
        # save the h5 file
        with h5py.File(h5format_dataset, "w") as f:
            f.attrs['columns']=list(self.v3d_data['01'].data_frame.columns)
            sub=f.create_group(self.subject_name)
            for trial in TRIALS:
                sub.create_dataset(trial,data=self.v3d_data[trial].data_frame)
                #print(self.vicon_data[trial].data_frame.head())




class ViconReader():
    '''
    Update:
    Date: 2021/09/25

    '''
    def __init__(self,subject_info,session):
        self.subject_name=subject_info.name
        vicon_calibrate_data_path = os.path.join(DATA_PATH, self.subject_name, session, self.subject_name+'static' + '.csv')
        self.vicon_data={}
        self.subject_name=subject_info.name
        self.folder_path=os.path.join(DATA_PATH,self.subject_name,session)
        self.session_trial_exists=False
        for trial in TRIALS:
            for trial_type in DYNAMIC_TRIALS:
                vicon_data_path = os.path.join(self.folder_path, self.subject_name + ' '+trial_type +' '+ trial + '.csv')
                if os.path.exists(vicon_data_path):
                    print(vicon_data_path)
                    self.vicon_data[trial]=wearable_toolkit.ViconCsvReader(vicon_data_path, trial, vicon_calibrate_data_path, subject_info=subject_info)
                    self.session_trial_exists=(self.session_trial_exists or True)
                else:
                    self.session_trial_exists=(self.session_trial_exists or False)
            
        
    def get_data_to_h5(self):
        
        # remove the exist h5 file
        h5format_dataset=os.path.join(self.folder_path,'labels_rawdataset.hdf5')
        if os.path.exists(h5format_dataset):
            try:
                os.remove(h5format_dataset)
            except IOError:
                print("cannot remove h5 file")
    
        # save the h5 file
        with h5py.File(h5format_dataset, "w") as f:
            f.attrs['columns']=list(self.vicon_data['01'].data_frame.columns)
            sub=f.create_group(self.subject_name)
            for trial in TRIALS:
                sub.create_dataset(trial,data=self.vicon_data[trial].data_frame)
                #print(self.vicon_data[trial].data_frame.head())



'''
Save each subject experiment data into two h5 data fromat (features and labels)

'''
def transfer_rawdata_to_h5():
    for subject in SUBJECTS: # subjects
        for session in SESSIONS:# trial types
            #- load subject information
            print("Subject {}, Session {}".format(subject,session))
            subject_info = subject_infos.loc[subject, :]

            #- read vicon data
            vicon=ViconReader(subject_info,session)

            #- read v3d data
            v3d=V3DReader(subject_info, re.sub('vicon','v3d',session))

            #- process IMU data
            xsen=XsenReader(subject_info,re.sub('vicon','xsen',session))

            #-- 1) Assign v3d and xsen data by crop v3d and xsen data. 
            #-- 2) Extract drop landing data of v3d, vicon and xsen. Because their effective periods
            #-- are not same, the vicon data waw croped when process it in Nexus. The v3d data and vicon data has same 
            #--  frame start and end
            for trial in TRIALS:
                if(vicon.session_trial_exists):
                    frame_start=vicon.vicon_data[trial].frame_start
                    frame_end=vicon.vicon_data[trial].frame_end
                    #-- crop xsen to assign it with vicon and v3d dataset
                    v3d.v3d_data[trial].crop()
                    xsen.xsen_data[trial].crop(frame_start,frame_end)
                    #-- extract drop landing period
                    v3d.v3d_data[trial].extract_droplanding_period()
                    xsen.xsen_data[trial].extract_droplanding_period()

            #-- save their data into a h5 file
            if(xsen.session_trial_exists):
                xsen.get_data_to_h5()
            if(v3d.session_trial_exists):
                v3d.get_data_to_h5()
            if(vicon.session_trial_exists):
                vicon.get_data_to_h5()



'''
Save all subjects' features and labels h5 format data into a h5 data format file: features_labels_rawdatasets.hdf5

'''
def transfer_allsubject_to_a_h5():
    # declare h5 file
    h5format_dataset=os.path.join(DATA_PATH,"features_labels_rawdatasets.hdf5")
    
    # remove the exist h5 file
    if os.path.exists(h5format_dataset):
        try:
            os.remove(h5format_dataset)
        except IOError:
            print("cannot remove h5 file")
            
    # save the h5 file
    with h5py.File(h5format_dataset, "w") as f:
        features_data={}
        for subject in SUBJECTS:
            sub=f.create_group(subject)
            f.attrs['subjects']=SUBJECTS
            for session in SESSIONS:
                features_path=os.path.join(DATA_PATH,subject,re.sub('vicon','xsen',session),'features_rawdataset.hdf5')
                labels_path=os.path.join(DATA_PATH,subject,re.sub('vicon','v3d',session),'labels_rawdataset.hdf5')
                if(os.path.exists(features_path) and os.path.exists(labels_path)):
                    print("combine feature and labels hdf5 datasets together at: ",h5format_dataset)
                    try:
                        with h5py.File(features_path,'r') as ff:
                            with h5py.File(labels_path,'r') as fl:
                                for trial in TRIALS:
                                    # - combine features and labels along with columns
                                    if(pd.DataFrame(ff[subject][trial]).shape[0]==pd.DataFrame(fl[subject][trial]).shape[0]):
                                        features_labels=pd.concat([pd.DataFrame(ff[subject][trial]),pd.DataFrame(fl[subject][trial])],axis=1)
                                        sub.create_dataset(trial,data=features_labels)
                                    else:
                                        print(termcolor.colored("subject: {} in trial:{} features anad lables have different rows".format(subject,trial),"red"))
                                        pdb.set_trace()
                                # set columns as attributes of the hdf5 file dataset
                                sub.attrs['columns']=list(ff.attrs['columns'])+list(fl.attrs['columns'])
                    except Exception as e: 
                        print(e)
                        print(termcolor.colored("Subject: {} h5 file path in session: {} is wrong".format(subject,session),'red'))
                        pdb.set_trace()
                        

"""
配置文件在const 中

"""
    
if __name__=="__main__":
    transfer_rawdata_to_h5()
    transfer_allsubject_to_a_h5()
                
