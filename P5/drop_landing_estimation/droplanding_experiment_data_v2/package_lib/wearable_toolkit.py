"""
wearalbe_toolkit.py

@Author: Dianxin

This package is used to preprocess data collected from walking trials.
Read v3d exported csv data, sage csv data, vicon exported csv data and openPose exported csv data.
Synchronize vicon data and sage data.

"""
import csv
import numpy as np
import math
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from scipy.signal import convolve2d


from const import IMU_SENSOR_LIST, IMU_FIELDS,EXT_KNEE_MOMENT, TARGETS_LIST, SUBJECT_HEIGHT, \
    EVENT_COLUMN,  V3D_DATA_FIELDS, V3D_LABELS_FIELDS,IMU_FEATURES_FIELDS, IMU_DATA_FIELDS
from const import SUBJECT_WEIGHT, STANCE, STANCE_SWING, STEP_TYPE, VIDEO_ORIGINAL_SAMPLE_RATE, FORCE_PLATE_DATA_FIELDS, FORCE_DATA_FIELDS, DROPLANDING_PERIOD
import wearable_math as wearable_math
import pdb
import os, re
import warnings
import termcolor



class VideoCsvReader:
    """
    read video exported csv file by openPose.
    """

    def __init__(self, file_path):
        self.data_frame = pd.read_csv(file_path, index_col=0)

    def get_column_position(self, marker_name):
        return self.data_frame[marker_name]

    def get_rshank_angle(self):
        ankle = self.data_frame[['RAnkle_x', 'RAnkle_y']]
        knee = self.data_frame[['RKnee_x', 'RKnee_y']]
        vector = knee.values - ankle.values
        return np.arctan2(-vector[:, 1], vector[:, 0])

    def fill_low_probability_data(self):
        columns_label = self.data_frame.columns.values.reshape([-1, 3]).tolist()
        for x, y, probability in columns_label:
            self.data_frame.loc[self.data_frame[probability] < 0.5, [x, y, probability]] = np.nan
        self.data_frame = self.data_frame.interpolate(method='linear', axis=0)

    def low_pass_filtering(self, cut_off_fre, sampling_fre, filter_order):

        # plt.figure()
        # plt.plot(self.data_frame['RKnee_x'])
        # plt.plot(data_filter(self.data_frame['RKnee_x'], 15, 100, 2))
        # plt.plot(data_filter(self.data_frame['RKnee_x'], 10, 100, 2))
        # plt.show()

        self.data_frame.loc[:, :] = data_filter(self.data_frame.values, cut_off_fre, sampling_fre, filter_order)

    def resample_to_100hz(self):
        target_sample_rate = 100.
        x, step = np.linspace(0., 1., self.data_frame.shape[0], retstep=True)
        # new_x = np.linspace(0., 1., int(self.data_frame.shape[0]*target_sample_rate/VIDEO_ORIGINAL_SAMPLE_RATE))
        new_x = np.arange(0., 1., step*VIDEO_ORIGINAL_SAMPLE_RATE/target_sample_rate)
        f = interp1d(x, self.data_frame, axis=0)
        self.data_frame = pd.DataFrame(f(new_x), columns=self.data_frame.columns)

    def crop(self, start_index):
        # keep index after start_index
        self.data_frame = self.data_frame.loc[start_index:]
        self.data_frame.index = range(self.data_frame.shape[0])


class Visual3dCsvReader:
    """
    read v3d export data. It should contain LEFT_KNEE_MOMENT,LEFT_KNEE_ANGLE etc.
    """
    TRUE_EVENT_INDEX = 0

    def __init__(self, file_path,subject_name,trial):
        # 读取V3D 输出数据
        self.file_path = file_path
        self.subject_name = subject_name
        self.trial = trial
        self.data = pd.read_csv(file_path, delimiter='\t', header=1, skiprows=[2, 3, 4])
        self.data = self.data.fillna(0)
        
        # extrect data specified by V3D_DATA_FIELDS
        actual_v3d_data_fields=[ss for ss in self.data.columns]
        try:
            if actual_v3d_data_fields==V3D_DATA_FIELDS:
                self.data=self.data[V3D_DATA_FIELDS]
                self.data.columns=V3D_LABELS_FIELDS
            else:
                exist_fields, unexist_fields = [],[]
                for fields in V3D_DATA_FIELDS:
                    if(fields in actual_v3d_data_fields):
                        exist_fields.append(fields)
                    else:
                        unexist_fields.append(fields)
                        print(termcolor.colored("V3D export data has less data feilds:",'yellow'),fields)
                        self.data.insert(self.data.shape[1],fields,0)
                self.data=self.data.reindex(columns=V3D_DATA_FIELDS)
                self.data.columns=V3D_LABELS_FIELDS
        except ValueError:
            pdb.set_trace()

        # 提取标签数据
        self.data_frame = self.data[V3D_LABELS_FIELDS].fillna(0)
        self.data_frame.columns = V3D_LABELS_FIELDS
        self.data_frame.index=range(self.data_frame.shape[0])

        row_num=self.data_frame.shape[0]
        if('FP' in V3D_DATA_FIELDS):# 如果是原始力数据，则需要求10帧的平均，且力数据和模型计算数据没有对其
            # moving average 10 sub frames and just pick the average of the 10 sub frames
            print('moving average force data')
            self.data_frame.loc[:int(row_num/10-1),FORCE_DATA_FIELDS]=self.data_frame[FORCE_DATA_FIELDS].rolling(window=10).mean().values[range(10-1,row_num,10),:]
            self.data_frame=self.data_frame.loc[:int(row_num/10-1)] # 截取前面有效的行数，对齐模型计算数据和力数据
        #convolve2d(self.data_frame[FORCE_PALTE_FEILDS],filter_weight,'valid')
        print('V3d raw data shape:', self.data.shape, 'raw data frame shape:', self.data_frame.shape)
        if(self.data.shape[0]==0):
            print(termcolor.colored("V3D export data is wrong, please check vicon data process:",'red'),fields)
            pdb.set_trace()

    def crop(self, start_index=0, end_index=None):
        '''
        Get valid experiment data, this should be excuted before extract_droplanding_perio()

        '''
        if end_index==None:
            end_index=self.data_frame.shape[0]

        # keep index after start_index
        try:
            self.data = self.data.loc[start_index:end_index]
        except Exception as e:
            print(e)
            pdb.set_trace()

        self.data.index = range(self.data.shape[0])

        self.data_frame = self.data_frame.loc[start_index:end_index]
        self.data_frame.index = range(self.data_frame.shape[0])
        print('v3d croped data frame shape:', self.data_frame.shape)

    def extract_droplanding_period(self):
        #- extract drop landing period
        #-- get touch moment, this is determined by the foot who touchs ground first
        left_touch_moment=self.data['LON'][0]
        right_touch_moment=self.data['RON'][0]

        # if there is no touch moment, then set it to a big value , eg. 1000000
        left_touch_moment=left_touch_moment if left_touch_moment >1 else 1000000
        right_touch_moment=right_touch_moment if right_touch_moment >1 else 1000000
        try:
            combined_touch_moment=int(min([left_touch_moment,right_touch_moment]))
            if(combined_touch_moment==0) or (combined_touch_moment==1000000):
                print(termcolor.colored("The trial has no right touch moment",'red'))
        except Exception as e:
            print(e)
            pdb.set_trace()

        print('v3d touch moment: {}'.format(combined_touch_moment))

        #-- determine start_index and end_index, the drop landing
        start_index=combined_touch_moment-DROPLANDING_PERIOD/4
        end_index=combined_touch_moment+DROPLANDING_PERIOD/4*3-1


        #-- check row_range is in start_index and end_index
        row_length=self.data_frame.shape[0]
        if((start_index<0) or (end_index>row_length)):
            print(termcolor.colored("Trial: {} of subject: {} extract period is out the effective data period".format(self.trial,self.subject_name),'red'))
            pdb.set_trace()

        #-- extract drop landing period data
        self.data_frame=self.data_frame.loc[start_index:end_index]
        if(self.data_frame.shape[0]!=DROPLANDING_PERIOD):
            print("data_frame row is less than sepcified DROPLANDING_PERIOD")
            pdb.set_trace()

        print('v3d extracted data frame shape:', self.data_frame.shape)



    def create_step_id(self, step_type):
        [LOFF, LON, ROFF, RON] = [self.data[event].dropna().values.tolist() for event in ['LOFF', 'LON', 'ROFF', 'RON']]

        events_dict = {'ROFF': ROFF, 'RON': RON, 'LOFF': LOFF, 'LON': LON}
        # Filter events_dict
        for _, frames in events_dict.items():
            for i in range(len(frames) - 1, -1, -1):
                if abs(frames[i] - frames[i - 1]) < 10:
                    frames.pop(i)





class XsenTxtReader():
    def __init__(self,folder_path, subject_name, trial):
        self.folder_path=folder_path
        self.trial=trial
        self.subject_name=subject_name

        is_verbose=False
        #- read txt data of a trial which has eight txt files
        file_name='MT_'+XSEN_IMU_ID['MASTER']+'_0'+trial+'-000_'+XSEN_IMU_ID['CHEST']+'.txt'
        file_path=os.path.join(folder_path,file_name)
        if is_verbose:
            print(file_path)
        chest_txt=np.loadtxt(file_path,comments='//',skiprows=6,dtype=float)
        chest_txt=self.interp_data(chest_txt)


        file_name='MT_'+XSEN_IMU_ID['MASTER']+'_0'+trial+'-000_'+XSEN_IMU_ID['WAIST']+'.txt'
        file_path=os.path.join(folder_path,file_name)
        if is_verbose:
            print(file_path)
        waist_txt=np.loadtxt(file_path,comments='//',skiprows=6,dtype=float)
        waist_txt=self.interp_data(waist_txt)

        file_name='MT_'+XSEN_IMU_ID['MASTER']+'_0'+trial+'-000_'+XSEN_IMU_ID['L_THIGH']+'.txt'
        file_path=os.path.join(folder_path,file_name)
        if is_verbose:
            print(file_path)
        l_thigh_txt=np.loadtxt(file_path,comments='//',skiprows=6,dtype=float)
        l_thigh_txt=self.interp_data(l_thigh_txt)
        
        
        file_name='MT_'+XSEN_IMU_ID['MASTER']+'_0'+trial+'-000_'+XSEN_IMU_ID['L_SHANK']+'.txt'
        file_path=os.path.join(folder_path,file_name)
        if is_verbose:
            print(file_path)
        l_shank_txt=np.loadtxt(file_path,comments='//',skiprows=6,dtype=float)
        l_shank_txt=self.interp_data(l_shank_txt)


        file_name='MT_'+XSEN_IMU_ID['MASTER']+'_0'+trial+'-000_'+XSEN_IMU_ID['L_FOOT']+'.txt'
        file_path=os.path.join(folder_path,file_name)
        if is_verbose:
            print(file_path)
        l_foot_txt=np.loadtxt(file_path,comments='//',skiprows=6,dtype=float)
        l_foot_txt=self.interp_data(l_foot_txt)
        
        file_name='MT_'+XSEN_IMU_ID['MASTER']+'_0'+trial+'-000_'+XSEN_IMU_ID['R_THIGH']+'.txt'
        file_path=os.path.join(folder_path,file_name)
        if is_verbose:
            print(file_path)
        r_thigh_txt=np.loadtxt(file_path,comments='//',skiprows=6,dtype=float)
        r_thigh_txt=self.interp_data(r_thigh_txt)
        
        
        file_name='MT_'+XSEN_IMU_ID['MASTER']+'_0'+trial+'-000_'+XSEN_IMU_ID['R_SHANK']+'.txt'
        file_path=os.path.join(folder_path,file_name)
        if is_verbose:
            print(file_path)
        r_shank_txt=np.loadtxt(file_path,comments='//',skiprows=6,dtype=float)
        r_shank_txt=self.interp_data(r_shank_txt)


        file_name='MT_'+XSEN_IMU_ID['MASTER']+'_0'+trial+'-000_'+XSEN_IMU_ID['R_FOOT']+'.txt'
        file_path=os.path.join(folder_path,file_name)
        if is_verbose:
            print(file_path)
        r_foot_txt=np.loadtxt(file_path,comments='//',skiprows=6,dtype=float)
        r_foot_txt=self.interp_data(r_foot_txt)


        # combine the eight IMU data into a csv file
        row_num=min(chest_txt.shape[0], waist_txt.shape[0],
                    l_thigh_txt.shape[0],l_shank_txt.shape[0],l_foot_txt.shape[0],
                    r_thigh_txt.shape[0],r_shank_txt.shape[0],r_foot_txt.shape[0]
                   )
        
        try:
            all_imu_data=np.hstack((chest_txt[:row_num,:], waist_txt[:row_num,1:]))
            all_imu_data=np.hstack((all_imu_data,l_thigh_txt[:row_num,1:]))
            all_imu_data=np.hstack((all_imu_data,l_shank_txt[:row_num,1:]))
            all_imu_data=np.hstack((all_imu_data,l_foot_txt[:row_num,1:]))
            all_imu_data=np.hstack((all_imu_data,r_thigh_txt[:row_num,1:]))
            all_imu_data=np.hstack((all_imu_data,r_shank_txt[:row_num,1:]))
            all_imu_data=np.hstack((all_imu_data,r_foot_txt[:row_num,1:]))
        except:
            pdb.set_trace()


        # butterworth filter
        filtered_all_imu_data=data_filter(all_imu_data, 10, 100, filter_order=2)
        pd_all_imu_data=pd.DataFrame(data=filtered_all_imu_data,columns=IMU_DATA_FIELDS)

        csv_path=os.path.join(folder_path,'features'+'_trial_'+trial+'.csv')
        if is_verbose:
            print(csv_path)
        pd_all_imu_data.to_csv(csv_path,index=False)
        
        self.data=pd_all_imu_data
        self.data_frame=self.data.loc[:,IMU_FEATURES_FIELDS] #read necessary columns
        self.data_frame.columns=IMU_FEATURES_FIELDS
        self.data_frame.index=range(self.data_frame.shape[0])

        print('xsen raw data shape:', self.data.shape, 'raw data frame shape:', self.data_frame.shape)

        '''
        warnings.warn("interpolate rows in IMU data, Please delete this in formal satus")
        extend_row=int(round(self.data_frame.shape[0]/40*100))
        self.data_frame.index=[int(x) for x in np.linspace(0,extend_row,num=self.data_frame.shape[0])]
        self.data_frame=self.data_frame.reindex([ int(x) for x in np.linspace(0,extend_row,num=extend_row)])
        self.data_frame.interpolate(inplace=True)
        '''






    def crop(self, start_index, end_index):
        '''

        crop the valid experiment data, this should be excuted before extract_droplanding_period()

        '''

        # keep index after start_index
        self.data = self.data.loc[start_index:end_index]
        self.data.index = range(self.data.shape[0])

        self.data_frame = self.data_frame.loc[start_index:end_index]
        self.data_frame.index = range(self.data_frame.shape[0])
        print('xsen  croped data frame shape:', self.data_frame.shape)


    def extract_droplanding_period(self):
        
        #-v3d file in a same session
        list_v3d_files=os.listdir(re.sub('xsen','v3d',self.folder_path))
        v3d_file=[x for x in list_v3d_files if self.trial+'.csv' in x][0]
        print("v3d file: {} for getting touch momemnt".format(v3d_file))
        v3d_file_path=os.path.join(re.sub('xsen','v3d', self.folder_path), v3d_file)
        v3d_data = pd.read_csv(v3d_file_path, delimiter='\t', header=1, skiprows=[2, 3, 4])
        v3d_data = v3d_data.fillna(0)

        #- extract drop landing period
        left_touch_moment=v3d_data['LON'][0]
        right_touch_moment=v3d_data['RON'][0]
        #-- get touch moment, this is determined by the foot who touchs ground first
        # NOTE: left_touch_moment and right_touch_moment are global variables, which are write by Visaul3D CsvReader,
        # So this class should be instanlizated after Visual3DCsvReader

        # if there is no touch moment, then set it to a big value , eg. 1000000
        left_touch_moment=left_touch_moment if left_touch_moment >1 else 1000000
        right_touch_moment=right_touch_moment if right_touch_moment >1 else 1000000
        try:
            combined_touch_moment=int(min([left_touch_moment,right_touch_moment]))
            if(combined_touch_moment==0) or (combined_touch_moment==1000000):
                print(termcolor.colored("The trial has no right touch moment",'red'))
                pdb.set_trace()
        except:
            pdb.set_trace()
        print('xsen touch moment: {}'.format(combined_touch_moment))

        #-- determine start_index and end_index, the drop landing
        start_index=combined_touch_moment-DROPLANDING_PERIOD/4 
        end_index=combined_touch_moment+DROPLANDING_PERIOD/4*3 - 1

        #-- check row_range is in start_index and end_index
        row_length=self.data_frame.shape[0]
        if((start_index<0) or (end_index>row_length)):
            print(termcolor.colored("Trial: {} of subject: {} extract period is out the effective data period".format(self.trial,self.subject_name),'red'))
            pdb.set_trace()
        
        #-- extract drop landing period data
        self.data_frame=self.data_frame.loc[start_index:end_index]
        if(self.data_frame.shape[0]!=DROPLANDING_PERIOD):
            print("data_frame row is less than sepcified DROPLANDING_PERIOD")
            pdb.set_trace()


        print('xsen  extracted data frame shape:', self.data_frame.shape)


    def interp_data(self,a_imu_txt):
        if(a_imu_txt.shape[0]==0):
            print(termcolor.colored("subject:{} in trial: {} loss IMU data".format(self.subject_name, self.trial),'red'))
            self.data_loss=True
            exit()
        # reindex the packacge counter (The first column)
        counter=a_imu_txt[:,0]
        reindex_counter=counter-counter[0] # the first countr number is set to 0
        reindex_counter=np.hstack((reindex_counter[reindex_counter>=0],reindex_counter[reindex_counter<0]+65535+1)) # 65535 is the max counter value
        a_imu_txt[:,0]=reindex_counter
        new_a_imu_txt=list(a_imu_txt.T)
        
        # interpolate package where they loss
        if(a_imu_txt.shape[0]<a_imu_txt[-1,0]):
            print(termcolor.colored("Trial:{} loss package".format(self.trial),"yellow"))
            nonexist_packages=[]
            exist_flag=False
            for package_idx in range(int(a_imu_txt[-1,0])):# The necessary package number
                if(package_idx in reindex_counter): # reindex_counter, actual package
                    exist_flag=True
                else:
                    exist_flag=False
                    nonexist_packages.append(package_idx)
                if ((exist_flag==True) and (len(nonexist_packages)!=0)):
                    for col_idx in range(0,a_imu_txt.shape[1]):
                        before_nonexist_package = nonexist_packages[0]-1 # 插值采样开始点
                        after_nonexist_package  = nonexist_packages[-1]+1# 插值采样结束点
                        start_index = np.where(reindex_counter==before_nonexist_package)[0][0]
                        end_index = np.where(reindex_counter==after_nonexist_package)[0][0]
                        print(nonexist_packages,termcolor.colored("ssssssssssssss",'red'))
                        fx=np.array([before_nonexist_package,after_nonexist_package])
                        fy=a_imu_txt[[start_index,end_index],col_idx]
                        temp=np.interp(nonexist_packages,fx,fy)
                        new_a_imu_txt[col_idx]=np.insert(new_a_imu_txt[col_idx],nonexist_packages[0],temp)
                    nonexist_packages=[]

        try:
            ret=np.array(new_a_imu_txt).T
        except Exception as e:
            print(e)
            pdb.set_trace()

        # round package_number
        ret[:,0]=np.round(ret[:,0])
        return ret

        

        
        
        

[PARENT_TITLE, SAMPLE_RATE, TITLE, DIRECTION, UNITS, DATA] = range(6)


from const import SEGMENT_DEFINITIONS, SUBJECTS, STATIC_TRIALS, TRIALS, SESSIONS, DATA_PATH, \
SUBJECT_HEIGHT, SUBJECT_WEIGHT, SUBJECT_ID, TRIAL_ID, XSEN_IMU_ID, IMU_DATA_FIELDS, FORCE_DATA_FIELDS
class ViconCsvReader:
    '''
    Read the csv files exported from vicon.
    The csv file should only contain Device and Model Outputs information
    '''

    # if static_trial is not none, it's used for filling missing data.
    def __init__(self, file_path,trial=None, segment_definitions=None, static_trial=None, subject_info=None):
        self.data_dict, self.sample_rate = ViconCsvReader.reading(file_path)
        self.trial=trial
        self.file_path=file_path
        '''
        # create segment marker data
        self.segment_data = dict()
        if segment_definitions is None:
            segment_definitions = {}
        for segment, markers in segment_definitions.items():
            self.segment_data[segment] = pd.Series(dict([(marker, self.data_dict[marker]) for marker in markers]))

        # used for filling missing marker data
        if static_trial is not None:
            calibrate_data, _ = ViconCsvReader.reading(static_trial)
            for segment, markers in segment_definitions.items():
                segment_data = pd.Series(dict([(marker, calibrate_data[marker]) for marker in markers]))
                self.fill_missing_marker(segment_data, self.segment_data[segment])
        '''

        
        plate_nums=['1','2']
        
        #force names orders
        force_names_ori = ['Imported AMTI 400600 V1.00 #' + plate_num + ' - ' + data_type for plate_num in plate_nums
                           for data_type in ['Force', 'CoP']]
        if force_names_ori[0] in self.data_dict.keys():
            if('Imported AMTI 400600 V1.00 #1 - Force' not in self.data_dict.keys()):
                print(self.data_dict.keys())
                print(force_names_ori)
        else:
            force_names_ori = ['AMTI 400600 V1.00 #' + plate_num + ' - ' + data_type for plate_num in plate_nums
                           for data_type in ['Force', 'CoP']]
            if('AMTI 400600 V1.00 #1 - Force' not in self.data_dict.keys()):
                print(self.data_dict.keys())
                print(force_names_ori)
        # resample force data
        filtered_force_array = np.concatenate([data_filter(self.data_dict[force_name], 50, 1000) for force_name in force_names_ori], axis=1)
        filtered_force_array = filtered_force_array[::10, :]# 取10次中的最后一次采用
        filtered_force_df = pd.DataFrame(filtered_force_array, columns=FORCE_PLATE_DATA_FIELDS)
        
        # calibration of force offset through adding an offset
        '''
        cal_offset = sub_info[['Caliwand for plate 1-x', 'Caliwand for plate 1-y', 'Caliwand for plate 1-z',
                               'Caliwand for plate 2-x', 'Caliwand for plate 2-y', 'Caliwand for plate 2-z']]
        filtered_force_df[['plate_1_cop_x', 'plate_1_cop_y', 'plate_1_cop_z',
                           'plate_2_cop_x', 'plate_2_cop_y', 'plate_2_cop_z']] += cal_offset.values
        '''
        
        # marker's data
        '''
        self.segment_definitions = segment_definitions
        if segment_definitions != {}:
            markers = [marker for markers in segment_definitions.values() for marker in markers]
            self.data_frame = pd.concat([self.data[marker] for marker in markers], axis=1)
            self.data_frame.columns = [marker + '_' + axis for marker in markers for axis in ['X', 'Y', 'Z']]

        '''
        # 如果数据文件中有关节数据
        if('LKneeAngles' in self.data_dict.keys()):
            # KneeJoint values
            left_knee=np.concatenate([self.data_dict[feild_name] for feild_name in ['LKneeAngles','LKneeForce','LKneeMoment']],axis=1)
            left_knee_df=pd.DataFrame(left_knee,columns= ['LKneeAngles','LKneeForce','LKneeMoment'])
        
            right_knee=np.concatenate([self.data_dict[feild_name] for feild_name in ['RKneeAngles','RKneeForce','RKneeMoment']],axis=1)
            right_knee_df=pd.DataFrame(right_knee,columns=['RKneeAngles','RKneeForce','RKneeMoment'])
            self.data = pd.concat([left_knee_df,right_knee_df],axis=1)
            self.data = pd.concat([self.data, filtered_force_df], axis=1)
        else:# 只有测力台数据
            self.data=filtered_force_df.copy()

        
        #fill some force data (None) to zero
        # The none, this is probably because the force data calucation process by Nexus
        self.data_frame=self.data.copy()
        self.data_frame.index=range(self.data_frame.shape[0])
        self.data_frame.fillna(0.0,inplace=True)
        self.generate_labels_csv()
        #process moments to trafer data unit to international unit
        '''
        
        temp_fields=['L_KneeForce_X','L_KneeForce_Y','L_KneeForce_Z',
                           'L_KneeMoment_X','L_KneeMoment_Y','L_KneeMoment_Z',
                           'R_KneeForce_X','R_KneeForce_X','R_KneeForce_Z',
                           'R_KneeMoment_X','R_KneeMoment_Y','R_KneeMoment_Z']
        for force_field in temp_fields:
            self.data[force_field]=self.data[force_field]*subject_info['weight']
        '''
        self.get_frame_range()
        
        print('vicon raw data shape:', self.data.shape, 'raw data frame shape:', self.data_frame.shape)
    
    def generate_labels_csv(self):
        self.data_frame.to_csv(os.path.join(os.path.dirname(self.file_path), 'vicon_data_'+self.trial+'.csv'),index=False)


    def get_frame_range(self):
        frames=[]
        with open(self.file_path, encoding='utf-8-sig') as f:
            for idx, row in enumerate(csv.reader(f)):
                if(idx>4):# 从第四行开始
                    if row!=[]:
                        try:
                            frames.append(int(row[0]))
                        except ValueError:
                            frames.append(np.nan)
                    else:
                        break
            self.frame_start=min(frames[0:2])
            self.frame_end=frames[-1]
        return self.frame_start, self.frame_end

        
    @staticmethod
    def reading(file_path):
        data_collection = dict()
        sample_rate_collection = {}
        state = PARENT_TITLE# state=0
        with open(file_path, encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                if state == PARENT_TITLE:# 第一行
                    parent_title = row[0] #第一列
                    state = SAMPLE_RATE
                elif state == SAMPLE_RATE:#state=1 第二行
                    sample_rate_collection[parent_title] = float(row[0])
                    state = TITLE
                elif state == TITLE: #state=2 第三行
                    titles = list()
                    for col in row[2:]:
                        if col != "":
                            if 'Model Outputs' == parent_title:
                                subject_name, title = col.split(':')
                            else:
                                title = col
                        titles.append(title)
                    state = DIRECTION
                elif state == DIRECTION: #state=3 第四行
                    directions = [i for i in row[2:]]
                    data = [[] for _ in directions]
                    state = UNITS
                elif state == UNITS:#state=4 第五行
                    # TODO@Dianxin: Record the units! never do it.
                    state = DATA
                elif state == DATA:#state=5, 第六行
                    if row == []: # 如果存在空行， 说明新的类型的数据输出 或数据被遍历完
                        state = PARENT_TITLE# state 被重置为零
                        for title in titles:
                            data_collection[title] = dict()
                        for i, direction in enumerate(directions):
                            data_collection[titles[i]][direction] = data[i]
                        for key, value in data_collection.items():
                            data_collection[key] = pd.DataFrame(value)
                        continue
                    for i, x in enumerate(row[2:]):
                        try:
                            data[i].append(float(x))
                        except ValueError:
                            data[i].append(np.nan)
        return data_collection, sample_rate_collection
 
    
    def get_joint_angles(self, sub_info,joint):
        '''
        AUthor: suntao
        
        
        P_04_kezhe:LKneeAngle, RKneeAngle
        '''
        assert(joint in ['L_KneeAngle','R_KneeAngle'])
        return self.data[sub_info['Name']+':'+joint]
        

    def get_right_external_kam(self, sub_info):
        sub_height, sub_weight = sub_info[[SUBJECT_HEIGHT, SUBJECT_WEIGHT]]
        # cal_offset = sub_info[['Caliwand for plate 2-x', 'Caliwand for plate 2-y', 'Caliwand for plate 2-z']]
        force_cop = self.data_frame[['plate_2_cop_x', 'plate_2_cop_y', 'plate_2_cop_z']].values
        # force_cop += cal_offset
        knee_origin = (self.data_frame[['RFME_X', 'RFME_Y', 'RFME_Z']].values +
                       self.data_frame[['RFLE_X', 'RFLE_Y', 'RFLE_Z']].values) / 2
        r = force_cop - knee_origin
        force_data = -self.data_frame[['plate_2_force_x', 'plate_2_force_y', 'plate_2_force_z']].values
        knee_moment = pd.DataFrame(np.cross(r, force_data), columns=EXT_KNEE_MOMENT)
        knee_moment /= (sub_height * sub_weight * 1000.)
        return knee_moment

    def get_angular_velocity_theta(self, segment, check_len):
        segment_data_series = self.segment_data[segment]
        sampling_rate = self.sample_rate['Trajectories']

        walking_data = pd.concat(segment_data_series.tolist(), axis=1).values
        check_len = min(walking_data.shape[0], check_len)
        marker_number = int(walking_data.shape[1] / 3)
        angular_velocity_theta = np.zeros([check_len])

        next_marker_matrix = walking_data[0, :].reshape([marker_number, 3])
        # vectiorize this for loop.
        for i_frame in range(check_len):
            if i_frame == 0:
                continue
            current_marker_matrix = next_marker_matrix
            next_marker_matrix = walking_data[i_frame, :].reshape([marker_number, 3])
            R_one_sample, _ = rigid_transform_3d(current_marker_matrix, next_marker_matrix)
            theta = np.math.acos((np.matrix.trace(R_one_sample) - 1) / 2)

            angular_velocity_theta[i_frame] = theta * sampling_rate / np.pi * 180
        return angular_velocity_theta

    def get_rshank_angle(self, direction):
        ankle = (self.data_dict['RTAM'] + self.data_dict['RFAL']) / 2
        knee = (self.data_dict['RFME'] + self.data_dict['RFLE']) / 2
        vector = (knee - ankle).values
        if direction == 'X':
            return np.arctan2(vector[:, 2], vector[:, 1])
        elif direction == 'Y':
            return np.arctan2(vector[:, 2], vector[:, 0])
        elif direction == 'Z':
            return np.arctan2(vector[:, 1], vector[:, 0])

    def get_angular_velocity(self, segment, direction):
        segment_data_series = self.segment_data[segment]
        sampling_rate = self.sample_rate['Trajectories']
        walking_data = pd.concat(segment_data_series.tolist(), axis=1).values
        data_len = walking_data.shape[0]
        marker_number = int(walking_data.shape[1] / 3)
        angular_velocity = np.zeros([data_len, 3])

        next_marker_matrix = walking_data[0, :].reshape([marker_number, 3])
        if direction == 'X':
            next_marker_matrix[:, 0] = 0
        elif direction == 'Y':
            next_marker_matrix[:, 1] = 0
        elif direction == 'Z':
            next_marker_matrix[:, 2] = 0
        # vectiorize this for loop.
        for i_frame in range(1, data_len):
            current_marker_matrix = next_marker_matrix
            next_marker_matrix = walking_data[i_frame, :].reshape([marker_number, 3])
            if direction == 'X':
                next_marker_matrix[:, 0] = 0
            elif direction == 'Y':
                next_marker_matrix[:, 1] = 0
            elif direction == 'Z':
                next_marker_matrix[:, 2] = 0
            R_one_sample, _ = rigid_transform_3d(current_marker_matrix, next_marker_matrix)
            theta = rotation_matrix_to_euler_angles(R_one_sample)

            angular_velocity[i_frame, :] = theta * sampling_rate / np.pi * 180
        angular_velocity = pd.DataFrame(angular_velocity)
        angular_velocity.columns = ['X', 'Y', 'Z']
        return angular_velocity[direction]

    def get_marker_position(self, marker_name):
        return self.data_dict[marker_name]

    def crop(self, start_index):
        # keep index after start_index
        self.data_frame = self.data_frame.loc[start_index:]
        self.data_frame.index = range(self.data_frame.shape[0])

        for segment, markers in self.segment_definitions.items():
            for marker in markers:
                self.segment_data[segment][marker] = self.segment_data[segment][marker].loc[start_index:]
                self.segment_data[segment][marker].index = range(self.segment_data[segment][marker].shape[0])

    def fill_missing_marker(self, calibrate_makers, motion_markers):
        if sum([motion_marker.isnull().sum().sum() for motion_marker in motion_markers.tolist()]) == 0:
            return

        # take the first frame of calibrate marker data for calibration
        calibrate_makers = pd.concat(calibrate_makers.tolist(), axis=1).values
        calibrate_makers = calibrate_makers[0, :].reshape([-1, 3])

        walking_data = pd.concat(motion_markers.tolist(), axis=1).values
        data_len = walking_data.shape[0]

        for i_frame in range(data_len):
            marker_matrix = walking_data[i_frame, :].reshape([-1, 3])
            coordinate_points = np.argwhere(~np.isnan(marker_matrix[:, 0])).reshape(-1)
            missing_points = np.argwhere(np.isnan(marker_matrix[:, 0])).reshape(-1)
            if len(missing_points) == 0:  # All the marker exist
                continue
            if len(coordinate_points) >= 3:
                origin, x, y, z = wearable_math.generate_coordinate(calibrate_makers[coordinate_points, :])
                origin_m, x_m, y_m, z_m = wearable_math.generate_coordinate(marker_matrix[coordinate_points, :])
                for missing_point in missing_points.tolist():
                    relative_point = wearable_math.get_relative_position(origin, x, y, z,
                                                                         calibrate_makers[missing_point, :])
                    missing_data = wearable_math.get_world_position(origin_m, x_m, y_m, z_m, relative_point)
                    motion_markers[missing_point]['X'][i_frame] = missing_data[0]
                    motion_markers[missing_point]['Y'][i_frame] = missing_data[1]
                    motion_markers[missing_point]['Z'][i_frame] = missing_data[2]
        for motion_marker in motion_markers:
            motion_marker.interpolate(method='linear', axis=0, inplace=True)

    def append_external_kam(self):
        # calibrate force plate
        pass

    
    
# butterworth low-pass filter
def data_filter(data, cut_off_fre, sampling_fre, filter_order=4):
    fre = cut_off_fre / (sampling_fre / 2)
    b, a = butter(filter_order, fre, 'lowpass')
    if len(data.shape) == 1:
        data_filtered = filtfilt(b, a, data)
    else:
        data_filtered = filtfilt(b, a, data, axis=0)
    return data_filtered
 

class SageCsvReader:
    """
    Read the csv file exported from sage systems
    """
    GUESSED_EVENT_INDEX = 0

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.sample_rate = 100
        self.data_frame = self.data[
            [field + '_' + str(index) for index, label in enumerate(IMU_SENSOR_LIST) for field in
             IMU_FIELDS]].copy()
        index = self.data['Package_0']
        for i in range(1, len(self.data['Package_0'])):
            if self.data['Package_0'].loc[i] < self.data['Package_0'].loc[i - 1]:
                self.data.loc[i:, 'Package_0'] += 65536
        index = index - self.data['Package_0'].loc[0]
        # fill dropout data with nan
        if index.size - 1 != index.iloc[-1]:
            print("Inconsistent shape")
        self.data_frame.index = index
        self.data_frame = self.data_frame.reindex(range(0, int(index.iloc[-1] + 1)))
        self.data_frame.columns = ["_".join([col.split('_')[0], IMU_SENSOR_LIST[int(col.split('_')[1])]]) for col in
                                   self.data_frame.columns]
        self.missing_data_index = self.data_frame.isnull().any(axis=1)
        self.data_frame = self.data_frame.interpolate(method='linear', axis=0)
        # self.data_frame.loc[:, :] = data_filter(self.data_frame.values, 15, 100, 2)

    def get_norm(self, sensor, field, is_plot=False):
        assert sensor in IMU_SENSOR_LIST
        assert field in ['Accel', 'Gyro']
        norm_array = np.linalg.norm(self.data_frame[[field + direct + '_' + sensor for direct in ['X', 'Y', 'Z']]],
                                    axis=1)
        if is_plot:
            plt.figure()
            plt.plot(norm_array)
            plt.show()
        return norm_array

    def get_first_event_index(self):
        for i in range(len(self.data['sync_event'])):
            if self.data['sync_event'].loc[i] == 1:
                return i
        return None

    def get_field_data(self, sensor, field):
        if sensor not in IMU_SENSOR_LIST:
            raise RuntimeError("No such a sensor")
        if field not in ['Accel', 'Gyro']:
            raise RuntimeError("{field} not in ['Accel', 'Gyro']")
        index = str(IMU_SENSOR_LIST.index(sensor))
        data = self.data_frame[[field + direct + '_' + str(index) for direct in ['X', 'Y', 'Z']]]
        data.columns = ['X', 'Y', 'Z']
        return data

    def crop(self, start_index):
        self.data = self.data.loc[start_index:]
        self.data.index = self.data.index - self.data.index[0]
        self.data_frame = self.data_frame.loc[start_index:]
        self.data_frame.index = self.data_frame.index - self.data_frame.index[0]

    def get_walking_strike_off(self, strike_delay, off_delay, segment, cut_off_fre_strike_off=None,
                               verbose=False):
        """ Reliable algorithm used in TNSRE first submission"""
        gyr_thd = np.rad2deg(2.6)
        acc_thd = 1.2
        max_distance = self.sample_rate * 2  # distance from stationary phase should be smaller than 2 seconds

        acc_data = np.array(
            self.data_frame[['_'.join([direct, segment]) for direct in ['AccelX', 'AccelY', 'AccelZ']]])
        gyr_data = np.array(self.data_frame[['_'.join([direct, segment]) for direct in ['GyroX', 'GyroY', 'GyroZ']]])

        if cut_off_fre_strike_off is not None:
            acc_data = data_filter(acc_data, cut_off_fre_strike_off, self.sample_rate, filter_order=2)
            gyr_data = data_filter(gyr_data, cut_off_fre_strike_off, self.sample_rate, filter_order=2)

        gyr_x = gyr_data[:, 0]
        data_len = gyr_data.shape[0]

        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        gyr_magnitude = np.linalg.norm(gyr_data, axis=1)
        acc_magnitude = acc_magnitude - 9.81

        stationary_flag = self.__find_stationary_phase(
            gyr_magnitude, acc_magnitude, acc_thd, gyr_thd)

        strike_list, off_list = [], []
        i_sample = 0

        while i_sample < data_len:
            # step 0, go to the next stationary phase
            if not stationary_flag[i_sample]:
                i_sample += 1
            else:
                front_crossing, back_crossing = self.__find_zero_crossing(gyr_x, gyr_thd, i_sample)

                if not back_crossing:  # if back zero crossing not found
                    break
                if not front_crossing:  # if front zero crossing not found
                    i_sample = back_crossing
                    continue

                the_strike = self.find_peak_max(gyr_x[front_crossing:i_sample], height=0)
                the_off = self.find_peak_max(gyr_x[i_sample:back_crossing], height=0)

                if the_strike is not None and i_sample - (the_strike + front_crossing) < max_distance:
                    strike_list.append(the_strike + front_crossing + strike_delay)
                if the_off is not None and the_off < max_distance:
                    off_list.append(the_off + i_sample + off_delay)
                i_sample = back_crossing
        if verbose:
            plt.figure()
            plt.plot(stationary_flag * 400)
            plt.plot(gyr_x)
            plt.plot(strike_list, gyr_x[strike_list], 'g*')
            plt.plot(off_list, gyr_x[off_list], 'r*')

        return strike_list, off_list

    @staticmethod
    def __find_stationary_phase(gyr_magnitude, acc_magnitude, foot_stationary_acc_thd, foot_stationary_gyr_thd):
        """ Old function, require 10 continuous setps """
        data_len = gyr_magnitude.shape[0]
        stationary_flag, stationary_flag_temp = np.zeros(gyr_magnitude.shape), np.zeros(gyr_magnitude.shape)
        stationary_flag_temp[
            (acc_magnitude < foot_stationary_acc_thd) & (abs(gyr_magnitude) < foot_stationary_gyr_thd)] = 1
        for i_sample in range(data_len):
            if stationary_flag_temp[i_sample - 5:i_sample + 5].all():
                stationary_flag[i_sample] = 1
        return stationary_flag

    @staticmethod
    def __find_stationary_phase_2(gyr_magnitude, acc_magnitude, foot_stationary_acc_thd, foot_stationary_gyr_thd):
        """ New function, removed 10 sample requirement """
        stationary_flag = np.zeros(gyr_magnitude.shape)
        stationary_flag[(acc_magnitude < foot_stationary_acc_thd) & (gyr_magnitude < foot_stationary_gyr_thd)] = 1
        return stationary_flag

    def __find_zero_crossing(self, gyr_x, foot_stationary_gyr_thd, i_sample):
        """
        Detected as a zero crossing if the value is lower than negative threshold.
        :return:
        """
        max_search_range = self.sample_rate * 3  # search 3 second front data at most
        front_crossing, back_crossing = False, False
        for j_sample in range(i_sample, max(0, i_sample - max_search_range), -1):
            if gyr_x[j_sample] < - foot_stationary_gyr_thd:
                front_crossing = j_sample
                break
        for j_sample in range(i_sample, gyr_x.shape[0]):
            if gyr_x[j_sample] < - foot_stationary_gyr_thd:
                back_crossing = j_sample
                break
        return front_crossing, back_crossing

    @staticmethod
    def find_peak_max(data_clip, height, width=None, prominence=None):
        """
        find the maximum peak
        :return:
        """
        peaks, properties = find_peaks(data_clip, width=width, height=height, prominence=prominence)
        if len(peaks) == 0:
            return None
        peak_heights = properties['peak_heights']
        max_index = np.argmax(peak_heights)
        return peaks[max_index]

    def create_step_id(self, segment, verbose=False):
        max_step_length = self.sample_rate * 2
        [RON, ROFF] = self.get_walking_strike_off(0, 0, segment, 10, verbose)
        events_dict = {'ROFF': ROFF, 'RON': RON}
        foot_events = translate_step_event_to_step_id(events_dict, max_step_length)
        self.data_frame.insert(0, EVENT_COLUMN, np.nan)
        if verbose:
            plt.figure()
        for _, event in foot_events.iterrows():
            self.data_frame.loc[event[0]:event[1], EVENT_COLUMN] = SageCsvReader.GUESSED_EVENT_INDEX
            SageCsvReader.GUESSED_EVENT_INDEX += 1
            if verbose:
                plt.plot(self.data_frame.loc[event[0]:event[1], 'GyroX_'+segment].values)
        if self.missing_data_index.any(axis=0):
            print("Steps containing corrupted data: {}. They are marked as minus".format(
                self.data_frame[self.missing_data_index][EVENT_COLUMN].dropna().drop_duplicates().tolist()))
            self.data_frame.loc[self.missing_data_index, EVENT_COLUMN] *= -1  # mark the missing IMU data as minus event
        if verbose:
            plt.show()


class DivideMaxScalar(MinMaxScaler):
    def partial_fit(self, X, y=None):
        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)
        data_range = data_max - data_min
        data_bi_max = np.nanmax(abs(X), axis=0)
        self.scale_ = 1 / data_bi_max
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        X *= self.scale_
        return X


def data_filter(data, cut_off_fre, sampling_fre, filter_order=4):
    fre = cut_off_fre / (sampling_fre / 2)
    b, a = butter(filter_order, fre, 'lowpass')
    if len(data.shape) == 1:
        data_filtered = filtfilt(b, a, data)
    else:
        data_filtered = filtfilt(b, a, data, axis=0)
    return data_filtered


def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rigid_transform_3d(a, b):
    """
    Get the Rotation Matrix and Translation array between A and B.
    return:
        R: Rotation Matrix, 3*3
        T: Translation Array, 1*3
    """
    assert len(a) == len(b)

    N = a.shape[0]  # total points
    centroid_A = np.mean(a, axis=0)
    centroid_B = np.mean(b, axis=0)
    # centre the points
    AA = a - np.tile(centroid_A, (N, 1))
    BB = b - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = np.dot(AA.T, BB)
    U, _, V_t = linalg.svd(np.nan_to_num(H))
    R = np.dot(V_t.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        # print
        # "Reflection detected"
        V_t[2, :] *= -1
        R = np.dot(V_t.T, U.T)
    T = -np.dot(R, centroid_A.T) + centroid_B.T
    return R, T


def sync_via_correlation(data1, data2, verbose=False):
    correlation = np.correlate(data1, data2, 'full')
    delay = len(data2) - np.argmax(correlation) - 1
    if verbose:
        plt.figure()
        if delay > 0:
            plt.plot(data1)
            plt.plot(data2[delay:])
        else:
            plt.plot(data1[-delay:])
            plt.plot(data2)
        plt.show()
    return delay


def translate_step_event_to_step_id(events_dict, max_step_length):
    # FILTER EVENTS
    event_list = sorted(
        [[i, event_type] for event_type in ['RON', 'ROFF'] for i in events_dict[event_type]], key=lambda x: x[0])
    event_type_dict = {i: event_type for i, event_type in event_list}
    event_ids = [i[0] for i in event_list]
    RON_events = events_dict['RON']

    def is_qualified_ron_event(ron_i):
        i = event_ids.index(RON_events[ron_i])
        prev_event_type, curr_event_type, next_event_type = map(lambda x: event_type_dict[event_ids[x]], [i-1, i, i+1])
        prev_step_length, current_step_length = np.diff(RON_events[ron_i - 2:ron_i+1])
        if curr_event_type not in [prev_event_type, next_event_type]\
                and 1.33 * prev_step_length > current_step_length > 0.75 * prev_step_length\
                and 50 < current_step_length < max_step_length:
            return True
        return False

    def transform_to_step_events(ron_i):
        """return consecutive events: off, on, off, on"""
        current_event_id_i = event_ids.index(RON_events[ron_i])
        return map(lambda i: event_ids[i], range(current_event_id_i-3, current_event_id_i + 1))

    r_steps = filter(is_qualified_ron_event, range(10, len(RON_events)))
    r_steps = map(transform_to_step_events, r_steps)
    r_steps = pd.DataFrame(r_steps)
    r_steps.columns = ['off_3', 'on_2', 'off_1', 'on_0']
    step_type_to_event_columns = {STANCE_SWING: ['on_2', 'on_0'], STANCE: ['on_2', 'off_1']}
    return r_steps[step_type_to_event_columns[STEP_TYPE]]


def calibrate_force_plate_center(file_path, plate_num):
    assert (plate_num in [1, 2])
    vicon_data = ViconCsvReader(file_path)
    data_DL = vicon_data.data['DL']
    data_DR = vicon_data.data['DR']
    data_ML = vicon_data.data['ML']
    center_vicon = (data_DL + data_DR) / 2 + (data_DL - data_ML)
    if plate_num == 1:
        center_plate = vicon_data.data['Imported Bertec Force Plate #1 - CoP']
    else:
        center_plate = vicon_data.data['Imported Bertec Force Plate #2 - CoP']
    center_plate.columns = ['X', 'Y', 'Z']
    plate_cop = np.mean(center_plate, axis=0)
    cop_offset = np.mean(center_vicon, axis=0) - plate_cop
    return plate_cop, cop_offset

