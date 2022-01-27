#!/bin/pyenv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error


# import customization robot control framework
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR+"/../")
from CRCF.data_manager import *


display_row_num=2000
joint_name='elbow'

joint_angle_in_C_path = os.path.join("~/workspace/joint-angle-estimation/data/results/",joint_name+"_joint_angle.csv") 
joint_angle_in_M_path = os.path.join("~/workspace/joint-angle-estimation/data/results/reference/",joint_name+"_joint_angle.csv") 

#joint_angle_in_C_path = os.path.join("~/workspace/M2C_HW/data/results/",joint_name+"_joint_angle.csv") 
#joint_angle_in_M_path = os.path.join("~/workspace/M2C_HW/data/results/reference/",joint_name+"_joint_angle.csv") 

joint_angle_in_C = pd.read_csv(joint_angle_in_C_path, sep='\t', index_col=None, header=0, skip_blank_lines=True, dtype=str)
data_C=joint_angle_in_C.iloc[:display_row_num,:].astype(float)

data_C_angle_columns=data_C.columns
data_C['Methods']='C'
data_C['Time']=np.linspace(0,data_C.shape[0]/100.0,data_C.shape[0])

joint_angle_in_M = pd.read_csv(joint_angle_in_M_path, sep=',', index_col=None, header=0, skip_blank_lines=True, dtype=str)
data_M=joint_angle_in_M.iloc[:display_row_num,:].astype(float)
data_M['Methods']='M'
data_M['Time']=np.linspace(0,data_M.shape[0]/100.0,data_M.shape[0])

diff=abs(data_C.iloc[:,1:4]-data_M.iloc[:,1:4])

MAE= diff.mean(axis=0)
print("MAE:",MAE)

print(data_C.head())
print(data_M.head())
print(diff.head())

data=pd.concat([data_C,data_M],axis=0)

data=data.melt(id_vars=['Count','Time','Methods'],var_name='cols',value_name='vals')

# plot

figwidth=12;figheight=3.5
subplot_left=0.08; subplot_right=0.97; subplot_top=0.9;subplot_bottom=0.14


g=sns.FacetGrid(data=data,col='cols',hue='Methods',sharey=False)
g.map_dataframe(sns.lineplot,x='Time',y='vals')
g.set_axis_labels("Time [s]", r"Joint angle [deg]")
g.add_legend()
#plt.legend(title='Program language', loc='best', labels=['Matlab', 'C'])
#g.set_xticklabels(xticklabels)
g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom)
g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)


plt.savefig(joint_name+".svg")


plt.show()



