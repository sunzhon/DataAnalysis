#!/usr/bin/env python
# -*- coding:utf-8 -*-
import loaddata as LD
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import pathlib
import os
import sys
import time

#1) load data
fileName="sensorfile_POSE"
culumnsName=["roll","pitch","yaw","x","y","z"]
fine_data=LD.loadData(fileName,culumnsName)
freq=40.0 # 40Hz,

#2) postprecessing 
if len(sys.argv)>=2:
    run_id = int(sys.argv[1]) # The id of the experiments
else:
    run_id = 0

data=fine_data[run_id]
read_rows=data.shape[0]

#2) plot diagram
cases=["Roll[rad]", "Pitch[rad]", "Yaw[rad]"]
fig, axs = plt.subplots(3, 1)
time = np.linspace(0,read_rows/freq,read_rows)
colors=['r','g','b']
for index,case in enumerate(cases):
    axs[index].plot(time,data.iloc[:,index],colors[index])
    #axs[index].legend(case, loc='upper left')
    axs[index].set_ylabel(case)
    axs[index].axis([0,max(time),-0.1,0.1])
    axs[index].grid()

    plt.xlabel('Time[s]')
    axs[0].set_title("The body attitude angles")

if __name__=='__main__':
    plt.show()
