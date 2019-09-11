#!/usr/bin/env python
# -*- coding:utf-8 -*-
import loaddata as LD
import sys
import matplotlib.pyplot as plt
import numpy as np

#1)load data
columnsName=['RF','RH','LF','LH']
fileName="sensorfile_GRF"
freq=40.0 # 40Hz,
fine_data=LD.loadData(fileName,columnsName)


#2) postprecessing 
if len(sys.argv)>=2:
    run_id = int(sys.argv[1]) # The id of the experiments
else:
    run_id = 0

data = fine_data[run_id]
read_rows=data.shape[0]

#2) plot diagram
cases=["RF", "RH", "LF", "LH"]
fig, axs = plt.subplots(4, 1)
time = np.linspace(0,read_rows/freq,read_rows)

for index,case in enumerate(cases):
    axs[index].plot(time,data.iloc[:,index],color="black")
    axs[index].legend([case], loc='upper left')
    axs[index].axis([0,read_rows/freq,-.5,1.0])
    axs[index].set_ylabel('Value')
    axs[index].set_xlabel('Time[s]')
    axs[0].set_title("Ground reaction force of foot")

if __name__=="__main__":
    plt.show()
