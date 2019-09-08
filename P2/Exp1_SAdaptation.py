#!/usr/bin/env python
# -*- coding:utf-8 -*-

import loaddata as LD
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    columnsName=['RFO1','RFO2','RHO1','RHO2','LFO1','LFO2','LHO1','LKO2','RFSA','RHSA','LFSA','LHSA',
'RFANC0','RFANC1','RHANC0','RHANC1','LFANC0','LFANC1','LHANC0','LHANC1','RFPA0','RFPA1','RHPA0','RHPA1','LFPA0','LFPA1','LHPA0','LHPA1'
                ]
    fileName="controlfile_CPG"
    freq=40.0 # 40Hz,
    fine_data=LD.loadData(fileName,columnsName)


    #2) postprecessing 
    CPGOutData=columnsName[8:13]
    if len(sys.argv)>=2:
        run_id = int(sys.argv[1]) # The id of the experiments
    else:
        run_id = 0
    data = fine_data[run_id][CPGOutData]
    read_rows=data.shape[0]

    #2) plot diagram
    
    cases=["RF", "RH", "LF", "LH"]
    fig, axs = plt.subplots(4, 1)
    time = np.linspace(0,read_rows/freq,read_rows)
    for index,case in enumerate(cases):
        axs[index].plot(time,data.iloc[:,index],'r')
        #axs[index].legend(['N0', 'N1'], loc='upper left')
        axs[index].set_ylabel(case)
        axs[index].axis([0,max(time),-0.02,0.14])
        axs[index].grid()
    plt.xlabel('Time[s]')
    axs[0].set_title("adaptive sensory feedback gain of four legs ")
