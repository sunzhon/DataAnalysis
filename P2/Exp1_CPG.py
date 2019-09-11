#!/usr/bin/env python
# -*- coding:utf-8 -*-
import loaddata as LD
import sys
import matplotlib.pyplot as plt
import numpy as np

#1) load data
columnsName_CPG=['RFO1','RFO2','RHO1','RHO2','LFO1','LFO2','LHO1','LKO2',
                 'RFSA','RHSA','LFSA','LHSA',
                 'RFACITerm0','RFACITerm1','RHACITerm0','RHACITerm1','LFACITerm0','LFACITerm1','LHACITerm0','LHACITerm1',
                 'RFSFTerm0','RFSFTerm1','RHSFTerm0','RHSFTerm1','LFSFTerm0','LFSFTerm1','LHSFTerm0','LHSFTerm1',
                 'RFFM','RHFM','LFFM','LHFM']
fileName="controlfile_CPG"
freq=40.0 # 40Hz,
fine_data=LD.loadData(fileName,columnsName_CPG)

#2) postprecessing 
CPGOutData=columnsName_CPG[0:8]
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
    axs[index].plot(time,data.iloc[:,2*index],'r')
    axs[index].plot(time,data.iloc[:,2*index+1],'b')
    axs[index].legend(['N0', 'N1'], loc='upper left')
    axs[index].set_ylabel(case)
    axs[index].axis([0,max(time),-1,1])

plt.xlabel('Time[s]')
axs[0].set_title("CPG outputs of four legs ")

if __name__=="__main__":
    plt.show()
