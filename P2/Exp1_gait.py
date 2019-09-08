#!/usr/bin/env python
# -*- coding:utf-8 -*-
import loaddata as LD
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
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
    #2) preprocess of the data
    threshold = 0.01
    state = np.zeros(data.shape, int)
    for i in range(0,4):
        for j in range(len(data)):
            if data.iloc[j,i] < threshold:
                state[j, i] = 0
            else:
                state[j, i] = 1
    #3) plot the gait diagram
    axprops = dict(xticks=[], yticks=[])
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest')
    #fig = plt.figure(figsize=(400, 400))
    fig = plt.figure()
    ax1 = fig.add_axes([0.2, 0.4, 0.75, 0.1], **axprops)
    ax2 = fig.add_axes([0.2, 0.3, 0.75, 0.1], **axprops)
    ax3 = fig.add_axes([0.2, 0.2, 0.75, 0.1], **axprops)
    ax4 = fig.add_axes([0.2, 0.1, 0.75, 0.1], **axprops)

    ax1.set_yticks([0.1]); ax1.set_yticklabels(["RF"])
    ax2.set_yticks([0.2]); ax2.set_yticklabels(["RH"])
    ax3.set_yticks([0.3]); ax3.set_yticklabels(["LF"])
    ax4.set_yticks([0.4]); ax4.set_yticklabels(["LH"])

    ax1.set_title("Gait diagram")
    cow,column=state.shape

    x1 = np.where(state[:,0] > 0.7, 1.0, 0.0)
    x2 = np.where(state[:,1] > 0.7, 1.0, 0.0)
    x3 = np.where(state[:,2] > 0.7, 1.0, 0.0)
    x4 = np.where(state[:,3] > 0.7, 1.0, 0.0)

    ax1.imshow(x1.reshape((1, -1)), **barprops)
    ax2.imshow(x2.reshape((1, -1)), **barprops)
    ax3.imshow(x3.reshape((1, -1)), **barprops)
    ax4.imshow(x4.reshape((1, -1)), **barprops)
    #plt.savefig('home/suntao/workspace/experiment_result/28_11_18/ground_reaction_force.jpg')
