#! /usr/bin/pyenv python
#!-coding utf-8-

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pdb

import So2Oscillator as cpg

def kinematics(theta1, theta2):
    L0=40.53
    L1=70.02;L2=86.36;

    x= L1*np.cos(theta1) + L2*np.cos(theta2+theta1)
    z= -L0 - (L1*np.sin(theta1) + L2*np.sin(theta2+theta1))

    return x, z



def fig(data):
    figsize=(6,8.5)#8.6614
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(4,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.08,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))
    axs.append(fig.add_subplot(gs1[2:4,0]))

    axs[0].plot()


    #plt.savefig('/media/suntao/DATA/Research/P3_workspace/Figures/ki.svg')
    plt.show()



def trajAngle(position):
    d_max=np.array([-1,-1,-1])
    d_min=np.array([-1,-1,100000])
    for idx in range(position.shape[0]):
        position_temp=np.delete(position,idx,axis=0)
        Pd=np.sum(np.power(position_temp-position[idx,:],2),axis=1)
        x_max=np.where(Pd==np.amax(Pd))
        x_min=np.where(Pd==np.amin(Pd))
        v_max=np.amax(Pd)
        v_min=np.amin(Pd)
        d_max=np.vstack([d_max,np.array([idx,x_max[0][0],v_max])])
        d_min=np.vstack([d_min,np.array([idx,x_min[0][0],v_min])])
    
    # angle and distance of the longest axis
    max_index = np.where(d_max[:,2]==np.amax(d_max[:,2]))
    i=int(d_max[max_index[0][0],0])
    j=int(d_max[max_index[0][0],1])
    pi=position[i,:]
    pj=position[j,:]
    pij=pj-pi
    angle=math.atan2(pij[1],pij[0])
    distance=np.linalg.norm(pij)


    # crosspoint number
    print('ss:', np.amin(d_min[:,2]))
    

    return angle/math.pi*180, distance



if __name__ == '__main__':

    os1 = cpg.So2Oscillator(0.04)
    data1=[]
    data2=[]
    
    offset1=.8
    offset2=1.7
    amplitude1=1.2
    amplitude2=1.0

    for i in range(1000):
        os1.step()
        data1.append(os1.getOutput(1))
        data2.append(os1.getOutput(2))
    

    x,z = kinematics(amplitude1*np.array(data1[100:215])+offset1, amplitude2*np.array(data2[100:215])+offset2)
    position=np.array([x,z]).T
    print(position.shape)
    print(trajAngle(position))
    plt.figure()
    plt.plot(x,z)
    plt.show()

