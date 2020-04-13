#! /usr/bin/pyenv python
#!-coding utf-8-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
def kinematics():
    figsize=(6,8.5)#8.6614
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(4,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.08,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))
    axs.append(fig.add_subplot(gs1[2:4,0]))


    theta0_elevation= np.pi/2.0;
    theta0_depression= np.pi/2.0*3.0;

    theta1_forward= np.pi/2.0;
    theta1_backward= np.pi/2.0*3;
    theta1_middle = (theta1_forward + theta1_backward)/2.0

    theta2_flexion = 54/180.0*np.pi;
    theta2_extension = 2.0*np.pi- theta2_flexion;
    theta2_middle = (theta2_flexion + theta2_extension )/2.0

    L0=40.53
    L1=70.02;L2=86.36;
    count = 2000

    theta0=np.linspace(theta0_elevation,theta0_depression,count)
    theta1=np.linspace(theta1_forward,theta1_backward,count)
    theta2=np.linspace(theta2_flexion,theta2_extension,count)


    x= L1*np.cos(theta1_forward-np.pi/2) + L2*np.cos(theta2+theta1_forward-1.5*np.pi)
    z= -L0 - (L1*np.sin(theta1_forward-np.pi/2) + L2*np.sin(theta2+theta1_forward-1.5*np.pi))
    axs[0].plot(x,z,'r')

    x= L1*np.cos(theta1_backward-np.pi/2) + L2*np.cos(theta2+theta1_backward-1.5*np.pi)
    z= -L0 - (L1*np.sin(theta1_backward-np.pi/2) + L2*np.sin(theta2+theta1_backward-1.5*np.pi))
    axs[0].plot(x,z,'g')

    x= L1*np.cos(theta1_middle-np.pi/2) + L2*np.cos(theta2+theta1_middle-1.5*np.pi)
    z= -L0 - (L1*np.sin(theta1_middle-np.pi/2) + L2*np.sin(theta2+theta1_middle-1.5*np.pi))
    axs[0].plot(x,z,'g:')

    x= L1*np.cos(theta1-np.pi/2) + L2*np.cos(theta2_flexion+theta1-1.5*np.pi)
    z= -L0 - (L1*np.sin(theta1-np.pi/2) + L2*np.sin(theta2_flexion+theta1-1.5*np.pi))
    axs[0].plot(x,z,'b')

    x= L1*np.cos(theta1-np.pi/2) + L2*np.cos(theta2_extension+theta1-1.5*np.pi)
    z= -L0 - (L1*np.sin(theta1-np.pi/2) + L2*np.sin(theta2_extension+theta1-1.5*np.pi))
    axs[0].plot(x,z,'k')


    x= L1*np.cos(theta1-np.pi/2) + L2*np.cos(theta2_middle+theta1-1.5*np.pi)
    z= -L0 - (L1*np.sin(theta1-np.pi/2) + L2*np.sin(theta2_middle+theta1-1.5*np.pi))
    axs[0].plot(x,z,'y')


    x= L1*np.cos(theta1-np.pi/2) + L2*np.cos(theta2_middle+theta1-1.5*np.pi)
    z= -L0 - (L1*np.sin(theta1-np.pi/2) + L2*np.sin(theta2_middle+theta1-1.5*np.pi))
    axs[0].plot(x,z,'y')
    
    axs[0].set_ylabel("z [mm]")
    axs[0].set_xlabel('x [mm]')


    virtual_leg_length= np.sqrt(L1*L1 +L2*L2 - 2.0*L1*L2*np.cos(theta2))


    min_vl= min(virtual_leg_length)
    max_vl = max(virtual_leg_length)
    print(max_vl)



    alpha=np.arccos((L1*L1+ min_vl*min_vl - L2*L2)/(2.0*L1*min_vl))
    length= L0 + min_vl*np.sin(theta1-alpha-np.pi/2)
    print(length)
    alpha=np.arccos((L1*L1+ min_vl*max_vl - L2*L2)/(2.0*L1*min_vl))
    length=L0 + max_vl*np.sin(theta1-alpha-np.pi/2)
    z= length* np.cos(theta0)
    y=length* np.sin(theta0)

    length = L0+L1+L2
    z= length* np.cos(theta0)
    y=length* np.sin(theta0)
    axs[1].plot(y,z,'b')



    length = L0-L2
    z= length* np.cos(theta0)
    y=length* np.sin(theta0)
    axs[1].plot(y,z,'g')

    axs[1].set_ylabel("z [mm]")
    axs[1].set_xlabel('y [mm]')
    #    y=virtual_leg_length*np.cos(theta0)
    #    z=-virtual_leg_length*np.sin(theta0)
    #    axs[1].plot(y,z)

    plt.savefig('/media/suntao/DATA/Research/P1_workspace/Figures/wkspace2.svg')
    plt.show()

if __name__ == '__main__':
        kinematics()
