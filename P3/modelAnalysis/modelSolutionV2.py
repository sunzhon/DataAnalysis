#! /bin/usr/pyenv python
# -*- coding:utf-8 -*-

import sys
import numpy as np
import gnureadline
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import pdb
from scipy.optimize import linprog
import math
from cvxopt import matrix, solvers


class SO2Control:
    MI=0.08
    weight=np.array([[1.4,0.18+MI],[-(0.18+MI),1.4]])
    alpha=0.2  # scale of the theta
    beta=0.0 # bias of the theta 

    gamma=0.05 # the sensory feedback gain in phase transition mechanism

    def __init__(self,leg_num):
        self.leg_num=leg_num

        self.activity=np.zeros((2,self.leg_num))
        self.output=np.zeros((2,self.leg_num))+0.01
        self.bias=np.zeros((2,self.leg_num))+0.01
        self.theta=np.zeros((2,self.leg_num))

        self.feedback=np.zeros((2,self.leg_num))

        self.force=np.zeros((self.leg_num,1))
        self.previous_force=np.zeros((self.leg_num,1))
        #self.force=np.array([1.0,0.1,0.0,0.0]).reshape(4,-1)
        self.force_threshold = 0.4594;
        self.stepCount=0

    def updateWeights(self):
        self.weight=np.array([[1.4,0.18+self.MI],[-(0.18+self.MI),1.4]])
        
    def updateActivities(self):
        self.activity=np.dot(self.weight,self.output) + self.bias + self.feedback

    def updateOutputs(self):
        self.output=np.tanh(self.activity)

    def setGRFs(self,force):
        assert(force.shape==self.force.shape)
        self.previous_force=self.force
        self.force= self.previous_force*0.5+force*0.5

    def updateFeedback_phaseTran(self):
        self.feedback= -self.gamma* np.dot(np.vstack((np.cos(self.output[0,:]),np.sin(self.output[1,:]))),np.diag(self.force[:,0]))

    def updateFeedback_phaseReset(self):
        self.updateWeights()
        self.updateActivities()
        for i in range(self.leg_num):
            if((self.previous_force[i,0]< self.force_threshold) and (self.force[i,0] >= self.force_threshold)):
                a1= 1.0- self.activity[0,i]
                a2= -self.activity[1,i]
                print("stepCount:, leg_num:, previous_force:,current_force:", self.stepCount, i, self.previous_force[i,0], self.force[i,0])
            else:
                a1=0.0
                a2=0.0
            self.feedback[0,i]=a1
            self.feedback[1,i]=a2

    def step(self):
        #self.updateFeedback_phaseTran()
        self.updateFeedback_phaseReset()
        self.updateWeights()
        self.updateActivities()
        self.updateOutputs()
        self.theta=np.tanh(self.alpha*self.output + self.beta)
        self.theta[[0,1],:]=self.theta[[1,0],:] #O2 to hip, O1 to knee 
        self.stepCount+=1

    def getOutput(self):
        return self.output

    def getTheta(self):
        return self.theta



class robotModel:
    #body parameters
    bodyMass=2.5 #kg
    body_length=0.30
    body_width=0.175
    body_height=0.05

    #environment parameters
    gravityGain=9.8

    #leg parameters
    L0=40.53/1000
    L1=70.02/1000;
    L2=86.36/1000;
    Fz=np.array([0,0,0,0])
    #robot position and orientatiion
    Pc=np.zeros(6)

    # robot joint commands
    theta=np.zeros((2,4))
    theta_old=theta
    theta_derivation=theta
    theta1_initial=-0.2; theta2_initial=0.4
    
    stepCount=0
    def Kinematics(self):
        '''
        Kinematics of quadruped robots having four legs,each leg has two joints, hip for forward and backward movement and knee for extension and flexion movement
        '''


        Px= self.L1*np.sin(self.theta[0,0]) + self.L2*np.sin(self.theta[1,0]+self.theta[0,0])
        Pz= -self.L0 - (self.L1*np.cos(self.theta[0,0]) + self.L2*np.cos(self.theta[1,0] + self.theta[0,0]))
        Px=Px+self.body_length/2
        Py=-self.body_width/2
        Pz=Pz-self.body_height/2
        self.P1=np.array([Px,Py,Pz])

        Px=  self.L1*np.sin(self.theta[0,1]) + self.L2*np.sin(self.theta[1,1]+self.theta[0,1])
        Pz= -self.L0 - (self.L1*np.cos(self.theta[0,1]) + self.L2*np.cos(self.theta[1,1]+self.theta[0,1]))
        Px=Px-self.body_length/2
        Py=-self.body_width/2
        Pz=Pz-self.body_height/2
        self.P2=np.array([Px,Py,Pz])

        Px= self.L1*np.sin(self.theta[0,2]) + self.L2*np.sin(self.theta[1,2]+self.theta[0,2])
        Pz= -self.L0 - (self.L1*np.cos(self.theta[0,2]) + self.L2*np.cos(self.theta[1,2]+self.theta[0,2]))
        Px=Px+self.body_length/2
        Py=self.body_width/2
        Pz=Pz-self.body_height/2
        self.P3=np.array([Px,Py,Pz])


        Px= self.L1*np.sin(self.theta[0,3]) + self.L2*np.sin(self.theta[1,3]+self.theta[0,3])
        Pz= -self.L0 - (self.L1*np.cos(self.theta[0,3]) + self.L2*np.cos(self.theta[1,3]+self.theta[0,3]))
        Px=Px-self.body_length/2
        Py=self.body_width/2
        Pz=Pz-self.body_height/2
        self.P4=np.array([Px,Py,Pz])
        
    def GRFs_simple(self):
        # just consider the vertical direction GRFs
        x1=math.fabs(self.P1[0])
        x2=math.fabs(self.P2[0])
        x3=math.fabs(self.P3[0])
        x4=math.fabs(self.P4[0])

        f3=self.bodyMass*self.gravityGain/2*(x2+x4)/(x1+x2+x3+x4)
        f4=self.bodyMass*self.gravityGain/2*(x1+x3)/(x1+x2+x3+x4)

        sigmal1= np.random.normal(0.1,f3/10)
        sigmal2= np.random.normal(0.1,f4/10)
        sigmal3= np.random.normal(0.1,f4/10)
        sigmal4= np.random.normal(0.1,f4/10)

        f1=f3+sigmal1
        f2=f4+sigmal2
        f3=f3+sigmal3
        f4=f4+sigmal4

        self.Fz=np.array([f1,f2,f3,f4]).reshape(4,-1)

    def GRFs_Matrix(self):
        # using this matrix
        x1=self.P1[0]
        x2=self.P2[0]
        x3=self.P3[0]
        x4=self.P4[0]

        y1=self.P1[1]
        y2=self.P2[1]
        y3=self.P3[1]
        y4=self.P4[1]
        
        # origianl
        aa=[0.0, 0.0, 0.0, 0.0]
        
        # decide the ai state, which legs are in swing phase
        theta_matrix = np.dot(self.theta_derivation[0,:].reshape(4,1), self.theta_derivation[0,:].reshape(1,4))
        #print(self.stepCount)
        self.stepCount = self.stepCount + 1
        
        # whether existing theta are not same phase
        if np.any(theta_matrix<0) or (np.any(theta_matrix==0) and np.any(theta_matrix!=0)):
            aa_true=np.where(self.theta_derivation[0,:] > 0)
            for idx in aa_true[0]:
                aa[idx] = 1.0
        
            if len(aa_true[0]) > 2: # if having more than three legs in planed swing, the planed swing cannot realize, so all legs in stance phase
                aa=[0.0,0.0,0.0,0.0]

        if (aa[0]==1.0 and aa[1]==1.0) or (aa[0]==1.0 and aa[2]==1.0) or (aa[1]==1.0 and aa[3]==1.0) or (aa[2]==1.0 and aa[3]==1.0): # if adjacent legs in planed swing, the palned swing cannot realize, so all legs in stance phase
            aa=[0.0,0.0,0.0,0.0]




        '''
        #pdb.set_trace()
        rank_Ab = np.linalg.matrix_rank(np.hstack((A,b.reshape(4,1))))
        rank_A = np.linalg.matrix_rank(A)
        if rank_Ab==rank_A:# existing solution
            if rank_Ab<4:
                self.GRFs_simple()
                print("Forcesss:",self.Fz)
            if rank_Ab==4:
                pdb.set_trace()
                f=np.linalg.solve(A, b)
                #self.Fz=f
                print("Force:",my_linprog_result.x)
        else:# No solution
            print("No solution")
        '''

        '''
        if (sum(aa)==0):
            # 四脚着地
            self.GRFs_simple()
            #print("simple")
        else:
            A = np.array([[x1,x2,x3,x4],[y1,y2,y3,y4],[1,1,1,1]])
            b = np.array([0, 0, self.bodyMass*self.gravityGain])
            c=np.array(aa)
            my_linprog_result=linprog(c,A_eq=A,b_eq=b,bounds=(0,24.5), method='revised simplex')
            #pdb.set_trace()
            if(my_linprog_result.success):
                self.Fz=my_linprog_result.x
                self.Fz=self.Fz.reshape(4,-1)
                #print(aa)
                #print(self.Fz)
                pdb.set_trace()
            else:
                self.GRFs_simple()
            
        '''

        Q = 2.0 * matrix([[3.0*(1.0-aa[0]), 0.0, 0.0, 0.0], [-2.0*(1.0-aa[0])*(1.0-aa[1]), 3.0*(1.0-aa[1]), 0.0, 0.0], [-2.0*(1.0-aa[0])*(1.0-aa[2]), -2.0*(1.0-aa[1])*(1.0-aa[2]), 3.0*(1.0-aa[2]), 0.0],[-2.0*(1.0-aa[0])*(1.0-aa[3]), -2.0*(1.0-aa[1])*(1.0-aa[3]), -2.0*(1.0-aa[2])*(1.0-aa[3]), 3.0*(1.0-aa[3])]])
        p = matrix(aa)
        G = matrix([[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, -1.0]])
        h = matrix([0.0, 0.0, 0.0, 0.0])

        A = np.array([[x1,x2,x3,x4],[y1,y2,y3,y4],[1,1,1,1]])
        b = np.array([0, 0, self.bodyMass*self.gravityGain])

        A = matrix(A)#原型为cvxopt.matrix(array,dims)，等价于A = matrix([[1.0],[1.0]]）
        b = matrix(b)

        result = solvers.qp(Q,p,G,h,A,b, options={'show_progress':False})
        self.Fz = np.array(result['x']).reshape(4,-1)
        self.Fz[0,0] +=np.random.normal(0.1,0.1)
        self.Fz[1,0] +=np.random.normal(0.1,0.1)
        self.Fz[2,0] +=np.random.normal(0.1,0.1)
        self.Fz[3,0] +=np.random.normal(0.1,0.1)


    def setCommand(self,theta):
        self.theta_old=self.theta
        self.theta=theta
        self.theta[0,:]=self.theta[0,:]+self.theta1_initial
        self.theta[1,:]=self.theta[1,:]+self.theta2_initial

        self.theta_derivation=(self.theta-self.theta_old)*1/60 # 60Hz 

    def getGRFs(self):
        return self.Fz

    def getFeetPosition(self):
        return [self.P1,self.P2,self.P3,self.P4]

    def step(self):
        self.Kinematics()
        self.GRFs_simple();
        self.GRFs_Matrix();
        
        
if __name__=="__main__":

    so2=SO2Control(4)
    robot=robotModel()

    o11=[]
    o12=[]
    o21=[]
    o22=[]
    o31=[]
    o32=[]
    o41=[]
    o42=[]

    theta11=[]
    theta12=[]
    theta21=[]
    theta22=[]
    theta31=[]
    theta32=[]
    theta41=[]
    theta42=[]
    
    P1=[]
    P2=[]
    P3=[]
    P4=[]

    GRFs1=[]
    GRFs2=[]
    GRFs3=[]
    GRFs4=[]
    count_length=1200;#
    for idx in range(count_length):
        so2.step()
        temp_theta=so2.getTheta()

        robot.setCommand(temp_theta)
        robot.step()
        temp_GRFs=robot.getGRFs()

        so2.setGRFs(0.1*temp_GRFs)

        o11.append(so2.getOutput()[0,0])
        o12.append(so2.getOutput()[1,0])
        o21.append(so2.getOutput()[0,1])
        o22.append(so2.getOutput()[1,1])
        o31.append(so2.getOutput()[0,2])
        o32.append(so2.getOutput()[1,2])
        o41.append(so2.getOutput()[0,3])
        o42.append(so2.getOutput()[1,3])


        theta11.append(so2.getTheta()[0,0])
        theta12.append(so2.getTheta()[1,0])

        theta21.append(so2.getTheta()[0,1])
        theta22.append(so2.getTheta()[1,1])

        theta31.append(so2.getTheta()[0,2])
        theta32.append(so2.getTheta()[1,2])

        theta41.append(so2.getTheta()[0,3])
        theta42.append(so2.getTheta()[1,3])

        P1.append(robot.getFeetPosition()[0])
        P2.append(robot.getFeetPosition()[1])
        P3.append(robot.getFeetPosition()[2])
        P4.append(robot.getFeetPosition()[3])

        GRFs1.append(robot.getGRFs()[0])
        GRFs2.append(robot.getGRFs()[1])
        GRFs3.append(robot.getGRFs()[2])
        GRFs4.append(robot.getGRFs()[3])


    # Plot
    freq=60 # Hz
    time=np.linspace(0,count_length/freq,count_length)
   
    # colors
    c4_1color=(46/255.0, 77/255.0, 129/255.0)
    c4_2color=(0/255.0, 198/255.0, 156/255.0)
    c4_3color=(255/255.0, 1/255.0, 118/255.0)
    c4_4color=(225/255.0, 213/255.0, 98/255.0)
    colors=[c4_1color, c4_2color, c4_3color, c4_4color]

    figsize=(7.5,5)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    columns_axs=1
    gs1=gridspec.GridSpec(6,columns_axs)
    gs1.update(hspace=0.2,top=0.97,bottom=0.1,left=0.1,right=0.98)
    axs=[]
    for idx in range(columns_axs):# how many columns, depends on       the experiment_classes
        axs.append(fig.add_subplot(gs1[0:2,idx]))
        axs.append(fig.add_subplot(gs1[2:4,idx]))
        axs.append(fig.add_subplot(gs1[4:6,idx]))


    #pdb.set_trace()


    idx=0
    axs[idx].plot(time,o11, color=c4_1color)
    axs[idx].plot(time,o21, color=c4_2color)
    axs[idx].plot(time,o31, color=c4_3color)
    axs[idx].plot(time,o41, color=c4_4color)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_ylabel(u'CPGs')
    axs[idx].set_yticks([-1.0,0.0,1.0])
    axs[idx].legend(['RF','RH','LF', 'LH'],ncol=4,loc='center right')
    axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])



    idx=1
    axs[idx].plot(time,theta11, color=c4_1color)
    axs[idx].plot(time,theta21, color=c4_2color)
    axs[idx].plot(time,theta31, color=c4_3color)
    axs[idx].plot(time,theta41, color=c4_4color)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_ylabel(u'MNs')
    axs[idx].set_yticks([-0.5,-0.2,0.1])
    axs[idx].legend(['RF','RH','LF', 'LH'],ncol=4,loc='center right')
    axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])


    '''
    axs[2].plot([math.fabs(pp[0]) for pp in P1])
    axs[2].plot([math.fabs(pp[0]) for pp in P2])
    axs[2].plot([math.fabs(pp[0]) for pp in P3])
    axs[2].plot([math.fabs(pp[0]) for pp in P4])
    axs[2].legend(['P1','P2','P3','P4'])
    axs[2].set_ylabel('Feet X position')
    '''

    idx=2
    axs[idx].plot(time,GRFs1, color=c4_1color)
    axs[idx].plot(time,GRFs2, color=c4_2color)
    axs[idx].plot(time,GRFs3, color=c4_3color)
    axs[idx].plot(time,GRFs4, color=c4_4color)
    axs[idx].grid(which='both',axis='x',color='k',linestyle=':')
    axs[idx].grid(which='both',axis='y',color='k',linestyle=':')
    axs[idx].set_ylabel(u'GRFs [N]')
    axs[idx].set_yticks([0,5.0,15.0])
    axs[idx].legend(['RF','RH','LF', 'LH'],ncol=4, loc='center right')
    #axs[idx].set_xticklabels([])
    axs[idx].set(xlim=[min(time),max(time)])
    axs[idx].set_xlabel('Time [s]')


    '''
    plt.figure()
    plt.plot([pp[0] for pp in P1],[pp[2] for pp in P1])
    plt.plot([pp[0] for pp in P2],[pp[2] for pp in P2])
    plt.plot(0,0,'ro')
    '''
    plt.show()

