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


class SO2Control:
    MI=0.08
    weight=np.array([[1.4,0.18+MI],[-(0.18+MI),1.4]])
    alpha=0.16  # scale of the theta
    beta=0.0 # bias of the theta 

    gamma=0.05 # the sensory feedback gain in phase transition mechanism

    def __init__(self,leg_num):
        self.leg_num=leg_num

        self.activity=np.zeros((2,self.leg_num))
        self.output=np.zeros((2,self.leg_num))+0.01
        self.bias=np.zeros((2,self.leg_num))+0.01
        self.theta=np.zeros((2,self.leg_num))

        self.feedback=np.zeros((2,self.leg_num))

        self.force=np.zeros(self.leg_num)

    def updateWeights(self):
        self.weight=np.array([[1.4,0.18+self.MI],[-(0.18+self.MI),1.4]])
        
    def updateActivities(self):
        self.activity=np.dot(self.weight,self.output) + self.bias + self.feedback

    def updateOutputs(self):
        self.output=np.tanh(self.activity)

    def setGRFs(self,force):
        assert(force.shape==self.force.shape)
        self.force=force

    def updateFeedback_phaseTran(self):
        self.feedback= -self.gamma* np.dot(np.vstack((np.cos(self.output[0,:]),np.sin(self.output[1,:]))),np.diag(self.force))

    def step(self):
        self.updateFeedback_phaseTran()
        self.updateWeights()
        self.updateActivities()
        self.updateOutputs()
        self.theta=np.tanh(self.alpha*self.output + self.beta)
        self.theta[[0,1],:]=self.theta[[1,0],:] #O2 to hip, O1 to knee 

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

        sigmal1= f3/20
        sigmal2= f4/20

        f1=f3+sigmal1
        f2=f4+sigmal2

        self.Fz=np.array([f1,f2,f3,f4])

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
        aa=[0,0,0,0]
        #x=np.array([self.f1, self.f2, self.f3, self.f4])
        
        # decide the ai state, which legs are in swing phase
        theta_matrix = np.dot(self.theta_derivation[0,:].reshape(4,1), self.theta_derivation[0,:].reshape(1,4))
        print(self.stepCount)
        self.stepCount = self.stepCount + 1
        
        # whether existing theta are not same phase
        if np.any(theta_matrix<0) or (np.any(theta_matrix==0) and np.any(theta_matrix!=0)):
            aa_true=np.where(self.theta_derivation[0,:] > 0)
            for idx in aa_true[0]:
                aa[idx] = 1
        
            if len(aa_true[0]) > 2: # if having more than three legs in planed swing, the planed swing cannot realize, so all legs in stance phase
                aa=[0,0,0,0]

        if (aa[0]==1 and aa[1]==1) or (aa[0]==1 and aa[2]==1) or (aa[1]==1 and aa[3]==1) or (aa[2]==1 and aa[3]==1): # if adjacent legs in planed swing, the palned swing cannot realize, so all legs in stance phase
            aa=[0,0,0,0]

        A = np.array([[x1,x2,x3,x4],[y1,y2,y3,y4],[1,1,1,1],aa])
        b = np.array([0, 0, self.bodyMass*self.gravityGain,0])

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

        if(sum(aa)==0): # all legs in stance phase
            self.GRFs_simple()
        else: #if have some legs probably in swing phase
            c=np.array([0,0,0,0])
            options={'tol': 2e-1} 
            my_linprog_result=linprog(c, A_eq=A, b_eq=b, bounds=(0,24.5), options=options, method='revised simplex')
            if (my_linprog_result.success==True): # if all legs in stance or line program has solution, then use simple_GRFs
                self.Fz=my_linprog_result.x
                print("linprog success")
            else:
                #pdb.set_trace()
                print(my_linprog_result.x)
                print(self.Fz)
                self.GRFs_simple()
            
        
        #print("aa:", aa)
        #print("Forcesss:",self.Fz)
        #if self.stepCount==799:
        #    pdb.set_trace()
        


        # determine the solution state of the equation 

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

    theta1=[]
    theta2=[]
    
    P1=[]
    P2=[]
    P3=[]
    P4=[]

    GRFs1=[]
    GRFs2=[]
    GRFs3=[]
    GRFs4=[]
    for idx in range(1800):
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


        theta1.append(so2.getTheta()[0,0])
        theta2.append(so2.getTheta()[1,0])

        P1.append(robot.getFeetPosition()[0])
        P2.append(robot.getFeetPosition()[1])
        P3.append(robot.getFeetPosition()[2])
        P4.append(robot.getFeetPosition()[3])

        GRFs1.append(robot.getGRFs()[0])
        GRFs2.append(robot.getGRFs()[1])
        GRFs3.append(robot.getGRFs()[2])
        GRFs4.append(robot.getGRFs()[3])


    fig=plt.figure()
    axs=[]
    axs.append(fig.add_subplot(4,1,1))
    axs.append(fig.add_subplot(4,1,2))
    axs.append(fig.add_subplot(4,1,3))
    axs.append(fig.add_subplot(4,1,4))
    
    #pdb.set_trace()
    axs[0].plot(o11)
    axs[0].plot(o21)
    axs[0].plot(o31)
    axs[0].plot(o41)
    axs[0].legend(['o11','o21','o31','o41'])
    axs[0].set_ylabel('CPGs')

    axs[1].plot(theta1)
    axs[1].plot(theta2)
    axs[1].set_ylabel('Joint commands')

    axs[2].plot([math.fabs(pp[0]) for pp in P1])
    axs[2].plot([math.fabs(pp[0]) for pp in P2])
    axs[2].plot([math.fabs(pp[0]) for pp in P3])
    axs[2].plot([math.fabs(pp[0]) for pp in P4])
    axs[2].legend(['P1','P2','P3','P4'])

    axs[3].plot(GRFs1)
    axs[3].plot(GRFs2)
    axs[3].plot(GRFs3)
    axs[3].plot(GRFs4)
    axs[3].legend(['RF','RH','LF','LH'])
    axs[3].set_ylabel('GRFs')

    plt.figure()
    plt.plot([pp[0] for pp in P1],[pp[2] for pp in P1])
    plt.plot([pp[0] for pp in P2],[pp[2] for pp in P2])
    plt.plot(0,0,'ro')
    plt.show()
