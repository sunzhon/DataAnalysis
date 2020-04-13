#!/usr/bin/pyenv
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class So2Oscillator:
    __a1=0.0
    __a2=0.0

    __o1 = 0.01
    __o2 = 0.01
    __b1 = 0.01
    __b2 = 0.01
    _input1 = 0.0
    _input2 = 0.0
    __gamma1, __gamma2 = 0.06, 0.06
    __w11, __w22 = 1.4, 1.4

    def __init__(self, mi):
        self.__MI = mi
        self.__w12 = + 0.18 + self.__MI
        self.__w21 = - 0.18 - self.__MI

    def setInput(self, input):
        self._input1 = input
        self._input2 = input
    def setBias(self, bias, index):
        if(index == 1):
            self.__b1 = bias
        else:
            self.__b2=bias
    def setMI(self, mi):
        self.__MI = mi

    def updateWeights(self):
        self.__w12 = + 0.18 + self.__MI
        self.__w21 = - 0.18 - self.__MI

    def step(self):
        self.updateWeights()
        self.__a1 = self.__w11*self.__o1+self.__w12*self.__o2 + self.__b1 - self.__gamma1 * self._input1 * np.cos(self.__o1)
        self.__a2 = self.__w22*self.__o2+self.__w21*self.__o1 + self.__b2 - self.__gamma2 * self._input2 * np.sin(self.__o2)

        self.__o1 = np.tanh(self.__a1)
        self.__o2 = np.tanh(self.__a2)

    def getOutput(self, i):
        if i == 1:
            return self.__o1
        elif i == 2:
            return self.__o2
        else:
            print("wrong")

    def getActivity(self, i):
        if i == 1:
            return self.__a1
        elif i == 2:
            return self.__a2
        else:
            print("wrong")

    def setOutput(self, index, value):
        if index==1:
            self.__o1 = value
        elif index==2:
            self.__o2 = value
        else:
            print("wrong")

    def setActivity(self, index, value):
        if index==1:
            self.__a1 = value
        elif index==2:
            self.__a2 = value
        else:
            print("wrong")


class PCPG:
    __input = [0.0, 0.0]
    __pcpg_step = [0.0, 0.0]
    __set = [0.0, 0.0]
    __setold = [0.0, 0.0]
    __countup = [0, 0]
    __countdown = [0, 0]
    __countupold = [0, 0]
    __countdownold = [0, 0]
    __diffset = [0, 0]

    __deltaxdown =[80, 80]
    __deltaxup = [100, 100]

    __xup = [0, 0]
    __xdown = [0, 0]

    __yup = [0, 0]
    __ydown = [0, 0]

    __output = [0, 0]
    __beta = [0.5, 0.5]

    def __init__(self):
        self.__output = [0.0, 0.0]

    def setBeta(self, beta):
        self.__beta[0] = beta
        self.__beta[1] = beta

    def setOutput(self, index, value):
        self.__output[i] = value

    def getOutput(self, index):
        return self.__output[index]

    def getInput(self, index):
        return self.__input[index]

    def getBeta(self):
        return self.__beta[0]

    def setInput(self, index, value):
        self.__input[index] = value

    def step(self):
        # CPG post processing
        for i in range(2):
            self.__pcpg_step[i] = self.__input[i]
            self.__setold[i] = self.__set[i]
            self.__countupold[i] = self.__countup[i]
            self.__countdownold[i] = self.__countdown[i]

        # 1) Linear threshold transfer function neuron 1 , or called step function neuron
        if self.__pcpg_step[i] >= self.__beta[i]:
            self.__set[i] = 1.0
        else:
            self.__set[i] = -1.0

            self.__diffset[i] = self.__set[i] - self.__setold[i] # double

            # 2) Count how many step of Swing

            if self.__set[i] == 1.0:
                self.__countup[i] = self.__countup[i] + 1.0 # Delta x0 up
                self.__countdown[i] = 0.0
            elif self.__set[i] == -1.0:
                self.__countdown[i] = self.__countdown[i] + 1.0 #Delta x0 down
                self.__countup[i] = 0.0

            # 3) Memorized the total steps of swing and stance
            if self.__countup[i] == 0.0 and self.__diffset[i] == -2.0 and self.__set[i] == -1.0:
                self.__deltaxup[i] = self.__countupold[i]

            if self.__countdown[i] == 0.0 and self.__diffset[i] == 2.0 and self.__set[i] == 1.0:
                self.__deltaxdown[i] = self.__countdownold[i]

            # 4)Comput y up and down
            self.__xup[i] = self.__countup[i]
            self.__xdown[i] = self.__countdown[i]
            # ////////////Scaling Slope Up calculation

            self.__yup[i] = ((2. / self.__deltaxup[i]) * self.__xup[i]) - 1.0
            self.__ydown[i] = ((-2. / self.__deltaxdown[i]) * self.__xdown[i]) + 1.0
            '''
            if self.__set[i] >= 0.0:
                self.__yup[i] = (1.0-self.__ydown[i])/self.__deltaxup[i] * self.__xup[i] + self.__ydown[i]
            else:
                self.__ydown[i] = (-1.0-self.__yup[i])/self.__deltaxdown[i] * self.__xdown[i] + self.__yup[i]
            '''

            # 5) Combine y up and down
            if self.__set[i] >= 0.0:
                self.__output[i] = self.__output[i] + .1   # self.__yup[i]
            else:
                self.__output[i] = self.__output[i] - .1
            # ********Limit upper and lower boundary
            # self.__output[i] = 1.0 if (self.__output[i] > 1.0) else \
            #    (-1.0 if (self.__output[i] < -1.0) else self.__output[i])


if __name__ == "__main__":
    os1 = So2Oscillator(0.1)
    os2 = So2Oscillator(0.09)
    pcpg = PCPG()
    # os2.setBias(0.01,1);
    # os2.setMI(0.15);

    list11 = []
    list12 = []
    list21 = []
    list22 = []
    list21a = []
    list22a = []
    grf = []
    list31, list32 = [], []
    list33 = []

    for i in range(800):
        if i < 400:  # have feedback
            grf.append(0.6 if os2.getOutput(1) < 0.6 else 0.0)
        elif (i >= 400) and (i< 600):   # without feedback
            grf.append(0.0)
        else:
            grf.append(0.6 if os2.getOutput(1) < 0.6 else 0.0)

        os1.setInput(grf[i])
        os1.step()
        os2.step()
        #os2.setInput(grf[i])
        if i==230:
            #os2.setOutput(1,0.9)
            #os2.setOutput(2,0.5)
            os2.setActivity(1,1.0-1.0*os2.getActivity(1))
            os2.setActivity(2,-1.0*os2.getActivity(2))
            #os2.setOutput(1,np.tanh(os2.getActivity(1)))
            os2.setOutput(2,np.tanh(os2.getActivity(2)))
        list11.append(os1.getOutput(1))
        list12.append(os1.getOutput(2))
        list21.append(os2.getOutput(1))
        list22.append(os2.getOutput(2))
        list21a.append(os2.getActivity(1))
        list22a.append(os2.getActivity(2))
        pcpg.setInput(0, os1.getOutput(1))
        pcpg.setInput(1, os1.getOutput(2))
        #pcpg.step()
        list31.append(pcpg.getOutput(0))
        list32.append(pcpg.getOutput(1))
        list33.append(pcpg.getBeta())

    plt.figure("first")
    plt.plot(list21,list22, 'r--')
    plt.plot(list21a,list22a, 'g--')
    plt.grid()
    plt.legend(["outputs","activity"], loc='upper left')
    print(max(list21a),max(list22a))

    plt.figure("second")
    p1 = plt.plot(list21[100:300],'r')
    p2 = plt.plot(list22[100:300],'b')
    p1a = plt.plot(list21a[100:300],'r:')
    p2a = plt.plot(list22a[100:300],'b:')
    plt.plot(grf[100:300],'k')
    plt.grid()
    plt.legend(["out 1", "out 2"], loc='upper left')

    st=list21[100:3000].index(max(list21[100:300]))
    print(st)
    print(list21[100+st],list22[100+st])
    print(max(list21[100:300]),max(list22[100:300]))

    plt.show()
