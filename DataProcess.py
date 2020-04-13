#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec
import numpy as np
import scipy 
import os
from itertools import cycle
import pdb 

class DataProcess:

    def __init__(self):
       self.pass_rate=0.001
       self.dataqueue=np.zeros(50)

    def set_low_passrate(self, passrate):
        self.pass_rate=passrate

    def lpprocess(self, data):
        fliterData=[]
        for index,value in enumerate(data):
            fliterData.append(self.pass_rate*value+(1.0-self.pass_rate)*data[index-1])

        return fliterData


    def ydpprocess(self, data):
        fliterData=[]
        index=cycle(range(len(self.dataqueue)))
        for d in data:
                self.dataqueue[next(index)]=d
                fliterData.append(self.dataqueue.mean())

        return fliterData




if __name__=='__main__':
    sd=DataProcess()
    print(sd.ydpprocess([1,2,3,4,5,6,7,8,9,10]))
