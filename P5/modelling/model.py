#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import gridspec
import os
import gnureadline
import pdb 
plt.rc('font',family='Arial')
import pandas as pd
from matplotlib.animation import FuncAnimation
import re
from brokenaxes import brokenaxes
import time as localtimepkg
import termcolor
import seaborn as sns



if __name__=="__main__":
    from keras.models import Sequential

    model = Sequential()
    from keras.layers import Dense

    model.add(Dense(units=64, activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
