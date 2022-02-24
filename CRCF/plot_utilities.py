#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import gridspec
import os
import pdb 
import termcolor
import gnureadline

plt.rc('font',family='Arial')
import pandas as pd
import re
import time as localtimepkg
from brokenaxes import brokenaxes
from mpl_toolkits import mplot3d
from matplotlib import animation
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import warnings
from matplotlib.animation import FuncAnimation
import matplotlib.font_manager

'''
Save figures

'''
def save_figure(fig_data_folder,fig_name):
    folder_fig = os.path.join(fig_data_folder,'data_visulization/')
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H_%M_%S", localtimepkg.localtime())) + '_'+fig_name+'.svg'
    plt.savefig(figPath)
    plt.show()
