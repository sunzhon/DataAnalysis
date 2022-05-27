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
def save_figure(fig_data_folder, fig_name='results', fig_format='.svg'):
    folder_fig = os.path.join(fig_data_folder,'data_visulization', str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime())))
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    figPath= os.path.join(folder_fig, str(localtimepkg.strftime("%H_%M_%S", localtimepkg.localtime()))+"_"+fig_name +'.' +fig_format)
    plt.savefig(figPath)
    plt.show()

    return figPath
