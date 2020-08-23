
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
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





import matplotlib.pyplot as plt
import numpy as np

def exam_box_plot():
    # Random test data
    np.random.seed(19680801)
    all_data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
    all_data_2 = [np.random.normal(0.1, std, size=100) for std in range(1, 4)]
    labels = ['x1', 'x2', 'x3']
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    
    # rectangular box plot
    bplot1 = ax1.boxplot(all_data, 
                         positions=[1-0.125, 3-0.125, 5-0.125],
                         widths=0.25,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         #labels=labels
                        )  # will be used to label x-ticks
    ax1.set_title('Rectangular box plot')
    

    bplot2 = ax1.boxplot(all_data_2, 
                          positions=[1.125, 3.125, 5.125],
                         widths=0.25,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                        #labels=labels
                        )  # will be used to label x-ticks

    # notch shape box plot
    bplot3 = ax2.boxplot(all_data,
                         notch=True,  # notch shape
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    ax2.set_title('Notched box plot')
    
    # fill with colors
    colors = ['pink', 'lightblue']
    for bplot, color in zip((bplot1, bplot2), colors):
        pdb.set_trace()
        for patch in bplot['boxes']:
            patch.set_facecolor(color)
    
    # adding horizontal grid lines
    for ax in [ax1, ax2]:
        ax.yaxis.grid(True)
        ax.set_xlabel('Three separate samples')
        ax.set_ylabel('Observed values')
    
    plt.show()





if __name__=='__main__':
    exam_box_plot()
