#! /usr/bin/pyenv python
# 

import numpy as np
import pandas as pd
import pdb
import gnureadline
import matplotlib.pyplot as plt



if __name__=='__main__':

    data_file="Trial_3.csv"
    data= pd.read_csv(data_file, sep=',',skip_blank_lines=True,dtype=str)
    pdb.set_trace()

    # 傅里叶变换后，绘制频域图像
    freqs = nf.fftfreq(times.size, times[1] - times[0])
    complex_array = nf.fft(noised_sigs)
    pows = np.abs(complex_array)
    
    plt.subplot(222)
    plt.title('Frequency Domain', fontsize=16)
    plt.ylabel('Power', fontsize=12)
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    # 指数增长坐标画图
    plt.semilogy(freqs[freqs > 0], pows[freqs > 0], c='limegreen', label='Noised')
    plt.legend()



