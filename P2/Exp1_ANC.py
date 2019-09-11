#! /usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
sns.set(color_codes = True)
import loaddata as LD
import pandas as pd
import numpy as np
import os
import sys







    #0) load input parameters
    #0.1) get the current date and time 

if len(sys.argv)>=2:
    run_id = int(sys.argv[1]) # The id of the experiments
else:
    run_id = 0

fileName="controlfile_ANC"
culumnsName=['A','B','C','D']
fine_data_w12,fine_data_w13,fine_data_w14,fine_stability=[],[],[],[]

resource_data=LD.loadData(fileName,culumnsName)
read_rows=resource_data[0].shape[0];
for data in resource_data:
    fine_data_w12.append(data.iloc[0:read_rows,0].astype(float))
    fine_data_w13.append(data.iloc[0:read_rows,1].astype(float))
    fine_data_w14.append(data.iloc[0:read_rows,2].astype(float))
    fine_stability.append(data.iloc[0:read_rows,3].astype(float))

fine_data_w12_array=np.array(fine_data_w12)
fine_data_w13_array=np.array(fine_data_w13)
fine_data_w14_array=np.array(fine_data_w14)
fine_stability_array=np.array(fine_stability)

times = np.linspace(0,read_rows,read_rows)/40 # The controller is 40Hz

fine_data_w12_df= pd.DataFrame(fine_data_w12_array.T, index=times, columns=range(0,fine_data_w12_array.shape[0]))
fine_data_w13_df= pd.DataFrame(fine_data_w13_array.T, index=times, columns=range(0,fine_data_w13_array.shape[0]))
fine_data_w14_df= pd.DataFrame(fine_data_w14_array.T, index=times, columns=range(0,fine_data_w14_array.shape[0]))
fine_stability_df= pd.DataFrame(fine_stability_array.T, index=times, columns=range(0,fine_stability_array.shape[0]))

fig= plt.figure()
sns.set_context('paper')

fig.add_subplot(2,1,1)
#sns.lineplot(data=fine_data_w12_df)
sns.tsplot(data=abs(fine_data_w12_df.to_numpy().T[run_id]),time=times,condition='RP12', color='r')
sns.tsplot(data=abs(fine_data_w13_df.to_numpy().T[run_id]),time=times,condition='RP13', color='b')
sns.tsplot(data=abs(fine_data_w14_df.to_numpy().T[run_id]),time=times,condition='RP14', color='g')
plt.xlabel("Time[s]")
plt.ylabel("Relative phase[rad]")

plt.title("Relative phase between legs and its variance")
fig.add_subplot(2,1,2)
sns.tsplot(data=fine_stability_df.to_numpy().T[run_id],time=times,condition='stability', color='y')
#sns.lineplot(data=fine_data_w12_df)
#plt.axis([0, max(times),-1.0,10.0])
plt.xlabel("Time[s]")
plt.ylabel("Variance")


if __name__=="__main__":
    plt.show()
