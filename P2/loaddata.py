#! /usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import re
import pathlib
import os
import sys
import time
import pdb
#0) load input parameters
def loadData(fileName,columnsName):
    '''
    load data from a file
    fileName: the name of file that you want to read
    columnsName: it the column name of the file
    Note: the args of sys is file_id and date of the file
    '''
    cdate_time=time.localtime(time.time())
    cdata_date=str(cdate_time.tm_mon)+str(cdate_time.tm_mday)

    if len(sys.argv)==3:
        data_date=str(sys.argv[2])
    else:
        data_date=cdata_date
        
    #1) load data from file
    data_files = "/home/suntao/workspace/experiment_data/"
    #data_files = "/home/suntao/Exp1withoutACITerm/"
    results_files = "/home/suntao/workspace/experiment_result/"
    os.chdir(data_files)
    folders=os.listdir()
    resource_data=[]
    rows_num=[]
    for folder in folders:
        if(not (re.match(data_date,folder)==None)):
            data_file = data_files + folder +"/"+fileName + ".csv"
            resource_data.append(pd.read_csv(data_file, sep='\t', index_col=0,header=None, names=columnsName, skip_blank_lines=True,dtype=str))
            rows_num.append(resource_data[-1].shape[0])# how many rows

    fine_data = []
    min_rows=min(rows_num)
    read_rows=min_rows-1

    for data in resource_data:
        fine_data.append(data.iloc[0:read_rows,:].astype(float))# 数据行对齐
    return fine_data

