import sqlite3
from sqlite3 import Error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb 
from cycler import cycler
import matplotlib as mpl
from matplotlib import gridspec

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
 
    return conn

def select_all_data(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM DATA")
 
    rows = cur.fetchall()
    return rows
    #for row in rows:
    #    print(row)
 
 
def select_data_by_priority(conn, step):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM DATA WHERE ID=?", (step,))
 
    rows = cur.fetchall()
    return rows
    
    #for row in rows:
    #    print(row)
def read_force_data(filename):
    freq=200# Hz sample
    #folderpath=r"F:\P2 workspace\Experiment data\Force plate test (2-07-19)\\"
    folderpath=r"/media/suntao/DATA/Experiment data/Force plate test (2-07-19)//"
    folderpath=r"/media/suntao/DATA/Research/Experiment data/Force test for self-organized locomotion//"

    filepath=folderpath + filename + r".db"
    conn=create_connection(filepath)
    with conn:
        datas=select_all_data(conn)
        pd_datas=pd.DataFrame(datas)
        
    Fx,Fy,Fz=[],[],[]
    for idx in range(3,82,4):
        Fx.append(pd_datas.iloc[:,idx])
        Fy.append(pd_datas.iloc[:,idx+1])
        Fz.append(pd_datas.iloc[:,idx+2])
    Fx_sum=Fx[0];Fy_sum=Fy[0];Fz_sum=Fz[0]
    for idx in range(1,len(Fx)):
        Fx_sum+=Fx[idx]
        Fy_sum+=Fy[idx]
        Fz_sum+=Fz[idx]
    sample_time=np.linspace(0,len(Fx_sum),len(Fx_sum))/freq
    print(len(sample_time))
    print(len(Fx_sum))
    return [sample_time,Fx_sum,Fy_sum,-1.0*Fz_sum, Fz]
    
    
if __name__=='__main__':
    filename="self-organized locomotion generation"

    [Time,Fx,Fy,Fz,Fzz]=read_force_data(filename)
    figsize=(16,8)
    fig=plt.figure(figsize=figsize)
    
    plt.subplot(3,1,1)
    plt.plot(Time,Fx)
    plt.subplot(3,1,2)
    plt.plot(Time,Fy)
    plt.subplot(3,1,3)
    plt.plot(Time,Fz)

    plt.xlabel('Time [s]')
    plt.ylabel('Fz [N]')
    '''
    figsize=(10,4.5)#8.6614
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(4,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.08,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))
    axs.append(fig.add_subplot(gs1[2:4,0]))
    '''



    colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf',
          '#1a55FF',
          '#1a55eF',
          '#1a55dF',
          '#1a55cF',
          '#1a55bF',
          '#1a55aF',
          '#d6272B',
          '#d6272C',
          '#d6272F',
          '#1a550F'
             ]

    # Set the plot curve with markers and a title
    figsize=(10,4.5)#8.6614
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1=gridspec.GridSpec(4,1)#13
    gs1.update(hspace=0.18,top=0.95,bottom=0.095,left=0.08,right=0.98)
    axs=[]
    axs.append(fig.add_subplot(gs1[0:2,0]))
    axs.append(fig.add_subplot(gs1[2:4,0]))


    axs[0].plot(Time, -1.0*Fzz[12], color=colors[1]) #, marker='o', label=str(cases[i]))
    axs[1].plot(Time, -1.0*Fzz[14], color=colors[2]) #, marker='o', label=str(cases[i]))

    plt.savefig('/media/suntao/DATA/Research/P1_workspace/Figures/Current_VGRF_GRF.svg')

    plt.show()





