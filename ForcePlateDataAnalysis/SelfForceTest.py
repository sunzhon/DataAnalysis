import sqlite3
from sqlite3 import Error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb 

        
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
    folderpath=r"/media/suntao/DATA/Experiment data/Force test for self-organized locomotion//"

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
    return [sample_time,Fx_sum,Fy_sum,Fz_sum]
    
    
if __name__=='__main__':
    #filename="self-organized locomotion low fre MI 0.06"# 61-80
    #filename="self-organized locomotion high fre MI 0.32"# 
    filename="self-organized locomotion generation"# 

    [Time,Fx,Fy,Fz]=read_force_data(filename)
    figsize=(16,8)
    fig=plt.figure(figsize=figsize)
    
    start_seconds=19*200;end_seconds=26*200

    plt.subplot(3,1,1)
    plt.plot(Time[start_seconds:end_seconds],Fx[start_seconds:end_seconds],'r')
    plt.ylabel('Fx [N]')
    plt.title("Ground reaction force")
    plt.subplot(3,1,2)
    plt.plot(Time[start_seconds:end_seconds],Fy[start_seconds:end_seconds],'g')
    plt.ylabel('Fy [N]')
    plt.subplot(3,1,3)
    plt.plot(Time[start_seconds:end_seconds],Fz[start_seconds:end_seconds],'b')
    plt.ylabel('Fz [N]')

    plt.xlabel('Time [s]')
    
    plt.savefig("/media/suntao/DATA/P2 workspace/Experimental Figs/P2Figs/Fig100.eps")
    plt.show()
