import sqlite3
from sqlite3 import Error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

        
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
    folderpath=r"F:\P2 workspace\Experiment data\Force plate test (2-07-19)\\"
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
    for idx in range(1,len(Px)):
        Fx_sum+=Fx[idx]
        Fy_sum+=Fy[idx]
        Fz_sum+=Fz[idx]
    sample_time=np.linspace(0,len(Fx_sum),len(Fx_sum))/freq
    print(len(sample_time))
    print(len(Fx_sum))
    return [sample_time,Fx_sum,Fy_sum,-1.0*Fz_sum]
    
    
if __name__=='__main__':
    filename="Lilibot 0 degree 5 (go)"

    [Time,Fx,Fy,Fz]=read_force_data(filename)
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
    
    plt.show()