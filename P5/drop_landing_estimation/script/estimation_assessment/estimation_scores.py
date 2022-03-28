import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import os
import yaml
import h5py

import seaborn as sns
import copy
import re

from sklearn.metrics import r2_score, mean_squared_error as mse
def get_scores(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse(y_true, y_pred))
    mae = np.mean(abs((y_true - y_pred)))
    r_rmse = rmse / (y_true.max() - y_true.min())

    return r2, rmse, mae, r_rmse
    

def combine_actual_prediction_data(labels,predictions,labels_names,fig_save_folder=None,**args):
    """
    transfer actual and prediction data into pandas dataframe and combine them into a pd dataframe

    """
    print("Plot the comparison between estimation and ground-truth results")
    
    #1) load dataset
    #i) Pandas DataFrame of the above datasets
    pd_labels = pd.DataFrame(labels,columns=labels_names)
    pd_predictions = pd.DataFrame(data=predictions,columns=labels_names)

    for idx in range(pd_labels.values.shape[1]):
        print("scores (r2, rmse, mae, r_rmse):", get_scores(pd_labels.values[:,idx],pd_predictions.values[:,idx]))


    #ii) add time and legends to pd_labels and pd_predictions
    freq=100.0;
    print("NOTE: Sampe frequency is 100 Hz in default")
    Time=np.linspace(0,labels.shape[0]/freq,num=labels.shape[0])

    pd_labels['Time']=Time
    pd_predictions['Time']=Time

    pd_labels['Legends']='Actual'
    pd_predictions['Legends']='Prediction'

    #iii) organize labels and predictions into a pandas dataframe
    pd_labels_predictions=pd.concat([pd_labels,pd_predictions],axis=0)
    pd_labels_predictions=pd_labels_predictions.melt(id_vars=['Time','Legends'],var_name='Variables',value_name='Values')

    return pd_labels_predictions
    



def plot_estimation_comparison(labels,predictions,labels_names,fig_save_folder=None,**args):
    """
    Plot the comparison between actual and prediction reslus

    """
    print("Plot the comparison between estimation and ground-truth results")
    pd_labels_predictions = combine_actual_prediction_data(labels,predictions,labels_names)
    
    #2) plot dataset and save figures
    #i) plot estimation results
    g=sns.FacetGrid(data=pd_labels_predictions,col='Variables',hue='Legends',sharey=False)
    g.map_dataframe(sns.lineplot,'Time','Values')
    g.add_legend()

    #ii) save figure
    if(fig_save_folder!=None):
        folder_fig = fig_save_folder+"/"
    else:
        folder_fig="./"
    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    # whether define the figsave_file
    if('prediction_file' in args.keys()):
        figPath=args['prediction_file']
    else:
        figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + '_test_prediction.svg'
    plt.savefig(figPath)


def plot_estimation_error(labels,predictions,labels_names,fig_save_folder=None,**args):
    """
    Plot the error between the atual and prediction

    """
    print("Plot the error beteen the actual and prediction results")

    #i) calculate estimation errors statistically: rmse. It is an average value
    pred_error=predictions-labels
    pred_mean=np.mean(pred_error,axis=0)
    pred_std=np.std(pred_error,axis=0)
    pred_rmse=np.sqrt(np.sum(np.power(pred_error,2),axis=0)/pred_error.shape[0])
    pred_rrmse=pred_rmse/np.mean(labels,axis=0)*100.0
    print("mean of ground-truth:",np.mean(labels))
    print("mean: {.2f}, std: {.2f}, RMSE: {.2f}, rRMSE: {.2f} of the errors between estimation and ground truth",pred_mean, pred_std, pred_rmse, pred_rrmse)


    #ii) calculate estimation errors realtime: normalized_absolute_error (nae)= abs(labels-prediction)/labels, along the time, each column indicates a labels
    nae = np.abs(pred_error)#/labels
    pd_nae=pd.DataFrame(data=nae,columns=labels_names);pd_nae['time']=Time
    pd_nae=pd_nae.melt(id_vars=['time'],var_name='GRF error [BW]',value_name='vals')


    #iii) plot absolute error and noramlized error (error-percentage)
    g=sns.FacetGrid(data=pd_nae,col='GRF error [BW]',col_wrap=3,sharey=False)
    g.map_dataframe(sns.lineplot,'time','vals')


    #ii) save figure
    if(fig_save_folder!=None):
        folder_fig = fig_save_folder+"/"
    else:
        folder_fig="./"

    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)

    # figure save file
    if('prediction_error_file' in args.keys()):
        figPath=args['prediction_error_file']
    else:
        figPath= folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + '_test_mes.svg'

    plt.savefig(figPath)



