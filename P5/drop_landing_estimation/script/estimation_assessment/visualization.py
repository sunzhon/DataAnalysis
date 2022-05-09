#! /bin/pyenv python
#coding: --utf-8

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

from estimation_assessment.scores import *

from vicon_imu_data_process.const import SAMPLE_FREQUENCY

'''
Plot the estimation results

'''
def plot_prediction(pd_labels, pd_predictions, testing_folder):

    #i) load hyper parameters
    hyperparams_file = os.path.join(testing_folder,"hyperparams.yaml")
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()

    #ii) create file name to save plot results
    test_subject_ids_names = hyperparams['test_subject_ids_names']
    prediction_file = os.path.join(testing_folder, test_subject_ids_names[0] + '_estimation.svg')

    #iii) plot the estimation results and errors
    plot_actual_estimation_curves(pd_labels, 
                                    pd_predictions, 
                                    testing_folder,
                                    figtitle=prediction_file)




def plot_prediction_statistic(features, labels, predictions,testing_folder):
    '''
    This function calculate the error between predicted and ground truth, and plot them for comparison
    '''
    
    # load hyperparameters, Note the values in hyperparams become string type
    hyperparams_file = os.path.join(testing_folder,"hyperparams.yaml")
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    
    # hyper parameters    
    features_names=hyperparams['features_names']
    labels_names=hyperparams['labels_names']
    
    
    # test subject idx, which one is for testing
    test_subject_ids_names = hyperparams['test_subject_ids_names']
    test_subject_ids_str=''
    for ii in test_subject_ids:
        test_subject_ids_str+='_'+str(ii)

    pd_error, pd_NRMSE = estimation_accuracy(predictions,labels,labels_names)

    plot_estimation_accuracy(pd_error, pd_NRMSE)
    


def plot_estimation_accuracy(pd_error, pd_NRMSE):
    # create experiment results folder
    # MAE
    fig=plt.figure(figsize=(10,2))
    style = ['darkgrid', 'dark', 'white', 'whitegrid', 'ticks']
    sns.set_style(style[4],{'grid.color':'k'})
    sns.catplot(data=pd_error,kind='bar', palette="Set3").set(ylabel='Absolute error [deg]')
    #plt.text(2.3,1.05, r"$\theta_{ae}(t)=abs(\hat{\theta}(t)-\theta)(t)$",horizontalalignment='center', fontsize=20)
    test_subject_ids_names = hyperparams['test_subject_ids_names']
    savefig_file=testing_folder+'/sub'+str(test_subject_ids_str)+'_mae.svg'
    plt.savefig(savefig_file)
    
    # NRMSE
    fig=plt.figure(figsize=(10,3))
    sns.catplot(data=pd_NRMSE,kind='bar', palette="Set3").set(ylabel='NRMSE [%]')
    #plt.text(2.3, 2.6, r"$NRMSE=\frac{\sqrt{\sum_{t=0}^{T}{\theta^2_{ae}(t)}/T}}{\theta_{max}-\theta_{min}} \times 100\%$",horizontalalignment='center', fontsize=20)
    savefig_file=testing_folder+'/sub'+str(test_subject_ids_str)+'_nrmse.svg'
    plt.savefig(savefig_file)



    
def estimation_accuracy(estimation, actual, labels_names):
    # Plot the statistical results of the estimation results and errors
    error=abs(estimation-actual)
    pd_error=pd.DataFrame(data=error,columns=labels_names)
    NRMSE=100.0*np.sqrt(pd_error.apply(lambda x: x**2).mean(axis=0).to_frame().transpose())/(actual.max(axis=0)-actual.min(axis=0))
    #*np.ones(pd_error.shape)*100
    pd_NRMSE=pd.DataFrame(data=NRMSE, columns = [col for col in list(pd_error.columns)])

    return pd_error, pd_NRMSE 



'''
Plot the history metrics in training process

'''

def plot_history(history):

    history_dict = history
    print(history_dict.keys())
    plt.plot(history_dict['loss'],'r')
    plt.plot(history_dict['val_loss'],'g')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(['train loss', 'valid loss'])

    plt.figure()
    plt.plot(history_dict['mae'],'r')
    plt.plot(history_dict['val_mae'],'g')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.legend(['train mae', 'valid mae'])

    print('Max train and validtion MAE: {:.4f} and {:.4f}'.format(max(history_dict['mae']),max(history_dict['val_mae'])))






def plot_actual_estimation_curves(pd_labels, pd_predictions, testing_folder, fig_save_folder=None,**kwargs):
    """
    Plot the comparison between actual and prediction reslus

    """
    #1) load dataset
    #i) add time and legends to pd_labels and pd_predictions
    if(not isinstance(pd_labels,pd.DataFrame)):
        pd_labels = pd.DataFrame(pd_labels)
        pd_labels = pd.DataFrame(pd_labels)

    pd_labels = copy.deepcopy(pd_labels)
    pd_predictions = copy.deepcopy(pd_predictions)

    Time=np.linspace(0,pd_labels.shape[0]/SAMPLE_FREQUENCY,num=pd_labels.shape[0])
    pd_labels['Time']=Time
    pd_predictions['Time']=Time

    pd_labels['Legends']='Actual'
    pd_predictions['Legends']='Prediction'

    #iii) organize labels and predictions into a pandas dataframe
    pd_labels_predictions=pd.concat([pd_labels,pd_predictions],axis=0)
    pd_labels_predictions=pd_labels_predictions.melt(id_vars=['Time','Legends'],var_name='Variables',value_name='Values')

    #2) plot dataset and save figures

    # plot configuration
    figwidth = 5; figheight = 5
    subplot_left=0.08; subplot_right=0.95; subplot_top=0.9;subplot_bottom=0.1; hspace=0.12; wspace=0.12

    #i) plot estimation results
    g=sns.FacetGrid(data=pd_labels_predictions,col='Variables',hue='Legends',sharey=False)
    g.map_dataframe(sns.lineplot,'Time','Values')
    g.add_legend()
    if(g.ax!=None):
        g.ax.grid(axis='both',which='major')
        if('figtitle' in kwargs.keys()):
            g.ax.set_title(kwargs['figtitle'])
        if('metrics' in kwargs.keys()):
            g.ax.text(0.6, 2,kwargs['metrics'], fontsize=12) #add text
    else:
        [ax.yaxis.grid(axis='both',which='major') for ax in g.axes]
        if('figtitle' in kwargs.keys()):
            [ax.set_title(kwargs['figtitle']) for ax in g.axes]
        if('metrics' in kwargs.keys()):
            [ax.text(0.45, 2, kwargs['metrics'], fontsize=12) for ax in g.axes]#add text

    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom,hspace=hspace,wspace=wspace)

    #ii) save figure
    # whether define the figsave_file
    if('figtitle' in kwargs.keys()):
        figPath = os.path.join(testing_folder, kwargs['figtitle']+".svg")
    else:
        figPath = os.path.join(testing_folder, str(localtimepkg.strftime("%H_%M_%S", localtimepkg.localtime())) + '.svg')

    plt.savefig(figPath)
    
    #iii) to show plot or not
    if('verbose' in kwargs.keys() and kwargs['verbose']==1):
        plt.show()

    plt.close()




'''
Plot statistic atucal and estimation values

'''

def plot_statistic_actual_estimation_curves(training_folder, testing_folder, **kwargs):
    
    # testing model in training folder 
    testing_results, testing_ingredients = test_model_on_unseen_subject(training_folder)

    #i) load actual values
    pd_actual_values = pd.concat(testing_results['labels'], axis=0)
    old_columns = pd_actual_values.columns
    new_columns = ['Actual ' + x for x in old_columns]
    pd_actual_values.rename(columns=dict(zip(old_columns,new_columns)), inplace=True)

    #ii) load prediction (estimation) values
    pd_prediction_values = pd.concat(testing_results['predictions'], axis=0)
    old_columns = pd_prediction_values.columns
    new_columns = ['Estimated ' + x for x in old_columns]
    pd_prediction_values.rename(columns=dict(zip(old_columns,new_columns)), inplace=True)

    #iii) combine actual and estimation values
    pd_actual_prediction_values = pd.concat([pd_actual_values,pd_prediction_values],axis=1)
    pd_actual_prediction_values.index = pd_actual_prediction_values.index/SAMPLE_FREQUENCY


    #iv) plot

    # plot configuration
    figwidth = 8; figheight = 8
    subplot_left=0.08; subplot_right=0.95; subplot_top=0.9;subplot_bottom=0.1; hspace=0.12; wspace=0.12

    #i) plot estimation results
    g=sns.FacetGrid(data=pd_actual_prediction_values)
    g.map_dataframe(sns.lineplot)
    g.add_legend()
    if(g.ax!=None):
        g.ax.grid(axis='both',which='major')
        g.ax.set_xlabel('Time [s]')
        g.ax.set_ylabel('GRF [BW]')
        if('figtitle' in kwargs.keys()):
            g.ax.set_title(kwargs['figtitle'])
        if('metrics' in kwargs.keys()):
            g.ax.text(0.6, 2,kwargs['metrics'], fontsize=12) #add text
    else:
        [ax.yaxis.grid(axis='both',which='major') for ax in g.axes]
        if('figtitle' in kwargs.keys()):
            [ax.set_title(kwargs['figtitle']) for ax in g.axes]
        if('metrics' in kwargs.keys()):
            [ax.text(0.45, 2, kwargs['metrics'], fontsize=12) for ax in g.axes]#add text

    # adjust figure size
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom,hspace=hspace,wspace=wspace)
    
    #ii) save figure
    # whether define the figsave_file
    if('figtitle' in kwargs.keys()):
        figPath = os.path.join(testing_folder, kwargs['figtitle']+".svg")
    else:
        figPath = os.path.join(testing_folder, str(localtimepkg.strftime("%H_%M_%S", localtimepkg.localtime())) + '.svg')

    plt.savefig(figPath)
    
    #iii) to show plot or not
    if('verbose' in kwargs.keys() and kwargs['verbose']==1):
        plt.show()

    plt.close()



def plot_estimation_error(labels,predictions,labels_names,fig_save_folder=None,**kwargs):
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
        folder_fig = fig_save_folder + "/"
    else:
        folder_fig = "./"

    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)

    # figure save file
    if('prediction_error_file' in kwargs.keys()):
        figPath = kwargs['prediction_error_file']
    else:
        figPath = folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + '_test_mes.svg'

    plt.savefig(figPath)



'''


Test each trained model on a trial of a testing subject

Plot the testing results of combination investigation


'''

def plot_combination_investigation_results(combination_testing_files, investigation_variable='Sensor configuration', displayed_variables=['r2','r_rmse']):

    pd_assessment = get_testing_metrics(combination_testing_files)

    #1) create folder
    combination_investigation_folder = re.search("[\s\S]+(\d)+", combination_testing_files).group()

    #2) plot statistical results
    figwidth = 13; figheight = 10
    subplot_left=0.06; subplot_right=0.97; subplot_top=0.95; subplot_bottom=0.06

    data = pd_assessment[pd_assessment['metrics'].isin(displayed_variables)]
    g = sns.catplot(data=data, x=investigation_variable, y='scores', col='metrics', col_wrap=2, kind='bar', hue='fields', height=3, aspect=0.8, sharey=False)
    g.fig.subplots_adjust(left = subplot_left, right=subplot_right, top=subplot_top, bottom=subplot_bottom, hspace=0.1, wspace=0.1)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    [ax.yaxis.grid(True) for ax in g.axes]
    g.savefig(os.path.join(combination_investigation_folder, "metrics.svg"))





'''
Test each trained model on the testing subjects' all trials and plot them:
    combination_investigation_files can be a list contains training folders or pd dataframe conatins metrics

'''

def plot_model_evaluation_on_unseen_subject(combination_investigation_files, investigation_variable='Sensor configurations', displayed_metrics = ['r2','r_rmse']):

    # testing multi models listed in combination_investigation_files on unseen subject's trials
    if(isinstance(combination_investigation_files,list)):
        pd_assessment = evaluate_models_on_unseen_subject(combination_investigation_files)

    if(isinstance(combination_investigation_files,pd.DataFrame)):
        pd_assessment = combination_investigation_files
    
    # save r2 scores
    pd_assessment.groupby('metrics').get_group('r2').to_csv(os.path.join(overall_metrics_folder,"r2_metrics.csv"))
    pd_assessment.groupby('metrics').get_group('r_rmse').to_csv(os.path.join(overall_metrics_folder,"r_rmse_metrics.csv"))

    # plot statistical results
    # i) plot configuration
    figwidth=13;figheight=10
    subplot_left=0.06; subplot_right=0.97; subplot_top=0.95;subplot_bottom=0.06

    # ii) plot 
    displayed_pd_assessment = pd_assessment[pd_assessment['metrics'].isin(displayed_metrics)]
    g=sns.catplot(data=displayed_pd_assessment, x=investigation_variable, y='scores',col='metrics',col_wrap=2,kind='bar',hue='fields',height=3, aspect=0.8,sharey=False)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom,hspace=0.1, wspace=0.1)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    [ax.yaxis.grid(True) for ax in g.axes]

    # iii) save plot figure
    g.savefig(overall_metrics_folder+ "/metrics.svg")
    
    return pd_assessment



def setup_plot(g, **kwargs):
               
    '''
    set up plot configurations

    '''
    xlabel = 'LSTM units'
    ylabel = 'R2'
    
    if('xlabel' in kwargs.keys()):
        xlabel = kwargs['xlabel']
    if('figtitle' in kwargs.keys()):
        figtitle = kwargs['figtitle']
    if('metrics' in kwargs.keys()):
        text = kwargs['metrics']
        
    if(hasattr(g, 'ax')): # only a subplot
        g.ax.grid(axis='both',which='major')
        g.ax.set_xlabel(xlabel)
        g.ax.set_ylabel('R2')
        if('figtitle' in kwargs.keys()):
            g.ax.set_title(figtitle)
        if('metrics' in kwargs.keys()):
            g.ax.text(0.6, 2,text, fontsize=12) #add text
    elif(hasattr(g, 'axes') and isinstance): # multi subplots
        try:
            iter(g.axes)
            pdb.set_trace()
            [ax.grid(axis='both',which='major') for ax in g.axes]
            [ax.set_xlabel(xlabel) for ax in g.axes]
            [ax.set_ylabel('R2') for ax in g.axes]
            if('figtitle' in kwargs.keys()):
                [ax.set_title(kwargs['figtitle']) for ax in g.axes]
            if('metrics' in kwargs.keys()):
                [ax.text(0.45, 2, kwargs['metrics'], fontsize=12) for ax in g.axes]#add text
        except TypeError: # only an axes
            g.axes.grid(axis='both',which='major')
            g.axes.set_xlabel(xlabel)
            g.axes.set_ylabel('R2')
            g.axes.legend(ncol=3,title='Sensor configurations',loc='lower right')
            
    if(isinstance(g,plt.Axes)):
        g.set_xlabel(xlabel)
        g.set_ylabel('R2')
        g.grid(visible=True, axis='both',which='major')
        g.set_ylim(0.7,1.0)
        g.legend(ncol=3,title='Sensor configurations',loc='lower right')
        g.get_legend().remove()
        
