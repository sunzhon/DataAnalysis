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

from vicon_imu_data_process.dataset import *
from vicon_imu_data_process import process_rawdata as pro_rd

from estimation_models.rnn_models import *

from sklearn.metrics import r2_score, mean_squared_error as mse

from vicon_imu_data_process.const import SAMPLE_FREQUENCY

def calculate_scores(y_true, y_pred):
    '''
    Calculate scores of the estimation value

    '''
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse(y_true, y_pred))
    mae = np.mean(abs((y_true - y_pred)))
    r_rmse = rmse / (y_true.max() - y_true.min())

    return round(r2*1000.0)/1000.0, round(rmse*1000.0)/1000.0, round(mae*1000.0)/1000.0, round(r_rmse*1000.0)/1000.0
 
def get_evaluation_metrics(pd_labels, pd_predictions,verbose=0):
    '''
    labels and predictions are pandas dataframe
    return metrics, it is pandas dataframe

    '''

    #i) calculate metrics
    scores={}
    for label in pd_labels.columns:
        scores[label] = list(calculate_scores(pd_labels[label].values, pd_predictions[label].values))
        if(verbose==1):
            print("{}: scores (r2, rmse, mae, r_rmse):".format(label), scores[label][0], scores[label][1], scores[label][2], scores[label][3])

    #ii) shape metrics
    metrics = pd.DataFrame(data=scores, index=['r2','rmse','mae','r_rmse'])
    metrics = metrics.reset_index().rename(columns={'index':'metrics'})
    metrics = metrics.melt(id_vars='metrics',var_name='fields',value_name='scores') 

    return metrics


def calculate_model_time_complexity(model, series, hyperparams):

    window_size = int(hyperparams['window_size'])
    batch_size = int(hyperparams['batch_size'])
    shift_step = int(hyperparams['shift_step'])
    labels_num = int(hyperparams['labels_num'])

    [row_num, column_num] = series.shape

    ds = tf.data.Dataset.from_tensor_slices(series[:, labels_num])

    start = time.time()

    forecast = model.predict(ds)

    end = time.time()

    time_cost = (end - start)/float(row_num)

    return time_cost


'''

 load (best) trained model

'''
def load_trained_model(training_folder, best_model=True):
    trained_model_file = os.path.join(training_folder,'trained_model','my_model.h5')
    #print("Trained model file: ", trained_model_file)
    
    trained_model = tf.keras.models.load_model(trained_model_file)
    
    if(best_model): # load the best model parameter
        best_trained_model_weights = training_folder + "/online_checkpoint/cp.ckpt"
        trained_model.load_weights(best_trained_model_weights)
        
    return trained_model




'''
Testing model
'''
def test_model(training_folder, xy_test, scaler, **kwargs):
    
    #1) create test results folder
    testing_folder = pro_rd.create_testing_files(training_folder)
    
    #2) load hyperparameters, note that the values in hyperparams become string type
    hyperparams_file = os.path.join(training_folder,"hyperparams.yaml")
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr, Loader=yaml.BaseLoader)
        fr.close()
    else:
        print("Not Found hyper params file at {}".format(hyperparams_file))
        exit()

    #3) load trained model
    trained_model = load_trained_model(training_folder)
    
    #4) test data
    model_output = model_forecast(trained_model, xy_test, hyperparams)
    model_prediction = model_output.reshape(-1,int(hyperparams['labels_num']))
    
    #5) reshape and inverse normalization
    prediction_xy_test = copy.deepcopy(xy_test) # deep copy of test data
    prediction_xy_test[:,-int(hyperparams['labels_num']):] = model_prediction # using same shape with all datasets
    predictions = scaler.inverse_transform(prediction_xy_test)[:,-int(hyperparams['labels_num']):] # inversed norm predition
    labels  = scaler.inverse_transform(xy_test)[:,-int(hyperparams['labels_num']):]
    features = scaler.inverse_transform(xy_test)[:,:-int(hyperparams['labels_num'])]
    
    #6) save params in testing
    hyperparams_file = os.path.join(testing_folder,"hyperparams.yaml")
    with open(hyperparams_file,'w') as fd:
        yaml.dump(hyperparams, fd)


    
    #7) transfer testing results' form into pandas Dataframe 
    pd_features = pd.DataFrame(data=features, columns=hyperparams['features_names'])
    pd_labels = pd.DataFrame(data = labels, columns=hyperparams['labels_names'])
    pd_predictions = pd.DataFrame(data = predictions, columns=hyperparams['labels_names'])

    #8) save testing results
    save_test_result(pd_features, pd_labels, pd_predictions, testing_folder)

    return features, labels, predictions, testing_folder




'''
save testing results: estimation (predictions) and estiamtion metrics
'''
def save_test_result(pd_features, pd_labels, pd_predictions, testing_folder):
    
    #1) save estiamtion of the testing
    # create testing result (estimation value) file
    saved_test_results_file = os.path.join(testing_folder, "test_results.h5")
    # save tesing results
    with h5py.File(saved_test_results_file,'w') as fd:
        fd.create_dataset('features',data=pd_features.values)
        fd.create_dataset('labels',data=pd_labels.values)
        fd.create_dataset('predictions',data=pd_predictions.values)
        fd['features'].attrs['features_names'] = list(pd_features.columns)
        fd['labels'].attrs['labels_names'] = list(pd_labels.columns)

    #2) save metrics of the estimation results

    # create testing metrics file   
    metrics_file = os.path.join(testing_folder, "test_metrics.csv")

    # calculate metrics
    metrics = get_evaluation_metrics(pd_labels, pd_predictions)

    # save metrics
    metrics.to_csv(metrics_file)



'''
Model evaluation:

Use a trained model to estimate labels using an unseen subject' all trials

subject_id_name = None, this means use the test subject specified in hyperprams
trials = None, means use all useful trials in the subject

'''
def test_model_on_unseen_subject(training_folder, subject_id_name=None, trials=None):

    #1) load hyperparameters
    hyperparams_file = os.path.join(training_folder, "hyperparams.yaml")
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    else:
        print("Not Found hyper params file at {}".format(hyperparams_file))
        exit()

    #2) load and norm dataset
    subjects_trials_data, norm_subjects_trials_data, scaler = load_normalize_data(hyperparams,syn_features_labels=True)

    #3) subject and trials for testing
    #i) the subject for testing
    if(subject_id_name==None):
        subject_id_name = hyperparams['test_subject_ids_names'][0]

    #ii) trials of the subject for testing
    if(trials==None): # use all trials of the subject, the return is lists
        trials = hyperparams['subjects_trials'][subject_id_name]

    if(not isinstance(trials,list)):
        trials = [trials]
    
    testing_results={'labels':[],'predictions': []}
    testing_ingredients = {'subjects': [], 'trials': [],'testing_folder':[]}
    for trial in trials:
        # load data for testing
        xy_test = norm_subjects_trials_data[subject_id_name][trial]
        # testing model
        features, labels, predictions, testing_folder = test_model(training_folder,xy_test,scaler)
        # reshape data
        pd_labels = pd.DataFrame(data = labels, columns=hyperparams['labels_names'])
        pd_predictions = pd.DataFrame(data = predictions, columns=hyperparams['labels_names'])
        # add time in dataframe
        #row_num = pd_labels.shape[0]
        #pd_labels['time'] = np.linspace(0,row_num/SAMPLE_FREQUENCY, row_num)
        #row_num = pd_predictions.shape[0]
        #pd_predictions['time'] = np.linspace(0,row_num/SAMPLE_FREQUENCY, row_num)

        # add data to list
        testing_ingredients['subjects'].append(subject_id_name)
        testing_ingredients['trials'].append(trial)
        testing_ingredients['testing_folder'].append(testing_folder)

        testing_results['labels'].append(pd_labels)
        testing_results['predictions'].append(pd_predictions)

    return testing_results, testing_ingredients

'''
Model evaluation:

    test multiple models listed in combination_investigation_files on a unseen subject's trials

'''

def evaluate_models_on_unseen_subject(combination_investigation_files):

    # open testing folder
    assessment = []
    for line in open(combination_investigation_files,'r'):
        #0) tesing results folder
        line=line.strip('\n')
        [sensor_config, model_size, testing_folder] = line.split('\t')

        #1) get folder of the trained model
        training_folder  = re.search(r".+(\d){2}-(\d){2}",testing_folder).group(0) + "/training" + re.search("_(\d)+", testing_folder).group(0)

        #2) testing the model by using all trials of the testing subject (specidied in hyperparams)
        testing_results, testing_ingredients = test_model_on_unseen_subject(training_folder)
        
        #3) estimation results
        for trial_id, testing_folder in enumerate(testing_ingredients['testing_folder']):
            #i) collect metric results
            metrics = pd.read_csv(os.path.join(testing_folder,"test_metrics.csv")) # get_evaluation_metrics(pd_labels, pd_predictions)
            metrics['Sensor configurations'] = sensor_config
            metrics['Model size'] = model_size
            metrics['Test ID'] = trial_id
            metrics['Subjects'] = testing_ingredients['subjects'][trial_id]
            metrics['Trials'] = testing_ingredients['trials'][trial_id]
            assessment.append(metrics)

    # concate to a pandas dataframe
    pd_assessment = pd.concat(assessment, axis=0)

    # save pandas DataFrame
    overall_metrics_folder = re.search("[\s\S]+(\d)+",combination_investigation_files).group()
    pd_assessment.to_csv(os.path.join(overall_metrics_folder,"metrics.csv"))
    print('Metrics file save at: {}'.format(overall_metrics_folder))
     
    return pd_assessment



'''
To get metrics of list of testing


'''

def get_testing_metrics(combination_testing_files):
    
    # open testing folder
    assessment=[]
    line_num=0
    for line in open(combination_testing_files,'r'):
        #0) tesing results folder
        if line =='\n':
            continue; # pass space line
        line = line.strip('\n')
        if(line_num==0):
            columns = line.split('\t')
            line_num = line_num + 1
            continue

        try:
            # get testing_folder and test id
            a_single_investigation_config_results = line.split('\t')
            testing_folder = a_single_investigation_config_results[-1]
            test_id = re.search("test_([0-9])+", testing_folder).group(0)
        except Exception as e:
            pdb.set_trace()

        #1) load testing results
        testing_results = os.path.join(testing_folder, 'test_results.h5')
        with h5py.File(testing_results,'r') as fd:
            features=fd['features'][:,:]
            predictions=fd['predictions'][:,:]
            labels=fd['labels'][:,:]
            labels_names=fd['labels'].attrs['labels_names']

        #2) estimation results
        #i) plot curves
        pd_labels = pd.DataFrame(data = labels, columns = labels_names)
        pd_predictions = pd.DataFrame(data = predictions, columns = labels_names)

        #ii) collect metric results
        try:
            metrics = get_evaluation_metrics(pd_labels, pd_predictions)
            for idx in range(len(a_single_investigation_config_results)-1):
                metrics[columns[idx]] = a_single_investigation_config_results[idx]
            metrics['Test ID'] = test_id
        except Exception as e:
            pdb.set_trace()

        assessment.append(metrics)

    #iii) concate to a pandas dataframe
    pd_assessment = pd.concat(assessment, axis=0)

    #3) save pandas DataFrame
    combination_investigation_folder = os.path.dirname(combination_testing_files)
    pd_assessment.to_csv(os.path.join(combination_investigation_folder, "metrics.csv"))

    return pd_assessment

