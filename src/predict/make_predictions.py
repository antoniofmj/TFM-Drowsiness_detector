'''
UCM - Máster en Big Data & Business Analytics
Trabajo Final de Máster: Drowsiness Detection
Authors: Beatriz Cuesta, Antonio Fernández, Federico Golffeld, Alejandro Lema, Álvaro López and Ginés Meca
Sept 2023
'''

from src.predict.rf_predictor import *
from src.predict.cnn_predictor import *
from src.predict.lstm_predictor import *
import pygame

def make_predictions(clf_model,
                     data,
                     counter,
                     static_pred,
                     config):
    '''
    Function that checks if a new prediction is required and computes it.
    Inputs:
    - clf_model (model): Sklearn or Keras model already trained and loaded.
    - data (dict): Dictionary that includes all the frames captured since the last prediction and their features and info.
    - counter (int): Number of frames captured.
    - static_pred (str): Last prediction generated.
    - config (dict): Run parameters.

    Outputs:
    - pred (str): Model prediction (new or last one the number of frames captured)
    - images (dict): Input data dict in case of not generating new prediction and empty otherwise (to capture new data for the next pred)
    '''
    # Calculating predictions:
    if config['clf_model']['name'] == 'RandomForestClassifier':
        pred, images = rf_predictor(clf_model = clf_model,
                                    data = data,
                                    static_pred = static_pred,
                                    config = config)
    elif config['clf_model']['name'] == 'CNN':
        pred, images = cnn_predictor(clf_model = clf_model,
                                     data = data,
                                     static_pred = static_pred,
                                     counter = counter,
                                     config = config)
    elif config['clf_model']['name'] == 'LSTM':
        pred, images = lstm_predictor(clf_model = clf_model,
                                      data = data,
                                      static_pred = static_pred,
                                      counter = counter,
                                      config = config)
    else:
        print(bcolors.WARNING + 'No valid model specified.' + bcolors.ENDC)



    return pred, images