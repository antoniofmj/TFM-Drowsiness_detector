'''
UCM - Máster en Big Data & Business Analytics
Trabajo Final de Máster: Drowsiness Detection
Authors: Beatriz Cuesta, Antonio Fernández, Federico Golffeld, Alejandro Lema, Álvaro López and Ginés Meca
Sept 2023
'''

import pickle
import keras


def clf_model_init(config):
    '''
    Function that loads the classification model to obtain the predictions.
    Inputs:
    - config (dict): Run parameters.
    '''
    if config['clf_model']['name'] == 'RandomForestClassifier':
        clf_model = pickle.load(open(config['clf_model']['path'], 'rb'))

    elif config['clf_model']['name'] in ['CNN', 'LSTM']:
        clf_model = keras.models.load_model(config['clf_model']['path'])

    else:
        print('Model no registered.')
        clf_model = None

    return clf_model