'''
UCM - Máster en Big Data & Business Analytics
Trabajo Final de Máster: Drowsiness Detection
Authors: Beatriz Cuesta, Antonio Fernández, Federico Golffeld, Alejandro Lema, Álvaro López and Ginés Meca
Sept 2023
'''

import pandas as pd
import numpy as np
from src.utils import *

def rf_predictor(clf_model,
                 data,
                 static_pred,
                 config):
    '''
    Function that checks if a new Random Forest prediction needs to be generated and computes it.
    Inputs:
    - clf_model (sklearn model): Random Forest model trained and loaded.
    - data (dict): Dictionary that includes all the frames captured since the last prediction and their features and info.
    - static_pred (str): Last prediction generated.
    - config (dict): Run parameters.

    Outputs:
    - pred (str): New prediction of the model or last prediction computed.
    - data (dict): Input data dict in case of not generating new prediction and empty otherwise (to capture new data for the next pred)
    '''
    # Predicting in case of having enough frames:
    if len(data) == config['clf_model']['n_preds']:
        data_df = pd.DataFrame.from_dict(data,
                                         orient = 'index')
        data_df = data_df[list(clf_model.feature_names_in_)]
        mean_pred = np.mean(clf_model.predict_proba(data_df)[:, 1])

        if mean_pred > config['clf_model']['threshold']:
            print('')
            print(bcolors.FAIL + 'ALERT: DROWSINESS DETECTED' + bcolors.ENDC)
            print('')
            pred = 'DROWSY SUBJECT'
        else:
            print('')
            print(bcolors.OKGREEN + 'ACTIVE SUBJECT.' + bcolors.ENDC)
            print('')
            pred = 'ACTIVE SUBJECT'

        return pred, {}

    # Keeping the last prediction otherwise:
    else:
        pred = static_pred
        return pred, data