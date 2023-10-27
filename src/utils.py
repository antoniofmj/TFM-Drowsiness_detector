'''
UCM - Máster en Big Data & Business Analytics
Trabajo Final de Máster: Drowsiness Detection
Authors: Beatriz Cuesta, Antonio Fernández, Federico Golffeld, Alejandro Lema, Álvaro López and Ginés Meca
Sept 2023
'''

import json
import pygame

class bcolors:
    '''
    class defined to print colors in console.
    '''
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_config(path):
    '''
    Function that reads the config with the params needed to run the model.
    Inputs:
    - path (str): config path.

    Outputs:
    - config (dict): Dictionary with all the params by keys.
    '''
    with open(path, 'r') as json_file:
        config = json.load(json_file)
    return config


def display_preds(data,
                  static_pred,
                  counter,
                  display_surface,
                  config):
    '''
    Function that displays the current prediction along with the prediction status (tracking the time to the next
    prediction) in the same screen where the face is shown.
    Inputs:
    - data (dict): Frames arrays and metadata extracted in the processing.
    - static_pred (str): Last prediction given.
    - counter (int): Number of frames captured.
    - display_surface (pygame object): Screen where the info is shown.
    - config (dict): Run parameters.
    Outputs:
    - display_surface (pygame object): Screen with all the info updated.
    '''
    if static_pred != False:
    # Show current prediction:
        my_font = pygame.font.SysFont('Oswald Bold', 40)
        if static_pred == 'ACTIVE SUBJECT':
            color_text = (0, 255, 0)
        else:
            color_text = (255, 0, 0)
        text_surface = my_font.render(static_pred, False, color_text)
        display_surface.blit(text_surface, (640 / 2 - 120, 400))

    # Update status:
    my_font = pygame.font.SysFont('Arial', 20)
    if (config['clf_model']['name'] in ['CNN', 'LSTM']) and (counter < config['clf_model']['init_n']):
        next_update_perc = len(data) / config['clf_model']['init_n'] * 100
    else:
        next_update_perc = len(data) / config['clf_model']['n_preds'] * 100
    text_update = my_font.render('Prediction update: ' + str(int(next_update_perc)) + '%', False, (0, 0, 0))
    display_surface.blit(text_update, (0, 0))

    return display_surface