'''
UCM - Máster en Big Data & Business Analytics
Trabajo Final de Máster: Drowsiness Detection
Authors: Beatriz Cuesta, Antonio Fernández, Federico Golffeld, Alejandro Lema, Álvaro López and Ginés Meca
Sept 2023
'''

import numpy as np
import mediapipe as mp
import cv2
import json

def eye_aspect_ratio(eye_landmarks):
    '''
    Function that computes the distances between one eye landmarks.
    Inputs:
    - eye_landmarks (numpy.array): Selected points from the face mesh associated to the eye.

    Outputs:
    - ear (float): EAR metric of the eye.
    '''
    # Horizontal
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Vertical
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    # Computing EAR:
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_landmarks):
    '''
    Function that computes the distances between mouth landmarks.
    Inputs:
    - mouth_landmarks (numpy.array): Selected points from the face mesh associated to the mouth.

    Outputs:
    - mar (float): MAR metric of the eye.
    '''
    # Checking the number of landmarks captured:
    if len(mouth_landmarks) < 4:
        return -1  # We cannot compute the MAR if we don't have enough landmarks

    # Main landmarks:
    upper_mouth = np.array(mouth_landmarks[1])  # 39
    lower_mouth = np.array(mouth_landmarks[3])  # 269

    # Computing MAR:
    mar = np.linalg.norm(upper_mouth - lower_mouth)
    return mar




