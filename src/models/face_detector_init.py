'''
UCM - Máster en Big Data & Business Analytics
Trabajo Final de Máster: Drowsiness Detection
Authors: Beatriz Cuesta, Antonio Fernández, Federico Golffeld, Alejandro Lema, Álvaro López and Ginés Meca
Sept 2023
'''

import mediapipe as mp


def create_face_detector_model():
    '''
    Function that initializes the model used to detect the faces during the processing to fit better each frame
    to the subject face.
    Outputs:
    - face_detector (mediapipe object): Detector initialized and ready to detect.
    '''
    face_detector = mp.solutions.face_detection.FaceDetection(
                                                              model_selection = 1,
                                                              min_detection_confidence = 0.5)
    return face_detector