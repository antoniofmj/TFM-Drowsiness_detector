'''
UCM - Máster en Big Data & Business Analytics
Trabajo Final de Máster: Drowsiness Detection
Authors: Beatriz Cuesta, Antonio Fernández, Federico Golffeld, Alejandro Lema, Álvaro López and Ginés Meca
Sept 2023
'''

import mediapipe as mp


def create_face_mesh_model():
    '''
    Function that initializes the model used to detect the face mesh during the processing.
    Outputs:
    - face_mesh (mediapipe object): Detector initialized and ready to detect.
    '''
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1)
    return face_mesh