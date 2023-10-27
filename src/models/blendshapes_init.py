'''
UCM - Máster en Big Data & Business Analytics
Trabajo Final de Máster: Drowsiness Detection
Authors: Beatriz Cuesta, Antonio Fernández, Federico Golffeld, Alejandro Lema, Álvaro López and Ginés Meca
Sept 2023
'''

from mediapipe.tasks.python import vision
from mediapipe.tasks import python


def create_blendshapes_model(config):
    '''
    Function that initializes the model used to detect the blendshapes during the processing.
    Outputs:
    - blend_detector (mediapipe object): Detector initialized and ready to detect.
    '''
    with open(config['blendshapes_task_path'], 'rb') as f:
        model_task = f.read()
    base_options = python.BaseOptions(model_asset_buffer = model_task)
    options = vision.FaceLandmarkerOptions(base_options = base_options,
                                           output_face_blendshapes = True,
                                           output_facial_transformation_matrixes = True,
                                           num_faces = 1)
    blend_detector = vision.FaceLandmarker.create_from_options(options)
    return blend_detector