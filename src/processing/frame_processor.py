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
import PIL
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from src.processing.utils import *


def frame_processor(frame,
                    ear_detector,
                    blendshapes_detector,
                    face_detector,
                    config):
    '''
    Function that, given a frame, creates a dict that contains all the information related to it.
    Input:
    - frame (cv2 image): Image captured through the computer camera.
    - ear_detector (mediapipe object): Face mesh model initialized.
    - blendshapes_detector (mediapipe object): Blendshapes model initialized.
    - face_detector (mediapipe object): Face detection model initialized.
    - config (dict): Run parameters.

    Output:
    - frame_info: Dict containing the image array (fitted to the face), the blendshapes and the EAR.
    '''

    # RGB transformation
    print('Transforming into RGB...')
    frame_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initializing the output variable:
    frame_info = {}

    # Landmarks detection
    print('Starting EAR&MAR detection...')
    print('Detecting face mesh...')
    results = ear_detector.process(frame_mp)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        left_eye_landmarks = np.array(
            [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in config['processor_idxs']['all_left_eye_idxs']])
        right_eye_landmarks = np.array(
            [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in config['processor_idxs']['all_right_eye_idxs']])

        # EAR
        print('Computing EAR...')
        frame_info['ear_left'] = eye_aspect_ratio(left_eye_landmarks)
        frame_info['ear_right'] = eye_aspect_ratio(right_eye_landmarks)

        # MAR
        print('Computing MAR...')
        frame_info['mar'] = mouth_aspect_ratio(
                            np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in config['processor_idxs']['mouth_idxs']]))

    else:
        print('No face detected.')
        frame_info['ear_left'] = -1.0
        frame_info['ear_right'] = -1.0
        frame_info['mar'] = -1.0


    # Blendshapes detection:
    print('Starting blendshapes detection...')
    frame_image = mp.Image(image_format = mp.ImageFormat.SRGB,
                           data = frame_mp)
    blend_detection_result = blendshapes_detector.detect(frame_image)
    if blend_detection_result.face_blendshapes:
        print('Computing blendshapes...')
        for category in blend_detection_result.face_blendshapes[0]:
            frame_info['index_' + str(category.index)] = category.score

    else:
        print('No face detected.')
        with open(config['detection_error_path'], 'r') as file:
            error_list = json.load(file)
        for feature in error_list:
            frame_info['index_' + str(feature['index'])] = feature['score']

    # Face zoom:
    if config['clf_model']['name'] == 'CNN':
        print('Detecting face to normalize...')
        face_results = face_detector.process(frame_mp).detections
        if face_results is not None:
            print('Face detected.')
            relative_bounding_box = face_results[0].location_data.relative_bounding_box

            # Fitting the minimums inside the frame box:
            if relative_bounding_box.xmin < 0:
                relative_bounding_box.xmin = 0
            if relative_bounding_box.ymin < 0:
                relative_bounding_box.ymin = 0

            # Normalizing min coords to pixel data:
            rect_start_point = _normalized_to_pixel_coordinates(
                relative_bounding_box.xmin,
                relative_bounding_box.ymin, frame.shape[1], frame.shape[0])

            # Fitting the maximums inside the frame box:
            x_max = min((relative_bounding_box.xmin + relative_bounding_box.width), 1)
            y_max = min((relative_bounding_box.ymin + relative_bounding_box.height), 1)

            # Normalizing max coords to pixel data:
            rect_end_point = _normalized_to_pixel_coordinates(
                x_max,
                y_max, frame.shape[1], frame.shape[0])

            # Bounding box coords:
            xleft, ytop = rect_start_point
            xright, ybot = rect_end_point

            # Face selection:
            frame = frame[ytop: ybot, xleft: xright]

    # Saving image array
    if config['clf_model']['name'] == 'CNN':
        # Frame resize
        print('Resizing frame...')
        frame = cv2.resize(frame, (config['frame_size'], config['frame_size']))
        if config['frame_color'] == 'rgb':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_info['frame_array'] = frame
        elif config['frame_color'] == 'gray':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_info['frame_array'] = frame

    return frame_info