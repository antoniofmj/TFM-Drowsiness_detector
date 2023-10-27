'''
UCM - Máster en Big Data & Business Analytics
Trabajo Final de Máster: Drowsiness Detection
Authors: Beatriz Cuesta, Antonio Fernández, Federico Golffeld, Alejandro Lema, Álvaro López and Ginés Meca
Sept 2023
'''

import sys
import time
from src.processing.frame_processor import *
from src.models.face_mesh_init import *
from src.models.blendshapes_init import *
from src.models.face_detector_init import *
from src.predict.make_predictions import *
from src.models.clf_model_init import *
import pygame


def drowsiness_detector():
    '''
    Main pipeline of the project. The pipeline opens the computer camera and displays the predictions on the same screen
    in a regular basis in order to track the drowsiness of the subject in real-time.
    '''

    print(bcolors.HEADER + 'Drowsiness detector execution started.' + bcolors.ENDC)

    #Reading config file:
    print('Reading config...')
    config_path = sys.argv[1]
    config = load_config(config_path)

    # Defining a video capture object:
    vid = cv2.VideoCapture(0)

    # Initializing necessary variables:
    print('Initializing variables...')
    images = {}
    frame_counter = 0
    pred = False
    start_time = time.time()

    # Initializing models that will be used in the processing
    # Mediapipe face detector
    print('Initializing face detector model...')
    face_mesh = create_face_mesh_model()

    # Blendshapes detector
    print('Initializing blendshapes detector model...')
    blend_detector = create_blendshapes_model(config)

    # Face detector:
    print('Initializing face detector model...')
    face_detector = create_face_detector_model()

    # Classification model:
    print('Initializing classification model...')
    clf_model = clf_model_init(config)

    # Initializing pygame visualization:
    pygame.init()
    pygame.font.init()
    surface = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Drowsiness detector")

    # Opening the frames loop:
    while (True):

        # Capturing the video frame:
        ret, frame = vid.read()

        # Displaying the resulting frame
        display_image = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "BGR")
        surface.blit(display_image, (0, 0))

        # Selecting only n frames per second:
        if time.time() - start_time >= 1 / config['frames_per_sec']:
            print('Frame no.', frame_counter, 'captured.')
            print('Starting frame processor...')
            frame_info = frame_processor(frame = frame,
                                         ear_detector = face_mesh,
                                         blendshapes_detector = blend_detector,
                                         face_detector = face_detector,
                                         config = config)

            # Saving the frame:
            print('Saving frame info...')
            images['frame_' + str(frame_counter)] = frame_info
            print('Frame', frame_counter, 'saved.')

            # Model execution:
            pred, images= make_predictions(clf_model = clf_model,
                                           data = images,
                                           static_pred = pred,
                                           counter = frame_counter,
                                           config = config)

            # Incresing the counter and restarting the start_time:
            frame_counter += 1
            start_time = time.time()

        surface = display_preds(data = images,
                                static_pred = pred,
                                counter = frame_counter,
                                display_surface = surface,
                                config = config)
        pygame.display.flip()

        # Press the X button to finish the process.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
                break
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    drowsiness_detector()