# forked/modified camera-opencv (github) miguelgrinberg code
# This code works for Streaming + predicting using high level tensorflow language - keras model .h5


import os
import cv2
import datetime
import numpy as np
import face_recognition
from models import cnn_model_2
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        class_labels = {0: 'Anger', 1: 'disgust', 2: 'fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        model = cnn_model_2()
        model.load_weights('trained_model/v2/model/model_195.h5')
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()
            frame = cv2.flip(img, 1)
            face_locations = face_recognition.face_locations(frame)
            for (top, right, bottom, left) in face_locations:
                frame = cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                face_frame = cv2.resize(frame[top:bottom, left:right], (48, 48))

                gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
                face = np.reshape(gray, (1, 48, 48, 1))

                prediction = class_labels[model.predict_classes(face)[0]]
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, prediction, (left, bottom), font, 1.5, (0, 255, 0), 1)

                timestamp = datetime.datetime.now()
                frame = cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),
                                    (10, 15), font, 0.6, (255, 0, 0), 1)
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()
