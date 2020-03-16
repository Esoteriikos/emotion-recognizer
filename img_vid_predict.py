import cv2
import time
import secrets
import numpy as np
import face_recognition
from models import cnn_model_2

img_file_ext = '.jpg'
vid_file_ext = '.mp4'

class_labels = {0: 'Anger', 1: 'disgust', 2: 'fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


# load model and model.predict_class
def load_model():
    model = cnn_model_2()
    model.load_weights('trained_model/v2/model/model_195.h5')
    return model


def prediction(image, model):
    # pre-process
    img = cv2.resize(image, (48, 48))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.reshape(gray, (1, 48, 48, 1))
    predictions = model.predict_classes(img)
    print('Predictions:', class_labels[predictions[0]])
    return class_labels[predictions[0]]


def annotate_face_predict(frame, face_locations, model, scaled_factor=1):
    for (top, right, bottom, left) in face_locations:
        # left, top, bottom, right = [x*(1/scaled_factor) for x in (top, right, bottom, left)]
        left = int(left/scaled_factor)
        top = int(top/scaled_factor)
        bottom = int(bottom/scaled_factor)
        right = int(right/scaled_factor)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        pred = prediction(frame[top:bottom, left:right], model)
        cv2.putText(frame, pred, (left, bottom), font, 1.5, (0, 255, 0), 1)


def scaled_image(width, height):
    """face_recognition face detection slow AF compare to MTCNN
    scaling the image as per resolution to process fast"""
    total = width*height
    scale_dict = {0.2: 6500000, 0.3: 4000000, 0.4: 2250000, 0.5: 1000000, 0.8: 500000, 1: 0}
    for k, v in scale_dict.items():
        if total > v:
            return k


# Helper functions
# frame resize pad is OPTIONAL. just helps website/app looks good
def frame_resize_pad(frame, desired_size=720):
    """https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/"""
    old_size = frame.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(frame, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im


def save_video(vid_output_path, fps, size, img_arr):
    random_hex = secrets.token_hex(8)
    vid_path = vid_output_path + random_hex + vid_file_ext
    out = cv2.VideoWriter(vid_path, 0x00000021, fps, size)

    # vid_path_2 = 'static/sentiment_gallery/videos/' + random_hex + '.mp4'
    # out_2 = cv2.VideoWriter(vid_path_2, cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    for i in range(len(img_arr)):
        # writing to a image array
        out.write(img_arr[i])
        # out_2.write(img_arr[i])
    out.release()


def save_image(frame, img_output_path):
    random_hex = secrets.token_hex(8)
    cv2.imwrite(img_output_path + random_hex + img_file_ext, frame)


# Process image
def image_predict(image_list, img_input_path, img_output_path):
    if len(image_list) == 0:
        return
    m1 = load_model()
    file_paths = [img_input_path + image for image in image_list]

    for file_path in file_paths:  # iterate through each file
        image = cv2.imread(file_path)
        image = frame_resize_pad(image, 720)
        face_locations = face_recognition.face_locations(image)
        annotate_face_predict(image, face_locations, m1)

        save_image(image, img_output_path)
        # img_path = r'static/sentiment_gallery/images/' + random_hex + '.jpg'
        # cv2.imwrite(img_path, image


# Process video
def video_predict(video_list, video_input_path, video_output_path):
    global face_locations, video
    face_locations = []
    if len(video_list) == 0:
        return
    model = load_model()

    file_paths = [video_input_path + image for image in video_list]

    # iterate through each file
    for file_path in file_paths:
        img_arr = []
        start_time = time.time()
        video = cv2.VideoCapture(file_path)

        if not video.isOpened():
            print("Can not open video : ", file_path)
            continue
        # get fps, height, width of the video
        fps = int(video.get(cv2.CAP_PROP_FPS))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        scale_factor = scaled_image(width, height)

        while True:
            ret, frame = video.read()
            if ret:
                small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                if time.time() - start_time > 0.1:  # Check if 1 sec has passed
                    face_locations = face_recognition.face_locations(small_frame)
                    start_time = time.time()

                annotate_face_predict(frame, face_locations, model, scale_factor)
                img_arr.append(frame)
            else:
                break
        video.release()
        save_video(video_output_path, fps, (width, height), img_arr)
