import os
import secrets
from flask_dropzone import Dropzone
from camera_stream_keras import Camera
from img_vid_predict import image_predict, video_predict
from flask import Flask, render_template, request, send_from_directory, Response


img_ext = ('.jpeg', '.jpg', '.png')
vid_ext = ('.mp4')

images_input_folder_path = 'static/input_files/images/'
images_output_folder_path = 'static/output_files/images/'
videos_input_folder_path = 'static/input_files/videos/'
videos_output_folder_path = 'static/output_files/videos/'

app = Flask(__name__)

# dropzone settings,  there are many more
# https://flask-dropzone.readthedocs.io/en/latest/configuration.html
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = '.jpg, .jpeg, .png, .mp4'
app.config['DROPZONE_MAX_FILE_SIZE'] = 100
app.config['DROPZONE_MAX_FILE'] = 50

# defaults
# app.config['DROPZONE_INVALID_FILE_TYPE'] = "You can't upload files of this type."
# app.config['DROPZONE_FILE_TOO_BIG'] = "File is too big {{filesize}}. Max filesize: {{maxFilesize}}MiB."
# app.config['DROPZONE_SERVER_ERROR'] = "Server error: {{statusCode}}"
# app.config['DROPZONE_BROWSER_UNSUPPORTED'] = "Your browser does not support drag'n'drop file uploads."
# app.config['DROPZONE_MAX_FILE_EXCEED'] = 	"Your can't upload any more files."

dropzone = Dropzone(app)


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


def delete_content(folder):  # Every experience to be new
    """Since we dont want previous result to be shown again
    or previous inputs to be process again"""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            pass


# Upload images videos zone
@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        delete_content(images_input_folder_path)
        delete_content(videos_input_folder_path)
        # uploaded files object
        file_object = request.files
        # Multiple files. iterating through each
        for f in file_object:
            file = request.files.get(f)

            # create random token
            random_hex = secrets.token_hex(8)

            #  extract the file extension
            _, f_ext = os.path.splitext(file.filename)
            file_name = random_hex + f_ext

            # check if its video or image to save in resp. folder
            if f_ext in img_ext:
                file_path = os.path.join(app.root_path, images_input_folder_path, file_name)
            else:
                # get the file path to save file
                file_path = os.path.join(app.root_path, videos_input_folder_path, file_name)

            file.save(file_path)

        return 'UPLOADING'
    return render_template('upload.html')


# To display results
@app.route('/result/<filename>')
def send_image(filename):
    return send_from_directory(images_output_folder_path, filename)


@app.route('/result/<filename>')
def send_video(filename):
    return '../' + videos_output_folder_path + filename


# To calculate results
@app.route("/results")
def results():
    delete_content(images_output_folder_path)
    image_names = os.listdir(images_input_folder_path)
    image_predict(image_names, images_input_folder_path, images_output_folder_path)
    delete_content(images_input_folder_path)

    delete_content(videos_output_folder_path)
    video_names = os.listdir(videos_input_folder_path)
    video_predict(video_names, videos_input_folder_path, videos_output_folder_path)
    delete_content(videos_input_folder_path)

    out_image_names = os.listdir(images_output_folder_path)
    out_video_names = os.listdir(videos_output_folder_path)
    return render_template("results.html", image_names=out_image_names, video_names=out_video_names)


# Web-cam Stream + real time predictions
@app.route('/stream')
def stream():
    return render_template('stream.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True, threaded=True, use_reloader=False)
