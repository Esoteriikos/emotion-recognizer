# esoteriikos/emotion-recognizer
**Input** - Image, video or webcam access.

**Intermediate** - Find faces, model.predict to predict the emotion, and lastly annotate the face and predicted emotion onto the image

**Output** - Images, Video, real time streaming (webcam) 

## Dataset : 
[FER](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

## Download the models from 
__[HERE](https://drive.google.com/open?id=1G05pHE3QN264BAmfKOf761MveEuQaxR8)__

paste them in respective folders
./trained_models/v2/model/<model_195/210.h5> and ./trained_models/v4/model/<model_60.h5>

In code we are using _model_195.h5_.


## Requirements : 

- face-recognition==1.3.0
- Flask==1.1.1
- Flask-Dropzone==1.5.4
- Keras==2.3.1
- numpy==1.18.1
- opencv-python==4.2.0.32
- tensorboard==2.1.1
- tensorflow==2.1.0

