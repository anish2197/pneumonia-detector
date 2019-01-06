from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model, Sequential
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.layers import *

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/vgg_tuned.h5'

vgg16 = keras.applications.vgg16.VGG16()
vgg16.layers.pop()
model = Sequential()

for layer in vgg16.layers:
    layer.trainable = False

for layers in vgg16.layers:
        model.add(layers)

model.add(Dense(2, activation="softmax"))
# Load your trained model
model.load_weights(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    return image

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    if img.mode != "RGB":
        img = image.convert("RGB")

    #img = image.resize((224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    # Preprocessing the image
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        result = "Probability of Pneumonia : " + str(round(preds[0][1],2))


        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class)               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(port=5002, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
