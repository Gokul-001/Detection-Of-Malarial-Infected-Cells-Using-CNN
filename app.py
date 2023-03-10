

from __future__ import division, print_function


import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ="malariagit97_final.h5"


# Load your trained model
model = load_model(MODEL_PATH)



def model_predict(img_path, model):
    test_image = image.load_img(img_path, target_size=(64, 64)) #image.load

    # Preprocessing the image # img_to_arr
    test_image= np.array(test_image)
    
    ## Scaling
    
    test_image = np.expand_dims(test_image, axis=0)
   
    result = model.predict(test_image)
    #training_set.class_indices
    if result[0][0] == 0:
        preds='The Person is Infected With Malaria'
    else:
        preds= 'The Person is not Infected With Malaria'
    return preds
"""

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    img_data= preprocess_input(x)

    preds = model.predict(img_data)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Person is Infected With Malaria"
    else:
        preds="The Person is not Infected With Malaria"
    
    """
    #return preds


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
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
