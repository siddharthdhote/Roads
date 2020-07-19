#!/usr/bin/env python
# coding: utf-8

# In[9]:

import tensorflow
import base64
import numpy as np
import io
import jinja2
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask import render_template

app= Flask(__name__)

def get_model():
    global model
    model=load_model('RoadCrack.h5')
    print("* Model loaded!")
    
def preprocess_image(image,target_size):
    if image.mode != "RGB":
        image=image.convert("RGB")
    image=image.resize(target_size)
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    
    return image

print("* Loading Keras model...")
get_model()

@app.route("/", methods=["GET","POST"])
def predict():
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded))
    processed_image=preprocess_image(image,target_size=(224,224))
    
    prediction=model.predict(processed_image).tolist()
    
    response={
        'prediction':{
            'Negative': prediction[0][0],
            'Positive': prediction[0][1]
        }
    }
    
    return jsonify(response)

if __name__=="__main__":
    app.debug=True
    app.run()
    

