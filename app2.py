#!flask/bin/python
from fastai.vision import *
import torch
import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import pickle
from PIL import Image
from io import BytesIO

app = Flask(__name__)
model = load_learner('.', 'base_model.pkl')


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit a empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/')
def index():
    return "Prediction of the Waste Class"

@app.route('/predict',methods=['POST'])
def predict():
    # get image from json
    image = request.get_data()
    image = open_image(io.BytesIO(image))
    image = image.resize((3, 384, 512))
    # predict the image
    pred_class,pred_idx,outputs = model.predict(image)

    # print the prediction
    return jsonify({"prediction":str(pred_class)})
    
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 80)