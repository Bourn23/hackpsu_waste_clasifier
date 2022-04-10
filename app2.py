#!flask/bin/python
from fastai.vision import *
import torch
import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import pickle

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

@app.route('/predict',methods=['GET'])
def predict():

    img = open_image('./glass2.jpg')

    # resize image to 384*512
    img = img.resize(size=(3, 384, 512))

    # predict the image
    pred_class,pred_idx,outputs = model.predict(img)

    # print the prediction
    print(pred_class)
    return jsonify({"prediction":str(pred_class)})
    # img = np.fromfile('glass2.jpg', dtype = np.uint8)
    # # Get the image from post request
    # # img = request.files['image'].read()
    # # img = np.fromstring(img, dtype=np.uint8)
    # # img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    # # img = cv2.resize(img, (512, 512))
    # print('image loaded')
    # img = img.reshape(1, 512, 512, 3)
    # img = torch.from_numpy(img).double()
    # # Make prediction
    # pred = model.predict(img)
    # pred = np.argmax(pred, axis=1)
    # pred = pred[0]
    # pred = jsonify({'type_of_waste': pred})

    # return pred

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 80)