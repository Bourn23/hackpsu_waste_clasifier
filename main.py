import torch
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # Get the image from post request
    img = request.files['image'].read()
    img = np.fromstring(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (512, 512))
    img = img.reshape(1, 512, 512, 3)
    img = torch.from_numpy(img).double()
    # Make prediction
    pred = model.predict(img)
    pred = np.argmax(pred, axis=1)
    pred = pred[0]

if __name__ == "__main__":
    app.run(debug=True)