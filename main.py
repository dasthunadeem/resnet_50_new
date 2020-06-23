# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array

# Flask 
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model = 'model_resnet.pb'

# Load your trained model
model = load_model(model)

#Creating a Function to predict
def model_predict(image_path, model):
    image=load_img(image_path,target_size=(224,224))

    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    image=preprocess_input(image)
    
    predict=model.predict(image)
    return predict


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static', secure_filename(file.filename))
        file.save(file_path)

        # Make prediction
        predict = model_predict(file_path, model)
        predict = decode_predictions(predict, top=1)   
        result = str(predict[0][0][1]) 
        accuracy= str(predict[0][0][2]*100)             
        return result,accuracy
    return None 


if __name__ == '__main__':
    app.run(debug=True,threaded=False)

