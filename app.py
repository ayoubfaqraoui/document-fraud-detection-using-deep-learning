from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '../uploads'
app.config['MODEL_PATH'] = '../model/document fraud detection using deep learning/model/my_model.h5'


img_height = 256
img_width = 256
class_names = ['fake', 'real']

# Load the model
MyModel = tf.keras.models.load_model(app.config['MODEL_PATH'])

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_array):
    predictions = MyModel.predict(img_array)
    pred_label = class_names[np.argmax(predictions)]
    return pred_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        img_array = preprocess_image(img_path)
        pred_label = predict_image(img_array)

        return render_template('result.html', img_path=img_path, pred_label=pred_label)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
