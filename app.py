from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the ResNet model
model = load_model('./model/ResNet50_knee_model.h5')

# Define image dimensions and classes
IMG_SIZE = (256, 256)
CATEGORIES = ['Normal', 'Doubtful', 'Mild','Moderate','Severe']  # Replace with your actual class names

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save uploaded file
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Preprocess the image
            img = load_img(file_path, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = CATEGORIES[np.argmax(prediction)]

            return render_template(
                'index.html',
                uploaded_image=file.filename,
                prediction=predicted_class
            )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
