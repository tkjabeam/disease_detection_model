from flask import Flask, request, jsonify, session, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
import logging

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your pre-trained model
model_path = 'skin_diseases.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(244, 244))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Define the class labels in the correct order
class_labels = [
    'Atopic Dermatitis', 
    'Eczema', 
    'Melanocytic Nevi', 
    'Psoriasis pictures Lichen Planus and related diseases', 
    'Seborrheic Keratoses and other Benign Tumors', 
    'Tinea Ringworm Candidiasis and other Fungal Infections', 
    'Warts Molluscum and other Viral Infections'
]

@app.route('/')
def home():
    return "Welcome to the Flask App!"


@app.route('/predict', methods=['POST'])
def upload_image():
    if 'predictions' not in session:
        session['predictions'] = []
    
    if 'file' not in request.files:
        flash("No file part")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logging.info(f"Saving file to {file_path}")
            file.save(file_path)

            # Preprocess the image
            img_array = preprocess_image(file_path)

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions[0])]
            session['predictions'].append(predicted_class)
            session.modified = True  # Ensure session is saved

            logging.info(f"Current predictions: {session['predictions']}")
            os.remove(file_path)

            if len(session['predictions']) == 5:
                # Determine the final prediction
                risk_counts = {label: 0 for label in class_labels}
                for pred in session['predictions']:
                    risk_counts[pred] += 1

                # Simple logic to determine the most frequent prediction
                final_prediction = max(risk_counts, key=risk_counts.get)

                session.pop('predictions', None)  # Clear the session
                return jsonify({"Model_Prediction": final_prediction})
            else:
                return jsonify({"message": "Image processed, please upload more images", "uploaded_images": len(session['predictions'])})

        except Exception as e:
            logging.error(f"Error processing the image: {e}")
            return jsonify({"Error": str(e)})

    return jsonify({"error": "Failed to process the image"})

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=10000)
