from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Custom F1 score metric function for Alzheimer's model
def f1_score(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1_val

# Register the custom metric function
tf.keras.utils.get_custom_objects().update({'f1_score': f1_score})

# Load the saved Alzheimer's model
alzheimer_model = tf.keras.models.load_model("alzheimer_inception_cnn_model")

# Load the saved Brain Tumor model
brain_tumor_model = tf.keras.models.load_model("brain_tumor_detection_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/alzheimer', methods=['GET', 'POST'])
def alzheimer():
    if request.method == 'POST':
        # Get the uploaded image from the request
        uploaded_image = request.files['image']

        if uploaded_image:
            # Extract image data and load it
            image_data = uploaded_image.read()
            img = tf.image.decode_image(image_data, channels=3)
            img = tf.image.resize(img, (176, 176))
            img = img.numpy()  # Convert to NumPy array
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Make a prediction
            prediction = alzheimer_model.predict(img)

            # Map the prediction to class names
            class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
            predicted_class = class_names[np.argmax(prediction)]

            return jsonify({'prediction': predicted_class})

    return render_template('alzheimer.html')

@app.route('/brain_tumor', methods=['GET', 'POST'])
def brain_tumor():
    if request.method == 'POST':
        # Get the uploaded image from the request
        uploaded_image = request.files['image']

        if uploaded_image:
            # Extract image data and load it
            image_data = uploaded_image.read()
            img = tf.image.decode_image(image_data, channels=3)
            img = tf.image.resize(img, (224, 224))
            img = img.numpy()  # Convert to NumPy array
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Make a prediction
            prediction = brain_tumor_model.predict(img)

            # Map the prediction to class names
            class_names = ['No Tumor', 'Tumor']
            predicted_class = class_names[int(np.round(prediction))]

            return jsonify({'prediction': predicted_class})

    return render_template('brain_tumor.html')

if __name__ == '__main__':
    app.run(debug=True)
