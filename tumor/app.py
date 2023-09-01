from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app= Flask(__name__)

# Load the brain tumor detection model
brain_tumor_model = tf.keras.models.load_model("brain_tumor_detection_model.h5")



# Define the route for the home page
@app.route('/')
def home():
    return render_template('brain_tumor.html')

# Define a route to handle brain tumor detection
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
            
            # Make a prediction for brain tumor
            prediction = brain_tumor_model.predict(img)
            
            # Map the prediction to class names
            class_names = ['No Tumor', 'Tumor']
            predicted_class = class_names[np.argmax(prediction)]
            
            return jsonify({'prediction': predicted_class})
    
    return render_template('brain_tumor.html')




if __name__ == '__main__':
    app.run(debug=True)
