from flask import Flask, request, jsonify
import io
import requests
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify

model = tf.keras.models.load_model('imageclassifier.h5')
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def hello():
    # Get the image URL from the request
    image_url = request.json['image_url']

    # Download the image from the URL
    response = requests.get(image_url)
    
    if response.status_code != 200:
        return jsonify({'error': 'Failed to download image.'}), 400

    # Decode and preprocess the image
    image = Image.open(io.BytesIO(response.content))
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    # Make predictions using the loaded TensorFlow model
    predictions = model.predict(image)

    # Process the predictions as needed
    # ...
    predictions1 = np.round(predictions)
    # Return the predictions as a JSON response
    return jsonify({'predictions': predictions1.tolist()})

if __name__ == '__main__':
    app.run()
