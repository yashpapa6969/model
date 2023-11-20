from flask import Flask, request, jsonify
import io
import requests
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
import os

model = tf.keras.models.load_model('imageclassifier.h5')
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def hello():
    image_url = request.json['image_url']

    response = requests.get(image_url)
    
    if response.status_code != 200:
        return jsonify({'error': 'Failed to download image.'}), 400

    image = Image.open(io.BytesIO(response.content))
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)

    predictions1 = np.round(predictions)
    return jsonify({'predictions': predictions1.tolist()})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
