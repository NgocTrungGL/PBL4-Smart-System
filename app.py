import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PBL import L1Dist

app = Flask(__name__)

model_path = 'siamesemodelv2.h5'
siamese_model = load_model(model_path, custom_objects={'L1Dist': L1Dist})

def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

def verify(detection_threshold=0.5, verification_threshold=0.5):
    results = []
    
    input_img_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
    input_img = preprocess_image(input_img_path)
    
    for image_name in os.listdir(os.path.join('application_data', 'verification_images')):
        validation_img_path = os.path.join('application_data', 'verification_images', image_name)
        validation_img = preprocess_image(validation_img_path)
        
        result = siamese_model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(results)
    verified = verification > verification_threshold
    
    return results, verified

@app.route('/verify', methods=['POST'])
def verify_user():
    file = request.files['file']
    file_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
    file.save(file_path)
    
    results, verified = verify()
    return jsonify({'verified': verified, 'results': results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
