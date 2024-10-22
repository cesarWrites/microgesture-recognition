from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Enable CORS
CORS(app)

#Load the trained model
model = load_model('model/microgesture_recognition_model.h5')

emotion_labels = ['anger', 'disgust', 'happy', 'sad', 'contempt', 'fear','neutral', 'surprise']


# Decode base64 image
def decode_base64_image(img_base64):
    img_data = base64.b64decode(img_base64.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

# Preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (48, 48))  # Resize to model input size
    image = image.astype('float32') / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Endpoint for emotion recognition
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_base64 = data['image']
    image = decode_base64_image(image_base64)
    preprocessed_image = preprocess_image(image)
    
    # Predict the emotion
    prediction = model.predict(preprocessed_image)
    emotion_index = np.argmax(prediction)
    emotion_label = emotion_labels[emotion_index]
    confidence = float(prediction[0][emotion_index])
    
    return jsonify({'emotion': emotion_label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

