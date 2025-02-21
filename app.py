from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import cv2
import numpy as np
import os
from torchvision import models
import torch.nn as nn
import random
from datetime import datetime

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load pre-trained ResNet model
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 29)  # Updated to match actual number of classes
    
    try:
        # Load the pre-trained weights for plant disease detection
        weights_path = 'plant_disease_model.pth'
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print("Successfully loaded pre-trained plant disease model")
        else:
            print("Warning: Pre-trained model not found. Using base ResNet model.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    
    model.eval()
    return model

# Initialize model
model = load_model()

# Transform for the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_disease(image):
    """Predict disease from image without requiring plant type specification"""
    try:
        # Convert to PIL Image if it's not already
        if not isinstance(image, Image.Image):
            image = Image.open(io.BytesIO(image))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # First check for human presence
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            return {
                'success': True,
                'human_detected': True,
                'message': 'Human detected in image'
            }
        
        # Transform image for disease detection
        image_tensor = transform(image).unsqueeze(0)
        
        # Debug: Check the shape of the image tensor
        print(f"Image tensor shape: {image_tensor.shape}")
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Debug: Check the output probabilities
            print(f"Output probabilities: {probabilities}")
            
            # Get top 3 predictions
            confidence, predicted = torch.topk(probabilities, k=3, dim=1)
            confidence = confidence.squeeze().tolist()
            predicted = predicted.squeeze().tolist()
            
            if isinstance(predicted, int):  # Handle single prediction case
                predicted = [predicted]
                confidence = [confidence]
            
            # Debug: Check the predicted classes and confidence scores
            print(f"Predicted classes: {predicted}")
            print(f"Confidence scores: {confidence}")
            
            if confidence[0] > 0.7:  # Increased confidence threshold for primary prediction
                results = []
                for conf, pred in zip(confidence, predicted):
                    disease_name = get_disease_name(pred)
                    results.append({
                        'disease': disease_name,
                        'confidence': min(conf * 100, 100)
                    })
                
                return {
                    'success': True,
                    'human_detected': False,
                    'predictions': results
                }
            else:
                return {
                    'success': False,
                    'error': 'No disease detected with high confidence. Please ensure the image shows clear plant symptoms.'
                }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_disease_name(class_idx):
    """Convert class index to disease name"""
    # This mapping should match your pre-trained model's classes
    disease_classes = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Rice___Bacterial_leaf_blight',
        'Rice___Brown_spot',
        'Rice___Leaf_blast',
        'Rice___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    return disease_classes[class_idx] if class_idx < len(disease_classes) else 'Unknown'

# Thresholds for alerts (adjusted for Kathmandu climate)
THRESHOLDS = {
    'temperature': {'min': 15, 'max': 30},  # Kathmandu's comfortable range
    'humidity': {'min': 40, 'max': 80},     # in percentage
    'soil_moisture': {'min': 30, 'max': 70}  # in percentage
}

# Plant care recommendations based on conditions
PLANT_CARE_TIPS = {
    'high_temperature': 'Temperature is high for Kathmandu! Consider providing shade or moving plants to a cooler spot.',
    'low_temperature': 'Temperature is low for Kathmandu! Move plants to a warmer location or provide protection.',
    'high_humidity': 'Humidity is high! Improve air circulation to prevent fungal diseases.',
    'low_humidity': 'Humidity is low! Consider using a humidifier or misting the plants.',
    'high_moisture': 'Soil is too wet! Reduce watering frequency and ensure proper drainage.',
    'low_moisture': 'Soil is too dry! Water your plants and consider adding mulch to retain moisture.'
}

def get_sensor_data():
    """Get current sensor readings with Kathmandu's typical ranges"""
    # Kathmandu's temperature typically ranges from 12°C to 30°C
    hour = datetime.now().hour
    
    # Simulate daily temperature variation
    if 6 <= hour < 12:  # Morning
        temp_range = (15, 25)
    elif 12 <= hour < 17:  # Afternoon
        temp_range = (20, 30)
    elif 17 <= hour < 22:  # Evening
        temp_range = (18, 25)
    else:  # Night
        temp_range = (12, 18)
    
    # Generate temperature within the appropriate range for the time of day
    temperature = round(random.uniform(temp_range[0], temp_range[1]), 1)
    
    # Humidity is typically higher in the morning and evening
    if 6 <= hour < 10 or 17 <= hour < 20:
        humidity_range = (60, 80)
    else:
        humidity_range = (40, 60)
    
    humidity = round(random.uniform(humidity_range[0], humidity_range[1]), 1)
    
    return {
        'temperature': temperature,
        'humidity': humidity,
        'soil_moisture': round(random.uniform(30, 70), 1)  # Soil moisture remains consistent
    }

def get_alerts(sensor_data):
    """Generate alerts based on sensor readings"""
    alerts = []
    
    # Temperature alerts
    if sensor_data['temperature'] > THRESHOLDS['temperature']['max']:
        alerts.append({
            'type': 'danger',
            'message': PLANT_CARE_TIPS['high_temperature'],
            'sensor': 'temperature',
            'value': sensor_data['temperature'],
            'unit': '°C'
        })
    elif sensor_data['temperature'] < THRESHOLDS['temperature']['min']:
        alerts.append({
            'type': 'warning',
            'message': PLANT_CARE_TIPS['low_temperature'],
            'sensor': 'temperature',
            'value': sensor_data['temperature'],
            'unit': '°C'
        })
    
    # Humidity alerts
    if sensor_data['humidity'] > THRESHOLDS['humidity']['max']:
        alerts.append({
            'type': 'danger',
            'message': PLANT_CARE_TIPS['high_humidity'],
            'sensor': 'humidity',
            'value': sensor_data['humidity'],
            'unit': '%'
        })
    elif sensor_data['humidity'] < THRESHOLDS['humidity']['min']:
        alerts.append({
            'type': 'warning',
            'message': PLANT_CARE_TIPS['low_humidity'],
            'sensor': 'humidity',
            'value': sensor_data['humidity'],
            'unit': '%'
        })
    
    # Soil moisture alerts
    if sensor_data['soil_moisture'] > THRESHOLDS['soil_moisture']['max']:
        alerts.append({
            'type': 'danger',
            'message': PLANT_CARE_TIPS['high_moisture'],
            'sensor': 'soil_moisture',
            'value': sensor_data['soil_moisture'],
            'unit': '%'
        })
    elif sensor_data['soil_moisture'] < THRESHOLDS['soil_moisture']['min']:
        alerts.append({
            'type': 'warning',
            'message': PLANT_CARE_TIPS['low_moisture'],
            'sensor': 'soil_moisture',
            'value': sensor_data['soil_moisture'],
            'unit': '%'
        })
    
    return alerts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    try:
        # Read the image file
        image_bytes = file.read()
        # Save the file
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(filename, 'wb') as f:
            f.write(image_bytes)
        
        # Predict disease
        result = predict_disease(image_bytes)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data received'})

        # Get the base64 string and convert to image
        image_data = data['image'].split('base64,')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Predict disease
        result = predict_disease(image_bytes)
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/sensor_data')
def sensor_data():
    """Endpoint to get current sensor readings and alerts"""
    data = get_sensor_data()
    alerts = get_alerts(data)
    return jsonify({
        'success': True,
        'data': data,
        'alerts': alerts,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    app.run(debug=True)