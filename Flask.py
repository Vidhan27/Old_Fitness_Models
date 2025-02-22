from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

try:
    # Load health status model and scaler
    with open('health_status_predictor.pkl', 'rb') as file:
        health_status = pickle.load(file)
    health_status_model = health_status['model']
    health_status_scaler = health_status['scaler']

    # Load sleep quality model and scaler
    with open('sleep_quality_predictor.pkl', 'rb') as file:
        sleep_quality = pickle.load(file)
    sleep_quality_model = sleep_quality['model']
    sleep_quality_scaler = sleep_quality['scaler']

    # Load calorie model and scaler separately
    with open('calorie_prediction_model.pkl', 'rb') as file:
        calorie = pickle.load(file)

    with open('calorie_scaler.pkl', 'rb') as file:
        calorie_scaler = pickle.load(file)

except Exception as e:
    print(f'Error loading models: {e}')
    raise

def create_features(features):
    df = features.copy()
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Intensity'] = df['Heart_Rate'] / df['Age']
    df['Energy_Index'] = df['Duration'] * df['Heart_Rate'] / 100
    df['Temperature_HR_Interaction'] = df['Body_Temp'] * df['Heart_Rate']
    df['Weight_Duration_Ratio'] = df['Weight'] / df['Duration']
    df['Metabolic Index'] = df['BMI'] * df['Heart_Rate'] / df['Age']
    df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
    return df

@app.route('/predict/health_status', methods=['POST'])
def predict_heart_status():
    try:
        data = request.get_json()
        features = [
            float(data['heart_rate']),
            float(data['temperature']),
            float(data['oxygen_level'])
        ]
        features_array = np.array(features).reshape(1, -1)
        scaled_features = health_status_scaler.transform(features_array)
        prediction = health_status_model.predict(scaled_features)[0]
        probability = health_status_model.predict_proba(scaled_features)

        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'probability': float(probability[0][1])
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/predict/calories', methods=['POST'])
def predict_calories():
    try:
        data = request.get_json()
        features = pd.DataFrame([[  
            str(data['gender']).lower(),
            int(data['age']),
            float(data['height']),
            float(data['weight']),
            float(data['duration']),
            float(data['heart_rate']),
            float(data['body_temp'])
        ]], columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

        features = create_features(features)
        scaled_features = calorie_scaler.transform(features)
        prediction = calorie.predict(scaled_features)

        return jsonify({
            'status': 'success',
            'prediction': float(prediction[0])
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/predict/sleep', methods=['POST'])
def predict_sleep():
    try:
        data = request.get_json()
        features = [
            float(data['avg_heart_rate']),
            float(data['movement_count']),
            float(data['room_temp']),
            float(data['sleep_duration']),
            float(data['oxygen_level'])
        ]
        features_array = np.array(features).reshape(1, -1)
        scaled_features = sleep_quality_scaler.transform(features_array)
        prediction = sleep_quality_model.predict(scaled_features)[0]
        probability = sleep_quality_model.predict_proba(scaled_features)

        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'probability': float(probability[0][1])
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'heart_status': health_status is not None,
            'calorie': calorie is not None,
            'calorie_scaler': calorie_scaler is not None,
            'sleep_quality': sleep_quality is not None
        }
    })

if __name__ == '__main__':
    PORT=os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=PORT)
