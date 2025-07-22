from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(_name_)

# Load the trained model
try:
    with open('heart_failure_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    print("Model loaded successfully!")
    print(f"Features: {feature_names}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded properly'})
        
        # Get form data
        features = []
        feature_values = {}
        
        # Expected features (typical heart failure dataset features)
        expected_features = [
            'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
            'ejection_fraction', 'high_blood_pressure', 'platelets',
            'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
        ]
        
        # Get values from form
        for feature in expected_features:
            value = request.form.get(feature)
            if value is not None:
                # Convert to appropriate type
                if feature in ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']:
                    feature_values[feature] = int(value)
                else:
                    feature_values[feature] = float(value)
                features.append(feature_values[feature])
        
        # Create input array
        input_data = np.array(features).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Prepare result
        result = {
            'prediction': int(prediction),
            'probability_no_death': round(probability[0] * 100, 2),
            'probability_death': round(probability[1] * 100, 2),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'input_values': feature_values
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'OK',
        'model_loaded': model is not None,
        'features': feature_names if feature_names else []
    })

if _name_ == '_main_':
    app.run(debug=True, host='0.0.0.0', port=5000)
