import joblib
import numpy as np
from flask import Flask, request, jsonify

# import app.app_config as c

app = Flask(__name__)
model = joblib.load(f'high_quality_red_wine_classifier_random_forest_2024-01-17-17-26-33.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)

        # Assume input data is a dictionary with feature names and values
        features = data.get('features', {})

        # Make sure there are features in the input data
        if not features:
            return jsonify({'error': 'No features provided'})

        model_feature_names = model.feature_names_in_
        feature_values = [features.get(name) for name in model_feature_names]

        # Make predictions using the pre-trained model
        prediction = model.predict([feature_values])[0]

        # Return the prediction as JSON
        return jsonify({'prediction': str(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)