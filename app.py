from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/tips_model.pkl')

@app.route('/')
def home():
    """
    Health check endpoint for the API.
    """
    return "Tips Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict total bill based on input features.
    Expected JSON input:
    {
        "sex": str,
        "smoker": str,
        "day": str,
        "time": str,
        "size": int
    }
    """
    try:
        # Parse input data
        data = request.get_json()

        # Validate input fields
        required_fields = ['sex', 'smoker', 'day', 'time', 'size']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing '{field}' in request"}), 400

        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])

        # Preprocess input data (ensure the same encoding as during training)
        label_encoders = {
            'sex': {'Male': 1, 'Female': 0},
            'smoker': {'Yes': 1, 'No': 0},
            'day': {'Thur': 0, 'Fri': 1, 'Sat': 2, 'Sun': 3},
            'time': {'Lunch': 0, 'Dinner': 1}
        }

        for col, mapping in label_encoders.items():
            if col in input_df:
                input_df[col] = input_df[col].map(mapping)

        # Ensure all columns are in the correct order
        input_df = input_df[['sex', 'smoker', 'day', 'time', 'size']]

        # Predict total bill
        prediction = model.predict(input_df)

        # Return prediction
        return jsonify({
            "input": data,
            "predicted_total_bill": prediction[0]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
