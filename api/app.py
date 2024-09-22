from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load('../models/loan_approval_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    result = 'Approved' if prediction == 1 else 'Not Approved'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
