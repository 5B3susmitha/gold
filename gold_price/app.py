# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)
model = joblib.load('gld_price_dataset_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    spx = float(request.form['spx'])
    uso = float(request.form['uso'])
    slv = float(request.form['slv'])
    eur_usd = float(request.form['eur_usd'])

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'spx': [spx],
        'uso': [uso],
        'slv': [slv],
        'eur_usd': [eur_usd]
    })

    # Make the prediction using the trained model
    prediction = model.predict(input_data)[0]

    # Return the prediction as JSON response
    return jsonify({'prediction': round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)
