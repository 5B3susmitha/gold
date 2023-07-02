import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('gold_price_prediction_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    input_features = [np.array(features)]
    prediction = model.predict(input_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted Gold Price: ${}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
