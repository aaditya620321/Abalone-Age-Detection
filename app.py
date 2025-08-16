from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Flask app
app = Flask(__name__)

# History storage (in-memory list)
prediction_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sex = int(request.form['sex'])
    length = float(request.form['length'])
    diameter = float(request.form['diameter'])
    height = float(request.form['height'])
    wholeWeight = float(request.form['wholeWeight'])
    Shuckedweight = float(request.form['Shuckedweight'])
    Visceraweight = float(request.form['Visceraweight'])
    Shellweight = float(request.form['Shellweight'])

    features = pd.DataFrame([{
        'Sex': sex,
        'Length': length,
        'Diameter': diameter,
        'Height': height,
        'Whole weight': wholeWeight,
        'Shucked weight': Shuckedweight,
        'Viscera weight': Visceraweight,
        'Shell weight': Shellweight
    }])

    age = model.predict(features)[0]

    # Save this prediction to history
    prediction_history.append({
        'Sex': sex,
        'Length': length,
        'Diameter': diameter,
        'Height': height,
        'Whole Weight': wholeWeight,
        'Shucked Weight': Shuckedweight,
        'Viscera Weight': Visceraweight,
        'Shell Weight': Shellweight,
        'Predicted Age': round(age, 2)
    })

    return render_template('index.html', age=round(age, 2))

@app.route('/history')
def history():
    return render_template('history.html', history=prediction_history)

if __name__ == "__main__":
    app.run(debug=True)
