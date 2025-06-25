from flask import Flask, render_template, request
from model import load_and_train
import numpy as np

app = Flask(__name__)
model, scaler, feature_list, metrics = load_and_train()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = [float(request.form.get(f)) for f in feature_list]
            input_scaled = scaler.transform([input_data])
            pred = model.predict(input_scaled)[0]
            prediction = 'High Risk of Osteoporosis' if pred == 1 else 'Low Risk'
        except:
            prediction = 'Invalid input — please enter numbers only.'

    return render_template('index.html', prediction=prediction, metrics=metrics, feature_list=feature_list)

if __name__ == '__main__':
    app.run(debug=True)
