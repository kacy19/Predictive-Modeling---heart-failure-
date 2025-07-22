# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(_name_)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        data = [float(request.form[key]) for key in request.form]
        input_array = np.array([data])
        prediction = model.predict(input_array)
        output = "Patient is likely to die." if prediction[0] == 1 else "Patient is likely to survive."
        return render_template("index.html", prediction_text=output)

if _name_ == "_main_":
    app.run(debug=True)
