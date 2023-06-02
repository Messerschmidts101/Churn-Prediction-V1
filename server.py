from flask import Flask,request,jsonify,render_template,session,send_from_directory
import joblib
import pandas as pd
import numpy as np

# Load classifier
churnprediction = joblib.load('model/churnprediction.pkl')

# Create flask app
flask_app = Flask(__name__,static_folder='static')

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST","GET"])
def predict():
    if request.method == "POST":
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = churnprediction.predict_proba(features)[0][0]
        print("Prediction: ", prediction)
        return render_template("index.html", Prediction_Here="The customer churn probability is: {}".format(prediction))
    else:
        return "Method Not Allowed"

if __name__ == "__main__":
    flask_app.run(debug=True, host='0.0.0.0',port=8080)
'''

@flask_app.route("/")
def Home():
    return render_template("index.html")
@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = classifier.predict(features)
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))
'''