from flask import Flask,request,jsonify,render_template,session,send_from_directory
import joblib
# import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
# from .models import Customer

# Load classifier
churnprediction = joblib.load('model/churnprediction.pkl')

# Create flask app
app = Flask(__name__, static_folder='static')
@app.route("/")
def Home():
    return "Welcome!"

@app.route("/prediction", methods = ["POST","GET"])
def prediction():
    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        features = [np.array(features)]
        prediction = churnprediction.predict_proba(features)[0][1]
        print("features: ",features)
        print("Prediction: ", prediction)

        #return render_template("client/public/index.html", Prediction_Here="The customer churn probability is: {}".format(prediction))
        return prediction
    else:
        return "Method Not Allowed"


@app.route("/feature_names", methods=["GET"])
def feature_names():
    id = "id"
    name = "name"
    features = [
        {
            id: 0,
            name:"state"
        },
        {
            id: 1,
            name: "account_length"
        },
        {
            id: 2,
            name: "area_code"
        },
        {
            id: 3,
            name: "international_plan"
        },
        {
            id: 4,
            name: "voice_mail_plan"
        },
        {
            id: 5,
            name: "number_vmail_messages"
        },
        {
            id: 6,
            name: "total_day_minutes"
        },
        {
            id: 7,
            name: "total_day_calls"
        },
        {
            id: 8,
            name: "total_day_charge"
        },
        {
            id: 9,
            name: "total_eve_minutes"
        },
        {
            id: 10,
            name: "total_eve_calls"
        },
        {
            id: 11,
            name: "total_eve_charge"
        },
        {
            id: 12,
            name: "total_night_minutes"
        },
        {
            id: 13,
            name: "total_night_calls"
        },
        {
            id: 14,
            name: "total_night_charge"
        },
        {
            id: 15,
            name: "total_intl_minutes"
        },
        {
            id: 16,
            name: "total_intl_calls"
        },
        {
            id: 17,
            name: "total_intl_charge"
        },
        {
            id: 18,
            name: "number_customer_service_calls"
        },
    ]
    return features

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=8080)

"""
predict([[0,26,161.6],[1,0,243.4],[1,0,299.4]])
:::['churned','not-churned','churned']

predict_proba([[0,26,161.6]])
:::[[.9,.1],[.35,.65],[.51,.49]]
"""

