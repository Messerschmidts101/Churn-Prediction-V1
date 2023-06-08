from flask import Flask,request,jsonify,render_template,session,send_from_directory
import joblib
# import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from .model import Customer

# Load classifier
churnprediction = joblib.load('model/churnprediction.pkl')

# Create flask app
app = Flask(__name__, static_folder='static')
app.comfig["SQLALCHEMY_DATABASE_URI"] = 'sqlite://database.db'
db = SQLAlchemy()
db.init_app(app=app)

@app.route("/")
def Home():
    return "Welcome!"

@app.route("/members")
def members():
    members = {"members" : ["mem1", "mem2"]}
    return members

@app.route("/prediction", methods = ["POST","GET"])
def prediction():
    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        features = [np.array(features)]
        prediction = churnprediction.predict_proba(features)[0][1]
        print("Prediction: ", prediction)
        #return render_template("client/public/index.html", Prediction_Here="The customer churn probability is: {}".format(prediction))
        return "Predict"
    else:
        return "Method Not Allowed"
    

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=8080)

"""
predict([[0,26,161.6],[1,0,243.4],[1,0,299.4]])
:::['churned','not-churned','churned']

predict_proba([[0,26,161.6]])
:::[[.9,.1],[.35,.65],[.51,.49]]
"""

