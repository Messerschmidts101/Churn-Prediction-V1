import io
import re
from flask import Flask,request,jsonify,redirect,session,send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
# from .models import Customer
from DataProcessor import DataProcessor
'''
dataprocessor = DataProcessor()
tblDataset = dataprocessor.one_hot_encode(data_set = 'insert dataset here', arrExcludedColumns=['churn'])
tblDataset = dataprocessor.process_csv('insert dataset here')
'''

# Load classifier
churnprediction = joblib.load('model/churnprediction.pkl')

# Create flask app
app = Flask(__name__, static_folder='static')
CORS(app)

data_processor =  DataProcessor()

@app.route("/")
def Home():
    return "Wilcommen!"

@app.route("/predict", methods = ["POST","GET"])
def predict():
    feature_post = request.get_json()
    if type(feature_post) == dict:
        features_list = feature_post["feature_post"]
        features = [{"name": feature["name"], "value": float(feature["value"])} for feature in features_list]
        features = {item['name']: item['value'] for item in features}
        features = data_processor.process_row(features)
        print("features: ", features)

        feature = [np.array([float(feature) for feature in features.values()])]
        prediction = churnprediction.predict_proba(feature)[0][1]
        print(" +++ === --- DEBUGGER LOGS START --- === +++  ")
        print("Features: ", re.sub(r'  ', '', str(feature)))
        print("Prediction: ", prediction)
        print(" +++ === --- DEBUGGER LOGS END --- === +++  ")

        return jsonify(prediction)
    else:
        return "Method not Allowed"

@app.route("/predict_dataset", methods=["POST","GET"])
def predict_dataset():
    file_data = request.data
    # Convert the file_data into a pandas DataFrame
    df = pd.read_csv(io.BytesIO(file_data))
    # Process the DataFrame as needed
    # df = df.drop("churn", axis=1)
    prediction = pd.DataFrame(churnprediction.predict_proba(df))[1].tolist()
    print(" +++ === --- DEBUGGER LOGS START --- === +++  ")
    print('Data shape:', df.shape)
    print('Data columns:', df.columns)
    print('Data head:', df.head())
    print("Prediction ",prediction.sort(reverse=True))
    print(" +++ === --- DEBUGGER LOGS END --- === +++  ")

    # return [{"name": i, "data": each} for i, each in enumerate(prediction)]
    return { "data": prediction }

@app.route("/predict_dataset_feature", methods=["POST","GET"])
def predict_dataset_feature():
    file_data = request.data
    # Convert the file_data into a pandas DataFrame
    df = pd.read_csv(io.BytesIO(file_data))
    # Process the DataFrame as needed
    df = data_processor.one_hot_encode(df, "churn")
    df = data_processor.process_csv(df)
    # df = df.drop("churn", axis=1)
    # prediction = pd.DataFrame(churnprediction.predict_proba(df))[1].tolist()
    df['churn'] = pd.DataFrame(churnprediction.predict_proba(df))[1].tolist()
    df = df.sort_values('churn', ascending=False).reset_index(drop=True)
    print(" +++ === --- DEBUGGER LOGS START --- === +++  ")
    print('Data shape:', df.shape)
    print('Data columns:', df.columns)
    print('Data head:', df.head())
    # sprint("Prediction ",prediction.sort(reverse=True))
    print(" +++ === --- DEBUGGER LOGS END --- === +++  ")

    # return [{"name": i, "data": each} for i, each in enumerate(prediction)]
    return df.to_json()

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
        {
            id: 19,
            name: "churn"
        },
    ]
    return features

if __name__ == "__main__":
    app.run(debug=True, host='localhost',port=8080)

"""
predict([[0,26,161.6],[1,0,243.4],[1,0,299.4]])
:::['churned','not-churned','churned']

predict_proba([[0,26,161.6]])
:::[[.9,.1],[.35,.65],[.51,.49]]
"""

