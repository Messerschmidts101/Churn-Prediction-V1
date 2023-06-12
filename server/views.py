from flask import Blueprint, request
from model import Customer

api = Blueprint('api', __name__)

@api.route("/predict", methods = ["POST", "GET"])
def predict():
    customer = request.get_json()
    customer_predict = Customer()