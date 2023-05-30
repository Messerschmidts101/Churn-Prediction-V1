from flask import Flask,request,jsonify,render_template,session,send_from_directory

# Create flask app
flask_app = Flask(__name__,static_folder='static')


@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/_get_data", methods = ["POST",'GET'])
def predict():
    data = request.get_json()
    print("DATA FROM JS: ",data)
    return data
    #return '', 200


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