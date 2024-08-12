from flask import Flask, render_template,request,jsonify
import sklearn
from utils import make_prediction



app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    email_text = request.form.get("email-content")
    prediction = make_prediction(email_text)
    return render_template("index.html", predictions = prediction, email_text = email_text)

@app.route("/api/predict", methods = ["POST"])
def api_predict():
    data = request.get_json(force = True) #forcing it to convert to a json format
    email_text = data["content"]
    prediction = make_prediction(email_text)
        
    return jsonify({"prediction" : prediction})
    
    
    
if __name__ == "__main__":
    app.run(debug=True)