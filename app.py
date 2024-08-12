from flask import Flask, render_template,request,jsonify
import pickle
import sklearn

cv = pickle.load(open("models/cv.pkl", "rb")) #since cv.pkl file is a binary file
model = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    email_text = request.form.get("email-content")
    tokenized_email = cv.transform([email_text])
    predictions = model.predict(tokenized_email)
    
    if predictions == 1:
        predictions = 1
    else:
        predictions = -1
    
    
    return render_template("index.html", predictions = predictions, email_text = email_text)

@app.route("/api/predict", methods = ["POST"])
def api_predict():
    data = request.get_json(force = True) #forcing it to convert to a json format
    email_text = data["content"]
    tokenized_email = cv.transform([email_text])
    predictions = model.predict(tokenized_email)
    
    if predictions == 1:
        predictions = 1
    else:
        predictions = -1
        
    return jsonify({"prediction" : predictions})
    
    
    
if __name__ == "__main__":
    app.run(debug=True)