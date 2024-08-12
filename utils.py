import pickle

cv = pickle.load(open("models/cv.pkl", "rb")) #since cv.pkl file is a binary file
model = pickle.load(open("models/clf.pkl", "rb"))

def make_prediction(email_text):
    tokenized_email = cv.transform([email_text])
    prediction = model.predict(tokenized_email)
    
    if prediction == 1:
        prediction = 1
    else:
        prediction = -1
    
    return prediction