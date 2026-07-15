import joblib

model = joblib.load("models/model.pkl")

def predict_message(message):

    prediction = model.predict([message])[0]

    return prediction