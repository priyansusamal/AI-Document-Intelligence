import joblib
import json

def load_model():
    return joblib.load("models/model.pkl")


def load_metrics():
    with open("models/metrics.json", "r") as f:
        return json.load(f)