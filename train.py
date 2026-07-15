"""
AI Document Intelligence Platform
Train Machine Learning Model
"""

import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Create models folder
os.makedirs("models", exist_ok=True)

# Load Dataset
df = pd.read_csv("dataset/spam.csv")

# Split data
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_model = None
best_accuracy = 0
best_name = ""

results = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english")),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    precision = precision_score(
        y_test,
        pred,
        pos_label="spam"
    )

    recall = recall_score(
        y_test,
        pred,
        pos_label="spam"
    )

    f1 = f1_score(
        y_test,
        pred,
        pos_label="spam"
    )

    results[name] = {
        "accuracy": round(accuracy,4),
        "precision": round(precision,4),
        "recall": round(recall,4),
        "f1": round(f1,4)
    }

    print(f"{name} Accuracy : {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_name = name
        best_model = pipeline

# Save Best Pipeline
joblib.dump(best_model, "models/model.pkl")

metrics = {
    "best_model": best_name,
    "accuracy": round(best_accuracy,4),
    "results": results
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nBest Model :", best_name)
print("Accuracy :", best_accuracy)
print("\nTraining Complete!")