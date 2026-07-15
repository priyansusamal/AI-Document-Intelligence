# AI Document Intelligence Platform

An AI-powered web application that classifies text messages as **Spam** or **Ham (Safe)** using **Machine Learning** and **Natural Language Processing (NLP)**. The application also provides interactive analytics, word cloud visualization, and batch prediction capabilities through a user-friendly Streamlit interface.

---

## Features

- **Spam Detection**
  - Predict whether a message is Spam or Ham.

- **Analytics Dashboard**
  - Dataset statistics
  - Spam vs Ham distribution
  - Message length analysis
  - Interactive charts

- **Word Cloud Analysis**
  - Spam Word Cloud
  - Ham Word Cloud
  - Most frequent words visualization

- **Batch Prediction**
  - Upload a CSV file containing multiple messages
  - Predict all messages at once
  - Download prediction results

- **Interactive Streamlit UI**
  - Clean and responsive interface
  - Easy navigation
  - Real-time predictions

---

## Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### Machine Learning
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Naive Bayes
- Random Forest

### Data Processing
- Pandas
- NumPy

### Visualization
- Plotly
- Matplotlib
- WordCloud

### Model Storage
- Joblib

---

## Project Structure

```text
AI_Document_Intelligence/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
│
├── assets/
│   └── style.css
│
├── dataset/
│   └── spam.csv
│
├── models/
│   ├── model.pkl
│   └── metrics.json
│
├── pages/
│   ├── 1_Analytics.py
│   ├── 2_WordCloud.py
│   └── 3_Batch_Prediction.py
│
└── utils/
    ├── loader.py
    ├── predictor.py
    └── history.py
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AI_Document_Intelligence.git

cd AI_Document_Intelligence
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train.py
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## Application Pages

-  Home
-  Analytics Dashboard
-  Word Cloud Analysis
-  Batch Prediction

---

## Machine Learning Workflow

1. Load Dataset
2. Preprocess Text
3. Convert Text to TF-IDF Features
4. Train Multiple Machine Learning Models
5. Save the Best Model
6. Predict New Messages
7. Visualize Results

---

## Future Improvements

-  Email Spam Detection
-  Multi-language Support
-  Cloud Deployment
-  Mobile Responsive Design
-  Deep Learning Models (LSTM/BERT)
-  PDF Report Generation
-  Prediction History Dashboard

---

## Deployed Link 
https://ai-document-intelligence-fvkehrkp6fkwmcnivsnvva.streamlit.app/

## 👨‍💻 Developed By

**Priyansu Samal**

ScholarX Machine Learning Project

2026

---

## License

This project is developed for educational and portfolio purposes.
