# 📩 SMS Spam Detection System

## 📌 Project Overview

This project is an SMS Spam Detection System built using Natural Language Processing (NLP) and Machine Learning.

The system classifies SMS messages into:

- ✅ Ham (Not Spam)
- 🚨 Spam

A Logistic Regression model combined with TF-IDF Vectorization is used to achieve high classification performance.

---

## 🚀 Features

- SMS Spam Detection
- NLP Text Preprocessing
- TF-IDF Vectorization
- Logistic Regression Classifier
- Streamlit Web Application
- Spam Probability Prediction
- Interactive User Interface

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-Learn
- Streamlit
- Joblib

---

## 📂 Project Structure

```text
SMS_Spam_Classifier/

│
├── app2.py
├── requirements.txt
├── SMS_Spam_Classifier.ipynb
├── spam_model.pkl
└── tfidf.pkl
```

### File Description

| File | Description |
|--------|-------------|
| app.py | Streamlit Web Application |
| requirements.txt | Required Python Libraries |
| SMS_Spam_Classifier.ipynb | Model Training Notebook |
| spam_model.pkl | Trained Logistic Regression Model |
| tfidf.pkl | Saved TF-IDF Vectorizer |

---

## 🔄 Machine Learning Workflow

### 1. Data Cleaning

- Lowercase conversion
- Special character removal
- Stopword removal
- Stemming

### 2. Feature Engineering

TF-IDF Vectorization

### 3. Model Training

Models Evaluated:

- Naive Bayes
- Logistic Regression
- Random Forest

### 4. Model Selection

Logistic Regression was selected as the final model based on overall performance.

---

## 📊 Model Performance

| Model | Accuracy | F1 Score |
|---------|---------|---------|
| Naive Bayes | 97.04% | 0.875 |
| Logistic Regression | **97.58%** | **0.910** |
| Random Forest | 97.22% | 0.884 |

### Final Model Metrics

- Accuracy: 97.58%
- F1 Score: 0.910
- Recall: 0.919
- ROC-AUC Score: 0.984

---

## ▶️ Run Locally

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 🎯 Project Goal

The objective of this project is to automatically identify spam SMS messages and reduce unwanted or potentially harmful communications.

---

## 👨‍💻 Author

**Meet Sabhaya**

Machine Learning & Data Science Enthusiast

---

⭐ If you found this project useful, consider giving it a star.
