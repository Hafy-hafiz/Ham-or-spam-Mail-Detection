# Spam Mail Detection using Machine Learning

A machine learning-based web application that classifies email or SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and supervised learning algorithms.

## Project Overview

Spam messages are unwanted or harmful messages sent in bulk, often containing scams, advertisements, or malicious links. This project uses **TF-IDF vectorization**, **text preprocessing**, and **machine learning models** to automatically detect whether a message is spam or not.

The project also includes a **Streamlit web app** where users can enter a message and instantly check whether it is spam or ham.

---

## Features

- Detects whether a message is **Spam** or **Ham**
- Uses **Natural Language Processing (NLP)** for text cleaning
- Converts text into numerical format using **TF-IDF Vectorization**
- Trains multiple machine learning models:
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
  - Support Vector Classifier (SVC)
- Compares models using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Selects the best model based on **Recall**
- Interactive **Streamlit UI** for real-time predictions
- Saves trained model and vectorizer using **Joblib**

---

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **NLTK**
- **Scikit-learn**
- **Imbalanced-learn (SMOTE)**
- **Matplotlib**
- **Seaborn**
- **Streamlit**
- **Joblib**

---

## Dataset

The dataset used is the **Spam/Ham SMS Dataset**, containing labeled text messages:

- **Ham** → Legitimate message
- **Spam** → Unwanted or promotional message

Dataset columns:
- `v1` → Label (`ham` / `spam`)
- `v2` → Message text

---

## Project Workflow

1. Load and clean dataset
2. Perform text preprocessing:
   - Lowercasing
   - Removing special characters
   - Tokenization
   - Lemmatization
   - Stopword removal
3. Convert text into vectors using **TF-IDF**
4. Split data into train/test sets
5. Handle class imbalance using **SMOTE**
6. Train multiple ML models
7. Evaluate model performance
8. Select best model based on **Recall**
9. Save best model and TF-IDF vectorizer
10. Build Streamlit app for prediction

---

## Model Evaluation Metrics

The models are evaluated using:

- **Accuracy** – Overall correctness
- **Precision** – How many predicted spam messages were actually spam
- **Recall** – How many actual spam messages were correctly detected
- **F1-Score** – Balance between Precision and Recall

### Why Recall?

For spam detection, Recall is the most important metric because missing a spam message (False Negative) is more harmful than incorrectly marking a ham message as spam.

