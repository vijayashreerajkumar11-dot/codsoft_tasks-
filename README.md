# 🚀 CodSoft ML Internship Projects

## 👨‍💻 Author
AI/ML Intern – CodSoft  
Student Name: Vijayashree Rajkumar

## 📌 Overview
This repository contains multiple Machine Learning projects developed during the CodSoft AI/ML Internship.  
Each project focuses on real-world applications of Natural Language Processing (NLP) and Machine Learning.
-----

# 🎬 Task 1: Movie Genre Classification

## 🚀 Project Description
This project predicts the genre of a movie based on its plot description using NLP and Machine Learning.

## 🧠 Problem Statement
Classify movie descriptions into genres like:
- Action
- Comedy
- Horror
- Romance
- Thriller
- Sci-fi
- Fantasy

## ⚙️ Tech Stack
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- LinearSVC (SVM)

## 🧪 Approach
- Text preprocessing using TF-IDF
- Feature extraction using n-grams (1,2)
- Model training using LinearSVC

## 📈 Output Example
Input: A hero saves the world from aliens
Prediction: Action

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 📊 Task 3: Customer Churn Prediction

## 🚀 Project Description
This project predicts whether a bank customer will stay or leave the bank (churn prediction).

## 🧠 Problem Statement
Identify customers likely to leave the bank so preventive actions can be taken.

## ⚙️ Tech Stack
- Python
- Pandas
- Scikit-learn
- Random Forest Classifier

## 🧪 Approach
- Data preprocessing and encoding
- Train-test split (80-20)
- Model training using Random Forest

## 📈 Model Performance
- Accuracy: ~86%
- Good performance for non-churn prediction

## Example Output
----------------------------------------
     ENTER CUSTOMER DETAILS
----------------------------------------
Credit Score >>> 250

Geography (0-France,1-Germany,2-Spain) >>> 1

Gender (0-Female,1-Male) >>> 1

Age >>> 22

Tenure >>> 3

Balance >>> 600000000

Number of Products >>> 8

Has Credit Card (0/1) >>> 1

Is Active Member (0/1) >>> 1

Estimated Salary >>> 455555799

----------------------------------------
     ENTERED DETAILS
----------------------------------------
Credit Score: 250

Geography: 1

Gender: 1

Age: 15

Tenure: 2500

Balance: 600000000.0

Products: 8

Credit Card: 1

Active Member: 1

Salary: 455555799.0

========================================
Prediction >>> Customer WILL CHURN
========================================
  
----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 📩 Task 4: Spam Message Detection

## 🚀 Project Description
This project classifies SMS messages as SPAM or LEGITIMATE using NLP.

## 🧠 Problem Statement
Detect spam messages automatically using machine learning models.

## ⚙️ Tech Stack
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Naive Bayes
- Logistic Regression
- LinearSVC (Best Model)

## 🧪 Approach
- Text cleaning and preprocessing
- TF-IDF feature extraction
- Training multiple ML models
- Selecting best-performing model (SVM)

## 📈 Model Performance
- SVM performed best among all models

## 📊 Output Example
Input: Congratulations! You won a prize

Prediction: SPAM
