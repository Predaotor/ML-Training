MAGIC Gamma Telescope ML Project
🌌 Project Overview

This project explores the MAGIC Gamma Telescope dataset, a real-world dataset from high-energy astrophysics.
The goal is to classify gamma-ray events vs. background hadron events using machine learning techniques.

This project demonstrates:

Data preprocessing & feature engineering

Training and evaluating multiple ML models

Comparing performance using key metrics

Visualizing results (confusion matrix, ROC/PR curves)

Interpreting results in the context of astrophysics

📂 Dataset

Name: MAGIC Gamma Telescope Dataset

Source: UCI Machine Learning Repository (link
)

Size: 19,020 instances, 10 features

Task: Binary classification (gamma vs. hadron events)

Features include:

fLength, fWidth, fSize, fConc, etc. (measured properties of telescope events).

Target variable:

class → gamma (signal) or hadron (background noise).

⚙️ Methods & Workflow

Data Preprocessing

Load & clean data

Handle missing values (if any)

Scale/normalize features

Exploratory Data Analysis (EDA)

Distribution plots of features

Correlation heatmaps

Model Training

Logistic Regression

Random Forest

XGBoost

(Optional) Neural Network

Evaluation

Accuracy, Precision, Recall, F1

Confusion Matrix

ROC & Precision-Recall curves

📊 Results
Model	Accuracy	Precision	Recall	F1-score# ML-Training
