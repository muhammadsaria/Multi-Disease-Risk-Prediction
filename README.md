# Multi-Disease-Risk-Prediction
A machine learning project to predict the risk of Diabetes and Heart Disease using health data.

## 👥 Team Members
- Muhammad Saria
- Muhammad Tahir

## 🚀 Project Overview
We developed a predictive system that analyzes patient health data to assess the risk of **Diabetes** and **Heart Disease**. This project followed an end-to-end data science pipeline including:

- ✅ Data Cleaning and Preprocessing
- 📊 Exploratory Data Analysis (EDA)
- 🧠 Feature Engineering
- 🔁 Model Training and Comparison
- 🧮 Model Explainability using SHAP
- 🌐 Deployment using Streamlit

---

## 📁 Datasets Used
- `diabetes_prediction_dataset.csv`
- `HeartDisease_Uncleaned_WithNoise.csv`

---

## 🔍 Exploratory Data Analysis
We visualized feature distributions, checked for imbalances and outliers, and explored feature-target relationships using:
- Correlation heatmaps
- Histograms and boxplots
- Countplots for categorical data

---

## 🧹 Data Preprocessing & Feature Engineering
- Handled missing and noisy values
- Encoded categorical variables
- Normalized/standardized features where necessary
- Created new features based on domain knowledge

---

## 🧪 Machine Learning Models Compared
We implemented and compared the performance of the following models for both datasets:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest
- Gradient Boosting
- XGBoost

📈 **Best Results:**
- **Heart Disease:** 97% Accuracy using **Random Forest**
- **Diabetes:** 91% Accuracy using **Gradient Boosting**

---

## 🤖 Model Explainability using SHAP
We used SHAP (SHapley Additive exPlanations) to interpret model decisions and understand feature importance for transparency and trustworthiness in predictions.

---

## 🌐 Streamlit Web App
We deployed the final models in a user-friendly **Streamlit app** allowing real-time predictions for heart and diabetes risk.

- App File: `multi_disease_prediction_app.py`



