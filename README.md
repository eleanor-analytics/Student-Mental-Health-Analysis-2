# Student Mental Health Prediction

## Overview

I built this project to explore a question that matters to me: what actually predicts depression in university students? Using a dataset of over 3,000 students, I cleaned and analysed the data, created new features from the raw information, and trained several machine learning models to see which factors mattered most. The results were clear — health, academic pressure, and financial stress all play a significant role, while social support acts as a protective factor. I built this end-to-end pipeline to practise the full data science workflow, from messy data to actionable insights.

---

## Dataset

- **Source:** Mendeley Data
- **Rows:** 3,351 students
- **Features:** 12 (demographics, CGPA, health, lifestyle)
- **Target:** Depression (Yes/No)

---


---

## Methodology

1. **Data Cleaning** – Handled missing values, converted data types
2. **Feature Engineering** – Created 6 new features (Academic Stress, Financial Stress, Social Support, Health Risk, Semester Pressure, Composite Risk)
3. **EDA** – Visualised relationships between features and depression
4. **Class Imbalance** – Applied SMOTE to balance training data
5. **Modelling** – Trained 4 models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
6. **Hyperparameter Tuning** – GridSearchCV for XGBoost
7. **Cross-Validation** – 5-fold StratifiedKFold
8. **Evaluation** – Accuracy, Precision, Recall, F1, ROC AUC

---

## Results

| Model | Accuracy | ROC AUC |
| :--- | :--- | :--- |
| Logistic Regression | 0.78 | 0.82 |
| Random Forest | 0.83 | 0.87 |
| Gradient Boosting | 0.84 | 0.88 |
| **XGBoost (tuned)** | **0.85** | **0.89** |

---

## Key Findings

- **Health condition** is the strongest predictor of depression
- **CGPA** and **financial stress** also significantly impact mental health
- **Social support** acts as a protective factor
- **Composite risk score** (combining all factors) achieves best prediction

---

# Skills demonstrated 

Python (pandas, scikit-learn, XGBoost)
Feature engineering
Machine learning (classification)
Hyperparameter tuning
Cross-validation
Data visualisation
Model interpretation
Git/GitHub workflow


## Connect

GitHub: github.com/eleanor-analytics
LinkedIn: linkedin.com/in/eleanor-bryan-35b922255



