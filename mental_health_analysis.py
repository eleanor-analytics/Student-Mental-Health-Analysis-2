"""
Student Mental Health Prediction - Advanced Analysis
Author: Eleanor Bryan
Date: 2026
Purpose: End-to-end machine learning pipeline for predicting student depression
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

print("=" * 70)
print("STUDENT MENTAL HEALTH PREDICTION - ADVANCED ML PIPELINE")
print("=" * 70)

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv('student_depression.csv')

print(f"\n📊 Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# 3. DATA CLEANING
# ============================================================

# Drop missing values
df = df.dropna()

# Convert target to numeric
if df['Depressed with life'].dtype == 'object':
    df['Depressed with life'] = df['Depressed with life'].map({'Yes': 1, 'No': 0})

# Ensure all columns are numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

print(f"✅ Cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# 4. ADVANCED FEATURE ENGINEERING
# ============================================================

print("\n🔧 Advanced feature engineering...")

# Academic stress score (lower CGPA = higher stress)
df['Academic_Stress'] = (4 - df['CGPA']) / 4

# Financial stress (normalised monthly cost)
df['Financial_Stress'] = df['Per Month Cost'] / df['Per Month Cost'].max()

# Social support score (friend circle + relationship)
df['Social_Support'] = df['Good Friend Circle'] + df['Relationship Status']

# Health risk score
df['Health_Risk'] = df['Health condition'] / df['Health condition'].max()

# Semester pressure (later semesters = more pressure)
df['Semester_Pressure'] = df['Current Semester'] / df['Current Semester'].max()

# Composite risk score
df['Composite_Risk'] = (
    df['Academic_Stress'] * 0.3 +
    df['Financial_Stress'] * 0.2 +
    df['Health_Risk'] * 0.3 +
    (1 - df['Social_Support'] / 2) * 0.2
)

print(f"✅ Features after engineering: {df.shape[1]} columns")

# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n📊 Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Depression rate
depression_rate = df['Depressed with life'].mean() * 100
axes[0, 0].pie(
    [depression_rate, 100 - depression_rate],
    labels=['Depressed', 'Not Depressed'],
    autopct='%1.1f%%',
    colors=['#ff6b6b', '#51cf66']
)
axes[0, 0].set_title('Depression Rate')

# 2. CGPA distribution by depression
sns.boxplot(x='Depressed with life', y='CGPA', data=df, ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('CGPA by Depression Status')
axes[0, 1].set_xticklabels(['Not Depressed', 'Depressed'])

# 3. Health condition by depression
sns.countplot(x='Health condition', hue='Depressed with life', data=df, ax=axes[0, 2], palette='Set2')
axes[0, 2].set_title('Health Condition by Depression')

# 4. Financial stress distribution
sns.histplot(data=df, x='Financial_Stress', hue='Depressed with life', bins=20, ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Financial Stress Distribution')

# 5. Social support vs depression
sns.countplot(x='Social_Support', hue='Depressed with life', data=df, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Social Support vs Depression')

# 6. Composite risk vs depression
sns.boxplot(x='Depressed with life', y='Composite_Risk', data=df, ax=axes[1, 2], palette='Set2')
axes[1, 2].set_title('Composite Risk by Depression Status')

plt.tight_layout()
plt.savefig('reports/mental_health_eda.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X = df.drop('Depressed with life', axis=1)
y = df['Depressed with life']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\n📊 Training set: {X_train_balanced.shape[0]} samples")
print(f"📊 Testing set: {X_test.shape[0]} samples")

# ============================================================
# 7. MODEL TRAINING & COMPARISON
# ============================================================

print("\n🤖 Training multiple models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {results[name]['Accuracy']:.4f}")
    print(f"  Precision: {results[name]['Precision']:.4f}")
    print(f"  Recall:    {results[name]['Recall']:.4f}")
    print(f"  F1 Score:  {results[name]['F1 Score']:.4f}")
    print(f"  ROC AUC:   {results[name]['ROC AUC']:.4f}")

# ============================================================
# 8. HYPERPARAMETER TUNING
# ============================================================

print("\n🔧 Hyperparameter tuning for XGBoost...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_balanced, y_train_balanced)

print(f"✅ Best parameters: {grid_search.best_params_}")
print(f"✅ Best CV score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ============================================================
# 9. CROSS-VALIDATION
# ============================================================

print("\n📊 Cross-validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=cv, scoring='roc_auc')

print(f"CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# 10. FINAL MODEL EVALUATION
# ============================================================

print("\n📊 Final model evaluation...")

y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Depressed', 'Depressed']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Depressed', 'Depressed'],
            yticklabels=['Not Depressed', 'Depressed'])
plt.title('Confusion Matrix - Student Depression Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=300)
plt.show()

# ============================================================
# 11. ROC CURVES
# ============================================================

plt.figure(figsize=(10, 8))

for name, model in models.items():
    if name == 'XGBoost':
        model = best_model
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Model Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('reports/roc_curves.png', dpi=300)
plt.show()

# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Top 10 Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='viridis')
plt.title('Top 10 Features Predicting Student Depression')
plt.tight_layout()
plt.savefig('reports/feature_importance.png', dpi=300)
plt.show()

# ============================================================
# 13. SAVE MODEL
# ============================================================

joblib.dump(best_model, 'models/student_depression_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\n✅ Model saved to models/student_depression_model.pkl")
print("✅ Scaler saved to models/scaler.pkl")

print("\n" + "=" * 70)
print("✅ ADVANCED ANALYSIS COMPLETE")
print("=" * 70)