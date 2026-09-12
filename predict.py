"""
Student Depression Prediction - Use the Trained Model
Author: Eleanor Bryan
Date: 2026
Purpose: Load the saved model and predict depression risk for new students
"""

import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('student_depression_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_depression(student_data):
    """
    Predict depression risk for a student.
    
    Parameters:
    student_data (dict): Student features
    
    Returns:
    tuple: (prediction, probability)
    """
    # Create DataFrame
    df = pd.DataFrame([student_data])
    
    # Add engineered features
    df['Academic_Stress'] = (4 - df['CGPA']) / 4
    df['Financial_Stress'] = df['Per Month Cost'] / 20000
    df['Social_Support'] = df['Good Friend Circle'] + df['Relationship Status']
    df['Health_Risk'] = df['Health condition'] / 3
    df['Semester_Pressure'] = df['Current Semester'] / 12
    df['Composite_Risk'] = (
        df['Academic_Stress'] * 0.3 +
        df['Financial_Stress'] * 0.2 +
        df['Health_Risk'] * 0.3 +
        (1 - df['Social_Support'] / 2) * 0.2
    )
    
    # Scale
    df_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    
    return prediction, probability

# Example: Predict for a new student
if __name__ == "__main__":
    new_student = {
        'Gender': 1,
        'Age': 20,
        'Family Background': 1,
        'Relationship Status': 0,
        'Marital Status': 0,
        'Living Style': 1,
        'Current Semester': 3,
        'CGPA': 3.8,
        'Good Friend Circle': 1,
        'Health condition': 1,
        'Per Month Cost': 12000
    }
    
    prediction, probability = predict_depression(new_student)
    
    print(f"Student: {new_student}")
    print(f"Prediction: {'Depressed' if prediction == 1 else 'Not Depressed'}")
    print(f"Risk Score: {probability * 100:.1f}%")