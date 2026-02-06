
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm

# Load saved model and preprocessor
model = joblib.load('lgbm_tuned_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes outcome.")

# Collect user input
age = st.number_input("Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
glucose = st.number_input("Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bp = st.number_input("Blood Pressure", min_value=50, max_value=200, value=80)
insulin = st.number_input("Insulin", min_value=0.0, max_value=500.0, value=80.0)
physical_activity = st.number_input("Physical Activity (min/week)", min_value=0, max_value=300, value=60)
gender = st.selectbox("Gender", ["Male", "Female"])
family_history = st.selectbox("Family History of Diabetes", ["Yes", "No"])
physical_state = st.selectbox("Physical State", ["Insufficiently Active", "Highly Active"])
glucose_status = st.selectbox("Glucose Status", ["Normal", "Prediabetic", "Diabetic"])
insulin_category = st.selectbox("Insulin Category", ["Optimal", "Normal", "Early-Resistance", "High Resistance"])
glucose_group = st.selectbox("Glucose Group", ["Group1", "Group2", "Group3"])  # example

# Combine into DataFrame
input_data = pd.DataFrame({
    'Age':[age],
    'BMI':[bmi],
    'Glucose_Level':[glucose],
    'Blood_Pressure':[bp],
    'Insulin':[insulin],
    'Physical_Activity':[physical_activity],
    'Gender':[gender],
    'Family_History':[family_history],
    'Physical_State':[physical_state],
    'Glucose_Status':[glucose_status],
    'Insulin_category':[insulin_category],
    'glucose_group':[glucose_group]
})

# Preprocess input
input_processed = preprocessor.transform(input_data)

# Make prediction
prediction = model.predict(input_processed)[0]
prediction_proba = model.predict_proba(input_processed)[0][1]

st.subheader("Prediction")
st.write("Diabetes Outcome:", "Positive" if prediction == 1 else "Negative")
st.write(f"Probability of Diabetes: {prediction_proba:.2f}")
