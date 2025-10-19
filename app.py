import streamlit as st
import numpy as np
import pickle

# Load saved model
model = pickle.load(open('logistic_model.pkl', 'rb'))

st.title("Diabetes Prediction App (Logistic Regression)")

st.write("This app uses a trained Logistic Regression model to predict whether a person is likely to have Diabetes.")

# Input fields for user
Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
Glucose = st.number_input("Glucose", 0, 200, 100)
BloodPressure = st.number_input("Blood Pressure", 0, 150, 70)
SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
Insulin = st.number_input("Insulin", 0, 900, 80)
BMI = st.number_input("BMI (Body Mass Index)", 0.0, 70.0, 25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
Age = st.number_input("Age", 18, 90, 30)

# Predict button
if st.button("Predict Diabetes Status"):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                            BMI, DiabetesPedigreeFunction, Age]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("The person is likely to have Diabetes.")
    else:
        st.success("The person is unlikely to have Diabetes.")
