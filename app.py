
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("adaboost_colorectal_model.pkl")

# Title
st.title("Colorectal Cancer 5-Year Survival Predictor")
st.write("This app predicts the 5-year survival chance of a patient with colorectal cancer using an AdaBoost model.")

def user_input():
    family_history = st.selectbox("Family History", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    mortality = st.selectbox("Mortality", ["Alive", "Deceased"])
    insurance = st.selectbox("Insurance Status", ["Insured", "Uninsured"])
    urban_rural = st.selectbox("Urban or Rural", ["Urban", "Rural"])
    country = st.selectbox("Country", ["Country A", "Country B", "Country C"])  # Adjust as needed
    incidence_rate = st.slider("Incidence Rate per 100K", 0, 100, 50)
    mortality_rate = st.slider("Mortality Rate per 100K", 0, 100, 50)
    healthcare_access = st.selectbox("Healthcare Access", ["Low", "Moderate", "High"])
    obesity_bmi = st.slider("Obesity BMI", 15.0, 45.0, 25.0)
    screening = st.selectbox("Screening History", ["No", "Yes"])
    treatment = st.selectbox("Treatment Type", ["Surgery", "Chemotherapy", "Radiation"])
    tumor_size = st.slider("Tumor Size (mm)", 0, 100, 30)
    tumor_category = st.selectbox("Tumor Size Category", ["Small", "Medium", "Large"])

    data = {
        'Family_History': 1 if family_history == "Yes" else 0,
        'Alcohol_Consumption': 1 if alcohol == "Yes" else 0,
        'Mortality': 1 if mortality == "Deceased" else 0,
        'Insurance_Status': 1 if insurance == "Insured" else 0,
        'Urban_or_Rural': 1 if urban_rural == "Urban" else 0,
        'Country': ["Country A", "Country B", "Country C"].index(country),
        'Incidence_Rate_per_100K': incidence_rate,
        'Mortality_Rate_per_100K': mortality_rate,
        'Healthcare_Access': ["Low", "Moderate", "High"].index(healthcare_access),
        'Obesity_BMI': obesity_bmi,
        'Screening_History': 1 if screening == "Yes" else 0,
        'Treatment_Type': ["Surgery", "Chemotherapy", "Radiation"].index(treatment),
        'Tumor_Size_mm': tumor_size,
        'Tumor_Size_Category': ["Small", "Medium", "Large"].index(tumor_category)
    }

    return pd.DataFrame([data])

input_df = user_input()

if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Predicted to survive 5 years. Probability: {prob:.2%}")
    else:
        st.error(f"Not predicted to survive 5 years. Probability: {prob:.2%}")

st.subheader("Input Summary")
st.write(input_df)

st.markdown("---")
st.caption("Developed using AdaBoost with Decision Tree on colorectal cancer dataset.")
