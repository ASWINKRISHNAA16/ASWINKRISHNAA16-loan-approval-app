import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('loan_model.pkl', 'rb'))

st.title("🏦 Loan Approval Prediction App")
st.image("pic.jpg", use_container_width=True)

# User inputs
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
Loan_Amount_Term = st.selectbox("Loan Term (months)", [360.0, 180.0, 120.0, 84.0, 240.0])
TotalIncome = st.number_input("Total Income", min_value=0.0)
LoanAmount = st.number_input("Loan Amount", min_value=0.0)

# Convert inputs to numbers
gender = 1 if Gender == "Male" else 0
married = 1 if Married == "Yes" else 0
dependents = {"0": 0, "1": 1, "2": 2, "3+": 3}[Dependents]
education = 0 if Education == "Graduate" else 1
self_employed = 1 if Self_Employed == "Yes" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[Property_Area]

# Predict button
if st.button("Predict Loan Approval"):
    if TotalIncome <= 0 or LoanAmount <= 0:
        st.error("❗ Please enter positive numbers for income and loan.")
    else:
        LoanAmount_log = np.log(LoanAmount + 1)
        input_data = np.array([[gender, married, dependents, education, self_employed,
                                LoanAmount_log, Loan_Amount_Term, Credit_History,
                                property_area, TotalIncome]])
        prediction = model.predict(input_data)
        result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
        st.success(f"Loan Status: {result}")
