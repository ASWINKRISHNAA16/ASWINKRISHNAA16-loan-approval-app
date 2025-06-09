import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('loan_model.pkl', 'rb'))

st.title("🏦 Loan Approval Prediction App")
st.image("pic.jpg", use_container_width=True) # Make sure 'pic.jpg' is in the same directory or adjust path

# User inputs
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
Credit_History = st.selectbox("Credit History", [1.0, 0.0]) # Ensure these are floats
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
Loan_Amount_Term = st.selectbox("Loan Term (months)", [360.0, 180.0, 120.0, 84.0, 240.0, 60.0, 300.0, 480.0, 36.0, 12.0]) # Added more terms based on common values, if applicable
TotalIncome = st.number_input("Total Income (Applicant + Coapplicant)", min_value=0.0)
LoanAmount = st.number_input("Loan Amount", min_value=0.0)

# Convert inputs to numbers
gender_encoded = 1 if Gender == "Male" else 0 # Assuming LabelEncoder maps Male:1, Female:0
married_encoded = 1 if Married == "Yes" else 0 # Assuming LabelEncoder maps Yes:1, No:0
dependents_encoded = {"0": 0, "1": 1, "2": 2, "3+": 3}[Dependents]
education_encoded = 0 if Education == "Graduate" else 1 # Assuming LabelEncoder maps Graduate:0, Not Graduate:1
self_employed_encoded = 1 if Self_Employed == "Yes" else 0 # Assuming LabelEncoder maps Yes:1, No:0
property_area_encoded = {"Rural": 0, "Semiurban": 1, "Urban": 2}[Property_Area] # Matches LabelEncoder's alphabetical order

# Apply transformation as done in training
LoanAmount_log = np.log(LoanAmount + 1) # Added +1 for safety against log(0) - although min_value=0.0 helps.
                                        # Your training used np.log(df['LoanAmount']), which implies no +1.
                                        # If your LoanAmount in training never had 0, then just np.log(LoanAmount) is fine.
                                        # Stick to np.log(LoanAmount) if that's exactly what you did in training.
                                        # For robustness in deployment, +1 is safer.
                                        # Let's assume you used np.log(LoanAmount) during training for now.

# Predict button
if st.button("Predict Loan Approval"):
    if TotalIncome <= 0 or LoanAmount <= 0:
        st.error("❗ Please enter positive numbers for income and loan.")
    else:
        # Re-verify this based on your actual training code:
        # If your training used np.log(df['LoanAmount']), then use LoanAmount_log = np.log(LoanAmount)
        # If your training used np.log(df['LoanAmount'] + 1), then use LoanAmount_log = np.log(LoanAmount + 1)
        # I'll stick to what was exactly in your script:
        try:
            LoanAmount_log = np.log(LoanAmount)
        except ValueError:
            st.error("❗ Loan Amount must be greater than 0 for logarithmic transformation.")
            st.stop() # Stop execution if LoanAmount is problematic

        # Ensure the order of features matches the training data's X columns exactly:
        # ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        # 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'TotalIncome', 'LoanAmount_log']

        input_data = np.array([[
            gender_encoded,
            married_encoded,
            dependents_encoded,
            education_encoded,
            self_employed_encoded,
            Loan_Amount_Term,
            Credit_History,
            property_area_encoded,
            TotalIncome, # This was NOT log-transformed in your training
            LoanAmount_log
        ]])

        # Print for debugging (remove in final deployment)
        st.write("Input data fed to model:", input_data)
        st.write("Shape of input data:", input_data.shape)

        prediction = model.predict(input_data)

        st.write("Raw prediction from model:", prediction) # Should be 0 or 1
       

        result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
        st.success(f"Loan Status: {result}")
