import streamlit as st
import pandas as pd
import joblib
import os

st.title("🎇 Diwali Sales Prediction App")

# Check files exist
if not os.path.exists("model.pkl"):
    st.error("model.pkl file not found. Please upload it to GitHub repository.")
    st.stop()

if not os.path.exists("columns.pkl"):
    st.error("columns.pkl file not found. Please upload it to GitHub repository.")
    st.stop()

# Load model
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# User Inputs
age = st.number_input("Age", 18, 70, 30)
marital_status = st.selectbox("Marital Status", [0, 1])

if st.button("Predict Sales Amount"):

    input_dict = {
        "Age": age,
        "Marital_Status": marital_status
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    prediction = model.predict(input_df)

    st.success(f"Predicted Purchase Amount: ₹ {prediction[0]:,.2f}")
