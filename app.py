import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🚗 AI-Powered Auto Insurance Fraud Detection")

st.write("Enter details below to predict if the claim is likely to be **FRAUDULENT** or **LEGITIMATE**.")

# Example inputs (these should match the features used during training)
age = st.number_input("Age of Insured", min_value=16, max_value=100, value=35)
policy_premium = st.number_input("Policy Premium", value=500.0)
vehicle_cost = st.number_input("Vehicle Cost", value=25000.0)
annual_mileage = st.number_input("Annual Mileage", value=12000.0)
num_vehicles = st.number_input("Number of Vehicles Involved", min_value=1, value=1)
witnesses = st.number_input("Number of Witnesses", min_value=0, value=0)
injuries = st.number_input("Number of Bodily Injuries", min_value=0, value=0)

# Collect features as array
features = [age, policy_premium, vehicle_cost, annual_mileage, num_vehicles, witnesses, injuries]
scaled_input = scaler.transform([features])

# Predict
if st.button("Predict Fraud"):
    prob = model.predict_proba(scaled_input)[0][1]
    pred = "FRAUD" if prob >= 0.35 else "LEGITIMATE"
    st.markdown(f"### Prediction: **{pred}**")
    st.write(f"Fraud Probability: **{prob:.2f}**")
