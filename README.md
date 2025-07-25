# 🚗 AI-Powered Auto Insurance Fraud Detection

This is a Streamlit web app that predicts whether an auto insurance claim is **fraudulent** or **legitimate** using a trained machine learning model.

---

##  Features

- Real-time fraud prediction based on user input
- Interactive web interface built with Streamlit
- Uses a pre-trained model (`fraud_detection_model.pkl`) and scaler (`scaler.pkl`)
- User inputs include age, premium, vehicle details, and accident info

---

##  Technologies Used

- **Python**
- **Streamlit**
- **Joblib**
- **NumPy**

---

---

## ✅ Features Used for Prediction

The model expects the following features:
- Age of Insured
- Policy Premium
- Vehicle Cost
- Annual Mileage
- Number of Vehicles Involved
- Number of Witnesses
- Number of Bodily Injuries

---

## ⚙️ Requirements

You must have Python 3.7+ installed.

Install required Python libraries using:

```bash
pip install streamlit scikit-learn joblib numpy

