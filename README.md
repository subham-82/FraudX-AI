# 🚗 AI-Powered Auto Insurance Fraud Detection

This is a Streamlit web app that predicts whether an auto insurance claim is **fraudulent** or **legitimate** using a trained machine learning model.

---

##  Features

- Real-time fraud prediction based on user input
- Interactive web interface built with Streamlit
- Uses a pre-trained model (`fraud_detection_model.pkl`) and scaler (`scaler.pkl`)
- User inputs include age, premium, vehicle details, and accident info
---

## 📁 Files Included

- `fraud_detection_model.pkl`: Trained RandomForest model on selected features.
- `scaler.pkl`: StandardScaler used to scale input features.
- `app.py`: Streamlit web app to interact with the fraud detection system.
- `README.md`: You're reading it!

---

## 🎯 Objective

To develop a supervised machine learning model that can predict whether an auto insurance claim is **fraudulent or legitimate** using structured data.

---

## 📊 Dataset Overview

The dataset contains auto insurance claim records with many policyholder, vehicle, and accident-related features.

### Target Variable
- `Fraud_Ind`: Whether the claim is fraudulent (`Y`) or not (`N`)

### Selected Features Used for Modeling
To simplify the model and integrate with a Streamlit app, we use 7 key features:

- `Age_Insured`: Age of the insured
- `Policy_Premium`: Premium amount paid
- `Vehicle_Cost`: Cost of the vehicle
- `Annual_Mileage`: Average annual mileage
- `Num_of_Vehicles_Involved`: Vehicles involved in the accident
- `Witnesses`: Number of witnesses
- `Bodily_Injuries`: Number of injuries reported

---

## 🧹 Preprocessing Steps

1. **Merge Datasets**: Combined 3 CSV files into a single DataFrame.
2. **Duplicate Removal**: Showed and removed duplicate rows.
3. **Missing Value Handling**:
   - Dropped rows with missing `Fraud_Ind`
   - Filled missing values in numeric features with the median
4. **Target Encoding**:
   - Converted `Fraud_Ind` into binary labels (`Y` → 1, `N` → 0)
5. **Feature Scaling**:
   - Used `StandardScaler` for normalizing numerical input

---

## 🧠 Model Used

- **RandomForestClassifier**
  - `n_estimators=100`
  - `class_weight='balanced'` to handle class imbalance
  - Trained on scaled 7-feature input

---

## 📈 Evaluation Metrics

- **Accuracy**
- **Precision, Recall, F1-score**
- **Confusion Matrix**
- **ROC AUC Score**
- **Precision-Recall vs Threshold Plot**

---

## ⚙️ How to Use

### In Notebook
1. Upload the 3 claim CSV files.
2. Run the notebook `AI_Fraud_Detection_7_Features_Corrected.ipynb`
3. The model will be saved as `fraud_detection_model.pkl` and `scaler.pkl`
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

