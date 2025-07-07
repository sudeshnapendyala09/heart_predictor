import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
# Load the trained model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Heart Disease Prediction", page_icon="💓")
st.title("💓 Heart Disease Risk Predictor")

st.sidebar.header("🧾 Patient Information")

def user_input():
    age = st.sidebar.slider("Age", 20, 80, 45)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", ["typical angina", "asymptomatic", "non-anginal", "atypical angina"])
    trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", ["True", "False"])
    restecg = st.sidebar.selectbox("Resting ECG", ["normal", "st-t wave abnormality", "lv hypertrophy"])
    thalch = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina", ["True", "False"])
    oldpeak = st.sidebar.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("ST Slope", ["upsloping", "flat", "downsloping"])
    ca = st.sidebar.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

    data = {
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'cp': ["typical angina", "asymptomatic", "non-anginal", "atypical angina"].index(cp),
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == "True" else 0,
        'restecg': ["normal", "st-t wave abnormality", "lv hypertrophy"].index(restecg),
        'thalch': thalch,
        'exang': 1 if exang == "True" else 0,
        'oldpeak': oldpeak,
        'slope': ["upsloping", "flat", "downsloping"].index(slope),
        'ca': ca,
        'thal': ["normal", "fixed defect", "reversable defect"].index(thal)
    }

    return pd.DataFrame([data])

input_df = user_input()

# Prediction
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)

st.subheader("🩺 Prediction Result")
if prediction == 1:
    st.error("⚠️ At Risk of Heart Disease.")
else:
    st.success("✅ No Heart Disease Risk.")

st.subheader("📊 Prediction Confidence")
st.write(pd.DataFrame(probability, columns=["No Disease", "Disease"], index=["Probability"]))
st.bar_chart(pd.DataFrame(probability, columns=["No Disease", "Disease"]).T)