import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load polynomial features
with open("poly.pkl", "rb") as f:
    poly = pickle.load(f)

st.title("Health Insurance Cost Prediction")
st.write("Enter customer details to predict insurance cost")

# ================= USER INPUTS =================
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)

sex = st.selectbox("Sex", ["Female", "Male"])
smoker = st.selectbox("Smoker", ["No", "Yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ================= MANUAL ENCODING =================
sex_male = 1 if sex == "Male" else 0
smoker_yes = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0
# (northeast is dropped to avoid dummy trap)

# ================= FEATURE ARRAY =================
features = np.array([[  
    age,
    bmi,
    children,
    sex_male,
    smoker_yes,
    region_northwest,
    region_southeast,
    region_southwest
]])

# ================= PREDICTION =================
if st.button("Predict Insurance Cost"):
    features_poly = poly.transform(features)
    prediction = model.predict(features_poly)
    st.success(f"Estimated Insurance Cost: ₹ {prediction[0]:,.2f}")