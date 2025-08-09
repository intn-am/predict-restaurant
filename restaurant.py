import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('best_random_forest_model.pkl')

st.title("Restaurant Profitability Prediction")

# Input user
location = st.selectbox("Location Type", ["Urban", "Suburban", "Rural"])
avg_meal_price = st.number_input("Average Meal Price", min_value=0.0, step=0.1)
monthly_customers = st.number_input("Monthly Customers", min_value=0)

# Prediksi
if st.button("Predict Profit"):
    # Sesuaikan format input dengan model
    input_df = pd.DataFrame({
        'location': [location],
        'avg_meal_price': [avg_meal_price],
        'monthly_customers': [monthly_customers]
    })

    pred = model.predict(input_df)[0]
    st.success(f"Estimated Monthly Profit: ${pred:,.2f}")
