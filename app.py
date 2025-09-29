import streamlit as st
import pandas as pd
import joblib

# Load trained model & scaler
model = joblib.load("models/random_forest_weather.pkl")
scaler = joblib.load("models/weather_scaler.pkl")

st.set_page_config(page_title="Seattle Weather Predictor", page_icon="🌦️", layout="centered")

st.title("🌦️ Seattle Weather Prediction App")
st.write("Enter weather conditions to predict the type of weather (sun, rain, snow, fog, etc.)")

# User Input Form
with st.form("weather_form"):
    precipitation = st.number_input("🌧️ Precipitation (inches)", min_value=0.0, step=0.1)
    temp_max = st.number_input("🌡️ Max Temperature (°C)", step=0.1)
    temp_min = st.number_input("❄️ Min Temperature (°C)", step=0.1)
    wind = st.number_input("💨 Wind Speed (m/s)", min_value=0.0, step=0.1)

    submitted = st.form_submit_button("Predict Weather")

# Prediction
if submitted:
    # Get current date for additional features
    current_date = pd.Timestamp.now()
    month = current_date.month
    day_of_week = current_date.dayofweek

    # Put inputs in a dataframe with all required features in the correct order
    input_data = pd.DataFrame([[temp_max, temp_min, precipitation, wind]],
                              columns=['temp_max', 'temp_min', 'precipitation', 'wind'
                                     ])

    # Scale the inputs
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"🌤️ The predicted weather is: **{prediction.upper()}**")