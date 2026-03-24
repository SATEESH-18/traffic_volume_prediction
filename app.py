import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("traffic_volume_prediction.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Traffic Volume Prediction", layout="centered")

st.title("🚗 Traffic Volume Prediction App")

st.write("Enter the details below to predict traffic volume")

# ---- USER INPUTS ---- #

temp = st.number_input("Temperature (°C)", value=20.0)

rain_1h = st.number_input("Rain in last 1 hour (mm)", value=0.0)
snow_1h = st.number_input("Snow in last 1 hour (mm)", value=0.0)
clouds_all = st.slider("Cloud Coverage (%)", 0, 100, 50)

weather_main = st.selectbox("Weather Condition", 
                           ["Clear", "Clouds", "Rain", "Snow", "Mist", "Drizzle"])

hour = st.slider("Hour of Day", 0, 23, 12)
month = st.slider("Month", 1, 12, 6)
day = st.slider("Day of Month", 1, 31, 15)
dayofweek = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)

is_holiday = st.selectbox("Is Holiday?", [0, 1])

# ---- FEATURE ENGINEERING ---- #

# Cyclical encoding
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

dow_sin = np.sin(2 * np.pi * dayofweek / 7)
dow_cos = np.cos(2 * np.pi * dayofweek / 7)

day_sin = np.sin(2 * np.pi * day / 31)
day_cos = np.cos(2 * np.pi * day / 31)

# ---- CREATE INPUT DATAFRAME ---- #

input_data = pd.DataFrame({
    "temp": [temp],
    "rain_1h": [rain_1h],
    "snow_1h": [snow_1h],
    "clouds_all": [clouds_all],
    "weather_main": [weather_main],
    "is_holiday": [is_holiday],

    # Cyclical features
    "hour_sin": [hour_sin],
    "hour_cos": [hour_cos],
    "month_sin": [month_sin],
    "month_cos": [month_cos],
    "dow_sin": [dow_sin],
    "dow_cos": [dow_cos],
    "day_sin": [day_sin],
    "day_cos": [day_cos]
})

# ---- PREDICTION ---- #

if st.button("Predict Traffic Volume"):
    prediction = model.predict(input_data)[0]
    st.success(f"🚦 Predicted Traffic Volume: {int(prediction)} vehicles")