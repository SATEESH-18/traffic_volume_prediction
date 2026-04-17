import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
import os

# =========================
# DOWNLOAD + LOAD MODEL
# =========================

MODEL_URL = "https://drive.google.com/uc?export=download&id=1gdhKW9hSPb3EpK_cuMdhlZg-TGHYLegt"  # 🔴 Replace with your ID
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================
# UI
# =========================

st.title("🚦 Traffic Volume Prediction")
st.markdown("### Predict highway traffic using Machine Learning")
st.markdown("---")

st.write("Provide current conditions and previous traffic data")

# =========================
# USER INPUTS
# =========================

st.subheader("🌦 Weather Conditions")

temp = st.slider("Temperature (Kelvin)", 250.0, 320.0, 290.0)
rain_1h = st.slider("Rain (last 1 hour)", 0.0, 10.0, 0.0)
clouds_all = st.slider("Cloud Coverage (%)", 0, 100, 50)

weather_main = st.selectbox(
    "Weather Condition",
    ["Clouds", "Clear", "Rain", "Snow", "Mist", "Drizzle", "Haze", "Fog"]
)

st.subheader("⏰ Time Information")

hour = st.slider("Hour", 0, 23, 12)
month = st.slider("Month", 1, 12, 6)
dayofweek = st.slider("Day of Week (0=Monday)", 0, 6, 3)

# =========================
# LAG INPUTS
# =========================

st.subheader("📊 Previous Traffic Data")

lag_1 = st.number_input("Traffic 1 hour ago", 0, 10000, 3000)
lag_2 = st.number_input("Traffic 2 hours ago", 0, 10000, 3200)
lag_24 = st.number_input("Traffic 24 hours ago", 0, 10000, 4000)

# =========================
# FEATURE ENGINEERING
# =========================

rolling_mean_3 = np.mean([lag_1, lag_2, lag_24])
rolling_mean_6 = np.mean([lag_1, lag_2, lag_24])
rolling_std_3 = np.std([lag_1, lag_2, lag_24])

hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)

month_sin = np.sin(2*np.pi*month/12)
month_cos = np.cos(2*np.pi*month/12)

dow_sin = np.sin(2*np.pi*dayofweek/7)
dow_cos = np.cos(2*np.pi*dayofweek/7)

is_rush_hour = int(hour in [7, 8, 9, 16, 17, 18])

# =========================
# CREATE INPUT DATAFRAME
# =========================

input_df = pd.DataFrame({
    "temp": [temp],
    "rain_1h": [rain_1h],
    "clouds_all": [clouds_all],

    "lag_1": [lag_1],
    "lag_2": [lag_2],
    "lag_24": [lag_24],

    "rolling_mean_3": [rolling_mean_3],
    "rolling_mean_6": [rolling_mean_6],
    "rolling_std_3": [rolling_std_3],

    "weather_main": [weather_main],

    "hour_sin": [hour_sin],
    "hour_cos": [hour_cos],
    "month_sin": [month_sin],
    "month_cos": [month_cos],
    "dow_sin": [dow_sin],
    "dow_cos": [dow_cos],

    "is_rush_hour": [is_rush_hour]
})

st.write("### 🔍 Input Summary")
st.dataframe(input_df)

# =========================
# PREDICTION
# =========================

if st.button("🚀 Predict Traffic Volume"):
    prediction = model.predict(input_df)[0]

    st.success(f"🚗 Predicted Traffic Volume: {int(prediction)}")

    if prediction > 5000:
        st.warning("⚠️ High traffic expected")
    elif prediction > 3000:
        st.info("🚦 Moderate traffic")
    else:
        st.success("🟢 Smooth traffic")