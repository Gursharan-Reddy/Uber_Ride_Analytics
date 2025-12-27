import streamlit as st
import pandas as pd
import joblib
import datetime
import pydeck as pdk
import numpy as np
import requests

# ---------------- Page Config ----------------
st.set_page_config(page_title="Uber Demand Control Center", layout="wide")

# ---------------- Load Assets ----------------
@st.cache_resource
def load_assets():
    model = joblib.load("outputs/model.pkl")
    zones = pd.read_csv("data/processed/zones.csv")
    data = pd.read_parquet("data/processed/model_data.parquet")
    return model, zones, data

model, zones, data = load_assets()

# ---------------- Weather ----------------
def fetch_nyc_weather(api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q=New York&appid={api_key}&units=imperial"
        r = requests.get(url).json()
        return r["main"]["temp"], r["weather"][0]["main"]
    except:
        return None, None

# ---------------- Sidebar ----------------
st.sidebar.header("Operational Controls")

use_live_weather = st.sidebar.toggle("📡 Live Weather", value=True)
API_KEY = "496697ebc96d923e8617116ae5ebd185"

if use_live_weather:
    temp, cond = fetch_nyc_weather(API_KEY)
    if temp:
        st.sidebar.success(f"🌡️ {temp:.1f}°F | {cond}")
        sim_temp = temp
        sim_rain = 1 if "Rain" in cond else 0
    else:
        sim_temp, sim_rain = 72, 0
else:
    sim_date = st.sidebar.date_input("Date", datetime.date(2024, 6, 1))
    sim_hour = st.sidebar.slider("Hour", 0, 23, 18)
    sim_temp = st.sidebar.slider("Temperature (F)", 0, 100, 72)
    sim_rain = st.sidebar.checkbox("Rainy")

surge = st.sidebar.slider("Surge Multiplier", 1.0, 3.0, 1.0)

st.sidebar.markdown("---")
p_zone = st.sidebar.selectbox("Pickup Zone", zones["Zone"].unique())
d_zone = st.sidebar.selectbox("Drop Zone", zones["Zone"].unique())
analyze = st.sidebar.button("Analyze Route")

# ---------------- Prediction Logic ----------------
def get_prediction_input(loc_id):
    now = datetime.datetime.now()
    hour = now.hour if use_live_weather else sim_hour
    day = now.weekday() if use_live_weather else sim_date.weekday()
    month = now.month if use_live_weather else sim_date.month

    row = {
        "PULocationID": loc_id,
        "DOLocationID": loc_id,
        "hour": hour,
        "day_of_week": day,
        "month": month,
        "is_weekend": 1 if day >= 5 else 0,
        "is_holiday": 0,
        "TAVG": sim_temp, # Pass raw temp too just in case
    }

    # Initialize all weather flags to 0
    weather_cols = [
        "weather_Clear", "weather_Freezing", "weather_Cold",
        "weather_Mild", "weather_Warm", "weather_Hot",
        "weather_Rain", "weather_Snow"
    ]
    for c in weather_cols:
        row[c] = 0

    # Categorize Temperature (Must match Feature Engineering logic)
    if sim_temp <= 32: row["weather_Freezing"] = 1
    elif sim_temp <= 50: row["weather_Cold"] = 1
    elif sim_temp <= 72: row["weather_Mild"] = 1
    elif sim_temp <= 85: row["weather_Warm"] = 1
    else: row["weather_Hot"] = 1

    # Set Precipitation
    if sim_rain:
        row["weather_Rain"] = 1
        row["weather_Clear"] = 0
    else:
        row["weather_Clear"] = 1 # Assume clear if not rainy

    # Create DF and align with model columns
    df = pd.DataFrame([row])
    
    if hasattr(model, 'feature_names_in_'):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    return df, hour

def get_demand(loc_id):
    df, _ = get_prediction_input(loc_id)
    return model.predict(df)[0] * surge

# ---------------- Main UI ----------------
st.title("Uber Demand Control Center")

col_map, col_metrics = st.columns([2, 1])

# ---------------- MAP (UNCHANGED) ----------------
with col_map:
    st.subheader("Demand Heatmap")

    map_df = zones.copy()
    map_df["demand"] = map_df["LocationID"].apply(get_demand)
    map_df["norm"] = map_df["demand"] / map_df["demand"].max()

    centers = np.array([
        [40.7580, -73.9855],
        [40.7306, -73.9866],
        [40.7831, -73.9712],
        [40.6782, -73.9442],
        [40.7282, -73.7949],
        [40.7357, -74.1724],
    ])

    cluster = map_df["LocationID"] % len(centers)
    base = centers[cluster]

    np.random.seed(42)
    map_df["lat"] = base[:, 0] + np.random.normal(0, 0.012, len(map_df))
    map_df["lon"] = base[:, 1] + np.random.normal(0, 0.012, len(map_df))

    heatmap = pdk.Layer(
        "HeatmapLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_weight="norm",
        radius_pixels=45,
        intensity=4.0,
        threshold=0.25,
        opacity=0.85,
    )

    deck = pdk.Deck(
        layers=[heatmap],
        initial_view_state=pdk.ViewState(
            latitude=40.73,
            longitude=-73.98,
            zoom=10.5,
        ),
        map_style="dark",
        tooltip={"text": "{Zone}\nDemand: {demand:.1f}"}
    )

    st.pydeck_chart(deck)

# ---------------- METRICS ----------------
with col_metrics:
    st.subheader("Demand Metrics")

    if analyze:
        if p_zone == d_zone:
            st.error("Pickup and Drop-off cannot be the same.")
        else:
            p_id = zones[zones["Zone"] == p_zone]["LocationID"].values[0]
            demand = get_demand(p_id)
            _, hr = get_prediction_input(p_id)

            st.metric("Forecasted Ride Volume", f"{demand:.2f} rides")

            if demand > 150:
                st.error("🔴 High Demand – Deploy more drivers")
            elif demand > 50:
                st.warning("🟠 Moderate Demand – Monitor closely")
            else:
                st.success("🟢 Stable Demand – Supply sufficient")
    else:
        st.info("Select zones and click Analyze to view demand metrics.")

# ---------------- HISTORICAL TRENDS ----------------
st.divider()
st.subheader("Historical System Trends")
st.line_chart(data.groupby("hour")["trip_count"].mean())