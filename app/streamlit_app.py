import streamlit as st
import pandas as pd
import joblib
import datetime
import pydeck as pdk
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(page_title="Uber Demand Control Center", layout="wide")

# ---------------- Load Assets ----------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("outputs/model.pkl")
        zones = pd.read_csv("data/processed/zones.csv")
        data = pd.read_parquet("data/processed/model_data.parquet")
        return model, zones, data
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None

model, zones, data = load_assets()

if model is None:
    st.stop()

# ---------------- Sidebar ----------------
st.sidebar.header("Simulation Controls")

date = st.sidebar.date_input("Date", datetime.date(2024, 6, 1))
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
temp = st.sidebar.slider("Temperature (F)", 0, 100, 70)
surge = st.sidebar.slider("Surge Multiplier (%)", 0, 100, 0) / 100

st.sidebar.markdown("---")
p_zone = st.sidebar.selectbox("Select Pickup Zone", zones["Zone"].unique())
d_zone = st.sidebar.selectbox("Select Drop Zone", zones["Zone"].unique())
analyze = st.sidebar.button("Analyze Demand")

# ---------------- Prediction Logic ----------------
def get_prediction(loc_id):
    row = {
        "PULocationID": loc_id,
        "DOLocationID": loc_id,
        "hour": hour,
        "day_of_week": date.weekday(),
        "month": date.month,
        "is_weekend": 1 if date.weekday() >= 5 else 0,
        "is_holiday": 0,
        "is_freezing": 1 if temp <= 32 else 0,
        "is_rainy": 0,
    }

    # UPDATED: Full list of weather categories
    weather_cols = [
        "weather_Clear",
        "weather_Freezing",
        "weather_Cold",
        "weather_Mild",
        "weather_Warm",
        "weather_Hot",
        "weather_Rain",
        "weather_Snow",
    ]

    # Initialize all weather columns to 0
    for col in weather_cols:
        row[col] = 0

    # UPDATED: Temperature Categorization Logic
    if temp <= 32:
        row["weather_Freezing"] = 1
    elif temp <= 50:
        row["weather_Cold"] = 1
    elif temp <= 72:
        row["weather_Mild"] = 1
    elif temp <= 85:
        row["weather_Warm"] = 1
    else:
        row["weather_Hot"] = 1

    # Realign columns to match model training data
    inp = pd.DataFrame([row])
    # Check if model has feature_names_in_ attribute (Scikit-Learn models)
    if hasattr(model, 'feature_names_in_'):
        inp = inp.reindex(columns=model.feature_names_in_, fill_value=0)
    
    return model.predict(inp)[0] * (1 + surge)

# ---------------- Main UI ----------------
st.title("Uber Demand Control Center")

col1, col2 = st.columns([2, 1])

# ---------------- Heatmap (Unchanged) ----------------
with col1:
    st.subheader("Demand Heatmap")

    map_df = zones.copy()
    map_df["predicted_demand"] = map_df["LocationID"].apply(get_prediction)

    # Organic clustering (no grid artifacts) - Preserved as requested
    np.random.seed(42)
    map_df["lat"] = 40.7128 + np.random.normal(0, 0.03, len(map_df))
    map_df["lon"] = -74.0060 + np.random.normal(0, 0.03, len(map_df))

    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_weight="predicted_demand",
        radius_pixels=80,
        intensity=1.1,
        threshold=0.05,
    )

    deck = pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=pdk.ViewState(
            latitude=40.7128,
            longitude=-74.0060,
            zoom=10,
            pitch=40,
        )
    )

    st.pydeck_chart(deck)

# ---------------- Demand Scoring (Updated) ----------------
with col2:
    st.subheader("Demand Scoring")

    if analyze:
        # --- IMPORTANT STEP: VALIDATION CHECK ---
        if p_zone == d_zone:
            st.error("⚠️ Invalid Route: Pickup and Drop-off zones cannot be the same.")
        else:
            p_id = zones[zones["Zone"] == p_zone]["LocationID"].values[0]
            val = get_prediction(p_id)

            st.metric("Forecasted Ride Volume", f"{val:.2f} rides")

            if val > 1.5:
                st.warning("High Demand: Driver rebalancing recommended.")
            else:
                st.success("Stable Demand: Supply is sufficient.")
    else:
        st.info("Select zones and click Analyze to view demand.")

# ---------------- Trends ----------------
st.divider()
st.subheader("Historical System Trends")
st.line_chart(data.groupby("hour")["trip_count"].mean())