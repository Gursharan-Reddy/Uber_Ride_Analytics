import streamlit as st
import pandas as pd
import joblib
import datetime
import pydeck as pdk

st.set_page_config(page_title="Uber Demand Center", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('outputs/model.pkl')
    zones = pd.read_csv('data/processed/zones.csv')
    data = pd.read_parquet('data/processed/model_data.parquet')
    return model, zones, data

model, zones, data = load_assets()

# --- Sidebar ---
st.sidebar.header("Simulation Controls")
date = st.sidebar.date_input("Date", datetime.date(2024, 6, 1))
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
temp = st.sidebar.slider("Temperature (F)", 0, 100, 70)
surge = st.sidebar.slider("Surge Multiplier (%)", 0, 100, 0) / 100.0

st.sidebar.markdown("---")
p_zone = st.sidebar.selectbox("Select Pickup Point", zones['Zone'].unique())
d_zone = st.sidebar.selectbox("Select Destination", zones['Zone'].unique())
analyze = st.sidebar.button("Analyze Demand")

# --- Logic ---
def get_prediction(loc_id):
    inp = pd.DataFrame([{
        'PULocationID': loc_id, 'hour': hour,
        'day_of_week': date.weekday(), 'month': date.month,
        'is_weekend': 1 if date.weekday() >= 5 else 0,
        'is_holiday': 0, 'TAVG': temp, 'TMAX': temp+5, 'TMIN': temp-5
    }])
    return model.predict(inp)[0] * (1 + surge)

# --- Main UI ---
st.title("Uber Demand Control Center")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Demand Heatmap")
    # Generate mock coordinates for mapping (NYC Centroids)
    map_df = zones.copy()
    map_df['predicted_demand'] = map_df['LocationID'].apply(get_prediction)
    map_df['lat'] = 40.7128 + (map_df['LocationID'] % 20) * 0.01 # Mock Lat
    map_df['lon'] = -74.0060 + (map_df['LocationID'] // 20) * 0.01 # Mock Lon
    
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("HeatmapLayer", map_df, get_position=['lon', 'lat'], get_weight="predicted_demand", radius_pixels=60)],
        initial_view_state=pdk.ViewState(latitude=40.7128, longitude=-74.0060, zoom=10)
    ))

with col2:
    st.subheader("Demand Scoring")
    if analyze:
        p_id = zones[zones['Zone'] == p_zone]['LocationID'].values[0]
        val = get_prediction(p_id)
        
        # We show 2 decimal places so you can see the model "reacting"
        st.metric("Forecasted Ride Volume", f"{val:.2f} rides")
        
        if val > 1.5:
            st.warning("High Demand: Suggesting driver rebalancing.")
        else:
            st.success("Stable Demand: Supply is sufficient.")
    else:
        st.info("Select locations and click 'Analyze' to begin.")

st.divider()
st.subheader("Historical System Trends")
st.line_chart(data.groupby('hour')['trip_count'].mean())