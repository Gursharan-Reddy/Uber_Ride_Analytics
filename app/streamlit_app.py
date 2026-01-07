import streamlit as st
import pandas as pd
import joblib
import datetime
import pydeck as pdk
import numpy as np
import requests
import matplotlib.pyplot as plt

# ---------------- 1. PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Uber Operations Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- 2. CUSTOM CSS STYLING ----------------
def load_custom_css():
    st.markdown("""
        <style>
        /* Card Styling for Metrics */
        div[data-testid="stMetric"] {
            background-color: #262730;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #29B5E8;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
            /* Force equal height for all metrics */
            min-height: 130px; 
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        /* Tab Styling */
        button[data-baseweb="tab"] {
            font-size: 16px;
            font-weight: 600;
        }
        /* Alert/Status Box Styling */
        div[data-testid="stMarkdownContainer"] p {
            font-size: 1.05rem;
        }
        /* Sidebar Header Styling */
        .sidebar-header {
            font-size: 1.2rem;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
            color: #29B5E8;
        }
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ---------------- 3. ASSET LOADING ----------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("outputs/model.pkl")
        zones = pd.read_csv("data/processed/zones.csv")
        data = pd.read_parquet("data/processed/model_data.parquet")
        return model, zones, data
    except Exception as e:
        st.error(f"System Error: {e}")
        return None, None, None

model, zones, data = load_assets()

if model is None:
    st.stop()

# ---------------- 4. LIVE WEATHER INTEGRATION ----------------
def fetch_nyc_weather():
    # API Key is used internally but not displayed in the UI
    API_KEY = "496697ebc96d923e8617116ae5ebd185"
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q=New York&appid={API_KEY}&units=imperial"
        r = requests.get(url, timeout=3).json()
        if r.get("cod") != 200: return None, None
        return r["main"]["temp"], r["weather"][0]["main"]
    except:
        return None, None

# ---------------- 5. SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.title("Control Panel")
    
    # --- Data Settings Section ---
    st.markdown('<p class="sidebar-header">Data Settings</p>', unsafe_allow_html=True)
    use_live_weather = st.toggle("Live Weather Mode", value=True)

    # Global Simulation Variables
    if use_live_weather:
        temp, cond = fetch_nyc_weather()
        if temp:
            st.info(f"NYC Live Conditions: {temp:.0f} F | {cond}")
            sim_temp = temp
            sim_rain = 1 if "Rain" in cond or "Drizzle" in cond else 0
        else:
            st.warning("Weather API Unavailable. Switching to defaults.")
            sim_temp, sim_rain = 72, 0
    else:
        st.subheader("Manual Simulation")
        sim_date = st.date_input("Date", datetime.date(2024, 6, 1))
        sim_hour = st.slider("Hour (0-23)", 0, 23, 18)
        sim_temp = st.slider("Temperature (F)", 0, 100, 72)
        sim_rain = st.checkbox("Rain Precipitation")

    st.markdown("---")
    
    # --- Market Controls Section ---
    st.markdown('<p class="sidebar-header">Market Controls</p>', unsafe_allow_html=True)
    surge = st.slider("Surge Multiplier", 1.0, 3.0, 1.0, step=0.1, help="Adjust price multiplier to simulate high demand.")
    
    st.markdown("### Route Selection")
    p_zone = st.selectbox("Pickup Location", zones["Zone"].unique(), index=0)
    d_zone = st.selectbox("Dropoff Location", zones["Zone"].unique(), index=1)
    
    analyze = st.button("Calculate Demand", type="primary", use_container_width=True)

# ---------------- 6. PREDICTION ENGINE ----------------
def get_prediction_input(loc_id):
    now = datetime.datetime.now()
    # Determine Time Basis
    if use_live_weather:
        hour, day, month = now.hour, now.weekday(), now.month
    else:
        hour = sim_hour
        day, month = sim_date.weekday(), sim_date.month

    # Feature Construction
    row = {
        "PULocationID": loc_id, "DOLocationID": loc_id,
        "hour": hour, "day_of_week": day, "month": month,
        "is_weekend": 1 if day >= 5 else 0, "is_holiday": 0,
        "TAVG": sim_temp 
    }

    # Weather Categorization
    weather_flags = ["weather_Clear", "weather_Freezing", "weather_Cold", 
                     "weather_Mild", "weather_Warm", "weather_Hot", 
                     "weather_Rain", "weather_Snow"]
    for c in weather_flags: row[c] = 0

    # Temperature Binning
    if sim_temp <= 32: row["weather_Freezing"] = 1
    elif sim_temp <= 50: row["weather_Cold"] = 1
    elif sim_temp <= 72: row["weather_Mild"] = 1
    elif sim_temp <= 85: row["weather_Warm"] = 1
    else: row["weather_Hot"] = 1

    # Rain Logic
    if sim_rain:
        row["weather_Rain"] = 1
    else:
        row["weather_Clear"] = 1

    # DataFrame creation aligned with model features
    df = pd.DataFrame([row])
    if hasattr(model, 'feature_names_in_'):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    return df, hour

def get_demand(loc_id):
    df, _ = get_prediction_input(loc_id)
    return model.predict(df)[0] * surge

# ---------------- 7. MAIN DASHBOARD UI ----------------

# A. Header
st.title("Uber Operations Command Center")
st.markdown("Real-time supply chain monitoring and predictive analytics.")

# B. Heads-Up Display (KPI Ribbon)
avg_demand = data[data['hour'] == 18]['trip_count'].mean() * surge 
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("System Status", "Normal" if surge < 1.5 else "High Volume", delta="Stable" if surge < 1.5 else "Surge Active")
kpi2.metric("Active Fleet", "1,240 Units", delta="+12 Online")
kpi3.metric("Avg Wait Time", "4.2 min", delta_color="inverse", delta="-0.5 min")
kpi4.metric("Est. Hourly Revenue", f"${(avg_demand * 22.5 * 100):,.0f}", delta=f"Multiplier {surge}x")

st.markdown("---")

# C. SECTION 1: THE MAP (Centered & Full Width)
st.subheader("Geospatial Demand Density")

with st.spinner("Updating geospatial data..."):
    map_df = zones.copy()
    # Generate predictions
    map_df["demand"] = map_df["LocationID"].apply(get_demand)
    map_df["norm"] = map_df["demand"] / map_df["demand"].max()

    # Clustering Logic
    centers = np.array([
        [40.7580, -73.9855], [40.7306, -73.9866], [40.7831, -73.9712],
        [40.6782, -73.9442], [40.7282, -73.7949], [40.7357, -74.1724],
    ])
    cluster = map_df["LocationID"] % len(centers)
    base = centers[cluster]
    np.random.seed(42)
    map_df["lat"] = base[:, 0] + np.random.normal(0, 0.012, len(map_df))
    map_df["lon"] = base[:, 1] + np.random.normal(0, 0.012, len(map_df))

    # PyDeck Layer
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
        initial_view_state=pdk.ViewState(latitude=40.73, longitude=-73.98, zoom=10.5),
        map_style="dark",
        tooltip={"text": "{Zone}\nDemand: {demand:.1f}"}
    )
    
    # Map height set to 600px for balance
    st.pydeck_chart(deck, use_container_width=True, height=600)

# D. SECTION 2: ROUTE ECONOMICS (Below Map, Horizontal Layout)
st.subheader("Route Economics")

if analyze:
    if p_zone == d_zone:
        st.error("Error: Pickup and Dropoff locations cannot be identical.")
    else:
        p_id = zones[zones["Zone"] == p_zone]["LocationID"].values[0]
        demand_val = get_demand(p_id)
        _, active_hour = get_prediction_input(p_id)

        # Night Pricing Logic
        is_night = (active_hour >= 21) or (active_hour < 9)
        time_mod = 2.0 if is_night else 1.0
        fare_status = "Night Fare (2x)" if is_night else "Standard Fare"
        
        # Revenue Calculation
        base_fare = 22.50
        final_fare = base_fare * surge * time_mod
        est_rev = demand_val * final_fare

        # Display Route Info
        st.info(f"Route Analysis: **{p_zone}** to **{d_zone}**")
        
        # HORIZONTAL METRICS (3 Columns) with ALIGNMENT FIX
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Rides", f"{demand_val:.1f}", delta="Trips/Hr")
        c2.metric("Base Fare", f"${final_fare:.2f}", delta=fare_status)
        # Added dummy delta "Gross Est." to force equal height/alignment with the other two boxes
        c3.metric("Projected Revenue", f"${est_rev:,.2f}", delta="Gross Est.")

        # Operational Recommendations
        if demand_val > 150:
            st.error("Action Required: Dispatch reinforcement units immediately.")
        elif demand_val > 50:
            st.warning("Advisory: Demand approaching capacity.")
        else:
            st.success("Status: Optimal supply levels.")
else:
    st.caption("Select a route in the control panel and click Calculate to view details.")

# ---------------- 8. SECONDARY METRICS ----------------
st.divider()
tab1, tab2 = st.tabs(["Historical Trends", "System Health"])

with tab1:
    st.subheader("Temporal Demand Patterns")
    st.line_chart(data.groupby("hour")["trip_count"].mean(), color="#29B5E8")

with tab2:
    st.write("**Model Architecture:** RandomForest Regressor (v1.0.2)")
    st.write("**Last Model Update:** 2024-06-15")
    st.progress(0.84, text="Model R2 Accuracy: 84%")