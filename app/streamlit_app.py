import warnings
import os
import sys

# -----------------------------------------------------------------------------
# 1. WARNING SUPPRESSION
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

import streamlit as st
import pandas as pd
import joblib
import datetime
import pydeck as pdk
import numpy as np
import sqlite3

# -----------------------------------------------------------------------------
# 2. PATH CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.external_apis import fetch_nyc_weather, get_traffic_congestion, get_nyc_events

# -----------------------------------------------------------------------------
# 3. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Uber Operations Command",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="Taxi"
)

# -----------------------------------------------------------------------------
# 4. PROFESSIONAL UI STYLING
# -----------------------------------------------------------------------------
def render_professional_ui():
    st.markdown("""
        <style>
        .stApp { background-color: #0E1117; }
        .css-card {
            background-color: #262730;
            border: 1px solid #363B47;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
            height: 100%;          
            min-height: 240px;     
            display: flex;         
            flex-direction: column;
            justify-content: space-between; 
        }
        div[data-testid="stMetric"] {
            background-color: #1F2229;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #29B5E8;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(41, 181, 232, 0.2);
        }
        div.stButton > button {
            background-color: #29B5E8;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #1C9AD6;
            box-shadow: 0 4px 12px rgba(41, 181, 232, 0.3);
        }
        section[data-testid="stSidebar"] {
            background-color: #161920;
            border-right: 1px solid #363B47;
        }
        h1, h2, h3 { color: #FFFFFF; font-family: 'Inter', sans-serif; }
        </style>
    """, unsafe_allow_html=True)

render_professional_ui()

# -----------------------------------------------------------------------------
# 5. STATE INITIALIZATION
# -----------------------------------------------------------------------------
if 'live_stats' not in st.session_state:
    st.session_state['live_stats'] = {"traffic": 0.0, "event": "No Data", "event_count": 0}
if 'calculation_done' not in st.session_state:
    st.session_state['calculation_done'] = False
if 'route_inputs' not in st.session_state:
    st.session_state['route_inputs'] = {}

# -----------------------------------------------------------------------------
# 6. ASSET LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        model_path = os.path.join(BASE_DIR, "outputs", "model.pkl")
        db_path = os.path.join(BASE_DIR, "data", "taxi_system.db")
        
        if not os.path.exists(model_path):
            return None, None, None
            
        model = joblib.load(model_path)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        zones = pd.read_sql("SELECT * FROM zones", conn)
        return model, zones, conn
    except:
        return None, None, None

model, zones, db_conn = load_assets()

if model is None:
    st.error("System Offline: Model or Database missing. Please run init_db.py first.")
    st.stop()

# -----------------------------------------------------------------------------
# 7. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Control Panel")
    
    st.markdown("### Live Feeds")
    use_live = st.toggle("Live Weather Sync", value=True)
    
    if use_live:
        temp, cond = fetch_nyc_weather()
        if temp:
            st.info(f"NYC: {temp:.0f}F | {cond}")
            sim_temp, sim_rain = temp, (1 if "Rain" in cond else 0)
        else:
            st.warning("Weather API Unreachable")
            sim_temp, sim_rain = 72, 0
        sim_hour = datetime.datetime.now().hour
        sim_date = datetime.datetime.now().date()
    else:
        sim_date = st.date_input("Simulation Date", datetime.date.today())
        sim_hour = st.slider("Simulation Hour", 0, 23, 12)
        sim_temp = st.slider("Temperature (F)", 0, 100, 72)
        sim_rain = st.checkbox("Rain Conditions")

    st.markdown("---")
    
    tomtom_key = st.secrets["api_keys"]["tomtom"]
    tm_key = st.secrets["api_keys"]["ticketmaster"]
    
    if st.button("Scan Traffic and Events", use_container_width=True):
        with st.spinner("Scanning real-time sources..."):
            event_name, event_count = get_nyc_events(tm_key)
            congestion = get_traffic_congestion(40.7128, -74.0060, tomtom_key)
            st.session_state['live_stats'] = {
                "traffic": congestion, 
                "event": event_name, 
                "event_count": event_count
            }
            st.success("Data Updated!")

    st.markdown("---")
    
    st.markdown("### Route Logistics")
    
    # --- UPDATE 1: SURGE SLIDER MODIFIED (0-100%) ---
    surge_pct = st.slider("Surge Percentage (%)", 0, 100, 0, step=5)
    
    # --- UPDATE 2: NORMALIZE TO MULTIPLIER ---
    # Convert 0-100% to a 1.0x-2.0x multiplier
    surge_multiplier = 1 + (surge_pct / 100.0)
    
    p_zone = st.selectbox("Pickup Point", zones["Zone"].unique(), index=0)
    d_zone = st.selectbox("Dropoff Point", zones["Zone"].unique(), index=1)
    
    if st.button("Calculate Demand", type="primary", use_container_width=True):
        st.session_state['calculation_done'] = True
        st.session_state['route_inputs'] = {
            "p_zone": p_zone,
            "d_zone": d_zone,
            "surge_pct": surge_pct,          # Store raw % for display
            "surge_mult": surge_multiplier   # Store multiplier for math
        }

# -----------------------------------------------------------------------------
# 8. PREDICTION ENGINE (Using Calculated Multiplier)
# -----------------------------------------------------------------------------
def predict_rides(loc_id, hour_val=None, active_surge_mult=None):
    if hour_val is None: 
        hour_val = sim_hour
    
    # Use the passed multiplier, or fall back to the global sidebar one
    if active_surge_mult is None:
        active_surge_mult = surge_multiplier
        
    day = sim_date.weekday()
    row = {
        "PULocationID": loc_id, "hour": hour_val, "day_of_week": day, 
        "month": sim_date.month, "is_weekend": 1 if day >= 5 else 0, "TAVG": sim_temp,
        "weather_Rain": 1 if sim_rain else 0, "weather_Clear": 0 if sim_rain else 1
    }
    df = pd.DataFrame([row])
    if hasattr(model, 'feature_names_in_'):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # --- UPDATE 3: USE MULTIPLIER IN PREDICTION ---
    return model.predict(df)[0] * active_surge_mult

# -----------------------------------------------------------------------------
# 9. MAIN DASHBOARD UI
# -----------------------------------------------------------------------------
st.title("Uber Operations Command Center")
st.markdown("Real-time predictive analytics and supply chain optimization.")

st.markdown("### System Status")
k1, k2, k3, k4 = st.columns(4)

try:
    avg_val = pd.read_sql(f"SELECT AVG(trip_count) FROM model_data WHERE hour = {sim_hour}", db_conn).iloc[0,0]
except:
    avg_val = 0

stats = st.session_state['live_stats']
k1.metric("Predicted Ride Vol", f"{avg_val:.1f}", delta="Normal Vol")
k2.metric("Active Fleet", "1,240", delta="+12 Online")
k3.metric("Traffic Congestion", f"{stats['traffic']*100:.0f}%", 
          delta="High" if stats['traffic'] > 0.3 else "Flowing", delta_color="inverse")
k4.metric("Live Events", f"{stats['event_count']}", delta=stats['event'][:10]+"..." if len(stats['event']) > 10 else stats['event'])

st.markdown("---")

st.markdown("### Demand Heatmap")
with st.container():
    map_df = zones.copy()
    if len(map_df) > 500: map_df = map_df.sample(500)
    
    map_df["demand"] = map_df["LocationID"].apply(lambda x: predict_rides(x))
    map_df["norm"] = map_df["demand"] / (map_df["demand"].max() + 1) 

    centers = np.array([[40.75, -73.98], [40.71, -74.00], [40.78, -73.96]])
    cluster = map_df["LocationID"] % len(centers)
    base = centers[cluster]
    map_df["lat"] = base[:, 0] + np.random.normal(0, 0.008, len(map_df))
    map_df["lon"] = base[:, 1] + np.random.normal(0, 0.008, len(map_df))

    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=40.73, longitude=-73.98, zoom=11, pitch=50),
        layers=[
            pdk.Layer("HeatmapLayer", data=map_df, get_position=["lon", "lat"], get_weight="norm", radius_pixels=60, intensity=2, threshold=0.3),
            pdk.Layer("ScatterplotLayer", data=map_df[map_df['demand'] > 100], get_position=["lon", "lat"], get_radius=100, get_fill_color=[255, 140, 0, 140], pickable=True)
        ],
        tooltip={"text": "{Zone}\nPredicted Demand: {demand:.1f}"}
    )
    st.pydeck_chart(deck, width="stretch")

if st.session_state['calculation_done']:
    st.markdown("---")
    st.markdown("### Route Economics Analysis")
    
    inputs = st.session_state['route_inputs']
    
    if inputs['p_zone'] == inputs['d_zone']:
        st.error("Pickup and Dropoff locations cannot be the same.")
    else:
        p_row = zones[zones["Zone"] == inputs['p_zone']]
        d_row = zones[zones["Zone"] == inputs['d_zone']]
        
        if not p_row.empty:
            p_id = p_row["LocationID"].values[0]
            
            # Use stored multiplier for consistent results
            active_mult = inputs['surge_mult']
            demand_val = predict_rides(p_id, active_surge_mult=active_mult)
            
            # --- UPDATE 4: REVENUE CALCULATION USES MULTIPLIER ---
            est_rev = demand_val * 22.5 * active_mult
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="css-card">
                    <div>
                        <h3 style="color:#29B5E8; margin-top:0">Route Details</h3>
                        <p style="color:white; font-size:18px; margin: 10px 0;"><b>From:</b> {inputs['p_zone']}</p>
                        <p style="color:white; font-size:18px; margin: 10px 0;"><b>To:</b> {inputs['d_zone']}</p>
                    </div>
                    <div>
                        <hr style="border-color:#363B47; margin: 15px 0;">
                        <p style="color:#A0A0A0; margin:0">Surge Applied: <span style="color:white">{inputs['surge_pct']}% ({active_mult}x)</span></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                st.markdown(f"""
                <div class="css-card">
                    <div>
                        <h3 style="color:#29B5E8; margin-top:0">Financial Projection</h3>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom: 10px;">
                        <div>
                            <p style="color:#A0A0A0; margin-bottom:5px">Zone Demand</p>
                            <span style="color:white; font-size:28px; font-weight:bold">{demand_val:.0f} <span style="font-size:16px; color:#A0A0A0">rides/hr</span></span>
                        </div>
                        <div style="text-align:right">
                            <p style="color:#A0A0A0; margin-bottom:5px">Est. Revenue</p>
                            <span style="color:#00FF99; font-size:28px; font-weight:bold">${est_rev:,.2f}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("### 24-Hour Demand Forecast")
            hours = list(range(24))
            forecasts = []
            for h in hours:
                # Use the active multiplier for the forecast chart too
                val = predict_rides(p_id, hour_val=h, active_surge_mult=active_mult)
                forecasts.append(val)
                
            chart_df = pd.DataFrame({"Hour": hours, "Predicted Demand": forecasts})
            st.line_chart(chart_df.set_index("Hour"), color="#29B5E8", width="stretch", height=300)