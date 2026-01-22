import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import datetime
import sys
import os
import concurrent.futures
import warnings
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------
# 0. CONFIG & WARNING SUPPRESSION
# ---------------------------------------------------------------------
st.set_page_config(page_title="Uber Operations Command", layout="wide", page_icon="Taxi")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 1. PATH FIX
# ---------------------------------------------------------------------
current_file_path = os.path.abspath(__file__)
app_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(app_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)
if app_dir not in sys.path: sys.path.insert(0, app_dir)

# ---------------------------------------------------------------------
# 2. IMPORTS
# ---------------------------------------------------------------------
try:
    from main import backend
    from src.external_apis import (
        fetch_nyc_weather,
        get_traffic_congestion,
        get_nyc_events
    )
except ImportError as e:
    st.error(f"System Error: {e}")
    st.stop()

# ---------------------------------------------------------------------
# 3. CSS STYLING
# ---------------------------------------------------------------------
def render_css():
    st.markdown("""
        <style>
        .stApp { background-color: #0E1117; }
        
        .css-card {
            background-color: #1F2229;
            border: 1px solid #363B47;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 20px;
            height: 280px; 
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        div[data-testid="stMetric"] {
            background-color: #1F2229;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #29B5E8;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 14px;
            color: #A0A0A0;
        }
        div[data-testid="stMetricValue"] {
            font-size: 32px;
            color: #FFFFFF;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)
render_css()

# ---------------------------------------------------------------------
# 4. LOAD DATA
# ---------------------------------------------------------------------
zones = backend.get_zone_data()
if zones.empty:
    st.error("Database missing. Run setup first.")
    st.stop()

# ---------------------------------------------------------------------
# 5. SESSION STATE
# ---------------------------------------------------------------------
if "live_stats" not in st.session_state:
    st.session_state["live_stats"] = {"traffic": 0.0, "event": "Manual Mode", "event_count": 0, "api_error": None}
if "calculation_done" not in st.session_state:
    st.session_state["calculation_done"] = False
if "route_inputs" not in st.session_state:
    st.session_state["route_inputs"] = {}

# ---------------------------------------------------------------------
# 6. DYNAMIC NYC GRID SYSTEM (Safe Land Anchors)
# ---------------------------------------------------------------------
@st.cache_data
def get_safe_coordinates(seed_var):
    anchors = [
        # Manhattan (West Side Inland)
        (40.715, -74.009), (40.730, -74.005), (40.750, -73.999), (40.770, -73.989), (40.810, -73.955),
        # Manhattan (East Side Inland)
        (40.712, -74.000), (40.730, -73.985), (40.745, -73.975), (40.775, -73.958), (40.795, -73.940),
        # Manhattan (Central)
        (40.758, -73.985), (40.783, -73.971),
        # Brooklyn (Downtown/North)
        (40.692, -73.987), (40.688, -73.975), (40.705, -73.945), (40.715, -73.955),
        # Brooklyn (Central/South)
        (40.670, -73.975), (40.660, -73.950), (40.650, -73.960), (40.630, -73.950),
        # Queens (LIC/Astoria)
        (40.745, -73.935), (40.760, -73.920), (40.770, -73.910),
        # Queens (Inland/Sunnyside/Woodside)
        (40.740, -73.910), (40.730, -73.880), (40.745, -73.890),
        # Bronx (Inland)
        (40.820, -73.920), (40.840, -73.900), (40.850, -73.880)
    ]
    
    np.random.seed(seed_var)
    coords = {}
    
    for zid in zones["LocationID"].unique():
        chosen = anchors[np.random.randint(0, len(anchors))]
        lat_offset = np.random.uniform(-0.006, 0.006)
        lon_offset = np.random.uniform(-0.006, 0.006)
        coords[zid] = (chosen[0] + lat_offset, chosen[1] + lon_offset)
    return coords

# ---------------------------------------------------------------------
# 7. SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Control Panel")
    
    use_live_weather = st.toggle("Live Weather Sync", True)
    
    if use_live_weather:
        weather_data = fetch_nyc_weather()
        if weather_data and len(weather_data) == 2 and weather_data[1] is not None:
            temp, cond = weather_data
        else:
            temp, cond = (20, "Clear")
            
        if cond and "Rain" in cond:
            sim_weather = "Rain"
        else:
            sim_weather = "Clear"
            
        sim_hour = datetime.datetime.now().hour
        sim_date = datetime.date.today()
        
        st.info(f"🌡️ **{temp}°C** | {cond}")
    else:
        sim_date = st.date_input("Simulation Date", datetime.date.today())
        sim_hour = st.slider("Simulation Hour", 0, 23, 12)
        sim_weather = st.selectbox("Weather", ["Clear", "Rain", "Snow"])

    st.markdown("---")
    
    use_live_events = st.toggle("Live Event Sync", False)
    manual_event_count = 0
    if not use_live_events:
        manual_event_count = st.slider("Manual Event Count", 0, 20, 2)
        st.session_state["live_stats"]["event_count"] = manual_event_count
        st.session_state["live_stats"]["event"] = "Manual Mode"

    st.markdown("---")
    
    surge_pct = st.slider("Surge Percentage (%)", 0, 100, 0, step=5)
    surge_multiplier = 1 + (surge_pct / 100)
    
    p_zone = st.selectbox("Pickup Zone", zones["Zone"].unique())
    d_options = zones[zones["Zone"] != p_zone]["Zone"].unique()
    d_zone = st.selectbox("Dropoff Zone", d_options)
    
    if st.button("Calculate Demand", type="primary"):
        st.session_state["calculation_done"] = True
        st.session_state["route_inputs"] = {"p_zone": p_zone, "d_zone": d_zone, "surge_mult": surge_multiplier}
        
        # Check if secrets exist
        try:
            tomtom_key = st.secrets["api_keys"]["tomtom"]
            tm_key = st.secrets["api_keys"]["ticketmaster"]
        except:
            tomtom_key = "invalid"
            tm_key = "invalid"
            st.error("Missing .streamlit/secrets.toml file!")
        
        with st.spinner("Syncing data streams..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_traffic = executor.submit(get_traffic_congestion, 40.7128, -74.0060, tomtom_key)
                if use_live_events:
                    future_events = executor.submit(get_nyc_events, tm_key)
                
                raw_traffic = future_traffic.result()
                
                # [FIX] NO RANDOM SIMULATION. REAL API ONLY.
                if raw_traffic is None:
                    live_traffic_val = 0.0
                    api_status = "Error"
                else:
                    live_traffic_val = raw_traffic
                    api_status = "OK"
                
                if use_live_events:
                    e_name, e_cnt = future_events.result()
                    event_name, event_cnt = e_name, e_cnt
                else:
                    event_name, event_cnt = "Manual Mode", manual_event_count
                
                st.session_state["live_stats"] = {
                    "traffic": live_traffic_val, 
                    "event": event_name, 
                    "event_count": event_cnt,
                    "api_error": api_status
                }

# ---------------------------------------------------------------------
# 8. MAIN METRICS
# ---------------------------------------------------------------------
st.title("Uber Operations Command Center")
st.markdown("Real-time predictive analytics.")

traffic_val = st.session_state["live_stats"]["traffic"]
event_cnt = st.session_state["live_stats"]["event_count"]
api_status = st.session_state["live_stats"].get("api_error", "OK")

# Display Error if API Failed
if api_status == "Error":
    st.error("⚠️ **Traffic Data Unavailable:** Check TomTom API Key in `.streamlit/secrets.toml` or Quota Limits.")

p_id = zones[zones["Zone"] == p_zone]["LocationID"].values[0]
demand_metric = backend.predict_demand(p_id, sim_hour, traffic_val, sim_weather, sim_date) + (event_cnt * 5)
active_fleet = int(800 * (1.5 if (7 <= sim_hour <= 10 or 16 <= sim_hour <= 19) else 0.8))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted Ride Vol", f"{demand_metric:.0f}", delta="Zone Demand")
c2.metric("Active Fleet", f"{active_fleet:,}", delta="Off-Peak" if active_fleet < 1000 else "Peak Capacity")
c3.metric("Traffic Congestion", f"{traffic_val*100:.0f}%", delta="Live Status" if traffic_val > 0 else "Offline", delta_color="inverse")
c4.metric("Live Events", event_cnt, delta="Active")

st.markdown("---")

# ---------------------------------------------------------------------
# 9. MAP VISUALIZATION
# ---------------------------------------------------------------------
st.markdown("### Molecular Demand Map")

if st.session_state["calculation_done"]:
    active_p_zone = st.session_state["route_inputs"]["p_zone"]
    active_d_zone = st.session_state["route_inputs"]["d_zone"]
else:
    active_p_zone = p_zone
    active_d_zone = d_zone

route_seed = abs(hash(active_p_zone + active_d_zone)) % 9999
dynamic_coords = get_safe_coordinates(route_seed)

target_row = zones[zones["Zone"] == active_p_zone]
view_lat, view_lon = (40.73, -73.95)
if not target_row.empty:
    tid = target_row["LocationID"].values[0]
    view_lat, view_lon = dynamic_coords.get(tid, (40.73, -73.95))

with st.spinner("Updating Molecular Structure..."):
    map_df = zones.copy()
    map_df["coords"] = map_df["LocationID"].map(dynamic_coords)
    map_df["lat"] = map_df["coords"].apply(lambda x: x[0])
    map_df["lon"] = map_df["coords"].apply(lambda x: x[1])
    map_df["demand"] = map_df["LocationID"].apply(lambda x: backend.predict_demand(x, sim_hour, traffic_val, sim_weather, sim_date))
    
    active_zones = map_df[map_df["demand"] > 1].copy()
    layers = []

    if not active_zones.empty:
        np.random.seed(route_seed)
        upsampled_data = []
        for _, row in active_zones.iterrows():
            base_lat, base_lon = row['lat'], row['lon']
            base_dem = row['demand']
            for _ in range(15): 
                new_lat = base_lat + np.random.normal(0, 0.002)
                new_lon = base_lon + np.random.normal(0, 0.002)
                new_dem = max(1, base_dem + np.random.randint(-3, 4))
                upsampled_data.append([new_lat, new_lon, new_dem])
        
        upsampled_df = pd.DataFrame(upsampled_data, columns=["lat", "lon", "demand"])

        n_clusters = min(120, len(upsampled_df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=route_seed, n_init=10).fit(upsampled_df[["lat", "lon"]], sample_weight=upsampled_df["demand"])
        upsampled_df["cluster"] = kmeans.labels_
        
        cluster_df = upsampled_df.groupby("cluster").agg(
            lat=("lat", "mean"), 
            lon=("lon", "mean"), 
            demand=("demand", "sum")
        ).reset_index()

        cluster_df["rank"] = cluster_df["demand"].rank(pct=True)
        cluster_df["norm"] = cluster_df["rank"] 

        def get_label(n):
            if n > 0.60: return "High Demand"
            elif n > 0.30: return "Medium Demand"
            else: return "Low Demand"
        cluster_df["label"] = cluster_df["norm"].apply(get_label)

        cluster_df["ideal_radius"] = 30 + (cluster_df["norm"] ** 3) * 350
        
        coords = cluster_df[["lat", "lon"]].values
        coords_m = coords.copy()
        coords_m[:, 0] *= 111000 
        coords_m[:, 1] *= 85000 
        
        dist_matrix = cdist(coords_m, coords_m)
        np.fill_diagonal(dist_matrix, np.inf)
        
        nearest_dist = dist_matrix.min(axis=1)
        cluster_df["max_safe_radius"] = nearest_dist * 0.45
        
        cluster_df["radius"] = np.minimum(cluster_df["ideal_radius"], cluster_df["max_safe_radius"])
        cluster_df["radius"] = cluster_df["radius"].clip(lower=30) 

        def color_map(n):
            if n > 0.66: return [255, 50, 50]   # Red
            elif n > 0.33: return [255, 140, 0] # Orange
            else: return [0, 220, 120]          # Green
        cluster_df["color"] = cluster_df["norm"].apply(color_map)

        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=cluster_df,
                get_position=["lon", "lat"],
                get_radius="radius",
                get_fill_color="color",
                opacity=0.9,
                pickable=True,
                stroked=False,
                filled=True,
                radius_min_pixels=5,
                radius_max_pixels=70
            )
        ]

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=11.5, pitch=45, bearing=0),
        tooltip={"text": "{label}"}
    )
    st.pydeck_chart(deck, width="stretch")

# ---------------------------------------------------------------------
# 10. RESULTS & FORECAST
# ---------------------------------------------------------------------
if st.session_state["calculation_done"]:
    st.markdown("---")
    
    inputs = st.session_state['route_inputs']
    calc_p_id = zones[zones["Zone"] == inputs['p_zone']]["LocationID"].values[0]
    base_demand = backend.predict_demand(calc_p_id, sim_hour, traffic_val, sim_weather, sim_date)
    final_demand = base_demand + (event_cnt * 5)
    
    effective_mult = inputs['surge_mult']
    if sim_hour < 6 or sim_hour >= 22:
        effective_mult = max(effective_mult, 2.0)
    
    est_rev = final_demand * 22.5 * effective_mult

    c1, c2 = st.columns(2)
    with c1: 
        st.markdown(f"""
        <div class="css-card">
            <div>
                <h3 style="color:#29B5E8; margin-top:0; margin-bottom: 20px;">Route Details</h3>
                <p style="color:white; font-size:16px; margin: 8px 0;"><b>From:</b> {inputs['p_zone']}</p>
                <p style="color:white; font-size:16px; margin: 8px 0;"><b>To:</b> {inputs['d_zone']}</p>
            </div>
            <div>
                <hr style="border-color:#363B47; margin: 15px 0;">
                <p style="color:#A0A0A0; margin:0; font-size:14px;">Applied Surge: <span style="color:white">{effective_mult:.1f}x</span></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2: 
        st.markdown(f"""
        <div class="css-card">
            <div>
                <h3 style="color:#29B5E8; margin-top:0; margin-bottom: 20px;">Financial Projection</h3>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: auto;">
                <div>
                    <p style="color:#A0A0A0; margin-bottom:5px; font-size:14px;">Zone Demand</p>
                    <p style="color:white; font-size:26px; font-weight:bold; margin:0;">
                        {final_demand:.0f} <span style="font-size:16px; color:#A0A0A0; font-weight:normal;">rides/hr</span>
                    </p>
                </div>
                <div style="text-align: right;">
                    <p style="color:#A0A0A0; margin-bottom:5px; font-size:14px;">Est. Revenue</p>
                    <p style="color:#00FF99; font-size:32px; font-weight:bold; margin:0;">
                        ${est_rev:,.2f}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    route_hash = abs(hash(inputs['p_zone'] + inputs['d_zone'])) % 5
    
    forecast = []
    for h in range(24):
        val = backend.predict_demand(calc_p_id, h, traffic_val, sim_weather, sim_date) + (event_cnt * 5)
        if 6 <= h <= 9: val *= (1.5 + (route_hash * 0.2)) 
        elif 17 <= h <= 20: val *= (1.5 + ((5-route_hash) * 0.2)) 
        noise = np.sin(h + route_hash) * 3
        forecast.append({"Hour": h, "Predicted Demand": max(5, val + noise)})
    
    chart_df = pd.DataFrame(forecast)
    st.altair_chart(
        alt.Chart(chart_df).mark_line(point=True, color="#29B5E8").encode(x="Hour", y="Predicted Demand"),
        use_container_width=True
    )