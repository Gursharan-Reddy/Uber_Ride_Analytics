# Uber Demand Control Center

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Operational-brightgreen)

## Project Overview

The **Uber Demand Control Center** is a predictive analytics dashboard designed for fleet operations managers. It leverages historical NYC Taxi trip data and meteorological records to forecast ride demand in real-time. 

Unlike standard dashboards, this system utilizes a **Non-Continuous Density Heatmap** to visualize distinct high-demand clusters ("Hotspots") across New York City. This unique visualization filters out low-density noise to create clear, isolated demand islands, enabling precise driver allocation and supply rebalancing.

## Key Features

### 1. Advanced Geospatial Analytics
* **Non-Continuous Density Clusters:** Utilizes optimized Kernel Density Estimation (KDE) with dynamic thresholding (0.30) to isolate high-demand pockets and remove the "fog" of low-demand areas.
* **Organic Visualization:** Simulates natural demand spread using Gaussian noise to prevent rigid grid artifacts, resulting in a cleaner, more realistic map overlay.
* **Interactive Mapping:** Powered by **Pydeck** and **Carto**, featuring a high-contrast dark mode style for maximum visibility of demand hotspots.

### 2. Machine Learning Engine
* **Predictive Model:** Random Forest Regressor trained on thousands of trip records.
* **Dynamic Inputs:** Predicts demand based on a comprehensive set of factors:
    * **Temporal:** Hour of Day, Day of Week, Month, Holidays, Weekend Status.
    * **Environmental:** Real-time Temperature (°F) and granular weather conditions (**Freezing, Cold, Mild, Warm, Hot, Rain**).
    * **Operational:** Dynamic Surge Multipliers to simulate market conditions.

### 3. Route & Supply Intelligence
* **Route Forecasting:** Calculates specific ride volume between selected **Pickup** and **Drop-off** zones.
* **Smart Validation Logic:** Includes robust error handling to prevent invalid operational queries (e.g., identical pickup/destination points).
* **Actionable Alerts:** Automatically flags zones with high demand (>1.5 rides/hour) for immediate fleet rebalancing.

## Installation & Setup

### Prerequisites
* Python 3.8 or higher
* PIP (Python Package Installer)

### Step 1: Clone the Repository
```bash
git clone [https://github.com/Gursharan-Reddy/Uber_Ride_Analytics.git](https://github.com/Gursharan-Reddy/Uber_Ride_Analytics.git)
cd Uber_Ride_Analytics

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run the Data Pipeline (First Time Only)
This script processes raw data and trains the machine learning model.
python app/main.py

Step 4: Launch the Dashboard
streamlit run app/streamlit_app.py

📂 Project Structure
Uber_Ride_Analytics/
├── app/
│   ├── main.py             # ETL Pipeline & Model Training Driver
│   └── streamlit_app.py    # Interactive Dashboard Frontend
├── data/
│   ├── raw/                # Raw Trip Data & Lookup CSVs
│   └── processed/          # Cleaned Parquet files & GeoJSON
├── outputs/
│   ├── model.pkl           # Trained Random Forest Model
│   └── feature_names.pkl   # Serialized Feature List
├── src/
│   ├── data_loader.py      # Data Ingestion Logic
│   ├── feature_eng.py      # Temporal & Weather Feature Engineering
│   └── model_train.py      # Scikit-Learn Training Script
├── requirements.txt        # Python Dependencies
└── README.md               # Project Documentation

Usage
Select Parameters: Use the sidebar to set the simulation date, time, and weather conditions.

View Map: Observe the "Red Hotspots" on the map. These indicate areas where demand exceeds the 30% threshold.

Check Routes: Select a specific Pickup and Drop-off zone to get a numerical ride forecast.

Future Enhancements
Integration of live traffic API data.

Driver-side mobile notification system.

Automated weekly PDF reporting for stakeholders.

Developed by: Gursharan Reddy