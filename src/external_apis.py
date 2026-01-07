import requests
import datetime

def fetch_nyc_weather():
    # Original Weather API integration
    API_KEY = "496697ebc96d923e8617116ae5ebd185"
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q=New York&appid={API_KEY}&units=imperial"
        r = requests.get(url, timeout=3).json()
        if r.get("cod") != 200: 
            return None, None
        return r["main"]["temp"], r["weather"][0]["main"]
    except:
        return None, None

def get_traffic_congestion(lat, lon, api_key):
    # TomTom Traffic API
    if not api_key: return 0.0
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={api_key}&point={lat},{lon}"
    try:
        r = requests.get(url, timeout=2).json()
        flow = r.get('flowSegmentData', {})
        curr, free = flow.get('currentSpeed', 1), flow.get('freeFlowSpeed', 1)
        return max(0.0, 1 - (curr / free))
    except: return 0.0

def get_nyc_events(api_key):
    # Ticketmaster Events API
    if not api_key: return "No Data", 0
    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    today = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"apikey": api_key, "city": "New York", "startDateTime": today, "size": 1}
    try:
        r = requests.get(url, params=params, timeout=3).json()
        events = r.get('_embedded', {}).get('events', [])
        return (events[0]['name'], len(events)) if events else ("None", 0)
    except: return "Error", 0