"""
Weather Auto-fill
------------------
Uses Open-Meteo (free, no API key required) to look up a city's coordinates
and current weather conditions, then maps them onto this project's 4
weather categories (Sunny, Rainy, Foggy, Icy) and derives time-of-day from
the location's own local time (not the server's clock).

Two calls:
1. Geocoding API - city name -> latitude/longitude
2. Forecast API   - lat/lon -> current weather code + local time

Both endpoints are public and require no signup/key. Every function here
fails soft (returns None) on any network/parsing error, so the app can
always fall back to manual dropdown selection.
"""

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes (used by Open-Meteo) mapped onto our 4 categories.
# Reference: https://open-meteo.com/en/docs (WMO Weather interpretation codes)
_FOG_CODES = {45, 48}
_ICY_CODES = {56, 57, 66, 67, 71, 73, 75, 77, 85, 86}
_RAINY_CODES = {51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99}


def geocode_city(city_name, timeout=6):
    """Returns {'lat', 'lon', 'name', 'country'} or None if not found/failed."""
    try:
        resp = requests.get(GEOCODE_URL, params={"name": city_name, "count": 1}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results")
        if not results:
            return None
        r = results[0]
        return {
            "lat": r["latitude"],
            "lon": r["longitude"],
            "name": r.get("name", city_name),
            "country": r.get("country", ""),
        }
    except Exception:
        return None


def geocode_city_suggestions(city_name, count=5, timeout=6):
    """
    Returns a list of matching cities (up to `count`) for autocomplete-style
    selection, so the user picks a real match instead of relying on getting
    the spelling exactly right. Returns [] on no matches or any failure.
    """
    try:
        resp = requests.get(GEOCODE_URL, params={"name": city_name, "count": count}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        return [
            {
                "lat": r["latitude"],
                "lon": r["longitude"],
                "name": r.get("name", city_name),
                "admin1": r.get("admin1", ""),
                "country": r.get("country", ""),
            }
            for r in results
        ]
    except Exception:
        return []


def fetch_current_conditions(lat, lon, timeout=6):
    """Returns {'weather_code', 'temperature', 'local_time'} or None on failure."""
    try:
        resp = requests.get(FORECAST_URL, params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code",
            "timezone": "auto",
        }, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current")
        if not current:
            return None
        return {
            "weather_code": current.get("weather_code"),
            "temperature": current.get("temperature_2m"),
            "local_time": current.get("time"),  # ISO string in the location's local time
        }
    except Exception:
        return None


def weather_code_to_category(code):
    """Maps a WMO weather code onto one of: Sunny, Rainy, Foggy, Icy."""
    if code in _FOG_CODES:
        return "Foggy"
    if code in _ICY_CODES:
        return "Icy"
    if code in _RAINY_CODES:
        return "Rainy"
    return "Sunny"  # clear, mainly clear, partly cloudy, overcast


def local_hour_to_time_of_day(local_time_str):
    """Parses an ISO local-time string (e.g. '2026-07-15T14:30') -> 'Day' or 'Night'."""
    try:
        hour = int(local_time_str.split("T")[1].split(":")[0])
    except Exception:
        return "Day"
    return "Day" if 6 <= hour < 19 else "Night"