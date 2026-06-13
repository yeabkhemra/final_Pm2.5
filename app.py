import os
from datetime import timedelta
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Project settings
# -----------------------------
CITY = "Phnom Penh"
LATITUDE = 11.5564
LONGITUDE = 104.9282
TIMEZONE = "Asia/Phnom_Penh"
H = 48  # prediction horizon in hours

MODEL_CANDIDATES = [
    "pm25_phnom_penh_xgboost_model.pkl",   # recommended file name
    "pm25_phnom_penh_best_model.pkl",      # fallback from your current notebook
]
FEATURE_COLS_PATH = "pm25_phnom_penh_feature_cols.pkl"
CONFIG_PATH = "pm25_phnom_penh_config.pkl"  # optional: cap values, horizon


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_resource
def load_model_assets():
    """Load trained model + feature column order from local files."""
    model_path = None
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError(
            "Model file not found. Put pm25_phnom_penh_xgboost_model.pkl "
            "in the same folder as app.py."
        )

    if not os.path.exists(FEATURE_COLS_PATH):
        raise FileNotFoundError(
            "Feature column file not found. Put pm25_phnom_penh_feature_cols.pkl "
            "in the same folder as app.py."
        )

    model = joblib.load(model_path)
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    config = {}
    if os.path.exists(CONFIG_PATH):
        config = joblib.load(CONFIG_PATH)

    return model, feature_cols, config, model_path


def call_api(url: str, params: dict) -> pd.DataFrame:
    """Call Open-Meteo API and return hourly data as a DataFrame."""
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data or "time" not in data["hourly"]:
        raise ValueError("Open-Meteo response does not contain hourly data.")

    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df


@st.cache_data(ttl=1800)
def fetch_open_meteo_data(latitude: float, longitude: float) -> pd.DataFrame:
    """
    Fetch recent and forecast hourly air-quality + weather data.
    Forecast rows are included only to create the target timestamp row.
    The model features still use the notebook's positive shifts only.
    """
    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    weather_url = "https://api.open-meteo.com/v1/forecast"

    air_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm2_5,pm10",
        "timezone": TIMEZONE,
        "past_hours": 240,
        "forecast_hours": 72,
    }

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "wind_speed_10m,wind_direction_10m,surface_pressure,cloud_cover"
        ),
        "timezone": TIMEZONE,
        "past_hours": 240,
        "forecast_hours": 72,
    }

    air = call_api(air_url, air_params)
    weather = call_api(weather_url, weather_params)

    df = pd.merge(air, weather, on="timestamp", how="outer").sort_values("timestamp")

    # Rename Open-Meteo variables to match your notebook column names
    rename_map = {
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "precipitation": "rain",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_dir",
        "surface_pressure": "pressure",
        "cloud_cover": "cloud",
    }
    df = df.rename(columns=rename_map)

    needed = ["pm2_5", "pm10", "temp", "humidity", "rain", "wind_speed", "wind_dir", "pressure", "cloud"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    # Fill small API gaps so lag/rolling features can still be calculated
    df[needed] = df[needed].interpolate(limit_direction="both").ffill().bfill()
    return df.reset_index(drop=True)


def add_methodology_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create the same no-leakage features from the notebook."""
    feat = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Optional: use the same caps saved from training
    cap_pm25 = config.get("cap_pm25") or config.get("PM2.5_cap")
    cap_pm10 = config.get("cap_pm10") or config.get("PM10_cap")
    if cap_pm25 is not None:
        feat["pm2_5"] = feat["pm2_5"].clip(upper=float(cap_pm25))
    if cap_pm10 is not None:
        feat["pm10"] = feat["pm10"].clip(upper=float(cap_pm10))

    # Time features are based on the timestamp we are predicting FOR
    feat["hour"] = feat["timestamp"].dt.hour
    feat["month"] = feat["timestamp"].dt.month
    feat["dayofweek"] = feat["timestamp"].dt.dayofweek

    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
    feat["month_sin"] = np.sin(2 * np.pi * (feat["month"] - 1) / 12)
    feat["month_cos"] = np.cos(2 * np.pi * (feat["month"] - 1) / 12)
    feat["dow_sin"] = np.sin(2 * np.pi * feat["dayofweek"] / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * feat["dayofweek"] / 7)
    feat["is_weekend"] = feat["dayofweek"].isin([5, 6]).astype(int)
    feat["is_rush_hour"] = feat["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    cambodian_holidays = [
        (1, 1), (1, 7), (3, 8),
        (4, 14), (4, 15), (4, 16),
        (5, 1), (5, 14), (5, 15), (5, 16),
        (6, 18), (9, 24), (10, 15), (10, 29), (11, 9),
    ]
    month_day = list(zip(feat["timestamp"].dt.month, feat["timestamp"].dt.day))
    feat["is_holiday"] = pd.Series(month_day).isin(cambodian_holidays).astype(int).values

    pm25_lags = [0, 1, 3, 6, 12, 24, 48, 72, 168]
    for lag in pm25_lags:
        feat[f"pm25_lag_{lag}h"] = feat["pm2_5"].shift(H + lag)

    for window in [3, 6, 12, 24, 48]:
        shifted = feat["pm2_5"].shift(H + 1)
        feat[f"pm25_roll_mean_{window}h"] = shifted.rolling(window).mean()
        feat[f"pm25_roll_std_{window}h"] = shifted.rolling(window).std()

    for lag in [0, 1, 3, 6, 12, 24, 48]:
        feat[f"pm10_lag_{lag}h"] = feat["pm10"].shift(H + lag)

    meteo_cols = ["temp", "humidity", "rain", "wind_speed", "wind_dir", "pressure", "cloud"]
    for col in meteo_cols:
        feat[f"{col}_now"] = feat[col].shift(H)
        feat[f"{col}_lag_12h"] = feat[col].shift(H + 12)
        feat[f"{col}_lag_24h"] = feat[col].shift(H + 24)

    feat["humid_x_wind"] = feat["humidity_now"] * feat["wind_speed_now"]
    feat["temp_x_humidity"] = feat["temp_now"] * feat["humidity_now"]

    return feat


def pm25_category(value: float) -> str:
    if value <= 12:
        return "Good"
    if value <= 35:
        return "Moderate"
    if value <= 55:
        return "Unhealthy for Sensitive Groups"
    if value <= 150:
        return "Unhealthy"
    return "Very Unhealthy"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PM2.5 Forecasting — Phnom Penh", page_icon="🌫️", layout="wide")

st.title("🌫️ PM2.5 Forecasting — Phnom Penh")
st.caption("XGBoost model · 48-hour ahead prediction · Open-Meteo live data")

with st.sidebar:
    st.header("Settings")
    latitude = st.number_input("Latitude", value=LATITUDE, format="%.4f")
    longitude = st.number_input("Longitude", value=LONGITUDE, format="%.4f")
    st.write("Prediction horizon:", f"{H} hours ahead")
    refresh = st.button("Refresh data")

if refresh:
    st.cache_data.clear()

try:
    model, feature_cols, config, model_path = load_model_assets()

    if model_path.endswith("best_model.pkl"):
        st.warning(
            "The app loaded pm25_phnom_penh_best_model.pkl. "
            "For your requirement, save and use pm25_phnom_penh_xgboost_model.pkl."
        )

    raw_df = fetch_open_meteo_data(latitude, longitude)
    feature_df = add_methodology_features(raw_df, config)

    local_now = pd.Timestamp.now(tz=ZoneInfo(TIMEZONE)).tz_localize(None).floor("H")
    prediction_time = raw_df[raw_df["timestamp"] <= local_now]["timestamp"].max()
    target_time = prediction_time + pd.Timedelta(hours=H)

    target_rows = feature_df[feature_df["timestamp"] == target_time]
    if target_rows.empty:
        raise ValueError("Target timestamp is not available from Open-Meteo yet. Try refreshing later.")

    row = target_rows.iloc[0]
    X = pd.DataFrame([row]).reindex(columns=feature_cols)

    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Some required model features are missing: {missing[:10]}")

    prediction = float(model.predict(X)[0])
    category = pm25_category(prediction)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted PM2.5", f"{prediction:.2f} µg/m³")
    col2.metric("Category", category)
    col3.metric("Target time", target_time.strftime("%Y-%m-%d %H:%M"))

    st.info(
        f"The model predicts PM2.5 for **{target_time:%Y-%m-%d %H:%M}**, "
        f"using only data available at **{prediction_time:%Y-%m-%d %H:%M}** and earlier."
    )

    st.subheader("Recent PM2.5 from Open-Meteo")
    chart_df = raw_df[raw_df["timestamp"] <= prediction_time].tail(72).set_index("timestamp")
    st.line_chart(chart_df[["pm2_5", "pm10"]])

    st.subheader("Feature row used by XGBoost")
    st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)

    with st.expander("Methodology used in this app"):
        st.markdown(
            """
            This app follows the same methodology as the notebook:

            1. Fetch hourly PM2.5, PM10, and weather data from Open-Meteo.
            2. Create cyclic time features: hour, month, and day of week.
            3. Add weekend, rush-hour, and Cambodian holiday indicators.
            4. Create PM2.5 and PM10 lag features using positive shifts only.
            5. Create PM2.5 rolling mean and rolling standard deviation.
            6. Use weather at prediction time, 12 hours before, and 24 hours before.
            7. Predict PM2.5 48 hours ahead using the saved XGBoost model.

            No CSV upload is required, and future PM2.5 is not used as a model input.
            """
        )

except Exception as e:
    st.error(str(e))
    st.markdown(
        """
        **Check these files are in the same folder as `app.py`:**

        - `pm25_phnom_penh_xgboost_model.pkl`
        - `pm25_phnom_penh_feature_cols.pkl`
        - optional: `pm25_phnom_penh_config.pkl`
        """
    )
