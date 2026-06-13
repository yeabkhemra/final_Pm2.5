# PM2.5 Forecasting Streamlit App

This app deploys the PM2.5 Phnom Penh 48-hour forecasting model using XGBoost.
It does not require dataset upload. It fetches hourly PM2.5, PM10, and weather data from Open-Meteo.

## Required model files
Put these files in the same folder as `app.py`:

- `pm25_phnom_penh_xgboost_model.pkl`
- `pm25_phnom_penh_feature_cols.pkl`
- optional: `pm25_phnom_penh_config.pkl`

To create them, run the code in `save_xgboost_files_cell.py` at the end of your notebook after training XGBoost.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and your `.pkl` files.
3. Go to Streamlit Community Cloud.
4. Choose your repository.
5. Set main file path to `app.py`.
6. Deploy.
