# Add this cell at the end of your notebook after training XGBoost.
# It saves XGBoost specifically for Streamlit deployment.

import joblib

prefix = "pm25_phnom_penh"

joblib.dump(xgb_model, f"{prefix}_xgboost_model.pkl")
joblib.dump(feature_cols, f"{prefix}_feature_cols.pkl")
joblib.dump({
    "cap_pm25": float(cap_pm25),
    "cap_pm10": float(cap_pm10),
    "horizon": int(H),
    "city": CITY,
}, f"{prefix}_config.pkl")

print("Saved Streamlit files:")
print(f"{prefix}_xgboost_model.pkl")
print(f"{prefix}_feature_cols.pkl")
print(f"{prefix}_config.pkl")
