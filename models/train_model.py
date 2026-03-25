import pandas as pd
import joblib
import os
 import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model():
    # Load data
    df = pd.read_csv("data/processed/mch_panel_data.csv").dropna()

    X = df[["gdp_per_capita", "fertility_rate"]]
    y = df["maternal_mortality"]

    # -----------------------------
    # MODELS (LIGHTWEIGHT)
    # -----------------------------
    panel_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=10, random_state=42)

    # Train models
    panel_model.fit(X, y)
    rf_model.fit(X, y)
    gb_model.fit(X, y)

    # Predictions (use panel model for metrics)
    preds = panel_model.predict(X)

    # Metrics
    r2 = r2_score(y, preds)   
rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)

    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # Save everything EXACTLY as dashboard expects
    joblib.dump({
        "panel_model": panel_model,
        "rf_model": rf_model,
        "gb_model": gb_model,
        "panel_metrics": {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae)
        }
    }, "models/maternal_model.pkl")