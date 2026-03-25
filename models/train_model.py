import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_PATH = "data/processed/mch_panel_data.csv"
MODEL_PATH = "models/maternal_model.pkl"

FEATURES = [
    "gdp_per_capita",
    "fertility_rate",
    "health_expenditure_per_capita",
    "female_secondary_enrollment"
]

TARGET = "maternal_mortality"


# ==========================================================
# LIGHTWEIGHT TRAINING FUNCTION (DEPLOYMENT SAFE)
# ==========================================================

def train_and_save_model():

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Random Forest (lighter)
    # -----------------------------
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # -----------------------------
    # Gradient Boosting (lighter)
    # -----------------------------
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)

    # -----------------------------
    # METRICS
    # -----------------------------
    model_output = {
        "rf_model": rf_model,
        "rf_metrics": {
            "r2": r2_score(y_test, rf_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, rf_pred)),
            "mae": mean_absolute_error(y_test, rf_pred)
        },
        "gb_model": gb_model,
        "gb_metrics": {
            "r2": r2_score(y_test, gb_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, gb_pred)),
            "mae": mean_absolute_error(y_test, gb_pred)
        }
    }

    joblib.dump(model_output, MODEL_PATH)

    print("Model trained and saved successfully")

    return model_output


# ==========================================================
# SAFE LOADER
# ==========================================================

def train_model():

    try:
        return joblib.load(MODEL_PATH)

    except:
        print("Training model (lightweight)...")
        return train_and_save_model()


# ==========================================================
# MANUAL RUN
# ==========================================================

if __name__ == "__main__":
    train_and_save_model()