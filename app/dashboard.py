
import os
import sys

# Fix import path for Streamlit Cloud
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import matplotlib

matplotlib.use("Agg")

from models.train_model import train_model

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="MCH Policy Dashboard",
    layout="wide"
)

# ==========================================================
# COLORS
# ==========================================================
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"

# ==========================================================
# LOAD DATA + MODEL
# ==========================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/mch_panel_data.csv")


MODEL_PATH = "models/maternal_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found. Training model... please wait ⏳")
        with st.spinner("Training model..."):
            train_model()

    return joblib.load(MODEL_PATH)


df = load_data().dropna()

model_data = load_model()

rf_model = model_data["rf_model"]
gb_model = model_data["gb_model"]

rf_r2 = model_data["rf_metrics"]["r2"]
rf_rmse = model_data["rf_metrics"]["rmse"]
rf_mae = model_data["rf_metrics"]["mae"]

gb_r2 = model_data["gb_metrics"]["r2"]
gb_rmse = model_data["gb_metrics"]["rmse"]
gb_mae = model_data["gb_metrics"]["mae"]

policy_vars = [
    "gdp_per_capita",
    "fertility_rate",
    "health_expenditure_per_capita",
    "female_secondary_enrollment"
]

# ==========================================================
# HEADER
# ==========================================================
st.title("📊 Maternal & Child Health Dashboard")
st.subheader("Policy Intelligence — Analytical View")
st.markdown("---")

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Filters")

country = st.sidebar.selectbox("Country", sorted(df["country"].unique()))
year = st.sidebar.selectbox("Year", sorted(df["year"].unique()))

filtered_df = df[df["country"] == country]
latest_data = filtered_df[filtered_df["year"] == year]

if latest_data.empty:
    latest_data = filtered_df.sort_values("year").iloc[[-1]]

row = latest_data.iloc[0]

# ==========================================================
# KPI
# ==========================================================
st.markdown("## Key Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Maternal Mortality", round(row["maternal_mortality"], 2))
c2.metric("Fertility Rate", round(row["fertility_rate"], 2))
c3.metric("GDP per Capita", round(row["gdp_per_capita"], 2))
c4.metric("Health Expenditure", round(row["health_expenditure_per_capita"], 2))

# ==========================================================
# POLICY SIMULATION
# ==========================================================
st.markdown("## Policy Scenario")

col1, col2 = st.columns(2)

with col1:
    gdp_shock = st.slider("GDP Change (%)", -20, 20, 0)
    health_shock = st.slider("Health Spending (%)", -30, 50, 0)

with col2:
    fertility_shock = st.slider("Fertility Change (%)", -50, 20, 0)
    edu_shock = st.slider("Education Change (%)", -20, 50, 0)

shock_input = pd.DataFrame({
    "gdp_per_capita": [row["gdp_per_capita"] * (1 + gdp_shock/100)],
    "fertility_rate": [row["fertility_rate"] * (1 + fertility_shock/100)],
    "health_expenditure_per_capita": [row["health_expenditure_per_capita"] * (1 + health_shock/100)],
    "female_secondary_enrollment": [row["female_secondary_enrollment"] * (1 + edu_shock/100)]
})

baseline_input = pd.DataFrame({
    "gdp_per_capita": [row["gdp_per_capita"]],
    "fertility_rate": [row["fertility_rate"]],
    "health_expenditure_per_capita": [row["health_expenditure_per_capita"]],
    "female_secondary_enrollment": [row["female_secondary_enrollment"]]
})

panel_prediction = rf_model.predict(shock_input)[0]
baseline_prediction = rf_model.predict(baseline_input)[0]
impact = baseline_prediction - panel_prediction

# ==========================================================
# VISUALS
# ==========================================================
colA, colB = st.columns(2)

with colA:
    fig_trend = px.line(
        filtered_df,
        x="year",
        y="maternal_mortality",
        title="Mortality Trend",
        color_discrete_sequence=[BLUE]
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with colB:
    comparison_df = pd.DataFrame({
        "Scenario": ["Baseline", "Policy"],
        "Value": [baseline_prediction, panel_prediction]
    })

    fig_policy = px.bar(
        comparison_df,
        x="Scenario",
        y="Value",
        color="Scenario",
        color_discrete_map={
            "Baseline": BLUE,
            "Policy": ORANGE
        },
        title="Policy Impact"
    )
    st.plotly_chart(fig_policy, use_container_width=True)

# ==========================================================
# PERFORMANCE
# ==========================================================
st.markdown("## Model Performance")

performance_df = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting"],
    "R²": [rf_r2, gb_r2],
    "RMSE": [rf_rmse, gb_rmse],
    "MAE": [rf_mae, gb_mae]
})

st.dataframe(performance_df.round(3))

# ==========================================================
# SHAP
# ==========================================================
st.markdown("## AI Explanation")

ml_input = shock_input[policy_vars]

explainer = shap.TreeExplainer(rf_model)
shap_vals = explainer.shap_values(ml_input)

fig, ax = plt.subplots()
shap.bar_plot(shap_vals[0], feature_names=policy_vars)
st.pyplot(fig)

# ==========================================================
# PDP
# ==========================================================
st.markdown("## Global Effects")

X_global = df[policy_vars]

fig_pdp, ax = plt.subplots(figsize=(10, 6))

PartialDependenceDisplay.from_estimator(
    rf_model,
    X_global,
    policy_vars,
    ax=ax
)

st.pyplot(fig_pdp)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption("Maternal & Child Health Policy Intelligence System")

