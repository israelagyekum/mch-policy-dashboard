import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import matplotlib
matplotlib.use("Agg")
import os
import joblib
import streamlit as st
import pandas as pd

from models.train_model import train_model
# ==========================================================
# PAGE CONFIG (ONLY ONCE)
# ==========================================================
st.set_page_config(
    page_title="MCH Policy Dashboard",
    layout="wide"
)

# ==========================================================
# COLOR SYSTEM (LIKE YOUR REFERENCE DASHBOARD)
# ==========================================================
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
DARK_BG = "#0B1F3A"

# ==========================================================
# LOAD DATA + MODEL (CACHED)
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

# ==========================================================
# EXTRACT MODELS (FIXED STRUCTURE)
# ==========================================================
global_model = model_data["panel_model"]

global_r2 = model_data["panel_metrics"]["r2"]
global_rmse = model_data["panel_metrics"]["rmse"]
global_mae = model_data["panel_metrics"]["mae"]

rf_model = model_data["rf_model"]
rf_r2 = model_data["rf_metrics"]["r2"]
rf_rmse = model_data["rf_metrics"]["rmse"]
rf_mae = model_data["rf_metrics"]["mae"]

gb_model = model_data["gb_model"]
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
st.markdown("# 📊 Maternal & Child Health Dashboard")
st.markdown("### Policy Intelligence — Analytical View")
st.markdown("---")

st.markdown("""
### 📘 How to Interpret This Dashboard

This platform provides decision-support insights for maternal health policy.

**Key Components:**
- **Key Indicators:** Current country-level metrics
- **Policy Scenario:** Simulate policy changes and observe impact
- **Policy Impact:** Estimated change in maternal mortality
- **Model Performance:** Accuracy of econometric vs machine learning models
- **Structural Effects:** Long-run relationships from panel regression
- **AI Interpretability:** Feature importance using SHAP

⚠️ Results are analytical estimates and should support—not replace—policy decisions.
""")

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Filters")

country = st.sidebar.selectbox("Country", sorted(df["country"].unique()))
year = st.sidebar.selectbox("Year", sorted(df["year"].unique()))

filtered_df = df[df["country"] == country]
latest_data = filtered_df[filtered_df["year"] == year]

if latest_data.empty:
    st.markdown('<div class="custom-warning">⚠️ No data available for selected year. Showing closest available year.</div>', unsafe_allow_html=True)

latest_data = filtered_df.sort_values("year").iloc[[-1]]

row = latest_data.iloc[0]

# ==========================================================
# KPI ROW (LIKE YOUR IMAGE)
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
    "female_secondary_enrollment": [row["female_secondary_enrollment"] * (1 + edu_shock/100)],
    "country": [country],
    "year": [year]
})

shock_input["country"] = shock_input["country"].astype("category")
shock_input["year"] = shock_input["year"].astype("category")

baseline_input = pd.DataFrame({
    "gdp_per_capita": [row["gdp_per_capita"]],
    "fertility_rate": [row["fertility_rate"]],
    "health_expenditure_per_capita": [row["health_expenditure_per_capita"]],
    "female_secondary_enrollment": [row["female_secondary_enrollment"]],
    "country": [country],
    "year": [year]
})

baseline_input["country"] = baseline_input["country"].astype("category")
baseline_input["year"] = baseline_input["year"].astype("category")

panel_prediction = global_model.predict(shock_input)[0]
baseline_prediction = global_model.predict(baseline_input)[0]
impact = baseline_prediction - panel_prediction

# ==========================================================
# MAIN DASHBOARD GRID (LIKE YOUR REFERENCE IMAGE)
# ==========================================================
colA, colB = st.columns(2)

# ---- LEFT: TREND ----
with colA:
    fig_trend = px.line(
        filtered_df,
        x="year",
        y="maternal_mortality",
        title="Mortality Trend",
        color_discrete_sequence=[BLUE]
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ---- RIGHT: POLICY IMPACT ----
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
# PERFORMANCE TABLE
# ==========================================================
st.markdown("## Model Performance")

performance_df = pd.DataFrame({
    "Model": ["Panel", "Random Forest", "Gradient Boosting"],
    "R²": [global_r2, rf_r2, gb_r2],
    "RMSE": [global_rmse, rf_rmse, gb_rmse],
    "MAE": [global_mae, rf_mae, gb_mae]
})

st.dataframe(performance_df.round(3))

# ==========================================================
# STRUCTURAL EFFECTS
# ==========================================================
st.markdown("## Policy Drivers (Econometric Effects)")

coef_df = pd.DataFrame({
    "Variable": policy_vars,
    "Coefficient": [global_model.params[v] for v in policy_vars]
})

fig_coef = px.bar(coef_df, x="Variable", y="Coefficient",
                  color_discrete_sequence=[GREEN])

st.plotly_chart(fig_coef, use_container_width=True)

# ==========================================================
# SHAP
# ==========================================================
st.markdown("## AI Explanation (What Drives Predictions)")

ml_input = shock_input[policy_vars].apply(pd.to_numeric)

rf_explainer = shap.TreeExplainer(rf_model)
shap_vals = rf_explainer.shap_values(ml_input)

fig, ax = plt.subplots()
shap.bar_plot(shap_vals[0], feature_names=policy_vars)
st.pyplot(fig)

# ==========================================================
# PDP (GLOBAL STRUCTURE)
# ==========================================================
st.markdown("## Global Nonlinear Effects")

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
st.caption("Institutional MCH Policy Intelligence Platform")

st.markdown("""
---
**Institutional Analytics Platform**  
Maternal & Child Health Policy Intelligence System  
Data Source: World Bank (2000–2022)
""")
st.caption("Models are trained on historical data and assume structural stability.")