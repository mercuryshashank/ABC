"""
ABC Ltd. — Machine Failure Prediction Tool
Streamlit app for non-technical managers/supervisors.

Loads six .pkl artifacts produced by ABC_Ltd_Machine_Failure_Prediction.ipynb:
  logistic_model.pkl, linear_model.pkl, scaler_clf.pkl, scaler_reg.pkl,
  clf_columns.pkl, reg_columns.pkl
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ABC Ltd. Machine Failure Predictor", page_icon="🛠️", layout="centered")

# ---------------------------------------------------------------
# Load models (cached so they only load once per session)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    log_model = joblib.load("logistic_model.pkl")
    lin_model = joblib.load("linear_model.pkl")
    scaler_clf = joblib.load("scaler_clf.pkl")
    scaler_reg = joblib.load("scaler_reg.pkl")
    clf_columns = joblib.load("clf_columns.pkl")
    reg_columns = joblib.load("reg_columns.pkl")
    return log_model, lin_model, scaler_clf, scaler_reg, clf_columns, reg_columns

log_model, lin_model, scaler_clf, scaler_reg, clf_columns, reg_columns = load_artifacts()

st.title("ABC Ltd. — Machine Failure Prediction Tool")
st.caption(
    "Enter a machine's current operating readings below. The tool estimates its "
    "failure risk and its expected torque under normal operation — built as "
    "decision support, not automation. You make the final call."
)

st.divider()

# ---------------------------------------------------------------
# Input form — plain language, no technical knowledge needed
# ---------------------------------------------------------------
st.subheader("1. Machine readings")

col1, col2 = st.columns(2)
with col1:
    tier = st.selectbox(
        "Machine quality tier",
        ["Low", "Medium", "High"],
        help="Low/Medium/High refers to the machine's build specification, not its condition.",
    )
    air_temp = st.number_input("Air temperature (K)", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
    process_temp = st.number_input("Process temperature (K)", min_value=295.0, max_value=315.0, value=310.0, step=0.1)

with col2:
    rpm = st.number_input("Rotational speed (RPM)", min_value=1000, max_value=3000, value=1500, step=10)
    torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=80.0, value=40.0, step=0.5)
    tool_wear = st.number_input("Tool wear so far (minutes)", min_value=0, max_value=260, value=100, step=1)

predict_clicked = st.button("Predict", type="primary", use_container_width=True)

st.divider()

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if predict_clicked:
    tier_low = 1 if tier == "Low" else 0
    tier_medium = 1 if tier == "Medium" else 0

    # --- Classification: Machine failure risk ---
    clf_row = {
        "Air_Temperature_K": air_temp,
        "Process_Temperature_K": process_temp,
        "Rotational_Speed_RPM": rpm,
        "Torque_Nm": torque,
        "Tool_Wear_Min": tool_wear,
        "Machine_Quality_Tier_Low": tier_low,
        "Machine_Quality_Tier_Medium": tier_medium,
    }
    X_clf_input = pd.DataFrame([clf_row])[clf_columns]
    X_clf_scaled = scaler_clf.transform(X_clf_input)
    failure_proba = log_model.predict_proba(X_clf_scaled)[0, 1]
    risk_label = "High" if failure_proba >= 0.5 else "Low"

    # --- Regression: expected torque under these conditions ---
    reg_row = {
        "Air_Temperature_K": air_temp,
        "Process_Temperature_K": process_temp,
        "Rotational_Speed_RPM": rpm,
        "Tool_Wear_Min": tool_wear,
        "Machine_Quality_Tier_Low": tier_low,
        "Machine_Quality_Tier_Medium": tier_medium,
    }
    X_reg_input = pd.DataFrame([reg_row])[reg_columns]
    X_reg_scaled = scaler_reg.transform(X_reg_input)
    expected_torque = lin_model.predict(X_reg_scaled)[0]
    torque_deviation = torque - expected_torque

    st.subheader("2. Prediction")

    c1, c2 = st.columns(2)
    with c1:
        if risk_label == "High":
            st.error(f"⚠️ Failure risk: **HIGH** ({failure_proba:.0%})")
        else:
            st.success(f"✅ Failure risk: **LOW** ({failure_proba:.0%})")
    with c2:
        st.metric("Expected torque for these settings", f"{expected_torque:.1f} Nm",
                   delta=f"{torque_deviation:+.1f} Nm vs. actual reading")

    if abs(torque_deviation) > 10:
        st.warning(
            f"The actual torque reading ({torque:.1f} Nm) is well outside what these "
            f"settings would normally produce (~{expected_torque:.1f} Nm). This can indicate "
            "a sensor fault or an abnormal load, and is worth checking even if the failure "
            "risk above is Low."
        )

    st.subheader("3. Why the model says this")
    coefs = pd.Series(log_model.coef_[0], index=clf_columns)
    input_scaled_series = pd.Series(X_clf_scaled[0], index=clf_columns)
    contributions = (coefs * input_scaled_series).sort_values(key=abs, ascending=False).head(5)

    st.write("Top factors driving this machine's risk score:")
    for factor, contrib in contributions.items():
        direction = "increases" if contrib > 0 else "reduces"
        label = factor.replace("_", " ")
        st.write(f"- **{label}** {direction} the predicted risk")

    st.caption(
        "This tool is decision support built on historical sensor data (AI4I 2020 "
        "predictive maintenance data, relabeled for ABC Ltd.). It does not replace "
        "inspection or engineering judgment."
    )

st.divider()
with st.expander("About this tool"):
    st.write(
        """
        **Model 1 — Logistic Regression:** predicts the probability a machine fails,
        based on its current sensor readings and quality tier. Trained with balanced
        class weights because failures are rare (~3.4% of records) — this tool is
        tuned to catch more real failures at the cost of some false alarms, since a
        missed failure is far costlier than an unnecessary inspection.

        **Model 2 — Linear Regression:** predicts the torque a machine should show
        given its speed, temperature and tier. Torque and rotational speed move
        together through a roughly constant-power relationship on this equipment —
        a real engineering property, not a coincidence. A large gap between actual
        and expected torque can flag a sensor fault or abnormal load before it
        shows up as a failure.
        """
    )
