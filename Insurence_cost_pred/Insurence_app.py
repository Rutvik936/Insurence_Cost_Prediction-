import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load Models
# -------------------------------
rf_model = joblib.load("rf_insurence.pkl")
xgb_model = joblib.load("xgb_insurence.pkl")

st.set_page_config(page_title="Insurance Cost Prediction", page_icon="💰", layout="centered")

st.title("💰 Insurance Cost Prediction App")
st.write("This app predicts insurance cost based on health and lifestyle factors using **Random Forest** and **XGBoost** models.")

# -------------------------------
# Input Form
# -------------------------------
st.header("🧾 Enter Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    diabetes = st.selectbox("Diabetes", ["No", "Yes"], help="No = 0, Yes = 1")
    bp = st.selectbox("Blood Pressure Problems", ["No", "Yes"], help="No = 0, Yes = 1")
    allergies = st.selectbox("Known Allergies", ["No", "Yes"], help="No = 0, Yes = 1")
    chronic = st.selectbox("Any Chronic Diseases", ["No", "Yes"], help="No = 0, Yes = 1")

with col2:
    transplant = st.selectbox("Any Transplants", ["No", "Yes"], help="No = 0, Yes = 1")
    cancer_history = st.selectbox("History of Cancer in Family", ["No", "Yes"], help="No = 0, Yes = 1")
    surgeries = st.number_input("Number of Major Surgeries", min_value=0, max_value=20, value=0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

# Convert Yes/No to 1/0
diabetes = 1 if diabetes == "Yes" else 0
bp = 1 if bp == "Yes" else 0
allergies = 1 if allergies == "Yes" else 0
chronic = 1 if chronic == "Yes" else 0
transplant = 1 if transplant == "Yes" else 0
cancer_history = 1 if cancer_history == "Yes" else 0

# Prepare input
features = np.array([[age, diabetes, bp, allergies, chronic,
                      transplant, cancer_history, surgeries, bmi]])

# -------------------------------
# Predictions
# -------------------------------
if st.button("🔮 Predict Insurance Cost"):
    rf_pred = rf_model.predict(features)[0]
    xgb_pred = xgb_model.predict(features)[0]
    avg_pred = (rf_pred + xgb_pred) / 2

    st.success(f"✅ Random Forest Prediction: **{rf_pred:.2f}**")
    st.success(f"✅ XGBoost Prediction: **{xgb_pred:.2f}**")
    st.info(f"📊 Average Predicted Insurance Cost: **{avg_pred:.2f}**")
# -------------------------------
# Feature Importance from RF model
# -------------------------------
with st.expander("📊 Random Forest Feature Importance"):
    # Get feature importances
    rf_importances = rf_model.feature_importances_

    feature_names = [
        "Age", "Diabetes", "BloodPressureProblems", "KnownAllergies", 
        "AnyChronicDiseases", "AnyTransplants", "HistoryOfCancerInFamily", 
        "NumberOfMajorSurgeries", "BMI"
    ]

    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_importances
    }).sort_values(by="Importance", ascending=False)

    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(10,5))
    importance_df.plot(kind="barh", x="Feature", y="Importance", legend=False, ax=ax, cmap="viridis")
    plt.title("Random Forest Feature Importance")
    st.pyplot(fig)

# -------------------------------
# Feature Importance from XGB model
# -------------------------------
with st.expander("📊 XGBoost Feature Importance"):
    xgb_importances = xgb_model.feature_importances_

    # Create DataFrame
    importance_df_xgb = pd.DataFrame({
        "Feature": feature_names,
        "Importance": xgb_importances
    }).sort_values(by="Importance", ascending=False)

    # Plot using matplotlib
    fig2, ax2 = plt.subplots(figsize=(10,5))
    importance_df_xgb.plot(kind="barh", x="Feature", y="Importance", legend=False, ax=ax2, color='orange')
    plt.title("XGBoost Feature Importance")
    st.pyplot(fig2)
# -------------------------------
st.markdown("---")
st.markdown("Developed by Rutvik Mahadik")