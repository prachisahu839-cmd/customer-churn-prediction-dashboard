import streamlit as st
import joblib
import numpy as np

# ==========================
# Load Model
# ==========================
model = joblib.load("models/churn_model.pkl")

# ==========================
# Dashboard Title
# ==========================
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Predict whether a customer is likely to churn.")

# ==========================
# Customer Inputs
# ==========================
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0

    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    senior = 1 if senior == "Yes" else 0

    partner = st.selectbox("Partner", ["No", "Yes"])
    partner = 1 if partner == "Yes" else 0

    dependents = st.selectbox("Dependents", ["No", "Yes"])
    dependents = 1 if dependents == "Yes" else 0

    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

# ==========================
# Predict Button
# ==========================
if st.button("🔍 Predict Churn"):

    features = np.array([
        gender,
        senior,
        partner,
        dependents,
        tenure,

        # Remaining encoded features
        1,  # PhoneService
        1,  # MultipleLines
        1,  # InternetService
        1,  # OnlineSecurity
        1,  # OnlineBackup
        1,  # DeviceProtection
        1,  # TechSupport
        1,  # StreamingTV
        1,  # StreamingMovies
        1,  # Contract
        1,  # PaperlessBilling
        1,  # PaymentMethod

        monthly_charges,
        total_charges

    ]).reshape(1, -1)

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is not likely to Churn")

# ==========================
# Model Information
# ==========================
st.markdown("---")

st.subheader("📈 Model Information")

st.metric(
    label="Model Accuracy",
    value="79.56%"
)

st.write("""
**Algorithm Used:** Random Forest Classifier

**Dataset:** Telco Customer Churn Dataset

**Goal:** Predict whether a customer will leave the company.
""")
st.markdown("---")

st.subheader("📊 Churn Analysis")

st.image(
    "visuals/churn_distribution.png",
    caption="Customer Churn Distribution"
)

st.image(
    "visuals/confusion_matrix.png",
    caption="Confusion Matrix"
)

st.image(
    "visuals/feature_importance.png",
    caption="Feature Importance"
)
# ==========================
# Visual Analytics Section
# ==========================

st.markdown("---")

st.header("📊 Churn Analytics")

# Churn Distribution
st.subheader("Customer Churn Distribution")
st.image(
    "visuals/churn_distribution.png",
    caption="Distribution of Churn vs Non-Churn Customers",
    use_container_width=True
)

# Confusion Matrix
st.subheader("Confusion Matrix")
st.image(
    "visuals/confusion_matrix.png",
    caption="Model Performance Evaluation",
    use_container_width=True
)

# Feature Importance
st.subheader("Feature Importance")
st.image(
    "visuals/feature_importance.png",
    caption="Most Important Features Affecting Churn",
    use_container_width=True
)