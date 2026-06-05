import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

css_path = BASE_DIR / "style.css"
model_path = PROJECT_DIR / "models" / "random_forest_churn_model.pkl"
columns_path = PROJECT_DIR / "models" / "model_columns.pkl"


# -----------------------------
# Load CSS
# -----------------------------
with open(css_path, "r", encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


# -----------------------------
# Load Model
# -----------------------------
with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(columns_path, "rb") as file:
    model_columns = pickle.load(file)


# -----------------------------
# Hero Section
# -----------------------------
st.markdown(
    """
    <div class="hero-section">
        <div>
            <p class="eyebrow">Machine Learning Project</p>
            <h1>Customer Churn Risk Predictor</h1>
            <p class="hero-text">
                Predict whether a customer is likely to churn using customer behavior,
                purchase activity, account manager status, and service usage details.
            </p>
        </div>
        <div class="hero-badge">
            <span>📊</span>
            <p>Random Forest Model</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([1, 1], gap="large")


# -----------------------------
# Input Section
# -----------------------------
with left_col:
    st.markdown(
        """
        <div class="custom-card-title">
            <h3>🧾 Enter Customer Details</h3>
            <p>Fill in the customer details below to estimate churn risk.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("churn_form"):
        age = st.slider(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35
        )

        total_purchase = st.number_input(
            "Total Purchase Amount",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

        account_manager = st.selectbox(
            "Does the customer have an account manager?",
            ["Yes", "No"]
        )

        years = st.number_input(
            "Years with Company",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.1
        )

        num_sites = st.slider(
            "Number of Sites",
            min_value=1,
            max_value=50,
            value=8
        )

        submitted = st.form_submit_button("Predict Churn Risk")


# -----------------------------
# Prepare Input
# -----------------------------
account_manager_value = 1 if account_manager == "Yes" else 0

input_data = pd.DataFrame({
    "Age": [age],
    "Total_Purchase": [total_purchase],
    "Account_Manager": [account_manager_value],
    "Years": [years],
    "Num_Sites": [num_sites]
})

input_data = input_data.reindex(columns=model_columns, fill_value=0)


# -----------------------------
# Result Section
# -----------------------------
with right_col:
    st.markdown(
        """
        <div class="custom-card-title">
            <h3>🔍 Prediction Result</h3>
            <p>The model prediction will appear here.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if submitted:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        probability_percentage = probability * 100

        if probability >= 0.70:
            risk_level = "High Risk"
            risk_class = "high-risk"
            risk_message = "This customer may need immediate retention action."
            icon = "🚨"
        elif probability >= 0.40:
            risk_level = "Medium Risk"
            risk_class = "medium-risk"
            risk_message = "This customer should be monitored carefully."
            icon = "⚠️"
        else:
            risk_level = "Low Risk"
            risk_class = "low-risk"
            risk_message = "This customer appears stable."
            icon = "✅"

        prediction_text = (
            "Customer is likely to churn"
            if prediction == 1
            else "Customer is not likely to churn"
        )

        st.markdown(
            f"""
            <div class="prediction-box {risk_class}">
                <div class="prediction-icon">{icon}</div>
                <div>
                    <h2>{prediction_text}</h2>
                    <p>{risk_level}</p>
                </div>
            </div>

            <div class="probability-card">
                <p>Churn Probability</p>
                <h1>{probability_percentage:.2f}%</h1>
                <div class="progress-bg">
                    <div class="progress-fill" style="width: {probability_percentage}%;"></div>
                </div>
            </div>

            <div class="insight-box">
                <h4>Business Recommendation</h4>
                <p>{risk_message}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            """
            <div class="empty-state">
                <h2>👈 Enter customer details</h2>
                <p>Your prediction result will appear here after clicking the button.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------
# Input Summary
# -----------------------------
st.markdown("---")

st.markdown("### 📌 Customer Input Summary")

display_data = input_data.copy()
display_data["Account_Manager"] = display_data["Account_Manager"].map({
    1: "Yes",
    0: "No"
})

st.dataframe(display_data, use_container_width=True)


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="footer">
        <p>Built with Python, Scikit-learn, Random Forest, and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)