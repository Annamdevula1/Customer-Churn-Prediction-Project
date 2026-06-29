import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("trained_model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)
st.image("project_banner.png", use_container_width=True)

st.title("📊 Customer Churn Prediction System")
st.markdown("Predict whether a customer is likely to stay or churn.")

st.sidebar.title("Customer Details")

# Customer Information

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

married = st.sidebar.selectbox(
    "Married",
    ["Yes", "No"]
)

dependents = st.sidebar.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=0
)

number_referrals = st.sidebar.number_input(
    "Number of Referrals",
    min_value=0,
    max_value=20,
    value=0
)

tenure = st.sidebar.number_input(
    "Tenure in Months",
    min_value=0,
    max_value=100,
    value=12
)

offer = st.sidebar.selectbox(
    "Offer",
    ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

avg_long_distance = st.sidebar.number_input(
    "Average Monthly Long Distance Charges",
    min_value=0.0,
    value=20.0
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["Yes", "No"]
)

internet_type = st.sidebar.selectbox(
    "Internet Type",
    ["DSL", "Fiber Optic", "Cable", "None"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection Plan",
    ["Yes", "No"]
)

premium_support = st.sidebar.selectbox(
    "Premium Tech Support",
    ["Yes", "No"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

unlimited_data = st.sidebar.selectbox(
    "Unlimited Data",
    ["Yes", "No"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-Month", "One Year", "Two Year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Bank Withdrawal",
        "Credit Card",
        "Mailed Check"
    ]
)

monthly_charge = st.sidebar.number_input(
    "Monthly Charge",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

total_refunds = st.sidebar.number_input(
    "Total Refunds",
    min_value=0.0,
    value=0.0
)

total_extra_data = st.sidebar.number_input(
    "Total Extra Data Charges",
    min_value=0.0,
    value=0.0
)

total_long_distance = st.sidebar.number_input(
    "Total Long Distance Charges",
    min_value=0.0,
    value=200.0
)

total_revenue = st.sidebar.number_input(
    "Total Revenue",
    min_value=0.0,
    value=1200.0
)
# Convert Yes/No values
yes_no = {"Yes": 1, "No": 0}

gender = 1 if gender == "Male" else 0
married = yes_no[married]
phone_service = yes_no[phone_service]
multiple_lines = yes_no[multiple_lines]
internet_service = yes_no[internet_service]
online_security = yes_no[online_security]
online_backup = yes_no[online_backup]
device_protection = yes_no[device_protection]
premium_support = yes_no[premium_support]
streaming_tv = yes_no[streaming_tv]
streaming_movies = yes_no[streaming_movies]
unlimited_data = yes_no[unlimited_data]
paperless = yes_no[paperless]

# Create Input DataFrame
input_data = pd.DataFrame({
    "Gender":[gender],
    "Age":[age],
    "Married":[married],
    "Number of Dependents":[dependents],
    "Number of Referrals":[number_referrals],
    "Tenure in Months":[tenure],
    "Offer":[offer],
    "Phone Service":[phone_service],
    "Avg Monthly Long Distance Charges":[avg_long_distance],
    "Multiple Lines":[multiple_lines],
    "Internet Service":[internet_service],
    "Internet Type":[internet_type],
    "Online Security":[online_security],
    "Online Backup":[online_backup],
    "Device Protection Plan":[device_protection],
    "Premium Tech Support":[premium_support],
    "Streaming TV":[streaming_tv],
    "Streaming Movies":[streaming_movies],
    "Unlimited Data":[unlimited_data],
    "Contract":[contract],
    "Paperless Billing":[paperless],
    "Payment Method":[payment_method],
    "Monthly Charge":[monthly_charge],
    "Total Charges":[total_charges],
    "Total Refunds":[total_refunds],
    "Total Extra Data Charges":[total_extra_data],
    "Total Long Distance Charges":[total_long_distance],
    "Total Revenue":[total_revenue]
})

# Encode categorical columns
input_data = pd.get_dummies(input_data)

# Prediction
if st.button("Predict Customer Status"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn.")
    else:
        st.success("✅ Customer is likely to Stay.")

# Graph Section
st.markdown("---")
st.header("📈 Project Visualizations")

st.image("project_banner.png", use_container_width=True)

st.image("customer_status.png", caption="Customer Status Distribution")

st.image("contract_distribution.png", caption="Contract Distribution")

st.image("payment_method.png", caption="Payment Method")

st.image("internet_service.png", caption="Internet Service")

st.image("monthly_charge_distribution.png", caption="Monthly Charges")

st.image("tenure_distribution.png", caption="Tenure Distribution")

# Footer
st.markdown("---")
st.markdown(
    "<center><h4>Customer Churn Prediction using Machine Learning</h4></center>",
    unsafe_allow_html=True
)
