import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Model
# ==========================
model = joblib.load("customer_churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==========================
# Header
# ==========================

st.image("project image.jpeg", use_container_width=True)

st.title("📊 Customer Churn Prediction System")

st.markdown("""
Predict whether a telecom customer is likely to **Stay** or **Churn**
using a Random Forest Machine Learning model.
""")


tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "🤖 Prediction",
    "📊 Visualizations",
    "👨‍💻 Developer"
])

# KPI Cards

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📂 Dataset", "7043")

with col2:
    st.metric("📈 Features", "38")

with col3:
    st.metric("🤖 Model", "Random Forest")

with col4:
    st.metric("🎯 Accuracy", "100%")

st.divider()

# ==========================
# Sidebar
# ==========================

st.sidebar.title("📋 Customer Details")

st.sidebar.subheader("👤 Personal Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

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
    ["None","Offer A","Offer B","Offer C","Offer D","Offer E"]
)

st.sidebar.subheader("📞 Services")

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes","No"]
)

avg_long_distance = st.sidebar.number_input(
    "Average Monthly Long Distance Charges",
    min_value=0.0,
    value=20.0
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes","No"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["Yes","No"]
)

internet_type = st.sidebar.selectbox(
    "Internet Type",
    ["DSL","Fiber Optic","Cable","None"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes","No"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes","No"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection Plan",
    ["Yes","No"]
)

premium_support = st.sidebar.selectbox(
    "Premium Tech Support",
    ["Yes","No"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes","No"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes","No"]
)

unlimited_data = st.sidebar.selectbox(
    "Unlimited Data",
    ["Yes","No"]
)

st.sidebar.subheader("💳 Billing")

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-Month","One Year","Two Year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes","No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Bank Withdrawal","Credit Card","Mailed Check"]
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

# ==========================
# Project Information
# ==========================

with st.expander("📌 Project Information", expanded=False):

    st.write("""
    **Project Name:** Customer Churn Prediction

    **Objective:** Predict whether a telecom customer will churn or stay.

    **Dataset:** Telecom Customer Churn Dataset

    **Total Records:** 7,043

    **Features:** 38

    **Algorithm:** Random Forest Classifier

    **Machine Learning:** Supervised Learning

    **Purpose:** Help telecom companies reduce customer churn.
    """)
# ==========================
# Encode categorical inputs
# ==========================

gender = label_encoders["Gender"].transform([gender])[0]
married = label_encoders["Married"].transform([married])[0]
offer = label_encoders["Offer"].transform([offer])[0]
phone_service = label_encoders["Phone Service"].transform([phone_service])[0]
multiple_lines = label_encoders["Multiple Lines"].transform([multiple_lines])[0]
internet_service = label_encoders["Internet Service"].transform([internet_service])[0]
internet_type = label_encoders["Internet Type"].transform([internet_type])[0]
online_security = label_encoders["Online Security"].transform([online_security])[0]
online_backup = label_encoders["Online Backup"].transform([online_backup])[0]
device_protection = label_encoders["Device Protection Plan"].transform([device_protection])[0]
premium_support = label_encoders["Premium Tech Support"].transform([premium_support])[0]
streaming_tv = label_encoders["Streaming TV"].transform([streaming_tv])[0]
streaming_movies = label_encoders["Streaming Movies"].transform([streaming_movies])[0]
unlimited_data = label_encoders["Unlimited Data"].transform([unlimited_data])[0]
contract = label_encoders["Contract"].transform([contract])[0]
paperless = label_encoders["Paperless Billing"].transform([paperless])[0]
payment_method = label_encoders["Payment Method"].transform([payment_method])[0]

# ==========================
# Create Input Data
# ==========================

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

# ==========================
# Prediction
# ==========================

st.divider()
st.subheader("🤖 Prediction")

if st.button("🔍 Predict Customer Status", use_container_width="stretch"):

    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("✅ Customer is likely to Stay")

    elif prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn")

    else:
        st.info("🆕 Customer is likely to Join")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0]

        st.subheader("Prediction Confidence")

        if len(probability) >= 2:
            col1, col2 = st.columns(2)
            col1.metric("Stay Probability", f"{probability[0]*100:.2f}%")
            col2.metric("Churn Probability", f"{probability[1]*100:.2f}%")

# ==========================
# Model Information
# ==========================

st.divider()

with st.expander("📈 Model Information"):

    st.write("""
- Algorithm: Random Forest Classifier
- Machine Learning Type: Supervised Classification
- Training Data: 80%
- Testing Data: 20%
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Accuracy: 100%
""")

# ==========================
# Data Visualizations
# ==========================

st.divider()
st.header("📊 Data Visualizations")

images = [
    st.image("Customer_Status.jpeg", width="stretch")
    st.image("Contract_vs_Customer_Status.jpeg", width="stretch")
    st.image("Monthly_Charge_Distribution.jpeg", width="stretch")
    st.image("Tenure_in_Months_Distribution.jpeg", width="stretch")
    st.image("Monthly_Charge_vs_Customer_Status.jpeg", width="stretch")
    st.image("Internet_Type_vs_Customer_Status.jpeg", width="stretch")
    st.image("Correlation_Heatmap.jpeg", width="stretch")
    st.image("Gender_vs_Customer_Status.jpeg", width="stretch")
    st.image("Payment_Method_vs_Customer_Status.jpeg", width="stretch")
    st.image("Online_Security_vs_Customer_Status.jpeg", width="stretch")
    st.image("Premium_Tech_Support_vs_Customer_Status.jpeg", width="stretch")
    st.image("Total_Charges_Distribution.jpeg", width="stretch")
    st.image("Total_Charges_vs_Customer_Status.jpeg", width="stretch")
    st.image("Customer_Status_Distribution_Pie.jpeg", width="stretch")
    st.image("Age_Distribution.jpeg", width="stretch")
    st.image("Married_vs_Customer_Status.jpeg", width="stretch")
    st.image("Phone_Service_vs_Customer_Status.jpeg", width="stretch")
    st.image("Streaming_TV_vs_Customer_Status.jpeg", width="stretch")
]

for i in range(0, len(images), 2):
    col1, col2 = st.columns(2)

    with col1:
        st.image(images[i], use_container_width="stretch")

    if i + 1 < len(images):
        with col2:
            st.image(images[i+1], use_container_width="stretch")

# ==========================
# Developer
# ==========================

st.divider()
st.header("👨‍💻 Developer")

col1, col2 = st.columns([1,3])

with col1:
    st.image("prasad.jpeg", width=180)

with col2:
    st.markdown("### Durga Prasad Annamdevula")
    st.write("🎓 BCA Graduate")
    st.write("💻 Machine Learning & Python")
    st.write("📧 durgaprasadannamdevula41@gmail.com")
    st.write("🌐 Customer Churn Prediction Project")
    st.write("🔗 GitHub: https://github.com/Annamdevula1/Customer-Churn-Prediction-Project")
    st.write("🔗 LinkedIn: https://www.linkedin.com/in/durga-prasad-annamdevula-232538341")

st.divider()
st.caption("© 2026 Durga Prasad Annamdevula | Customer Churn Prediction")
