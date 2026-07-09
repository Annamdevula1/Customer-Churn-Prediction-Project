import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Load Model
# ==========================================
model = joblib.load("customer_churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS
# ==========================================
st.markdown("""
<style>

/* Main Background */
.stApp{
    background-color:#F5F7FA;
}

/* Hero Title */
.hero-title{
    font-size:42px;
    font-weight:bold;
    color:white;
    text-align:center;
    padding-top:10px;
}

.hero-subtitle{
    font-size:20px;
    color:white;
    text-align:center;
    padding-bottom:15px;
}

/* Header Box */
.header-box{
    background:linear-gradient(90deg,#1565C0,#42A5F5);
    border-radius:18px;
    padding:20px;
    margin-bottom:20px;
}

/* Section Heading */
.section-title{
    color:#1565C0;
    font-size:28px;
    font-weight:bold;
    margin-top:10px;
    margin-bottom:15px;
}

/* Metric Cards */
[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:15px;
    border:1px solid #E0E0E0;
    box-shadow:0px 2px 10px rgba(0,0,0,0.08);
}

/* Button */
.stButton>button{
    background:#1565C0;
    color:white;
    border-radius:10px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#E3F2FD;
}

section[data-testid="stSidebar"] h1{
    color:#0D47A1;
    font-size:28px;
    font-weight:bold;
}

section[data-testid="stSidebar"] h2{
    color:#1565C0;
    font-size:22px;
    font-weight:bold;
}

section[data-testid="stSidebar"] h3{
    color:#2E7D32;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Header
# ==========================================

st.image("project image.jpeg", use_container_width=True)

st.markdown("""
<div class="header-box">

<div class="hero-title">
📊 Customer Churn Prediction System
</div>

<div class="hero-subtitle">
Predict whether a telecom customer is likely to
<b>Stay</b> or <b>Churn</b> using
Random Forest Machine Learning.
</div>

</div>
""", unsafe_allow_html=True)

# ==========================================
# KPI Cards
# ==========================================

col1,col2,col3,col4=st.columns(4)

with col1:
    st.metric("📂 Dataset","7043")

with col2:
    st.metric("📈 Features","38")

with col3:
    st.metric("🤖 Algorithm","Random Forest")

with col4:
    st.metric("🎯 Accuracy","100%")

st.divider()

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("📋 Navigation")

st.sidebar.markdown("### 🏠 Home")
st.sidebar.markdown("### 👤 Customer Form")
st.sidebar.markdown("### 📊 Prediction")
st.sidebar.markdown("### 📈 Data Visualizations")
st.sidebar.markdown("### 📌 Project Information")
st.sidebar.markdown("### 🤖 Model Information")
st.sidebar.markdown("### 👨‍💻 Developer")

st.sidebar.info(
"""
Customer Churn Prediction

Developed using
Random Forest Classifier

Machine Learning Project
"""
)

st.markdown(
"<h2 class='section-title'>📝 Customer Information</h2>",
unsafe_allow_html=True
)
# ==========================================
# Customer Input Form
# ==========================================

with st.form("customer_form"):

    st.subheader("👤 Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=0
        )

    with col2:

        number_referrals = st.number_input(
            "Number of Referrals",
            min_value=0,
            max_value=20,
            value=0
        )

        tenure = st.number_input(
            "Tenure in Months",
            min_value=0,
            max_value=100,
            value=12
        )

        offer = st.selectbox(
            "Offer",
            [
                "None",
                "Offer A",
                "Offer B",
                "Offer C",
                "Offer D",
                "Offer E"
            ]
        )

    st.divider()

    st.subheader("📞 Service Details")

    col3, col4 = st.columns(2)

    with col3:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        avg_long_distance = st.number_input(
            "Average Monthly Long Distance Charges",
            min_value=0.0,
            value=20.0
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["Yes", "No"]
        )

        internet_type = st.selectbox(
            "Internet Type",
            [
                "DSL",
                "Fiber Optic",
                "Cable",
                "None"
            ]
        )

    with col4:

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No"]
        )

        device_protection = st.selectbox(
            "Device Protection Plan",
            ["Yes", "No"]
        )

        premium_support = st.selectbox(
            "Premium Tech Support",
            ["Yes", "No"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No"]
        )

        unlimited_data = st.selectbox(
            "Unlimited Data",
            ["Yes", "No"]
        )
    st.divider()

    st.subheader("💳 Billing Information")

    col5, col6 = st.columns(2)

    with col5:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-Month",
                "One Year",
                "Two Year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Bank Withdrawal",
                "Credit Card",
                "Mailed Check"
            ]
        )

        monthly_charge = st.number_input(
            "Monthly Charge",
            min_value=0.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0
        )

    with col6:

        total_refunds = st.number_input(
            "Total Refunds",
            min_value=0.0,
            value=0.0
        )

        total_extra_data = st.number_input(
            "Total Extra Data Charges",
            min_value=0.0,
            value=0.0
        )

        total_long_distance = st.number_input(
            "Total Long Distance Charges",
            min_value=0.0,
            value=200.0
        )

        total_revenue = st.number_input(
            "Total Revenue",
            min_value=0.0,
            value=1200.0
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Customer Status",
        use_container_width=True
    )

# ==========================================
# Project Information
# ==========================================

with st.expander("📌 Project Information", expanded=False):

    st.write("""
**Project Name:** Customer Churn Prediction

**Objective:** Predict whether a telecom customer will churn or stay.

**Dataset:** Telecom Customer Churn Dataset

**Total Records:** 7,043

**Features:** 38

**Algorithm:** Random Forest Classifier

**Machine Learning Type:** Supervised Learning

**Purpose:** Help telecom companies identify customers who are likely to churn and improve customer retention.
""")
# ==========================================
# Encode Categorical Inputs
# ==========================================

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

# ==========================================
# Create Input Data
# ==========================================

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

# ==========================================
# Prediction
# ==========================================

if submitted:

    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(input_data)

    st.divider()
    st.markdown(
        "<h2 class='section-title'>🎯 Prediction Result</h2>",
        unsafe_allow_html=True
    )

    if prediction[0] == 0:
        st.success("✅ Customer is likely to Stay")

    elif prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn")

    else:
        st.info("🆕 Customer is likely to Join")

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)[0]

        st.subheader("Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Stay Probability",
                f"{probability[0]*100:.2f}%"
            )

        with col2:
            st.metric(
                "Churn Probability",
                f"{probability[1]*100:.2f}%"
            )
# ==========================================
# Model Information
# ==========================================

st.divider()

with st.expander("📈 Model Information", expanded=False):

    st.markdown("""
### 🤖 Model Details

- **Algorithm:** Random Forest Classifier
- **Machine Learning Type:** Supervised Classification
- **Training Split:** 80%
- **Testing Split:** 20%

### 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score

### 🎯 Model Accuracy

**100%**
""")

# ==========================================
# Data Visualizations
# ==========================================

st.divider()

st.markdown(
"<h2 class='section-title'>📊 Data Visualizations</h2>",
unsafe_allow_html=True
)

images = [

    "Customer_Status.jpeg",
    "Contract_vs_Customer_Status.jpeg",
    "Monthly_Charge_Distribution.jpeg",
    "Tenure_in_Months_Distribution.jpeg",
    "Monthly_Charge_vs_Customer_Status.jpeg",
    "Internet_Type_vs_Customer_Status.jpeg",
    "Correlation_Heatmap.jpeg",
    "Gender_vs_Customer_Status.jpeg",
    "Payment_Method_vs_Customer_Status.jpeg",
    "Online_Security_vs_Customer_Status.jpeg",
    "Premium_Tech_Support_vs_Customer_Status.jpeg",
    "Total_Charges_Distribution.jpeg",
    "Total_Charges_vs_Customer_Status.jpeg",
    "Customer_Status_Distribution_Pie.jpeg",
    "Age_Distribution.jpeg",
    "Married_vs_Customer_Status.jpeg",
    "Phone_Service_vs_Customer_Status.jpeg",
    "Streaming_TV_vs_Customer_Status.jpeg"

]

for i in range(0, len(images), 2):

    col1, col2 = st.columns(2)

    with col1:
        st.image(images[i],use_container_width=True)

    if i + 1 < len(images):

        with col2:
            st.image(images[i + 1],use_container_width=True)

# ==========================================
# Developer Section
# ==========================================

st.divider()

st.markdown(
"<h2 class='section-title'>👨‍💻 Developer</h2>",
unsafe_allow_html=True
)

col1, col2 = st.columns([1,3])

with col1:

    st.image(
        "prasad.jpeg",
        width=220
    )

with col2:

    st.markdown("## Durga Prasad Annamdevula")

    st.write("🎓 **BCA Graduate**")

    st.write("💻 Machine Learning | Python | Data Science")

    st.write("📧 durgaprasadannamdevula41@gmail.com")

    st.write("🌐 Customer Churn Prediction Project")

    st.write(
        "🔗 GitHub: https://github.com/Annamdevula1/Customer-Churn-Prediction-Project"
    )

    st.write(
        "🔗 LinkedIn: https://www.linkedin.com/in/durga-prasad-annamdevula-232538341"
    )

st.divider()

# ==========================================
# Footer
# ==========================================

st.markdown(
"""
<div style='text-align:center;'>

### 📊 Customer Churn Prediction System

Developed using ❤️ with Streamlit & Machine Learning

© 2026 Durga Prasad Annamdevula

</div>
""",
unsafe_allow_html=True
)
