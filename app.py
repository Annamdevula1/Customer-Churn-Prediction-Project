import streamlit as st
import pandas as pd
import joblib

# Load Model and Label Encoders
model = joblib.load("customer_churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)
st.image("project image.jpeg", width="stretch")

st.title("📊 Customer Churn Prediction System")

st.markdown("Predict whether a customer is likely to stay or churn.")
st.sidebar.title("Customer Details")

# ==========================
# Project Information
# ==========================

st.markdown("---")

st.subheader("📌 Project Information")


st.write("""
**Project Name:** Customer Churn Prediction
**Objective:** Predict whether a telecom customer will churn or stay.
**Dataset:** Telecom Customer Churn Dataset
**Total Records:** 7,043
**Features:** 38
**Target Variable:** Customer Status
**Programming Language:** Python
**Machine Learning Model:** Random Forest Classifier


**Libraries Used:**
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

**Development Environment:** Google Colab

**Version Control:** GitHub

**Purpose:** Help telecom companies identify customers who are likely to churn and support customer retention strategies.
""")



# Customer Information

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
# Encode categorical inputs using saved LabelEncoders

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

# Create Input DataFrame

input_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Married": [married],
    "Number of Dependents": [dependents],
    "Number of Referrals": [number_referrals],
    "Tenure in Months": [tenure],
    "Offer": [offer],
    "Phone Service": [phone_service],
    "Avg Monthly Long Distance Charges": [avg_long_distance],
    "Multiple Lines": [multiple_lines],
    "Internet Service": [internet_service],
    "Internet Type": [internet_type],
    "Online Security": [online_security],
    "Online Backup": [online_backup],
    "Device Protection Plan": [device_protection],
    "Premium Tech Support": [premium_support],
    "Streaming TV": [streaming_tv],
    "Streaming Movies": [streaming_movies],
    "Unlimited Data": [unlimited_data],
    "Contract": [contract],
    "Paperless Billing": [paperless],
    "Payment Method": [payment_method],
    "Monthly Charge": [monthly_charge],
    "Total Charges": [total_charges],
    "Total Refunds": [total_refunds],
    "Total Extra Data Charges": [total_extra_data],
    "Total Long Distance Charges": [total_long_distance],
    "Total Revenue": [total_revenue]
})

# Prediction

if st.button("Predict Customer Status"):
   st.write("Model expects:", model.feature_names_in_.tolist())
   st.write("Input columns:", input_data.columns.tolist())

input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

prediction = model.predict(input_data)



   
if prediction[0] == 0:
        st.success("✅ Customer is likely to Stay")

elif prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn")

else:
    st.info("🆕 Customer is likely to Join") 
    st.markdown("---")
st.subheader("🤖 Model Information")

st.write("""
**Algorithm:** Random Forest Classifier

**Machine Learning Type:** Supervised Learning (Classification)

**Training Data:** 80%

**Testing Data:** 20%

**Target Variable:** Customer Status

**Evaluation Metrics:** Accuracy, Precision, Recall, and F1-Score

**Model Accuracy:** 100% (on the test dataset)

**Prediction Output:** Predicts whether a customer is likely to Stay or Churn based on the provided customer information.
""")

        # ---------------- Data Visualizations ----------------

st.header("📊 Data Visualizations")

st.subheader("Customer Status")
from PIL import Image

img = Image.open("Customer_Status.jpeg")
st.image(img)

st.subheader("Contract vs Customer Status")
st.image('Contract_vs_Customer_Status.jpeg')

st.subheader("Monthly Charge Distribution")
st.image('Monthly_Charge_Distribution.jpeg')

st.subheader("Tenure in Months Distribution")
st.image('Tenure_in_Months_Distribution.jpeg')

st.subheader("Monthly Charge vs Customer Status")
st.image('Monthly_Charge_vs_Customer_Status.jpeg')

st.subheader("Internet Type vs Customer Status")
st.image('Internet_Type_vs_Customer_Status.jpeg')

st.subheader("Correlation Heatmap")
st.image('Correlation_Heatmap.jpeg')

st.subheader("Gender vs Customer Status")
st.image('Gender_vs_Customer_Status.jpeg')

st.subheader("Payment Method vs Customer Status")
st.image('Payment_Method_vs_Customer_Status.jpeg')

st.subheader("Online Security vs Customer Status")
st.image('Online_Security_vs_Customer_Status.jpeg')

st.subheader("Premium Tech Support vs Customer Status")
st.image('Premium_Tech_Support_vs_Customer_Status.jpeg')

st.subheader("Total Charges Distribution")
st.image('Total_Charges_Distribution.jpeg')

st.subheader("Total Charges vs Customer Status")
st.image('Total_Charges_vs_Customer_Status.jpeg')

st.subheader("Customer Status Distribution (Pie Chart)")
st.image('Customer_Status_Distribution_Pie.jpeg')

st.subheader("Age Distribution")
st.image('Age_Distribution.jpeg')

st.subheader("Married vs Customer Status")
st.image('Married_vs_Customer_Status.jpeg')

st.subheader("Phone Service vs Customer Status")
st.image('Phone_Service_vs_Customer_Status.jpeg')

st.subheader("Streaming TV vs Customer Status")
st.image('Streaming_TV_vs_Customer_Status.jpeg')
# ---------------- Footer ----------------

st.markdown("---")

st.markdown("---")
st.subheader("👨‍💻 Developer")

col1, col2 = st.columns([1, 3])

with col1:
    st.image("prasad.jpeg", width=150)  

with col2:
    
  st.markdown("### 👨‍💻 Durga Prasad Annamdevula")

  st.markdown("🎓 Recently Graduate And Fresher")

  st.markdown("📧 **Email:**durgaprasadannamdevula41@gmail.com")

  st.markdown("💻 **GitHub:** https://github.com/Annamdevula1/Customer-Churn-Prediction-Project.git")

  st.markdown("🔗 **LinkedIn:** https://www.linkedin.com/in/durga-prasad-annamdevula-232538341")

  st.markdown("🌐 **Project:** Customer Churn Prediction using Machine Learning")

  st.markdown("© 2026 Durga Prasad Annamdevula. All Rights Reserved.")





