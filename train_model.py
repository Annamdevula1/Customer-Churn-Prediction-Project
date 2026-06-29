# ==========================================
# PART 1 : Import Libraries & Preprocessing
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("telecom_customer_churn 1.csv")
df["Offer"] = df["Offer"].fillna("None")

# Drop Unnecessary Columns
df.drop(["Customer ID", "Churn Category", "Churn Reason"], axis=1, inplace=True)

# Label Encoding
label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

print("Part 1 Completed Successfully!")
# ==========================================
# PART 2 : Train the Model
# ==========================================

# Define Features (X) and Target (y)
X = df.drop("Customer Status", axis=1)
y = df["Customer Status"]

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the Random Forest Model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

print("✅ Part 2 Completed Successfully!")
# ==========================================
# PART 3 : Evaluate & Save the Model
# ==========================================

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model and Label Encoders
joblib.dump(model, "customer_churn_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n✅ Model saved as customer_churn_model.pkl")
print("✅ Label Encoders saved as label_encoders.pkl")

# Download Files (Google Colab)
from google.colab import files

files.download("customer_churn_model.pkl")
# ==========================================
# PART 1 : Import Libraries & Preprocessing
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("telecom_customer_churn 1.csv")
df["Offer"] = df["Offer"].fillna("None")

# Drop Unnecessary Columns
df.drop(["Customer ID", "Churn Category", "Churn Reason"], axis=1, inplace=True)

# Label Encoding
label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

print("Part 1 Completed Successfully!")
# ==========================================
# PART 2 : Train the Model
# ==========================================

# Define Features (X) and Target (y)
X = df.drop("Customer Status", axis=1)
y = df["Customer Status"]

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the Random Forest Model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

print("✅ Part 2 Completed Successfully!")
# ==========================================
# PART 3 : Evaluate & Save the Model
# ==========================================

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model and Label Encoders
joblib.dump(model, "customer_churn_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n✅ Model saved as customer_churn_model.pkl")
print("✅ Label Encoders saved as label_encoders.pkl")

# Download Files (Google Colab)
from google.colab import files

files.download("customer_churn_model.pkl")
files.download("label_encoders.pkl")
