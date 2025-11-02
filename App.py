import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Load trained model
model = joblib.load("final_gb_classifier.pkl")

# Page setup
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Light gradient background + colorful UI
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to bottom right, #ffffff, #f5f5f5);
            color: black;
            font-family: 'Poppins', sans-serif;
        }
        h1, h2, h3 {
            text-align: center;
            font-weight: 700;
            background: linear-gradient(90deg, #ff007f, #ff9900, #007bff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stButton>button {
            background: linear-gradient(90deg, #ff007f, #ff9900);
            color: white;
            border: none;
            border-radius: 10px;
            height: 3em;
            width: 12em;
            font-size: 16px;
            transition: 0.3s;
            box-shadow: 0px 0px 10px #ff80c0;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #007bff, #00c6ff);
            transform: scale(1.05);
        }
        .stSelectbox, .stNumberInput, .stSlider {
            color: black;
        }
        .css-1d391kg, .stMarkdown {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("💖 Customer Churn Prediction Dashboard")
st.markdown("### Step 2: Load & Predict")
st.success("✅ Gradient Boosting Model Loaded Successfully!")

# Input section
st.markdown("### 🎯 Enter Customer Details Below:")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col3:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

col4, col5 = st.columns(2)
with col4:
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
with col5:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=1200.0, value=100.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=2000.0)

# Prepare input data
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# One-hot encode
input_encoded = pd.get_dummies(input_data)
for col in model.feature_names_in_:
    if col not in input_encoded:
        input_encoded[col] = 0
input_encoded = input_encoded[model.feature_names_in_]

# Prediction
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_encoded)
    probability = model.predict_proba(input_encoded)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ This customer is likely to **Churn** (Probability: {probability:.2f})")
    else:
        st.success(f"🎉 This customer is likely to **Stay (Safe)** (Probability: {1 - probability:.2f})")

    # -------------------------------
    # Visualization Section (Telco data based)
    # -------------------------------
    df = pd.read_csv("Telco-Customer-Churn.csv")

    st.markdown("---")
    st.markdown("## 📊 Customer Insights Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(df, names='Churn', title="Customer Churn Distribution",
                      color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x='tenure', color='Churn', nbins=30,
                            title="Tenure Distribution by Churn",
                            color_discrete_sequence=['#ff007f', '#007bff'])
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.bar(df, x='Contract', color='Churn',
                      title="Contract Type vs Churn Rate",
                      color_discrete_sequence=['#00c6ff', '#ff9900'])
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.box(df, x='InternetService', y='MonthlyCharges', color='Churn',
                      title="Monthly Charges by Internet Service",
                      color_discrete_sequence=['#ff007f', '#007bff'])
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.caption("💻 Built with ❤️ using Streamlit | Model: Gradient Boosting Classifier")
