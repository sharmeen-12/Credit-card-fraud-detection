import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Financial Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# Load saved model artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
    
    st.title("💳 Real-Time Financial Fraud Detection System")
    st.write("This interactive system utilizes a **SMOTE + Cost-Sensitive XGBoost Model** to analyze transaction feature spaces and compute real-time fraud probability risks.")
    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Transaction Metadata")
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0)
        time_elapsed = st.number_input("Time Elapsed (Seconds)", min_value=0.0, value=3600.0, step=100.0)
        
        st.divider()
        st.subheader("Key PCA Features (V1 - V4)")
        v1 = st.slider("V1 (Principal Component)", -10.0, 10.0, -0.5, step=0.1)
        v2 = st.slider("V2 (Principal Component)", -10.0, 10.0, 0.2, step=0.1)
        v3 = st.slider("V3 (Principal Component)", -10.0, 10.0, 1.2, step=0.1)
        v4 = st.slider("V4 (Principal Component)", -10.0, 10.0, -0.8, step=0.1)

    with col2:
        st.subheader("Real-Time Prediction Engine")
        st.info("Adjust the transaction inputs on the left panel and click 'Analyze Transaction Risk' to execute model inference.")

        if st.button("🚀 Analyze Transaction Risk", use_container_width=True):
            # Scale raw amount feature using exact scaler
            scaled_amount = scaler.transform(np.array([[amount]]))[0][0]
            
            # Standardization for time feature (Dataset Mean = 94813.86, Std = 47488.15)
            scaled_time = (time_elapsed - 94813.86) / 47488.15

            # Construct full 30-feature input vector aligned with training sequence
            # Index 0-27: V1 to V28 | Index 28: scaled_amount | Index 29: scaled_time
            input_vector = np.zeros((1, 30))
            input_vector[0, 0] = v1
            input_vector[0, 1] = v2
            input_vector[0, 2] = v3
            input_vector[0, 3] = v4
            input_vector[0, 28] = scaled_amount
            input_vector[0, 29] = scaled_time

            # Model Inference
            prediction = model.predict(input_vector)[0]
            probability = model.predict_proba(input_vector)[0][1]

            st.write("---")
            # Operational Risk Threshold for Fraud Detection set to 15% (0.15)
            if prediction == 1 or probability >= 0.15:
                st.error(f"⚠️ **HIGH FRAUD RISK DETECTED**\n\n**Fraud Risk Probability:** {probability * 100:.2f}%")
                st.warning("Action Recommended: Flagged for fraud analyst review or trigger secondary multi-factor authentication (OTP).")
            else:
                st.success(f"✅ **TRANSACTION VERIFIED LEGITIMATE**\n\n**Fraud Risk Probability:** {probability * 100:.2f}%")
                st.info("Action: Transaction approved for automatic execution.")

except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.info("Please verify that 'fraud_model.pkl' and 'scaler.pkl' are present in the current working directory.")