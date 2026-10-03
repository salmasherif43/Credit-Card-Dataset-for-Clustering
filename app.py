import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved scaler and model
@st.cache_resource
def load_assets():
    model = joblib.load('kmeans_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

kmeans, scaler = load_assets()

# Streamlit App UI
st.set_page_config(page_title='Credit Card Customer Segmentation', layout='wide')
st.title('💳 Credit Card Customer Segmentation App')
st.write("Enter a customer's details below to identify which segmentation cluster they belong to.")

# Sidebar or form for user inputs
st.subheader("Customer Feature Input")

# Define our features
features = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']

# Layout inputs in columns
cols = st.columns(3)
inputs = {}

for i, col_name in enumerate(features):
    with cols[i % 3]:
        # Use reasonable default values based on common dataset distributions
        inputs[col_name] = st.number_input(f"{col_name}", value=0.0, step=10.0 if 'LIMIT' in col_name or 'BALANCE' in col_name or 'PAYMENTS' in col_name or 'PURCHASES' in col_name else 0.1)

# Predict button
if st.button('Segment Customer', type='primary'):
    # Prepare the input DataFrame
    input_df = pd.DataFrame([inputs])
    
    # Scale features
    scaled_inputs = scaler.transform(input_df)
    
    # Predict cluster
    cluster_id = kmeans.predict(scaled_inputs)[0]
    
    st.success(f"🎯 This customer belongs to **Cluster {{cluster_id}}**")
    
    # Cluster interpretation
    st.subheader("Cluster Insights:")
    if cluster_id == 0:
        st.info("**High Spenders (Power Shoppers):** High purchases, high one-off purchases, and very high credit limit. Active spenders.")
    elif cluster_id == 1:
        st.info("**Frugal/Inactive Users:** Low balances, low overall purchases, and lower credit limits.")
    elif cluster_id == 2:
        st.info("**Balanced/Installment Users:** Moderate balances, moderate purchases, and highly active with installment purchases.")
    elif cluster_id == 3:
        st.info("**Cash Advance Users:** High cash advance transactions with higher balances but lower direct purchase volume.")
