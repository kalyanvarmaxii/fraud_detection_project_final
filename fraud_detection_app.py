import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model and feature columns
@st.cache_resource
def load_model():
    model, feature_cols = joblib.load("fraud_xgb_model.pkl")
    return model, feature_cols

model, feature_cols = load_model()

st.title("💳 Fraud Detection Demo – XGBoost Model")
st.write("This app uses the final trained XGBoost model to predict whether a transaction is fraudulent.")

st.sidebar.header("Enter Transaction Details")

# Inputs (match original features before get_dummies)
step = st.sidebar.number_input("Step (time step of transaction)", min_value=0, max_value=1_000_000, value=1)
txn_type = st.sidebar.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT"])
amount = st.sidebar.number_input("Amount", min_value=0.0, value=10000.0, step=100.0)
oldbalanceOrg = st.sidebar.number_input("Sender Old Balance", min_value=0.0, value=50000.0, step=100.0)
newbalanceOrig = st.sidebar.number_input("Sender New Balance", min_value=0.0, value=40000.0, step=100.0)
oldbalanceDest = st.sidebar.number_input("Receiver Old Balance", min_value=0.0, value=0.0, step=100.0)
newbalanceDest = st.sidebar.number_input("Receiver New Balance", min_value=0.0, value=10000.0, step=100.0)

# Recreate engineered features exactly like in Colab
balanceDiffOrig = oldbalanceOrg - newbalanceOrig
balanceDiffDest = oldbalanceDest - newbalanceDest

# Build single-row DataFrame before dummies
input_df = pd.DataFrame([{
    "step": step,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "balanceDiffOrig": balanceDiffOrig,
    "balanceDiffDest": balanceDiffDest,
    "type": txn_type
}])

st.subheader("Input Transaction Preview")
st.write(input_df)

# Apply same one-hot encoding as in training
input_encoded = pd.get_dummies(input_df, columns=["type"], drop_first=True)

# Align with training feature columns (missing columns → 0)
input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)

if st.button("🔍 Predict Fraud"):
    pred = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"🚨 Fraudulent Transaction\nProbability of Fraud: {proba*100:.2f}%")
    else:
        st.success(f"✅ Legitimate Transaction\nProbability of Fraud: {proba*100:.2f}%")
