"""
Credit Risk Prediction System - Streamlit Application
======================================================
Loads the saved end-to-end pipeline (preprocessing + tuned Gradient Boosting
classifier) and predicts HIGH RISK / LOW RISK for a loan applicant.
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

from feature_engineering import add_engineered_features

MODEL_PATH = "models/credit_risk_model.joblib"
METADATA_PATH = "models/metadata.json"

st.set_page_config(page_title="Credit Risk Prediction System", page_icon="💳", layout="centered")

st.title("💳 Credit Risk Prediction System")
st.write(
    "Predicts whether a loan applicant is **High Risk** or **Low Risk** using a "
    "Gradient Boosting model trained on historical applicant data. Risk is derived "
    "from the dataset's continuous RiskScore (median split); see the project README "
    "for the full methodology."
)

# ---------------------------------------------------------------------------
# Load model & metadata with error handling
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
    st.error(
        "Model file(s) not found. Please run `python build.py` first to train "
        "and save the model before launching this app."
    )
    st.stop()

try:
    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
except Exception as e:
    st.error(f"Failed to load the model or metadata: {e}")
    st.stop()

numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
numeric_ranges = metadata["numeric_ranges"]
categorical_options = metadata["categorical_options"]

# Engineered features are computed automatically and are NOT collected directly
ENGINEERED_NUMERIC = {"TotalObligationRatio"}
ENGINEERED_CATEGORICAL = {"AgeGroup"}

raw_numeric_features = [c for c in numeric_features if c not in ENGINEERED_NUMERIC]
raw_categorical_features = [c for c in categorical_features if c not in ENGINEERED_CATEGORICAL]

st.header("Applicant Information")

user_input = {}

with st.form("applicant_form"):
    st.subheader("Personal Details")
    col1, col2 = st.columns(2)
    with col1:
        user_input["Age"] = st.number_input(
            "Age", min_value=18, max_value=100,
            value=int(numeric_ranges["Age"]["median"]), step=1)
        user_input["MaritalStatus"] = st.selectbox("Marital Status", categorical_options["MaritalStatus"])
        user_input["NumberOfDependents"] = st.number_input(
            "Number of Dependents", min_value=0, max_value=15,
            value=int(numeric_ranges["NumberOfDependents"]["median"]), step=1)
    with col2:
        user_input["EducationLevel"] = st.selectbox("Education Level", categorical_options["EducationLevel"])
        user_input["EmploymentStatus"] = st.selectbox("Employment Status", categorical_options["EmploymentStatus"])
        user_input["Experience"] = st.number_input(
            "Years of Work Experience", min_value=0, max_value=60,
            value=int(numeric_ranges["Experience"]["median"]), step=1)
        user_input["JobTenure"] = st.number_input(
            "Job Tenure (years at current job)", min_value=0, max_value=50,
            value=int(numeric_ranges["JobTenure"]["median"]), step=1)

    st.subheader("Income & Employment")
    col3, col4 = st.columns(2)
    with col3:
        user_input["AnnualIncome"] = st.number_input(
            "Annual Income ($)", min_value=0.0,
            value=float(numeric_ranges["AnnualIncome"]["median"]), step=1000.0)
        user_input["MonthlyIncome"] = st.number_input(
            "Monthly Income ($)", min_value=1.0,
            value=float(numeric_ranges["MonthlyIncome"]["median"]), step=100.0)
    with col4:
        user_input["HomeOwnershipStatus"] = st.selectbox("Home Ownership Status", categorical_options["HomeOwnershipStatus"])

    st.subheader("Loan Details")
    col5, col6 = st.columns(2)
    with col5:
        user_input["LoanAmount"] = st.number_input(
            "Loan Amount ($)", min_value=0.0,
            value=float(numeric_ranges["LoanAmount"]["median"]), step=500.0)
        user_input["LoanDuration"] = st.number_input(
            "Loan Duration (months)", min_value=1,
            value=int(numeric_ranges["LoanDuration"]["median"]), step=1)
        user_input["MonthlyLoanPayment"] = st.number_input(
            "Expected Monthly Loan Payment ($)", min_value=0.0,
            value=float(numeric_ranges["MonthlyLoanPayment"]["median"]), step=50.0)
    with col6:
        user_input["LoanPurpose"] = st.selectbox("Loan Purpose", categorical_options["LoanPurpose"])
        user_input["InterestRate"] = st.number_input(
            "Interest Rate (decimal, e.g. 0.12 = 12%)", min_value=0.0, max_value=1.0,
            value=float(numeric_ranges["InterestRate"]["median"]), step=0.01, format="%.3f")
        user_input["BaseInterestRate"] = st.number_input(
            "Base Interest Rate (decimal)", min_value=0.0, max_value=1.0,
            value=float(numeric_ranges["BaseInterestRate"]["median"]), step=0.01, format="%.3f")

    st.subheader("Credit & Debt Profile")
    col7, col8 = st.columns(2)
    with col7:
        user_input["CreditScore"] = st.number_input(
            "Credit Score", min_value=300, max_value=850,
            value=int(numeric_ranges["CreditScore"]["median"]), step=1)
        user_input["DebtToIncomeRatio"] = st.number_input(
            "Debt-to-Income Ratio", min_value=0.0, max_value=5.0,
            value=float(numeric_ranges["DebtToIncomeRatio"]["median"]), step=0.01, format="%.3f")
        user_input["TotalDebtToIncomeRatio"] = st.number_input(
            "Total Debt-to-Income Ratio", min_value=0.0, max_value=10.0,
            value=float(numeric_ranges["TotalDebtToIncomeRatio"]["median"]), step=0.01, format="%.3f")
        user_input["MonthlyDebtPayments"] = st.number_input(
            "Monthly Debt Payments ($)", min_value=0.0,
            value=float(numeric_ranges["MonthlyDebtPayments"]["median"]), step=50.0)
        user_input["CreditCardUtilizationRate"] = st.number_input(
            "Credit Card Utilization Rate (0-1)", min_value=0.0, max_value=1.0,
            value=float(numeric_ranges["CreditCardUtilizationRate"]["median"]), step=0.01, format="%.3f")
    with col8:
        user_input["NumberOfOpenCreditLines"] = st.number_input(
            "Number of Open Credit Lines", min_value=0, max_value=30,
            value=int(numeric_ranges["NumberOfOpenCreditLines"]["median"]), step=1)
        user_input["NumberOfCreditInquiries"] = st.number_input(
            "Number of Credit Inquiries", min_value=0, max_value=30,
            value=int(numeric_ranges["NumberOfCreditInquiries"]["median"]), step=1)
        user_input["LengthOfCreditHistory"] = st.number_input(
            "Length of Credit History (years)", min_value=0,
            value=int(numeric_ranges["LengthOfCreditHistory"]["median"]), step=1)
        user_input["PaymentHistory"] = st.number_input(
            "Payment History Score", min_value=0,
            value=int(numeric_ranges["PaymentHistory"]["median"]), step=1)
        user_input["BankruptcyHistory"] = st.selectbox("Bankruptcy History", [0, 1],
                                                         format_func=lambda x: "Yes" if x == 1 else "No")
        user_input["PreviousLoanDefaults"] = st.selectbox("Previous Loan Defaults", [0, 1],
                                                            format_func=lambda x: "Yes" if x == 1 else "No")

    st.subheader("Assets & Other Financial Details")
    col9, col10 = st.columns(2)
    with col9:
        user_input["TotalAssets"] = st.number_input(
            "Total Assets ($)", min_value=0.0,
            value=float(numeric_ranges["TotalAssets"]["median"]), step=1000.0)
        user_input["TotalLiabilities"] = st.number_input(
            "Total Liabilities ($)", min_value=0.0,
            value=float(numeric_ranges["TotalLiabilities"]["median"]), step=1000.0)
        user_input["NetWorth"] = st.number_input(
            "Net Worth ($)", value=float(numeric_ranges["NetWorth"]["median"]), step=1000.0)
    with col10:
        user_input["SavingsAccountBalance"] = st.number_input(
            "Savings Account Balance ($)", min_value=0.0,
            value=float(numeric_ranges["SavingsAccountBalance"]["median"]), step=100.0)
        user_input["CheckingAccountBalance"] = st.number_input(
            "Checking Account Balance ($)", min_value=0.0,
            value=float(numeric_ranges["CheckingAccountBalance"]["median"]), step=100.0)
        user_input["UtilityBillsPaymentHistory"] = st.number_input(
            "Utility Bills Payment History (0-1, fraction paid on time)",
            min_value=0.0, max_value=1.0,
            value=float(numeric_ranges["UtilityBillsPaymentHistory"]["median"]), step=0.01, format="%.3f")

    submitted = st.form_submit_button("Predict Risk")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    try:
        input_df = pd.DataFrame([user_input])
        input_df = add_engineered_features(input_df)

        # Ensure exact column set/order expected by the pipeline
        expected_cols = metadata["feature_cols"]
        missing_cols = [c for c in expected_cols if c not in input_df.columns]
        if missing_cols:
            st.error(f"Missing required input(s): {missing_cols}")
            st.stop()
        input_df = input_df[expected_cols]

        prediction = pipeline.predict(input_df)[0]
        label = "HIGH RISK" if prediction == 1 else "LOW RISK"

        st.header("Prediction Result")
        if prediction == 1:
            st.error(f"### {label}")
        else:
            st.success(f"### {label}")

        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(input_df)[0, 1]
            st.write(f"**Risk Probability (High Risk):** {proba * 100:.1f}%")

        with st.expander("Model Explanation"):
            st.write(
                "This prediction is generated by a tuned Gradient Boosting "
                "classifier trained on historical applicant data. Feature "
                "importance (from training) indicates which variables most "
                "influenced the model's predictions in general; it does not "
                "prove causation for this individual case."
            )
            fi_path = "outputs/plots/feature_importance.png"
            if os.path.exists(fi_path):
                st.image(fi_path, caption="Top Feature Importances (from model training)")

    except Exception as e:
        st.error(f"Prediction failed due to an unexpected error: {e}")

st.markdown("---")
st.caption(
    "Credit Risk Prediction System — Machine Learning Capstone Project. "
    "Predictions are model estimates for academic purposes and should not be "
    "used for real lending decisions."
)
