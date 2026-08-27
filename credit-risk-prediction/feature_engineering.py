"""
Shared feature-engineering logic used by both the training script (build.py /
the notebook) and the Streamlit app (app.py), so that engineered features are
computed identically at training time and at prediction time.
"""
import numpy as np
import pandas as pd

AGE_BINS = [0, 30, 50, 200]
AGE_LABELS = ["Young", "Middle-aged", "Senior"]


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add TotalObligationRatio and AgeGroup to a raw applicant DataFrame.

    TotalObligationRatio = (TotalLiabilities + MonthlyLoanPayment) / (12 * MonthlyIncome + 1)
        Captures total financial burden (existing liabilities + the new loan's
        monthly repayment) relative to annual income capacity. The "+1" in the
        denominator guards against division by zero.

    AgeGroup: Young (<=30), Middle-aged (31-50), Senior (51+).
    """
    data = data.copy()
    annual_income_proxy = data["MonthlyIncome"] * 12
    data["TotalObligationRatio"] = (
        (data["TotalLiabilities"] + data["MonthlyLoanPayment"]) /
        (annual_income_proxy + 1)
    )
    data["AgeGroup"] = pd.cut(data["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    data["AgeGroup"] = data["AgeGroup"].astype(str)
    return data
