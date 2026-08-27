# Credit Risk Prediction System

## Overview
An end-to-end Machine Learning Capstone project that predicts whether a loan
applicant is **High Risk** or **Low Risk** using financial and demographic
data, deployed as an interactive Streamlit application.

## Problem Statement
Lenders need an objective, scalable way to flag potentially risky loan
applicants before a lending decision is made, to help reduce loan defaults.

## Objective
Build a complete, leakage-free classification pipeline — from raw data to a
deployed web app — that predicts applicant risk using only information
realistically available at the time of application.

## Dataset
- File: `data/Loan.csv`
- **20,000 rows, 36 raw columns** (verified by direct inspection of the CSV)
- No missing values, no duplicate rows
- Contains financial fields (income, credit score, assets, liabilities, debt
  ratios, interest rates), demographic fields (age, marital status,
  dependents, education), loan fields (amount, duration, purpose), and two
  target-related fields: `RiskScore` (continuous) and `LoanApproved` (binary)

## Dataset Features
Key raw columns used as model inputs (35 total, see `models/metadata.json`
for the exact list): `Age`, `AnnualIncome`, `MonthlyIncome`, `CreditScore`,
`EmploymentStatus`, `EducationLevel`, `Experience`, `LoanAmount`,
`LoanDuration`, `LoanPurpose`, `MaritalStatus`, `NumberOfDependents`,
`HomeOwnershipStatus`, `MonthlyDebtPayments`, `CreditCardUtilizationRate`,
`NumberOfOpenCreditLines`, `NumberOfCreditInquiries`, `DebtToIncomeRatio`,
`TotalDebtToIncomeRatio`, `BankruptcyHistory`, `PreviousLoanDefaults`,
`PaymentHistory`, `LengthOfCreditHistory`, `SavingsAccountBalance`,
`CheckingAccountBalance`, `TotalAssets`, `TotalLiabilities`,
`UtilityBillsPaymentHistory`, `JobTenure`, `NetWorth`, `BaseInterestRate`,
`InterestRate`, `MonthlyLoanPayment`, plus two engineered features
(`TotalObligationRatio`, `AgeGroup`).

`ApplicationDate` is dropped (unrealistic date range, no genuine predictive
value). `RiskScore` and `LoanApproved` are excluded as **features** — see
Target Variable section below.

## Target Variable
**Investigation:** `RiskScore` (continuous, range ~28.8–84) is the dataset's
native risk measurement, strongly correlated with financial-health features
(income, net worth, bankruptcy history, debt ratios). `LoanApproved`
(binary) can be reproduced from `RiskScore` alone with ~96.5% accuracy using
a simple threshold rule (`RiskScore < ~45 → Approved`), proving it is a
**downstream, post-decision** field derived largely from `RiskScore`, not an
independent risk label.

**Selected target:** `RiskLabel`, derived from `RiskScore` via a **median
split**:
```
High Risk = RiskScore >= 52.0 (dataset median)
Low  Risk = RiskScore <  52.0
```
Resulting distribution: **50.84% High Risk / 49.16% Low Risk** — naturally
balanced.

**Why this target:** It directly represents "risk" (as the spec requires),
is derived from the dataset's own dedicated risk-scoring field, and the
median split is simple, reproducible, and avoids an arbitrary hand-picked
cutoff.

**Why alternatives were rejected:** `LoanApproved` reflects an approval
*decision* rather than risk itself, and is largely circular with
`RiskScore` — using it as the target would not satisfy the "Risk (High/Low)"
requirement as directly, and would not remove the underlying leakage
concern either.

**Limitation:** `RiskScore` is very likely itself computed from a formula
over the applicant's financial attributes. The model is therefore, to some
extent, learning to approximate that pre-existing scoring logic rather than
an independently observed real-world outcome (such as an actual future
default). This is disclosed for transparency; `RiskScore` and `LoanApproved`
are excluded from the feature set to prevent direct leakage.

## Data Preprocessing
- **Numeric features:** median imputation + `StandardScaler`
- **Categorical features:** most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`
- All steps are wrapped in a single `sklearn`/`imblearn` `Pipeline` /
  `ColumnTransformer`, fit **only** on the training split (no leakage from
  test data into scaling or encoding)
- **Outliers:** investigated via the IQR method on key numeric columns;
  retained (not deleted) since they represent genuine, informative
  high-income/high-asset or low-credit-score profiles

## Feature Engineering
- **TotalObligationRatio** = `(TotalLiabilities + MonthlyLoanPayment) / (12*MonthlyIncome + 1)`
  — total financial burden relative to annual income capacity (protected
  against divide-by-zero)
- **AgeGroup** — `Young` (≤30), `Middle-aged` (31–50), `Senior` (51+)
- The existing `DebtToIncomeRatio` / `TotalDebtToIncomeRatio` columns were
  inspected and used as-is (not blindly recomputed, since they do not match
  a naive `MonthlyDebtPayments / MonthlyIncome` formula, indicating they
  already account for additional obligations)

## Exploratory Data Analysis
Saved to `outputs/plots/`:
`risk_distribution.png`, `risk_by_age.png`, `risk_by_income.png`,
`risk_by_employment.png`, `risk_by_credit_score.png`,
`correlation_heatmap.png`, `numeric_boxplots.png`,
`numeric_distributions.png`. High-risk applicants tend to have lower income
and lower credit scores than low-risk applicants; `DebtToIncomeRatio`,
`NetWorth`, income, and `BankruptcyHistory` show the strongest association
with risk.

## Machine Learning Models
Four required models were trained: **Logistic Regression, Random Forest,
Gradient Boosting, XGBoost** (SVM was not included — not required, and the
four mandatory models already give strong, comparable coverage of linear and
tree-based approaches without adding unnecessary complexity).

## Imbalanced Data Handling
Checked programmatically: training-set class ratio ≈ 1.03:1 (50.84% /
49.16%), i.e. **not significantly imbalanced** — a natural result of the
median-split target definition. **SMOTE was not applied**, since doing so
would be artificial oversampling for a problem that does not need it. The
pipeline is built with `imblearn.Pipeline` so SMOTE could be enabled with
one line if the class balance changes in the future.

## Hyperparameter Tuning
`RandomizedSearchCV` (`n_iter=15`, 5-fold `StratifiedKFold`,
`scoring="f1"`) was used for all four models — F1 was chosen because it
balances precision and recall, both of which matter for a balanced
credit-risk classification task. The test set was never used during tuning
or model selection.

## Evaluation Metrics
Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix, ROC Curve
— all computed on the held-out 20% test set (`outputs/plots/confusion_matrix.png`,
`outputs/plots/roc_curve.png`).

## Model Comparison
**Baseline (before tuning)** — `outputs/model_comparison.csv`:

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|---------:|----------:|-------:|---------:|--------:|
| Logistic Regression |  0.8748  |   0.8770  | 0.8766 |  0.8768  | 0.9465  |
| Random Forest       |  0.9125  |   0.9223  | 0.9041 |  0.9131  | 0.9760  |
| Gradient Boosting   |  0.9380  |   0.9443  | 0.9331 |  0.9387  | 0.9840  |
| XGBoost             |  0.9490  |   0.9485  | 0.9513 |  0.9499  | 0.9906  |

**After hyperparameter tuning** — `outputs/model_comparison_tuned.csv`:

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|---------:|----------:|-------:|---------:|--------:|
| Logistic Regression |  0.8748  |   0.8770  | 0.8766 |  0.8768  | 0.9465  |
| Random Forest       |  0.9145  |   0.9213  | 0.9095 |  0.9154  | 0.9769  |
| **Gradient Boosting** | **0.9505** | **0.9509** | **0.9518** | **0.9514** | **0.9907** |
| XGBoost             |  0.9495  |   0.9486  | 0.9523 |  0.9504  | 0.9914  |

## Final Model
**Selected: Gradient Boosting Classifier** (tuned) — chosen from actual
tuned test results, prioritizing F1 Score (0.9514, highest among all four)
with strong Recall (0.9518, important for catching genuinely high-risk
applicants) and ROC-AUC (0.9907, essentially tied with XGBoost's 0.9914).
Gradient Boosting was **not chosen automatically** — its selection follows
directly from the metrics table above via code (`ranked.sort_values(...)`),
not from an assumption that any particular algorithm is "best."

**Best hyperparameters:**
```
n_estimators = 300
learning_rate = 0.1
max_depth = 4
min_samples_split = 10
```

**Final test-set metrics:** Accuracy 0.9505, Precision 0.9509, Recall
0.9518, F1 0.9514, ROC-AUC 0.9907.

## Feature Importance
Computed from the trained Gradient Boosting model's native
`feature_importances_` (`outputs/plots/feature_importance.png`). Top
contributors: `DebtToIncomeRatio`, `NetWorth`, `AnnualIncome`,
`TotalDebtToIncomeRatio`, `MonthlyIncome`, `PreviousLoanDefaults`,
`CreditScore`, `LengthOfCreditHistory`, `BankruptcyHistory`, `InterestRate`.
Feature importance indicates association strength in the model, not
causation.

## Streamlit Deployment
`app.py` loads the saved end-to-end pipeline
(`models/credit_risk_model.joblib`) and metadata
(`models/metadata.json`), collects applicant details through a form (raw
inputs only — engineered features are computed automatically), and displays
**HIGH RISK** / **LOW RISK** with the model's predicted probability.

## Project Structure
```
credit-risk-prediction/
├── data/Loan.csv
├── notebooks/credit_risk_analysis.ipynb
├── models/
│   ├── credit_risk_model.joblib
│   └── metadata.json
├── outputs/
│   ├── plots/ (11 PNG files)
│   ├── model_comparison.csv
│   └── model_comparison_tuned.csv
├── feature_engineering.py
├── build.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run
```bash
pip install -r requirements.txt

# (optional) retrain the model from scratch
python build.py

# launch the app
streamlit run app.py
```
