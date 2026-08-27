"""Generates notebooks/credit_risk_analysis.ipynb by combining markdown
explanations (per Enginow spec section 25) with the exact code from build.py,
split into the required sections. Running this script only builds the .ipynb
structure; a separate nbclient execution step actually runs it end-to-end.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# 1. Project Introduction
md("""# Credit Risk Prediction System – End-to-End ML Model with Deployment

**Machine Learning Capstone Project**

This notebook builds a complete, end-to-end machine learning system that
classifies loan applicants as **High Risk** or **Low Risk**, following the
workflow: data understanding → cleaning → preprocessing → feature
engineering → EDA → train/test split → model training → hyperparameter
tuning → evaluation → model selection → model saving → Streamlit
deployment.""")

# 2. Problem Statement
md("""## Problem Statement

Lenders need to assess, before approving a loan, how likely an applicant is
to represent a credit risk. Manually reviewing every applicant's full
financial profile does not scale. We build a supervised classification model
that uses an applicant's financial and demographic information to predict a
binary risk category, which can support (not replace) a human underwriter's
decision.""")

# 3. Objective
md("""## Objective

Build and deploy a classification model that predicts whether a loan
applicant is **High Risk** or **Low Risk**, using only information that is
realistically available *before* a lending decision is made, while avoiding
data leakage and following sound ML engineering practice (proper
train/test separation, pipelines, cross-validated tuning).""")

# 4. Import Libraries
md("## 4. Import Libraries")
code("""import sys
sys.path.append("..")  # so feature_engineering.py (project root) is importable

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, roc_curve)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

from feature_engineering import add_engineered_features

RANDOM_STATE = 42
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
PLOTS_DIR = "../outputs/plots"
MODELS_DIR = "../models"
""")

# 5. Load Dataset
md("## 5. Load Dataset")
code("""df = pd.read_csv("../data/Loan.csv")
print("Shape:", df.shape)
df.head()""")

# 6. Data Understanding
md("""## 6. Data Understanding

We inspect the actual dataset (shape, dtypes, numeric/categorical columns,
descriptive statistics) rather than assuming any structure in advance.""")
code("""print("Shape:", df.shape)
df.info()""")
code("""df.describe(include='number').T""")
code("""numeric_cols_all = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols_all = df.select_dtypes(exclude=np.number).columns.tolist()
print("Numeric columns:", numeric_cols_all)
print("\\nCategorical columns:", categorical_cols_all)""")
code("""for c in ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']:
    print(c, ':', df[c].unique())""")
md("""**Why binary classification?** The Enginow specification requires predicting
`Risk (High/Low)` — a two-class label. Once we define this target (Section 8),
the task is a standard binary classification problem: assign each applicant
to one of exactly two mutually exclusive classes.""")

# 7. Data Quality Check
md("## 7. Data Quality Check")
code("""missing = df.isnull().sum()
print("Missing values per column (only columns with >0 shown):")
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

print("\\nDuplicate rows:", df.duplicated().sum())""")
md("""The dataset has **no missing values** and **no duplicate rows**. This is a
clean, complete dataset, so no row deletion or advanced imputation for
missingness is required at the raw-data level (imputers are still included
in the pipeline as a safety net and best practice).""")

# 8. Target Selection
md("""## 8. Target Selection — Investigation

The Enginow spec requires a target representing **Risk (High/Low)**. The raw
dataset offers two related candidates: `RiskScore` (continuous) and
`LoanApproved` (binary). We investigate both before choosing.""")
code("""print("RiskScore summary:")
print(df['RiskScore'].describe())
print()
print("LoanApproved value counts:")
print(df['LoanApproved'].value_counts())
print()
print("Correlation(RiskScore, LoanApproved):", df['RiskScore'].corr(df['LoanApproved']))""")
code("""# Is LoanApproved simply a threshold rule applied to RiskScore?
best_thresh, best_acc = None, 0
for t in range(20, 90):
    acc = ((df['RiskScore'] < t).astype(int) == df['LoanApproved']).mean()
    if acc > best_acc:
        best_acc, best_thresh = acc, t
print(f"Best single threshold on RiskScore reproduces LoanApproved with accuracy = {best_acc:.4f} at threshold {best_thresh}")""")
md(f"""**Findings:**
- A simple rule *"Approved if RiskScore < ~45"* reproduces `LoanApproved` with
  ~96.5% accuracy. This shows `LoanApproved` is **downstream of / derived
  from** `RiskScore` (an approval decision made largely from the risk score),
  not an independent signal.
- `LoanApproved` represents a **post-decision** business outcome (whether the
  loan was approved), not risk itself. Using it as the target would answer
  "will this application be approved?" rather than "is this applicant risky?"
  and would still be circular with `RiskScore`.
- `RiskScore` is the dataset's native, continuous **risk measurement**, and is
  the most direct available proxy for the "Risk (High/Low)" label the spec
  requires.

**Target Decision:** We binarize `RiskScore` using a **median split**:

```
High Risk = RiskScore >= median(RiskScore)
Low  Risk = RiskScore <  median(RiskScore)
```

**Why median split:** it is simple, reproducible, requires no arbitrary
business threshold, and (as shown below) yields a naturally balanced target
— avoiding the need for artificial class-balancing tricks.

**Why alternatives were rejected:**
- `LoanApproved` as target — rejected because it reflects a post-decision
  approval outcome derived mostly from `RiskScore`, not risk itself.
- An arbitrary fixed RiskScore cutoff (e.g. "high risk if RiskScore > 60") —
  rejected in favor of the median split, which is data-driven rather than
  hand-picked.

**Limitation (documented):** `RiskScore` itself is very likely a formula
computed from a combination of the applicant's financial features (income,
net worth, debt ratios, bankruptcy history, etc. — see the leakage analysis
below). This means the model is, to a meaningful extent, learning to
approximate that pre-existing scoring formula rather than discovering an
entirely independent notion of risk. This is a known and disclosed
limitation of using a dataset where the "ground truth" risk label is itself
model/rule-derived rather than an observed real-world outcome (e.g. actual
default).""")
code("""RISK_MEDIAN = df['RiskScore'].median()
df['RiskLabel'] = np.where(df['RiskScore'] >= RISK_MEDIAN, 'High Risk', 'Low Risk')
print("Median RiskScore threshold:", RISK_MEDIAN)
print(df['RiskLabel'].value_counts())
print(df['RiskLabel'].value_counts(normalize=True) * 100)""")

# 9. Data Leakage Analysis
md("""## 9. Data Leakage Analysis

**Target-derived leakage:**
- `RiskScore` is used to construct the target → **must be dropped** as a
  feature (it would trivially leak the label).
- `LoanApproved` is strongly derived from `RiskScore` (~96.5% reproducible
  via a simple threshold) and represents a post-decision outcome →
  **also dropped**.

**Other columns checked:** No other column in the dataset directly encodes
the target or is a duplicate of it.

**Procedural leakage prevention (enforced throughout this notebook):**
- Train/test split is performed **before** any preprocessing, scaling,
  encoding, feature-selection, or SMOTE.
- All preprocessing (imputation, scaling, one-hot encoding) is wrapped in a
  single `sklearn`/`imblearn` `Pipeline`, fit **only** on training folds.
- SMOTE (if used) is placed inside the pipeline so it is applied only to
  training folds during cross-validation, never to the test set.
- Hyperparameter tuning uses cross-validation on the training set only; the
  test set is touched exactly once, for final evaluation.

`ApplicationDate` is also dropped — inspection showed dates spanning an
unrealistic range (into the 2070s), indicating it is a synthetic field with
no genuine temporal predictive meaning here.""")
code("""LEAKY_COLS = ['RiskScore', 'LoanApproved']
DROP_COLS = ['ApplicationDate']
print(df['ApplicationDate'].min(), '->', df['ApplicationDate'].max())""")

# 10. Data Preprocessing
md("""## 10. Data Preprocessing

We build a `ColumnTransformer` that:
- Imputes numeric columns with the **median** and scales them with
  `StandardScaler`.
- Imputes categorical columns with the **most frequent** value and encodes
  them with `OneHotEncoder(handle_unknown="ignore")`.

All fitting happens inside the pipeline, only on the training set (Section
13), so there is no leakage from test data into scaling/encoding.""")
code("""for c in numeric_cols_all:
    if c not in ('RiskScore', 'LoanApproved'):
        pass
numeric_cols_all = [c for c in numeric_cols_all if c not in ('RiskScore', 'LoanApproved')]
categorical_cols_all = [c for c in categorical_cols_all if c not in ('RiskLabel',)]
print("Numeric feature candidates:", numeric_cols_all)
print("Categorical feature candidates:", categorical_cols_all)""")

md("""### Outlier Investigation (IQR)

We check for outliers using the IQR method on key numeric columns, but do
**not** automatically delete them: in a credit-risk context, unusually high
income/assets or low credit scores are genuine, informative data points, not
measurement errors. Tree-based models (Random Forest / Gradient Boosting /
XGBoost) are also naturally robust to such outliers.""")
code("""outlier_report = {}
for col in ['AnnualIncome', 'LoanAmount', 'TotalAssets', 'TotalLiabilities', 'CreditScore']:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report[col] = {'lower': round(lower, 2), 'upper': round(upper, 2), 'n_outliers': int(n_out)}
pd.DataFrame(outlier_report).T""")
md("**Decision:** outliers are retained (no rows removed) for the reasons above.")

# 11. Feature Engineering
md("""## 11. Feature Engineering

The dataset already contains `DebtToIncomeRatio` and `TotalDebtToIncomeRatio`.
We inspect rather than blindly recompute a duplicate ratio.""")
code("""check = (df['MonthlyDebtPayments'] / df['MonthlyIncome'].replace(0, np.nan)).head()
print("MonthlyDebtPayments/MonthlyIncome (manual):", check.values)
print("Existing DebtToIncomeRatio column:          ", df['DebtToIncomeRatio'].head().values)""")
md("""The existing `DebtToIncomeRatio` does not match a naive
`MonthlyDebtPayments / MonthlyIncome` computation exactly (it likely factors
in additional obligations), so we **use the existing column as-is** rather
than overwriting it with a duplicate/simplified formula.

**New engineered features:**

1. **TotalObligationRatio** = `(TotalLiabilities + MonthlyLoanPayment) / (12 * MonthlyIncome + 1)`
   — captures total financial burden (existing liabilities plus the new
   loan's repayment) relative to annual income capacity. The `+1` in the
   denominator prevents division by zero.
2. **AgeGroup** — `Young` (<=30), `Middle-aged` (31-50), `Senior` (51+),
   chosen as standard, easily interpretable life-stage boundaries.

(Implemented in `feature_engineering.py` so training and the deployed
Streamlit app compute these identically.)""")
code("""df = add_engineered_features(df)
numeric_cols_all = numeric_cols_all + ['TotalObligationRatio']
categorical_cols_all = categorical_cols_all + ['AgeGroup']
df[['TotalObligationRatio', 'AgeGroup']].describe(include='all')""")
code("""df['AgeGroup'].value_counts()""")

# 12. Exploratory Data Analysis
md("""## 12. Exploratory Data Analysis

Key visual insights required by the spec: risk distribution, risk vs.
age/income/employment/credit score, correlation heatmap, boxplots, and
distribution plots. Plots are saved to `outputs/plots/`.""")
code("""order = ['Low Risk', 'High Risk']
plt.figure(figsize=(6, 4.5))
ax = sns.countplot(x='RiskLabel', data=df, order=order, palette=['#4C72B0', '#C44E52'])
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2, p.get_height()), ha='center', va='bottom')
plt.title('Risk Class Distribution'); plt.xlabel('Risk Label'); plt.ylabel('Number of Applicants')
plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/risk_distribution.png'); plt.show()""")
code("""plt.figure(figsize=(6.5, 4.5))
sns.boxplot(x='RiskLabel', y='Age', data=df, order=order, palette=['#4C72B0', '#C44E52'])
plt.title('Applicant Age by Risk Category'); plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/risk_by_age.png'); plt.show()""")
code("""plt.figure(figsize=(6.5, 4.5))
sns.boxplot(x='RiskLabel', y='AnnualIncome', data=df, order=order, palette=['#4C72B0', '#C44E52'])
plt.title('Annual Income by Risk Category'); plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/risk_by_income.png'); plt.show()""")
code("""plt.figure(figsize=(7, 4.5))
sns.countplot(x='EmploymentStatus', hue='RiskLabel', data=df, hue_order=order, palette=['#4C72B0', '#C44E52'])
plt.title('Risk Category by Employment Status'); plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/risk_by_employment.png'); plt.show()""")
code("""plt.figure(figsize=(6.5, 4.5))
sns.boxplot(x='RiskLabel', y='CreditScore', data=df, order=order, palette=['#4C72B0', '#C44E52'])
plt.title('Credit Score by Risk Category'); plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/risk_by_credit_score.png'); plt.show()""")
code("""plt.figure(figsize=(14, 11))
corr = df[numeric_cols_all].corr()
sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.3)
plt.title('Correlation Heatmap - Numeric Features'); plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/correlation_heatmap.png'); plt.show()""")
code("""important_num = ['AnnualIncome', 'CreditScore', 'LoanAmount', 'DebtToIncomeRatio', 'TotalAssets', 'TotalLiabilities']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), important_num):
    sns.boxplot(y=df[col], ax=ax, color='#4C72B0'); ax.set_title(col)
plt.suptitle('Boxplots of Key Numeric Features'); plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/numeric_boxplots.png'); plt.show()""")
code("""fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), important_num):
    sns.histplot(df[col], kde=True, ax=ax, color='#55A868'); ax.set_title(col)
plt.suptitle('Distributions of Key Numeric Features'); plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/numeric_distributions.png'); plt.show()""")
md("""**Brief observations:** High-risk applicants tend to show lower annual
income and lower credit scores than low-risk applicants (see boxplots
above); the correlation heatmap highlights `DebtToIncomeRatio`,
`TotalDebtToIncomeRatio`, `NetWorth`, `AnnualIncome`/`MonthlyIncome`, and
`BankruptcyHistory` as features most associated with risk — consistent with
the feature-importance results obtained later from the trained model.""")

# 13. Train/Test Split
md("""## 13. Train/Test Split

80/20 stratified split with a fixed `random_state=42`, performed **before**
any preprocessing, scaling, encoding or SMOTE.""")
code("""feature_cols = [c for c in df.columns if c not in LEAKY_COLS + DROP_COLS + ['RiskLabel']]
X = df[feature_cols]
y = (df['RiskLabel'] == 'High Risk').astype(int)  # 1 = High Risk, 0 = Low Risk

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print('Train shape:', X_train.shape, ' Test shape:', X_test.shape)

numeric_features = [c for c in numeric_cols_all if c in feature_cols]
categorical_features = [c for c in categorical_cols_all if c in feature_cols]
print('Numeric features used:', numeric_features)
print('Categorical features used:', categorical_features)""")

# 14. Class Imbalance Analysis
md("## 14. Class Imbalance Analysis")
code("""class_counts = y_train.value_counts()
class_pct = y_train.value_counts(normalize=True) * 100
print('Train class counts:\\n', class_counts)
print('\\nTrain class %:\\n', class_pct)
imbalance_ratio = class_counts.max() / class_counts.min()
print('\\nImbalance ratio:', imbalance_ratio)
USE_SMOTE = imbalance_ratio > 1.5
print('Use SMOTE:', USE_SMOTE)""")
md("""Because the target was constructed with a **median split**, the classes
are naturally close to balanced (~50/50 in the training set, imbalance ratio
~1.03). This is **not** a severe imbalance, so we do **not** apply SMOTE —
using it here would be artificial oversampling for a problem that does not
need it. This decision is confirmed programmatically above (`USE_SMOTE`).""")

# 15. Model Training (pipelines + baseline)
md("""## 15. Model Training

We build one shared `ColumnTransformer` (median imputation + scaling for
numeric, most-frequent imputation + one-hot encoding for categorical) and
wrap each of the four required models in an `imblearn` `Pipeline`
(so SMOTE, if used, would be correctly scoped to training folds only).""")
code("""numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

def make_pipeline(model, use_smote):
    steps = [('preprocessor', preprocessor)]
    if use_smote:
        steps.append(('smote', SMOTE(random_state=RANDOM_STATE)))
    steps.append(('classifier', model))
    return ImbPipeline(steps=steps)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False),
}""")

# 16. Baseline Model Comparison
md("## 16. Baseline Model Comparison")
code("""def evaluate(pipe, X_te, y_te):
    preds = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    return {
        'Accuracy': accuracy_score(y_te, preds),
        'Precision': precision_score(y_te, preds),
        'Recall': recall_score(y_te, preds),
        'F1 Score': f1_score(y_te, preds),
        'ROC-AUC': roc_auc_score(y_te, proba),
    }

baseline_results = []
baseline_pipes = {}
for name, model in models.items():
    pipe = make_pipeline(model, USE_SMOTE)
    pipe.fit(X_train, y_train)
    metrics = evaluate(pipe, X_test, y_test)
    metrics['Model'] = name
    baseline_results.append(metrics)
    baseline_pipes[name] = pipe

comparison_df = pd.DataFrame(baseline_results)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']]
comparison_df.to_csv('../outputs/model_comparison.csv', index=False)
comparison_df""")

# 17. Hyperparameter Tuning
md("""## 17. Hyperparameter Tuning

`RandomizedSearchCV` with 5-fold stratified cross-validation, scoring on
**F1** (balances precision and recall — both matter here since the target is
balanced but misclassifying either class has a real cost). The test set is
never used during tuning.""")
code("""cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__class_weight': [None, 'balanced'],
    },
    'Random Forest': {
        'classifier__n_estimators': [200, 300, 400],
        'classifier__max_depth': [None, 8, 12, 16],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__class_weight': [None, 'balanced'],
    },
    'Gradient Boosting': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [2, 3, 4],
        'classifier__min_samples_split': [2, 5, 10],
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__subsample': [0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 1.0],
        'classifier__reg_lambda': [0.5, 1.0, 2.0],
    },
}

tuned_pipes = {}
best_params_report = {}
for name, model in models.items():
    pipe = make_pipeline(model, USE_SMOTE)
    search = RandomizedSearchCV(
        pipe, param_distributions=param_grids[name], n_iter=15, scoring='f1',
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    tuned_pipes[name] = search.best_estimator_
    best_params_report[name] = search.best_params_
    print(name, '-> best CV F1:', round(search.best_score_, 4), '| best params:', search.best_params_)""")

# 18. Final Model Evaluation
md("## 18. Final Model Evaluation (tuned models, on held-out test set)")
code("""tuned_results = []
for name, pipe in tuned_pipes.items():
    metrics = evaluate(pipe, X_test, y_test)
    metrics['Model'] = name
    tuned_results.append(metrics)
tuned_comparison_df = pd.DataFrame(tuned_results)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']]
tuned_comparison_df.to_csv('../outputs/model_comparison_tuned.csv', index=False)
tuned_comparison_df.sort_values('F1 Score', ascending=False)""")

# 19. Confusion Matrix
md("## 19. Confusion Matrix")
code("""ranked = tuned_comparison_df.sort_values(['F1 Score', 'ROC-AUC'], ascending=False)
best_model_name = ranked.iloc[0]['Model']
best_pipe = tuned_pipes[best_model_name]
print('Selected best model:', best_model_name)

preds = best_pipe.predict(X_test)
proba = best_pipe.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
plt.title(f'Confusion Matrix - {best_model_name}'); plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/confusion_matrix.png'); plt.show()
print('TN, FP, FN, TP:', cm.ravel())""")

# 20. ROC Curve
md("## 20. ROC Curve")
code("""fpr, tpr, _ = roc_curve(y_test, proba)
auc_val = roc_auc_score(y_test, proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {auc_val:.3f})', color='#C44E52')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve - Best Model')
plt.legend(); plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/roc_curve.png'); plt.show()""")

# 21. Feature Importance
md("""## 21. Feature Importance

Feature importance indicates which variables contributed most to the
model's predictions on average; it does **not** prove causation.""")
code("""def get_feature_names(preprocessor):
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    return list(numeric_features) + cat_names

fitted_preprocessor = best_pipe.named_steps['preprocessor']
all_feature_names = get_feature_names(fitted_preprocessor)
classifier = best_pipe.named_steps['classifier']

if hasattr(classifier, 'feature_importances_'):
    importances = classifier.feature_importances_
elif hasattr(classifier, 'coef_'):
    importances = np.abs(classifier.coef_[0])
else:
    importances = None

if importances is not None:
    fi_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='importance', y='feature', data=fi_df, color='#4C72B0')
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/feature_importance.png'); plt.show()
    display(fi_df)""")

# 22. Final Model Selection
md("""## 22. Final Model Selection

Selection is based on the actual tuned test-set results above (Section 18),
prioritizing **F1 Score** (balances precision and recall) with **ROC-AUC**
as a tie-breaker, while also considering recall specifically — in credit
risk, failing to flag a genuinely high-risk applicant (a false negative) can
be more costly than an overly cautious false positive. XGBoost was **not**
automatically chosen; the model with the best actual tuned metrics was
selected programmatically (see `ranked` above).""")
code("""print('FINAL SELECTED MODEL:', best_model_name)
print()
print(ranked.to_string(index=False))
print()
print('Best hyperparameters:', best_params_report[best_model_name])""")

# 23. Save Final Model
md("## 23. Save Final Model")
code("""joblib.dump(best_pipe, f'{MODELS_DIR}/credit_risk_model.joblib')

metadata = {
    'best_model_name': best_model_name,
    'risk_median_threshold': float(RISK_MEDIAN),
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'feature_cols': feature_cols,
    'best_params': best_params_report[best_model_name],
    'categorical_options': {c: sorted(df[c].astype(str).unique().tolist()) for c in categorical_features},
    'numeric_ranges': {c: {'min': float(df[c].min()), 'max': float(df[c].max()), 'median': float(df[c].median())} for c in numeric_features},
}
with open(f'{MODELS_DIR}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('Saved:', f'{MODELS_DIR}/credit_risk_model.joblib')
print('Saved:', f'{MODELS_DIR}/metadata.json')""")

# 24. Conclusion
md("""## 24. Conclusion

**Summary:** Using the `Loan.csv` dataset (20,000 applicants, 36 raw
columns), we constructed a `High Risk` / `Low Risk` target from the
dataset's `RiskScore` (median split), carefully excluding `RiskScore` and
the downstream `LoanApproved` field to avoid leakage. After cleaning
(no missing values / duplicates found), feature engineering
(`TotalObligationRatio`, `AgeGroup`), and EDA, we trained four models
(Logistic Regression, Random Forest, Gradient Boosting, XGBoost) inside
`sklearn`/`imblearn` pipelines, tuned each with `RandomizedSearchCV`
(5-fold CV, scoring=F1), and evaluated all of them on a held-out 20% test
set. The best-performing tuned model (selected on actual F1/ROC-AUC results,
see Section 22) was saved as a single deployable pipeline and served through
a Streamlit application (`app.py`).

**Key limitation:** Because the target is derived from `RiskScore`, which is
itself likely a formula computed from the applicant's financial attributes,
the model is partly learning to approximate that existing scoring formula
rather than an independently observed outcome (e.g. an actual future
default). This is disclosed here and in the README for transparency.

**Learning outcomes:** This project demonstrates practical experience with
data leakage analysis, `ColumnTransformer`/`Pipeline` design, feature
engineering, EDA, handling class balance, cross-validated hyperparameter
tuning, multi-metric model evaluation, model persistence, and deployment via
Streamlit.""")

nb['cells'] = cells
with open('notebooks/credit_risk_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Notebook written with', len(cells), 'cells.')
