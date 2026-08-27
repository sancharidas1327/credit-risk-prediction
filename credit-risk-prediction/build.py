"""
Credit Risk Prediction System - Build Script
==============================================
Runs the full pipeline: load -> clean -> feature engineer -> EDA -> split ->
train (4 models) -> compare -> tune best candidates -> evaluate -> select ->
save pipeline. Produces all files under outputs/ and models/.

This script is the executable source of truth; the notebook mirrors it with
explanations for academic submission.
"""
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
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
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "data/Loan.csv"
PLOTS_DIR = "outputs/plots"
MODELS_DIR = "models"

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)

# ---------------------------------------------------------------------------
# 2. DATA QUALITY CHECK
# ---------------------------------------------------------------------------
missing = df.isnull().sum()
missing = missing[missing > 0]
duplicates = df.duplicated().sum()
print("Missing values (columns with >0):\n", missing)
print("Duplicate rows:", duplicates)

# ---------------------------------------------------------------------------
# 3. TARGET SELECTION
# ---------------------------------------------------------------------------
# Investigation performed prior to this script (see notebook / analysis):
#   - RiskScore (continuous, range ~28.8-84) is a computed risk metric, strongly
#     correlated with financial-health features (Income, NetWorth, BankruptcyHistory,
#     DebtToIncomeRatio, etc.)
#   - LoanApproved (binary) can be reproduced from RiskScore alone with ~96.5%
#     accuracy using a simple threshold (RiskScore < 45 -> Approved). This proves
#     LoanApproved is a DOWNSTREAM / derived decision made largely from RiskScore,
#     i.e. a post-decision artifact, not an independent risk label.
#   - Using LoanApproved as the target would (a) represent an approval decision
#     rather than "risk" as required by the spec, and (b) still be circular with
#     RiskScore.
#   - RiskScore itself is the dataset's native, continuous risk measurement and
#     is the most direct available proxy for "Risk (High/Low)" required by the
#     Enginow spec. We therefore binarize RiskScore using a MEDIAN SPLIT:
#         High Risk = RiskScore >= median(RiskScore)
#         Low  Risk = RiskScore <  median(RiskScore)
#   - The median split produces a naturally balanced target without artificial
#     resampling and gives a defensible, reproducible cut point.
# LEAKAGE PREVENTION: both RiskScore and LoanApproved are DROPPED from the
# feature set because the target is derived from RiskScore (using it as a
# feature would leak the answer) and LoanApproved is downstream of the same
# information.
RISK_MEDIAN = df["RiskScore"].median()
df["RiskLabel"] = np.where(df["RiskScore"] >= RISK_MEDIAN, "High Risk", "Low Risk")
print("\nTarget threshold (RiskScore median):", RISK_MEDIAN)
print(df["RiskLabel"].value_counts())
print(df["RiskLabel"].value_counts(normalize=True) * 100)

LEAKY_COLS = ["RiskScore", "LoanApproved"]
DROP_COLS = ["ApplicationDate"]  # synthetic/unrealistic dates (some in year 2072), not a genuine predictor

# ---------------------------------------------------------------------------
# 4. DATA UNDERSTANDING SUMMARY (saved for README/notebook use)
# ---------------------------------------------------------------------------
numeric_cols_all = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols_all = df.select_dtypes(exclude=np.number).columns.tolist()
for c in ["RiskScore", "LoanApproved"]:
    if c in numeric_cols_all:
        numeric_cols_all.remove(c)
if "RiskLabel" in categorical_cols_all:
    categorical_cols_all.remove("RiskLabel")
if "ApplicationDate" in categorical_cols_all:
    categorical_cols_all.remove("ApplicationDate")

print("\nNumeric feature candidates:", numeric_cols_all)
print("Categorical feature candidates:", categorical_cols_all)

# ---------------------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
# DebtToIncomeRatio and TotalDebtToIncomeRatio already exist in the raw data.
# We inspect rather than blindly recompute duplicates.
print("\nExisting DebtToIncomeRatio sample check vs MonthlyDebtPayments/MonthlyIncome:")
check = (df["MonthlyDebtPayments"] / df["MonthlyIncome"].replace(0, np.nan)).head()
print(check.values, df["DebtToIncomeRatio"].head().values)
# -> Existing DebtToIncomeRatio and TotalDebtToIncomeRatio are usable as-is;
#    we will NOT recompute a duplicate ratio.

# Total Obligation Ratio: (TotalLiabilities + MonthlyLoanPayment) / (MonthlyIncome*12 + 1)
# Rationale: captures total financial burden (existing liabilities + new loan
# repayment) relative to annual income capacity. Avoids divide-by-zero with +1.
# (Implementation lives in feature_engineering.py so build.py and app.py stay
# in sync.)
df = add_engineered_features(df)
numeric_cols_all += ["TotalObligationRatio"]
categorical_cols_all += ["AgeGroup"]

print("\nAgeGroup boundaries: Young <=30, Middle-aged 31-50, Senior 51+")
print(df["AgeGroup"].value_counts())

# ---------------------------------------------------------------------------
# 6. OUTLIER INVESTIGATION (IQR) - informational, no automatic deletion
# ---------------------------------------------------------------------------
outlier_report = {}
for col in ["AnnualIncome", "LoanAmount", "TotalAssets", "TotalLiabilities", "CreditScore"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report[col] = {"lower": float(lower), "upper": float(upper), "n_outliers": int(n_out)}
print("\nOutlier report (IQR method):")
for k, v in outlier_report.items():
    print(k, v)
# Decision: outliers are retained. These are genuine (if extreme) financial
# profiles (e.g. high-income/high-asset applicants) that are realistic and
# informative for a credit-risk model; StandardScaler + tree-based models
# handle this reasonably. No rows removed.

# ---------------------------------------------------------------------------
# 7. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
# 7.1 Risk distribution
plt.figure(figsize=(6, 4.5))
order = ["Low Risk", "High Risk"]
ax = sns.countplot(x="RiskLabel", data=df, order=order, palette=["#4C72B0", "#C44E52"])
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.title("Risk Class Distribution")
plt.xlabel("Risk Label")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/risk_distribution.png")
plt.close()

# 7.2 Risk by Age
plt.figure(figsize=(6.5, 4.5))
sns.boxplot(x="RiskLabel", y="Age", data=df, order=order, palette=["#4C72B0", "#C44E52"])
plt.title("Applicant Age by Risk Category")
plt.xlabel("Risk Label")
plt.ylabel("Age (years)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/risk_by_age.png")
plt.close()

# 7.3 Risk by Income
plt.figure(figsize=(6.5, 4.5))
sns.boxplot(x="RiskLabel", y="AnnualIncome", data=df, order=order, palette=["#4C72B0", "#C44E52"])
plt.title("Annual Income by Risk Category")
plt.xlabel("Risk Label")
plt.ylabel("Annual Income")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/risk_by_income.png")
plt.close()

# 7.4 Risk by Employment Status
plt.figure(figsize=(7, 4.5))
sns.countplot(x="EmploymentStatus", hue="RiskLabel", data=df, hue_order=order,
              palette=["#4C72B0", "#C44E52"])
plt.title("Risk Category by Employment Status")
plt.xlabel("Employment Status")
plt.ylabel("Number of Applicants")
plt.legend(title="Risk Label")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/risk_by_employment.png")
plt.close()

# 7.5 Risk by Credit Score
plt.figure(figsize=(6.5, 4.5))
sns.boxplot(x="RiskLabel", y="CreditScore", data=df, order=order, palette=["#4C72B0", "#C44E52"])
plt.title("Credit Score by Risk Category")
plt.xlabel("Risk Label")
plt.ylabel("Credit Score")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/risk_by_credit_score.png")
plt.close()

# 7.6 Correlation heatmap (numeric features only, excluding leaky cols)
plt.figure(figsize=(14, 11))
corr_cols = [c for c in numeric_cols_all]
corr = df[corr_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
plt.title("Correlation Heatmap - Numeric Features")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png")
plt.close()

# 7.7 Boxplots for important numeric features
important_num = ["AnnualIncome", "CreditScore", "LoanAmount", "DebtToIncomeRatio",
                  "TotalAssets", "TotalLiabilities"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), important_num):
    sns.boxplot(y=df[col], ax=ax, color="#4C72B0")
    ax.set_title(col)
plt.suptitle("Boxplots of Key Numeric Features")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/numeric_boxplots.png")
plt.close()

# 7.8 Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), important_num):
    sns.histplot(df[col], kde=True, ax=ax, color="#55A868")
    ax.set_title(col)
plt.suptitle("Distributions of Key Numeric Features")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/numeric_distributions.png")
plt.close()

print("\nEDA plots saved to", PLOTS_DIR)

# ---------------------------------------------------------------------------
# 8. TRAIN / TEST SPLIT (before any preprocessing/scaling/SMOTE)
# ---------------------------------------------------------------------------
feature_cols = [c for c in df.columns if c not in LEAKY_COLS + DROP_COLS + ["RiskLabel"]]
X = df[feature_cols]
y = (df["RiskLabel"] == "High Risk").astype(int)  # 1 = High Risk, 0 = Low Risk

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

numeric_features = [c for c in numeric_cols_all if c in feature_cols]
categorical_features = [c for c in categorical_cols_all if c in feature_cols]
print("Final numeric features:", numeric_features)
print("Final categorical features:", categorical_features)

# ---------------------------------------------------------------------------
# 9. CLASS IMBALANCE ANALYSIS
# ---------------------------------------------------------------------------
class_counts = y_train.value_counts()
class_pct = y_train.value_counts(normalize=True) * 100
print("\nTrain class counts:\n", class_counts)
print("Train class %:\n", class_pct)
imbalance_ratio = class_counts.max() / class_counts.min()
print("Imbalance ratio:", imbalance_ratio)
USE_SMOTE = imbalance_ratio > 1.5  # median split -> near 50/50, expect False
print("Use SMOTE:", USE_SMOTE)

# ---------------------------------------------------------------------------
# 10. PREPROCESSING PIPELINE (fit only on training data, inside CV via Pipeline)
# ---------------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])


def make_pipeline(model, use_smote):
    steps = [("preprocessor", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("classifier", model))
    return ImbPipeline(steps=steps)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                              use_label_encoder=False),
}

# ---------------------------------------------------------------------------
# 11. BASELINE MODEL TRAINING & COMPARISON
# ---------------------------------------------------------------------------
def evaluate(pipe, X_te, y_te):
    preds = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    return {
        "Accuracy": accuracy_score(y_te, preds),
        "Precision": precision_score(y_te, preds),
        "Recall": recall_score(y_te, preds),
        "F1 Score": f1_score(y_te, preds),
        "ROC-AUC": roc_auc_score(y_te, proba),
    }

baseline_results = []
baseline_pipes = {}
for name, model in models.items():
    pipe = make_pipeline(model, USE_SMOTE)
    pipe.fit(X_train, y_train)
    metrics = evaluate(pipe, X_test, y_test)
    metrics["Model"] = name
    baseline_results.append(metrics)
    baseline_pipes[name] = pipe
    print(name, metrics)

comparison_df = pd.DataFrame(baseline_results)[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
comparison_df.to_csv("outputs/model_comparison.csv", index=False)
print("\nBaseline comparison:\n", comparison_df)

# ---------------------------------------------------------------------------
# 12. HYPERPARAMETER TUNING (RandomizedSearchCV, StratifiedKFold=5, scoring=f1)
# ---------------------------------------------------------------------------
# f1 is chosen because the target is balanced but we still want a metric that
# jointly rewards precision and recall (missing a High Risk applicant and
# wrongly flagging a Low Risk applicant both carry real costs).
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

param_grids = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__class_weight": [None, "balanced"],
    },
    "Random Forest": {
        "classifier__n_estimators": [200, 300, 400],
        "classifier__max_depth": [None, 8, 12, 16],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": [None, "balanced"],
    },
    "Gradient Boosting": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_depth": [2, 3, 4],
        "classifier__min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
        "classifier__reg_lambda": [0.5, 1.0, 2.0],
    },
}

tuned_pipes = {}
best_params_report = {}
for name, model in models.items():
    pipe = make_pipeline(model, USE_SMOTE)
    search = RandomizedSearchCV(
        pipe, param_distributions=param_grids[name], n_iter=15, scoring="f1",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    tuned_pipes[name] = search.best_estimator_
    best_params_report[name] = search.best_params_
    print(name, "best params:", search.best_params_, "best CV f1:", search.best_score_)

# ---------------------------------------------------------------------------
# 13. FINAL EVALUATION (tuned models, on untouched test set)
# ---------------------------------------------------------------------------
tuned_results = []
for name, pipe in tuned_pipes.items():
    metrics = evaluate(pipe, X_test, y_test)
    metrics["Model"] = name
    tuned_results.append(metrics)
tuned_comparison_df = pd.DataFrame(tuned_results)[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
print("\nTuned comparison:\n", tuned_comparison_df)
tuned_comparison_df.to_csv("outputs/model_comparison_tuned.csv", index=False)

# ---------------------------------------------------------------------------
# 14. MODEL SELECTION
# ---------------------------------------------------------------------------
# Rank primarily by F1 (balances precision/recall) with ROC-AUC as tie-break.
ranked = tuned_comparison_df.sort_values(["F1 Score", "ROC-AUC"], ascending=False)
best_model_name = ranked.iloc[0]["Model"]
best_pipe = tuned_pipes[best_model_name]
print("\nSelected best model:", best_model_name)
print(ranked)

# ---------------------------------------------------------------------------
# 15. CONFUSION MATRIX & ROC CURVE for the best model
# ---------------------------------------------------------------------------
preds = best_pipe.predict(X_test)
proba = best_pipe.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low Risk", "High Risk"], yticklabels=["Low Risk", "High Risk"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, proba)
auc_val = roc_auc_score(y_test, proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC = {auc_val:.3f})", color="#C44E52")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Best Model")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/roc_curve.png")
plt.close()

# ---------------------------------------------------------------------------
# 16. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
def get_feature_names(preprocessor):
    num_names = numeric_features
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    return list(num_names) + cat_names

fitted_preprocessor = best_pipe.named_steps["preprocessor"]
all_feature_names = get_feature_names(fitted_preprocessor)
classifier = best_pipe.named_steps["classifier"]

if hasattr(classifier, "feature_importances_"):
    importances = classifier.feature_importances_
    fi_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(15)
elif hasattr(classifier, "coef_"):
    importances = np.abs(classifier.coef_[0])
    fi_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(15)
else:
    fi_df = pd.DataFrame(columns=["feature", "importance"])

if not fi_df.empty:
    plt.figure(figsize=(8, 6))
    sns.barplot(x="importance", y="feature", data=fi_df, color="#4C72B0")
    plt.title(f"Top 15 Feature Importances - {best_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/feature_importance.png")
    plt.close()
    print("\nTop features:\n", fi_df)

# ---------------------------------------------------------------------------
# 17. SAVE FINAL MODEL
# ---------------------------------------------------------------------------
joblib.dump(best_pipe, f"{MODELS_DIR}/credit_risk_model.joblib")
print("\nSaved model to", f"{MODELS_DIR}/credit_risk_model.joblib")

# Save metadata needed by the Streamlit app
metadata = {
    "best_model_name": best_model_name,
    "risk_median_threshold": float(RISK_MEDIAN),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "feature_cols": feature_cols,
    "best_params": best_params_report[best_model_name],
    "categorical_options": {c: sorted(df[c].astype(str).unique().tolist()) for c in categorical_features},
    "numeric_ranges": {
        c: {"min": float(df[c].min()), "max": float(df[c].max()), "median": float(df[c].median())}
        for c in numeric_features
    },
}
with open(f"{MODELS_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nBUILD COMPLETE.")
