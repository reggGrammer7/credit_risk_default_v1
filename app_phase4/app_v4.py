# =============================================
# ENHANCED CREDIT RISK SCORECARD APP
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import matplotlib.pyplot as plt
import joblib
import os
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

st.set_page_config(page_title="Enhanced Credit Risk App", layout="wide")

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    credit = fetch_openml(data_id=31, as_frame=True)
    df = credit.frame.copy()
    df.rename(columns={"class": "Default"}, inplace=True)
    df["Default"] = df["Default"].map({"bad": 1, "good": 0}).astype("int64")
    return df

df = load_data()

# ------------------------------
# Missing Value Handling
# ------------------------------
def handle_missing(df):
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif is_categorical_dtype(df[col]):
            if "Missing" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["Missing"])
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna("Missing")
    return df

df = handle_missing(df)

# ------------------------------
# Interaction Feature
# ------------------------------
df["income_loan_interaction"] = df["credit_amount"] * df["duration"]

# ------------------------------
# Train-Test Split
# ------------------------------
X = df.drop("Default", axis=1)
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------
# Binning Numeric Variables
# ------------------------------
def bin_numeric(X, cols, bins=5):
    X_binned = X.copy()
    for col in cols:
        X_binned[col] = pd.qcut(X[col], bins, duplicates='drop')
    return X_binned

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
X_train_binned = bin_numeric(X_train, numeric_cols)
X_test_binned = bin_numeric(X_test, numeric_cols)

# ------------------------------
# WOE Transformation
# ------------------------------
def compute_woe(df, feature, target):
    woe_df = df[[feature, target]].copy()
    woe_df[target] = pd.to_numeric(woe_df[target], errors="coerce").fillna(0).astype("int64")
    grouped = woe_df.groupby(feature, observed=False)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']
    grouped['dist_good'] = grouped['good'] / grouped['good'].sum()
    grouped['dist_bad'] = grouped['bad'] / grouped['bad'].sum()
    grouped['WOE'] = np.log((grouped['dist_good'] + 1e-6) / (grouped['dist_bad'] + 1e-6))
    return grouped['WOE'].to_dict()

woe_maps = {}
for col in X_train_binned.columns:
    woe_maps[col] = compute_woe(pd.concat([X_train_binned[[col]], y_train], axis=1), col, "Default")
    X_train[col] = X_train_binned[col].map(woe_maps[col])
    X_test[col] = X_test_binned[col].map(woe_maps[col])

# ------------------------------
# Model with GridSearchCV
# ------------------------------
model_path = "saved_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    base_model = LogisticRegression(max_iter=1000, class_weight='balanced')

    param_grid = {
        'C': [0.01, 0.1, 1, 10]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(base_model, param_grid, cv=cv, scoring='roc_auc')
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    joblib.dump(model, model_path)

# ------------------------------
# Predictions
# ------------------------------
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ------------------------------
# Metrics
# ------------------------------
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Confusion Matrix
threshold = 0.5
y_pred = (y_pred_proba >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# ------------------------------
# KS Threshold Optimization
# ------------------------------
def ks_threshold(y_true, y_prob):
    thresholds = np.linspace(0,1,100)
    ks_vals = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        ks = abs(recall_score(y_true, pred) - (1 - precision_score(y_true, pred)))
        ks_vals.append(ks)
    return thresholds[np.argmax(ks_vals)]

best_thresh = ks_threshold(y_test, y_pred_proba)

# ------------------------------
# Credit Score Conversion
# ------------------------------
def pd_to_score(pd, base_score=600, pdo=50):
    odds = (1-pd)/pd
    return base_score + pdo * np.log(odds)

scores = pd_to_score(y_pred_proba)

# ------------------------------
# Calibration Curve
# ------------------------------
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Enhanced Credit Risk Dashboard")

st.subheader("Model Performance")
st.write(f"ROC-AUC: {roc_auc:.3f}")
st.write(f"Precision: {precision:.3f}, Recall: {recall:.3f}")

st.write("Confusion Matrix")
st.write(cm)

# Calibration Plot
fig, ax = plt.subplots()
ax.plot(prob_pred, prob_true, marker='o')
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Optimized Threshold")
st.write(best_thresh)

st.subheader("Credit Score Distribution")
fig2, ax2 = plt.subplots()
ax2.hist(scores, bins=20)
st.pyplot(fig2)
