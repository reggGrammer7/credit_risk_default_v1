# app_credit_risk_full.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Scorecard App", layout="wide")
st.title("📊 Credit Risk Assessment Platform")

# ------------------------------
# 1️⃣ Load Data
# ------------------------------
@st.cache_data
def load_data():
    credit = fetch_openml(data_id=31, as_frame=True)
    df = credit.frame.copy()
    df.rename(columns={"class": "Default"}, inplace=True)
    df["Default"] = df["Default"].map({"bad": 1, "good": 0})
    numeric_cols = ["duration","credit_amount","installment_commitment","residence_since","age","existing_credits","num_dependents"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

df = load_data()
st.sidebar.header("Data Preview")
st.sidebar.dataframe(df.head())

# ------------------------------
# 2️⃣ Train-Test Split
# ------------------------------
X = df.drop("Default", axis=1)
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------
# 3️⃣ WOE for Logistic Regression
# ------------------------------
def compute_woe(df, feature, target):
    df[target] = df[target].astype(int)
    grouped = df.groupby(feature)[target].agg(['count','sum'])
    grouped.columns = ['total','bad']
    grouped['good'] = grouped['total'] - grouped['bad']
    grouped['dist_good'] = grouped['good'] / grouped['good'].sum()
    grouped['dist_bad'] = grouped['bad'] / grouped['bad'].sum()
    grouped['dist_good'] = grouped['dist_good'].replace(0, 0.0001)
    grouped['dist_bad'] = grouped['dist_bad'].replace(0, 0.0001)
    grouped['WOE'] = np.log(grouped['dist_good']/grouped['dist_bad'])
    return grouped['WOE'].to_dict()

cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
woe_maps = {}
for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    woe_maps[col] = compute_woe(pd.concat([X_train, y_train], axis=1), col, 'Default')
    X_train[col+'_woe'] = X_train[col].map(woe_maps[col])
    X_test[col+'_woe'] = X_test[col].map(woe_maps[col])

woe_features = [c+'_woe' for c in cat_cols] + ["age","credit_amount","duration"]

# ------------------------------
# 4️⃣ Target Encoding for Trees
# ------------------------------
def target_encode_cv(X, y, cols, n_splits=5):
    X_enc = X.copy()
    y_num = y.astype(int)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for col in cols:
        X_enc[col] = X_enc[col].astype(str)
        encoded_col = pd.Series(index=X.index, dtype=float)
        for train_idx, val_idx in kf.split(X, y_num):
            train_col = X.iloc[train_idx][col].astype(str)
            mapping = y_num.iloc[train_idx].groupby(train_col).mean()
            encoded_col.iloc[val_idx] = X.iloc[val_idx][col].astype(str).map(mapping)
        encoded_col.fillna(y_num.mean(), inplace=True)
        X_enc[col+'_te'] = encoded_col
    return X_enc

tree_cols = cat_cols
X_train_te = target_encode_cv(X_train, y_train, tree_cols)
X_test_te = target_encode_cv(X_test, y_test, tree_cols)
tree_features = [c+'_te' for c in tree_cols] + ["age","credit_amount","duration"]

# ------------------------------
# 5️⃣ Sidebar: Model Selection
# ------------------------------
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression (WOE)","XGBoost","LightGBM"]
)

# ------------------------------
# 6️⃣ Train Model
# ------------------------------
if model_choice=="Logistic Regression (WOE)":
    model = LogisticRegression(max_iter=1000)
    X_train_model = X_train[woe_features]
    X_test_model = X_test[woe_features]
elif model_choice=="XGBoost":
    model = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42)
    X_train_model = X_train_te[tree_features]
    X_test_model = X_test_te[tree_features]
elif model_choice=="LightGBM":
    model = LGBMClassifier(n_estimators=200, random_state=42)
    X_train_model = X_train_te[tree_features]
    X_test_model = X_test_te[tree_features]

model.fit(X_train_model, y_train)
y_pred_proba = model.predict_proba(X_test_model)[:,1]

# ------------------------------
# 7️⃣ Metrics: ROC-AUC, KS, GINI
# ------------------------------
roc_auc = roc_auc_score(y_test, y_pred_proba)
gini = 2*roc_auc -1

def ks(y_true, y_score):
    y_num = pd.Series(y_true).astype(int)
    data = pd.DataFrame({"y": y_num, "score": y_score})
    data = data.sort_values("score")
    data["cum_bad"] = data["y"].cumsum()/data["y"].sum()
    data["cum_good"] = ((1-data["y"]).cumsum())/((1-data["y"]).sum())
    return max(abs(data["cum_bad"]-data["cum_good"]))

ks_val = ks(y_test, y_pred_proba)

# ------------------------------
# 8️⃣ PSI Calculation
# ------------------------------
def psi(expected, actual, bins=10):
    expected_num = pd.Series(expected, dtype="float64")
    actual_num = pd.Series(actual, dtype="float64")
    expected_bins, bin_edges = pd.qcut(expected_num, bins, duplicates="drop", retbins=True)
    actual_bins = pd.cut(actual_num, bins=bin_edges, include_lowest=True)
    psi_val = 0.0
    for b in expected_bins.cat.categories:
        e_perc = (expected_bins == b).sum() / len(expected_bins)
        a_perc = (actual_bins == b).sum() / len(actual_bins)
        psi_val += (e_perc - a_perc) * np.log((e_perc + 1e-6) / (a_perc + 1e-6))
    return psi_val

psi_val = psi(y_train, y_pred_proba)

# ------------------------------
# 9️⃣ Tabs for Streamlit
# ------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Applicant Risk",
    "📈 Model Evaluation",
    "🧠 SHAP Explainability",
    "⚡ Portfolio Monitoring"
])

# ------------------------------
# TAB 1: Individual Borrower Risk
# ------------------------------
with tab1:
    st.subheader("Individual Credit Risk Assessment")
    age = st.number_input("Age", 18, 100, 30)
    credit_amount = st.number_input("Loan Amount", 500, 50000, 5000)
    duration = st.number_input("Loan Duration (months)", 6, 72, 24)
    monthly_income = st.number_input("Monthly Income", 200, 20000, 3000)
    monthly_obligations = st.number_input("Existing Monthly Debt Obligations", 0, 15000, 500)

    monthly_installment = credit_amount/duration
    dti = monthly_installment/monthly_income
    disposable_income = monthly_income - monthly_obligations

    col1, col2, col3 = st.columns(3)
    col1.metric("Monthly Installment", f"${monthly_installment:,.2f}")
    col2.metric("Debt-to-Income Ratio", f"{dti:.1%}")
    col3.metric("Disposable Income", f"${disposable_income:,.2f}")

    user_data = pd.DataFrame({"age":[age],"credit_amount":[credit_amount],"duration":[duration]})
    for f in X_train_model.columns:
        if f not in user_data.columns:
            if pd.api.types.is_numeric_dtype(X_train_model[f]):
                user_data[f] = X_train_model[f].median()
            else:
                user_data[f] = X_train_model[f].mode()[0]

    pd_score = model.predict_proba(user_data[X_train_model.columns])[0,1]
    st.metric("Probability of Default", f"{pd_score:.2%}")

    if pd_score<0.30 and dti<0.40:
        st.success("✅ APPROVED")
    elif pd_score<0.60 and dti<0.60:
        st.warning("🟡 REFER FOR MANUAL REVIEW")
    else:
        st.error("❌ REJECTED")

# ------------------------------
# TAB 2: Model Evaluation
# ------------------------------
with tab2:
    st.subheader(f"{model_choice} Performance")
    st.write(f"ROC-AUC: {roc_auc:.3f}, GINI: {gini:.3f}, KS: {ks_val:.3f}")
    st.write(f"Population Stability Index (PSI): {psi_val:.4f}")

# ------------------------------
# TAB 3: SHAP Explainability
# ------------------------------
with tab3:
    if model_choice in ["XGBoost","LightGBM"]:
        st.subheader("SHAP Feature Importance")
        explainer = shap.Explainer(model, X_train_model)
        shap_values = explainer(X_test_model)
        fig, ax = plt.subplots(figsize=(10,6))
        shap.summary_plot(shap_values, X_test_model, show=False)
        st.pyplot(fig)
    else:
        st.info("SHAP is only available for tree-based models.")

# ------------------------------
# TAB 4: Portfolio Monitoring (Interactive)
# ------------------------------
with tab4:
    st.subheader("Portfolio Monitoring")

    st.write("### Filter Portfolio")
    age_min, age_max = st.slider("Age range", 18, 100, (18,100))
    loan_min, loan_max = st.slider("Loan Amount", 500, 50000, (500,50000))
    duration_min, duration_max = st.slider("Duration (months)", 6, 72, (6,72))

    portfolio_mask = (
        (X_test_model["age"] >= age_min) & (X_test_model["age"] <= age_max) &
        (X_test_model["credit_amount"] >= loan_min) & (X_test_model["credit_amount"] <= loan_max) &
        (X_test_model["duration"] >= duration_min) & (X_test_model["duration"] <= duration_max)
    )

    filtered_y_pred = y_pred_proba[portfolio_mask.values]
    filtered_y_true = y_test.iloc[portfolio_mask.values]

    # PD distribution
    st.write("Distribution of predicted default probabilities:")
    fig1, ax1 = plt.subplots()
    ax1.hist(filtered_y_pred, bins=20, edgecolor='k')
    ax1.set_xlabel("Predicted Probability of Default (PD)")
    ax1.set_ylabel("Number of Borrowers")
    st.pyplot(fig1)

    # PSI per decile
    st.write("Population Stability Index (PSI) per decile:")
    n_bins = 10
    y_train_bins = pd.qcut(y_train.rank(method='first'), n_bins, labels=False)
    filtered_pred_bins = pd.qcut(pd.Series(filtered_y_pred).rank(method='first'), n_bins, labels=False)

    psi_bins = []
    for i in range(n_bins):
        e_perc = (y_train_bins==i).sum()/len(y_train_bins)
        a_perc = (filtered_pred_bins==i).sum()/len(filtered_pred_bins)
        psi_bins.append((e_perc - a_perc)*np.log((e_perc+1e-6)/(a_perc+1e-6)))

    fig2, ax2 = plt.subplots()
    ax2.bar(range(1,n_bins+1), psi_bins)
    ax2.set_xlabel("Decile")
    ax2.set_ylabel("PSI Contribution")
    st.pyplot(fig2)
