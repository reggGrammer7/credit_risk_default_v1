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
# 3️⃣ WOE for Logistic Regression + IV
# ------------------------------
def compute_woe_iv(df, feature, target):
    df = df.copy()
    df[target] = df[target].astype(int)
    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']
    grouped['dist_good'] = grouped['good'] / grouped['good'].sum()
    grouped['dist_bad'] = grouped['bad'] / grouped['bad'].sum()
    grouped['dist_good'] = grouped['dist_good'].replace(0, 0.0001)
    grouped['dist_bad'] = grouped['dist_bad'].replace(0, 0.0001)
    grouped['WOE'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
    grouped['IV'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['WOE']
    iv = grouped['IV'].sum()
    return grouped['WOE'].to_dict(), iv

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

# ------------------------------
# 3.1️⃣ Sidebar: Feature Filtering
# ------------------------------
st.sidebar.header("Feature Filtering")
enable_filtering = st.sidebar.checkbox("Enable feature filtering", value=True)
show_iv_table = st.sidebar.checkbox("Show IV table", value=False)
show_baseline = st.sidebar.checkbox("Show no-filtering baseline", value=False)
missing_thresh = st.sidebar.slider("Max missing rate", 0.0, 0.6, 0.4, 0.05)
iv_thresh = st.sidebar.slider("Min IV (categorical)", 0.0, 0.3, 0.02, 0.01)
corr_thresh = st.sidebar.slider("Max correlation (numeric)", 0.7, 0.99, 0.9, 0.01)

def filter_numeric_features(X_df, base_cols, missing_limit, corr_limit):
    cols = [c for c in base_cols if c in X_df.columns]
    missing_rate = X_df[cols].isna().mean()
    cols = [c for c in cols if missing_rate[c] <= missing_limit]
    if len(cols) <= 1:
        return cols
    corr = X_df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [column for column in upper.columns if any(upper[column] > corr_limit)]
    return [c for c in cols if c not in drop]

def filter_categorical_features(X_df, y_series, cols, missing_limit, iv_limit):
    keep = []
    iv_map = {}
    missing_rate = X_df[cols].isna().mean()
    for col in cols:
        if missing_rate[col] > missing_limit:
            continue
        X_col = X_df[col].astype(str)
        woe_map, iv = compute_woe_iv(pd.concat([X_col, y_series], axis=1), col, "Default")
        iv_map[col] = iv
        if iv >= iv_limit:
            keep.append(col)
    return keep, iv_map

if enable_filtering:
    filtered_num_cols = filter_numeric_features(X_train, numeric_cols, missing_thresh, corr_thresh)
    filtered_cat_cols, iv_scores = filter_categorical_features(X_train, y_train, cat_cols, missing_thresh, iv_thresh)
else:
    filtered_num_cols = numeric_cols
    filtered_cat_cols = cat_cols
    iv_scores = {}

if show_iv_table:
    if not iv_scores:
        _, iv_scores = filter_categorical_features(X_train, y_train, cat_cols, missing_thresh, 0.0)
    iv_df = (
        pd.DataFrame({"feature": list(iv_scores.keys()), "iv": list(iv_scores.values())})
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
    st.sidebar.dataframe(iv_df)

woe_maps = {}
for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    woe_map, _iv = compute_woe_iv(pd.concat([X_train[[col]], y_train], axis=1), col, 'Default')
    woe_maps[col] = woe_map
    X_train[col + '_woe'] = X_train[col].map(woe_maps[col])
    X_test[col + '_woe'] = X_test[col].map(woe_maps[col])

woe_features = [c + '_woe' for c in filtered_cat_cols] + filtered_num_cols

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

tree_cols = filtered_cat_cols
X_train_te = target_encode_cv(X_train, y_train, tree_cols)
X_test_te = target_encode_cv(X_test, y_test, tree_cols)
tree_features = [c+'_te' for c in tree_cols] + filtered_num_cols

if show_baseline:
    X_train_te_all = target_encode_cv(X_train, y_train, cat_cols)
    X_test_te_all = target_encode_cv(X_test, y_test, cat_cols)
    baseline_woe_features = [c + '_woe' for c in cat_cols] + numeric_cols
    baseline_tree_features = [c + '_te' for c in cat_cols] + numeric_cols

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
# 8.1️⃣ Baseline Metrics (No Filtering)
# ------------------------------
if show_baseline:
    if model_choice == "Logistic Regression (WOE)":
        baseline_model = LogisticRegression(max_iter=1000)
        X_train_base = X_train[baseline_woe_features]
        X_test_base = X_test[baseline_woe_features]
    elif model_choice == "XGBoost":
        baseline_model = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42)
        X_train_base = X_train_te_all[baseline_tree_features]
        X_test_base = X_test_te_all[baseline_tree_features]
    else:
        baseline_model = LGBMClassifier(n_estimators=200, random_state=42)
        X_train_base = X_train_te_all[baseline_tree_features]
        X_test_base = X_test_te_all[baseline_tree_features]

    baseline_model.fit(X_train_base, y_train)
    baseline_pred_proba = baseline_model.predict_proba(X_test_base)[:, 1]
    baseline_roc_auc = roc_auc_score(y_test, baseline_pred_proba)
    baseline_gini = 2 * baseline_roc_auc - 1
    baseline_ks = ks(y_test, baseline_pred_proba)
    baseline_psi = psi(y_train, baseline_pred_proba)

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
    if enable_filtering:
        st.write(f"Selected numeric features: {filtered_num_cols}")
        st.write(f"Selected categorical features (WOE/TE): {filtered_cat_cols}")
    if show_baseline:
        st.write("Baseline (no filtering):")
        st.write(f"ROC-AUC: {baseline_roc_auc:.3f}, GINI: {baseline_gini:.3f}, KS: {baseline_ks:.3f}")
        st.write(f"Population Stability Index (PSI): {baseline_psi:.4f}")

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
