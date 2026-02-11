# 📊 Credit Risk Default Scoring App

A production-style **Credit Risk Default Scoring application** built with **Streamlit**, designed to simulate how financial institutions assess borrower default risk using classical scorecard techniques and modern machine learning models.

This app represents **Phase 1** of the project and focuses on:

* Robust data preprocessing
* Multiple model pipelines
* Industry-standard risk metrics
* Explainability and decision support

---

## 🚀 Overview

The application uses the **German Credit Dataset** (OpenML) to predict the **Probability of Default (PD)** for borrowers. Users can:

* Explore the dataset interactively
* Train and compare multiple credit risk models
* Evaluate models using banking-standard metrics (ROC-AUC, Gini, KS, PSI)
* Interpret model behavior using SHAP values
* Perform **individual borrower credit assessments** with automated credit decisions

---

## 🧠 Models Implemented

The app supports **three modeling approaches**, each reflecting real-world credit risk practices:

### 1️⃣ Logistic Regression (Scorecard-style)

* Uses **Weight of Evidence (WOE)** encoding for categorical variables
* Interpretable and regulator-friendly
* Commonly used in traditional credit scorecards

### 2️⃣ XGBoost

* Gradient boosting tree-based model
* Uses **cross-validated target encoding**
* Captures nonlinear relationships and interactions

### 3️⃣ LightGBM

* High-performance gradient boosting model
* Efficient and scalable for large datasets
* Also uses target encoding

Model selection is handled dynamically through the sidebar.

---

## 📂 Dataset

* **Source**: OpenML (German Credit Dataset)
* **Target Variable**: `Default`

  * `1` → Bad credit (default)
  * `0` → Good credit (non-default)

### Key Feature Groups

* Demographics (e.g., age)
* Loan characteristics (amount, duration)
* Credit history
* Existing financial obligations

---

## ⚙️ Data Processing Pipeline

### 🔹 Numeric Features

Converted to numeric and cleaned using coercion.

### 🔹 Categorical Encoding

Two parallel encoding strategies are used:

#### ➤ Weight of Evidence (WOE)

* Applied to Logistic Regression only
* Ensures monotonicity and interpretability
* Aligns with Basel and regulatory expectations

#### ➤ Target Encoding (Cross-Validated)

* Applied to tree-based models
* Prevents target leakage using Stratified K-Fold CV

---

## 📈 Model Evaluation Metrics

The app reports **industry-standard credit risk metrics**:

* **ROC-AUC** – Overall discriminatory power
* **Gini Coefficient** – Model strength (2×AUC − 1)
* **KS Statistic** – Maximum separation between good and bad borrowers
* **PSI (Population Stability Index)** – Detects population drift

These metrics are critical in real-world model validation and monitoring.

---

## 🔍 Model Explainability (SHAP)

For **XGBoost and LightGBM**, the app generates:

* **SHAP summary plots**
* Feature-level contribution analysis
* Transparent insight into how variables affect default risk

This is essential for:

* Model governance
* Regulatory audits
* Stakeholder trust

---

## 👤 Individual Borrower Credit Assessment

The app allows users to simulate a **real credit application** by entering:

* Age
* Loan amount
* Loan duration
* Monthly income
* Existing debt obligations

### Automatically computed indicators:

* Monthly installment
* Debt-to-Income (DTI) ratio
* Disposable income

### Final Outputs:

* **Probability of Default (PD)**
* Automated credit decision:

  * ✅ Approved
  * 🟡 Refer for manual review
  * ❌ Rejected

Decision logic combines **model PD** with **financial affordability rules**, mirroring real lending policies.

---

## 🖥️ Tech Stack

* **Frontend**: Streamlit
* **Data**: Pandas, NumPy
* **Models**: Scikit-learn, XGBoost, LightGBM
* **Explainability**: SHAP
* **Visualization**: Matplotlib

---

## ▶️ How to Run the App

```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn xgboost lightgbm shap matplotlib

# Run the app
streamlit run app_credit_risk_full.py
```

---

## 📌 Project Status

✅ **Phase 1 Completed**

Current phase focuses on:

* Model development
* Evaluation
* Explainability
* Decision logic

---

## 🔮 Planned Enhancements (Next Phases)

* Credit score scaling (PDO-based scorecards)
* Feature binning & monotonic constraints
* Reject inference
* Model calibration
* Model monitoring dashboard
* Database & API integration
* Deployment-ready architecture

---

## 👨‍💻 Author

**Regent Yao Agbemafle**
Data Analyst | Data Scientist | Credit Risk & ML Enthusiast

---

## ⭐ Final Note

This project is designed to reflect **real-world credit risk modeling workflows**, blending regulatory best practices with modern machine learning techniques. It serves as a strong foundation for advanced risk analytics, research, or production deployment.

Feel free to fork, explore, and build upon it 🚀
