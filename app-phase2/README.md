# 📊 Credit Risk Assessment Platform — Phase 2

## 🔥 Project Overview

**Phase 2** upgrades the Credit Risk Default App from a *model-centric prototype* into a **portfolio-aware, decision-driven credit risk platform**. While Phase 1 focused on model development and evaluation, Phase 2 introduces **user experience improvements, monitoring capabilities, and governance-style analytics** that closely resemble real-world banking risk systems.

Built with **Streamlit**, this phase integrates modeling, explainability, borrower-level decisioning, and portfolio monitoring into a single interactive application.

---

## 🚀 What’s New in Phase 2 (Why This Is an Upgrade)

Phase 2 is not just a refactor — it is a **functional leap** toward production-grade credit risk analytics.

### ✅ Key Upgrades from Phase 1

| Area             | Phase 1             | Phase 2 (Upgrade)                                      |
| ---------------- | ------------------- | ------------------------------------------------------ |
| UI / UX          | Single-page flow    | **Tabbed, modular interface**                          |
| Decisioning      | PD output           | **PD + affordability-based credit decisions**          |
| Explainability   | Global SHAP         | **Integrated SHAP tab with model gating**              |
| Monitoring       | Static PSI          | **Interactive portfolio-level PSI & PD distributions** |
| Risk Perspective | Individual borrower | **Borrower + portfolio risk views**                    |
| Governance       | Metrics only        | **Metrics + stability diagnostics**                    |

This phase demonstrates a deeper understanding of **model lifecycle management**, not just model training.

---

## 🧠 Modeling Framework

The platform supports **three industry-relevant models**, selectable dynamically:

### 1️⃣ Logistic Regression (WOE Scorecard)

* Weight of Evidence (WOE) encoding for categorical variables
* High interpretability and regulatory alignment
* Commonly used in Basel-compliant scorecards

### 2️⃣ XGBoost

* Gradient boosting decision trees
* Cross-validated target encoding to avoid leakage
* Captures nonlinear effects and feature interactions

### 3️⃣ LightGBM

* Efficient, scalable gradient boosting framework
* Optimized for performance and speed
* Suitable for large credit portfolios

---

## ⚙️ Data & Feature Engineering

* **Dataset**: German Credit Dataset (OpenML)
* Target variable: `Default` (1 = bad, 0 = good)

### Encoding Strategy

* **WOE encoding** → Logistic Regression only
* **Cross-validated Target Encoding** → Tree-based models

This dual-pipeline design mirrors **real bank architectures**, where interpretability and predictive power are balanced.

---

## 📈 Model Evaluation Metrics

Phase 2 reports **banking-standard validation metrics**:

* **ROC-AUC** – Model discrimination
* **Gini Coefficient** – Credit model strength
* **KS Statistic** – Separation between good and bad borrowers
* **Population Stability Index (PSI)** – Population drift detection

Metrics are surfaced in a **dedicated evaluation tab**, reinforcing governance and audit readiness.

---

## 🧠 Explainability & Transparency (SHAP)

For XGBoost and LightGBM models, Phase 2 integrates:

* SHAP summary plots in a dedicated tab
* Feature contribution analysis
* Model-specific explainability controls

This aligns with regulatory expectations around **model transparency and explainability**.

---

## 👤 Individual Borrower Risk Assessment

Users can simulate a real credit application by entering borrower details:

* Age
* Loan amount
* Loan duration
* Monthly income
* Existing debt obligations

### Automatically derived indicators:

* Monthly installment
* Debt-to-Income (DTI) ratio
* Disposable income

### Outputs:

* **Probability of Default (PD)**
* Automated credit decision:

  * ✅ Approved
  * 🟡 Refer for manual review
  * ❌ Rejected

This combines **statistical risk** with **affordability rules**, reflecting real lending policies.

---

## ⚡ Portfolio Monitoring & Risk Oversight (NEW)

A major Phase 2 enhancement is the **Portfolio Monitoring tab**, enabling:

* Interactive filtering by age, loan amount, and duration
* Visualization of PD distributions across borrower segments
* **Decile-level PSI analysis** for population stability

This introduces a **portfolio manager’s perspective**, moving beyond single-loan decisioning to systemic risk oversight.

---

## 🖥️ Application Structure (Phase 2)

The app is organized into four functional tabs:

1. **Applicant Risk** – Individual borrower scoring and decisions
2. **Model Evaluation** – Performance and stability metrics
3. **SHAP Explainability** – Model transparency
4. **Portfolio Monitoring** – Drift and distribution analysis

This modular layout reflects how credit risk tools are used in practice across teams.

---

## 🧰 Tech Stack

* **Frontend**: Streamlit
* **Data Processing**: Pandas, NumPy
* **Models**: Scikit-learn, XGBoost, LightGBM
* **Explainability**: SHAP
* **Visualization**: Matplotlib

---

## ▶️ How to Run

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm shap matplotlib
streamlit run app_credit_risk_full.py
```

---

## 🎯 Why Recruiters Should Care

This project demonstrates:

* End-to-end credit risk modeling
* Regulatory-aware feature engineering (WOE, KS, PSI)
* Model explainability and governance readiness
* Portfolio-level risk monitoring
* Clean, interactive decision-support tooling

It bridges the gap between **academic machine learning** and **real-world financial risk systems**.

---

## 🔮 Next Enhancements (Phase 3)

* Scorecard scaling (PDO, odds-to-score)
* Model calibration & backtesting
* Time-based population drift analysis
* Model versioning & challenger frameworks
* API & database integration

---

## 👨‍💻 Author

**Regent Yao Agbemafle**
Data Analyst | Data Scientist | Credit Risk & Machine Learning

---

⭐ *Phase 2 elevates this project from a modeling exercise to a full credit risk decision and monitoring platform.*
