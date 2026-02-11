# Credit Risk Scorecard App – Phase 3

## Overview

This is the **third and final phase** of the Credit Risk Assessment Platform, designed to evaluate the creditworthiness of loan applicants. Built using **Python, Streamlit, and advanced machine learning techniques**, this version introduces **dynamic feature filtering, improved model explainability, and portfolio-level monitoring**—making it a significant upgrade over the previous phase.

---

## 🔹 Key Features

1. **Multiple Modeling Options**  
   - Logistic Regression with **WOE (Weight of Evidence)**  
   - XGBoost  
   - LightGBM  

2. **Advanced Feature Engineering & Selection**  
   - **WOE + IV calculation** for categorical features  
   - **Dynamic feature filtering** based on missing rate, IV, and correlation thresholds  
   - Target encoding with cross-validation for tree-based models  

3. **Individual Borrower Risk Assessment**  
   - Calculates **Probability of Default (PD)** for custom inputs  
   - Visualizes key metrics like **monthly installment, debt-to-income ratio, and disposable income**  
   - Provides actionable approval decisions:
     - ✅ Approved  
     - 🟡 Refer for manual review  
     - ❌ Rejected  

4. **Model Evaluation**  
   - Displays **ROC-AUC, Gini coefficient, KS statistic, and Population Stability Index (PSI)**  
   - Allows comparison between **filtered features** and **baseline (no filtering)**  

5. **Explainability & Transparency**  
   - Integrated **SHAP plots** for tree-based models  
   - Feature importance visualization enhances **model interpretability** for stakeholders  

6. **Portfolio Monitoring**  
   - Interactive filtering by **age, loan amount, and duration**  
   - Visualizes **PD distributions and PSI per decile**  
   - Enables **risk monitoring at the portfolio level**, critical for financial institutions  

---

## 🔹 Upgrades from Phase 2

- **Dynamic Feature Filtering:** Users can now filter numeric and categorical features based on missing rates, IV scores, and correlation thresholds. This reduces noise and improves model robustness.  
- **Information Value (IV) Insights:** IV scores allow users to quickly identify the most predictive categorical features.  
- **Baseline Comparison:** Ability to compare filtered models with unfiltered baseline models to assess the impact of feature selection.  
- **Improved Portfolio Analytics:** Phase 3 allows decile-level PSI visualization for portfolio-level risk monitoring.  
- **Enhanced UI Controls:** Multiple checkboxes and sliders for fine-tuning the modeling pipeline without changing the code.  

---

## 🔹 Why This Matters to Industry

- **Banks & Lending Institutions:** Enables **automated and transparent credit scoring** while maintaining regulatory compliance.  
- **FinTech Companies:** Helps rapidly evaluate applicants and manage risk at both individual and portfolio levels.  
- **Risk Management Teams:** Provides actionable insights through **metrics, explainability, and portfolio monitoring**, supporting informed decision-making.  
- **Recruiters & Employers:** Demonstrates practical skills in **machine learning, feature engineering, model explainability, and interactive dashboards**, which are highly sought after in financial data science roles.  

---

## 🔹 How to Run

1. Clone the repository:  
   ```bash
   git clone <your-repo-url>
   cd credit-risk-default-app
