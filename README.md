# 📘 Safe-Haven Assets & Crash Prediction (2000–2025)

### **Master in Finance — Machine Learning for Asset Pricing**

---

## 🧭 1. Project Overview

This project analyzes whether several assets — **Gold**, **Dollar Index**, and **US 10Y Treasuries** — behave as **hedges** or **safe havens** during U.S. stock market downturns.

The study covers **2000–2025** and combines:

🔹 **Econometric models**  
- OLS regressions  
- Crisis-period regressions  
- Quantile regressions (Baur & Lucey 2010)

🔹 **Machine learning models**  
- Logistic Regression (baseline)  
- Random Forest (non-linear tree model)  
- XGBoost (boosted trees → best performer)

🔹 **Unsupervised learning**  
- K-Means clustering  
- PCA visualization

🔹 **Model interpretability**  
- SHAP feature analysis

🎯 **Goal:** Determine whether safe-haven assets protect portfolios during extreme S&P 500 declines.

---

## 📊 2. Dataset

Daily data for:

- **S&P 500 returns**
- **Gold returns**
- **Dollar Index returns**
- **US 10Y Treasury returns**
- **VIX index**

**Frequency:** Daily  
**Period:** 2000–2025  
**Source:** Pre-cleaned file `merged_daily_returns.csv`

---

## 🧪 3. Methodology

---

### ### **3.1 Econometric Approach**

#### **📌 (a) Global Hedge Regression (OLS)**  
Tests whether each asset hedges equity risk in normal periods.

#### **📌 (b) Crisis Regressions**  
OLS regressions restricted to:

- Global Financial Crisis (2007–09)  
- Euro Debt Crisis (2010–12)  
- COVID-19 Shock (2020)  
- Ukraine/Inflation Crisis (2022–23)

Objective: assess **safe-haven behavior**.

#### **📌 (c) Quantile Regression (Extreme 1–5% S&P 500 Days)**  
Tests if assets protect against **extreme market crashes**.

---

### ### **3.2 Machine Learning Approach**

#### **🤖 (a) Crash Prediction Classifiers**

Models used:

| Model | Role | Notes |
|-------|------|-------|
| Logistic Regression | Baseline | Linear, fast, interpretable |
| Random Forest | Tree ensemble | Strong non-linear model |
| XGBoost | Gradient boosting | **Best crash predictor** |

**Target variable:** bottom **5%** S&P 500 returns → `is_crash = 1`.

**Features:**

- VIX level and VIX changes  
- Gold/Dollar/US10Y returns  
- SP500 5–10 day momentum  

#### **📈 Metrics:**

- **Accuracy**  
- **Precision**  
- **Recall (Crash Detection)**  
- **ROC AUC**  

#### **📊 (b) SHAP Analysis**  
Explains model drivers:

- VIX level = most important  
- VIX volatility  
- SP500 momentum  
- Treasury market sensitivity  

#### **📈 (c) Safe-Haven Test on Predicted Crash Days**

For days predicted as crashes, we compute:

- Gold avg return  
- Dollar Index avg return  
- US10Y avg return  
- SP500 avg return (should be negative)

---

### ### **3.3 Unsupervised Learning (K-Means)**

Clustering on actual crash days (bottom 5%) identifies:

- **3 market crisis types**
- PCA shows separable crisis dynamics

Across all clusters, **US Treasuries consistently rise during stress**, proving strong safe-haven properties.

---

## 📁 4. Repository Structure

```text
project/
│
├── data/
│   └── merged_daily_returns.csv
│
├── notebooks/
│   └── final_project_analysis.ipynb
│
├── src/
│   ├── Global analysis.py
│   ├── Linear Regression.py
│   ├── XGBoost, Random Forest & K-Means.py
│   └── Importing and preparing data.py
│
├── results/
│   ├── global_regression_table.csv
│   ├── crisis_regression_table.csv
│   ├── quantile_safe_haven_table.csv
│   ├── shap_feature_importance.png
│   ├── kmeans_clusters.png
│   └── safe_haven_summary.txt
│
└── README.md   ← (this file)
