# Hedge and Safe-Haven Assets (2000–2025)

---

## Research Question
Are gold, the US dollar, and US Treasury bonds hedges and/or safe havens against S&P 500 market downturns during major financial crises between 2000 and 2025?

---

## Project Overview

This project studies defensive asset behaviour using **daily financial data from 2000 to 2025**, combining econometric methods and machine-learning techniques.
The analysis distinguishes between **hedge properties** (average market conditions) and **safe-haven behaviour** (extreme equity downturns).

The methodology includes:

- **Econometrics (hedge vs safe haven tests)**
  - Global linear regressions (hedge behaviour in normal times)
  - Crisis-specific regressions (behaviour during major crises)
  - Quantile regressions (tail-risk and extreme downside days)
  
- **Supervised machine learning (crash detection)**
  - XGBoost, Random Forest, LSTM (time-series deep learning)
  - Performance emphasis on **recall** (missing a crash is costlier than false alarms)
  
- **Unsupervised learning (regime structure)**
  - PCA (visualisation of stress regimes)
  - K-means (clustering of extreme crash days)
  
- **Explainability**
  - SHAP (drivers of crash predictions, mainly VIX-related features)

## Dataset

Daily market data (Yahoo Finance), including:
- S&P 500, Gold, DXY, US10Y (yield-based proxy), VIX  
Main processed file used in the pipeline:
- `data/merged_daily_returns.csv`

---

## Setup

#### Create environment -> ⚠️ Requires Python 3.12 (not compatible with 3.13)
python3.12 -m venv .venv

#### Activate (Mac/Linux)
source .venv/bin/activate

#### Install dependencies
pip install -r requirements.txt

---

## Usage

python main.py

Expected output : Comparative analysis of asset returns using econometric and machine-learning methods

---

## Project Structure

```text
SafeHavenAnalysis/
├── data/
│   └── merged_daily_returns.csv
├── notebooks/
│   └── (optional analysis notebooks)
├── results/
│   └── (exported tables, figures, model outputs)
├── src/
│   ├── importing_and_preparing_data.py
│   ├── global_analysis.py
│   ├── linear_regression.py
│   └── ml_crash_detection_and_clustering.py
├── main.py
├── requirements.txt
├── AI_Usage.md
└── README.md
``` 
---

## Results

The analysis highlights heterogeneous safe-haven properties across assets, depending on market regimes and tail events:

- **US Treasuries**: strongest hedge and most robust safe haven (strong protection in crises and deep tails).
- **US Dollar (DXY)**: weak hedge on average but **systematic safe haven** during stress.
- **Gold**: **conditional safe haven** (protects in some crisis regimes, weaker and more regime-dependent).

---

## Requirements

- Python 3.12.12
- pandas, numpy
- statsmodels
- scikit-learn
- xgboost
- tensorflow
- shap
- matplotlib, seaborn