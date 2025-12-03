
# — LINEAR REGRESSIONS (GLOBAL, CRISIS, QUANTILES)

# ===============================================================
# GLOBAL IMPORTS & CONFIGURATION
# ===============================================================

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# Load corrected daily returns file
returns = pd.read_csv("data/merged_daily_returns.csv")
returns["Date"] = pd.to_datetime(returns["Date"])
returns = returns.set_index("Date").dropna()



# ===============================================================
# 1) 📊 GLOBAL LINEAR REGRESSION (HEDGE TEST)
# ===============================================================

# List of assets to analyze
assets = {
    "Gold": "Gold_Return",
    "Dollar Index (DXY)": "Dollar_Index_Return",
    "US 10Y Treasury": "US10Y_Return"
}

results = []

for asset_name, y_col in assets.items():

    # Dependent variable
    Y = returns[y_col]

    # Single explanatory variable: S&P 500 return
    X = returns[["SP500_Return"]]
    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()

    beta = model.params["SP500_Return"]
    pval = model.pvalues["SP500_Return"]
    r2 = model.rsquared

    results.append([
        asset_name,
        beta,
        pval,
        r2
    ])

# Final regression table
global_reg_table = pd.DataFrame(
    results,
    columns=["Asset", "Beta_SP500", "Pvalue_SP500", "R_squared"]
)

print("\n📊 GLOBAL REGRESSION — Hedge Test (Corrected)\n")
print(global_reg_table.round(4))



# ===============================================================
# 2) 📉 LINEAR REGRESSION BY CRISIS PERIODS (vs S&P 500 only)
# ===============================================================

# Crisis windows (exact dates used in your analysis)
crises = {
    "Global_Financial_Crisis": ("2007-10-01", "2009-03-31"),
    "Euro_Debt_Crisis": ("2010-05-01", "2012-06-30"),
    "COVID19_Shock": ("2020-02-01", "2020-12-31"),
    "Ukraine_Inflation": ("2022-01-01", "2023-12-31")
}

# Models: only S&P 500 as explanatory variable
models = {
    "Gold": "Gold_Return",
    "Dollar Index (DXY)": "Dollar_Index_Return",
    "US 10Y Treasury": "US10Y_Return"
}

rows = []

for asset_name, y_col in models.items():
    for crisis_name, (start, end) in crises.items():

        # Extract crisis window
        sub = returns.loc[start:end]
        if len(sub) < 10:  # safety check
            continue

        X = sm.add_constant(sub["SP500_Return"])
        y = sub[y_col]

        model = sm.OLS(y, X).fit()

        beta_sp500 = model.params["SP500_Return"]
        pval_sp500 = model.pvalues["SP500_Return"]

        rows.append({
            "Asset": asset_name,
            "Crisis": crisis_name,
            "Beta_SP500": round(beta_sp500, 4),
            "Pvalue_SP500": round(pval_sp500, 4),
            "R_squared": round(model.rsquared, 4)
        })

# Final table identical to your structure
crisis_reg_table = pd.DataFrame(
    rows,
    columns=["Asset", "Crisis", "Beta_SP500", "Pvalue_SP500", "R_squared"]
)

crisis_reg_table = crisis_reg_table.sort_values(["Asset", "Crisis"]).reset_index(drop=True)

print("\n📉 CRISIS LINEAR REGRESSIONS — Behavior During Crises (vs S&P 500 only)\n")
print(crisis_reg_table.round(4))



# ===============================================================
# 3) 🟦 SAFE HAVEN TEST — QUANTILE REGRESSIONS (Baur & Lucey 2010)
# ===============================================================

# Assets to test
assets = {
    "Gold": "Gold_Return",
    "Dollar Index (DXY)": "Dollar_Index_Return",
    "US 10Y Treasury": "US10Y_Return"
}

# Worst S&P500 days: 1%, 2.5%, 5%
quantiles = [0.01, 0.025, 0.05]
rows = []

for asset_name, y_col in assets.items():
    y = returns[y_col]
    sp500 = returns["SP500_Return"]

    for q in quantiles:

        # Threshold for extreme days
        threshold = sp500.quantile(q)

        # Interaction term: SP500 * extreme indicator
        sp500_extreme = sp500.where(sp500 < threshold, 0)

        X = pd.concat([sp500, sp500_extreme], axis=1)
        X.columns = ["SP500", "SP500_extreme"]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        beta_normal = model.params["SP500"]
        beta_extreme = model.params["SP500_extreme"]  # key coefficient
        total_beta_crash = beta_normal + beta_extreme

        p_extreme = model.pvalues["SP500_extreme"]

        rows.append({
            "Asset": asset_name,
            "Quantile": f"{int(q*100)}%",
            "β normal": round(beta_normal, 4),
            "β extreme (added)": round(beta_extreme, 4),
            "β total crash": round(total_beta_crash, 4),
            "p-value extreme": round(p_extreme, 4),
            "R²": round(model.rsquared, 4)
        })

# Final table
safehaven_table = pd.DataFrame(rows)
print("\n🟦 SAFE HAVEN TEST – Quantile approach (Baur & Lucey 2010)\n")
print(safehaven_table.round(4))



# ===============================================================
# 4) SCATTERPLOTS — PERFORMANCE DURING EXTREME S&P500 DECLINES
# ===============================================================

# Clean again for safety
returns = returns.dropna()

# Quantile to highlight (5% worst days)
q = 0.05
threshold = returns['SP500_Return'].quantile(q)
normal = returns['SP500_Return'] >= threshold
extreme = returns['SP500_Return'] < threshold


# -------------------------------------------------
# GOLD vs S&P500
# -------------------------------------------------
plt.figure(figsize=(10,7))

plt.scatter(
    returns.loc[normal, 'SP500_Return'],
    returns.loc[normal, 'Gold_Return'],
    c='lightgray', alpha=0.7, s=30, label='Normal days (95%)'
)

plt.scatter(
    returns.loc[extreme, 'SP500_Return'],
    returns.loc[extreme, 'Gold_Return'],
    c='#E69F00', edgecolors='black', linewidth=0.8, s=70, alpha=0.95,
    label='Worst 5% S&P500 days'
)

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('S&P 500 — Daily Return')
plt.ylabel('Gold — Daily Return')
plt.title('Gold vs S&P 500 – Extreme Crash Behavior (2000–2025)', fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# -------------------------------------------------
# DXY vs S&P500
# -------------------------------------------------
plt.figure(figsize=(10,7))

plt.scatter(
    returns.loc[normal, 'SP500_Return'],
    returns.loc[normal, 'Dollar_Index_Return'],
    c='lightgray', alpha=0.7, s=30, label='Normal days (95%)'
)

plt.scatter(
    returns.loc[extreme, 'SP500_Return'],
    returns.loc[extreme, 'Dollar_Index_Return'],
    c='#56B4E9', edgecolors='black', linewidth=0.8, s=70, alpha=0.95,
    label='Worst 5% S&P500 days'
)

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('S&P 500 — Daily Return')
plt.ylabel('Dollar Index (DXY) — Daily Return')
plt.title('Dollar Index vs S&P 500 – Flight-to-Safety Behavior', fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# -------------------------------------------------
# US 10Y Treasury vs S&P500
# -------------------------------------------------
plt.figure(figsize=(10,7))

plt.scatter(
    returns.loc[normal, 'SP500_Return'],
    returns.loc[normal, 'US10Y_Return'],
    c='lightgray', alpha=0.7, s=30, label='Normal days (95%)'
)

plt.scatter(
    returns.loc[extreme, 'SP500_Return'],
    returns.loc[extreme, 'US10Y_Return'],
    c='#D55E00', edgecolors='black', linewidth=0.8, s=80, alpha=0.95,
    label='Worst 5% S&P500 days'
)

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('S&P 500 — Daily Return')
plt.ylabel('US 10Y Treasury — Daily Return (price)')
plt.title('US 10Y Treasury vs S&P 500 – Ultimate Safe Haven', fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ===============================================================
# 5) SHORT SUMMARY — Hedge & Safe Haven
# ===============================================================

print("\n===============================================================")
print("📝 SHORT SUMMARY — Hedge & Safe Haven Interpretation")
print("===============================================================\n")

# ------- 1) Hedge Summary (Global Betas) -------
print("➡️ Hedge (Global OLS):")

for _, r in global_reg_table.iterrows():
    beta = r["Beta_SP500"]
    asset = r["Asset"]

    if beta < 0:
        status = "Hedge"
    elif abs(beta) < 0.02:
        status = "Weak/Neutral"
    else:
        status = "Not a hedge"

    print(f"• {asset}: {status} (β = {beta:.4f})")

# ------- 2) Safe Haven Summary (Quantile Betas) -------
print("\n➡️ Safe Haven (Quantile Regressions):")

for asset in safehaven_table["Asset"].unique():
    sub = safehaven_table[safehaven_table["Asset"] == asset]
    negative_extremes = (sub["β extreme (added)"] < 0).sum()

    if negative_extremes >= 2:
        status = "Safe haven"
    elif negative_extremes == 1:
        status = "Partial safe haven"
    else:
        status = "Not a safe haven"

    print(f"• {asset}: {status} (negative extreme betas in {negative_extremes}/3 quantiles)")

# ------- 3) Crisis Summary (OLS by Crisis) -------
print("\n➡️ Crisis Protection (Crisis Betas):")

for asset in crisis_reg_table["Asset"].unique():
    sub = crisis_reg_table[crisis_reg_table["Asset"] == asset]
    neg = (sub["Beta_SP500"] < 0).sum()
    total = len(sub)

    if neg >= 3:
        status = "Strong protection"
    elif neg >= 1:
        status = "Occasional protection"
    else:
        status = "No protection"

    print(f"• {asset}: {status} ({neg}/{total} crises with β < 0)")

print("\n===============================================================\n")