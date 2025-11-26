
# General data Analysis

# ===============================================================
# GLOBAL IMPORTS & CONFIGURATION
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")

# Pandas display options (Jupyter-like)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)


# ===============================================================
# 1) 📊 DESCRIPTIVE ANALYSIS: MEANS, VARIANCES & CORRELATIONS
# ===============================================================

# Load the merged daily dataset
data = pd.read_csv("../data/merged_daily_data.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.set_index("Date")

# Your 5 final variables
cols = ["Gold", "VIX", "Dollar_Index", "SP500", "US10Y"]

# Descriptive statistics table: Mean + Variance
desc_stats = pd.DataFrame({
    "Mean": data[cols].mean(),
    "Variance": data[cols].var()
})

print("\n📊 DESCRIPTIVE STATISTICS (2000–2025)\n")
print(desc_stats.round(4))

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (2000–2025)")
plt.show()


# ===============================================================
# 2) 📈 VIX ANALYSIS & STRESS PERIOD DETECTION
# ===============================================================

# Verify that the VIX column exists
assert "VIX" in data.columns, "❌ VIX column not found in merged_daily_data.csv"

vix = data["VIX"]

# 1) Detection of high VIX periods
vix_mean = vix.mean()
vix_std = vix.std()

# Stress threshold: VIX > mean + 1 standard deviation
threshold = vix_mean + vix_std
high_vix = vix > threshold

# 2) Definition of selected crisis periods
crises = {
    "Global_Financial_Crisis": ("2007-10-01", "2009-03-31"),
    "Euro_Debt_Crisis": ("2010-05-01", "2012-06-30"),
    "COVID19_Shock": ("2020-02-01", "2020-12-31"),
    "Ukraine_Inflation": ("2022-01-01", "2023-12-31")
}

# 3) GRAPH: VIX + threshold + crisis shading
plt.figure(figsize=(16, 6))

# VIX line
plt.plot(vix.index, vix, label="VIX", color="purple")

# Stress threshold line
plt.axhline(threshold, color="red", linestyle="--",
            label=f"Stress threshold (mean + 1 std = {threshold:.2f})")

# Crisis shaded regions
for crisis, (start, end) in crises.items():
    plt.axvspan(pd.to_datetime(start), pd.to_datetime(end),
                color="orange", alpha=0.15, label=crisis)

# Points where VIX > threshold
plt.scatter(vix.index[high_vix], vix[high_vix],
            color="black", s=15, label="High VIX")

plt.title("Stress Detection via VIX (2000–2025)")
plt.ylabel("VIX Level")
plt.legend(loc="upper left", fontsize=9)
plt.grid(True)
plt.show()


# ===============================================================
# 3) 📉 CRISIS DESCRIPTIVE ANALYSIS (RETURNS)
# ===============================================================

# Load corrected daily returns
returns = pd.read_csv("../data/merged_daily_returns.csv")
returns["Date"] = pd.to_datetime(returns["Date"])
returns = returns.set_index("Date").dropna()

# Columns used
ret_cols = ["Gold_Return", "SP500_Return",
            "Dollar_Index_Return", "US10Y_Return", "VIX_Return"]

rows = []

for crisis, (start, end) in crises.items():
    sub = returns.loc[start:end]

    # Cumulative growth = (1+r1)*(1+r2)*... - 1
    cum = (1 + sub[ret_cols]).prod() - 1

    # Daily volatility
    vol = sub[ret_cols].std()

    rows.append([
        crisis,
        cum["Gold_Return"] * 100,
        cum["SP500_Return"] * 100,
        cum["Dollar_Index_Return"] * 100,
        cum["US10Y_Return"] * 100,
        vol["Gold_Return"] * 100,
        vol["SP500_Return"] * 100,
        vol["Dollar_Index_Return"] * 100,
        vol["US10Y_Return"] * 100,
        ((1 + sub["VIX_Return"]).prod() - 1) * 100
    ])

# Build final summary table
crisis_summary = pd.DataFrame(
    rows,
    columns=[
        "Crisis",
        "Gold_CumRet_%", "SP500_CumRet_%", "DXY_CumRet_%", "US10Y_CumRet_%",
        "Gold_Vol_%", "SP500_Vol_%", "DXY_Vol_%", "US10Y_Vol_%",
        "VIX_CumChange_%"
    ]
).set_index("Crisis")

print("\n📉 FINAL CRISIS TABLE (Daily Returns)\n")
print(crisis_summary.round(3).to_string())


# ===============================================================
# 4) 📈 CUMULATIVE GROWTH CHARTS (Gold + SP500 + DXY)
# ===============================================================

# Cumulative growth function
def cum_growth(r):
    return (1 + r).cumprod()

# Create subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Crisis loop
for ax, (crisis, (start, end)) in zip(axes, crises.items()):
    sub = returns.loc[start:end]

    ax.plot(sub.index, cum_growth(sub["Gold_Return"]), label="Gold")
    ax.plot(sub.index, cum_growth(sub["SP500_Return"]), label="SP500")
    ax.plot(sub.index, cum_growth(sub["Dollar_Index_Return"]), label="Dollar Index (DXY)")

    ax.set_title(f"{crisis}\n({start} → {end})", fontsize=10)
    ax.set_ylabel("Cumulative Growth (base = 1)")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
