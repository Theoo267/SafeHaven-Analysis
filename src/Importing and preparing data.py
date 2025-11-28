
# — DATA IMPORT, CLEANING & MERGING

# ===============================================================
# GLOBAL IMPORTS & DIRECTORY SETUP
# ===============================================================

import pandas as pd
import os
import yfinance as yf
import datetime
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# Create /data folder if needed
os.makedirs("data", exist_ok=True)

# Date interval for all downloaded datasets
start = datetime.datetime(2000, 9, 1)
end   = datetime.datetime(2025, 9, 1)



# ===============================================================
# 1) DOWNLOAD & SAVE RAW FINANCIAL DATA (Yahoo Finance)
# ===============================================================

# ---------------------------
# GOLD PRICE (GC=F)
# ---------------------------
gold = yf.download("GC=F", start=start, end=end, auto_adjust=True, progress=False)
gold = gold[["Close"]].dropna().reset_index()
gold = gold.rename(columns={"Date": "Date", "Close": "Gold"})
gold = gold.sort_values(by="Date")
gold["Date"] = pd.to_datetime(gold["Date"])
gold.to_csv("data/gold_2000_2025.csv", index=False)

print("✅ GOLD data recorded : data/gold_2000_2025.csv")


# ---------------------------
# S&P 500 INDEX (^GSPC)
# ---------------------------
sp500 = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
sp500 = sp500[["Close"]].dropna().reset_index()
sp500 = sp500.rename(columns={"Date": "Date", "Close": "SP500"})
sp500 = sp500.sort_values(by="Date")
sp500["Date"] = pd.to_datetime(sp500["Date"])
sp500.to_csv("data/sp500_2000_2025.csv", index=False)

print("✅ S&P500 data recorded : data/sp500_2000_2025.csv")


# ---------------------------
# DOLLAR INDEX (DXY)
# ---------------------------
dxy = yf.download("DX-Y.NYB", start=start, end=end, auto_adjust=True, progress=False)
dxy = dxy[["Close"]].dropna().reset_index()
dxy = dxy.rename(columns={"Date": "Date", "Close": "Dollar_Index"})
dxy = dxy.sort_values(by="Date")
dxy["Date"] = pd.to_datetime(dxy["Date"])
dxy.to_csv("data/dollar_index_2000_2025.csv", index=False)

print("✅ Dollar Index data recorded : data/dollar_index_2000_2025.csv")


# ---------------------------
# VIX INDEX (^VIX)
# ---------------------------
vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
vix = vix[["Close"]].dropna().reset_index()
vix = vix.rename(columns={"Date": "Date", "Close": "VIX"})
vix = vix.sort_values(by="Date")
vix["Date"] = pd.to_datetime(vix["Date"])
vix.to_csv("data/vix_2000_2025.csv", index=False)

print("✅ VIX data recorded : data/vix_2000_2025.csv")


# ---------------------------
# US 10-YEAR TREASURY YIELD (^TNX)
# ---------------------------
tnx = yf.download("^TNX", start=start, end=end, auto_adjust=True, progress=False)
tnx = tnx[["Close"]].dropna().reset_index()
tnx = tnx.rename(columns={"Date": "Date", "Close": "US10Y"})
tnx = tnx.sort_values(by="Date")
tnx["Date"] = pd.to_datetime(tnx["Date"])
tnx.to_csv("data/us10y_2000_2025.csv", index=False)

print("✅ US 10Y Yield data recorded : data/us10y_2000_2025.csv")



# ===============================================================
# 2) GRAPHICAL REPRESENTATION OF THE 5 SERIES
# ===============================================================

# List of datasets: (plot title, file path)
datasets = [
    ("Gold Price (USD/troy ounce)",       "data/gold_2000_2025.csv"),
    ("VIX Volatility Index",              "data/vix_2000_2025.csv"),
    ("Dollar Index (DXY)",                "data/dollar_index_2000_2025.csv"),
    ("S&P 500 Index",                     "data/sp500_2000_2025.csv"),
    ("US 10-year interest rate",          "data/us10y_2000_2025.csv")
]

# Generic function for loading + cleaning daily CSV files
def load_and_prepare(path):
    df = pd.read_csv(path)

    # Ensure date column exists and clean it
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        date_col = df.columns[0]
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Detect numeric columns
    value_cols = [c for c in df.columns if c != "Date"]

    # Convert numeric values
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only non-empty numeric columns
    value_cols = [c for c in value_cols if df[c].notna().sum() > 0]

    if not value_cols:
        raise ValueError(f"No numeric columns found in {path}")

    # Choose the most complete column
    y_col = max(value_cols, key=lambda c: df[c].notna().sum())

    # Clean dataset
    df = df[["Date", y_col]].dropna()
    df = df.sort_values("Date").drop_duplicates(subset="Date")

    return df, y_col

# Create plots for the 5 daily time series
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("Daily evolution of economic indicators (2000–2025)",
             fontsize=14, weight="bold")

for (title, path), ax in zip(datasets, axes.flat):
    df, y_col = load_and_prepare(path)
    ax.plot(df["Date"], df[y_col])
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



# ===============================================================
# 3) MERGING ALL DAILY SERIES INTO ONE MASTER FILE
# ===============================================================

# Generic function for loading a daily CSV file
def load_daily(path, rename_dict=None):
    df = pd.read_csv(path)

    # Convert date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Rename columns if needed
    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Sort & remove duplicate dates
    df = df.sort_values("Date").drop_duplicates(subset="Date")

    return df

# Load the 5 datasets
gold  = load_daily("data/gold_2000_2025.csv", {"Gold": "Gold"})
vix   = load_daily("data/vix_2000_2025.csv", {"VIX": "VIX"})
dxy   = load_daily("data/dollar_index_2000_2025.csv", {"Dollar_Index": "Dollar_Index"})
sp500 = load_daily("data/sp500_2000_2025.csv", {"SP500": "SP500"})
us10y = load_daily("data/us10y_2000_2025.csv", {"US10Y": "US10Y"})

# Date coverage checking
for name, df in [
    ("Gold", gold), ("VIX", vix), ("DXY", dxy),
    ("SP500", sp500), ("US10Y", us10y)
]:
    print(f"{name:6s} : {df['Date'].min().date()} -> {df['Date'].max().date()}   ({len(df)} points)")

# Merge all datasets by inner join on Date
merged = (
    gold.merge(vix,   on="Date", how="inner")
        .merge(dxy,   on="Date", how="inner")
        .merge(sp500, on="Date", how="inner")
        .merge(us10y, on="Date", how="inner")
)

print("\nMerged dataset dimensions :", merged.shape)
print(merged.head())

# Save merged dataset
merged.to_csv("data/merged_daily_data.csv", index=False)
print("\n🎉 Merged file saved : data/merged_daily_data.csv")



# ===============================================================
# 4) GENERATION OF DAILY RETURNS FILE
# ===============================================================

# Load merged dataset
data = pd.read_csv("data/merged_daily_data.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.set_index("Date")

# 1) Standard pct_change returns for price series
price_cols = ["Gold", "VIX", "Dollar_Index", "SP500"]

for col in price_cols:
    data[col + "_Return"] = data[col].pct_change()

# 2) Correct transformation for bond yield return
data["US10Y_Change"]  = data["US10Y"].diff()
data["US10Y_Return"] = - data["US10Y_Change"]

# 3) Remove NA rows
returns = data.dropna()

# 4) Save returns file
returns.to_csv("data/merged_daily_returns.csv")
print("✔ New return file created with correct US10Y transformation.")
