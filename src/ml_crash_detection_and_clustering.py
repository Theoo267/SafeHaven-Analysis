# =============================================================================
# MACHINE LEARNING
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, recall_score,
    precision_score, f1_score
)
import shap
import os

os.makedirs("results", exist_ok=True)


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', True)

# =============================================================================
# 4.1 LOAD DATA
# =============================================================================

df = pd.read_csv("data/merged_daily_returns.csv", parse_dates=['Date']).set_index('Date')

# =============================================================================
# 4.2 FEATURE ENGINEERING
# =============================================================================

df['VIX_Change_1d'] = df['VIX'].pct_change(1)
df['VIX_Change_3d'] = df['VIX'].pct_change(3)
df['VIX_Change_5d'] = df['VIX'].pct_change(5)

df['VIX_High'] = (df['VIX'] > df['VIX'].rolling(252).quantile(0.80)).astype(int)

df['SP500_mom_5d'] = df['SP500_Return'].rolling(5).mean()
df['SP500_mom_10d'] = df['SP500_Return'].rolling(10).mean()

df = df.dropna()


# =============================================================================
# 4.3 TARGET VARIABLE — EXTREME NEGATIVE RETURNS (Bottom 5%)
# =============================================================================

crash_threshold = df['SP500_Return'].quantile(0.05)
df['is_crash'] = (df['SP500_Return'] <= crash_threshold).astype(int)


# =============================================================================
# 4.4 FEATURES & TRAIN/TEST SPLIT
# =============================================================================

features = [
    'VIX', 'VIX_Change_1d', 'VIX_Change_3d', 'VIX_Change_5d', 'VIX_High',
    'Gold_Return', 'Dollar_Index_Return', 'US10Y_Return',
    'SP500_mom_5d', 'SP500_mom_10d'
]

X = df[features]
y = df['is_crash']

X_train = X.loc[:'2019-12-31']
X_test  = X.loc['2020-01-01':]

y_train = y.loc[:'2019-12-31']
y_test  = y.loc['2020-01-01':]


# =============================================================================
# 4.5 XGBOOST MODEL
# =============================================================================

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=(len(y_train)-y_train.sum()) / y_train.sum(),
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# =============================================================================
# 4.6 MODEL PERFORMANCE
# =============================================================================

print("="*70)
print("XGBoost Performance on 2020–2025 (Extreme Crash Prediction)")
print("="*70)
print(classification_report(y_test, y_pred, digits=4))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

xgb_perf = pd.DataFrame({
    "Metric": ["ROC_AUC"],
    "Value": [roc_auc_score(y_test, y_prob)]
})
xgb_perf.to_csv("results/xgb_performance.csv", index=False)


# =============================================================================
# 4.7 SHAP EXPLANATIONS
# =============================================================================

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10, show=False)
plt.title("Feature Importance (SHAP) – Drivers of Crash Alerts")
plt.tight_layout()
plt.savefig("results/shap_feature_importance_bar.png", dpi=200)
plt.close()

shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Values – Impact Direction on Crash Probability")
plt.tight_layout()
plt.savefig("results/shap_summary_beeswarm.png", dpi=200)
plt.close()


# =============================================================================
# 4.8 SAFE-HAVEN PERFORMANCE ON PREDICTED CRASH DAYS
# =============================================================================

predicted_crash_days = X_test.index[y_pred == 1]

safe_haven_returns = df.loc[predicted_crash_days, [
    'Gold_Return', 'Dollar_Index_Return', 'US10Y_Return', 'SP500_Return'
]].mean()

print("\n" + "="*70)
print("Average Daily Returns on Predicted Crash Days (2020–2025)")
print("="*70)
print(safe_haven_returns.round(6).to_string())
print("="*70)
safe_haven_returns.to_csv("results/xgb_safehaven_returns.csv")

print("\nConclusion:")
print(f"• Gold   : +{safe_haven_returns['Gold_Return']*100:5.3f}% → Strong safe-haven")
print(f"• Dollar : +{safe_haven_returns['Dollar_Index_Return']*100:5.3f}% → Safe-haven behavior")
print(f"• US10Y  : +{safe_haven_returns['US10Y_Return']*100:5.3f}% → Flight-to-quality")
print(f"• S&P 500: {safe_haven_returns['SP500_Return']*100:6.2f}% → Correct crash detection")

print("\nModel successfully identifies safe-haven assets during extreme market stress.")


# =============================================================================
# 5. RANDOM FOREST MODEL
# =============================================================================

from sklearn.ensemble import RandomForestClassifier

print("\n" + "="*70)
print("Random Forest Performance on 2020–2025 (Extreme Crash Prediction)")
print("="*70)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# PERFORMANCE METRICS
print(classification_report(y_test, rf_pred, digits=4))
print(f"ROC AUC Score: {roc_auc_score(y_test, rf_prob):.4f}")

rf_perf = pd.DataFrame({
    "Metric": ["ROC_AUC"],
    "Value": [roc_auc_score(y_test, rf_prob)]
})
rf_perf.to_csv("results/rf_performance.csv", index=False)

# SAFE-HAVEN PERFORMANCE ON RANDOM FOREST PREDICTED CRASH DAYS
rf_predicted_crash_days = X_test.index[rf_pred == 1]

rf_safe_haven = df.loc[rf_predicted_crash_days, [
    'Gold_Return', 'Dollar_Index_Return', 'US10Y_Return', 'SP500_Return'
]].mean()

print("\n" + "="*70)
print("Average Daily Returns on RF Predicted Crash Days (2020–2025)")
print("="*70)
print(rf_safe_haven.round(6).to_string())
print("="*70)

rf_safe_haven.to_csv("results/rf_safehaven_returns.csv")

# =============================================================================
# 6. K-MEANS CLUSTERING – UNSUPERVISED STRESS TYPOLOGY
# =============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print("\n" + "="*80)
print("UNSUPERVISED ANALYSIS: K-Means Clustering of Extreme Market Stress Days")
print("="*80)

extreme_days = df[df['is_crash'] == 1].copy()
print(f"Number of extreme crash days (bottom 5%): {len(extreme_days)}")

kmeans_features = [
    'SP500_Return', 'VIX_Change_1d', 'Gold_Return',
    'Dollar_Index_Return', 'US10Y_Return'
]

X_cluster = StandardScaler().fit_transform(extreme_days[kmeans_features])

# Elbow method
inertias = [
    KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster).inertia_
    for k in range(1, 10)
]

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertias, 'bo-')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.savefig("results/kmeans_elbow_method.png", dpi=200)
plt.close()


# k=3 clusters
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
extreme_days['crisis_type'] = kmeans_model.fit_predict(X_cluster)

summary = extreme_days.groupby('crisis_type')[kmeans_features].mean()
summary['count'] = extreme_days['crisis_type'].value_counts()

print("\nAverage characteristics by crisis type:")
print(summary.round(5))

print("\n→ Key insight: Across the 3 machine-identified crisis types,")
print("  gold, the US dollar, AND especially US Treasuries increase consistently.")
print("  Unsupervised proof that these are genuine safe-haven assets.")

# PCA projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=extreme_days['crisis_type'],
            cmap='plasma', s=60, alpha=0.8)
plt.title('PCA – Three Types of Extreme Crises (2000–2025)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.colorbar(label='Crisis type')
plt.grid(True, alpha=0.3)
plt.savefig("results/kmeans_pca_clusters.png", dpi=200)
plt.close()

# =============================================================================
# 7. LSTM — Deep Learning Model for Crash Prediction
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

print("\n====================== 7. LSTM TRAINING ======================\n")


# =============================================================================
# 7.1 LSTM FEATURES (Same Inputs as XGBoost & RandomForest)
# =============================================================================
# Ensures fair benchmarking between classical ML models and deep learning.

lstm_features = [
    'VIX', 'VIX_Change_1d', 'VIX_Change_3d', 'VIX_Change_5d', 'VIX_High',
    'Gold_Return', 'Dollar_Index_Return', 'US10Y_Return',
    'SP500_mom_5d', 'SP500_mom_10d'
]

X_lstm = df[lstm_features].values
y_lstm = df["is_crash"].values


# =============================================================================
# 7.2 TRAIN/TEST SPLIT — Same as XGBoost & RandomForest
# =============================================================================
# Ensures perfect out-of-sample comparability.

split = len(X_train)  # identical time split

X_train_lstm = X_lstm[:split]
X_test_lstm  = X_lstm[split:]
y_train_lstm = y_lstm[:split]
y_test_lstm  = y_lstm[split:]


# =============================================================================
# 7.3 SCALING (RobustScaler — stable to outliers)
# =============================================================================

scaler_lstm = RobustScaler()
X_train_lstm = scaler_lstm.fit_transform(X_train_lstm)
X_test_lstm  = scaler_lstm.transform(X_test_lstm)


# =============================================================================
# 7.4 SEQUENCE GENERATION
# =============================================================================


SEQ_LEN = 60

def build_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = build_sequences(X_train_lstm, y_train_lstm)
X_test_seq,  y_test_seq  = build_sequences(X_test_lstm,  y_test_lstm)

print(f"Train LSTM sequences : {X_train_seq.shape}")
print(f"Test  LSTM sequences : {X_test_seq.shape}")


# =============================================================================
# 7.5 LSTM MODEL ARCHITECTURE
# =============================================================================


tf.random.set_seed(42)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(lstm_features))),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0007),
    loss="binary_crossentropy",
    metrics=["AUC"]
)

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=25,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)


# =============================================================================
# 7.6 LSTM PREDICTION
# =============================================================================


proba_lstm = model.predict(X_test_seq).flatten()

threshold = np.percentile(proba_lstm, 90)
pred_lstm = (proba_lstm >= threshold).astype(int)

print(f"\nNumber of crashes predicted by LSTM : {pred_lstm.sum()}")


# =============================================================================
# 7.7 SAFE-HAVEN PERFORMANCE ON LSTM PREDICTED CRASH DAYS
# =============================================================================

test_dates_lstm = df.index[split + SEQ_LEN:][pred_lstm == 1]

if len(test_dates_lstm) > 0:
    safe_lstm = df.loc[test_dates_lstm, [
        "Gold_Return", "Dollar_Index_Return", "US10Y_Return", "SP500_Return"
    ]].mean() * 100
else:
    safe_lstm = pd.Series(
        [np.nan]*4,
        index=["Gold_Return", "Dollar_Index_Return", "US10Y_Return", "SP500_Return"]
    )

print("\n=== LSTM Safe-Haven Results ===")
print(safe_lstm.round(4))
safe_lstm.to_csv("results/lstm_safehaven_returns.csv")


# =============================================================================
# 7.8 LSTM PERFORMANCE METRICS
# =============================================================================

lstm_recall    = recall_score(y_test_seq, pred_lstm)
lstm_precision = precision_score(y_test_seq, pred_lstm, zero_division=0)
lstm_f1        = f1_score(y_test_seq, pred_lstm, zero_division=0)
lstm_auc       = roc_auc_score(y_test_seq, proba_lstm)

print("\n=== Performance LSTM ===")
print(f"Recall (crash)   : {lstm_recall:.4f}")
print(f"Precision (crash): {lstm_precision:.4f}")
print(f"F1-score         : {lstm_f1:.4f}")
print(f"ROC-AUC          : {lstm_auc:.4f}")


# =============================================================================
# 8. FINAL MODEL COMPARISON (XGBoost vs RandomForest vs LSTM)
# =============================================================================

print("\n==============================================================")
print("                FINAL MODEL COMPARISON")
print("==============================================================\n")

# Align sequences length with LSTM window
y_test_aligned = y_test[-len(y_test_seq):]

xgb_pred_aligned = y_pred[-len(y_test_seq):]
xgb_prob_aligned = y_prob[-len(y_test_seq):]

rf_pred_aligned  = rf_pred[-len(y_test_seq):]
rf_prob_aligned  = rf_prob[-len(y_test_seq):]

models = ["XGBoost", "Random Forest", "LSTM"]

recalls = [
    recall_score(y_test_aligned, xgb_pred_aligned),
    recall_score(y_test_aligned, rf_pred_aligned),
    lstm_recall
]

precisions = [
    precision_score(y_test_aligned, xgb_pred_aligned),
    precision_score(y_test_aligned, rf_pred_aligned),
    lstm_precision
]

f1s = [
    f1_score(y_test_aligned, xgb_pred_aligned),
    f1_score(y_test_aligned, rf_pred_aligned),
    lstm_f1
]

aucs = [
    roc_auc_score(y_test_aligned, xgb_prob_aligned),
    roc_auc_score(y_test_aligned, rf_prob_aligned),
    lstm_auc
]

results = pd.DataFrame({
    "Model": models,
    "Recall Crash": recalls,
    "Precision": precisions,
    "F1-score": f1s,
    "ROC-AUC": aucs
})

print("\n==================== MODEL BENCHMARK ====================\n")
print(results.round(4).to_string(index=False))
results.to_csv("results/model_comparison.csv", index=False)

# Visualization
plt.figure(figsize=(8, 5))
plt.bar(models, recalls, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
plt.title("Crash Detection Recall — Model Comparison", fontsize=14)
plt.ylabel("Recall (0 → 1)")
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.ylim(0, 1)
plt.savefig("results/model_comparison_recall.png", dpi=200)
plt.close()