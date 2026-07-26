"""
Train fraud detection model and save artifacts.
Run: python train.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from features import engineer_features
np.random.seed(42)
n_samples = 10000
transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']

data = {
    'step': np.random.randint(0, 744, n_samples),  # 31 days * 24 hours
    'type': np.random.choice(transaction_types, n_samples),
    'amount': np.random.exponential(scale=5000, size=n_samples),
    'oldbalanceOrg': np.random.exponential(scale=50000, size=n_samples),
    'newbalanceOrig': np.random.exponential(scale=50000, size=n_samples),
    'oldbalanceDest': np.random.exponential(scale=50000, size=n_samples),
    'newbalanceDest': np.random.exponential(scale=50000, size=n_samples),
    'hoursSinceLastTransaction': np.random.exponential(scale=24, size=n_samples),
    'transactionsLast24h': np.random.poisson(5, n_samples),
    'isNewDestination': np.random.choice([0, 1], n_samples),
}

df = pd.DataFrame(data)

fraud_prob = np.random.random(n_samples)
is_fraud = (
    ((df['oldbalanceOrg'] > 0) & ((df['oldbalanceOrg'] - df['newbalanceOrig']) / df['oldbalanceOrg'] > 0.9)) |
    (df['isNewDestination'] == 1) & (df['amount'] > 5000) |
    (df['transactionsLast24h'] >= 20) |
    ((df['hoursSinceLastTransaction'] < 0.2) & (df['amount'] > 10000))
)

random_fraud_idx = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
is_fraud[random_fraud_idx] = ~is_fraud[random_fraud_idx]

df['isFraud'] = is_fraud.astype(int)

print(f"Dataset: {len(df)} samples, {df['isFraud'].sum()} frauds ({100*df['isFraud'].mean():.2f}%)")

# Engineer features
X = engineer_features(df.drop('isFraud', axis=1))
y = df['isFraud']

print(f"Features: {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print("\n=== Model Performance ===")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

# Find optimal threshold for ~95% recall
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
idx = np.argmin(np.abs(recall - 0.95))
optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

print(f"\nOptimal Threshold (95% recall): {optimal_threshold:.4f}")

BASE = os.path.dirname(os.path.dirname(__file__))  
os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)

model.save_model(os.path.join(BASE, 'models', 'fraud_xgb.json'))
joblib.dump(X.columns.tolist(), os.path.join(BASE, 'models', 'feature_columns.pkl'))
joblib.dump(optimal_threshold, os.path.join(BASE, 'models', 'threshold.pkl'))

print("\n✅ Model saved to models/")
print("   - fraud_xgb.json")
print("   - feature_columns.pkl")
print("   - threshold.pkl")
