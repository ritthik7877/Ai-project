"""
Feature engineering for fraud detection system.
"""
import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Engineer features from raw transaction data.
    
    Args:
        df: DataFrame with columns:
            step, type, amount, oldbalanceOrg, newbalanceOrig,
            oldbalanceDest, newbalanceDest, hoursSinceLastTransaction,
            transactionsLast24h, isNewDestination
    
    Returns:
        DataFrame with engineered features
    """
    feat = df.copy()
    
    # Balance-based features
    feat['errorBalanceOrig'] = (
        (feat['oldbalanceOrg'] + feat['amount']) - feat['newbalanceOrig']
    ).abs()
    feat['errorBalanceDest'] = (
        (feat['oldbalanceDest'] - feat['amount']) - feat['newbalanceDest']
    ).abs()
    
    # Drained account ratio
    feat['drainedRatio'] = (feat['oldbalanceOrg'] - feat['newbalanceOrig']) / feat['oldbalanceOrg'].clip(lower=1)
    
    # Amount ratio to origin balance
    feat['amountRatioOrg'] = feat['amount'] / feat['oldbalanceOrg'].clip(lower=1)
    
    # Transaction timing features
    feat['hourOfDay'] = feat['step'] % 24
    feat['isNightTime'] = ((feat['hourOfDay'] >= 22) | (feat['hourOfDay'] <= 5)).astype(int)
    
    # Velocity features
    feat['transactionsPerHour'] = feat['transactionsLast24h'] / feat['hoursSinceLastTransaction'].clip(lower=0.1)
    feat['isBurst'] = (feat['transactionsLast24h'] >= 10).astype(int)
    
    # Amount features
    feat['isLargeAmount'] = (feat['amount'] > 10000).astype(int)
    feat['log_amount'] = np.log1p(feat['amount'])
    
    # Type features (one-hot encoding)
    type_dummies = pd.get_dummies(feat['type'], prefix='type_', drop_first=False)
    feat = pd.concat([feat, type_dummies], axis=1)
    
    # Drop raw columns
    feat = feat.drop(['step', 'type', 'oldbalanceOrg', 'oldbalanceDest'], axis=1)
    
    return feat
