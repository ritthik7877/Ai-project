"""
FastAPI backend for the Transaction Fraud Detection System.
Run: uvicorn app:app --reload --port 8000
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import engineer_features

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "models", "fraud_xgb.json")
COLS_PATH = os.path.join(BASE, "models", "feature_columns.pkl")
THRESH_PATH = os.path.join(BASE, "models", "threshold.pkl")

app = FastAPI(title="Transaction Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
feature_columns = joblib.load(COLS_PATH)
threshold = float(joblib.load(THRESH_PATH))


class TransactionRequest(BaseModel):
    amount: float
    hoursSinceLastTransaction: float = Field(..., ge=0)
    transactionsLast24h: int = Field(..., ge=0)
    isNewDestination: bool
    hourOfDayInput: int = Field(..., ge=0, le=23)
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: str  # PAYMENT | TRANSFER | CASH_OUT | CASH_IN | DEBIT


def build_analysis_summary(row: dict, proba: float, is_fraud: bool) -> str:
    """Rule-based natural-language explanation of the model's decision.
    Swap this out for a real LLM call (e.g. GPT-3.5/Claude) using the same
    signals as the prompt context if you want live-generated prose instead."""
    reasons = []

    if row["oldbalanceOrg"] > 0:
        drained_ratio = (row["oldbalanceOrg"] - row["newbalanceOrig"]) / row["oldbalanceOrg"]
        if drained_ratio > 0.85:
            reasons.append(
                f"the transaction empties {drained_ratio*100:.0f}% of the origin account's balance "
                f"(${row['oldbalanceOrg']:,.2f} \u2192 ${row['newbalanceOrig']:,.2f}), consistent with an account takeover"
            )
    if row["isNewDestination"]:
        reasons.append("the funds are moving to a destination account with no prior balance history")
    if row["transactionsLast24h"] >= 10:
        reasons.append(f"{row['transactionsLast24h']} transactions occurred in the last 24 hours, an unusually high burst")
    if row["hoursSinceLastTransaction"] < 0.5:
        reasons.append(f"this transaction followed the previous one after only {row['hoursSinceLastTransaction']:.2f} hours")
    if row["type"] in ("TRANSFER", "CASH_OUT"):
        reasons.append(f"the transaction type ({row['type']}) is the category most associated with fraud in this system")
    if row["amount"] > 10000:
        reasons.append(f"the amount (${row['amount']:,.2f}) is large relative to typical transactions")

    if is_fraud:
        if not reasons:
            reasons.append("the model's learned balance and velocity patterns match historical fraud cases")
        body = (
            f"This transaction has been flagged as high-risk fraud with {proba*100:.2f}% confidence. "
            f"Key signals: " + "; ".join(reasons) + "."
        )
    else:
        body = (
            f"This transaction looks consistent with normal account activity "
            f"(fraud probability {proba*100:.2f}%). "
        )
        if reasons:
            body += "Some minor risk signals were present but not enough to cross the threshold: " + "; ".join(reasons) + "."
        else:
            body += "No significant behavioral anomalies were detected."

    return body


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: TransactionRequest):
    raw = {
        "step": req.hourOfDayInput,  # hour proxy; hourOfDay feature derives step % 24
        "type": req.type,
        "amount": req.amount,
        "oldbalanceOrg": req.oldbalanceOrg,
        "newbalanceOrig": req.newbalanceOrig,
        "oldbalanceDest": req.oldbalanceDest,
        "newbalanceDest": req.newbalanceDest,
        "hoursSinceLastTransaction": req.hoursSinceLastTransaction,
        "transactionsLast24h": req.transactionsLast24h,
        "isNewDestination": int(req.isNewDestination),
    }
    df = pd.DataFrame([raw])
    feat = engineer_features(df)

    # align columns to training-time schema (missing dummy cols -> 0)
    for col in feature_columns:
        if col not in feat.columns:
            feat[col] = 0
    feat = feat[feature_columns]

    proba = float(model.predict_proba(feat)[0, 1])
    is_fraud = proba >= threshold

    if proba >= 0.75:
        risk_level = "HIGH RISK"
    elif proba >= 0.3:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "LOW RISK"

    summary = build_analysis_summary(raw, proba, is_fraud)

    return {
        "isFraud": bool(is_fraud),
        "fraudProbability": round(proba * 100, 2),
        "riskLevel": risk_level,
        "analysisSummary": summary,
    }
