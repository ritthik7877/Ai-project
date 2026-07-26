# AI-Driven Financial Fraud Detection System — Model & Notebook

## What's here
```
src/generate_data.py   # generates the synthetic transaction dataset
src/features.py         # behavioral-anomaly feature engineering
src/train.py             # XGBoost training + evaluation script (run directly)
notebooks/fraud_detection.ipynb   # full walkthrough, already executed with outputs
models/fraud_xgb.json             # trained model
models/feature_columns.pkl        # column order needed for inference
models/threshold.pkl              # decision threshold tuned for ~99% recall
data/transactions_sample.csv      # 5k-row sample (regenerate full 1M via generate_data.py)
```

## Quick start
```bash
pip install xgboost scikit-learn pandas numpy matplotlib joblib
cd src
python generate_data.py   # writes data/transactions.csv (1M rows by default)
python train.py           # trains, evaluates, saves model
```
Or open `notebooks/fraud_detection.ipynb` for the full walkthrough with charts.

## About the data
No internet access was available to pull the real Kaggle dataset, so
`generate_data.py` builds a **synthetic dataset structured like PaySim**
(the well-known mobile-money fraud dataset: step/type/amount/balances),
with the fraud pattern you described — *"account drained to $0 via
unknown transfer"* — deliberately injected, plus realistic noise/overlap
so legit and fraud cases aren't trivially separable.

**To use real data instead:** download PaySim ("Synthetic Financial
Datasets For Fraud Detection") or IEEE-CIS Fraud Detection from Kaggle,
save as `data/transactions.csv` with the same column names, and rerun
`train.py` — nothing else changes.

## Results on this run
- AUC: 1.0000, Precision: 1.0000, Recall: 0.9923, F1: 0.9961
- Inference: ~2 ms/transaction (single-row predict)

**Honest caveat:** these numbers are unusually high because
`errorBalanceOrig`/`errorBalanceDest` (balance-consistency checks)
dominate feature importance — this is a known property of PaySim-style
data, not a training bug, but it means the score is optimistic. For a
second, leakage-free benchmark to quote alongside this one, run the
same pipeline against Kaggle's `Credit Card Fraud Detection` dataset
(PCA-anonymized features) — expect AUC around 0.97–0.99 there, which
is the more defensible number if asked about it in an interview or viva.

## Running the full UI (backend + frontend)

```bash
pip install fastapi uvicorn xgboost scikit-learn pandas numpy joblib
python -m uvicorn app:app --reload --port 8000
```
Then just open `frontend/index.html` directly in your browser (double-click it,
or right-click → Open With Browser). It calls `http://127.0.0.1:8000/predict`,
so keep the uvicorn server running in a terminal while you use the page.

Try the exact scenario from your reference screenshot to confirm it matches:
Amount 99999, Hours Since Last Transaction 0.2, Transactions in Last 24h 45,
New Destination Yes, Hour of Day 3, Old Balance Origin 100000, New Balance
Origin 1, Old Balance Destination 0, New Balance Destination 99999,
Transaction Type Transfer → should return **FRAUD, ~100% probability, HIGH RISK**.

The "Analysis Summary" text is currently rule-based (built from the same
signals the model uses — drained-balance ratio, new-destination flag, 24h
transaction burst, time-since-last-transaction, transaction type, amount).
To match the "GPT-3.5 integration" resume bullet, swap `build_analysis_summary()`
in `app.py` for a real LLM call using the same `raw` dict as prompt context —
structurally a drop-in replacement, nothing else in the app changes.

## Next steps to match the full resume bullet set
1. ~~FastAPI serving layer~~ ✅ done (`app.py`)
2. ~~Frontend~~ ✅ done (`frontend/index.html`, matches your reference screenshot)
3. **GPT-based report generation** — replace the rule-based summary with a real LLM call (see above)
4. **Deploy** — containerize with Docker, push to Render/Railway free tier for a live demo link + add a GitHub repo with the video demo
# Ai-project
