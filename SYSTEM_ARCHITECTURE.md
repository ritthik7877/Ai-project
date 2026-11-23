# System Architecture

Project: [ritthik7877/Ai-project](https://github.com/ritthik7877/Ai-project)  
Hosted static frontend: [ritthik7877.github.io/Ai-project](https://ritthik7877.github.io/Ai-project/)

## Architecture

```mermaid
flowchart TB
    subgraph Training["Offline model training"]
        A[src/train.py] --> B[Generate synthetic transactions]
        B --> C[src/features.py]
        C --> D[Stratified train/test split]
        D --> E[XGBoost training and evaluation]
        E --> F[Select threshold near 95% recall]
        F --> G[(models/fraud_xgb.json)]
        F --> H[(models/feature_columns.pkl)]
        F --> I[(models/threshold.pkl)]
    end

    subgraph Runtime["Local prediction runtime"]
        J[Browser: index.html] -->|POST /predict JSON| K[FastAPI: app.py]
        K --> L[Validate request with Pydantic]
        L --> M[src/features.py]
        G --> N[XGBoost predict_proba]
        H --> N
        M --> N
        N --> O[Apply saved threshold and risk bands]
        I --> O
        O --> P[JSON prediction response]
        P --> J
        Q[GET /health] --> K
    end

    subgraph Publish["Static site publishing"]
        R[Push to main or manual workflow run] --> S[.github/workflows/static.yml]
        S --> T[GitHub Pages deployment]
        T --> J
    end
```

## Components

| Component | Role |
|---|---|
| `src/train.py` | Creates synthetic examples, trains and evaluates XGBoost, chooses a probability threshold, and saves artifacts |
| `src/features.py` | Converts transaction input columns into the engineered feature representation expected by the model |
| `app.py` | Loads model artifacts at startup and serves health and prediction endpoints |
| `index.html` | Static browser form that posts transaction details and renders the prediction |
| `models/fraud_xgb.json` | Serialized XGBoost classifier |
| `models/feature_columns.pkl` | Ordered model feature names used to align prediction inputs |
| `models/threshold.pkl` | Threshold used to convert fraud probability into a fraud/legitimate decision |
| `.github/workflows/static.yml` | Publishes static repository content to GitHub Pages on pushes to `main`, or on manual dispatch |

## Training Workflow

1. `src/train.py` generates 10,000 synthetic transaction examples with fields for amount, balances, transaction type, time since the previous transaction, transaction count, and whether the destination is new.
2. Synthetic fraud labels are generated from configured balance-drain, new-destination, high-velocity, and short-interval patterns, with random label flips to add noise.
3. `engineer_features()` in `src/features.py` derives:
   - Absolute origin and destination balance inconsistencies
   - Drained-balance and amount-to-origin-balance ratios
   - Hour-of-day and night-time flags
   - Transaction velocity and burst indicators
   - Large-amount and log-amount values
   - One-hot encoded transaction type
4. The feature data is divided into stratified training and test sets.
5. XGBoost is trained with positive-class weighting to account for the class distribution.
6. Test probabilities are evaluated. The threshold is chosen from the precision-recall curve near a 95% recall target.
7. The model, feature order, and threshold are written to `models/`.

The training examples are synthetic; their metrics should not be interpreted as expected real-world accuracy.

## Prediction Workflow

```mermaid
sequenceDiagram
    participant Browser as index.html
    participant API as FastAPI app.py
    participant Features as src/features.py
    participant Model as XGBoost + saved artifacts

    Browser->>API: POST /predict with transaction JSON
    API->>API: Validate fields and ranges
    API->>Features: Engineer transaction features
    Features-->>API: Feature row
    API->>API: Add missing training columns and order columns
    API->>Model: predict_proba(feature row)
    Model-->>API: Fraud probability
    API->>API: Apply saved decision threshold and risk band
    API-->>Browser: Fraud flag, probability, risk, explanation
```

The request model requires `amount`, `hoursSinceLastTransaction`, `transactionsLast24h`, `isNewDestination`, `hourOfDayInput`, the four before/after balance values, and `type`. Pydantic also constrains time-since-last-transaction and transaction count to non-negative values, and hour to 0–23.

The response contains:

- `isFraud`: threshold-based boolean classification
- `fraudProbability`: percentage probability rounded to two decimals
- `riskLevel`: risk band derived from probability
- `analysisSummary`: rule-based text based on transaction attributes and model output

## Hosting and Network Boundaries

The GitHub Pages workflow deploys static files only. It does not run the FastAPI backend or host the model inference service. The frontend's API URL is currently `http://127.0.0.1:8000/predict`, which points to the browser user's own machine. As a result:

- Local end-to-end use requires starting FastAPI locally and opening `index.html`.
- The published Pages frontend cannot reach a developer's local API.
- Public end-to-end use requires deploying the API separately, configuring the frontend to use its HTTPS URL, and restricting CORS to the hosted site.

There is no database, transaction persistence, authentication, or deployed model registry in the current architecture.

## Artifact and Feature Contract

Training and inference must use the same `engineer_features()` implementation and preserve the feature names and order stored in `feature_columns.pkl`. At inference, `app.py` fills absent one-hot columns with zero before selecting the saved order.

`app.py` loads `fraud_xgb.json`, `feature_columns.pkl`, and `threshold.pkl` during application startup. If any artifact is absent or incompatible, the API will fail to initialize; retrain the model to regenerate the set.
