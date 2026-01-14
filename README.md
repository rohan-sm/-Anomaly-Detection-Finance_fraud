# Financial Fraud / Anomaly Detection API

This project is a **fraud detection system** that identifies **anomalous financial transactions** using **unsupervised machine learning**, specifically **Isolation Forest**, and exposes predictions through a **FastAPI REST API**.

The focus of this project is on **real-world fraud behavior modeling**, clean backend design, and explainable predictions — not just model accuracy.

---

## 🔍 Problem Statement

In real financial systems:
- Fraud is **rare and highly imbalanced**
- Labels are often **missing or delayed**
- Fraud patterns change over time

Because of this, treating fraud detection as a **supervised classification problem** is often unreliable.

➡️ This project approaches fraud as an **anomaly detection problem**, where transactions that deviate strongly from normal user behavior are flagged as potential fraud.

---

## 🏗️ System Architecture



```
FRAUD-ANAMILY-DETECTION
├── app/                        # API & Inference logic
├── data/
│   ├── metadata/
│   ├── processed/
│   └── raw/
├── models/                     # Saved models (.pkl, .keras) & artifacts
├── notebooks/                  # Experiments & Analysis
├── reports/                    # Generated Metrics
├── src/                        # Core Logic & Pipelines
│   ├── data_generation/
│   ├── feature_engineering/
│   ├── models/                 # pipelines
│   └── utils/
├── tests/                      # Unit Tests
├── .gitignore
├── README.md
├── requirements.txt
└── run_pipeline.py
```
---

## 🤖 Models Used

### 1️⃣ Isolation Forest

### 2️⃣ One-Class SVM

### 3️⃣ Autoencoder (Neural Network)


---

## 📊 Feature Engineering Philosophy

Fraud is rarely about raw values.
It’s about **contextual deviation**.

### Example Features
| Feature | Why It Matters |
|------|---------------|
| `amount_dev_log` | Detects abnormal spend magnitude |
| `avg_amount_24h` | Personal baseline modeling |
| `txn_count_1h` | Velocity attacks |
| `hour_sin / hour_cos` | Time-of-day anomalies |
| `location_change` | Geo inconsistency |
| `device_change` | Account takeover signal |

⚠️ **Same preprocessing pipeline is used during training and inference**  
(No training–serving skew)
---
--------------------
## 🏗️ High-Level Architecture
```
Transaction Input
      │
      ▼
FastAPI (/predict)
      │
      ▼
Feature Engineering
      │
      ▼
Isolation Forest Model
      │
      ▼
Anomaly Score
      │
      ▼
Fraud Probability + Reasoning
      │
      ▼
   Logging
```
## 🧠 Why Isolation Forest?

Multiple anomaly detection approaches were experimented with during development.

**Isolation Forest was chosen for the final system because:**
- It performed **most consistently** on validation data
- It scales well to large transaction volumes
- It works naturally with **tabular financial features**
- It provides a clear anomaly score that can be converted into a fraud probability

Rather than overengineering, the final system uses **one strong, explainable model**.

---


---

## 🔮 API Endpoint

### POST /predict

#### Request
```json
{
  "amount": 4500,
  "avg_amount_24h": 1200,
  "txn_count_1h": 6,
  "hour": 2,
  "location_change": 1,
  "device_change": 1
}
```
## Responce
```json
{
  "fraud_probability": 0.83,
  "is_fraud": true,
  "reasoning": [
    "Transaction amount deviates from user's normal spending",
    "Unusual transaction time detected",
    "High transaction velocity observed",
    "Isolation Forest anomaly score is high"
  ]
}
```

---
## 🧮 Fraud Probability Logic

Isolation Forest outputs an anomaly score, not a probability.
To make the output interpretable:
* Raw anomaly scores are normalized
* Scores are mapped to a range between 0 and 1
* A configurable threshold is used to classify fraud vs non-fraud
This keeps the decision logic:
* Simple
* Transparent
* Easy to explain in interviews

---
## 🪵 Logging

Every prediction is logged for traceability and debugging.
Each log entry includes:
* Timestamp
* Key input attributes
* Anomaly score
* Fraud probability
* Final fraud decision

---
## Validation & Error Handling

* Request validation using Pydantic
* Clear error messages for invalid inputs
* Safe handling of model loading and inference errors

---
## ▶️ Running the Application
```
pip install -r requirements.txt
uvicorn app.main:app --reload
```
---
## API documentation is available at:

```http://127.0.0.1:8000/docs```
