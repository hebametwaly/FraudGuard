# 🛡️ FraudGuard — Credit Card Fraud Detection

A machine learning web application that detects fraudulent credit card transactions in real time, built as a graduation project using **XGBoost** and **Streamlit**.

---

## Overview

FraudGuard is an end-to-end fraud detection system that allows users to analyze credit card transactions — either one at a time or in bulk — and instantly receive a fraud risk score. The model is trained on the well-known ULB Credit Card Fraud dataset and handles the severe class imbalance (only 0.172% fraud) using cost-sensitive learning.

---

## Features

- **Single Transaction Analysis** — Enter PCA features (V1–V28), Amount, and Time to get an instant fraud probability score with a risk level label (Very Low → Critical).
- **Batch Analysis** — Upload a CSV file to score thousands of transactions at once. Supports optional `Class` column for automatic evaluation metrics.
- **Interactive Visualizations** — Fraud score distributions, risk level breakdowns, amount comparisons, and score rank charts.
- **Model Info Page** — Explore model architecture, dataset statistics, and feature importance rankings.
- **Export Results** — Download batch prediction results as a CSV file.

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost (Gradient Boosted Trees) |
| Estimators | 200 |
| Learning Rate | 0.05 |
| Max Depth | 6 |
| Imbalance Handling | Cost-Sensitive (`scale_pos_weight`) |
| Train/Test Split | Time-Based (80/20) |
| Evaluation Metric | AUPRC |

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [ULB Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total Transactions | 284,807 |
| Fraud Rate | 0.172% |
| Features | V1–V28 (PCA) + Amount + Time |
| Preprocessing | StandardScaler on Amount and Time |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/fraudguard.git
cd fraudguard

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run App.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
fraudguard/
│
├── App.py                              # Main Streamlit application
├── Credit Card Fraud Detection.ipynb   # Model training & EDA notebook
├── fraud_model_xgb.json                # Trained XGBoost model
├── scaler_amount.pkl                   # Fitted scaler for Amount feature
├── scaler_time.pkl                     # Fitted scaler for Time feature
├── creditcard_sample for Evaluation.csv # Sample dataset for batch testing
└── requirements.txt                    # Python dependencies
```

---

## 📦 Dependencies

```
streamlit>=1.35.0
xgboost>=2.0.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
joblib>=1.3.0
```

---

## 🖥️ App Pages

### 1. Single Transaction
Enter the 30 transaction features manually and click **Predict** to receive:
- A fraud probability score (0–1)
- A color-coded risk level badge
- A visual score bar

### 2. Batch Analysis
Upload a `.csv` file containing columns: `Time`, `V1`–`V28`, `Amount` (and optionally `Class` for evaluation). The app will:
- Score every transaction
- Display aggregate metrics (fraud count, fraud rate, average score)
- Generate distribution plots and a downloadable results table

### 3. Model Info
View the model's hyperparameters, dataset statistics, and a full feature importance chart.

---

## 📈 How to Interpret the Fraud Score

| Score Range | Risk Level | Meaning |
|---|---|---|
| 0.00 – 0.20 | Very Low | Almost certainly legitimate |
| 0.20 – 0.40 | Low | Likely legitimate |
| 0.40 – 0.60 | Moderate | Review recommended |
| 0.60 – 0.80 | High | Likely fraudulent |
| 0.80 – 1.00 | Critical | Almost certainly fraud |

Transactions scoring **≥ 0.5** are flagged as fraud by default. In production, adjust the threshold based on your tolerance for false positives vs. false negatives.

---

## 🙌 Acknowledgements

- [ULB Machine Learning Group](https://mlg.ulb.ac.be/) for the credit card fraud dataset
- [Streamlit](https://streamlit.io/) for the web framework
- [XGBoost](https://xgboost.readthedocs.io/) for the gradient boosting library
