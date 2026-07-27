# Credit Card Fraud Detection using Anomaly Detection

Production-ready Data Scientist project for detecting fraudulent credit card transactions using both unsupervised anomaly detection and supervised machine learning techniques.

---

# Project Objectives

- Learn anomaly detection from scratch.
- Build a production-level Data Scientist project.
- Compare multiple anomaly detection algorithms.
- Deploy the best model using FastAPI.
- Monitor model performance.
- Retrain the model when required.

---

# Dataset

This project uses the **Credit Card Fraud Detection** dataset.

Dataset Link:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download the dataset manually and extract:

```
creditcard.csv
```

Place it here:

```
data/
└── raw/
    └── creditcard.csv
```

**Note:** The dataset is not included in this repository because of its size and licensing. Please download it directly from Kaggle. The project expects the CSV file to be located in `data/raw/`. :contentReference[oaicite:1]{index=1}

---

# Project Structure

```
Credit-Card-Fraud-Detection-using-Anomaly-Detection/
│
├── .github/
├── artifacts/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docker/
├── logs/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── src/
├── tests/
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

# Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastAPI
- SHAP
- MLflow
- PostgreSQL
- Docker
- Grafana
- Prometheus

---

# Learning Roadmap

- Business Understanding
- Data Ingestion
- Data Validation
- Exploratory Data Analysis
- Feature Engineering
- PCA
- Isolation Forest
- Local Outlier Factor
- One-Class SVM
- Autoencoder
- Random Forest
- XGBoost
- Explainability (SHAP)
- FastAPI Deployment
- Monitoring
- Retraining
- Docker
- GitHub Actions
- Chainguard

---

# Author

Nandini Arjunan

## 📊 Model Performance Comparison

The project first trains baseline anomaly detection models using default parameters. Hyperparameter optimization is then performed using **Optuna** to improve model performance.

### Baseline vs Tuned Isolation Forest

| Metric | Baseline Model | Tuned Model (Optuna) |
|---------|---------------:|---------------------:|
| Accuracy | 96.13% | **99.68%** |
| Precision | 3.33% | **26.06%** |
| Recall | **77.92%** | 48.05% |
| F1 Score | 6.39% | **33.79%** |
| ROC AUC | **87.04%** | 73.91% |

### Interpretation

The baseline Isolation Forest achieved a high recall, detecting approximately **78%** of fraudulent transactions. However, its precision was only **3.33%**, meaning it generated a large number of false positive alerts.

After hyperparameter optimization with **Optuna**, the model became more selective:

- Accuracy improved from **96.13%** to **99.68%**.
- Precision increased from **3.33%** to **26.06%**, significantly reducing false positive predictions.
- F1 Score improved from **6.39%** to **33.79%**, providing a much better balance between precision and recall.
- Recall decreased from **77.92%** to **48.05%**, indicating that the tuned model detects fewer fraud cases but produces fewer false alarms.

This reflects the typical **precision–recall trade-off** encountered in anomaly detection systems.

### Final Model Selection

The project uses **F1 Score** as the primary model selection criterion because it balances both precision and recall on highly imbalanced fraud detection datasets.

Based on this criterion, the **tuned Isolation Forest** was selected as the final production model.

### Best Model Metrics

| Metric | Value |
|---------|------:|
| Model | Isolation Forest |
| Accuracy | **99.68%** |
| Precision | **26.06%** |
| Recall | **48.05%** |
| F1 Score | **33.79%** |
| ROC AUC | **73.91%** |

---

**Why F1 Score instead of Accuracy?**

Credit card fraud datasets are highly imbalanced, with fraudulent transactions representing only a small fraction of all transactions. A model can achieve very high accuracy simply by predicting most transactions as normal.

For this reason, **F1 Score** is used as the primary optimization and model selection metric because it provides a balanced evaluation of both fraud detection capability (recall) and prediction reliability (precision).