# Exploratory Data Analysis Report

## Project

Credit Card Fraud Detection using Anomaly Detection

---

# Executive Summary

This report summarizes the exploratory analysis performed on the credit card transaction dataset before model development.

---


# Dataset Overview

| Property | Value |
|----------|------:|
| Total Rows | 227845 |
| Total Columns | 31 |
| Feature Columns | 30 |
| Target Column | Class |
| Memory Usage | 53.89 MB |
| Fraud Transactions | 394 |
| Normal Transactions | 227451 |

---

# Data Quality Assessment

| Check | Result |
|------|------:|
| Missing Values | 0 |
| Missing Percentage | 0.00% |
| Duplicate Rows | 718 |
| Duplicate Percentage | 0.32% |

### Observation

- No missing values detected.
- 718 duplicate rows detected.

### Recommendation

Duplicate rows should be removed during preprocessing.

---

# Class Distribution

| Class | Count | Percentage |
|------|------:|-----------:|
| Normal | 227451 | 99.83% |
| Fraud | 394 | 0.17% |

### Dataset Imbalance
| Metric | Value |
|--------|------:|
| Imbalance Ratio | 577.29 : 1 |
### Observation

The dataset is extremely imbalanced.

### Business Impact

- Accuracy is not an appropriate evaluation metric.
- Precision, Recall, F1-score, ROC-AUC and PR-AUC should be used for evaluation.

---

# Transaction Amount Analysis

| Statistic | Value |
|-----------|------:|
| Minimum Amount | $0.00 |
| Maximum Amount | $25691.16 |
| Mean Amount | $88.18 |
| Median Amount | $22.00 |
| Standard Deviation | $250.72 |
| 25th Percentile (Q1) | $5.64 |
| 75th Percentile (Q3) | $77.49 |

### Observation

- Transaction amounts are highly right-skewed.
- Large transaction amounts are relatively rare.
- Amount contains several outliers.

### Recommendation

- Apply feature scaling.
- Preserve outliers because they may represent fraudulent transactions.

---

# Transaction Time Analysis


| Statistic | Value |
|-----------|------:|
| Minimum Time | 0 |
| Maximum Time | 172792 |
| Mean Time | 94885.09 |
| Median Time | 84805.00 |
| Standard Deviation | 47488.42 |
| 25th Percentile (Q1) | 54228.00 |
| 75th Percentile (Q3) | 139364.00 |


### Observation
 - Transactions span the complete observation period.
 - The Time feature alone does not clearly distinguish fraudulent and legitimate transactions.
### Recommendation

- Retain the Time feature during model training.
- Consider deriving additional temporal features if real timestamps are available.


---
---

# Correlation Analysis

### Top Positive Correlations

| Feature | Correlation |
|---------|------------:|
| V11 | 0.1537 |
| V4 | 0.1350 |
| V2 | 0.0906 |
| V21 | 0.0356 |
| V19 | 0.0324 |


### Top Negative Correlations

| Feature | Correlation |
|---------|------------:|
| V17 | -0.3219 |
| V14 | -0.3011 |
| V12 | -0.2600 |
| V10 | -0.2179 |
| V3 | -0.1941 |


### Observation

- The strongest positively correlated features are listed above.
- The strongest negatively correlated features are listed above.
- Fraud detection relies on combinations of multiple variables rather than a single feature.

---


# Generated Figures

- class_distribution.png
- amount_distribution.png
- log_amount_distribution.png
- amount_boxplot.png
- amount_by_class.png
- time_distribution.png
- time_by_class.png
- correlation_heatmap.png
- feature_target_correlation.png

---

# Key Findings

- No missing values detected.
- 718 duplicate rows detected.
- Dataset is highly imbalanced.
- Transaction Amount is heavily right-skewed.
- Amount contains significant outliers.
- PCA features exhibit low correlation.
- Fraud detection requires learning complex patterns across multiple variables.

---

# Next Steps

The next phase is **Data Preprocessing**, which includes:

- Remove duplicate rows
- Split features and target
- Train/Test split
- Feature scaling
- Save preprocessing artifacts
