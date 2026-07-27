import os

from src.logger.logger import logger


class EDAReportGenerator:
    """
    Generates a Markdown EDA report.
    """

    REPORT_DIR = os.path.join("reports")
    REPORT_PATH = os.path.join("reports", "eda_report.md")

    @staticmethod
    def generate_report(summary):
        """
        Generate EDA report in Markdown format.
        """

        os.makedirs(
            EDAReportGenerator.REPORT_DIR,
            exist_ok=True,
        )

        report = ""

        report += EDAReportGenerator.build_header()

        report += EDAReportGenerator.build_dataset_analysis(summary)

        report += EDAReportGenerator.build_feature_analysis(summary)

        report += EDAReportGenerator.build_conclusion(summary)

        EDAReportGenerator.save_report(report)

    @staticmethod
    def build_header():
        """
        Build report header.
        """

        return """# Exploratory Data Analysis Report

## Project

Credit Card Fraud Detection using Anomaly Detection

---

# Executive Summary

This report summarizes the exploratory analysis performed on the credit card transaction dataset before model development.

---

"""
    @staticmethod
    def build_dataset_analysis(summary):
        """
        Build dataset overview, data quality,
        and class distribution sections.
        """
        total_rows = summary["total_rows"]
        
        total_columns = summary["total_columns"]
        memory_usage = summary["memory_usage_mb"]
        feature_count = len(summary["feature_names"]) - 1
        target_column = "Class"

        total_missing_values = summary["total_missing_values"]
        duplicate_rows = summary["duplicate_rows"]
        data_quality = summary["data_quality_statistics"]
        missing_percentage = data_quality["missing_percentage"]
        duplicate_percentage = data_quality["duplicate_percentage"]

        class_distribution = summary["class_distribution"]
        class_percentage = summary["class_percentage"]
        class_statistics = summary["class_statistics"]
        imbalance_ratio = class_statistics["imbalance_ratio"]

        normal = class_distribution.get(0, 0)
        fraud = class_distribution.get(1, 0)

        normal_percentage = class_percentage.get(0, 0)
        fraud_percentage = class_percentage.get(1, 0)

        # Dynamic observations

        if total_missing_values == 0:
            missing_message = "No missing values detected."
        else:
            missing_message = (
                f"{total_missing_values} missing values detected."
            )

        if duplicate_rows == 0:
            duplicate_message = "No duplicate rows detected."
        else:
            duplicate_message = (
                f"{duplicate_rows} duplicate rows detected."
            )

        if fraud_percentage < 1:
            imbalance_message = (
                "The dataset is extremely imbalanced."
            )
        elif fraud_percentage < 5:
            imbalance_message = (
                "The dataset is moderately imbalanced."
            )
        else:
            imbalance_message = (
                "The dataset is relatively balanced."
            )

        return f"""
# Dataset Overview

| Property | Value |
|----------|------:|
| Total Rows | {total_rows} |
| Total Columns | {total_columns} |
| Feature Columns | {feature_count} |
| Target Column | {target_column} |
| Memory Usage | {memory_usage:.2f} MB |
| Fraud Transactions | {fraud} |
| Normal Transactions | {normal} |

---

# Data Quality Assessment

| Check | Result |
|------|------:|
| Missing Values | {total_missing_values} |
| Missing Percentage | {missing_percentage:.2f}% |
| Duplicate Rows | {duplicate_rows} |
| Duplicate Percentage | {duplicate_percentage:.2f}% |

### Observation

- {missing_message}
- {duplicate_message}

### Recommendation

Duplicate rows should be removed during preprocessing.

---

# Class Distribution

| Class | Count | Percentage |
|------|------:|-----------:|
| Normal | {normal} | {normal_percentage:.2f}% |
| Fraud | {fraud} | {fraud_percentage:.2f}% |

### Dataset Imbalance
| Metric | Value |
|--------|------:|
| Imbalance Ratio | {imbalance_ratio:.2f} : 1 |
### Observation

{imbalance_message}

### Business Impact

- Accuracy is not an appropriate evaluation metric.
- Precision, Recall, F1-score, ROC-AUC and PR-AUC should be used for evaluation.

---
"""
    @staticmethod
    def build_feature_analysis(summary):
        """
        Build feature analysis section.
        """

        time = summary["time_statistics"]
        amount = summary["amount_statistics"]
        correlation = summary["correlation_statistics"]
        positive = correlation["top_positive"]
        negative = correlation["top_negative"]
        positive_rows = ""
        for feature, value in positive.items():
            positive_rows += f"| {feature} | {value:.4f} |\n"
        negative_rows = ""
        
        for feature, value in negative.items():    
            negative_rows += f"| {feature} | {value:.4f} |\n"

        return f"""
# Transaction Amount Analysis

| Statistic | Value |
|-----------|------:|
| Minimum Amount | ${amount["minimum"]:.2f} |
| Maximum Amount | ${amount["maximum"]:.2f} |
| Mean Amount | ${amount["mean"]:.2f} |
| Median Amount | ${amount["median"]:.2f} |
| Standard Deviation | ${amount["std"]:.2f} |
| 25th Percentile (Q1) | ${amount["q1"]:.2f} |
| 75th Percentile (Q3) | ${amount["q3"]:.2f} |

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
| Minimum Time | {time["minimum"]} |
| Maximum Time | {time["maximum"]} |
| Mean Time | {time["mean"]:.2f} |
| Median Time | {time["median"]:.2f} |
| Standard Deviation | {time["std"]:.2f} |
| 25th Percentile (Q1) | {time["q1"]:.2f} |
| 75th Percentile (Q3) | {time["q3"]:.2f} |


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
{positive_rows}

### Top Negative Correlations

| Feature | Correlation |
|---------|------------:|
{negative_rows}

### Observation

- The strongest positively correlated features are listed above.
- The strongest negatively correlated features are listed above.
- Fraud detection relies on combinations of multiple variables rather than a single feature.

---

"""




    
    
    @staticmethod
    def build_conclusion(summary):
        """
        Build conclusion section.
        """

        total_missing_values = summary["total_missing_values"]
        duplicate_rows = summary["duplicate_rows"]


        
        
        class_percentage = summary["class_percentage"]
        

        fraud_percentage = class_percentage.get(1, 0)

        if total_missing_values == 0:
            missing_message = "No missing values detected."
        else:
            missing_message = (
                f"{total_missing_values} missing values detected."
            )

        if duplicate_rows == 0:
            duplicate_message = "No duplicate rows detected."
        else:
            duplicate_message = (
                f"{duplicate_rows} duplicate rows detected."
            )

        if fraud_percentage < 1:
            imbalance_message = "Dataset is highly imbalanced."
        elif fraud_percentage < 5:
            imbalance_message = "Dataset is moderately imbalanced."
        else:
            imbalance_message = "Dataset is relatively balanced."

        return f"""
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

- {missing_message}
- {duplicate_message}
- {imbalance_message}
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
"""

    @staticmethod
    def save_report(report):
        """
        Save report to disk.
        """

        with open(
            EDAReportGenerator.REPORT_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(report)

        logger.info(
            "EDA Report generated successfully: %s",
            EDAReportGenerator.REPORT_PATH,
        )    