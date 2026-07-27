import io

from src.logger.logger import logger


class DataSummary:
    """
    Performs summary analysis on the dataset.
    """

    @staticmethod
    def generate_summary(df):
        """
        Generate dataset summary, log the results,
        and return summary statistics for report generation.
        """

        logger.info("=" * 60)
        logger.info("Dataset Summary")
        logger.info("=" * 60)

        # =====================================================
        # Dataset Shape
        # =====================================================

        total_rows, total_columns = df.shape

        logger.info("Dataset Shape")
        logger.info(f"Rows    : {total_rows}")
        logger.info(f"Columns : {total_columns}")

        # =====================================================
        # Dataset Information
        # =====================================================

        buffer = io.StringIO()
        df.info(buf=buffer)
        dataset_info = buffer.getvalue()

        logger.info("\nDataset Information")
        logger.info(dataset_info)

        # =====================================================
        # Data Types
        # =====================================================

        data_types = df.dtypes

        logger.info("\nData Types")
        logger.info("\n%s", data_types)

        numerical_columns = (
            df.select_dtypes(include=["number"])
            .columns
            .tolist()
        )

        categorical_columns = (
            df.select_dtypes(exclude=["number"])
            .columns
            .tolist()
        )

        # =====================================================
        # Memory Usage
        # =====================================================

        memory_usage_mb = round(
            df.memory_usage(deep=True).sum() / (1024 ** 2),
            2
        )

        logger.info("\nMemory Usage")
        logger.info(f"{memory_usage_mb} MB")

        # =====================================================
        # Missing Values
        # =====================================================

        missing_values = df.isnull().sum()

        total_missing_values = int(
            missing_values.sum()
        )

        logger.info("\nMissing Values")
        logger.info("\n%s", missing_values)

        # =====================================================
        # Duplicate Rows
        # =====================================================

        duplicate_rows = int(
            df.duplicated().sum()
        )

        logger.info("\nDuplicate Rows")
        logger.info(duplicate_rows)

        # =====================================================
        # Data Quality Statistics
        # =====================================================

        missing_percentage = round(
        (total_missing_values / total_rows) * 100,
        2,
        )

        duplicate_percentage = round(
        (duplicate_rows / total_rows) * 100,
        2,
        )

        data_quality_statistics = {
        "missing_percentage": missing_percentage,
        "duplicate_percentage": duplicate_percentage,
        }

        logger.info("\nData Quality Statistics")
        logger.info("\n%s", data_quality_statistics)

        # =====================================================
        # Summary Statistics
        # =====================================================

        summary_statistics = df.describe()

        logger.info("\nSummary Statistics")
        logger.info("\n%s", summary_statistics)

        # =====================================================
        # Amount Statistics
        # =====================================================

        amount_statistics = {
            "minimum": round(df["Amount"].min(), 2),
            "maximum": round(df["Amount"].max(), 2),
            "mean": round(df["Amount"].mean(), 2),
            "median": round(df["Amount"].median(), 2),
            "std": round(df["Amount"].std(), 2),
            "q1": round(df["Amount"].quantile(0.25), 2),
            "q3": round(df["Amount"].quantile(0.75), 2),
        }

        logger.info("\nAmount Statistics")
        logger.info("\n%s", amount_statistics)
        # =====================================================
        # Time Statistics
        # =====================================================

        time_statistics = {
        "minimum": int(df["Time"].min()),
        "maximum": int(df["Time"].max()),
        "mean": round(df["Time"].mean(), 2),
        "median": round(df["Time"].median(), 2),
        "std": round(df["Time"].std(), 2),
        "q1": round(df["Time"].quantile(0.25), 2),
        "q3": round(df["Time"].quantile(0.75), 2),
        }

        logger.info("\nTime Statistics")
        logger.info("\n%s", time_statistics)

        # =====================================================
        # Class Distribution
        # =====================================================

        class_distribution = (
            df["Class"]
            .value_counts()
            .sort_index()
        )

        logger.info("\nClass Distribution")
        logger.info("\n%s", class_distribution)

        # =====================================================
        # Class Percentage
        # =====================================================

        class_percentage = (
            df["Class"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        )

        logger.info("\nClass Percentage")
        logger.info("\n%s", class_percentage)

        # =====================================================
        # Class Statistics
        # =====================================================

        normal_count = int(class_distribution.get(0, 0))
        fraud_count = int(class_distribution.get(1, 0))

        imbalance_ratio = (
        round(normal_count / fraud_count, 2)
        if fraud_count > 0
        else None
        )

        class_statistics = {
        "normal_count": normal_count,
        "fraud_count": fraud_count,
        "normal_percentage": round(
        class_percentage.get(0, 0), 2
        ),
        "fraud_percentage": round(
        class_percentage.get(1, 0), 2
        ),
        "imbalance_ratio": imbalance_ratio,
        }

        logger.info("\nClass Statistics")
        logger.info("\n%s", class_statistics)        

        # =====================================================
        # Correlation Analysis
        # =====================================================

        correlation_matrix = df.corr(numeric_only=True)

        target_correlation = (
            correlation_matrix["Class"]
            .drop("Class")
            .sort_values(key=abs, ascending=False)
        )

        top_positive_features = (
            target_correlation[target_correlation > 0]
            .head(5)
        )

        top_negative_features = (
            target_correlation[target_correlation < 0]
            .head(5)
        )

        logger.info("\nTop Positive Correlations")
        logger.info("\n%s", top_positive_features)

        logger.info("\nTop Negative Correlations")
        logger.info("\n%s", top_negative_features)
        # =====================================================
        # Correlation Statistics
        # =====================================================
        correlation_statistics = {
            "top_positive": top_positive_features.to_dict(),
            "top_negative": top_negative_features.to_dict(),
        }
        logger.info("\nCorrelation Statistics")
        logger.info("\n%s", correlation_statistics)


        logger.info("=" * 60)
        logger.info("Dataset Summary Completed")
        logger.info("=" * 60)

        return {

            # =================================================
            # Dataset Overview
            # =================================================

            "total_rows": total_rows,
            "total_columns": total_columns,
            "dataset_shape": (total_rows, total_columns),
            "dataset_info": dataset_info,
            "memory_usage_mb": memory_usage_mb,

            # =================================================
            # Features
            # =================================================

            "feature_names": df.columns.tolist(),
            "data_types": data_types.to_dict(),
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,

            # =================================================
            # Data Quality
            # =================================================

            "missing_values": missing_values.to_dict(),
            "total_missing_values": total_missing_values,
            "duplicate_rows": duplicate_rows,
            "data_quality_statistics": data_quality_statistics,

            # =================================================
            # Statistics
            # =================================================

            "summary_statistics": summary_statistics.to_dict(),
            "amount_statistics": amount_statistics,
            "time_statistics": time_statistics,

            # =================================================
            # Target Analysis
            # =================================================

            "class_distribution": class_distribution.to_dict(),
            "class_percentage": class_percentage.to_dict(),
            "class_statistics": class_statistics,

            # =================================================
            # Correlation
            # =================================================

            "top_positive_correlations":
                top_positive_features.to_dict(),

            "top_negative_correlations":
                top_negative_features.to_dict(),
            "correlation_statistics":
                correlation_statistics,

        }