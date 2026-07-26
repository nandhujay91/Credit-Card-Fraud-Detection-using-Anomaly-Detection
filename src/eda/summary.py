import io

from src.logger.logger import logger


class DataSummary:
    """
    Performs summary analysis on the dataset.
    """

    @staticmethod
    def generate_summary(df):
        """
        Generate and log dataset summary.
        """

        logger.info("=" * 60)
        logger.info("Dataset Summary")
        logger.info("=" * 60)

        # Dataset Shape
        logger.info(f"Dataset Shape : {df.shape}")

        # Dataset Information
        buffer = io.StringIO()
        df.info(buf=buffer)

        logger.info("\nDataset Information")
        logger.info(buffer.getvalue())

        # Missing Values
        logger.info("\nMissing Values")
        logger.info("\n%s", df.isnull().sum())

        # Duplicate Rows
        logger.info("\nDuplicate Rows")
        logger.info(df.duplicated().sum())

        # Summary Statistics
        logger.info("\nSummary Statistics")
        logger.info("\n%s", df.describe())

        # Class Distribution
        logger.info("\nClass Distribution")
        logger.info("\n%s", df["Class"].value_counts())

        # Class Percentage
        logger.info("\nClass Percentage")
        logger.info(
            "\n%s",
            df["Class"].value_counts(normalize=True) * 100
        )

        logger.info("=" * 60)
        logger.info("Dataset Summary Completed")
        logger.info("=" * 60)