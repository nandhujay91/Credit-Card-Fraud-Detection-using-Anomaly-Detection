import sys

import pandas as pd

from src.config.configuration import ConfigurationManager
from src.exception.exception import CustomException
from src.logger.logger import logger


class DataValidation:
    """
    Performs validation on the training dataset.
    """

    def __init__(self):
        self.config = ConfigurationManager().get_data_ingestion_config()

    def validate_dataset(self):
        """
        Validate the training dataset.
        """

        try:
            logger.info("=" * 50)
            logger.info("Starting Data Validation")
            logger.info("=" * 50)

            # Read training dataset
            df = pd.read_csv(self.config.train_data_path)

            logger.info(
                f"Training dataset loaded successfully. Shape: {df.shape}"
            )

            # Check target column
            if "Class" not in df.columns:
                raise ValueError(
                    "Target column 'Class' is missing."
                )

            logger.info("Target column 'Class' found.")

            # Check missing values
            missing_values = df.isnull().sum().sum()

            if missing_values > 0:
                raise ValueError(
                    f"Dataset contains {missing_values} missing values."
                )

            logger.info("No missing values found.")

            # Check duplicate rows
            duplicate_rows = df.duplicated().sum()

            logger.info(
                f"Duplicate rows found: {duplicate_rows}"
            )

            # Dataset dimensions
            logger.info(
                f"Rows: {df.shape[0]}"
            )

            logger.info(
                f"Columns: {df.shape[1]}"
            )

            logger.info("=" * 50)
            logger.info("Data Validation completed successfully.")
            logger.info("=" * 50)

            return True

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":

    validator = DataValidation()

    validator.validate_dataset()