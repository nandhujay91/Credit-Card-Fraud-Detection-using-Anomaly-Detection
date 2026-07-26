import os
import sys

import pandas as pd

from src.exception.exception import CustomException
from src.logger.logger import logger


class DataValidation:
    """
    Performs basic validation on the training dataset.
    """

    def __init__(self, train_data_path: str):
        self.train_data_path = train_data_path

    def validate_dataset(self):
        """
        Validate the training dataset.
        """

        try:
            logger.info("Starting Data Validation")

            # Check file exists
            if not os.path.exists(self.train_data_path):
                raise FileNotFoundError(
                    f"Dataset not found: {self.train_data_path}"
                )

            logger.info("Dataset file exists.")

            # Read dataset
            df = pd.read_csv(self.train_data_path)

            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

            # Check target column
            if "Class" not in df.columns:
                raise ValueError("Target column 'Class' not found.")

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

            logger.info(f"Duplicate rows: {duplicate_rows}")

            # Check dataset shape
            logger.info(
                f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
            )

            logger.info("Data Validation completed successfully.")

            return True

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":

    train_path = os.path.join(
        "artifacts",
        "data_ingestion",
        "train.csv"
    )

    validator = DataValidation(train_path)

    validator.validate_dataset()