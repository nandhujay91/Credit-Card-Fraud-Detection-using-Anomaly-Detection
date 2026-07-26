import os
import shutil
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.configuration import ConfigurationManager
from src.exception.exception import CustomException
from src.logger.logger import logger


class DataIngestion:
    """
    Handles data ingestion:
    1. Copies raw dataset to artifacts directory
    2. Reads the dataset
    3. Splits into train and test sets
    4. Saves processed datasets
    """

    def __init__(self):
        self.config = ConfigurationManager().get_data_ingestion_config()

    def initiate_data_ingestion(self):
        """
        Executes the complete data ingestion pipeline.
        """

        logger.info("Started Data Ingestion")

        try:
            # Source dataset location
            source_path = os.path.join(
                "data",
                "raw",
                "creditcard.csv"
            )

            # Check whether dataset exists
            if not os.path.exists(source_path):
                raise FileNotFoundError(
                    f"Dataset not found at {source_path}"
                )

            logger.info("Raw dataset found successfully.")

            # Copy dataset into artifacts directory
            shutil.copy(
                source_path,
                self.config.raw_data_path
            )

            logger.info(
                f"Dataset copied to {self.config.raw_data_path}"
            )

            # Read dataset
            df = pd.read_csv(self.config.raw_data_path)

            logger.info(
                f"Dataset loaded successfully. Shape: {df.shape}"
            )

            # Train-Test Split
            train_df, test_df = train_test_split(
                df,
                test_size=0.20,
                random_state=42,
                stratify=df["Class"]
            )

            logger.info("Train-Test split completed.")

            # Save train dataset
            train_df.to_csv(
                self.config.train_data_path,
                index=False
            )

            # Save test dataset
            test_df.to_csv(
                self.config.test_data_path,
                index=False
            )

            logger.info("Train dataset saved.")
            logger.info("Test dataset saved.")

            logger.info("Data Ingestion completed successfully.")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    print(f"Train Data : {train_path}")
    print(f"Test Data  : {test_path}")