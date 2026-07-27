import os
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

        self.validated_data_path = os.path.join(
            "data",
            "interim",
            "validated_creditcard.csv",
        )

    def validate_dataset(self):
        """
        Validate the training dataset and
        save the validated dataset.
        """

        try:

            logger.info("=" * 60)
            logger.info("Starting Data Validation")
            logger.info("=" * 60)

            # =====================================================
            # Load Dataset
            # =====================================================

            df = pd.read_csv(
                self.config.train_data_path
            )

            logger.info(
                "Training dataset loaded successfully."
            )

            logger.info(
                "Dataset Shape : %s",
                df.shape,
            )

            # =====================================================
            # Validate Target Column
            # =====================================================

            if "Class" not in df.columns:
                raise ValueError(
                    "Target column 'Class' is missing."
                )

            logger.info(
                "Target column 'Class' found."
            )

            # =====================================================
            # Validate Missing Values
            # =====================================================

            missing_values = int(
                df.isnull().sum().sum()
            )

            if missing_values > 0:
                raise ValueError(
                    f"Dataset contains {missing_values} missing values."
                )

            logger.info(
                "No missing values found."
            )

            # =====================================================
            # Check Duplicate Rows
            # =====================================================

            duplicate_rows = int(
                df.duplicated().sum()
            )

            logger.info(
                "Duplicate rows found : %s",
                duplicate_rows,
            )

            # =====================================================
            # Dataset Information
            # =====================================================

            logger.info(
                "Rows : %s",
                df.shape[0],
            )

            logger.info(
                "Columns : %s",
                df.shape[1],
            )

            # =====================================================
            # Save Validated Dataset
            # =====================================================

            os.makedirs(
                os.path.dirname(
                    self.validated_data_path
                ),
                exist_ok=True,
            )

            df.to_csv(
                self.validated_data_path,
                index=False,
            )

            logger.info(
                "Validated dataset saved successfully."
            )

            logger.info(
                "Location : %s",
                self.validated_data_path,
            )

            logger.info("=" * 60)
            logger.info("Data Validation Completed Successfully")
            logger.info("=" * 60)

            return df

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":

    validator = DataValidation()

    validator.validate_dataset()