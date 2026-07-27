"""
Prediction Schema

Defines the expected input schema for prediction.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

import sys
import pandas as pd

from src.logger import logger
from src.exception import CustomException


class PredictionSchema:
    """
    Validate prediction input data.
    """

    REQUIRED_COLUMNS = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount"
    ]

    @staticmethod
    def validate(data: pd.DataFrame):

        try:

            logger.info("=" * 70)
            logger.info("Validating Prediction Schema")

            missing_columns = [
                column
                for column in PredictionSchema.REQUIRED_COLUMNS
                if column not in data.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing Columns : {missing_columns}"
                )

            logger.info("Prediction Schema Validation Successful")
            logger.info("=" * 70)

            return data[
                PredictionSchema.REQUIRED_COLUMNS
            ]

        except Exception as e:

            logger.error(str(e))

            raise CustomException(
                e,
                sys
            )