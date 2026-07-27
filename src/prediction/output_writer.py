"""
Prediction Output Writer

Saves prediction results.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

import os
import sys
import pandas as pd

from src.logger import logger
from src.exception import CustomException


class OutputWriter:
    """
    Save prediction results.
    """

    OUTPUT_PATH = os.path.join(
        "artifacts",
        "prediction",
        "predictions.csv"
    )

    @staticmethod
    def save(
            input_data: pd.DataFrame,
            predictions,
            scores
    ):

        try:

            logger.info("=" * 70)
            logger.info("Saving Prediction Results")

            output_df = input_data.copy()

            output_df["Prediction"] = predictions
            output_df["Anomaly Score"] = scores

            os.makedirs(
                os.path.dirname(
                    OutputWriter.OUTPUT_PATH
                ),
                exist_ok=True
            )

            output_df.to_csv(
                OutputWriter.OUTPUT_PATH,
                index=False
            )

            logger.info(
                f"Prediction file saved at : {OutputWriter.OUTPUT_PATH}"
            )

            logger.info("=" * 70)

        except Exception as e:

            logger.error(str(e))

            raise CustomException(
                e,
                sys
            )