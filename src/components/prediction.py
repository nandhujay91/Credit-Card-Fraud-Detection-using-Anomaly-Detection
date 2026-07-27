"""
Prediction Pipeline Component

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

import sys

from src.logger import logger
from src.exception import CustomException

from src.prediction.prediction_pipeline import PredictionPipeline


class PredictionComponent:
    """
    Executes the complete prediction pipeline.
    """

    @staticmethod
    def run():

        try:

            logger.info("=" * 80)
            logger.info("PREDICTION PIPELINE STARTED")
            logger.info("=" * 80)

            prediction_pipeline = PredictionPipeline(

                input_path="data/prediction/new_transactions.csv"

            )

            prediction_pipeline.run()

            logger.info("=" * 80)
            logger.info("PREDICTION PIPELINE COMPLETED")
            logger.info("=" * 80)

        except Exception as e:

            logger.error(str(e))

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":

    PredictionComponent.run()