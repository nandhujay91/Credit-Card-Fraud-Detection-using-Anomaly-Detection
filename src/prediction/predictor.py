"""
Prediction Module

Uses the trained model to predict fraud transactions.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

import sys
import pandas as pd

from src.logger import logger
from src.exception import CustomException


class Predictor:
    """
    Predict fraud using the trained model.
    """

    @staticmethod
    def predict(model, scaler, data: pd.DataFrame):

        try:

            logger.info("=" * 70)
            logger.info("Starting Prediction")

            # Scale features
            X_scaled = scaler.transform(data)

            # Prediction
            predictions = model.predict(X_scaled)

            # Isolation Forest returns:
            #  1  -> Normal
            # -1  -> Fraud

            labels = [
                "Fraud" if p == -1 else "Normal"
                for p in predictions
            ]

            # Anomaly score
            scores = model.decision_function(X_scaled)

            logger.info("Prediction Completed Successfully")
            logger.info("=" * 70)

            return labels, scores

        except Exception as e:

            logger.error(str(e))

            raise CustomException(
                e,
                sys
            )