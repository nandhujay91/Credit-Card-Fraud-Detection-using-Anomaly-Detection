"""
Model Loader

Loads the trained model and preprocessing artifacts
required for prediction.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

import os
import joblib
import sys

from src.logger import logger
from src.exception import CustomException


class ModelLoader:
    """
    Load trained model and scaler.
    """

    MODEL_PATH = os.path.join(
        "artifacts",
        "model_training",
        "best_model.pkl"
    )

    SCALER_PATH = os.path.join(
        "artifacts",
        "preprocessing",
        "scaler.pkl"
    )

    @staticmethod
    def load():

        try:

            logger.info("=" * 70)
            logger.info("Loading Prediction Artifacts")

            model = joblib.load(
                ModelLoader.MODEL_PATH
            )

            scaler = joblib.load(
                ModelLoader.SCALER_PATH
            )

            logger.info("Model Loaded Successfully")
            logger.info("Scaler Loaded Successfully")

            logger.info("=" * 70)

            return model, scaler

        except Exception as e:

            logger.error(str(e))

            raise CustomException(
                e,
                sys
            )