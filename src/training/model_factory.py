"""
Model Factory Module

Purpose:
--------
Creates and returns all machine learning models used for anomaly detection.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.logger import logger
from src.exception import CustomException


class ModelFactory:
    """
    Factory class for creating anomaly detection models.
    """

    @staticmethod
    def get_models():
        """
        Returns all anomaly detection models.

        Returns
        -------
        dict
            Dictionary containing model name and initialized model object.
        """

        try:

            logger.info("=" * 60)
            logger.info("Initializing anomaly detection models...")

            models = {

                "IsolationForest": IsolationForest(
                    n_estimators=200,
                    contamination="auto",
                    random_state=42,
                    n_jobs=-1
                ),

                "LocalOutlierFactor": LocalOutlierFactor(
                    n_neighbors=20,
                    contamination="auto",
                    novelty=True
                )

            }

            logger.info("Models initialized successfully.")

            for model_name in models.keys():
                logger.info(f"Loaded Model : {model_name}")

            logger.info("=" * 60)

            return models

        except Exception as e:
            logger.error("Error while creating models.")
            raise CustomException(e)