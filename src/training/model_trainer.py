"""
Model Trainer Module

Purpose
-------
Train all anomaly detection models.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

from src.logger import logger
from src.exception import CustomException
from src.training.model_factory import ModelFactory
from src.training.autoencoder import AutoencoderModel


class ModelTrainer:
    """
    Train all anomaly detection models.
    """

    @staticmethod
    def train_models(X_train):
        """
        Train all models.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training feature matrix.

        Returns
        -------
        dict
            Dictionary containing trained models.
        """

        try:

            logger.info("=" * 70)
            logger.info("Starting Model Training...")

            trained_models = {}

            # ---------------------------------------
            # Classical ML Models
            # ---------------------------------------

            models = ModelFactory.get_models()

            for model_name, model in models.items():

                logger.info(f"Training {model_name}...")

                model.fit(X_train)

                trained_models[model_name] = model

                logger.info(f"{model_name} training completed.")

            # ---------------------------------------
            # Autoencoder
            # ---------------------------------------

            logger.info("Training Autoencoder...")

            autoencoder = AutoencoderModel(
                input_dim=X_train.shape[1]
            )

            autoencoder.train(X_train)

            trained_models["Autoencoder"] = autoencoder

            logger.info("Autoencoder training completed.")

            logger.info("=" * 70)
            logger.info("All models trained successfully.")

            return trained_models

        except Exception as e:
            logger.error("Error occurred during model training.")
            raise CustomException(e)