"""
Model Selector Module

Purpose
-------
Select the best anomaly detection model based on evaluation metrics.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

from src.logger import logger
from src.exception import CustomException


class ModelSelector:
    """
    Select the best performing model.
    """

    @staticmethod
    def select_best_model(
            trained_models,
            evaluation_results
    ):
        """
        Select the best model using F1 Score.

        Parameters
        ----------
        trained_models : dict
            Dictionary containing trained models.

        evaluation_results : dict
            Dictionary containing evaluation metrics.

        Returns
        -------
        tuple
            best_model_name,
            best_model,
            best_metrics
        """

        try:

            logger.info("=" * 70)
            logger.info("Selecting Best Model...")

            best_model_name = None
            best_model = None
            best_metrics = None

            best_f1 = -1

            for model_name, metrics in evaluation_results.items():

                current_f1 = metrics["F1 Score"]

                logger.info(
                    f"{model_name} --> F1 Score : {current_f1:.4f}"
                )

                if current_f1 > best_f1:

                    best_f1 = current_f1
                    best_model_name = model_name
                    best_model = trained_models[model_name]
                    best_metrics = metrics

            logger.info("=" * 70)
            logger.info(
                f"Best Model Selected : {best_model_name}"
            )

            logger.info(
                f"Best F1 Score : {best_f1:.4f}"
            )

            logger.info("=" * 70)

            return (

                best_model_name,
                best_model,
                best_metrics

            )

        except Exception as e:

            logger.error(
                "Error while selecting best model."
            )

            raise CustomException(e)