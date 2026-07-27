"""
Model Saver Module

Purpose
-------
Save the best anomaly detection model and its metadata.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

import os
import json
import joblib
import tensorflow as tf

from src.logger import logger
from src.exception import CustomException


class ModelSaver:
    """
    Save the best trained model.
    """

    @staticmethod
    def save_model(
            best_model_name,
            best_model,
            best_metrics
    ):

        try:

            logger.info("=" * 70)
            logger.info("Saving Best Model...")

            save_directory = os.path.join(
                "artifacts",
                "model_training"
            )

            os.makedirs(
                save_directory,
                exist_ok=True
            )

            # -----------------------------------------
            # Save Model
            # -----------------------------------------

            if best_model_name == "Autoencoder":

                model_path = os.path.join(
                    save_directory,
                    "best_autoencoder.keras"
                )

                best_model.model.save(model_path)

            else:

                model_path = os.path.join(
                    save_directory,
                    "best_model.pkl"
                )

                joblib.dump(
                    best_model,
                    model_path
                )

            logger.info(f"Model saved at : {model_path}")

            # -----------------------------------------
            # Save Metadata
            # -----------------------------------------

            metadata = {

                "Best Model": best_model_name,
                "Model Path": model_path,
                "Accuracy": float(best_metrics["Accuracy"]),
                "Precision": float(best_metrics["Precision"]),
                "Recall": float(best_metrics["Recall"]),
                "F1 Score": float(best_metrics["F1 Score"]),
                "ROC AUC": float(best_metrics["ROC AUC"])

            }

            metadata_path = os.path.join(
                save_directory,
                "best_model_metadata.json"
            )

            with open(
                    metadata_path,
                    "w"
            ) as file:

                json.dump(
                    metadata,
                    file,
                    indent=4
                )

            logger.info(f"Metadata saved at : {metadata_path}")

            logger.info("=" * 70)
            logger.info("Best model saved successfully.")

        except Exception as e:

            logger.error("Error while saving model.")

            raise CustomException(e)