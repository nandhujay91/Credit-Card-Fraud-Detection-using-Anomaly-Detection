"""
Model Training Pipeline

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

import sys
import numpy as np

from src.logger import logger
from src.exception import CustomException

from src.training.model_trainer import ModelTrainer
from src.training.hyperparameter_tuning import HyperparameterTuner
from src.training.model_evaluator import ModelEvaluator
from src.training.model_selector import ModelSelector
from src.training.model_saver import ModelSaver
from src.training.training_report import TrainingReport


class ModelTrainingPipeline:
    """
    Complete Model Training Pipeline
    """

    @staticmethod
    def run():

        try:

            logger.info("=" * 80)
            logger.info("MODEL TRAINING PIPELINE STARTED")
            logger.info("=" * 80)

            # ==========================================================
            # Load Preprocessed Data
            # ==========================================================

            X_train = np.load(
                "artifacts/preprocessing/X_train.npy"
            )

            X_test = np.load(
                "artifacts/preprocessing/X_test.npy"
            )

            y_train = np.load(
                "artifacts/preprocessing/y_train.npy"
            )

            y_test = np.load(
                "artifacts/preprocessing/y_test.npy"
            )

            logger.info("Preprocessed artifacts loaded successfully.")

            # ==========================================================
            # Train Baseline Models
            # ==========================================================

            logger.info("Training baseline models...")

            trained_models = ModelTrainer.train_models(
                X_train
            )

            logger.info("Baseline model training completed.")

            # ==========================================================
            # Hyperparameter Tuning
            # ==========================================================

            logger.info("Running Hyperparameter Tuning...")

            trained_models = HyperparameterTuner.optimize_models(
                trained_models=trained_models,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test
            )

            logger.info("Hyperparameter tuning completed.")

            # ==========================================================
            # Evaluate Models
            # ==========================================================

            logger.info("Evaluating trained models...")

            evaluation_results = ModelEvaluator.evaluate_models(
                trained_models,
                X_test,
                y_test
            )

            logger.info("Model evaluation completed.")

            # ==========================================================
            # Select Best Model
            # ==========================================================

            logger.info("Selecting best model...")

            (
                best_model_name,
                best_model,
                best_metrics
            ) = ModelSelector.select_best_model(
                trained_models,
                evaluation_results
            )

            logger.info(
                f"Best Model Selected : {best_model_name}"
            )

            # ==========================================================
            # Save Best Model
            # ==========================================================

            logger.info("Saving best model...")

            ModelSaver.save_model(
                best_model_name,
                best_model,
                best_metrics
            )

            logger.info("Best model saved successfully.")

            # ==========================================================
            # Generate Training Report
            # ==========================================================

            logger.info("Generating training report...")

            TrainingReport.generate_report(
                evaluation_results,
                best_model_name,
                best_metrics
            )

            logger.info("Training report generated successfully.")

            logger.info("=" * 80)
            logger.info("MODEL TRAINING PIPELINE COMPLETED")
            logger.info("=" * 80)

        except Exception as e:

            logger.error(f"Model Training Pipeline Failed : {str(e)}")

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":

    ModelTrainingPipeline.run()