"""
Model Evaluator Module

Purpose
-------
Evaluate all trained anomaly detection models.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

import sys
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.logger import logger
from src.exception import CustomException


class ModelEvaluator:
    """
    Evaluate all trained anomaly detection models.
    """

    @staticmethod
    def evaluate_models(
        trained_models,
        X_test,
        y_test
    ):

        try:

            logger.info("=" * 80)
            logger.info("MODEL EVALUATION STARTED")
            logger.info("=" * 80)

            evaluation_results = {}

            for model_name, model in trained_models.items():

                logger.info(f"Evaluating Model : {model_name}")

                # =====================================================
                # Isolation Forest
                # =====================================================

                if model_name == "IsolationForest":

                    raw_predictions = model.predict(X_test)

                    predictions = np.where(
                        raw_predictions == -1,
                        1,
                        0
                    )

                    anomaly_scores = -model.decision_function(
                        X_test
                    )

                # =====================================================
                # Local Outlier Factor
                # =====================================================

                elif model_name == "LocalOutlierFactor":

                    raw_predictions = model.predict(X_test)

                    predictions = np.where(
                        raw_predictions == -1,
                        1,
                        0
                    )

                    anomaly_scores = -model.decision_function(
                        X_test
                    )

                # =====================================================
                # Autoencoder
                # =====================================================

                elif model_name == "Autoencoder":

                    reconstruction_errors = (
                        model.reconstruction_error(
                            X_test
                        )
                    )

                    threshold = np.percentile(
                        reconstruction_errors,
                        95
                    )

                    predictions = model.predict(
                        X_test,
                        threshold
                    )

                    anomaly_scores = reconstruction_errors

                else:

                    logger.warning(
                        f"Skipping unsupported model : {model_name}"
                    )

                    continue

                # =====================================================
                # Metrics
                # =====================================================

                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                )

                try:

                    roc_auc = roc_auc_score(
                        y_test,
                        anomaly_scores
                    )

                except ValueError:

                    roc_auc = 0.0

                cm = confusion_matrix(
                    y_test,
                    predictions
                )

                report = classification_report(
                    y_test,
                    predictions,
                    zero_division=0
                )

                # =====================================================
                # Store Results
                # =====================================================

                evaluation_results[model_name] = {

                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "ROC AUC": roc_auc,
                    "Confusion Matrix": cm,
                    "Classification Report": report

                }

                # =====================================================
                # Logging
                # =====================================================

                logger.info(f"{model_name} Evaluation Completed")

                logger.info(f"Accuracy  : {accuracy:.4f}")
                logger.info(f"Precision : {precision:.4f}")
                logger.info(f"Recall    : {recall:.4f}")
                logger.info(f"F1 Score  : {f1:.4f}")
                logger.info(f"ROC AUC   : {roc_auc:.4f}")

                logger.info("Confusion Matrix")

                logger.info(f"\n{cm}")

                logger.info("=" * 80)

            logger.info("MODEL EVALUATION COMPLETED")

            return evaluation_results

        except Exception as e:

            logger.error(
                f"Model Evaluation Failed : {str(e)}"
            )

            raise CustomException(
                e,
                sys
            )