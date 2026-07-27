"""
Training Report Module

Purpose
-------
Generate training summary report.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

import os
from datetime import datetime

from src.logger import logger
from src.exception import CustomException


class TrainingReport:

    @staticmethod
    def generate_report(
        evaluation_results,
        best_model_name,
        best_metrics
    ):

        try:

            logger.info("Generating training report...")

            report_directory = os.path.join(
                "artifacts",
                "model_training"
            )

            os.makedirs(
                report_directory,
                exist_ok=True
            )

            report_path = os.path.join(
                report_directory,
                "training_report.txt"
            )

            with open(report_path, "w") as file:

                file.write("=" * 80 + "\n")
                file.write("MODEL TRAINING REPORT\n")
                file.write("=" * 80 + "\n\n")

                file.write(
                    f"Generated On : {datetime.now()}\n\n"
                )

                file.write("MODEL COMPARISON\n")
                file.write("-" * 80 + "\n")

                for model_name, metrics in evaluation_results.items():

                    file.write(f"\nModel : {model_name}\n")
                    file.write(f"Accuracy : {metrics['Accuracy']:.4f}\n")
                    file.write(f"Precision : {metrics['Precision']:.4f}\n")
                    file.write(f"Recall : {metrics['Recall']:.4f}\n")
                    file.write(f"F1 Score : {metrics['F1 Score']:.4f}\n")
                    file.write(f"ROC AUC : {metrics['ROC AUC']:.4f}\n")

                file.write("\n")
                file.write("=" * 80 + "\n")
                file.write("BEST MODEL\n")
                file.write("=" * 80 + "\n\n")

                file.write(f"Model : {best_model_name}\n")
                file.write(f"Accuracy : {best_metrics['Accuracy']:.4f}\n")
                file.write(f"Precision : {best_metrics['Precision']:.4f}\n")
                file.write(f"Recall : {best_metrics['Recall']:.4f}\n")
                file.write(f"F1 Score : {best_metrics['F1 Score']:.4f}\n")
                file.write(f"ROC AUC : {best_metrics['ROC AUC']:.4f}\n")

            logger.info(f"Training report saved at : {report_path}")

        except Exception as e:

            raise CustomException(e)