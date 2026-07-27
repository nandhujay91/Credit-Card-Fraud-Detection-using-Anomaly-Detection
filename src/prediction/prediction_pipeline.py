"""
Prediction Pipeline

Loads new transactions and predicts fraud.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

import sys
import pandas as pd

from src.logger import logger
from src.exception import CustomException

from src.prediction.model_loader import ModelLoader
from src.prediction.predictor import Predictor
from src.prediction.output_writer import OutputWriter


class PredictionPipeline:
    """
    Complete prediction pipeline.
    """

    def __init__(self, input_path):

        self.input_path = input_path

    def run(self):

        try:

            logger.info("=" * 80)
            logger.info("PREDICTION PIPELINE STARTED")
            logger.info("=" * 80)

            # -------------------------------
            # Load Input Data
            # -------------------------------

            data = pd.read_csv(
                self.input_path
            )

            logger.info(
                "Input Shape : %s",
                data.shape
            )

            # -------------------------------
            # Load Model & Scaler
            # -------------------------------

            model, scaler = ModelLoader.load()

            # -------------------------------
            # Predict
            # -------------------------------

            predictions, scores = Predictor.predict(
                model,
                scaler,
                data
            )

            # -------------------------------
            # Save Output
            # -------------------------------

            OutputWriter.save(
                data,
                predictions,
                scores
            )

            logger.info("=" * 80)
            logger.info("PREDICTION PIPELINE COMPLETED")
            logger.info("=" * 80)

        except Exception as e:

            logger.error(str(e))

            raise CustomException(
                e,
                sys
            )