import sys

import pandas as pd

from src.exception.exception import CustomException
from src.logger.logger import logger
from src.preprocessing.artifact_saver import ArtifactSaver
from src.preprocessing.duplicate_handler import DuplicateHandler
from src.preprocessing.feature_scaler import FeatureScaler
from src.preprocessing.feature_target_split import FeatureTargetSplit
from src.preprocessing.train_test_splitter import TrainTestSplitter


class DataPreprocessing:
    """
    Executes the complete
    data preprocessing pipeline.
    """

    def __init__(
        self,
        input_path="data/interim/validated_creditcard.csv",
    ):
        self.input_path = input_path

    def run(self):
        """
        Run the complete preprocessing pipeline.
        """

        try:

            logger.info("=" * 60)
            logger.info("Data Preprocessing Started")
            logger.info("=" * 60)

            # Load validated dataset
            df = pd.read_csv(
                self.input_path
            )

            logger.info(
                "Validated dataset loaded successfully."
            )

            logger.info(
                "Dataset Shape : %s",
                df.shape,
            )

            # Remove duplicates
            df = DuplicateHandler.remove_duplicates(
                df
            )

            # Feature / Target Split
            X, y = FeatureTargetSplit.split(
                df
            )

            # Train Test Split
            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = TrainTestSplitter.split(
                X,
                y,
            )

            # Feature Scaling
            (
                X_train_scaled,
                X_test_scaled,
                scaler,
            ) = FeatureScaler.scale(
                X_train,
                X_test,
            )

            # Save artifacts
            ArtifactSaver.save(
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
                scaler,
            )

            logger.info("=" * 60)
            logger.info("Data Preprocessing Completed Successfully")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":

    preprocessing = DataPreprocessing()

    preprocessing.run()