import os
import joblib
import numpy as np

from src.logger.logger import logger


class ArtifactSaver:
    """
    Saves preprocessing artifacts
    required for model training
    and inference.
    """

    @staticmethod
    def save(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        artifact_dir="artifacts/preprocessing",
    ):
        """
        Save preprocessing artifacts.

        Args:
            X_train:
                Scaled training features.

            X_test:
                Scaled testing features.

            y_train:
                Training labels.

            y_test:
                Testing labels.

            scaler:
                Trained StandardScaler.

            artifact_dir:
                Directory to save artifacts.
        """

        logger.info("=" * 60)
        logger.info("Saving Preprocessing Artifacts Started")
        logger.info("=" * 60)

        os.makedirs(
            artifact_dir,
            exist_ok=True,
        )

        # Save datasets
        np.save(
            os.path.join(
                artifact_dir,
                "X_train.npy",
            ),
            X_train,
        )

        np.save(
            os.path.join(
                artifact_dir,
                "X_test.npy",
            ),
            X_test,
        )

        np.save(
            os.path.join(
                artifact_dir,
                "y_train.npy",
            ),
            y_train,
        )

        np.save(
            os.path.join(
                artifact_dir,
                "y_test.npy",
            ),
            y_test,
        )

        # Save scaler
        joblib.dump(
            scaler,
            os.path.join(
                artifact_dir,
                "scaler.pkl",
            ),
        )

        logger.info(
            "All preprocessing artifacts saved successfully."
        )

        logger.info(
            "Artifact Location : %s",
            artifact_dir,
        )

        logger.info("=" * 60)
        logger.info("Saving Preprocessing Artifacts Completed")
        logger.info("=" * 60)