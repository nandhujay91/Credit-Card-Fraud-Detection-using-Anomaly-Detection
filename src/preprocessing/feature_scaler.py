from sklearn.preprocessing import StandardScaler

from src.logger.logger import logger


class FeatureScaler:
    """
    Scales numerical features using
    StandardScaler.
    """

    @staticmethod
    def scale(
        X_train,
        X_test,
    ):
        """
        Scale training and testing datasets.

        Args:
            X_train (pd.DataFrame):
                Training features.

            X_test (pd.DataFrame):
                Testing features.

        Returns:
            tuple:
                X_train_scaled,
                X_test_scaled,
                scaler
        """

        logger.info("=" * 60)
        logger.info("Feature Scaling Started")
        logger.info("=" * 60)

        scaler = StandardScaler()

        # Fit only on training data
        X_train_scaled = scaler.fit_transform(
            X_train
        )

        # Transform test data
        X_test_scaled = scaler.transform(
            X_test
        )

        logger.info(
            "Training features scaled successfully."
        )

        logger.info(
            "Testing features scaled successfully."
        )

        logger.info(
            "Scaled X_train Shape : %s",
            X_train_scaled.shape,
        )

        logger.info(
            "Scaled X_test Shape : %s",
            X_test_scaled.shape,
        )

        logger.info("=" * 60)
        logger.info("Feature Scaling Completed")
        logger.info("=" * 60)

        return (
            X_train_scaled,
            X_test_scaled,
            scaler,
        )