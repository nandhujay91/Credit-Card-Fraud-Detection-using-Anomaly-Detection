from sklearn.model_selection import train_test_split

from src.logger.logger import logger


class TrainTestSplitter:
    """
    Splits the dataset into
    training and testing sets.
    """

    @staticmethod
    def split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    ):
        """
        Split features and target into
        training and testing datasets.

        Args:
            X (pd.DataFrame):
                Feature dataset.

            y (pd.Series):
                Target dataset.

            test_size (float):
                Test dataset size.

            random_state (int):
                Random seed.

        Returns:
            tuple:
                X_train,
                X_test,
                y_train,
                y_test
        """

        logger.info("=" * 60)
        logger.info("Train Test Split Started")
        logger.info("=" * 60)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        logger.info(
            "Train-Test Split completed successfully."
        )

        logger.info(
            "X_train Shape : %s",
            X_train.shape,
        )

        logger.info(
            "X_test Shape : %s",
            X_test.shape,
        )

        logger.info(
            "y_train Shape : %s",
            y_train.shape,
        )

        logger.info(
            "y_test Shape : %s",
            y_test.shape,
        )

        logger.info("=" * 60)
        logger.info("Train Test Split Completed")
        logger.info("=" * 60)

        return (
            X_train,
            X_test,
            y_train,
            y_test,
        )