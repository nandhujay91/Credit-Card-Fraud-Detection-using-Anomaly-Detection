from src.logger.logger import logger


class FeatureTargetSplit:
    """
    Splits the dataset into
    features (X) and target (y).
    """

    @staticmethod
    def split(df):
        """
        Split the dataset into
        features and target.

        Args:
            df (pd.DataFrame):
                Input dataset.

        Returns:
            tuple:
                X (pd.DataFrame): Feature dataset.
                y (pd.Series): Target variable.
        """

        logger.info("=" * 60)
        logger.info("Feature Target Split Started")
        logger.info("=" * 60)

        # Separate Features
        X = df.drop(
            columns=["Class"]
        )

        # Separate Target
        y = df["Class"]

        logger.info(
            "Feature dataset created successfully."
        )

        logger.info(
            "Target dataset created successfully."
        )

        logger.info(
            "Feature Shape : %s",
            X.shape,
        )

        logger.info(
            "Target Shape : %s",
            y.shape,
        )

        logger.info(
            "Feature Columns : %s",
            list(X.columns),
        )

        logger.info(
            "Target Column : %s",
            y.name,
        )

        logger.info("=" * 60)
        logger.info("Feature Target Split Completed")
        logger.info("=" * 60)

        return X, y