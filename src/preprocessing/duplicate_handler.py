from src.logger.logger import logger


class DuplicateHandler:
    """
    Handles duplicate row detection
    and removal.
    """

    @staticmethod
    def remove_duplicates(df):
        """
        Detect and remove duplicate rows.
        """

        logger.info("=" * 60)
        logger.info("Duplicate Removal Started")
        logger.info("=" * 60)

        # Count rows before removal
        rows_before = len(df)

        # Count duplicate rows
        duplicate_count = df.duplicated().sum()

        logger.info(
            "Duplicate Rows Found : %s",
            duplicate_count,
        )

        # Remove duplicates
        df = df.drop_duplicates()

        # Count rows after removal
        rows_after = len(df)

        logger.info(
            "Rows Before Removal : %s",
            rows_before,
        )

        logger.info(
            "Rows After Removal : %s",
            rows_after,
        )

        logger.info(
            "Rows Removed : %s",
            rows_before - rows_after,
        )

        logger.info("=" * 60)
        logger.info("Duplicate Removal Completed")
        logger.info("=" * 60)

        return df