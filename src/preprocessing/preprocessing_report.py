import os
from datetime import datetime

from src.logger.logger import logger


class PreprocessingReport:
    """
    Generates a preprocessing report.
    """

    @staticmethod
    def generate(
        original_rows,
        cleaned_rows,
        duplicate_rows,
        X_train,
        X_test,
        y_train,
        y_test,
        scaler_name="StandardScaler",
        artifact_dir="artifacts/preprocessing",
    ):
        """
        Generate preprocessing report.

        Args:
            original_rows (int):
                Rows before preprocessing.

            cleaned_rows (int):
                Rows after duplicate removal.

            duplicate_rows (int):
                Number of duplicate rows removed.

            X_train:
                Training features.

            X_test:
                Testing features.

            y_train:
                Training labels.

            y_test:
                Testing labels.

            scaler_name (str):
                Name of scaler used.

            artifact_dir (str):
                Directory to save report.
        """

        logger.info("=" * 60)
        logger.info("Generating Preprocessing Report")
        logger.info("=" * 60)

        os.makedirs(
            artifact_dir,
            exist_ok=True,
        )

        report_path = os.path.join(
            artifact_dir,
            "preprocessing_report.txt",
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as report:

            report.write("=" * 60 + "\n")
            report.write("PREPROCESSING REPORT\n")
            report.write("=" * 60 + "\n\n")

            report.write(
                f"Generated On : {datetime.now()}\n\n"
            )

            report.write("Dataset Summary\n")
            report.write("-" * 60 + "\n")

            report.write(
                f"Original Rows           : {original_rows}\n"
            )

            report.write(
                f"Rows After Cleaning     : {cleaned_rows}\n"
            )

            report.write(
                f"Duplicate Rows Removed  : {duplicate_rows}\n\n"
            )

            report.write("Train Test Split\n")
            report.write("-" * 60 + "\n")

            report.write(
                f"X_train Shape : {X_train.shape}\n"
            )

            report.write(
                f"X_test Shape  : {X_test.shape}\n"
            )

            report.write(
                f"y_train Shape : {y_train.shape}\n"
            )

            report.write(
                f"y_test Shape  : {y_test.shape}\n\n"
            )

            report.write("Feature Scaling\n")
            report.write("-" * 60 + "\n")

            report.write(
                f"Scaler Used : {scaler_name}\n\n"
            )

            report.write("Artifacts Generated\n")
            report.write("-" * 60 + "\n")

            report.write("✓ X_train.npy\n")
            report.write("✓ X_test.npy\n")
            report.write("✓ y_train.npy\n")
            report.write("✓ y_test.npy\n")
            report.write("✓ scaler.pkl\n\n")

            report.write("Status\n")
            report.write("-" * 60 + "\n")
            report.write(
                "Preprocessing Completed Successfully\n"
            )

            report.write("\n")
            report.write("=" * 60)

        logger.info(
            "Preprocessing report generated successfully."
        )

        logger.info(
            "Location : %s",
            report_path,
        )

        logger.info("=" * 60)
        logger.info("Preprocessing Report Completed")
        logger.info("=" * 60)