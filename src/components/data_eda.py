import sys
import pandas as pd

from src.config.configuration import ConfigurationManager
from src.eda.summary import DataSummary
from src.eda.visualization import DataVisualization
from src.eda.report_generator import EDAReportGenerator
from src.exception.exception import CustomException
from src.logger.logger import logger


class DataEDA:
    """
    Orchestrates the Exploratory Data Analysis (EDA) workflow.
    """

    def __init__(self):
        self.config = ConfigurationManager().get_data_ingestion_config()

    def perform_eda(self):
        """
        Execute the complete EDA workflow.
        """

        try:
            logger.info("=" * 60)
            logger.info("Starting Exploratory Data Analysis")
            logger.info("=" * 60)

            # ==========================================
            # Load Training Dataset
            # ==========================================

            df = pd.read_csv(self.config.train_data_path)

            logger.info("Training dataset loaded successfully.")

            # ==========================================
            # Generate Dataset Summary
            # ==========================================

            summary = DataSummary.generate_summary(df)

            # ==========================================
            # Generate Visualizations
            # ==========================================

            DataVisualization.plot_class_distribution(df)
            DataVisualization.plot_amount_distribution(df)
            DataVisualization.plot_log_amount_distribution(df)
            DataVisualization.plot_amount_boxplot(df)
            DataVisualization.plot_amount_by_class(df)
            DataVisualization.plot_time_distribution(df)
            DataVisualization.plot_time_by_class(df)
            DataVisualization.plot_correlation_heatmap(df)
            DataVisualization.plot_feature_target_correlation(df)

            # ==========================================
            # Generate EDA Report
            # ==========================================

            EDAReportGenerator.generate_report(summary)

            logger.info("=" * 60)
            logger.info("EDA Completed Successfully")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":

    eda = DataEDA()
    eda.perform_eda()