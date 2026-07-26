import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.logger.logger import logger


class DataVisualization:
    """
    Generates visualizations for Exploratory Data Analysis.
    """

    FIGURE_DIR = os.path.join("reports", "figures")

    @staticmethod
    def plot_class_distribution(df):
        """
        Generate and save the class distribution bar chart.
        """

        logger.info("=" * 60)
        logger.info("Generating Class Distribution Plot")
        logger.info("=" * 60)

        os.makedirs(DataVisualization.FIGURE_DIR, exist_ok=True)

        class_counts = df["Class"].value_counts().sort_index()

        labels = ["Normal", "Fraud"]

        colors = ["steelblue", "crimson"]

        plt.figure(figsize=(8, 6))

        bars = plt.bar(
            labels,
            class_counts.values,
            color=colors,
            edgecolor="black",
            linewidth=1.2
        )

        plt.title(
            "Credit Card Transaction Class Distribution",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel(
            "Transaction Type",
            fontsize=12
        )

        plt.ylabel(
            "Number of Transactions",
            fontsize=12
        )

        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

        total = len(df)

        for bar, count in zip(bars, class_counts.values):

            percentage = (count / total) * 100

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                count,
                f"{count:,}\n({percentage:.2f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

        plt.tight_layout()

        save_path = os.path.join(
            DataVisualization.FIGURE_DIR,
            "class_distribution.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            f"Class Distribution chart saved to : {save_path}"
        )

    @staticmethod
    def plot_amount_distribution(df):
        """
        Generate and save the transaction amount distribution histogram.
        """

        logger.info("=" * 60)
        logger.info("Generating Amount Distribution Plot")
        logger.info("=" * 60)

        os.makedirs(DataVisualization.FIGURE_DIR, exist_ok=True)

        plt.figure(figsize=(10, 6))

        plt.hist(
            df["Amount"],
            bins=50,
            color="steelblue",
            edgecolor="black"
        )

        plt.title(
            "Transaction Amount Distribution",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel(
            "Transaction Amount",
            fontsize=12
        )

        plt.ylabel(
            "Frequency",
            fontsize=12
        )

        plt.grid(
            linestyle="--",
            alpha=0.4
        )

        plt.tight_layout()

        save_path = os.path.join(
            DataVisualization.FIGURE_DIR,
            "amount_distribution.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            f"Amount Distribution chart saved to : {save_path}"
        )
    @staticmethod
    def plot_log_amount_distribution(df):
        """
        Generate and save the log-transformed transaction amount distribution.
        """

        logger.info("=" * 60)
        logger.info("Generating Log Amount Distribution Plot")
        logger.info("=" * 60)

        os.makedirs(DataVisualization.FIGURE_DIR, exist_ok=True)

        log_amount = np.log1p(df["Amount"])

        plt.figure(figsize=(10, 6))

        plt.hist(
            log_amount,
            bins=50,
            color="darkorange",
            edgecolor="black"
        )

        plt.title(
            "Log-Transformed Transaction Amount Distribution",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel(
            "Log(Transaction Amount + 1)",
            fontsize=12
        )

        plt.ylabel(
            "Frequency",
            fontsize=12
        )

        plt.grid(
            linestyle="--",
            alpha=0.4
        )

        plt.tight_layout()

        save_path = os.path.join(
            DataVisualization.FIGURE_DIR,
            "log_amount_distribution.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            f"Log Amount Distribution chart saved to : {save_path}"
        )
    @staticmethod
    def plot_amount_boxplot(df):
        """
        Generate and save the transaction amount boxplot.
        """

        logger.info("=" * 60)
        logger.info("Generating Amount Boxplot")
        logger.info("=" * 60)

        os.makedirs(DataVisualization.FIGURE_DIR, exist_ok=True)

        plt.figure(figsize=(10, 6))

        plt.boxplot(
            df["Amount"],
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="steelblue"),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(
                marker="o",
                markerfacecolor="crimson",
                markersize=3,
                linestyle="none"
            )
        )

        plt.title(
            "Transaction Amount Boxplot",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel(
            "Transaction Amount",
            fontsize=12
        )

        plt.grid(
            linestyle="--",
            alpha=0.4
        )

        plt.tight_layout()

        save_path = os.path.join(
            DataVisualization.FIGURE_DIR,
            "amount_boxplot.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            f"Amount Boxplot saved to : {save_path}"
        )
    @staticmethod
    def plot_amount_by_class(df):
        """
        Generate and save the transaction amount distribution by class.
        """

        logger.info("=" * 60)
        logger.info("Generating Amount by Class Boxplot")
        logger.info("=" * 60)

        os.makedirs(DataVisualization.FIGURE_DIR, exist_ok=True)

        plt.figure(figsize=(10, 6))

        data = [
            df[df["Class"] == 0]["Amount"],
            df[df["Class"] == 1]["Amount"]
        ]

        plt.boxplot(
            data,
            tick_labels=["Normal", "Fraud"],
            patch_artist=True,
            boxprops=dict(facecolor="steelblue"),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(
                marker="o",
                markerfacecolor="crimson",
                markersize=3,
                linestyle="none"
            )
        )

        plt.title(
            "Transaction Amount by Class",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel(
            "Transaction Class",
            fontsize=12
        )

        plt.ylabel(
            "Transaction Amount",
            fontsize=12
        )

        plt.grid(
            linestyle="--",
            alpha=0.4
        )

        plt.tight_layout()

        save_path = os.path.join(
            DataVisualization.FIGURE_DIR,
            "amount_by_class.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            f"Amount by Class chart saved to : {save_path}"
        )
    @staticmethod
    def plot_time_distribution(df):
        """
        Generate and save the transaction time distribution histogram.
        """

        logger.info("=" * 60)
        logger.info("Generating Time Distribution Plot")
        logger.info("=" * 60)

        os.makedirs(DataVisualization.FIGURE_DIR, exist_ok=True)

        plt.figure(figsize=(10, 6))

        plt.hist(
            df["Time"],
            bins=50,
            color="mediumseagreen",
            edgecolor="black"
        )

        plt.title(
            "Transaction Time Distribution",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel(
            "Time (seconds)",
            fontsize=12
        )

        plt.ylabel(
            "Frequency",
            fontsize=12
        )

        plt.grid(
            linestyle="--",
            alpha=0.4
        )

        plt.tight_layout()

        save_path = os.path.join(
            DataVisualization.FIGURE_DIR,
            "time_distribution.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            f"Time Distribution chart saved to : {save_path}"
        )
    @staticmethod
    def plot_time_by_class(df):
        """
        Generate and save the transaction time distribution by class.
        """

        logger.info("=" * 60)
        logger.info("Generating Time by Class Boxplot")
        logger.info("=" * 60)

        os.makedirs(DataVisualization.FIGURE_DIR, exist_ok=True)

        plt.figure(figsize=(10, 6))

        data = [
        df[df["Class"] == 0]["Time"],
        df[df["Class"] == 1]["Time"]
        ]

        plt.boxplot(
        data,
        tick_labels=["Normal", "Fraud"],
        patch_artist=True,
        boxprops=dict(facecolor="mediumseagreen"),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(
            marker="o",
            markerfacecolor="crimson",
            markersize=3,
            linestyle="none"
        )
        )

        plt.title(
        "Transaction Time by Class",
        fontsize=14,
        fontweight="bold"
        )

        plt.xlabel(
        "Transaction Class",
        fontsize=12
        )

        plt.ylabel(
        "Time (seconds)",
        fontsize=12
        )

        plt.grid(
        linestyle="--",
        alpha=0.4
        )

        plt.tight_layout()

        save_path = os.path.join(
        DataVisualization.FIGURE_DIR,
        "time_by_class.png"
        )

        plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
        )

        plt.close()

        logger.info(
        f"Time by Class chart saved to : {save_path}"
        )
    @staticmethod
    def plot_correlation_heatmap(df):
        """
        Generate and save the feature correlation heatmap.
        """

        logger.info("=" * 60)
        logger.info("Generating Correlation Heatmap")
        logger.info("=" * 60)

        os.makedirs(DataVisualization.FIGURE_DIR, exist_ok=True)

        correlation_matrix = df.corr()

        plt.figure(figsize=(16, 12))

        sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar=True
        )

        plt.title(
        "Feature Correlation Heatmap",
        fontsize=16,
        fontweight="bold"
        )

        plt.tight_layout()

        save_path = os.path.join(
        DataVisualization.FIGURE_DIR,
        "correlation_heatmap.png"
        )

        plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
        )

        plt.close()

        logger.info(
        f"Correlation Heatmap saved to : {save_path}"
        ) 
    @staticmethod
    def plot_feature_target_correlation(df):
        """
        Generate and save the feature correlation with target (Class).
        """

        logger.info("=" * 60)
        logger.info("Generating Feature Correlation with Target")
        logger.info("=" * 60)

        os.makedirs(DataVisualization.FIGURE_DIR, exist_ok=True)

        correlations = (
            df.corr(numeric_only=True)["Class"]
            .drop("Class")
            .sort_values()
        )

        plt.figure(figsize=(10, 8))

        plt.barh(
            correlations.index,
            correlations.values,
            color="steelblue",
            edgecolor="black"
        )

        plt.title(
            "Feature Correlation with Fraud Class",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel(
            "Correlation Coefficient",
            fontsize=12
        )

        plt.ylabel(
            "Features",
            fontsize=12
        )

        plt.grid(
            axis="x",
            linestyle="--",
            alpha=0.4
        )

        plt.tight_layout()

        save_path = os.path.join(
            DataVisualization.FIGURE_DIR,
            "feature_target_correlation.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            f"Feature Target Correlation chart saved to : {save_path}"
        )                       
    