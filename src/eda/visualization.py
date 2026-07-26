import os

import matplotlib.pyplot as plt

from src.logger.logger import logger


class DataVisualization:
    """
    Generates visualizations for Exploratory Data Analysis.
    """

    @staticmethod
    def plot_class_distribution(df):
        """
        Generate and save the class distribution bar chart.
        """

        logger.info("=" * 60)
        logger.info("Generating Class Distribution Plot")
        logger.info("=" * 60)

        reports_dir = os.path.join("reports", "figures")
        os.makedirs(reports_dir, exist_ok=True)

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

        plt.xlabel("Transaction Type", fontsize=12)

        plt.ylabel("Number of Transactions", fontsize=12)

        plt.grid(axis="y", linestyle="--", alpha=0.4)

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
            reports_dir,
            "class_distribution.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(f"Class Distribution chart saved to : {save_path}")