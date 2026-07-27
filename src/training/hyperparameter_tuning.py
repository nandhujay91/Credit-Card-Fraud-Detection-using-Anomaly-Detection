"""
Hyperparameter Tuning Module

Purpose
-------
Optimize anomaly detection models using Optuna.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

import sys
import optuna

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

from src.logger import logger
from src.exception import CustomException


class HyperparameterTuner:
    """
    Hyperparameter tuning for anomaly detection models.
    """

    @staticmethod
    def optimize_models(
        trained_models,
        X_train,
        X_test,
        y_test
    ):
        """
        Optimize all supported models.
        """

        try:

            logger.info("=" * 80)
            logger.info("HYPERPARAMETER TUNING STARTED")
            logger.info("=" * 80)

            # ---------------------------------------------------------
            # Tune Isolation Forest
            # ---------------------------------------------------------

            trained_models["IsolationForest"] = (
                HyperparameterTuner._tune_isolation_forest(
                    X_train,
                    X_test,
                    y_test
                )
            )

            # ---------------------------------------------------------
            # Future Extensions
            # ---------------------------------------------------------

            # trained_models["LocalOutlierFactor"] = (
            #     HyperparameterTuner._tune_lof(...)
            # )

            # trained_models["Autoencoder"] = (
            #     HyperparameterTuner._tune_autoencoder(...)
            # )

            logger.info("=" * 80)
            logger.info("HYPERPARAMETER TUNING COMPLETED")
            logger.info("=" * 80)

            return trained_models

        except Exception as e:

            logger.error(str(e))

            raise CustomException(
                e,
                sys
            )

    @staticmethod
    def _tune_isolation_forest(
        X_train,
        X_test,
        y_test
    ):
        """
        Tune Isolation Forest using Optuna.
        """

        sampler = optuna.samplers.TPESampler(
            seed=42
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )

        def objective(trial):

            model = IsolationForest(

                n_estimators=trial.suggest_int(
                    "n_estimators",
                    100,
                    300,
                    step=50
                ),

                max_samples=trial.suggest_float(
                    "max_samples",
                    0.6,
                    1.0
                ),

                contamination=trial.suggest_float(
                    "contamination",
                    0.001,
                    0.005
                ),

                max_features=trial.suggest_float(
                    "max_features",
                    0.6,
                    1.0
                ),

                bootstrap=trial.suggest_categorical(
                    "bootstrap",
                    [True, False]
                ),

                random_state=42

            )

            model.fit(X_train)

            prediction = model.predict(X_test)

            prediction = [
                1 if p == -1 else 0
                for p in prediction
            ]

            return f1_score(
                y_test,
                prediction
            )

        study.optimize(
            objective,
            n_trials=30,
            show_progress_bar=True
        )

        logger.info(
            f"Best Isolation Forest F1 : {study.best_value:.4f}"
        )

        logger.info(
            f"Best Parameters : {study.best_params}"
        )

        best_model = IsolationForest(

            **study.best_params,

            random_state=42

        )

        best_model.fit(X_train)

        return best_model