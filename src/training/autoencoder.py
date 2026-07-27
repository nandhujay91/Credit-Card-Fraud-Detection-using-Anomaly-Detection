"""
Autoencoder Module

Purpose
-------
Build and train a Deep Autoencoder for anomaly detection.

Author : Nandini Arjunan
Project: Credit Card Fraud Detection using Anomaly Detection
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from src.logger import logger
from src.exception import CustomException


class AutoencoderModel:
    """
    Deep Autoencoder for anomaly detection.
    """

    def __init__(self, input_dim):

        self.input_dim = input_dim
        self.model = self.build_model()

    def build_model(self):
        """
        Build Autoencoder Architecture.
        """

        try:

            logger.info("Building Autoencoder model...")

            input_layer = Input(shape=(self.input_dim,))

            # Encoder
            encoder = Dense(32, activation="relu")(input_layer)
            encoder = Dense(16, activation="relu")(encoder)
            encoder = Dense(8, activation="relu")(encoder)

            # Bottleneck
            bottleneck = Dense(4, activation="relu")(encoder)

            # Decoder
            decoder = Dense(8, activation="relu")(bottleneck)
            decoder = Dense(16, activation="relu")(decoder)
            decoder = Dense(32, activation="relu")(decoder)

            output_layer = Dense(
                self.input_dim,
                activation="linear"
            )(decoder)

            model = Model(inputs=input_layer, outputs=output_layer)

            model.compile(
                optimizer="adam",
                loss="mse"
            )

            logger.info("Autoencoder created successfully.")

            return model

        except Exception as e:
            raise CustomException(e)

    def train(
        self,
        X_train,
        epochs=50,
        batch_size=256,
        validation_split=0.2
    ):
        """
        Train Autoencoder.
        """

        try:

            logger.info("Training Autoencoder...")

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )

            history = self.model.fit(
                X_train,
                X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                shuffle=True,
                callbacks=[early_stop],
                verbose=1
            )

            logger.info("Autoencoder training completed.")

            return history

        except Exception as e:
            raise CustomException(e)

    def reconstruction_error(self, X):
        """
        Compute reconstruction error.
        """

        try:

            reconstructed = self.model.predict(
                X,
                verbose=0
            )

            mse = tf.reduce_mean(
                tf.square(X - reconstructed),
                axis=1
            ).numpy()

            return mse

        except Exception as e:
            raise CustomException(e)

    def predict(self, X, threshold):
        """
        Predict anomalies.

        Returns
        -------
        0 -> Normal
        1 -> Fraud
        """

        try:

            errors = self.reconstruction_error(X)

            predictions = (errors > threshold).astype(int)

            return predictions

        except Exception as e:
            raise CustomException(e)