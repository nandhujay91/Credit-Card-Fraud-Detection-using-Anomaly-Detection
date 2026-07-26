import os

from src.constants import (
    ARTIFACT_DIR,
    DATA_INGESTION_DIR,
    RAW_DATA_FILE_NAME,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
)

from src.entity.config_entity import DataIngestionConfig


class ConfigurationManager:

    def get_data_ingestion_config(self):

        ingestion_dir = os.path.join(
            ARTIFACT_DIR,
            DATA_INGESTION_DIR
        )

        os.makedirs(ingestion_dir, exist_ok=True)

        return DataIngestionConfig(

            artifact_dir=ingestion_dir,

            raw_data_path=os.path.join(
                ingestion_dir,
                RAW_DATA_FILE_NAME
            ),

            train_data_path=os.path.join(
                ingestion_dir,
                TRAIN_FILE_NAME
            ),

            test_data_path=os.path.join(
                ingestion_dir,
                TEST_FILE_NAME
            ),
        )