from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    artifact_dir: str
    raw_data_path: str
    train_data_path: str
    test_data_path: str