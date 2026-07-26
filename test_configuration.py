from src.config.configuration import ConfigurationManager

config = ConfigurationManager().get_data_ingestion_config()

print(config)