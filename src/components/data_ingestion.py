import os
import sys
from src.entity.config_entity import DataIngestionConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml
from pathlib import Path
import pandas as pd



class DataIngestion:
    def __init__(self):
        # Read config yaml
        self.ingestion_config = DataIngestionConfig(

        )

    def initiate_data_ingestion(self):
        logging.info("Starting Data Ingestion")
        try:
            # Make artifacts folder
            # Download file and store into artifacts folder
            pass
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    bob = read_yaml(Path("config/config.yaml"))
    print(bob)
