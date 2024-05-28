import os
import zipfile
import sys
from src.entity.config_entity import DataIngestionConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml, create_directories
from pathlib import Path
import pandas as pd



class DataIngestion:
    def __init__(self):
        config = read_yaml(Path("config/config.yaml"))
        self.ingestion_config = DataIngestionConfig(
            root_dir = config["data_ingestion"]["root_dir"],
            source_files_path = config["data_ingestion"]["source_files_path"]
        )

    def initiate_data_ingestion(self):
        logging.info("Starting Data Ingestion")
        try:
            
            create_directories([self.ingestion_config.root_dir])
            
            # For simplicity, we just source the files from a local folder
            unzip_dir = self.ingestion_config.root_dir
            with zipfile.ZipFile(self.ingestion_config.source_files_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)
            logging.info(f"Extracted source files to {unzip_dir} successfully")

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    bob = DataIngestion()
    bob.initiate_data_ingestion()
