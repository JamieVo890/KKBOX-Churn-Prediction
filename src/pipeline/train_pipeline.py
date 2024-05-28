from src.components.data_ingestion import DataIngestion
from src.components.data_transform import DataTransform
from src.components.model_trainer import ModelTraining
from src.logger import logging
from src.exception import CustomException
import sys

class TrainPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            data_ingestion  = DataIngestion()
            data_transform = DataTransform()
            model_trainer = ModelTraining()

            logging.info("Beginning Training Pipeline")
            data_ingestion.initiate_data_ingestion()
            data_transform.initiate_data_transform()
            model_trainer.initiate_model_training()
            logging.info("Successfully completed Training Pipeline")

        except Exception as e:
            logging.info("Training Pipeline Failed")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    bob = TrainPipeline()
    bob.main()
