import zipfile
import sys
from src.entity.config_entity import EvaluationConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml, create_directories
from pathlib import Path
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

class EvaluationPipeline:
    def __init__(self):
        config = read_yaml(Path("config/config.yaml"))
        self.evaluation_config = EvaluationConfig(
            root_dir = config["evaluation"]["root_dir"],
            final_model_path = config["evaluation"]["final_model_path"],
            evaluation_X_path = config["evaluation"]["evaluation_X_path"],
            evaluation_y_path = config["evaluation"]["evaluation_y_path"],
            accuracy_txt_path = config["evaluation"]["accuracy_txt_path"]
        )

    def initiate_evaluation(self):
        logging.info("Starting evaluation")
        try:
            
            create_directories([self.evaluation_config.root_dir])
            
            # Just for simplicity, we use the test data for our scheduled evaluation of our model during the deployment
            X_test = np.load(self.evaluation_config.evaluation_X_path)
            y_test = np.load(self.evaluation_config.evaluation_y_path)

            with open(self.evaluation_config.final_model_path, 'rb') as file:
                model = pickle.load(file)

            
            # Load model 
            # Evaluate final model on test set
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test,preds)

            if accuracy < 100:
                logging.info(f"Model evaluation set accuracy: {accuracy} \n Retraining required")
            else:
                logging.info(f"Model evaluation set accuracy: {accuracy}")
            
            with open(self.evaluation_config.accuracy_txt_path, 'w') as f:
                    f.write(str(accuracy))

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    bob = EvaluationPipeline()
    bob.initiate_evaluation()
