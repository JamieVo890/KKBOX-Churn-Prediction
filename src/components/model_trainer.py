import sys
from src.entity.config_entity import ModelTrainConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml, create_directories
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import mlflow
import pickle


class ModelTraining:
    def __init__(self):
        config = read_yaml(Path("config/config.yaml"))
        self.train_config = ModelTrainConfig(
            root_dir = config["model_train"]["root_dir"],
            X_train_data_path = config["model_train"]["X_train_data_path"],
            X_test_data_path = config["model_train"]["X_test_data_path"],
            y_train_data_path = config["model_train"]["y_train_data_path"],
            y_test_data_path = config["model_train"]["y_test_data_path"],
            final_model_path = config["model_train"]["final_model_path"]
        )

    def load_data(self):
        X_train = np.load(self.train_config.X_train_data_path)
        y_train = np.load(self.train_config.y_train_data_path)
        X_test = np.load(self.train_config.X_test_data_path)
        y_test = np.load(self.train_config.y_test_data_path)
        return X_train, y_train, X_test, y_test
    
    def cross_validate_models(self, model_name, model, param_grid, X, y):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_log_loss', n_jobs=5)
        grid_search.fit(X, y)

        # Iterate over all parameter combinations and their results
        for i in range(len(grid_search.cv_results_['params'])):
            params = grid_search.cv_results_['params'][i]
            mean_test_score = grid_search.cv_results_['mean_test_score'][i]
            std_test_score = grid_search.cv_results_['std_test_score'][i]

            with mlflow.start_run(run_name=f"{model_name}_run_{i}"):
                mlflow.log_params(params)
                mlflow.log_metric('mean_neg_log_loss', mean_test_score)
                mlflow.log_metric('std_neg_log_loss', std_test_score)

                if params == grid_search.best_params_:
                    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
                    logging.info(f"Logged {model_name} model with best params: {params} and neg_log_loss: {mean_test_score}")

        return grid_search
    
    def train_final_model(self, model, params, X, y):
        final_model = model(**params)
        final_model.fit(X,y)
        return final_model
    
    def initiate_model_training(self):
        logging.info("Starting Model Training")
        try:
            
            create_directories([self.train_config.root_dir])
            
            logging.info("Loading Data")
            X_train, y_train, X_test, y_test = self.load_data()

            # Hyperparameter grids
            logging.info("Reading model training parameters")
            rf_param_grid = read_yaml(Path("params/params.yaml"))["rf_param_grid"]
            xgb_param_grid = read_yaml(Path("params/params.yaml"))["xgb_param_grid"]

            # Experiment name
            experiment_name = "model_comparison_experiment"
            mlflow.set_experiment(experiment_name)

            logging.info("Starting Random Forest cross-validation")
            random_forest_grid = self.cross_validate_models("Random_Forest", RandomForestClassifier(), rf_param_grid, X_train, y_train)
            
            logging.info("Starting XGBoost cross-validation")
            XGBoost_grid = self.cross_validate_models("XGBoost", XGBClassifier(), xgb_param_grid, X_train, y_train)  

            if random_forest_grid.best_score_ > XGBoost_grid.best_score_:
                final_model = self.train_final_model(RandomForestClassifier(), random_forest_grid.best_params_, X_train, y_train)
            else:
                final_model = self.train_final_model(XGBClassifier, XGBoost_grid.best_params_, X_train, y_train)

            # Save the final model to a pickle file
            model_filename = self.train_config.final_model_path
            with open(model_filename, 'wb') as file:
                pickle.dump(final_model, file)

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    bob = ModelTraining()
    bob.initiate_model_training()
