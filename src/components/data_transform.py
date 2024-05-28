import sys
from src.entity.config_entity import DataTransformConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml, create_directories
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

class DataTransform:
    def __init__(self):
        config = read_yaml(Path("config/config.yaml"))
        self.transform_config = DataTransformConfig(
            root_dir = config["data_transform"]["root_dir"],
            raw_members_data_path = config["data_transform"]["raw_members_data_path"],
            raw_transactions_data_path = config["data_transform"]["raw_transactions_data_path"],
            raw_user_logs_data_path = config["data_transform"]["raw_user_logs_data_path"],
            raw_train_data_path = config["data_transform"]["raw_train_data_path"],
            final_X_train_data_path = config["data_transform"]["final_X_train_data_path"],
            final_X_test_data_path = config["data_transform"]["final_X_test_data_path"],
            y_train_data_path = config["data_transform"]["y_train_data_path"],
            y_test_data_path = config["data_transform"]["y_test_data_path"]
        )

    def load_data(self):
        logging.info("Reading data")
        members_df = pd.read_csv(self.transform_config.raw_members_data_path)
        transactions_df = pd.read_csv(self.transform_config.raw_transactions_data_path)
        user_logs_df = pd.read_csv(self.transform_config.raw_user_logs_data_path)
        train_df = pd.read_csv(self.transform_config.raw_train_data_path)
        return members_df, transactions_df, user_logs_df, train_df
    
    def transform_members_data(self, members_df):
        logging.info("Transforming members dataset.")
        members_df_drop = members_df.drop(["bd","gender","registration_init_time"], axis=1)
        return members_df_drop
    
    def transform_user_logs_data(self, user_logs_df):
        logging.info("Transforming user_logs dataset")
        user_logs_df_summed = user_logs_df.groupby("msno").sum()
        user_logs_df_summed.reset_index(inplace=True)
        return user_logs_df_summed
        
    def transform_transaction_data(self, transactions_df):
        logging.info("Transforming transactions dataset")
        latest_transactions = transactions_df.copy()
        latest_transactions = latest_transactions.sort_values('transaction_date').drop_duplicates(['msno'], keep='last')


        prev_transactions = transactions_df.copy()
        mask = prev_transactions.apply(tuple, axis=1).isin(latest_transactions.apply(tuple, axis=1))
        prev_transactions = prev_transactions[~mask]

        num_transactions = prev_transactions.groupby('msno').size().reset_index(name='num_prev_transactions')
        total_prev_paid = prev_transactions.groupby('msno')['actual_amount_paid'].sum().reset_index(name='total_prev_paid')
        total_prev_cancelled = prev_transactions.groupby('msno')['is_cancel'].sum().reset_index(name='total_prev_cancelled')
        num_prev_discounts = prev_transactions[prev_transactions['plan_list_price'] > prev_transactions['actual_amount_paid']].groupby('msno').size().reset_index(name='num_prev_discounts')

        # Add new features
        latest_transactions = latest_transactions.merge(num_transactions, on='msno', how='left')
        latest_transactions = latest_transactions.merge(total_prev_paid, on='msno', how='left')
        latest_transactions = latest_transactions.merge(num_prev_discounts, on='msno', how='left')
        latest_transactions = latest_transactions.merge(total_prev_cancelled, on='msno', how='left')

        # Filling NA values with 0 (for users with no previous transactions)
        latest_transactions['num_prev_transactions'] = latest_transactions['num_prev_transactions'].fillna(0)
        latest_transactions['total_prev_paid'] = latest_transactions['total_prev_paid'].fillna(0)
        latest_transactions['num_prev_discounts'] = latest_transactions['num_prev_discounts'].fillna(0)
        latest_transactions['total_prev_cancelled'] = latest_transactions['total_prev_cancelled'].fillna(0)

        latest_transactions["curr_discount"] = (latest_transactions['plan_list_price'] > latest_transactions['actual_amount_paid']).astype(int)
        latest_transactions = latest_transactions.drop('membership_expire_date',axis=1)
        latest_transactions = latest_transactions.drop('transaction_date',axis=1)

        return latest_transactions


    def merge_datasets(self, members_df, user_logs_df, transactions_df, train_df):
        logging.info("Merging datasets")
        final_dataset = train_df.merge(transactions_df, on='msno', how='inner')
        final_dataset = final_dataset.merge(user_logs_df, on='msno', how='inner')
        final_dataset = final_dataset.merge(members_df, on='msno', how='inner')
        final_dataset = final_dataset.drop("msno", axis=1)

        return final_dataset
    
    def get_preprocessor(self, categorical_var, numerical_var):
        logging.info("Obtaining preprocessor")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_var),
                ('cat', OneHotEncoder(drop='first'), categorical_var)
            ],
            remainder='passthrough' 
        )
        return preprocessor

    def initiate_data_transform(self):
        logging.info("Starting data transform")
        try:

            create_directories([self.transform_config.root_dir])

            members_df, transactions_df, user_logs_df, train_df = self.load_data()
            members_transformed_df = self.transform_members_data(members_df)
            user_logs_transformed_df = self.transform_user_logs_data(user_logs_df)
            transactions_transformed_df = self.transform_transaction_data(transactions_df)
            merged_dataset = self.merge_datasets(members_transformed_df, user_logs_transformed_df, transactions_transformed_df, train_df)

            X = merged_dataset.drop("is_churn", axis=1)
            y = merged_dataset["is_churn"]


            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

            categorical_var = ["payment_method_id", "city", "registered_via"]
            numerical_var = X_train.columns.difference(categorical_var)
            preprocessor = self.get_preprocessor(categorical_var, numerical_var)

            logging.info("Transforming train set")
            X_train_final = preprocessor.fit_transform(X_train).toarray()
            X_test_final = preprocessor.transform(X_test).toarray()

            np.save(self.transform_config.final_X_train_data_path, X_train_final)
            np.save(self.transform_config.final_X_test_data_path, X_test_final)
            np.save(self.transform_config.y_train_data_path, y_train)
            np.save(self.transform_config.y_test_data_path, y_test)


        except Exception as e:
            logging.info("Data transform failed")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    bob = DataTransform()
    bob.initiate_data_transform()
