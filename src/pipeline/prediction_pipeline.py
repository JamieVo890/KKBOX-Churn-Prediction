import sys
import pandas as pd
from src.exception import CustomException
import joblib
from src.components.data_transform import DataTransform

class PredictPipeline:
    def __init__(self, users, members, transactions):
        self.users_df = pd.read_csv(users)
        self.members_df = pd.read_csv(members)
        self.transactions_df = pd.read_csv(transactions)
        self.model = joblib.load('artifacts/model_train/model.pkl')
        self.transformer = DataTransform(training=False)
        pass
   
    def transform(self):
        user_logs_transformed_df = self.transformer.transform_user_logs_data(self.users_df)
        members_transformed_df = self.transformer.transform_members_data(self.members_df)
        transactions_transformed_df = self.transformer.transform_transaction_data(self.transactions_df)
        
        final_dataset = user_logs_transformed_df.merge(members_transformed_df, on='msno', how='inner')
        final_dataset = final_dataset.merge(transactions_transformed_df, on='msno', how='inner')
        final_dataset = final_dataset.drop("msno", axis=1)
        final_dataset = final_dataset[~final_dataset["payment_method_id"].isin([25,2])]

        categorical_var = ["payment_method_id", "city", "registered_via"]
        numerical_var = final_dataset.columns.difference(categorical_var)
        preprocessor = self.transformer.get_preprocessor(categorical_var, numerical_var)
        final_dataset = preprocessor.fit_transform(final_dataset).toarray()
        
        return final_dataset

    def predict(self):
        try:
            final_dataset = self.transform()
            predictions = self.model.predict(final_dataset)
            return predictions
        
        except Exception as e:
            raise CustomException(e,sys)
        
#print("test")