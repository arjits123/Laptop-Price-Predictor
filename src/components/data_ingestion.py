import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from dataclasses import dataclass

from data_transformation import DataTransformation
from model_development import ModelTrainer

import pandas as pd
import numpy as np

#data ingestion class
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    logging.info('Data ingestion configuration completed')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config =  DataIngestionConfig()

    def initiate_data_ingestion(self):

        try:
            df = pd.read_csv('data/laptop_data.csv')
            df.drop(columns=['Unnamed: 0'],inplace=True)
            logging.info('Imported the dataset from the local drive')

            #making the artifacts directory
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            #Saving the raw csv file
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Data ingestion completed')
            return self.data_ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    
    # Data Ingestion
    data_ingestion = DataIngestion()
    raw_data_path = data_ingestion.initiate_data_ingestion()

    #Data Transformation
    data_transformation = DataTransformation()
    cleaned_df = data_transformation.clean_data(data_path = raw_data_path)
    # print(cleaned_df.head())
    X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(final_data = cleaned_df)
    # print(X_test.shape)

    #Model trainer 
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test))

