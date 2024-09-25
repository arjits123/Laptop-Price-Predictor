import sys
import pandas as pd # type: ignore
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from components import data_ingestion, data_transformation, model_development

if __name__ == '__main__':
    # Data Ingestion
    ingesiton = data_ingestion.DataIngestion()
    raw_data_path = ingesiton.initiate_data_ingestion()

    #Data Transformation
    transformation = data_transformation.DataTransformation()
    cleaned_df = transformation.clean_data(data_path = raw_data_path)
    X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(final_data = cleaned_df)

    #Model trainer 
    model_trainer = model_development.ModelTrainer()
    print(model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test))
