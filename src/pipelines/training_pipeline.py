import sys
import pandas as pd # type: ignore
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from components import data_ingestion, data_transformation, model_development

class TrainingPipeline:
    def __init__(self) -> None:
        pass

    def train(self):
        data_extraction = data_ingestion.DataIngestion()
        raw_data_path = data_extraction.initiate_data_ingestion()

        #Data Transformation
        Transformation = data_transformation.DataTransformation()
        cleaned_df = Transformation.clean_data(data_path = raw_data_path)
        X_train, X_test, y_train, y_test = Transformation.initiate_data_transformation(final_data = cleaned_df)

        #Model Trainer
        model_trainer = model_development.ModelTrainer()
        print(model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test))
