import sys
import pandas as pd # type: ignore
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import load_obj


class CustomData:
    def __init__(self, company, type_name, Ram, weight, touchScreen, ips, screenSize, screenResolution, 
                 cpu_brand, HDD, SSD, GPU_brand, os):
        self.company = company
        self.type_name = type_name
        self.Ram = Ram
        self.weight = weight
        self.touchScreen = touchScreen
        self.ips = ips
        self.screenSize = screenSize
        self.screenResolution = screenResolution
        self.cpu_brand = cpu_brand
        self.HDD = HDD
        self.SSD = SSD
        self.GPU_brand = GPU_brand
        self.os = os

    def get_data_as_df(self):
        try:
            #Create a dictionary to create a df
            custom_data_dictionary = {
                'Company' : [self.company],
                'TypeName': [self.type_name],
                'Ram': [self.Ram],
                'Weight': [self.weight],
                'Touchscreen': [self.touchScreen],
                'Ips': [self.ips],
                'screen_size': [self.screenSize],
                'screen_resolution': [self.screenResolution],
                'Cpu brand': [self.cpu_brand],
                'HDD': [self.HDD],
                'SSD': [self.SSD],
                'Gpu brand': [self.GPU_brand],
                'os': [self.os]
            }
            df = pd.DataFrame(custom_data_dictionary)
            logging.info('Custom Data frame from the input is created')
            return df
        except Exception as e:
            raise CustomException(e,sys)

class PredictionPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            # define the pkl file paths
            model_path = 'artifacts/model_trainer.pkl'
            feature_engineering_path = 'artifacts/feature_engineering.pkl'

            #load the pkl files
            model = load_obj(model_path)
            feature_engineering = load_obj(feature_engineering_path)

            #apply transformation on the featurs
            tranformed_data = feature_engineering.transform(features)

            #perform perdiction
            prediction = model.predict(tranformed_data)

            logging.info('Prediction pipeline created')
            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)
    