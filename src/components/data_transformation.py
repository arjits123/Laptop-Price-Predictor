import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from dataclasses import dataclass
from utils import fetch_processor, memory_column_cleaning, cat_os, save_obj

# ML libraries
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

#Data transformation config
@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation."""
    # Handel missing values, outliers, scale the variables and encode (OneEncoder, StandardScaler, MinMaxscaler)
    feature_engineering_obj_path : str = os.path.join('artifacts', 'feature_engineering.pkl')  
    cleaned_data_obj_path : str = os.path.join('artifacts', 'cleaned_data.csv')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def clean_data(self,data_path):
        try:
            df = pd.read_csv(data_path)
            
            # Handle Ram and weight columns
            df['Ram'] = df['Ram'].str.replace('GB','').astype('int32')
            df['Weight'] = df['Weight'].str.replace('kg','').astype('float64')

            # Separating ScreenResolution in 3 columns - Touchscreen, Ips, Screens size(PPI)
            df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
            df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
            new = df['ScreenResolution'].str.split('x',n=1,expand=True)
            df['X_res'] = new[0]
            df['Y_res'] = new[1]
            df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])
            df['X_res'] = df['X_res'].astype('int')
            df['Y_res'] = df['Y_res'].astype('int')
            df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')
            df.drop(columns=['ScreenResolution', 'Inches','X_res','Y_res'],inplace=True)

            # Handelling Cpu Column
            df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
            df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
            df.drop(columns=['Cpu','Cpu Name'],inplace=True)

            # handelling memory column 
            df = memory_column_cleaning(data=df)
            df.drop(columns=['Memory', 'Hybrid','Flash_Storage'],inplace=True)

            # Handelling Gpu Column
            df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])
            df = df[df['Gpu brand'] != 'ARM']
            df.drop(columns=['Gpu'],inplace=True)

            #Handelling Opsys column 
            df['os'] = df['OpSys'].apply(cat_os)
            df.drop(columns=['OpSys'],inplace=True)

            logging.info('Data cleaning completed')
            
            return df

        except Exception as e:
            raise CustomException(e,sys)
        
    def get_data_tranformation_object(self):
        try:
            categorical_features = [0,1,7,10,11]
            preprocessor = ColumnTransformer(transformers=[
                ('ohe', OneHotEncoder(drop='first'), categorical_features )
            ], remainder= 'passthrough')

            logging.info('Preprocessor created')
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, final_data):
        try:
            df = final_data

            #split the dataset into X and y variables
            X = df.drop(columns = ['Price'], axis = 1)
            y = np.log(df['Price'])
            logging.info('X and y created')

            #train test split
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 43)
            logging.info('Train and test split completed')

            #obtain preprocessing object
            preprocessing_obj = self.get_data_tranformation_object()
            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)
            logging.info('Data transformation completed')

            #save the preprocessing object
            save_obj(
                file_path= self.transformation_config.feature_engineering_obj_path,
                obj = preprocessing_obj
            )
            logging.info('feature_engieering object file saved')

            return(
                X_train,
                X_test,
                y_train,
                y_test
            )

        except Exception as e:
            raise CustomException(e,sys)

        

    

