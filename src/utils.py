import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

def memory_column_cleaning(data):
    data['Memory'] = data['Memory'].astype(str).replace('\.0', '', regex=True)
    data["Memory"] = data["Memory"].str.replace('GB', '')
    data["Memory"] = data["Memory"].str.replace('TB', '000')
    new = data["Memory"].str.split("+", n = 1, expand = True)

    data["first"]= new[0]
    data["first"]=data["first"].str.strip()
    data["second"]= new[1]

    data["Layer1HDD"] = data["first"].apply(lambda x: 1 if "HDD" in x else 0)
    data["Layer1SSD"] = data["first"].apply(lambda x: 1 if "SSD" in x else 0)
    data["Layer1Hybrid"] = data["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
    data["Layer1Flash_Storage"] = data["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

    data['first'] = data['first'].str.extract(r'(\d+)').astype(int) # regular expression applied

    data["second"] = data["second"].fillna("0")
    data["Layer2HDD"] = data["second"].apply(lambda x: 1 if "HDD" in x else 0)
    data["Layer2SSD"] = data["second"].apply(lambda x: 1 if "SSD" in x else 0)
    data["Layer2Hybrid"] = data["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
    data["Layer2Flash_Storage"] = data["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

    data['second'] = data['second'].str.extract(r'(\d+)').astype(int) # regular expression applied
 
    data["HDD"]=(data["first"]*data["Layer1HDD"]+data["second"]*data["Layer2HDD"])
    data["SSD"]=(data["first"]*data["Layer1SSD"]+data["second"]*data["Layer2SSD"])
    data["Hybrid"]=(data["first"]*data["Layer1Hybrid"]+data["second"]*data["Layer2Hybrid"])
    data["Flash_Storage"]=(data["first"]*data["Layer1Flash_Storage"]+data["second"]*data["Layer2Flash_Storage"])

    data.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
           'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
           'Layer2Flash_Storage'],inplace=True)
    
    return data

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
    

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e,sys)

def load_obj(file_path):
    try:
        with open(file_path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)















# def model_training(X_train, y_train, X_test, y_test, models):
#     try:
#         report = {}
#         for i in range(0,len(models)):
#             print(i)
#             model = list(models.values())[i]
#             print(model)

#             #train the model
#             model.fit(X_train, y_train)
#             #make prediction
#             y_pred = model.predict(X_test)
#             testing_score = r2_score(y_test, y_pred)

#             #for every model testing score
#             report[list(models.keys())[i]] = testing_score

#             return report
#     except Exception as e:
#         raise CustomException(e,sys)