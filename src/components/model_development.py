import os
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import save_obj

from sklearn.linear_model import LinearRegression,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    """Configuration for model trainer."""
    model_trainer_file_path : str = os.path.join('artifacts', 'model_trainer.pkl')
    logging.info('model trainer config created')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Random_forest": RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15),
                'xgboost': XGBRegressor(n_estimators=83,learning_rate=0.2111),
                'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=500),
                'AdaBoostRegressor': AdaBoostRegressor(n_estimators=15,learning_rate=1.0),
                'DecisionTree': DecisionTreeRegressor(max_depth= 10)
            }

            #Model_training
            model_report = {}
            for i in range(len(models)):
                model = list(models.values())[i]
                model.fit(X_train,y_train)
                Y_pred = model.predict(X_test)
                testing_score = r2_score(y_test, Y_pred)

                model_report[list(models.keys())[i]] = testing_score
            
            logging.info('model training complete')

            # Best model score
            best_model_score = max(sorted(model_report.values())) # eg = 88

            #Best model name - gives the key of dictionary
            model_keys_list = list(model_report.keys())
            model_values_list = list(model_report.values())
            index_of_best_model = model_values_list.index(best_model_score) # lets say 2
            best_model_name = model_keys_list[index_of_best_model]

             # best model - gives the value of dictionary
            best_model = models[best_model_name]
            logging.info('Best model name declared')

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            save_obj(
                file_path = self.model_trainer_config.model_trainer_file_path,
                obj = best_model
            )
            logging.info('best model saved as a object')

            #Prediction of the best model
            ypred = best_model.predict(X_test)
            score = r2_score(y_test, ypred)
            logging.info('best model name and score printed')
            return best_model_name, score

        except Exception as e:
            raise CustomException(e,sys)

