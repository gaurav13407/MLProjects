import os
import sys
import logging
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.utlils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            
            if train_array.shape[1] < 2:
                raise CustomException(f"train_array has only {train_array.shape[1]} column(s), expected at least 2.")
            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Ensure y_train and y_test are correctly shaped
            y_train = y_train.ravel() if y_train.ndim > 1 else y_train
            y_test = y_test.ravel() if y_test.ndim > 1 else y_test
            
            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            models = {
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "AdaBoostRegressor": AdaBoostRegressor()
            }
            
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R² score >= 0.6")

            logging.info(f"Best model found: {best_model_name} with R² score: {best_model_score}")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            
            predictions = best_model.predict(X_test)
            r2_square = r2_score(y_test, predictions)
            
            logging.info(f"Final model R² score: {r2_square}")
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)