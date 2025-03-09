import os
import sys
import numpy as np
import pandas as pd
import pandas as pd
from src.exception import CustomException 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import dill

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb")as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
import numpy as np

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        # 🚀 Ensure y_train and y_test have correct shape (reshape if needed)
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train = y_train.ravel()  # Convert (N, 1) or (N, C) to (N,)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = y_test.ravel()

        print(f"Fixed Shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Fixed Shapes -> X_test: {X_test.shape}, y_test: {y_test.shape}")

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[name] = test_model_score  # Save test accuracy
            
        return report

    except Exception as e:
        raise CustomException(e, sys)

        
    except Exception as e:
        raise CustomException(e,sys)