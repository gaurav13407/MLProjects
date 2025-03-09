import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  # ✅ Corrected instantiation

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            # Define numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler()),
                ]
            )

            # Define categorical pipeline
            cat_pipeline = Pipeline(  #  Fixed incorrect `-`
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore")),  # ✅ Avoids errors on unseen categories
                    ("Scaler", StandardScaler(with_mean=False)),  # ✅ Avoids issues with sparse matrices
                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            # Define ColumnTransformer
            preprocessing = ColumnTransformer(
                transformers=[  #  Fixed missing comma
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            

            return preprocessing  # Return the transformer object

        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {str(e)}")
            raise CustomException(e, sys)
        
        
    def initate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            
            logging.info("obtaning preproceesing objects")
            
            preprocesing_obj=self.get_data_transformer_object()
            
            target_column_name="math score"
            numerical_columns = ["writing score", "reading score"]
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on traning dataframe and testing dataframe.")
            
            input_feature_train_arr=preprocesing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocesing_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing objects.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocesing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
