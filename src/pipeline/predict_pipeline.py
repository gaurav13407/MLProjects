import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            print("Loading model and preprocessor...")
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            print("Model and preprocessor loaded successfully.")
            
            # Print input columns for debugging
            print("Input Data Columns:", features.columns.tolist())
            print("Input Data Sample:\n", features.head())
            
            # Ensure column names match expected ones
            required_columns = {"gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course", "reading score", "writing score"}
            missing_columns = required_columns - set(features.columns)
            if missing_columns:
                raise ValueError(f"Missing columns in input data: {missing_columns}")
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            print("Generated DataFrame:\n", df)  # Debugging statement
            return df
        except Exception as e:
            raise CustomException(e, sys)