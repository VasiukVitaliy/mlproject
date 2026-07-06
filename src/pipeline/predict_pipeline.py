import sys
import pandas as pd
from pathlib import Path

from src.utils.utils import load_model
from src.exeptions import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self, model_path: Path, preprocessor_path: Path):
        try:
            logging.info("Loading model and preprocessor for inference")
            self.model = load_model(model_path)
            self.preprocessor = load_model(preprocessor_path)
            logging.info("Model and preprocessor loaded successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        data = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score],
        }
        return pd.DataFrame(data)