from src.entities.artifact import DataIngestionArtifact
from src.exeptions import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import sys


class DataIngestions:
    def __init__(self, config, params):
        try:
            self.config = config
            self.params = params
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_ingestion(self):
        try:
            logging.info("Reading data...")
            data = pd.read_csv(self.config.data_path)
            logging.info("Reading data was successfully. Splitting data...")
        
            train_data, test_data = train_test_split(data, test_size = self.params.test_size, random_state=self.params.random_state)
            logging.info("Splitting data was successfully. Saving data...")
        
            train_data.to_csv(self.config.train_path)
            test_data.to_csv(self.config.test_path)
            logging.info("Saving data was successfully. Creating artifact...")
        
            artifact = DataIngestionArtifact(
                train_path= self.config.train_path,
                test_path= self.config.train_path
                )
            logging.info("Creating artifact was successfully. Data Ingestion done.")
        
            return artifact
        except Exception as e:
            logging.error(f"Error: {e}")
            raise CustomException(e, sys)