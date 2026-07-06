from src.entities.configs import DataTransformationConfig
from src.entities.artifact import DataTransformationArtifact
from src.logger import logging
from src.exeptions import CustomException
from src.utils.utils import write_npy, save_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import sys
import pandas as pd
import numpy as np



class DataTransformations:
    def __init__(self, config, artifact):
        self.config = config
        self.artifact = artifact

    def start_transformation(self):
        try:
            logging.info("Starting data transformation stage")

            validation_status = self.artifact.validation_status
            if not validation_status:
                logging.error("Validation stage failed. Stopping transformation stage")
                raise CustomException("Validation stage was failed. Transformation stage was stopped")

            logging.info("Reading train and test data")
            train_data = pd.read_csv(self.artifact.valid_train_file_path)
            test_data = pd.read_csv(self.artifact.valid_test_file_path)
            logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")



            X_train = train_data.drop(columns=["math_score"])
            y_train = train_data["math_score"]

            X_test = test_data.drop(columns=["math_score"])
            y_test = test_data["math_score"]
            
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()

            num_features = [col for col in X_train.columns if pd.api.types.is_numeric_dtype(X_train[col])]
            cat_features = [col for col in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[col])]
            logging.info(f"Numeric features: {num_features}")
            logging.info(f"Categorical features: {cat_features}")

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, cat_features),
                    ("StandardScaler", numeric_transformer, num_features),
                ]
            )
            logging.info("Preprocessor object (ColumnTransformer) created")

            logging.info("Applying preprocessing object on train and test data")
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
            logging.info(f"Transformed X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

            self.config.train_folder.mkdir(parents= True, exist_ok=True)
            self.config.test_folder.mkdir(parents= True, exist_ok=True)
            logging.info("Train and test output folders created (or already exist)")

            logging.info("Saving transformed arrays to .npy files")
            write_npy(X_train, self.config.train_folder / self.config.input_data_name)
            write_npy(X_test, self.config.test_folder / self.config.input_data_name)
            write_npy(y_train, self.config.train_folder / self.config.output_data_name)
            write_npy(y_test, self.config.test_folder / self.config.output_data_name)
            logging.info("All transformed arrays saved successfully")
            
            save_model(preprocessor, self.config.transformer)

            artifact = DataTransformationArtifact(
                transformer=self.config.transformer,
                input_data_name=self.config.input_data_name,
                output_data_name=self.config.output_data_name,
                train_folder=self.config.train_folder,
                test_folder=self.config.test_folder,
            )
            logging.info("Data transformation stage completed successfully")

            return artifact
        except Exception as e:
            logging.error(f"Error occurred in data transformation stage: {e}")
            raise CustomException(e, sys)