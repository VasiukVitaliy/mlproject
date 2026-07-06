from src.entities.configs import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainingConfig
from src.exeptions import CustomException
from src.utils.utils import read_yaml
from src.logger import logging
from pathlib import Path
from datetime import datetime
import sys

class ConfigManager:
    def __init__(self, config_path, params_path):
        try:
            logging.info("Reading configs...")
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)
            logging.info("Reading was successfully. Creating artifact folder...")
        
            general_config = self.config.general_config
            vers = datetime.now().strftime("%m_%d-%H-%M-%S")

            self.artifact_path = Path.cwd() / general_config.artifact_folder / vers
            self.artifact_path.mkdir(parents=True, exist_ok=True)
            logging.info("Creating artifact folder was successfully")
        except Exception as e:
            logging.error(f"Error: {e}")
            raise CustomException(e, sys)
        

    def get_ingestion_config(self)->DataIngestionConfig:
        try:
            ingestion_path = self.artifact_path /self.config.data_ingestion.folder_name
            ingestion_path.mkdir(exist_ok=True)
            config = DataIngestionConfig(
            data_path= self.config.data_ingestion.data_path,
            train_path= ingestion_path / self.config.data_ingestion.train_path,
            test_path= ingestion_path / self.config.data_ingestion.test_path
            )
            return config
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_validation_config(self)->DataValidationConfig:
        try:
            data_validation = self.config.data_validation
            validation_path = self.artifact_path / data_validation.folder_name
            validation_path.mkdir(exist_ok=True)
            
            config = DataValidationConfig(
                valid_train_data_path= validation_path / data_validation.valid_train_data_path,
                invalid_train_data_path= validation_path / data_validation.invalid_train_data_path,
                valid_test_data_path= validation_path / data_validation.valid_test_data_path,
                invalid_test_data_path= validation_path / data_validation.invalid_test_data_path,
                data_drift_report= validation_path / data_validation.data_drift_report,
                schema_path= Path(data_validation.schema_path),
                missing_threshold= self.params.missing_threshold,
                drift_p_value_threshold=self.params.pvalue_threshold
                
            )
            return config
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_transformation_config(self)->DataTransformationConfig:
        try:
            data_transformation = self.config.data_transformation
            tranformation_path = self.artifact_path / data_transformation.folder_name
            tranformation_path.mkdir(exist_ok = True)
            config = DataTransformationConfig(
                transformer= tranformation_path / data_transformation.transformer,
                train_folder= Path.cwd() / data_transformation.folder_name / data_transformation.train_folder,
                test_folder= Path.cwd() / data_transformation.folder_name / data_transformation.test_folder,
                input_data_name=data_transformation.input_file_name,
                output_data_name=data_transformation.output_file_name
            )
            return config
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_training_config(self)->ModelTrainingConfig:
        try:
            model_training = self.config.model_training
            training_path = self.artifact_path / model_training.folder_name
            training_path.mkdir(exist_ok = True)
            config = ModelTrainingConfig(
                model_path= training_path / model_training.model_filename,
                models= self.params.models,
                params=self.params.model_params,
                report_path=  training_path / model_training.report_filename
            )
            return config
        except Exception as e:
            raise CustomException(e, sys)