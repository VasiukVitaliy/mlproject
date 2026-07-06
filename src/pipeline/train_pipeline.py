from src.components.data_ingestions import DataIngestions
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformations
from src.components.model_training import ModelTraining
from src.config.manager import ConfigManager
from pathlib import Path
import shutil

if __name__ == "__main__":
    
    manager = ConfigManager(Path('configs\config.yaml'), Path('configs\params.yaml'))
    ingestion_config = manager.get_ingestion_config()
    ingestion_step = DataIngestions(ingestion_config, manager.params)
    ingestion_artifact = ingestion_step.start_ingestion()
    
    validation_config = manager.get_validation_config()
    validation_step = DataValidation(validation_config, ingestion_artifact)
    validation_artifact = validation_step.start_validation()
    
    transformation_config = manager.get_transformation_config()
    transformation_step = DataTransformations(transformation_config, validation_artifact)
    transformation_artifact = transformation_step.start_transformation()
    
    training_config = manager.get_training_config()
    training_step = ModelTraining(training_config, transformation_artifact)
    training_artifact = training_step.test_models()
    
    latest_model_version_path = Path.cwd() / "latest"
    latest_model_version_path.mkdir(exist_ok=True)
    
    model_filename = training_artifact.model_path.name
    preprocessor_filename = transformation_artifact.transformer.name
    
    shutil.copy2(training_artifact.model_path, latest_model_version_path / model_filename)
    shutil.copy2(transformation_artifact.transformer, latest_model_version_path / preprocessor_filename)
    
    