from src.components.data_ingestions import DataIngestions
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformations
from src.config.manager import ConfigManager
from pathlib import Path
if __name__ == "__main__":
    
    manager = ConfigManager(Path('configs\config.yaml'), Path('configs\params.yaml'))
    ingestion_config = manager.get_ingestion_config()
    ingestion_step = DataIngestions(ingestion_config, manager.params)
    ingestion_artefact = ingestion_step.start_ingestion()
    
    validation_config = manager.get_validation_config()
    validation_step = DataValidation(validation_config, ingestion_artefact)
    validation_artefact = validation_step.start_validation()
    
    transformation_config = manager.get_transformation_config()
    transformation_step = DataTransformations(transformation_config, validation_artefact)
    transformation_artefact = transformation_step.start_transformation()