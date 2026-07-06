from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    train_path: Path
    test_path: Path
    
@dataclass
class DataValidationArtifact:
        validation_status: bool
        valid_train_file_path: Path
        valid_test_file_path: Path
        invalid_train_file_path: Path
        invalid_test_file_path: Path
        drift_report_file_path: Path
        validation_status: bool       
        
@dataclass
class DataTransformationArtifact:
    transformer: Path
    input_data_name: str
    output_data_name: str
    train_folder: Path
    test_folder: Path
    
@dataclass
class ModelTrainingArtifact:
    model_path: Path
    report_path: Path