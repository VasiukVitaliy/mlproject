from dataclasses import dataclass
from pathlib import Path
from box import ConfigBox

@dataclass
class DataIngestionConfig:
    data_path: Path
    train_path: Path
    test_path: Path

@dataclass
class DataValidationConfig:
    valid_train_data_path: Path
    invalid_train_data_path: Path
    valid_test_data_path: Path
    invalid_test_data_path: Path
    
    data_drift_report: Path
    schema_path: Path
    
    missing_threshold: float
    drift_p_value_threshold: float
    
@dataclass
class DataTransformationConfig:
    transformer: Path
    input_data_name: str
    output_data_name: str
    train_folder: Path
    test_folder: Path