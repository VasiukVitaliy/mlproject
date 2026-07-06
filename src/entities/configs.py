from dataclasses import dataclass
from typing import Literal
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

@dataclass
class ModelTrainingConfig:
    model_path: Path
    models: Literal["AdaBoostRegressor","GradientBoostingRegressor","RandomForestRegressor",
                    "CatBoostRegressor","LinearRegressor", "DecisionTreeRegressor", "XGBRegressor"]
    params: ConfigBox
    report_path: Path
    