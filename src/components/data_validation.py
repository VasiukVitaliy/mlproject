from src.entities.configs import DataValidationConfig
from src.entities.artifact import DataValidationArtifact
from src.exeptions import CustomException
from src.logger import logging
from src.utils.utils import read_yaml
from scipy.stats import ks_2samp, chi2_contingency
import sys
import pandas as pd

class DataValidation:
    def __init__(self, config, artifact):
        self.config = config
        self.artifact = artifact
        
    def check_data_drift(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
        drift_report = {}
        
        try:
            
            numerical_cols = [col for col in train_data.columns if pd.api.types.is_numeric_dtype(train_data[col])]
            categorical_cols = [col for col in train_data.columns if not pd.api.types.is_numeric_dtype(train_data[col])]
            
            for col in numerical_cols:
                res = ks_2samp(train_data[col].to_list(), test_data[col].to_list())
                drift_report[col] = {
                    "type": "numerical",
                    "test": "ks_2samp",
                    "p_value": float(res.pvalue),
                    "drift_detected": bool(res.pvalue < 0.05)
                }

            for col in categorical_cols:
                train_counts = train_data[col].value_counts()
                test_counts = test_data[col].value_counts()

                contingency_table = pd.DataFrame({
                    "train": train_counts,
                    "test": test_counts
                }).fillna(0)

                _, p_value, _, _ = chi2_contingency(contingency_table)

                drift_report[col] = {
                "type": "categorical",
                "test": "chi2_contingency",
                "p_value": float(p_value),
                "drift_detected": bool(p_value < 0.05)
                }

            return drift_report
        except Exception as e:
            raise CustomException(e, sys)
    
    def check_data_types(self, data_schema, data):
        try:
            types = {col: str(dtype) for col, dtype in data.dtypes.items()}
            type_test = data_schema == types
            return type_test
        except Exception as e:
            raise CustomException(e, sys)
    
    def check_missing_data(self, data, missing_threshold):
        try:
            n = len(data)
            missing_n = data.isna().sum().sum()
            missing_ratio =  missing_n / (n * data.shape[1])
            return missing_ratio <= missing_threshold
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def start_validation(self) -> DataValidationArtifact:
        try:
            train_data = pd.read_csv(self.artifact.train_path)
            test_data = pd.read_csv(self.artifact.test_path)

            schema = read_yaml(self.config.schema_path)
            data_schema = schema.columns

            train_type_test = self.check_data_types(data_schema, train_data)
            test_type_test = self.check_data_types(data_schema, test_data)

            missing_threshold = self.config.missing_threshold

            train_missing_test = self.check_missing_data(train_data, missing_threshold)
            test_missing_test = self.check_missing_data(test_data, missing_threshold)


            drift_report = self.check_data_drift(train_data, test_data)
            drift_detected = any(result["drift_detected"] for result in drift_report.values())

            report_df = pd.DataFrame(drift_report)
            report_df.to_html(self.config.data_drift_report)

            train_valid = train_type_test and train_missing_test
            test_valid = test_type_test and test_missing_test
            validation_status = train_valid and test_valid and not drift_detected

            logging.info(f"Train type match: {train_type_test}, missing ratio ok: {train_missing_test}")
            logging.info(f"Test type match: {test_type_test}, missing ratio ok: {test_missing_test}")
            logging.info(f"Drift detected: {drift_detected}")
            logging.info(f"Overall validation status: {validation_status}")

            self.config.valid_train_data_path.mkdir(exist_ok = True)
            self.config.invalid_train_data_path.mkdir(exist_ok = True)
            self.config.valid_test_data_path.mkdir(exist_ok = True)
            self.config.invalid_test_data_path.mkdir(exist_ok = True)

            if train_valid:
                train_data.to_csv(self.config.valid_train_data_path / "train_data.csv", index=False, header=True)
            else:
                train_data.to_csv(self.config.invalid_train_data_path  / "train_data.csv", index=False, header=True)

            if test_valid:
                test_data.to_csv(self.config.valid_test_data_path / "test_data.csv", index=False, header=True)
            else:
                test_data.to_csv(self.config.invalid_test_data_path / "test_data.csv", index=False, header=True)

            return DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.config.valid_train_data_path / "train_data.csv" if train_valid else None,
                valid_test_file_path=self.config.valid_test_data_path / "test_data.csv" if test_valid else None,
                invalid_train_file_path=None if train_valid else self.config.invalid_train_data_path  / "train_data.csv",
                invalid_test_file_path=None if test_valid else self.config.invalid_test_data_path / "test_data.csv",
                drift_report_file_path=self.config.data_drift_report
            )
        except Exception as e:
            raise CustomException(e, sys)