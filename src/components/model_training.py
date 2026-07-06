from src.entities.artifact import ModelTrainingArtifact
from src.utils.utils import load_npy, save_model
from src.exeptions import CustomException
from src.logger import logging
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import optuna

import sys

MODEL_REGISTRY = {
    "AdaBoostRegressor": AdaBoostRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "LinearRegressor": LinearRegression,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "XGBRegressor": XGBRegressor,
    "CatBoostRegressor": CatBoostRegressor
}

SUGGEST_DISPATCH = {
    "int": lambda trial, name, spec: trial.suggest_int(
        name, spec["val"][0], spec["val"][1],
        step=spec.get("step", 1), log=spec.get("log", False)
    ),
    "float": lambda trial, name, spec: trial.suggest_float(
        name, spec["val"][0], spec["val"][1],
        step=spec.get("step"), log=spec.get("log", False)
    ),
    "categorical": lambda trial, name, spec: trial.suggest_categorical(
        name, spec["val"]
    ),
}


def generate_regression_report(y_true, y_pred, n_features, model_name="model", params=None):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    report = {
        "model": model_name,
        "params": params,
        "r2": r2,
        "adj_r2": adj_r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
    }

    return report


class Tuner:
    """
    Тюнер працює ТІЛЬКИ з X_train/y_train.
    Всередині objective train ще раз ділиться на фолди через KFold —
    X_test/y_test сюди взагалі не передаються і не використовуються.
    """
    def __init__(self, *, X_train, y_train, models, parameters, cv=5, random_state=42):
        self.models = models
        self.params = parameters
        self.X_train = X_train
        self.y_train = y_train.reshape(-1, 1)
        self.cv = cv
        self.random_state = random_state

        self.full_report = []

    def adj_r2(self, true, pred, n_features):
        r2 = r2_score(true, pred)
        n = len(true)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    def objective(self, trial):
        try:
            regressor_name = trial.suggest_categorical("models", self.models)

            params = {}
            for key, spec in self.params[regressor_name].items():
                if spec is None:
                    continue
                params[key] = SUGGEST_DISPATCH[spec["type"]](trial, key, spec)

            model_cls = MODEL_REGISTRY[regressor_name]
            n_features = self.X_train.shape[1]

            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            fold_scores = []

            for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(self.X_train)):
                X_tr, X_val = self.X_train[tr_idx], self.X_train[val_idx]
                y_tr, y_val = self.y_train[tr_idx], self.y_train[val_idx]

                model = model_cls(**params)
                model.fit(X_tr, y_tr)

                y_pred = model.predict(X_val)
                fold_score = self.adj_r2(y_val, y_pred, n_features)
                fold_scores.append(fold_score)

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))

            # Звіт по цьому trial (усереднений по фолдах)
            report = {
                "model": regressor_name,
                "params": params,
                "cv_mean_adj_r2": mean_score,
                "cv_std_adj_r2": std_score,
                "cv_scores": fold_scores,
            }
            self.full_report.append(report)

            return mean_score
        except Exception as e:
            raise CustomException(e, sys)


class ModelTraining:
    def __init__(self, config, artifact):
        self.config = config
        self.artifact = artifact

    def test_models(self):
        try:
            logging.info("Starting model training stage")

            logging.info("Reading train and test data")
            X_train = load_npy(self.artifact.train_folder / self.artifact.input_data_name)
            X_test = load_npy(self.artifact.test_folder / self.artifact.input_data_name)

            y_train = load_npy(self.artifact.train_folder / self.artifact.output_data_name)
            y_test = load_npy(self.artifact.test_folder / self.artifact.output_data_name)
            logging.info(
                f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, "
                f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}"
            )

            # ВАЖЛИВО: X_test/y_test НЕ передаються в Tuner —
            # Optuna їх жодного разу не бачить під час тюнінгу
            tuner = Tuner(
                X_train=X_train, y_train=y_train,
                models=self.config.models, parameters=self.config.params,
                cv=5
            )
            logging.info("Tuner object created (тюнінг через 5-fold CV на train)")

            logging.info(f"Starting Optuna study with models: {self.config.models}")
            study = optuna.create_study(direction="maximize")
            study.optimize(tuner.objective, n_trials=100)
            logging.info(
                f"Optuna study completed. Best CV adj_r2: {study.best_value}"
            )

            # Звіт по всіх trials (крос-валідаційні результати, БЕЗ test)
            cv_report_df = pd.DataFrame(tuner.full_report)
            logging.info(f"CV report:\n{cv_report_df}")

            best_model_info = study.best_params.copy()
            model_name = best_model_info.pop("models")
            logging.info(f"Best model selected (за CV): {model_name}, params: {best_model_info}")

            best_model = MODEL_REGISTRY[model_name](**best_model_info)
            logging.info("Fitting best model on full train data")
            best_model.fit(X_train, y_train)
            logging.info("Best model fitted successfully")

            n_features = X_train.shape[1]
            y_test_pred = best_model.predict(X_test)
            final_report = generate_regression_report(
                y_test, y_test_pred, n_features,
                model_name=model_name, params=best_model_info
            )
            logging.info(f"FINAL (untouched) test report: {final_report}")

            final_report_df = pd.DataFrame([final_report])

            with open(self.config.report_path, "w", encoding="utf-8") as f:
                f.write("<h2>Cross-validation results (all trials, train only)</h2>\n")
                f.write(cv_report_df.to_html(index=False))
                f.write("<h2>Final untouched test evaluation (best model)</h2>\n")
                f.write(final_report_df.to_html(index=False))

            logging.info(f"Saved models evaluating report in {self.config.report_path}")

            logging.info(f"Saving best model to {self.config.model_path}")
            save_model(best_model, self.config.model_path)
            logging.info("Model training stage completed successfully")
            
            artifact = ModelTrainingArtifact(
                model_path=self.config.model_path,
                report_path=self.config.report_path,
            )

            return artifact

        except Exception as e:
            logging.error(f"Error occurred in model training stage: {e}")
            raise CustomException(e, sys)