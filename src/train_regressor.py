import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.config import (
    PROCESSED_CSV_PATH, MODELS_DIR, REPORTS_DIR,
    RANDOM_STATE, TEST_SIZE, TARGET_REGRESSION,
    FEATURE_NUMERIC, FEATURE_CATEGORICAL, FEATURE_SPATIAL,
    FORBIDDEN_LEAKAGE_FEATURES
)
from src.preprocessor import PreprocessorBuilder
from src.mlflow_tracker import MLflowTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class RegressorTrainer:
    """Trains and benchmarks genuine property valuation regression models with zero leakage."""

    def __init__(self, data_path=PROCESSED_CSV_PATH):
        self.data_path = Path(data_path)
        self.models_dir = Path(MODELS_DIR)
        self.reports_dir = Path(REPORTS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_tracker = MLflowTracker()

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Loads data, asserts zero leakage, and creates train-test splits."""
        logger.info(f"Loading data for property valuation regression from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        feature_cols = FEATURE_NUMERIC + FEATURE_CATEGORICAL + FEATURE_SPATIAL

        # Hard assertion: Guarantee zero target leakage in feature matrix
        for forbidden in FORBIDDEN_LEAKAGE_FEATURES:
            assert forbidden not in feature_cols, f"CRITICAL LEAKAGE: {forbidden} found in feature matrix!"

        X = df[feature_cols].copy()
        y = df[TARGET_REGRESSION].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        logger.info(f"Sanitized Regression splits: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates pipeline predictions across regression metrics."""
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2_score": round(float(r2), 4)
        }

    def train_and_compare(self) -> Dict[str, Any]:
        """Trains Ridge Baseline, Random Forest, and LightGBM regressors."""
        X_train, X_test, y_train, y_test = self.prepare_data()
        prep_builder = PreprocessorBuilder()

        candidate_models = {
            "Ridge_Regression_Baseline": {
                "estimator": Ridge(alpha=10.0, random_state=RANDOM_STATE),
                "params": {"alpha": 10.0}
            },
            "Random_Forest_Regressor": {
                "estimator": RandomForestRegressor(
                    n_estimators=100, max_depth=10, min_samples_split=20,
                    random_state=RANDOM_STATE, n_jobs=-1
                ),
                "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 20}
            },
            "LightGBM_Regressor": {
                "estimator": LGBMRegressor(
                    n_estimators=150, max_depth=6, learning_rate=0.08,
                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
                ),
                "params": {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.08}
            }
        }

        results = {}
        best_model_name = None
        best_rmse = float("inf")
        best_pipeline = None

        for name, config in candidate_models.items():
            logger.info(f"Training sanitized regression candidate: {name}...")
            preprocessor = prep_builder.build_transformer()
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', config['estimator'])
            ])

            pipeline.fit(X_train, y_train)
            metrics = self.evaluate_model(pipeline, X_test, y_test)
            logger.info(f"Results for {name}: R2={metrics['r2_score']}, MAE={metrics['mae']}, RMSE={metrics['rmse']}")

            results[name] = {
                "params": config['params'],
                "metrics": metrics
            }

            self.mlflow_tracker.log_run(
                run_name=f"Sanitized_Valuation_{name}",
                parameters=config['params'],
                metrics=metrics,
                model=pipeline,
                artifact_path="regression_model",
                tags={"model_type": "regression_valuation", "candidate": name, "leakage_free": "true"}
            )

            # Pick best model based on lowest test RMSE
            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_model_name = name
                best_pipeline = pipeline

        # Save champion model pipeline
        best_model_path = self.models_dir / "regression_model.joblib"
        logger.info(f"Persisting champion valuation model ({best_model_name}) to {best_model_path}...")
        joblib.dump(best_pipeline, best_model_path)

        # Feature importance
        fitted_prep = best_pipeline.named_steps['preprocessor']
        feature_names = prep_builder.get_feature_names(fitted_prep)

        feature_importance_dict = {}
        reg_step = best_pipeline.named_steps['regressor']
        if hasattr(reg_step, "feature_importances_"):
            importances = reg_step.feature_importances_
            feature_importance_dict = {
                name: round(float(imp), 5)
                for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:25]
            }
        elif hasattr(reg_step, "coef_"):
            importances = np.abs(reg_step.coef_)
            feature_importance_dict = {
                name: round(float(imp), 5)
                for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:25]
            }

        metadata = {
            "best_model_name": best_model_name,
            "target": TARGET_REGRESSION,
            "comparison": results,
            "feature_names": feature_names,
            "top_feature_importances": feature_importance_dict,
            "numeric_features": FEATURE_NUMERIC,
            "categorical_features": FEATURE_CATEGORICAL,
            "spatial_features": FEATURE_SPATIAL
        }

        with open(self.reports_dir / "regression_results.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(self.models_dir / "regression_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Regression champion: {best_model_name} with RMSE: {best_rmse:.4f}")
        return metadata

if __name__ == "__main__":
    trainer = RegressorTrainer()
    res = trainer.train_and_compare()
    print("Sanitized regression training complete.")
