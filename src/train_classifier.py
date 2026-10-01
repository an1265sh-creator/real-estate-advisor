import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from src.config import (
    PROCESSED_CSV_PATH, MODELS_DIR, REPORTS_DIR,
    RANDOM_STATE, TEST_SIZE, TARGET_CLASSIFICATION,
    FEATURE_NUMERIC, FEATURE_CATEGORICAL, FEATURE_SPATIAL,
    FORBIDDEN_LEAKAGE_FEATURES
)
from src.preprocessor import PreprocessorBuilder
from src.mlflow_tracker import MLflowTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ClassifierTrainer:
    """Trains and benchmarks sanitized investment classification surrogate models with zero leakage."""

    def __init__(self, data_path=PROCESSED_CSV_PATH):
        self.data_path = Path(data_path)
        self.models_dir = Path(MODELS_DIR)
        self.reports_dir = Path(REPORTS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_tracker = MLflowTracker()

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Loads data, strictly enforces zero leakage, and creates stratified train-test splits."""
        logger.info(f"Loading data for classification from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        feature_cols = FEATURE_NUMERIC + FEATURE_CATEGORICAL + FEATURE_SPATIAL
        
        # Hard assertion: Guarantee zero target leakage in feature matrix
        for forbidden in FORBIDDEN_LEAKAGE_FEATURES:
            assert forbidden not in feature_cols, f"CRITICAL LEAKAGE: {forbidden} found in feature matrix!"

        X = df[feature_cols].copy()
        y = df[TARGET_CLASSIFICATION].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logger.info(f"Sanitized Classification splits: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates pipeline predictions across classification metrics."""
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        return {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm
        }

    def train_and_compare(self) -> Dict[str, Any]:
        """Trains Logistic Regression Baseline, Random Forest, and LightGBM models."""
        X_train, X_test, y_train, y_test = self.prepare_data()
        prep_builder = PreprocessorBuilder()

        candidate_models = {
            "Logistic_Regression_Baseline": {
                "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                "params": {"max_iter": 1000, "solver": "lbfgs"}
            },
            "Random_Forest_Classifier": {
                "estimator": RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=20,
                    random_state=RANDOM_STATE, n_jobs=-1
                ),
                "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 20}
            },
            "LightGBM_Classifier": {
                "estimator": LGBMClassifier(
                    n_estimators=150, max_depth=6, learning_rate=0.08,
                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
                ),
                "params": {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.08}
            }
        }

        results = {}
        best_model_name = None
        best_f1 = -1.0
        best_pipeline = None

        for name, config in candidate_models.items():
            logger.info(f"Training sanitized classification candidate: {name}...")
            preprocessor = prep_builder.build_transformer()
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', config['estimator'])
            ])

            pipeline.fit(X_train, y_train)
            metrics = self.evaluate_model(pipeline, X_test, y_test)
            logger.info(f"Results for {name}: F1={metrics['f1_score']}, AUC={metrics['roc_auc']}, Acc={metrics['accuracy']}")

            results[name] = {
                "params": config['params'],
                "metrics": metrics
            }

            self.mlflow_tracker.log_run(
                run_name=f"Sanitized_Classification_{name}",
                parameters=config['params'],
                metrics={k: v for k, v in metrics.items() if k != "confusion_matrix"},
                model=pipeline,
                artifact_path="classification_model",
                tags={"model_type": "classification", "candidate": name, "leakage_free": "true"}
            )

            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_model_name = name
                best_pipeline = pipeline

        # Save champion model pipeline
        best_model_path = self.models_dir / "classification_model.joblib"
        logger.info(f"Persisting champion classification model ({best_model_name}) to {best_model_path}...")
        joblib.dump(best_pipeline, best_model_path)

        # Feature importance
        fitted_prep = best_pipeline.named_steps['preprocessor']
        feature_names = prep_builder.get_feature_names(fitted_prep)

        feature_importance_dict = {}
        clf_step = best_pipeline.named_steps['classifier']
        if hasattr(clf_step, "feature_importances_"):
            importances = clf_step.feature_importances_
            feature_importance_dict = {
                name: round(float(imp), 5)
                for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:25]
            }
        elif hasattr(clf_step, "coef_"):
            importances = np.abs(clf_step.coef_[0])
            feature_importance_dict = {
                name: round(float(imp), 5)
                for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:25]
            }

        metadata = {
            "best_model_name": best_model_name,
            "target": TARGET_CLASSIFICATION,
            "comparison": results,
            "feature_names": feature_names,
            "top_feature_importances": feature_importance_dict,
            "numeric_features": FEATURE_NUMERIC,
            "categorical_features": FEATURE_CATEGORICAL,
            "spatial_features": FEATURE_SPATIAL
        }

        with open(self.reports_dir / "classification_results.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(self.models_dir / "classification_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Classification champion: {best_model_name} with F1-Score: {best_f1:.4f}")
        return metadata

if __name__ == "__main__":
    trainer = ClassifierTrainer()
    res = trainer.train_and_compare()
    print("Sanitized classification training complete.")
