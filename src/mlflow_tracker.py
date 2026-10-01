import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow
import mlflow.sklearn

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

class MLflowTracker:
    """Manages MLflow experiment tracking, runs, metric logging, and artifact persistence."""

    def __init__(self, tracking_uri: str = MLFLOW_TRACKING_URI, experiment_name: str = MLFLOW_EXPERIMENT_NAME):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._init_mlflow()

    def _init_mlflow(self):
        """Configures MLflow tracking backend."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow initialized with tracking URI: {mlflow.get_tracking_uri()} and experiment: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow initialization notice: {e}. Attempting fallback...")
            try:
                mlflow.set_tracking_uri(f"sqlite:///{Path('./mlflow.db').resolve().as_posix()}")
                mlflow.set_experiment(self.experiment_name)
            except Exception as e2:
                logger.error(f"Fallback tracking also encountered: {e2}")

    def log_run(
        self,
        run_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        model: Optional[Any] = None,
        artifact_path: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Logs parameters, metrics, tags, and model inside an MLflow run."""
        try:
            with mlflow.start_run(run_name=run_name) as run:
                if tags:
                    mlflow.set_tags(tags)
                if parameters:
                    mlflow.log_params(parameters)
                if metrics:
                    mlflow.log_metrics(metrics)
                if model is not None and artifact_path:
                    try:
                        mlflow.sklearn.log_model(model, artifact_path=artifact_path, serialization_format="cloudpickle")
                    except Exception as me:
                        logger.warning(f"Model artifact logging notice: {me}. Parameters and metrics are safely logged.")
                logger.info(f"MLflow run '{run_name}' logged successfully (Run ID: {run.info.run_id})")
                return run.info.run_id
        except Exception as e:
            logger.error(f"Error logging MLflow run '{run_name}': {e}")
            return None
