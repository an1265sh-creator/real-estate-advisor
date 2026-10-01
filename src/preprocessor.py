import logging
from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_NUMERIC,
    FEATURE_CATEGORICAL,
    FEATURE_SPATIAL
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PreprocessorBuilder:
    """Constructs Scikit-learn ColumnTransformer for sanitized property features."""

    def __init__(
        self,
        numeric_features: List[str] = FEATURE_NUMERIC,
        categorical_features: List[str] = FEATURE_CATEGORICAL,
        spatial_features: List[str] = FEATURE_SPATIAL
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.spatial_features = spatial_features

    def build_transformer(self) -> ColumnTransformer:
        """Builds standard ColumnTransformer."""
        num_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])

        cat_cols = self.categorical_features + self.spatial_features
        cat_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self.numeric_features),
                ('cat', cat_pipeline, cat_cols)
            ],
            remainder='drop'
        )
        return preprocessor

    def get_feature_names(self, fitted_preprocessor: ColumnTransformer) -> List[str]:
        """Extracts engineered feature names after one-hot encoding."""
        output_features = list(self.numeric_features)
        cat_encoder = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_cols = self.categorical_features + self.spatial_features
        cat_encoded_names = cat_encoder.get_feature_names_out(cat_cols)
        output_features.extend(list(cat_encoded_names))
        return output_features
