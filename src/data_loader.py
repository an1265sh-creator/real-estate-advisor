import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, RAW_NUMERIC_COLS, RAW_CATEGORICAL_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles raw data ingestion and structural validation for india_housing_prices.csv."""

    def __init__(self, data_path=RAW_DATA_PATH):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Loads dataset from CSV path."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Source dataset not found at {self.data_path}")
        logger.info(f"Loading raw dataset from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validates column presence, datatypes, missing values, duplicates, and key constraints."""
        expected_cols = set(RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS + ['ID'])
        actual_cols = set(df.columns)
        missing_cols = list(expected_cols - actual_cols)
        extra_cols = list(actual_cols - expected_cols)

        null_counts = df.isnull().sum().to_dict()
        total_nulls = sum(null_counts.values())

        total_dups = int(df.duplicated().sum())
        non_id_cols = [c for c in df.columns if c != 'ID']
        dups_excl_id = int(df.duplicated(subset=non_id_cols).sum())

        ref_year_unique = (df['Year_Built'] + df['Age_of_Property']).unique().tolist()
        floor_anomaly_count = int((df['Floor_No'] > df['Total_Floors']).sum())

        validation_report = {
            "shape": df.shape,
            "missing_expected_columns": missing_cols,
            "unexpected_columns": extra_cols,
            "total_nulls": total_nulls,
            "total_duplicates": total_dups,
            "duplicates_excl_id": dups_excl_id,
            "reference_year_invariance": ref_year_unique,
            "floor_no_exceeds_total": floor_anomaly_count,
            "floor_anomaly_percentage": round((floor_anomaly_count / len(df)) * 100, 2)
        }
        return validation_report

    def load_and_validate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = self.load_data()
        report = self.validate_schema(df)
        return df, report

if __name__ == "__main__":
    loader = DataLoader()
    df, report = loader.load_and_validate()
    print("Schema Validation Success:", report["total_nulls"] == 0 and report["total_duplicates"] == 0)
