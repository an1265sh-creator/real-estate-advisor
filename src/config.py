import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = PROJECT_ROOT / "india_housing_prices.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_housing_data.parquet"
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "cleaned_housing_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

# MLflow Settings
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "real_estate_investment_advisor")

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Raw Expected Columns
RAW_NUMERIC_COLS = [
    'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt',
    'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property',
    'Nearby_Schools', 'Nearby_Hospitals'
]

RAW_CATEGORICAL_COLS = [
    'State', 'City', 'Locality', 'Property_Type', 'Furnished_Status',
    'Public_Transport_Accessibility', 'Parking_Space', 'Security',
    'Amenities', 'Facing', 'Owner_Type', 'Availability_Status'
]

# Sanitized Engineered Features for Modeling (Strictly Zero Target Leakage)
FEATURE_NUMERIC = [
    'BHK', 'Size_in_SqFt',
    'Floor_No', 'Total_Floors', 'Age_of_Property',
    'Nearby_Schools', 'Nearby_Hospitals',
    'amenity_count', 'floor_ratio'
]

FEATURE_CATEGORICAL = [
    'Property_Type', 'Furnished_Status', 'Public_Transport_Accessibility',
    'Parking_Space', 'Security', 'Facing', 'Owner_Type', 'Availability_Status'
]

# Spatial features (State, City)
FEATURE_SPATIAL = ['State', 'City']

# Targets
# Regression Target: Current Property Market Valuation (Price_in_Lakhs)
TARGET_REGRESSION = 'Price_in_Lakhs'

# Classification Target: High Investment Potential (MCDA Score >= 70.0)
TARGET_CLASSIFICATION = 'High_Investment_Potential'

# Explicit Leakage Exclusion List (Asserted in unit tests)
FORBIDDEN_LEAKAGE_FEATURES = [
    'Price_per_SqFt', 'infra_score', 'investment_score_raw',
    'Price_After_5_Years', 'annual_growth_rate', 'Price_Growth_Pct'
]

# Ensure directories exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
