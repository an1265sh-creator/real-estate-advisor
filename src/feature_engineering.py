import logging
from typing import Tuple, Any, Dict
import pandas as pd
import numpy as np
from src.config import (
    PROCESSED_DATA_PATH, PROCESSED_CSV_PATH,
    TARGET_CLASSIFICATION, TARGET_REGRESSION
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles legitimate domain feature engineering, transparent Multi-Criteria Decision Analysis
    (MCDA) investment potential scoring, and 5-year illustrative scenario compounding.
    Strictly prevents target-leakage features from entering the ML training matrices.
    """

    TRANSPORT_MAP = {'High': 1.0, 'Medium': 0.5, 'Low': 0.0}
    INVESTMENT_THRESHOLD = 70.0  # MCDA High Investment Potential Score threshold (>= 70.0)

    @staticmethod
    def count_amenities(amenity_str: Any) -> int:
        """Parses comma-separated amenities into an integer count."""
        if not isinstance(amenity_str, str) or not amenity_str.strip():
            return 0
        return len([a.strip() for a in amenity_str.split(',') if a.strip()])

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives pure legitimate domain features that are safe for ML modeling.
        Does NOT compute or append target-leaking scores here.
        """
        df = df.copy()

        # 1. Legitimate Amenity Count (0 to 5)
        df['amenity_count'] = df['Amenities'].apply(self.count_amenities)

        # 2. Legitimate Bounded Floor Position Ratio
        clamped_floor = np.minimum(df['Floor_No'], df['Total_Floors'])
        safe_total = np.maximum(df['Total_Floors'], 1)
        df['floor_ratio'] = (clamped_floor / safe_total).round(4)

        return df

    def compute_investment_potential_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Module B: Transparent Multi-Criteria Decision Analysis (MCDA) Scoring Engine.
        Calculates an explainable 0 to 100 Investment Potential Score.
        """
        transport_score = df['Public_Transport_Accessibility'].map(self.TRANSPORT_MAP).fillna(0.0)
        security_score = (df['Security'] == 'Yes').astype(float)
        parking_score = (df['Parking_Space'] == 'Yes').astype(float)
        amenity_cnt = df['amenity_count'] if 'amenity_count' in df.columns else df['Amenities'].apply(self.count_amenities)

        # 1. Infrastructure & Utility Component (0 to 1)
        infra_component = (
            0.20 * transport_score +
            0.20 * (amenity_cnt / 5.0) +
            0.15 * (df['Nearby_Schools'] / 10.0) +
            0.15 * (df['Nearby_Hospitals'] / 10.0) +
            0.15 * security_score +
            0.15 * parking_score
        )

        # 2. Valuation & Margin of Safety Component (0 to 1)
        p_sqft = df['Price_per_SqFt']
        p_min, p_max = p_sqft.min(), p_sqft.max()
        value_component = 1.0 - ((p_sqft - p_min) / (p_max - p_min if p_max > p_min else 1.0))

        # 3. Asset Freshness Component (0 to 1)
        age = df['Age_of_Property']
        age_min, age_max = age.min(), age.max()
        age_component = 1.0 - ((age - age_min) / (age_max - age_min if age_max > age_min else 1.0))

        # Weighted Composite Score scaled to 0-100
        score = 100.0 * (
            0.40 * infra_component +
            0.35 * value_component +
            0.25 * age_component
        )
        return score.round(2)

    @staticmethod
    def calculate_scenario_projections(current_price: float) -> Dict[str, Dict[str, float]]:
        """
        Module C: 5-Year Scenario-Based Capital Value Projections.
        Uses illustrative compounding annual growth assumptions (5.0%, 7.5%, 10.0%).
        """
        scenarios = {
            "Conservative": {"annual_rate": 0.050, "label": "5.0% p.a."},
            "Moderate": {"annual_rate": 0.075, "label": "7.5% p.a."},
            "Optimistic": {"annual_rate": 0.100, "label": "10.0% p.a."}
        }
        results = {}
        for name, cfg in scenarios.items():
            r = cfg["annual_rate"]
            future_val = round(current_price * ((1.0 + r) ** 5), 2)
            net_gain = round(future_val - current_price, 2)
            growth_pct = round((net_gain / current_price) * 100, 2) if current_price > 0 else 0.0
            results[name] = {
                "annual_rate_pct": r * 100,
                "projected_price_lakhs": future_val,
                "net_gain_lakhs": net_gain,
                "growth_pct": growth_pct
            }
        return results

    def process_and_save(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes raw dataset, computes legitimate features, and attaches evaluation labels."""
        df_clean = self.create_features(df)

        # Attach transparent MCDA Investment Score for reference & classification evaluation
        df_clean['investment_potential_score'] = self.compute_investment_potential_score(df_clean)
        df_clean[TARGET_CLASSIFICATION] = (
            df_clean['investment_potential_score'] >= self.INVESTMENT_THRESHOLD
        ).astype(int)

        logger.info(f"Target Classification ({TARGET_CLASSIFICATION}) Class Balance: "
                    f"{df_clean[TARGET_CLASSIFICATION].value_counts(normalize=True).to_dict()}")
        logger.info(f"Target Regression ({TARGET_REGRESSION}) Summary: Mean ₹{df_clean[TARGET_REGRESSION].mean():.2f} Lakhs")

        logger.info(f"Saving processed dataset to {PROCESSED_CSV_PATH} and {PROCESSED_DATA_PATH}...")
        df_clean.to_csv(PROCESSED_CSV_PATH, index=False)
        try:
            df_clean.to_parquet(PROCESSED_DATA_PATH, index=False)
        except Exception as e:
            logger.warning(f"Could not save parquet: {e}")

        return df_clean

if __name__ == "__main__":
    from src.data_loader import DataLoader
    loader = DataLoader()
    raw_df = loader.load_data()
    fe = FeatureEngineer()
    processed_df = fe.process_and_save(raw_df)
    print("Done. Processed shape:", processed_df.shape)
