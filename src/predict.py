import logging
from pathlib import Path
from typing import Dict, Any, Union
import joblib
import pandas as pd
import numpy as np

from src.config import (
    MODELS_DIR, TARGET_CLASSIFICATION, TARGET_REGRESSION,
    FEATURE_NUMERIC, FEATURE_CATEGORICAL, FEATURE_SPATIAL
)
from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PropertyPredictor:
    """
    Unified inference and evaluation engine for Real Estate Investment Advisor.
    Coordinates:
    - Module A: Genuine ML Property Valuation
    - Module B: Transparent Multi-Criteria Investment Potential Scoring
    - Module C: 5-Year Scenario-Based Capital Value Projections
    """

    def __init__(self, models_dir: Union[str, Path] = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.clf_pipeline = None
        self.reg_pipeline = None
        self.fe = FeatureEngineer()
        self.load_models()

    def load_models(self):
        """Loads serialized champion models from disk."""
        clf_path = self.models_dir / "classification_model.joblib"
        reg_path = self.models_dir / "regression_model.joblib"

        if clf_path.exists():
            logger.info(f"Loading classification model from {clf_path}...")
            self.clf_pipeline = joblib.load(clf_path)
        else:
            logger.warning(f"Classification model not yet found at {clf_path}.")

        if reg_path.exists():
            logger.info(f"Loading regression model from {reg_path}...")
            self.reg_pipeline = joblib.load(reg_path)
        else:
            logger.warning(f"Regression model not yet found at {reg_path}.")

    def prepare_input(self, raw_input: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Converts raw input dictionary or DataFrame and derives legitimate features."""
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        else:
            df = raw_input.copy()

        # Defaults for optional / omitted columns
        defaults = {
            'State': 'Maharashtra',
            'City': 'Mumbai',
            'Locality': 'Locality_1',
            'Property_Type': 'Apartment',
            'BHK': 3,
            'Size_in_SqFt': 1500,
            'Price_in_Lakhs': 150.0,
            'Year_Built': 2020,
            'Age_of_Property': 5,
            'Floor_No': 5,
            'Total_Floors': 10,
            'Nearby_Schools': 5,
            'Nearby_Hospitals': 5,
            'Public_Transport_Accessibility': 'Medium',
            'Parking_Space': 'No',
            'Security': 'No',
            'Amenities': 'Clubhouse',
            'Furnished_Status': 'Semi-furnished',
            'Facing': 'North',
            'Owner_Type': 'Owner',
            'Availability_Status': 'Ready_to_Move'
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        # Derive Price_per_SqFt for MCDA scoring (kept outside ML feature matrix)
        if 'Price_per_SqFt' not in df.columns or df['Price_per_SqFt'].isnull().all():
            size_safe = np.maximum(df['Size_in_SqFt'].astype(float), 1.0)
            df['Price_per_SqFt'] = (df['Price_in_Lakhs'].astype(float) / size_safe).round(2)

        # Derive Age_of_Property if missing
        if 'Age_of_Property' not in df.columns and 'Year_Built' in df.columns:
            df['Age_of_Property'] = 2025 - df['Year_Built']
        elif 'Year_Built' not in df.columns and 'Age_of_Property' in df.columns:
            df['Year_Built'] = 2025 - df['Age_of_Property']

        # Derive pure legitimate features (amenity_count, floor_ratio)
        df = self.fe.create_features(df)
        return df

    def evaluate_property(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes complete evaluation across Modules A, B, and C.
        """
        df_input = self.prepare_input(property_data)
        current_price = float(df_input['Price_in_Lakhs'].iloc[0])
        size_sqft = float(df_input['Size_in_SqFt'].iloc[0])
        price_per_sqft = float(df_input['Price_per_SqFt'].iloc[0])
        amenity_cnt = int(df_input['amenity_count'].iloc[0])

        # Feature matrix for ML models (strict zero leakage)
        feature_cols = FEATURE_NUMERIC + FEATURE_CATEGORICAL + FEATURE_SPATIAL
        X_ml = df_input[feature_cols].copy()

        # -------------------------------------------------------------
        # MODULE A: Genuine ML Valuation
        # -------------------------------------------------------------
        if self.reg_pipeline is not None:
            ml_estimated_price = float(self.reg_pipeline.predict(X_ml)[0])
            ml_valuation_lakhs = round(max(ml_estimated_price, 10.0), 2)
            valuation_diff = round(current_price - ml_valuation_lakhs, 2)
        else:
            ml_valuation_lakhs = current_price
            valuation_diff = 0.0

        # -------------------------------------------------------------
        # MODULE B: Transparent MCDA Investment Potential Score
        # -------------------------------------------------------------
        score_series = self.fe.compute_investment_potential_score(df_input)
        investment_score = float(score_series.iloc[0])

        if investment_score >= 70.0:
            rating_tier = "High Investment Potential"
            rating_badge = "TIER 1 — HIGH POTENTIAL"
            badge_color = "good"
        elif investment_score >= 55.0:
            rating_tier = "Moderate Investment Potential"
            rating_badge = "TIER 2 — MODERATE POTENTIAL"
            badge_color = "moderate"
        else:
            rating_tier = "Low Investment Potential / Cautious"
            rating_badge = "TIER 3 — CAUTIOUS / LOW"
            badge_color = "bad"

        # Surrogate classifier probability
        if self.clf_pipeline is not None:
            clf_prob = float(self.clf_pipeline.predict_proba(X_ml)[0][1]) if hasattr(self.clf_pipeline, "predict_proba") else (1.0 if investment_score >= 70.0 else 0.0)
            clf_pred = int(self.clf_pipeline.predict(X_ml)[0])
        else:
            clf_prob = investment_score / 100.0
            clf_pred = 1 if investment_score >= 70.0 else 0

        # Contributing highlights
        strengths = []
        cautions = []
        if df_input['Public_Transport_Accessibility'].iloc[0] == 'High':
            strengths.append("High public transport accessibility (+)")
        elif df_input['Public_Transport_Accessibility'].iloc[0] == 'Low':
            cautions.append("Low public transit connectivity (-)")

        if amenity_cnt >= 4:
            strengths.append(f"Rich lifestyle amenities ({amenity_cnt}/5) (+)")
        elif amenity_cnt <= 1:
            cautions.append("Minimal lifestyle amenities (<=1) (-)")

        if df_input['Age_of_Property'].iloc[0] <= 8:
            strengths.append(f"Recent construction (Age: {int(df_input['Age_of_Property'].iloc[0])} yrs) (+)")
        elif df_input['Age_of_Property'].iloc[0] >= 25:
            cautions.append(f"Aging structure (Age: {int(df_input['Age_of_Property'].iloc[0])} yrs) (-)")

        if price_per_sqft <= 0.08:
            strengths.append(f"Favorable entry cost per sqft (₹{int(price_per_sqft*100000)}/sqft) (+)")
        elif price_per_sqft >= 0.20:
            cautions.append(f"Elevated price per sqft (₹{int(price_per_sqft*100000)}/sqft) (-)")

        if df_input['Security'].iloc[0] == 'Yes':
            strengths.append("Gated security & surveillance (+)")
        if df_input['Parking_Space'].iloc[0] == 'Yes':
            strengths.append("Dedicated parking reserved (+)")

        # -------------------------------------------------------------
        # MODULE C: 5-Year Illustrative Scenario Projections
        # -------------------------------------------------------------
        scenario_projections = self.fe.calculate_scenario_projections(current_price)

        return {
            "current_price_lakhs": current_price,
            "size_sqft": size_sqft,
            "price_per_sqft_lakhs": price_per_sqft,
            "ml_valuation_lakhs": ml_valuation_lakhs,
            "valuation_diff_lakhs": valuation_diff,
            "investment_potential_score": investment_score,
            "rating_tier": rating_tier,
            "rating_badge": rating_badge,
            "badge_color": badge_color,
            "classifier_predicted_high_potential": clf_pred,
            "classifier_confidence_pct": round(clf_prob * 100, 1),
            "strengths": strengths,
            "cautions": cautions,
            "scenarios": scenario_projections
        }

    # Backward compatibility wrapper for existing tests
    def predict_single(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        report = self.evaluate_property(property_data)
        mod_scenario = report["scenarios"]["Moderate"]
        return {
            "current_price_lakhs": report["current_price_lakhs"],
            "price_per_sqft": report["price_per_sqft_lakhs"],
            "infra_score": report["investment_potential_score"] / 100.0,
            "amenity_count": property_data.get("Amenities", "").count(",") + 1 if property_data.get("Amenities") else 0,
            "is_good_investment": report["classifier_predicted_high_potential"],
            "recommendation": "GOOD INVESTMENT" if report["classifier_predicted_high_potential"] == 1 else "NOT RECOMMENDED",
            "confidence_pct": report["classifier_confidence_pct"],
            "estimated_5y_price_lakhs": mod_scenario["projected_price_lakhs"],
            "absolute_gain_lakhs": mod_scenario["net_gain_lakhs"],
            "expected_growth_pct": mod_scenario["growth_pct"],
            "annual_cagr_pct": mod_scenario["annual_rate_pct"],
            "key_factors": report["strengths"]
        }
