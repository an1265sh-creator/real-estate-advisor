import unittest
import pandas as pd
from src.config import (
    FEATURE_NUMERIC, FEATURE_CATEGORICAL, FEATURE_SPATIAL,
    FORBIDDEN_LEAKAGE_FEATURES, TARGET_REGRESSION, TARGET_CLASSIFICATION
)
from src.train_classifier import ClassifierTrainer
from src.train_regressor import RegressorTrainer
from src.predict import PropertyPredictor

class TestTargetLeakage(unittest.TestCase):
    """
    Automated test suite enforcing that target-derived variables, algebraic proxies,
    and synthetic future prices NEVER enter machine learning feature matrices.
    """

    def test_feature_list_excludes_forbidden_leakage(self):
        all_features = set(FEATURE_NUMERIC + FEATURE_CATEGORICAL + FEATURE_SPATIAL)
        for forbidden in FORBIDDEN_LEAKAGE_FEATURES:
            self.assertNotIn(
                forbidden, all_features,
                f"AUDIT VIOLATION: {forbidden} was found in the ML feature list!"
            )

    def test_price_per_sqft_excluded_from_valuation_features(self):
        all_features = set(FEATURE_NUMERIC + FEATURE_CATEGORICAL + FEATURE_SPATIAL)
        self.assertNotIn('Price_per_SqFt', all_features)
        self.assertNotIn('infra_score', all_features)
        self.assertNotIn('investment_score_raw', all_features)
        self.assertNotIn('Price_After_5_Years', all_features)

    def test_targets_not_in_features(self):
        all_features = set(FEATURE_NUMERIC + FEATURE_CATEGORICAL + FEATURE_SPATIAL)
        self.assertNotIn(TARGET_REGRESSION, all_features)
        self.assertNotIn(TARGET_CLASSIFICATION, all_features)

    def test_classifier_trainer_splits_leakage_free(self):
        trainer = ClassifierTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data()
        for forbidden in FORBIDDEN_LEAKAGE_FEATURES:
            self.assertNotIn(forbidden, X_train.columns)
            self.assertNotIn(forbidden, X_test.columns)

    def test_regressor_trainer_splits_leakage_free(self):
        trainer = RegressorTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data()
        for forbidden in FORBIDDEN_LEAKAGE_FEATURES:
            self.assertNotIn(forbidden, X_train.columns)
            self.assertNotIn(forbidden, X_test.columns)

    def test_predict_single_evaluates_without_leakage(self):
        predictor = PropertyPredictor()
        sample = {
            'State': 'Maharashtra', 'City': 'Mumbai', 'Locality': 'Locality_1',
            'Property_Type': 'Apartment', 'BHK': 3, 'Size_in_SqFt': 1500,
            'Price_in_Lakhs': 150.0, 'Age_of_Property': 5, 'Floor_No': 3,
            'Total_Floors': 10, 'Nearby_Schools': 6, 'Nearby_Hospitals': 5,
            'Public_Transport_Accessibility': 'High', 'Parking_Space': 'Yes',
            'Security': 'Yes', 'Amenities': 'Gym, Pool, Clubhouse',
            'Furnished_Status': 'Semi-furnished', 'Facing': 'East',
            'Owner_Type': 'Owner', 'Availability_Status': 'Ready_to_Move'
        }
        res = predictor.evaluate_property(sample)
        self.assertIn("ml_valuation_lakhs", res)
        self.assertIn("investment_potential_score", res)
        self.assertIn("scenarios", res)
        self.assertIn("Conservative", res["scenarios"])
        self.assertIn("Moderate", res["scenarios"])
        self.assertIn("Optimistic", res["scenarios"])

if __name__ == '__main__':
    unittest.main()
