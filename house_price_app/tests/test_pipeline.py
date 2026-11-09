"""
Unit and Integration Tests for Multi-City House Price Prediction & Recommendation System.
Tests coverage across Bengaluru, Mumbai, Hyderabad, and Chennai.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ml.preprocess import clean_data, load_raw_data
from ml.train import load_trained_artifacts
from ml.recommender import HouseRecommender
from app.app import app


class TestMultiCityPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test client and load default city artifacts."""
        cls.app = app
        cls.client = cls.app.test_client()
        cls.default_city = config.DEFAULT_CITY
        cls.model, cls.preprocessor = load_trained_artifacts(cls.default_city)
        cls.recommender = HouseRecommender(city=cls.default_city)

    # ── 1. City Registry Tests ────────────────────────────────────────────────
    def test_city_registry(self):
        """Ensure all 4 cities are configured and default city is Bengaluru."""
        self.assertEqual(config.DEFAULT_CITY, "Bengaluru")
        self.assertIn("Bengaluru", config.CITIES)
        self.assertIn("Mumbai", config.CITIES)
        self.assertIn("Hyderabad", config.CITIES)
        self.assertIn("Chennai", config.CITIES)

        # Test case-insensitive resolution and short codes
        self.assertEqual(config.normalize_city_name("mumbai"), "Mumbai")
        self.assertEqual(config.normalize_city_name("BLR"), "Bengaluru")
        self.assertEqual(config.normalize_city_name("HYD"), "Hyderabad")
        self.assertEqual(config.normalize_city_name("chennai"), "Chennai")
        self.assertEqual(config.normalize_city_name("unknown_city"), "Bengaluru")

    def test_city_locations(self):
        """Ensure each city has non-empty locality options."""
        for city in ["Bengaluru", "Mumbai", "Hyderabad", "Chennai"]:
            locs = config.get_city_locations(city)
            self.assertIsInstance(locs, list)
            self.assertGreater(len(locs), 0, f"Locations empty for {city}")

    # ── 2. Data Cleaning Tests ────────────────────────────────────────────────
    def test_datasets_present_and_clean(self):
        """Ensure cleaned datasets exist and follow canonical schema without NaNs."""
        for city in ["Bengaluru", "Mumbai", "Hyderabad", "Chennai"]:
            path = config.get_dataset_path(city)
            self.assertTrue(os.path.exists(path), f"Dataset missing for {city} at {path}")
            df = clean_data(pd.read_csv(path))
            self.assertFalse(df[config.TARGET_COLUMN].isnull().any())
            for col in config.ALL_FEATURE_COLUMNS:
                self.assertIn(col, df.columns)
            self.assertGreater(len(df), 100, f"Dataset too small for {city}")

    # ── 3. Model Prediction Tests for Each City ───────────────────────────────
    def test_multi_city_predictions(self):
        """Test model predictions across all supported cities."""
        sample_queries = {
            "Bengaluru": {"location": "Whitefield", "total_sqft": 1400.0, "bath": 2, "BHK": 3},
            "Mumbai": {"location": "Andheri", "total_sqft": 1000.0, "bath": 2, "BHK": 2},
            "Hyderabad": {"location": "Gachibowli", "total_sqft": 1500.0, "bath": 2, "BHK": 3},
            "Chennai": {"location": "Adyar", "total_sqft": 1200.0, "bath": 2, "BHK": 2},
        }

        for city, payload in sample_queries.items():
            model, prep = load_trained_artifacts(city)
            input_df = pd.DataFrame([payload])
            X_trans = prep.transform(input_df)
            pred_price = float(model.predict(X_trans)[0])

            self.assertIsInstance(pred_price, float)
            self.assertGreater(pred_price, 10.0, f"Prediction too low for {city}: {pred_price}")
            self.assertLess(pred_price, 2000.0, f"Prediction too high for {city}: {pred_price}")

    # ── 4. Recommender Tests ──────────────────────────────────────────────────
    def test_recommenders_per_city(self):
        """Ensure recommenders rank and return properties for each city."""
        for city in ["Bengaluru", "Mumbai", "Hyderabad", "Chennai"]:
            rec = HouseRecommender(city=city)
            recs = rec.recommend_properties(top_n=3)

            self.assertIsInstance(recs, list)
            self.assertGreater(len(recs), 0, f"No recommendations for {city}")
            first = recs[0]
            self.assertIn("property_id", first)
            self.assertIn("predicted_price", first)
            self.assertIn("value_tag", first)
            self.assertEqual(first["city"], city)

    def test_find_similar_properties(self):
        """Ensure cosine-similarity finds similar properties by ID."""
        target_id = self.recommender.df["property_id"].iloc[0]
        sims = self.recommender.find_similar_properties(target_id, top_n=2)
        self.assertEqual(len(sims), 2)
        self.assertIn("similarity_score", sims[0])

    # ── 5. Flask Route & REST API Tests ───────────────────────────────────────
    def test_api_cities(self):
        """GET /api/cities should return all 4 cities."""
        res = self.client.get("/api/cities")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["cities"]), 4)

    def test_api_locations_city_param(self):
        """GET /api/locations with ?city=Mumbai should return Mumbai localities."""
        res = self.client.get("/api/locations?city=Mumbai")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["city"], "Mumbai")
        self.assertIn("Andheri", data["locations"])

    def test_api_predict_post(self):
        """POST /api/predict with Mumbai payload should calculate price."""
        payload = {
            "city": "Mumbai",
            "location": "Bandra",
            "total_sqft": 1100.0,
            "bath": 2,
            "BHK": 2,
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["city"], "Mumbai")
        self.assertGreater(data["predicted_price_lakhs"], 0)

    def test_api_stats_city_param(self):
        """GET /api/stats with ?city=Hyderabad should return stats."""
        res = self.client.get("/api/stats?city=Hyderabad")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["stats"]["city"], "Hyderabad")
        self.assertGreater(data["stats"]["total_properties"], 0)

    def test_web_routes_render(self):
        """Ensure web UI routes render without error."""
        for path in ["/", "/predict?city=Mumbai", "/recommend?city=Chennai", "/dashboard?city=Hyderabad", "/about"]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200, f"Failed rendering route: {path}")


if __name__ == "__main__":
    unittest.main()
