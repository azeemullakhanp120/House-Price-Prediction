"""
Recommendation Engine — Multi-City House Price System.
Supports Bengaluru, Mumbai, Hyderabad, and Chennai.

Strategy
--------
1. At init, load city dataset and trained model+preprocessor for the city.
2. Run inference on every row to compute "predicted_price" and
   "value_score" (predicted / listed price).
3. Build a feature matrix for cosine-similarity lookup.
4. recommend_properties()  — filter by budget/BHK, rank by value+amenity/area proxy.
5. find_similar_properties() — cosine similarity on the feature matrix.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ml.preprocess import clean_data, load_raw_data
from ml.train import load_trained_artifacts


class HouseRecommender:
    """
    Content-Based Property Recommender supporting multiple cities.

    Parameters
    ----------
    data_path : str | None
        Path to clean CSV file (defaults to config.get_dataset_path(city)).
    city : str | None
        Target city name (defaults to config.DEFAULT_CITY).
    """

    def __init__(self, data_path=None, city=config.DEFAULT_CITY):
        self.city = config.normalize_city_name(city)
        if data_path is None:
            data_path = config.get_dataset_path(self.city)
        self.data_path = data_path

        # ── Load & clean dataset ──────────────────────────────────────────────
        self.df = clean_data(load_raw_data(data_path))

        # ── Load city model & preprocessor ────────────────────────────────────
        self.model, self.preprocessor = load_trained_artifacts(self.city)

        # ── Pre-compute predictions + value tags ──────────────────────────────
        X = self.df[config.ALL_FEATURE_COLUMNS]
        X_trans = self.preprocessor.transform(X)

        self.df["predicted_price"] = np.round(self.model.predict(X_trans), 2)
        self.df["value_score"] = np.round(
            self.df["predicted_price"] / self.df[config.TARGET_COLUMN].replace(0, 1), 2
        )
        self.df["value_tag"] = self.df["value_score"].apply(self._value_tag)

        # Feature matrix for cosine similarity
        self.feature_matrix = X_trans

        # Add synthetic property_id if not present
        cfg = config.get_city_config(self.city)
        short_code = cfg.get("short_code", "PROP")
        if "property_id" not in self.df.columns:
            self.df["property_id"] = [
                f"{short_code}-{1000 + i}" for i in range(len(self.df))
            ]

        self.df["city"] = self.city
        self.df = self.df.reset_index(drop=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _value_tag(score):
        """Map value_score to a human-readable badge."""
        if score >= config.VALUE_SCORE_THRESHOLDS["great_deal"]:
            return "Great Deal 🚀"
        elif score >= config.VALUE_SCORE_THRESHOLDS["fair_deal"]:
            return "Fair Price 👍"
        else:
            return "Overpriced ⚠️"

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend_properties(
        self,
        location=None,
        min_budget=None,
        max_budget=None,
        min_bedrooms=None,
        min_area=None,
        top_n=config.TOP_N_RECOMMENDATIONS,
    ):
        """
        Return the top-N properties matching user preferences,
        ranked by value_score and area.
        """
        filtered = self.df.copy()

        if location:
            filtered = filtered[
                filtered["location"].str.lower() == location.strip().lower()
            ]

        if min_budget is not None and not np.isnan(min_budget):
            filtered = filtered[filtered["price"] >= float(min_budget)]

        if max_budget is not None and not np.isnan(max_budget):
            filtered = filtered[filtered["price"] <= float(max_budget)]

        if min_bedrooms is not None and not np.isnan(min_bedrooms):
            filtered = filtered[filtered["BHK"] >= int(min_bedrooms)]

        if min_area is not None and not np.isnan(min_area):
            filtered = filtered[filtered["total_sqft"] >= float(min_area)]

        if filtered.empty:
            return []

        # Normalise total_sqft to [0, 1] range as a size proxy
        sqft_min = self.df["total_sqft"].min()
        sqft_max = self.df["total_sqft"].max()
        sqft_range = max(sqft_max - sqft_min, 1)

        area_proxy = (filtered["total_sqft"] - sqft_min) / sqft_range

        # Composite score: 70% value ratio + 30% area proxy
        filtered["composite_score"] = (
            filtered["value_score"] * 0.70 + area_proxy * 0.30
        )

        ranked = filtered.sort_values(by="composite_score", ascending=False).head(top_n)

        return ranked.to_dict(orient="records")

    def find_similar_properties(self, property_id, top_n=3):
        """Find the top-N properties most similar to property_id via cosine similarity."""
        idx_matches = self.df.index[self.df["property_id"] == property_id].tolist()
        if not idx_matches:
            return []

        target_idx = idx_matches[0]
        target_vec = self.feature_matrix[target_idx].reshape(1, -1)

        # Compute cosine similarity across all properties
        sim_scores = cosine_similarity(target_vec, self.feature_matrix).flatten()

        sim_indices = sim_scores.argsort()[::-1]
        sim_indices = [i for i in sim_indices if i != target_idx][:top_n]

        results = []
        for i in sim_indices:
            row = self.df.iloc[i].to_dict()
            row["similarity_score"] = round(float(sim_scores[i]), 3)
            results.append(row)

        return results


if __name__ == "__main__":
    for c in ["Bengaluru", "Mumbai", "Hyderabad", "Chennai"]:
        rec = HouseRecommender(city=c)
        recs = rec.recommend_properties(top_n=2)
        print(f"Recommender test for {c}: {len(recs)} properties found. First: {recs[0]['property_id'] if recs else 'None'}")
