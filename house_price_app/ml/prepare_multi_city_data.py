"""
Multi-City Data Preparation Script for House Price Prediction System.
Prepares clean datasets for Bengaluru, Mumbai, Hyderabad, and Chennai.
Standard schema: location, total_sqft, bath, BHK, price (in Lakhs INR).
"""
import os
import sys
import re
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _clean_bengaluru():
    """Ensure data/bengaluru_clean.csv exists."""
    blr_path = os.path.join(config.DATA_DIR, "bengaluru_clean.csv")
    if os.path.exists(blr_path):
        df = pd.read_csv(blr_path)
        print(f"[OK] Bengaluru dataset ready: {df.shape[0]} rows.")
        return df

    # Fallback to prepare_bengaluru_data logic if not found
    from ml.prepare_bengaluru_data import prepare_bengaluru_dataset
    df = prepare_bengaluru_dataset()
    return df


def _clean_mumbai_and_hyderabad():
    """Extract and standardize Mumbai and Hyderabad from housing.csv."""
    housing_path = os.path.join(config.DATA_DIR, "housing.csv")
    if not os.path.exists(housing_path):
        print(f"[WARN] housing.csv not found at {housing_path}")
        return None, None

    raw_df = pd.read_csv(housing_path)

    # 1. Mumbai
    mumbai_raw = raw_df[raw_df["city"].str.strip().str.lower() == "mumbai"].copy()
    mumbai_clean = pd.DataFrame({
        "location": mumbai_raw["location"].str.strip(),
        "total_sqft": mumbai_raw["area_sqft"].astype(float),
        "bath": mumbai_raw["bathrooms"].astype(int),
        "BHK": mumbai_raw["bedrooms"].astype(int),
        "price": np.round(mumbai_raw["price"].astype(float), 2)
    }).dropna().drop_duplicates()
    mumbai_out = os.path.join(config.DATA_DIR, "mumbai_clean.csv")
    mumbai_clean.to_csv(mumbai_out, index=False)
    print(f"[OK] Mumbai dataset created: {mumbai_clean.shape[0]} rows -> {mumbai_out}")

    # 2. Hyderabad
    hyd_raw = raw_df[raw_df["city"].str.strip().str.lower() == "hyderabad"].copy()
    hyd_clean = pd.DataFrame({
        "location": hyd_raw["location"].str.strip(),
        "total_sqft": hyd_raw["area_sqft"].astype(float),
        "bath": hyd_raw["bathrooms"].astype(int),
        "BHK": hyd_raw["bedrooms"].astype(int),
        "price": np.round(hyd_raw["price"].astype(float), 2)
    }).dropna().drop_duplicates()
    hyd_out = os.path.join(config.DATA_DIR, "hyderabad_clean.csv")
    hyd_clean.to_csv(hyd_out, index=False)
    print(f"[OK] Hyderabad dataset created: {hyd_clean.shape[0]} rows -> {hyd_out}")

    return mumbai_clean, hyd_clean


def _clean_chennai():
    """Clean and standardize Chennai dataset from chennai_raw.csv."""
    chennai_raw_path = os.path.join(config.DATA_DIR, "chennai_raw.csv")
    if not os.path.exists(chennai_raw_path):
        print(f"[WARN] chennai_raw.csv not found at {chennai_raw_path}")
        return None

    df_ch = pd.read_csv(chennai_raw_path)

    # Fix known typos in raw area names
    area_fix = {
        "Chrompt": "Chrompet",
        "Chrmpet": "Chrompet",
        "Chormpet": "Chrompet",
        "Karapakam": "Karapakkam",
        "TNagar": "T Nagar",
        "Ana Nagar": "Anna Nagar",
        "Ann Nagar": "Anna Nagar",
        "Velchery": "Velachery",
        "Adyr": "Adyar",
        "KKNagar": "KK Nagar"
    }
    df_ch["AREA"] = df_ch["AREA"].replace(area_fix)

    # Drop missing values in crucial columns
    df_ch = df_ch.dropna(subset=["AREA", "INT_SQFT", "N_BEDROOM", "N_BATHROOM", "SALES_PRICE"])

    # SALES_PRICE in raw file is in Rupees (e.g. 7600000 = 76 Lakhs)
    # Convert to Lakhs INR
    chennai_clean = pd.DataFrame({
        "location": df_ch["AREA"].str.strip(),
        "total_sqft": df_ch["INT_SQFT"].astype(float),
        "bath": df_ch["N_BATHROOM"].astype(int),
        "BHK": df_ch["N_BEDROOM"].astype(int),
        "price": np.round(df_ch["SALES_PRICE"] / 100000.0, 2)
    }).drop_duplicates()

    # Outlier filter on price_per_sqft (optional IQR cleanup)
    chennai_clean["ppsf"] = (chennai_clean["price"] * 100000.0) / chennai_clean["total_sqft"]
    q1 = chennai_clean["ppsf"].quantile(0.01)
    q3 = chennai_clean["ppsf"].quantile(0.99)
    chennai_clean = chennai_clean[(chennai_clean["ppsf"] >= q1) & (chennai_clean["ppsf"] <= q3)].drop(columns=["ppsf"])

    chennai_out = os.path.join(config.DATA_DIR, "chennai_clean.csv")
    chennai_clean.to_csv(chennai_out, index=False)
    print(f"[OK] Chennai dataset created: {chennai_clean.shape[0]} rows -> {chennai_out}")
    return chennai_clean


def prepare_all_city_datasets():
    """Run data preparation for all 4 cities."""
    print("Preparing datasets for Bengaluru, Mumbai, Hyderabad, and Chennai...")
    _clean_bengaluru()
    _clean_mumbai_and_hyderabad()
    _clean_chennai()
    print("All city datasets successfully prepared!")


if __name__ == "__main__":
    prepare_all_city_datasets()

