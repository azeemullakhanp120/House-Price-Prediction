"""
Data Preparation Script for Bengaluru House Price Prediction System.

Pipeline:
  1. Load the raw Bengaluru Housing dataset (e.g. Kaggle "Bengaluru House Data").
  2. Filter to city == "Bengaluru" if a 'city' column exists, then drop it.
  3. Rename columns to a consistent internal schema:
       location, total_sqft, bath, BHK, price (Lakhs)
  4. Convert total_sqft ranges (e.g. "1200 - 1500") to their numeric midpoint.
  5. Extract BHK from the 'size' column (e.g. "3 BHK" -> 3).
  6. Drop nulls, duplicates, and impossible values.
  7. Remove per-location price_per_sqft outliers (IQR method).
  8. Group locations with < 10 listings into "Other".
  9. Save the result as data/bengaluru_clean.csv.

Usage:
    python ml/prepare_bengaluru_data.py
"""
import os
import sys
import re
import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Step 0 – locate the raw source file ───────────────────────────────────────
# Fallback chain: look for bengaluru_house_data.csv or housing.csv in data/
_CANDIDATE_RAW = [
    os.path.join(config.DATA_DIR, "bengaluru_house_data.csv"),
    os.path.join(config.DATA_DIR, "housing.csv"),
]


def _find_raw_file():
    """Return the first existing raw data file from the candidate list."""
    for path in _CANDIDATE_RAW:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Raw dataset not found. Please place 'bengaluru_house_data.csv' "
        f"(or 'housing.csv') inside {config.DATA_DIR}."
    )


# ── Step 1 – convert total_sqft to a single float ─────────────────────────────
def _parse_sqft(val):
    """
    Convert a raw total_sqft value to a float:
      - "1200.0"      -> 1200.0
      - "1200 - 1500" -> 1350.0   (midpoint)
      - "34.46Sq. Meter" -> convert m² -> sqft
      - other non-numeric -> NaN
    """
    if pd.isnull(val):
        return np.nan
    val = str(val).strip()

    # Range like "1200 - 1500"
    if " - " in val:
        parts = val.split(" - ")
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except ValueError:
            return np.nan

    # Square metres (rare)
    if "Sq. Meter" in val:
        try:
            return float(val.replace("Sq. Meter", "").strip()) * 10.7639
        except ValueError:
            return np.nan

    # Plain numeric / "1200.0 sqft"
    numeric = re.sub(r"[^\d.]", "", val)
    try:
        return float(numeric) if numeric else np.nan
    except ValueError:
        return np.nan


# ── Step 2 – extract BHK integer from 'size' column ──────────────────────────
def _extract_bhk(size_val):
    """
    '3 BHK'       -> 3
    '4 Bedroom'   -> 4
    '1 RK'        -> 1
    """
    if pd.isnull(size_val):
        return np.nan
    match = re.search(r"(\d+)", str(size_val))
    return int(match.group(1)) if match else np.nan


# -- Step 3 - IQR-based outlier removal per location -------------------------
def _remove_ppsf_outliers(df):
    """
    Remove rows whose price_per_sqft is an extreme outlier within each location.
    Uses a +/-1.5xIQR fence on log-price_per_sqft for robustness.
    Returns a clean copy with the helper column dropped.
    """
    df = df.copy()
    df["price_per_sqft"] = df["price"] * 1e5 / df["total_sqft"]   # price is in Lakhs

    keep_mask = pd.Series(True, index=df.index)

    for loc, grp in df.groupby("location"):
        if len(grp) < 4:          # too small to compute meaningful IQR
            continue
        log_ppsf = np.log1p(grp["price_per_sqft"])
        q1, q3 = log_ppsf.quantile(0.25), log_ppsf.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_idx = grp.index[(log_ppsf < lo) | (log_ppsf > hi)]
        keep_mask[outlier_idx] = False

    cleaned = df[keep_mask].drop(columns=["price_per_sqft"])
    return cleaned.reset_index(drop=True)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def prepare_bengaluru_data(raw_path=None, out_path=config.DATASET_PATH):
    """
    Full Bengaluru data-preparation pipeline.

    Parameters
    ----------
    raw_path : str | None
        Path to the raw CSV.  Autodiscovered if None.
    out_path : str
        Destination path for bengaluru_clean.csv.

    Returns
    -------
    pd.DataFrame  – the cleaned dataset.
    """
    if raw_path is None:
        raw_path = _find_raw_file()

    print(f"[1/9] Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"      Raw shape: {df.shape}")

    # ── Filter to Bengaluru if 'city' column present ──
    if "city" in df.columns:
        print("[2/9] Filtering city == 'Bengaluru' ...")
        df = df[df["city"].str.strip().str.lower() == "bengaluru"].copy()
        df = df.drop(columns=["city"])
        print(f"      After city filter: {df.shape}")
    else:
        print("[2/9] No 'city' column found – treating entire file as Bengaluru data.")

    # ── Normalise column names ────────────────────────────────────────────────
    # The Kaggle Bengaluru dataset columns:
    #   area_type, availability, location, size, society, total_sqft, bath, balcony, price
    # The old synthetic multi-city dataset columns:
    #   property_id, city, location, area_sqft, bedrooms, bathrooms, floor, age_years,
    #   parking, furnishing, amenities_score, price
    # We normalise both to: location, total_sqft, bath, BHK, price
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename old synthetic column names -> canonical Bengaluru schema
    rename_map = {
        "area_sqft":  "total_sqft",   # synthetic -> canonical
        "bathrooms":  "bath",          # synthetic -> canonical
        "bedrooms":   "BHK",           # synthetic -> canonical (will skip size step)
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required_raw = {"location", "total_sqft", "bath", "price"}
    missing_req = required_raw - set(df.columns)
    if missing_req:
        raise ValueError(f"Raw dataset is missing required columns: {missing_req}")

    # ── Parse total_sqft ──────────────────────────────────────────────────────
    print("[3/9] Parsing total_sqft ...")
    df["total_sqft"] = df["total_sqft"].apply(_parse_sqft)

    # ── Extract BHK ───────────────────────────────────────────────────────────
    print("[4/9] Extracting BHK from 'size' column (or using already-renamed 'BHK') ...")
    if "BHK" not in df.columns:
        # Kaggle dataset: parse from 'size' column e.g. "3 BHK"
        if "size" in df.columns:
            df["BHK"] = df["size"].apply(_extract_bhk)
        else:
            # Last fallback: infer from bath count
            df["BHK"] = df["bath"].astype(float)

    # ── Keep only the columns we need ────────────────────────────────────────
    keep_cols = ["location", "total_sqft", "bath", "BHK", "price"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # ── Drop nulls and duplicates ─────────────────────────────────────────────
    print("[5/9] Dropping nulls and duplicates ...")
    before = len(df)
    df = df.dropna(subset=["location", "total_sqft", "bath", "BHK", "price"])
    df = df.drop_duplicates()
    print(f"      Rows: {before} -> {len(df)}")

    # ── Sanity-check numeric ranges ───────────────────────────────────────────
    print("[6/9] Removing impossible / extreme rows ...")
    before = len(df)
    df["bath"]       = df["bath"].astype(float)
    df["BHK"]        = df["BHK"].astype(int)
    df["total_sqft"] = df["total_sqft"].astype(float)
    df["price"]      = df["price"].astype(float)

    df = df[
        (df["total_sqft"] >= 300) & (df["total_sqft"] <= 30000) &
        (df["bath"] >= 1)         & (df["bath"] <= 20)           &
        (df["BHK"]  >= 1)         & (df["BHK"]  <= 10)           &
        (df["price"] > 0)
    ]
    # Drop rows where bath count is absurdly higher than BHK
    df = df[df["bath"] < df["BHK"] + 2]
    print(f"      Rows after range filter: {before} -> {len(df)}")

    # ── Per-location price_per_sqft outlier removal ───────────────────────────
    print("[7/9] Removing per-location price_per_sqft outliers ...")
    before = len(df)
    df = _remove_ppsf_outliers(df)
    print(f"      Rows: {before} -> {len(df)}")

    # ── Normalise location strings ─────────────────────────────────────────────
    print("[8/9] Normalising locations (group rare ones into 'Other') ...")
    df["location"] = df["location"].str.strip()
    loc_counts = df["location"].value_counts()
    rare_locs  = loc_counts[loc_counts < 10].index
    df["location"] = df["location"].replace(rare_locs, "Other")
    print(f"      Unique locations after grouping: {df['location'].nunique()}")

    # ── Save ───────────────────────────────────────────────────────────────────
    print(f"[9/9] Saving cleaned dataset to: {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"      Final dataset: {len(df)} rows x {len(df.columns)} columns")
    print(f"      Columns: {list(df.columns)}")
    print(f"      Price range: Rs.{df['price'].min():.2f}L – Rs.{df['price'].max():.2f}L")

    return df


if __name__ == "__main__":
    prepare_bengaluru_data()
