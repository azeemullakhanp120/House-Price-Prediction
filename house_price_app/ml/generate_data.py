"""
Synthetic Multi-City Housing Data Generator.

Generates realistic simulated property data for Bengaluru, Mumbai, Hyderabad, and Chennai.
Produces standardized clean datasets matching the system schema:
    property_id, location, total_sqft, bath, BHK, price (in Lakhs INR)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Representative localities and base price-per-sqft (INR) for each city
CITY_LOCATION_SPECS = {
    "Bengaluru": {
        "short_code": "BLR",
        "locations": {
            "Indiranagar": 12500,
            "Koramangala": 11500,
            "Jayanagar": 10800,
            "HSR Layout": 9200,
            "Bellandur": 7800,
            "Sarjapur Road": 7000,
            "Whitefield": 6800,
            "Marathahalli": 6400,
            "Yelahanka": 5500,
            "Electronic City": 4600,
            "Hebbal": 9800,
            "Bannerghatta Road": 6200,
            "JP Nagar": 7500,
            "Rajaji Nagar": 8800,
            "Basavangudi": 9600,
            "Banaswadi": 6700,
            "Kadugodi": 5800,
            "Hoodi": 6100,
            "Thanisandra": 5300,
            "Yeshwanthpur": 7900,
        },
        "base_sqft_mult": 1.0,
        "base_price_mult": 1.0,
    },
    "Mumbai": {
        "short_code": "BOM",
        "locations": {
            "Worli": 42000,
            "Bandra": 38000,
            "Powai": 22000,
            "Andheri": 21000,
            "Goregaon": 17500,
            "Borivali": 16000,
            "Malad": 15500,
            "Kurla": 13500,
            "Thane": 11000,
            "Navi Mumbai": 9500,
        },
        "base_sqft_mult": 0.85,
        "base_price_mult": 1.4,
    },
    "Hyderabad": {
        "short_code": "HYD",
        "locations": {
            "Jubilee Hills": 14000,
            "Banjara Hills": 13000,
            "Hitec City": 9200,
            "Madhapur": 8800,
            "Gachibowli": 8500,
            "Kondapur": 7500,
            "Begumpet": 7000,
            "Secunderabad": 6500,
            "Kukatpally": 6200,
            "Miyapur": 5400,
            "Uppal": 4800,
        },
        "base_sqft_mult": 1.1,
        "base_price_mult": 0.85,
    },
    "Chennai": {
        "short_code": "MAA",
        "locations": {
            "T Nagar": 14000,
            "Adyar": 13500,
            "Besant Nagar": 13000,
            "Anna Nagar": 12000,
            "KK Nagar": 8200,
            "Velachery": 7800,
            "OMR": 6200,
            "Porur": 6000,
            "Karapakkam": 5800,
            "Chrompet": 5500,
        },
        "base_sqft_mult": 1.0,
        "base_price_mult": 0.9,
    },
}


def generate_city_dataset(city_name: str,
                          n_samples: int = 1600,
                          random_seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """
    Generate synthetic property listings for a given city and save to data/<city>_clean.csv.
    """
    canonical_city = config.normalize_city_name(city_name)
    spec = CITY_LOCATION_SPECS.get(canonical_city, CITY_LOCATION_SPECS["Bengaluru"])

    np.random.seed(random_seed)
    locations = list(spec["locations"].keys())
    base_psfts = list(spec["locations"].values())
    short_code = spec["short_code"]

    data = []
    for i in range(1, n_samples + 1):
        loc_idx = np.random.randint(0, len(locations))
        location = locations[loc_idx]
        base_psft = base_psfts[loc_idx]

        bhk = int(np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.35, 0.42, 0.12, 0.03]))

        if bhk == 1:
            sqft = np.random.uniform(450, 900) * spec["base_sqft_mult"]
            bath = 1
        elif bhk == 2:
            sqft = np.random.uniform(850, 1400) * spec["base_sqft_mult"]
            bath = int(np.random.choice([1, 2], p=[0.25, 0.75]))
        elif bhk == 3:
            sqft = np.random.uniform(1300, 2200) * spec["base_sqft_mult"]
            bath = int(np.random.choice([2, 3], p=[0.35, 0.65]))
        elif bhk == 4:
            sqft = np.random.uniform(2000, 3400) * spec["base_sqft_mult"]
            bath = int(np.random.choice([3, 4], p=[0.45, 0.55]))
        else:
            sqft = np.random.uniform(2800, 4800) * spec["base_sqft_mult"]
            bath = int(np.random.choice([4, 5], p=[0.5, 0.5]))

        sqft = round(float(sqft), 1)

        # Price calculation (INR -> Lakhs)
        raw_inr = (
            sqft * base_psft
            + bhk * 250_000 * spec["base_price_mult"]
            + bath * 150_000 * spec["base_price_mult"]
        )
        noise = np.random.normal(0, raw_inr * 0.06)
        price_lakhs = round(float(max(15.0, (raw_inr + noise) / 1e5)), 2)

        data.append({
            "property_id": f"{short_code}-{1000 + i}",
            "location": location,
            "total_sqft": sqft,
            "bath": bath,
            "BHK": bhk,
            "price": price_lakhs,
        })

    df = pd.DataFrame(data)

    # Inject minor NaNs into Bengaluru to exercise the median/mode imputer
    if canonical_city == "Bengaluru":
        nan_idx = np.random.choice(df.index, size=10, replace=False)
        df.loc[nan_idx[:5], "bath"] = np.nan
        df.loc[nan_idx[5:], "total_sqft"] = np.nan

    out_path = config.get_dataset_path(canonical_city)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Synthetic {canonical_city} dataset saved: {out_path} ({len(df)} rows)")
    return df


def generate_bengaluru_dataset(n_samples: int = 1600,
                                random_seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Generate synthetic Bengaluru housing data (backwards-compatible alias)."""
    return generate_city_dataset("Bengaluru", n_samples=n_samples, random_seed=random_seed)


def generate_all_city_datasets(n_samples: int = 1600) -> dict:
    """Generate synthetic datasets for all configured cities and create unified housing.csv."""
    dfs = {}
    combined_rows = []

    for idx, (city_name, spec) in enumerate(CITY_LOCATION_SPECS.items()):
        df_city = generate_city_dataset(city_name, n_samples=n_samples, random_seed=config.RANDOM_STATE + idx)
        dfs[city_name] = df_city

        # Append to multi-city unified housing.csv
        for _, row in df_city.iterrows():
            combined_rows.append({
                "property_id": row["property_id"],
                "city": city_name,
                "location": row["location"],
                "area_sqft": row["total_sqft"],
                "bedrooms": row["BHK"],
                "bathrooms": row["bath"],
                "price": row["price"],
            })

    # Save housing.csv for multi-city reference
    housing_df = pd.DataFrame(combined_rows)
    housing_path = os.path.join(config.DATA_DIR, "housing.csv")
    housing_df.to_csv(housing_path, index=False)
    print(f"[OK] Unified synthetic housing.csv saved: {housing_path} ({len(housing_df)} rows)")

    return dfs


if __name__ == "__main__":
    generate_all_city_datasets()
