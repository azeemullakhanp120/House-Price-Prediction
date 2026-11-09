"""
Data Preprocessing Module for Bengaluru House Price Prediction System.

Handles:
  - Loading bengaluru_clean.csv
  - Basic cleaning (duplicates, missing values, IQR outlier capping)
  - Building a sklearn Pipeline with OneHotEncoder for location
  - Fitting and persisting the preprocessor to models/preprocessor.pkl
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Raw Data Loading ──────────────────────────────────────────────────────────

def load_raw_data(data_path=config.DATASET_PATH):
    """Load the clean CSV dataset from disk. If not yet generated, generates synthetic data."""
    if not os.path.exists(data_path):
        from ml.generate_data import generate_all_city_datasets
        generate_all_city_datasets()
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please run: python ml/generate_data.py"
        )
    df = pd.read_csv(data_path)
    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_data(df, handle_outliers=True):
    """
    Clean the Bengaluru dataset:
      1. Drop duplicate rows.
      2. Impute missing numeric values with column median.
      3. Impute missing categorical values with column mode.
      4. Optionally cap total_sqft and price outliers using 1.5×IQR.

    Parameters
    ----------
    df              : raw DataFrame (must contain config.ALL_FEATURE_COLUMNS + target)
    handle_outliers : bool – whether to apply IQR capping

    Returns
    -------
    pd.DataFrame – cleaned copy
    """
    df = df.copy()

    # 1. Duplicates
    df = df.drop_duplicates()

    # 2. Numeric imputation (median)
    for col in config.NUMERICAL_COLUMNS:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 3. Categorical imputation (mode)
    for col in config.CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. IQR outlier capping on total_sqft and price
    if handle_outliers:
        for col in ["total_sqft", config.TARGET_COLUMN]:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df[col] = np.clip(df[col], lower, upper)

    return df


# ── sklearn Pipeline ──────────────────────────────────────────────────────────

def create_preprocessor():
    """
    Build a ColumnTransformer pipeline:
      - Numeric branch : median imputer + StandardScaler
      - Categorical branch : mode imputer + OneHotEncoder (handle_unknown='ignore')

    The 'location' column receives OneHotEncoding so the model learns per-area
    price premiums without leaking future unseen categories.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore" → unseen locations during inference map to
        # all-zero OHE vector rather than raising an error.
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,      config.NUMERICAL_COLUMNS),
            ("cat", categorical_pipeline,  config.CATEGORICAL_COLUMNS),
        ],
        remainder="drop",    # drop any extra columns not in the feature list
    )
    return preprocessor


def get_feature_names(preprocessor):
    """Return human-readable feature names after OHE transformation."""
    feature_names = list(config.NUMERICAL_COLUMNS)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    ohe_names   = cat_encoder.get_feature_names_out(config.CATEGORICAL_COLUMNS).tolist()
    feature_names.extend(ohe_names)
    return feature_names


# ── Fit & Persist ─────────────────────────────────────────────────────────────

def fit_and_save_preprocessor(df, save_path=config.PREPROCESSOR_PATH):
    """Fit the preprocessor on X and save to disk with joblib."""
    preprocessor = create_preprocessor()
    X = df[config.ALL_FEATURE_COLUMNS]
    preprocessor.fit(X)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)
    print(f"Preprocessor saved to {save_path}")
    return preprocessor


def load_preprocessor(save_path=config.PREPROCESSOR_PATH):
    """Load a previously fitted preprocessor from disk."""
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Preprocessor not found at {save_path}")
    return joblib.load(save_path)


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_raw   = load_raw_data()
    df_clean = clean_data(df_raw)
    print(f"Clean dataset: {len(df_clean)} rows")
    prep = fit_and_save_preprocessor(df_clean)
    print("Done.")
