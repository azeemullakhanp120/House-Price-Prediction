"""
Model Training & Evaluation Module — Multi-City Housing Prediction System.
Trains models across Bengaluru, Mumbai, Hyderabad, and Chennai.

Pipeline
--------
1. Load clean synthetic city datasets (data/<city>_clean.csv).
2. Clean and preprocess using scikit-learn ColumnTransformer.
3. Compare:
     - Baseline: Ridge Regression
     - Tree-based: Random Forest Regressor
4. Evaluate on held-out test sets (80/20 split).
5. Report MAE, RMSE, and R2:
     - Overall across all test listings
     - Per-city breakdown for Bengaluru, Mumbai, Hyderabad, and Chennai
6. Persist models and preprocessors:
     - models/price_model_<city>.pkl & preprocessor_<city>.pkl
     - models/price_model.pkl & preprocessor.pkl (for backwards compatibility)
     - models/metrics.json
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ml.preprocess import clean_data


def train_and_evaluate_models():
    """
    Train baseline and Random Forest models across all cities,
    evaluate overall and per-city metrics, and persist trained artifacts.
    """
    city_dfs = []
    for city_name in config.CITIES.keys():
        path = config.get_dataset_path(city_name)
        if not os.path.exists(path):
            from ml.generate_data import generate_all_city_datasets
            generate_all_city_datasets()
            break

    for city_name in config.CITIES.keys():
        path = config.get_dataset_path(city_name)
        df_c = clean_data(pd.read_csv(path))
        df_c["city"] = city_name
        city_dfs.append(df_c)

    df = pd.concat(city_dfs, ignore_index=True)
    print(f"\n[*] Loaded {len(df)} listings across {len(city_dfs)} cities:")
    for city, count in df["city"].value_counts().items():
        print(f"      - {city}: {count} properties")

    # Features and Target
    cat_features = ["city", "location"]
    num_features = ["total_sqft", "bath", "BHK"]
    all_features = cat_features + num_features
    target_col = "price"

    X = df[all_features]
    y = df[target_col].values

    # Preprocessor with imputers
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
        ],
        remainder="drop",
    )

    # 80/20 Stratified Train/Test Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, random_state=config.RANDOM_STATE, stratify=df["city"]
    )
    print(f"\n[*] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    print("[*] Fitting preprocessing pipeline ...")
    X_tr_trans = preprocessor.fit_transform(X_train)
    X_te_trans = preprocessor.transform(X_test)

    # ── 1. Baseline Model (Ridge Regression) ─────────────────────────────────
    print("\n[*] Training Baseline Model (Ridge Regression, alpha=10.0) ...")
    baseline_model = Ridge(alpha=10.0)
    baseline_model.fit(X_tr_trans, y_train)
    y_pred_base = baseline_model.predict(X_te_trans)

    base_r2 = float(r2_score(y_test, y_pred_base))
    base_mae = float(mean_absolute_error(y_test, y_pred_base))
    base_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_base)))

    print(f"    Baseline Overall -> R2: {base_r2:.4f} | MAE: Rs. {base_mae:.2f}L | RMSE: Rs. {base_rmse:.2f}L")

    # ── 2. Tree-based Model (Random Forest Regressor) ─────────────────────────
    print("\n[*] Training Tree-based Model (Random Forest Regressor) ...")
    rf_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=16,
        min_samples_split=3,
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    rf_model.fit(X_tr_trans, y_train)
    y_pred_rf = rf_model.predict(X_te_trans)

    rf_r2 = float(r2_score(y_test, y_pred_rf))
    rf_mae = float(mean_absolute_error(y_test, y_pred_rf))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))

    print(f"    Random Forest Overall -> R2: {rf_r2:.4f} | MAE: Rs. {rf_mae:.2f}L | RMSE: Rs. {rf_rmse:.2f}L")

    # ── 3. Per-City Evaluation ────────────────────────────────────────────────
    test_df = df.loc[idx_test].copy()
    test_df["y_true"] = y_test
    test_df["y_pred"] = y_pred_rf

    print("\n" + "=" * 70)
    print("ACCURACY METRICS PER CITY (Held-Out Test Set - Random Forest)")
    print("=" * 70)
    print(f"{'City':<16}{'Samples':<11}{'R2 Score':<13}{'MAE (Lakhs)':<16}{'RMSE (Lakhs)':<15}")
    print("-" * 70)

    per_city_metrics = {}
    for city_name in config.CITIES.keys():
        c_sub = test_df[test_df["city"] == city_name]
        if len(c_sub) == 0:
            continue
        c_r2 = float(r2_score(c_sub["y_true"], c_sub["y_pred"]))
        c_mae = float(mean_absolute_error(c_sub["y_true"], c_sub["y_pred"]))
        c_rmse = float(np.sqrt(mean_squared_error(c_sub["y_true"], c_sub["y_pred"])))
        per_city_metrics[city_name] = {
            "samples": len(c_sub),
            "r2": round(c_r2, 4),
            "mae": round(c_mae, 2),
            "rmse": round(c_rmse, 2),
        }
        print(f"{city_name:<16}{len(c_sub):<11}{c_r2:<13.4f}Rs. {c_mae:<12.2f}Rs. {c_rmse:<11.2f}")
    print("=" * 70)

    metrics = {
        "overall": {
            "baseline": {"r2": round(base_r2, 4), "mae": round(base_mae, 2), "rmse": round(base_rmse, 2)},
            "random_forest": {"r2": round(rf_r2, 4), "mae": round(rf_mae, 2), "rmse": round(rf_rmse, 2)},
        },
        "per_city": per_city_metrics,
    }

    # ── 4. Train Per-City Specialized Models ──────────────────────────────────
    print("\n[*] Training Per-City Specialized Models for Maximum Precision ...")
    for city_name in config.CITIES.keys():
        city_csv = config.get_dataset_path(city_name)
        if not os.path.exists(city_csv):
            continue
        c_df = clean_data(pd.read_csv(city_csv))
        c_X = c_df[["location", "total_sqft", "BHK", "bath"]]
        c_y = c_df["price"].values

        c_num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        c_cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ])
        c_prep = ColumnTransformer([
            ("num", c_num_pipe, ["total_sqft", "BHK", "bath"]),
            ("cat", c_cat_pipe, ["location"]),
        ])
        c_X_trans = c_prep.fit_transform(c_X)
        c_model = RandomForestRegressor(
            n_estimators=100, max_depth=16, min_samples_split=3,
            random_state=config.RANDOM_STATE, n_jobs=1
        )
        c_model.fit(c_X_trans, c_y)

        # Save city model and preprocessor
        cfg = config.get_city_config(city_name)
        c_model_path = os.path.join(config.MODELS_DIR, cfg["model_file"])
        c_prep_path = os.path.join(config.MODELS_DIR, cfg["preprocessor_file"])
        joblib.dump(c_model, c_model_path)
        joblib.dump(c_prep, c_prep_path)
        print(f"      Saved {city_name} specialized model -> {c_model_path}")

    # Save combined model and preprocessor
    combined_model_path = os.path.join(config.MODELS_DIR, "price_model_combined.pkl")
    combined_prep_path = os.path.join(config.MODELS_DIR, "preprocessor_combined.pkl")
    joblib.dump(rf_model, combined_model_path)
    joblib.dump(preprocessor, combined_prep_path)

    # Save legacy defaults for Bengaluru
    joblib.dump(rf_model, os.path.join(config.MODELS_DIR, "price_model.pkl"))
    joblib.dump(preprocessor, os.path.join(config.MODELS_DIR, "preprocessor.pkl"))

    # Save metrics JSON
    metrics_path = os.path.join(config.MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[*] Metrics saved to {metrics_path}")

    return metrics


def load_trained_artifacts(city_name=config.DEFAULT_CITY):
    """
    Load the saved model and preprocessor for the specified city.
    If artefacts are missing, trains them on demand.
    """
    canonical_city = config.normalize_city_name(city_name)
    cfg = config.get_city_config(canonical_city)

    model_path = os.path.join(config.MODELS_DIR, cfg["model_file"])
    prep_path = os.path.join(config.MODELS_DIR, cfg["preprocessor_file"])

    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        train_and_evaluate_models()

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    return model, preprocessor


if __name__ == "__main__":
    train_and_evaluate_models()
