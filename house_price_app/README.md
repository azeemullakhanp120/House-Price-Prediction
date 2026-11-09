# 🏡 House Price Prediction & Recommendation System

An end-to-end, production-ready Python Machine Learning & Flask Web Application for predicting residential house prices and delivering content-based property recommendations across **Bengaluru, Mumbai, Hyderabad, and Chennai**.

---

## 🌟 Multi-City Architecture & Synthetic Data Pipeline

All models, analytics, and recommendations are powered by synthetic property datasets with city-calibrated localities, square footage, bedroom/bathroom configurations, and realistic market valuations:

- **Bengaluru**: 20 major tech corridors and residential hubs (Indiranagar, Koramangala, Whitefield, HSR Layout, etc.).
- **Mumbai**: High-density and suburban micro-markets (Bandra, Worli, Powai, Andheri, Borivali, etc.).
- **Hyderabad**: Fast-growing IT corridors and prime zones (Jubilee Hills, Banjara Hills, Gachibowli, Hitec City, etc.).
- **Chennai**: Prime coastal and residential areas (Adyar, Anna Nagar, T Nagar, Besant Nagar, Velachery, etc.).

---

## 📊 Model Evaluation on Held-Out Test Set (80/20 Stratified Split)

Models compared: **Ridge Regression (Baseline)** vs. **Random Forest Regressor (Ensemble Tree)**.

### Overall Performance

| Model | $R^2$ Score | MAE (Lakhs) | RMSE (Lakhs) |
|---|---|---|---|
| **Ridge Regression (Baseline)** | 0.9064 | ₹23.30 L | ₹35.35 L |
| **Random Forest Regressor (Ensemble)** | **0.9745** | **₹14.41 L** | **₹18.43 L** |

### Per-City Accuracy Breakdown (Held-Out Test Set — Random Forest)

| City | Test Samples | $R^2$ Score | MAE (Lakhs) | RMSE (Lakhs) |
|---|---|---|---|---|
| **Mumbai** | 320 | **0.9871** | ₹12.95 L | ₹18.08 L |
| **Chennai** | 320 | **0.9532** | ₹13.71 L | ₹17.10 L |
| **Hyderabad** | 320 | **0.9397** | ₹14.09 L | ₹17.51 L |
| **Bengaluru** | 320 | **0.8954** | ₹16.88 L | ₹20.80 L |

---

## 📁 Directory Structure

```
house_price_app/
├── config.py                  # Multi-city registry (CITIES), paths, and location helpers
├── data/
│   ├── housing.csv            # Unified multi-city dataset (6,400 properties)
│   ├── bengaluru_clean.csv    # 1,600 Bengaluru listings
│   ├── mumbai_clean.csv       # 1,600 Mumbai listings
│   ├── hyderabad_clean.csv    # 1,600 Hyderabad listings
│   ├── chennai_clean.csv      # 1,600 Chennai listings
│   └── users.csv              # User preferences dataset
├── models/
│   ├── price_model_combined.pkl   # Unified multi-city model
│   ├── preprocessor_combined.pkl  # Unified preprocessor
│   ├── price_model_<city>.pkl     # Per-city specialized models
│   ├── preprocessor_<city>.pkl    # Per-city preprocessors
│   └── metrics.json               # Persisted test set evaluation metrics
├── ml/
│   ├── generate_data.py       # Multi-city synthetic dataset generator
│   ├── preprocess.py          # Data cleaning, imputation, and feature pipelines
│   ├── train.py               # Baseline vs RF model training and evaluation
│   ├── recommender.py         # Multi-city cosine-similarity recommender
│   └── eda.py                 # Multi-city visual plot generator
├── app/
│   ├── app.py                 # Flask server routes and REST API endpoints
│   ├── templates/             # HTML templates (base, index, predict, recommend, dashboard, result, about)
│   └── static/                # Light-theme CSS, JS with localStorage persistence, EDA charts
├── tests/
│   └── test_pipeline.py       # Comprehensive unit & integration test suite
├── .gitignore                 # Excludes raw data archives and bytecode
└── README.md
```

---

## 🚀 How to Run and Retrain

```bash
cd house_price_app

# 1. Generate multi-city synthetic datasets
python ml/generate_data.py

# 2. Train baseline and tree-based models and output accuracy metrics
python ml/train.py

# 3. Regenerate EDA visualization plots
python ml/eda.py

# 4. Run the automated test suite
python -m unittest tests/test_pipeline.py

# 5. Start the web application (from house_price_app or workspace root)
python run.py
```
Open `http://127.0.0.1:5000` in your browser.
