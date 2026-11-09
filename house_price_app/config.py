"""
Configuration file for House Price Prediction & Recommendation System.
Supports multiple cities: Bengaluru, Mumbai, Hyderabad, and Chennai.
All paths, column names, model parameters, and city settings are configured here.
"""
import os
import pandas as pd

# ── Base Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_PATH = os.path.join(DATA_DIR, "users.csv")
DB_PATH = os.path.join(DATA_DIR, "users.db")

# Model Paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Static Assets Path
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# ── Multi-City Registry ────────────────────────────────────────────────────────
# Central configurable registry for all supported cities.
# Adding a new city in the future only requires adding an entry here.
CITIES = {
    "Bengaluru": {
        "id": "bengaluru",
        "name": "Bengaluru",
        "short_code": "BLR",
        "state": "Karnataka",
        "coordinates": {"lat": 12.9716, "lng": 77.5946},
        "currency": "₹",
        "data_file": "bengaluru_clean.csv",
        "model_file": "price_model_bengaluru.pkl",
        "preprocessor_file": "preprocessor_bengaluru.pkl",
        "available": True,
        "description": "Silicon Valley of India — Tech corridors, premium apartments & villas.",
    },
    "Mumbai": {
        "id": "mumbai",
        "name": "Mumbai",
        "short_code": "BOM",
        "state": "Maharashtra",
        "coordinates": {"lat": 19.0760, "lng": 72.8777},
        "currency": "₹",
        "data_file": "mumbai_clean.csv",
        "model_file": "price_model_mumbai.pkl",
        "preprocessor_file": "preprocessor_mumbai.pkl",
        "available": True,
        "description": "Financial Capital of India — High-density, sea-facing & suburban prime real estate.",
    },
    "Hyderabad": {
        "id": "hyderabad",
        "name": "Hyderabad",
        "short_code": "HYD",
        "state": "Telangana",
        "coordinates": {"lat": 17.3850, "lng": 78.4867},
        "currency": "₹",
        "data_file": "hyderabad_clean.csv",
        "model_file": "price_model_hyderabad.pkl",
        "preprocessor_file": "preprocessor_hyderabad.pkl",
        "available": True,
        "description": "Cyberabad & Pearl City — Rapidly expanding IT hubs & luxury gated communities.",
    },
    "Chennai": {
        "id": "chennai",
        "name": "Chennai",
        "short_code": "MAA",
        "state": "Tamil Nadu",
        "coordinates": {"lat": 13.0827, "lng": 80.2707},
        "currency": "₹",
        "data_file": "chennai_clean.csv",
        "model_file": "price_model_chennai.pkl",
        "preprocessor_file": "preprocessor_chennai.pkl",
        "available": True,
        "description": "Detroit of Asia — Cultural hub, coastal residences, and thriving industrial corridors.",
    },
}

DEFAULT_CITY = "Bengaluru"


def normalize_city_name(city_input: str) -> str:
    """Resolve user or URL input string to canonical city name (case-insensitive)."""
    if not city_input:
        return DEFAULT_CITY
    c_lower = city_input.strip().lower()
    for canonical_name, data in CITIES.items():
        if c_lower in [canonical_name.lower(), data["id"].lower(), data["short_code"].lower()]:
            return canonical_name
    return DEFAULT_CITY


def get_city_config(city_name: str = DEFAULT_CITY) -> dict:
    """Return dictionary configuration for the specified city."""
    canonical = normalize_city_name(city_name)
    return CITIES.get(canonical, CITIES[DEFAULT_CITY])


def get_dataset_path(city_name: str = DEFAULT_CITY) -> str:
    """Return absolute file path to the cleaned dataset for the given city."""
    cfg = get_city_config(city_name)
    return os.path.join(DATA_DIR, cfg["data_file"])


def get_model_path(city_name: str = DEFAULT_CITY) -> str:
    """Return path to the trained model for the given city."""
    cfg = get_city_config(city_name)
    primary = os.path.join(MODELS_DIR, cfg["model_file"])
    if os.path.exists(primary):
        return primary
    # Backwards compatibility fallback for Bengaluru
    legacy = os.path.join(MODELS_DIR, "price_model.pkl")
    return legacy if os.path.exists(legacy) else primary


def get_preprocessor_path(city_name: str = DEFAULT_CITY) -> str:
    """Return path to the fitted preprocessor for the given city."""
    cfg = get_city_config(city_name)
    primary = os.path.join(MODELS_DIR, cfg["preprocessor_file"])
    if os.path.exists(primary):
        return primary
    # Backwards compatibility fallback for Bengaluru
    legacy = os.path.join(MODELS_DIR, "preprocessor.pkl")
    return legacy if os.path.exists(legacy) else primary


# Cache for dynamic locations per city
_LOCATIONS_CACHE = {}


def get_city_locations(city_name: str = DEFAULT_CITY) -> list:
    """
    Return sorted list of available localities for the specified city.
    Extracts directly from the clean dataset CSV and caches results.
    """
    canonical = normalize_city_name(city_name)
    if canonical in _LOCATIONS_CACHE:
        return _LOCATIONS_CACHE[canonical]

    data_file = get_dataset_path(canonical)
    if os.path.exists(data_file):
        try:
            df = pd.read_csv(data_file, usecols=["location"])
            locs = sorted([str(loc).strip() for loc in df["location"].dropna().unique() if str(loc).strip()])
            _LOCATIONS_CACHE[canonical] = locs
            return locs
        except Exception:
            pass

    # Fallback to hardcoded defaults if dataset not yet present
    fallback = {
        "Bengaluru": sorted(BENGALURU_LOCATIONS_FALLBACK),
        "Mumbai": ["Andheri", "Bandra", "Borivali", "Goregaon", "Kurla", "Malad", "Navi Mumbai", "Powai", "Thane", "Worli"],
        "Hyderabad": ["Banjara Hills", "Begumpet", "Gachibowli", "Jubilee Hills", "Kondapur", "Kukatpally", "Madhapur", "Miyapur", "Secunderabad", "Uppal"],
        "Chennai": ["Adyar", "Anna Nagar", "Chrompet", "Karapakkam", "KK Nagar", "T Nagar", "Velachery"],
    }
    locs = fallback.get(canonical, [])
    _LOCATIONS_CACHE[canonical] = locs
    return locs


# ── Backwards-Compatibility Aliases ───────────────────────────────────────────
DATASET_PATH = get_dataset_path(DEFAULT_CITY)
MODEL_PATH = get_model_path(DEFAULT_CITY)
PREPROCESSOR_PATH = get_preprocessor_path(DEFAULT_CITY)

# Fallback Bengaluru locations
BENGALURU_LOCATIONS_FALLBACK = [
    "1st Block Jayanagar", "1st Phase JP Nagar", "2nd Phase Judicial Layout", "2nd Stage Nagarbhavi",
    "5th Block Hbr Layout", "5th Phase JP Nagar", "6th Phase JP Nagar", "7th Phase JP Nagar",
    "8th Phase JP Nagar", "9th Phase JP Nagar", "AECS Layout", "Abbigere", "Akshaya Nagar",
    "Ambalipura", "Ambedkar Nagar", "Amruth Nagar", "Anandapura", "Ananth Nagar", "Anjanapura",
    "Ardendale", "Arekere", "Attibele", "BEML Layout", "BTM 2nd Stage", "BTM Layout", "Babusapalpalya",
    "Badavala Nagar", "Balagere", "Banashankari", "Banashankari Stage II", "Banashankari Stage III",
    "Banashankari Stage V", "Banashankari Stage VI", "Banaswadi", "Banjara Layout", "Bannerghatta",
    "Bannerghatta Road", "Basavangudi", "Basaveshwara Nagar", "Battarahalli", "Begur", "Begur Road",
    "Bellandur", "Benson Town", "Bharathi Nagar", "Bhoganhalli", "Billekahalli", "Binny Pete",
    "Bisuvanahalli", "Bommanahalli", "Bommasandra", "Bommasandra Industrial Area", "Bommenahalli",
    "Brooke Bond", "Budigere", "CV Raman Nagar", "Chamrajpet", "Chandapura", "Channasandra",
    "Chikka Tirupathi", "Chikkabanavara", "Chikkalasandra", "Choodasandra", "Cooke Town",
    "Cox Town", "Cunningham Road", "Dasanapura", "Dasarahalli", "Devanahalli", "Devarachikkanahalli",
    "Dodda Nekkundi", "Doddaballapur", "Doddakallasandra", "Doddathoguru", "Domlur", "Dommasandra",
    "EPIP Zone", "Electronic City", "Electronic City Phase II", "Electronics City Phase 1",
    "Frazer Town", "GM Palaya", "Garudachar Palya", "Giri Nagar", "Gollarapalya Hosahalli",
    "Gottigere", "Green Glen Layout", "Gubbalala", "Gunjur", "HAL 2nd Stage", "HBR Layout",
    "HRBR Layout", "HOSA Road", "Harlur", "Harrows Road", "Hebbal", "Hebbal Kempapura",
    "Hegde Nagar", "Hennur", "Hennur Road", "Hoodi", "Horamavu Agara", "Horamavu Banaswadi",
    "Hormavu", "Hosa Road", "Hosakerehalli", "Hoskote", "Hosur Road", "Hulimavu",
    "ISRO Layout", "ITPL", "Iblur Village", "Indiranagar", "JP Nagar", "Jakkur",
    "Jalahalli", "Jalahalli East", "Jigani", "Judicial Layout", "KR Puram", "Kadubeesanahalli",
    "Kadugodi", "Kaggadasapura", "Kaggalipura", "Kaikondrahalli", "Kalena Agrahara", "Kalyan Nagar",
    "Kambipura", "Kammanahalli", "Kammasandra", "Kanakapura", "Kanakpura Road", "Kannamangala",
    "Karuna Nagar", "Kasavanhalli", "Kasturi Nagar", "Kathriguppe", "Kaval Byrasandra",
    "Kenchenahalli", "Kengeri", "Kengeri Satellite Town", "Kereguddadahalli", "Kogilu",
    "Konanakunte", "Koramangala", "Kothannur", "Kothanur", "Kudlu", "Kudlu Gate",
    "Kumaraswamy Layout", "Kundalahalli", "LB Shastri Nagar", "Laggere", "Lakshminarayana Pura",
    "Lingadheeranahalli", "Magadi Road", "Mahadevpura", "Mahalakshmi Layout", "Mallasandra",
    "Malleshpalya", "Malleshwaram", "Marathahalli", "Margondanahalli", "Marsur", "Mico Layout",
    "Munnekollal", "Murugeshpalya", "Mysore Road", "NGR Layout", "NRI Layout", "Nagarbhavi",
    "Nagasandra", "Nagavara", "Nagreshwara Nagenahalli", "Neeladri Nagar", "Nehru Nagar",
    "OMBR Layout", "Old Airport Road", "Old Madras Road", "Padmanabhanagar", "Pai Layout",
    "Panathur", "Parappana Agrahara", "Pattandur Agrahara", "Poorna Pragna Layout",
    "Prithvi Layout", "R.T. Nagar", "Rachenahalli", "Raja Rajeshwari Nagar", "Rajaji Nagar",
    "Rajiv Gandhi Nagar", "Ramagondanahalli", "Ramamurthy Nagar", "Rayasandra", "Sadashiv Nagar",
    "Sahakara Nagar", "Sanjay nagar", "Sarakki Nagar", "Sarjapur", "Sarjapur  Road",
    "Sarjapura - Attibele Road", "Sector 2 HSR Layout", "Sector 7 HSR Layout", "Seegehalli",
    "Shampura", "Shivaji Nagar", "Singasandra", "Somasundara Palya", "Sompura", "Sonnenahalli",
    "Subramanyapura", "Sultan Palaya", "TC Palaya", "Talaghattapura", "Thanisandra",
    "Thigalarapalya", "Thubarahalli", "Tindlu", "Tumkur Road", "Ulsoor", "Uttarahalli",
    "Varthur", "Varthur Road", "Vasanthapura", "Vidyaranyapura", "Vijayanagar",
    "Vishveshwarya Layout", "Vishwapriya Nagar", "Vittasandra", "Whitefield",
    "Yelachenahalli", "Yelahanka", "Yelahanka New Town", "Yelenahalli", "Yeshwanthpur", "Other"
]

BENGALURU_LOCATIONS = BENGALURU_LOCATIONS_FALLBACK
LOCATION_OPTIONS = sorted(BENGALURU_LOCATIONS)

# ── Column Definitions ─────────────────────────────────────────────────────────
TARGET_COLUMN = "price"
CATEGORICAL_COLUMNS = ["location"]
NUMERICAL_COLUMNS = ["total_sqft", "bath", "BHK"]
ALL_FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS

# ── Categorical Options ────────────────────────────────────────────────────────
FURNISHING_OPTIONS = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]

# ── Model Hyperparameters & Training Settings ──────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ── Recommendation Parameters ──────────────────────────────────────────────────
TOP_N_RECOMMENDATIONS = 8
VALUE_SCORE_THRESHOLDS = {
    "great_deal": 1.05,
    "fair_deal": 0.95
}

# ── Currency Formatting ────────────────────────────────────────────────────────
CURRENCY_SYMBOL = "₹"
