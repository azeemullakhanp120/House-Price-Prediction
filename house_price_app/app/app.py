"""
Flask Web Application — Multi-City House Price Prediction & Recommendation System.
Supports Bengaluru, Mumbai, Hyderabad, and Chennai.
"""
import os
import sys

# ── Ensure project root is on sys.path ───────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
APP_DIR = os.path.dirname(THIS_FILE)
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, flash, redirect, url_for, session

import config
from ml.recommender import HouseRecommender
from ml.train import load_trained_artifacts
from ml.eda import generate_eda_plots

app = Flask(
    __name__,
    template_folder=os.path.join(APP_DIR, "templates"),
    static_folder=os.path.join(APP_DIR, "static"),
)
app.secret_key = "estate_iq_multi_city_secret_key_change_in_prod"

# ── Multi-City Recommenders & Models Cache ─────────────────────────────────────
_RECOMMENDERS_CACHE = {}
_MODELS_CACHE = {}


def get_recommender(city_name: str) -> HouseRecommender:
    """Return cached HouseRecommender instance for the given city."""
    canonical = config.normalize_city_name(city_name)
    if canonical not in _RECOMMENDERS_CACHE:
        try:
            _RECOMMENDERS_CACHE[canonical] = HouseRecommender(city=canonical)
        except Exception as exc:
            print(f"[WARN] Could not initialize recommender for {canonical}: {exc}")
            return None
    return _RECOMMENDERS_CACHE[canonical]


def get_model_and_preprocessor(city_name: str):
    """Return (model, preprocessor) tuple for the given city."""
    canonical = config.normalize_city_name(city_name)
    if canonical not in _MODELS_CACHE:
        try:
            m, p = load_trained_artifacts(canonical)
            _MODELS_CACHE[canonical] = (m, p)
        except Exception as exc:
            print(f"[WARN] Could not load model artefacts for {canonical}: {exc}")
            return None, None
    return _MODELS_CACHE[canonical]


# Pre-load default city at startup
try:
    get_model_and_preprocessor(config.DEFAULT_CITY)
    get_recommender(config.DEFAULT_CITY)
    print(f"[OK] Flask: Pre-loaded {config.DEFAULT_CITY} artefacts.")
except Exception as exc:
    print(f"[WARN] Startup pre-load warning: {exc}")


def resolve_active_city() -> str:
    """Resolve active city from query params, session, or default."""
    requested_city = request.args.get("city")
    if requested_city:
        canonical = config.normalize_city_name(requested_city)
        session["current_city"] = canonical
        return canonical
    session_city = session.get("current_city")
    if session_city:
        return config.normalize_city_name(session_city)
    return config.DEFAULT_CITY


@app.context_processor
def inject_city_context():
    """Inject active city configuration and all cities list into all templates."""
    active_city_name = resolve_active_city()
    active_city_cfg = config.get_city_config(active_city_name)
    all_cities = list(config.CITIES.values())
    return {
        "current_city": active_city_cfg,
        "active_city_name": active_city_name,
        "cities": all_cities,
        "default_city": config.DEFAULT_CITY,
    }


# ── Utility ───────────────────────────────────────────────────────────────────

def format_inr(price_lakhs: float) -> str:
    """Format a numeric price in Lakhs to '₹ X.XX Crore' or '₹ X.XX Lakhs'."""
    if price_lakhs >= 100.0:
        return f"₹{price_lakhs / 100.0:.2f} Crore"
    return f"₹{price_lakhs:.2f} Lakhs"


# ── Web UI Routes ─────────────────────────────────────────────────────────────

@app.route("/set-city/<city_name>")
def set_city(city_name: str):
    """Switch active city and redirect to referring page."""
    canonical = config.normalize_city_name(city_name)
    session["current_city"] = canonical
    referrer = request.referrer or url_for("home")
    # Preserve page path while replacing or adding city param
    return redirect(referrer)


@app.route("/")
def home():
    """Homepage — overview and stats for the active city."""
    active_city = resolve_active_city()
    rec = get_recommender(active_city)
    city_locations = config.get_city_locations(active_city)

    # Compute quick stats
    total_properties = len(rec.df) if rec else 0
    avg_price = round(float(rec.df["price"].mean()), 1) if rec is not None and not rec.df.empty else 0

    return render_template(
        "index.html",
        total_properties=total_properties,
        avg_price=avg_price,
        locations_count=len(city_locations),
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET  – render prediction form with active city's locations.
    POST – process form, call the city's ML model, render result.html.
    """
    active_city = resolve_active_city()

    if request.method == "POST":
        try:
            city = config.normalize_city_name(request.form.get("city", active_city))
            session["current_city"] = city
            location = request.form.get("location", "").strip()
            total_sqft = float(request.form.get("total_sqft", 1000))
            bath = int(request.form.get("bath", 2))
            BHK = int(request.form.get("BHK", 2))

            model, preprocessor = get_model_and_preprocessor(city)
            if model is None or preprocessor is None:
                raise ValueError(f"No prediction model available for {city}.")

            input_df = pd.DataFrame([{
                "location": location,
                "total_sqft": total_sqft,
                "bath": bath,
                "BHK": BHK,
            }])

            X_trans = preprocessor.transform(input_df)
            pred_lakhs = float(np.round(model.predict(X_trans)[0], 2))

            formatted = format_inr(pred_lakhs)
            conf_low = round(pred_lakhs * 0.95, 2)
            conf_high = round(pred_lakhs * 1.05, 2)

            rec = get_recommender(city)
            recs = rec.recommend_properties(
                location=location, min_bedrooms=BHK, top_n=3
            ) if rec else []

            return render_template(
                "result.html",
                price_lakhs=pred_lakhs,
                formatted_price=formatted,
                confidence_low=conf_low,
                confidence_high=conf_high,
                input_data=input_df.iloc[0].to_dict(),
                recommendations=recs,
                city=city,
            )

        except Exception as exc:
            flash(f"Prediction error: {exc}", "danger")
            return redirect(url_for("predict", city=active_city))

    # GET: render form
    locations = config.get_city_locations(active_city)
    return render_template(
        "predict.html",
        locations=locations,
        furnishings=config.FURNISHING_OPTIONS,
        city=active_city,
    )


@app.route("/recommend")
def recommend():
    """
    Smart Recommender page for the active city.
    """
    active_city = resolve_active_city()
    location = request.args.get("location", "")
    min_budget = request.args.get("min_budget", type=float)
    max_budget = request.args.get("max_budget", type=float)
    min_bedrooms = request.args.get("min_bedrooms", type=int)
    property_id = request.args.get("property_id", "")

    rec = get_recommender(active_city)
    locations = config.get_city_locations(active_city)

    if property_id and rec:
        recommendations = rec.find_similar_properties(
            property_id, top_n=config.TOP_N_RECOMMENDATIONS
        )
    elif rec:
        recommendations = rec.recommend_properties(
            location=location or None,
            min_budget=min_budget,
            max_budget=max_budget,
            min_bedrooms=min_bedrooms,
            top_n=config.TOP_N_RECOMMENDATIONS,
        )
    else:
        recommendations = []

    return render_template(
        "recommend.html",
        recommendations=recommendations,
        locations=locations,
        selected_location=location,
        selected_min_budget=min_budget,
        selected_max_budget=max_budget,
        selected_min_bedrooms=min_bedrooms,
        city=active_city,
    )


@app.route("/dashboard")
def dashboard():
    """EDA charts dashboard for the active city."""
    active_city = resolve_active_city()
    # Ensure plots exist for this city
    generate_eda_plots(city=active_city)
    cfg = config.get_city_config(active_city)
    city_id = cfg["id"]

    return render_template(
        "dashboard.html",
        city=active_city,
        city_id=city_id,
    )


@app.route("/about")
def about():
    """Methodology & API docs page."""
    return render_template("about.html")


# ── REST API Endpoints ────────────────────────────────────────────────────────

@app.route("/api/cities", methods=["GET"])
def api_cities():
    """JSON API — list all configured cities and metadata."""
    return jsonify({
        "status": "success",
        "default_city": config.DEFAULT_CITY,
        "cities": list(config.CITIES.values()),
    }), 200


@app.route("/api/locations", methods=["GET"])
def api_locations():
    """
    JSON API — get localities for a specific city.
    Query param: ?city=Mumbai
    """
    city_name = request.args.get("city", resolve_active_city())
    canonical = config.normalize_city_name(city_name)
    locations = config.get_city_locations(canonical)
    return jsonify({
        "status": "success",
        "city": canonical,
        "count": len(locations),
        "locations": locations,
    }), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API — multi-city house price prediction.
    Request body (JSON):
        {
          "city": "Mumbai",
          "location": "Bandra",
          "total_sqft": 1200.0,
          "bath": 2,
          "BHK": 2
        }
    """
    try:
        data = request.get_json(force=True) or {}
        city_name = data.get("city", config.DEFAULT_CITY)
        canonical = config.normalize_city_name(city_name)

        for field in config.ALL_FEATURE_COLUMNS:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing field: {field}"}), 400

        model, preprocessor = get_model_and_preprocessor(canonical)
        if model is None or preprocessor is None:
            return jsonify({"status": "error", "message": f"Model not available for city: {canonical}"}), 404

        input_df = pd.DataFrame([{
            "location": data["location"],
            "total_sqft": float(data["total_sqft"]),
            "bath": int(data["bath"]),
            "BHK": int(data["BHK"]),
        }])

        X_trans = preprocessor.transform(input_df)
        pred_lakhs = float(np.round(model.predict(X_trans)[0], 2))

        return jsonify({
            "status": "success",
            "city": canonical,
            "predicted_price_lakhs": pred_lakhs,
            "formatted_price": format_inr(pred_lakhs),
            "confidence_interval": {
                "low": round(pred_lakhs * 0.95, 2),
                "high": round(pred_lakhs * 1.05, 2),
            },
        }), 200

    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    JSON API — property recommendations for a specific city.
    Request body (JSON):
        {
          "city": "Hyderabad",
          "location": "Gachibowli",
          "min_budget": 50,
          "max_budget": 150,
          "min_bedrooms": 2,
          "top_n": 5
        }
    """
    try:
        data = request.get_json(force=True) or {}
        city_name = data.get("city", config.DEFAULT_CITY)
        canonical = config.normalize_city_name(city_name)

        rec = get_recommender(canonical)
        if not rec:
            return jsonify({"status": "error", "message": f"Recommender not available for {canonical}"}), 404

        recs = rec.recommend_properties(
            location=data.get("location"),
            min_budget=data.get("min_budget"),
            max_budget=data.get("max_budget"),
            min_bedrooms=data.get("min_bedrooms"),
            top_n=data.get("top_n", config.TOP_N_RECOMMENDATIONS),
        )
        return jsonify({
            "status": "success",
            "city": canonical,
            "count": len(recs),
            "recommendations": recs,
        }), 200

    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """JSON API — dataset summary statistics for the chosen city."""
    try:
        city_name = request.args.get("city", resolve_active_city())
        canonical = config.normalize_city_name(city_name)
        rec = get_recommender(canonical)
        df = rec.df if rec else pd.DataFrame()
        price_col = config.TARGET_COLUMN

        stats = {
            "city": canonical,
            "total_properties": int(len(df)),
            "locations_count": int(df["location"].nunique()) if "location" in df else 0,
            "avg_price_lakhs": float(np.round(df[price_col].mean(), 2)) if price_col in df and not df.empty else 0,
            "min_price_lakhs": float(df[price_col].min()) if price_col in df and not df.empty else 0,
            "max_price_lakhs": float(df[price_col].max()) if price_col in df and not df.empty else 0,
        }
        return jsonify({"status": "success", "stats": stats}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── Entry-point ───────────────────────────────────────────────────────────────

def _open_browser_when_ready(url):
    import time
    time.sleep(1.2)
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    target_url = f"http://127.0.0.1:{port}"

    import threading
    threading.Thread(target=_open_browser_when_ready, args=[target_url], daemon=True).start()

    print("\n" + "=" * 65)
    print("  EstateIQ House Price Prediction Web App")
    print(f"  --> Status: LIVE & RUNNING")
    print(f"  --> Opening in browser automatically: {target_url}")
    print(f"  --> Or open manually: {target_url}")
    print("  --> (Keep this terminal window OPEN while browsing)")
    print("  --> Press CTRL+C to stop the server")
    print("=" * 65 + "\n")
    try:
        app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
    except OSError as e:
        if "10048" in str(e) or "already in use" in str(e).lower():
            fallback_port = 5001
            fallback_url = f"http://127.0.0.1:{fallback_port}"
            print(f"[!] Port {port} was already in use, switching to: {fallback_url}")
            threading.Thread(target=_open_browser_when_ready, args=[fallback_url], daemon=True).start()
            app.run(host="127.0.0.1", port=fallback_port, debug=True, use_reloader=False)
        else:
            raise e
