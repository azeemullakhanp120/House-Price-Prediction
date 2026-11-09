"""
EDA (Exploratory Data Analysis) Module — Multi-City Edition.
Generates and saves visualisation plots for Bengaluru, Mumbai, Hyderabad, and Chennai.
All plots are saved to app/static/images/ and shown on the dashboard.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ml.preprocess import clean_data, load_raw_data


def generate_eda_plots(city=config.DEFAULT_CITY, save_dir=config.IMAGES_DIR, force=False):
    """
    Generate and save visual figures for the given city dataset.

    Plots produced:
      1. price_dist_<city_id>.png
      2. price_by_location_<city_id>.png
      3. price_vs_area_<city_id>.png
      4. corr_heatmap_<city_id>.png

    Returns dict mapping plot names to absolute file paths.
    """
    canonical_city = config.normalize_city_name(city)
    cfg = config.get_city_config(canonical_city)
    city_id = cfg["id"]

    os.makedirs(save_dir, exist_ok=True)

    dist_filename = f"price_dist_{city_id}.png"
    loc_filename = f"price_by_location_{city_id}.png"
    area_filename = f"price_vs_area_{city_id}.png"
    corr_filename = f"corr_heatmap_{city_id}.png"

    dist_path = os.path.join(save_dir, dist_filename)
    loc_path = os.path.join(save_dir, loc_filename)
    area_path = os.path.join(save_dir, area_filename)
    corr_path = os.path.join(save_dir, corr_filename)

    # If already generated and not forced, return paths directly
    if not force and os.path.exists(dist_path) and os.path.exists(loc_path) and os.path.exists(area_path) and os.path.exists(corr_path):
        return {
            "price_dist": dist_path,
            "price_by_location": loc_path,
            "price_vs_area": area_path,
            "corr_heatmap": corr_path,
        }

    data_path = config.get_dataset_path(canonical_city)
    if not os.path.exists(data_path):
        return {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    df = clean_data(pd.read_csv(data_path))

    # ── 1. Price Distribution ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["price"], bins=35, color="#6366f1", edgecolor="#3730a3", alpha=0.85)
    ax.set_title(f"{canonical_city} Property Price Distribution (Rs. Lakhs)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Price (Rs. Lakhs)", fontsize=11)
    ax.set_ylabel("Number of Properties", fontsize=11)
    plt.tight_layout()
    plt.savefig(dist_path, dpi=120)
    plt.close()

    # ── 2. Average Price by Location (top 20) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    loc_avg = (
        df.groupby("location")["price"]
        .mean()
        .sort_values(ascending=True)
        .tail(20)
    )
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(loc_avg)))
    bars = ax.barh(loc_avg.index, loc_avg.values, color=colors)
    ax.set_title(f"Top Locations by Average Price — {canonical_city}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Average Price (Rs. Lakhs)", fontsize=11)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"Rs. {w:.1f}L", va="center", ha="left", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(loc_path, dpi=120)
    plt.close()

    # ── 3. Price vs Area scatter ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        df["total_sqft"], df["price"],
        c=df["BHK"], cmap="plasma", alpha=0.6, edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("BHK", fontsize=10)
    ax.set_title(f"Price vs Built-Up Area — {canonical_city}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Total sqft", fontsize=11)
    ax.set_ylabel("Price (Rs. Lakhs)", fontsize=11)
    plt.tight_layout()
    plt.savefig(area_path, dpi=120)
    plt.close()

    # ── 4. Correlation Heatmap ────────────────────────────────────────────────
    num_cols = config.NUMERICAL_COLUMNS + [config.TARGET_COLUMN]
    fig, ax = plt.subplots(figsize=(7, 6))
    num_df = df[[c for c in num_cols if c in df.columns]]
    corr = num_df.corr()
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)

    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.6 else "white", fontsize=9)
    ax.set_title(f"Feature Correlations — {canonical_city}", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(corr_path, dpi=120)
    plt.close()

    # Backwards compatibility copies for default city
    if canonical_city == config.DEFAULT_CITY:
        import shutil
        shutil.copy(dist_path, os.path.join(save_dir, "price_dist.png"))
        shutil.copy(loc_path, os.path.join(save_dir, "price_by_location.png"))
        shutil.copy(area_path, os.path.join(save_dir, "price_vs_area.png"))
        shutil.copy(corr_path, os.path.join(save_dir, "corr_heatmap.png"))

    return {
        "price_dist": dist_path,
        "price_by_location": loc_path,
        "price_vs_area": area_path,
        "corr_heatmap": corr_path,
    }


def generate_all_city_plots():
    """Pre-generate EDA plots for all configured cities."""
    for city_name in config.CITIES.keys():
        print(f"Generating EDA plots for {city_name}...")
        generate_eda_plots(city=city_name)
    print("All EDA charts generated.")


if __name__ == "__main__":
    generate_all_city_plots()
