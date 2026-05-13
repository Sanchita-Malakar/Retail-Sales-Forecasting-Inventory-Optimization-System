"""
STEP 4 — Master Pipeline
Run this once to generate all data, train models, and produce all output files.
Usage: python main.py
       python main.py --quick   (3-month window for fast testing)
"""
import os, sys, time, argparse
from datetime import date
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)

def run_pipeline(quick=False):
    t0 = time.time()

    # ── dirs ────────────────────────────────────────────────────────────────
    for d in ["data/raw","data/processed","data/forecasts","models","outputs"]:
        os.makedirs(d, exist_ok=True)

    # ── STEP 1: Generate data ────────────────────────────────────────────────
    banner("STEP 1 — Generating synthetic retail data")
    from src.data_generation import generate_all
    start = date(2023, 6, 1) if quick else date(2022, 1, 1)
    end   = date(2024, 3, 31)
    catalog, weather, sales, inventory = generate_all(start=start, end=end)

    # ── STEP 2A: Clean ──────────────────────────────────────────────────────
    banner("STEP 2A — Cleaning data")
    from src.data_cleaning import clean_sales, clean_inventory, clean_weather
    sales_clean = clean_sales(sales)
    inv_clean   = clean_inventory(inventory)
    weather_c   = clean_weather(weather)
    sales_clean.to_csv("data/processed/cleaned_sales.csv",     index=False)
    inv_clean.to_csv("data/processed/cleaned_inventory.csv",   index=False)
    weather_c.to_csv("data/processed/cleaned_weather.csv",     index=False)
    print(f"  Sales: {len(sales_clean):,} rows | Inventory: {len(inv_clean):,} rows")

    # ── STEP 2B: Feature engineering ────────────────────────────────────────
    banner("STEP 2B — Feature engineering")
    from src.feature_engineering import build_daily_product_features
    features = build_daily_product_features(sales_clean, inv_clean, weather_c, catalog)
    features.to_csv("data/processed/featured_data.csv", index=False)
    print(f"  Feature table: {features.shape[0]:,} rows × {features.shape[1]} cols")

    # ── STEP 3A: Forecasting ─────────────────────────────────────────────────
    banner("STEP 3A — Training XGBoost forecast model")
    from src.forecasting import train_forecast_model, generate_15day_forecast
    model, feat_cols, metrics, importances = train_forecast_model(features)
    forecast_df = generate_15day_forecast(features, model, feat_cols)
    forecast_df.to_csv("data/forecasts/forecast_15days.csv", index=False)
    print(f"  Forecast: {len(forecast_df)} rows | MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f}")
    print(f"  Top-5 features: {list(importances.head(5).index)}")

    # ── STEP 3B: Inventory intelligence ─────────────────────────────────────
    banner("STEP 3B — Inventory intelligence")
    from src.inventory_optimizer import compute_inventory_intelligence
    inv_rec = compute_inventory_intelligence(sales_clean, inv_clean, forecast_df, catalog)
    inv_rec.to_csv("outputs/inventory_intelligence.csv", index=False)
    alerts = inv_rec["alert_status"].value_counts()
    print(f"  Alert summary:\n{alerts.to_string()}")

    # ── STEP 3C: Intelligence engine ─────────────────────────────────────────
    banner("STEP 3C — Trend & velocity engine")
    from src.intelligence_engine import (
        compute_trend_scores, compute_velocity_table,
        compute_seasonal_summary, compute_weather_impact
    )
    trends   = compute_trend_scores(features)
    velocity = compute_velocity_table(features, inv_rec)
    seasonal = compute_seasonal_summary(sales_clean)
    weather_impact = compute_weather_impact(features)

    trends.to_csv("outputs/trend_scores.csv",      index=False)
    velocity.to_csv("outputs/velocity_table.csv",  index=False)
    seasonal.to_csv("outputs/seasonal_summary.csv",index=False)
    weather_impact.to_csv("outputs/weather_impact.csv", index=False)

    hot = trends[trends["trend_label"] == "Hot 🔥"]
    dead = inv_rec[inv_rec["is_dead_stock"] == 1]
    print(f"  Hot products: {len(hot)} | Dead stock: {len(dead)}")

    # ── Done ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE in {elapsed:.0f}s")
    print("  Output files:")
    for root, _, files in os.walk("data"):
        for f in files:
            path = os.path.join(root, f)
            print(f"    {path}  ({os.path.getsize(path)//1024} KB)")
    for root, _, files in os.walk("outputs"):
        for f in files:
            path = os.path.join(root, f)
            print(f"    {path}  ({os.path.getsize(path)//1024} KB)")
    print("\n  Run the dashboard: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use 10-month window for fast testing")
    args = parser.parse_args()
    run_pipeline(quick=args.quick)