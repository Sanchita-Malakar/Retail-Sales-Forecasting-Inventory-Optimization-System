# RetailPulse AI ⚡
### Retail Sales Forecasting & Inventory Optimization System

> An end-to-end ML-powered retail intelligence platform — XGBoost demand forecasting, automated inventory alerts, trend detection, and an interactive Streamlit dashboard.

---

## Overview

RetailPulse AI is a full-stack data science project that simulates a real-world retail operation across 8 product categories and 40 SKUs, trains an XGBoost forecasting model on 2+ years of synthetic transaction data, and surfaces actionable inventory decisions through a dark-themed, Plotly-powered dashboard.

**What it does in one sentence:** Given historical sales, inventory levels, and weather data, it predicts the next 15 days of demand per SKU, computes reorder points with safety stock, flags dead/overstock items, and visualizes all of this in a live dashboard.

---

## Features

- **Synthetic Data Generation** — 2+ years of realistic retail transactions across 8 categories, 40 SKUs, 3 stores, with seasonality, holiday boosts, weather effects, and trend drift built in
- **XGBoost Demand Forecasting** — Trained on 30+ lag/rolling/calendar/weather features; rolls forward 15 days per product with auto-updating lag windows
- **Inventory Intelligence** — Safety stock (95% service level), reorder point calculation, days-to-stockout, suggested order quantities, and cost estimates
- **Trend & Velocity Engine** — Momentum scoring (vs. 30-day prior), selling velocity (7d/30d), dead stock detection, overstock flagging, and "Hot 🔥" product ranking
- **Seasonal & Weather Analytics** — Category × Month heatmaps, revenue trend lines, Pearson correlation between weather features and sales per category
- **Interactive Dashboard (RetailPulse AI)** — 6-tab Streamlit app with live KPI cards, alert banners, restock planner, and per-product drill-down

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        PIPELINE                            │
│                                                            │
│  data_generation.py  →  data_cleaning.py                  │
│         ↓                      ↓                          │
│   Raw CSVs (4 files)    Cleaned CSVs                       │
│                              ↓                             │
│                   feature_engineering.py                   │
│                   (lag, rolling, velocity,                 │
│                    seasonal, dead-stock flags)             │
│                              ↓                             │
│          ┌───────────────────┼───────────────┐             │
│    forecasting.py   inventory_optimizer.py  intelligence_engine.py │
│    (XGBoost model)  (reorder, safety stock) (trends, velocity)     │
│          └───────────────────┼───────────────┘             │
│                              ↓                             │
│                     streamlit_app.py                       │
│                   (6-tab live dashboard)                   │
└────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
retail-pulse-ai/
│
├── src/
│   ├── data_generation.py       # Step 1 — Synthetic data generator
│   ├── data_cleaning.py         # Step 2A — Null/outlier/type fixes
│   ├── feature_engineering.py   # Step 2B — Lag, rolling, velocity, flags
│   ├── forecasting.py           # Step 3A — XGBoost training + 15-day forecast
│   ├── inventory_optimizer.py   # Step 3B — Reorder, safety stock, alerts
│   └── intelligence_engine.py   # Step 3C — Trends, velocity, seasonal, weather
│
├── app/
│   └── streamlit_app.py         # Step 5 — Interactive dashboard
│
├── main.py                      # Step 4 — Master pipeline runner
│
├── data/
│   ├── raw/                     # Generated CSVs (catalog, weather, sales, inventory)
│   └── processed/               # Cleaned + feature-engineered data
│
├── models/                      # Saved XGBoost model (joblib)
├── outputs/                     # Inventory intelligence, trends, velocity CSVs
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data generation & processing | `pandas`, `numpy` |
| Machine learning | `xgboost`, `scikit-learn` |
| Statistical modeling | `scipy` (normal distribution for safety stock) |
| Model persistence | `joblib` |
| Dashboard | `streamlit`, `plotly` |
| Language | Python 3.9+ |

---

## Data Model

The pipeline generates and processes **4 core CSV files**:

| File | Description | Key Columns |
|---|---|---|
| `product_catalog.csv` | 40 SKUs across 8 categories | `product_id`, `category`, `base_price`, `reorder_lead_days`, `min_order_qty` |
| `sales_transactions.csv` | ~2 years of daily transactions | `date`, `product_id`, `quantity`, `unit_price`, `discount_pct`, `total_spent` |
| `inventory_log.csv` | Daily stock levels per SKU per store | `date`, `product_id`, `store_id`, `opening_stock`, `closing_stock`, `units_restocked` |
| `weather_data.csv` | Daily weather per day | `date`, `temp_c`, `rainfall_mm`, `is_rainy`, `is_cold`, `is_hot` |

**Data characteristics:**
- 8 categories: Electronics, Clothing, Food, Beverages, Furniture, Sports, Beauty, Books
- 40 SKUs with category-appropriate price ranges and demand levels
- 3 stores with independent inventory tracking
- Seasonality patterns: Christmas-peak (Electronics, Beauty), Summer-peak (Clothing, Beverages, Sports), Flat (Food, Furniture, Books)
- Holiday boosts for 9 key dates (New Year, Valentine's, Diwali, Christmas, etc.)
- Weather effects per category (e.g. rainy days boost Electronics, suppress Sports)
- Positive demand trend drift simulating YoY growth

---

## ML Model Details

### XGBoost Forecaster (`forecasting.py`)

**Feature set (30+ features):**

| Group | Features |
|---|---|
| Lag features | `lag_1`, `lag_3`, `lag_7`, `lag_14`, `lag_21`, `lag_28` |
| Rolling stats | `roll_mean_7/14/30`, `roll_std_7/14`, `roll_max_7` |
| Calendar | `dayofweek`, `month`, `quarter`, `dayofyear`, `is_weekend`, `is_month_end`, `is_holiday` |
| Weather | `temp_c`, `rainfall_mm`, `is_rainy`, `is_cold`, `is_hot` |
| Product | `avg_price`, `avg_discount`, `base_price` |

**Training strategy:**
- Global model across all SKUs (sufficient data density for XGBoost)
- Train/test split: last 15 days held out as test set
- Evaluation metrics: MAE and RMSE reported at training time
- Hyperparameters: 300 estimators, LR=0.05, max_depth=6, subsample=0.8

**Inference:**
- Rolls forward 15 days per product from the last known date
- Auto-updates lag windows from rolling history on each step
- Clamps predictions to ≥ 0

### Inventory Intelligence (`inventory_optimizer.py`)

Uses last 90 days of sales to compute per-product stats:

```
Safety Stock   = Z(service_level) × √(lead_time) × σ(daily_demand)
Reorder Point  = μ(daily_demand) × lead_time + safety_stock
Suggested Qty  = max(min_order_qty, reorder_point − current_stock + forecast_demand_15d)
Days to Stockout = current_stock / μ(daily_demand)
```

Default service level: **95%** (configurable via dashboard slider).

**Alert statuses:**

| Status | Condition |
|---|---|
| 🔴 OUT OF STOCK | `current_stock == 0` |
| 🟠 REORDER NOW | `current_stock ≤ reorder_point` |
| ⚫ DEAD STOCK | No sales in last 30 days |
| 🔵 OVERSTOCK | Days-to-stockout > 90 |
| 🟢 OK | None of the above |

---

## Dashboard — RetailPulse AI

Run with:
```bash
streamlit run app/streamlit_app.py
```

The dashboard auto-runs the full pipeline on first launch if output files are missing (~3 minutes).

**6 tabs:**

| Tab | Contents |
|---|---|
| 📦 Storage Overview | Stock vs reorder point bar chart, inventory detail table, days-to-stockout urgency strip |
| 🔮 15-Day Forecast | Top products by forecast volume, per-product historical + forecast line with confidence band |
| 📈 Trend & Velocity | Momentum % bar chart, 7d/30d velocity grouped bars, trend label distribution, revenue/day table |
| 🌤 Season & Weather | Category × Month heatmap, monthly revenue lines, weather correlation bars, sales vs temperature overlay |
| 🏷 Category Analysis | Revenue/unit/transaction share pies & bars, day-of-week patterns, payment method distribution |
| 🚨 Alerts & Restock | Live alert banners by severity, suggested restock schedule with cost estimates, dead stock value chart |

**Sidebar controls:**
- Category filter (applies across all tabs)
- Product selector (for per-SKU drill-down views)
- Service level slider (80–99%)
- Lead time slider (1–30 days)

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/retail-pulse-ai.git
cd retail-pulse-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
# Full 2-year dataset (~3 min)
python main.py

# Quick mode: 10-month window for fast testing
python main.py --quick
```

### 5. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Requirements

```
pandas>=1.5
numpy>=1.23
xgboost>=1.7
scikit-learn>=1.2
scipy>=1.10
joblib>=1.2
streamlit>=1.28
plotly>=5.15
```

---

## Output Files

After running the pipeline, the following files are generated:

```
data/raw/
  product_catalog.csv         — 40 SKUs with metadata
  sales_transactions.csv      — All raw transactions
  inventory_log.csv           — Daily stock per SKU per store
  weather_data.csv            — Daily weather

data/processed/
  cleaned_sales.csv           — Cleaned transactions
  cleaned_inventory.csv       — Cleaned inventory log
  cleaned_weather.csv         — Cleaned weather data
  featured_data.csv           — Master feature table (one row per date × product)

data/forecasts/
  forecast_15days.csv         — XGBoost predictions for next 15 days per SKU

models/
  xgboost_forecaster.pkl      — Saved model + feature list + metrics

outputs/
  inventory_intelligence.csv  — Per-SKU reorder/alert/stockout analysis
  trend_scores.csv            — Momentum, trend labels, category ranks
  velocity_table.csv          — 7d/30d selling rates + revenue/day
  seasonal_summary.csv        — Month × category aggregates
  weather_impact.csv          — Pearson correlations: weather → sales
```

---

## Extending the Project

| Idea | Where to change |
|---|---|
| Add real data | Replace `data_generation.py` with your own CSV loader |
| Change forecast horizon | `forecast_days` parameter in `forecasting.py` |
| Add more SKUs/categories | `CATEGORIES` dict in `data_generation.py` |
| Tune XGBoost | Hyperparameters in `train_forecast_model()` |
| Add email alerts | Hook into `inventory_optimizer.py` alert logic |
| Deploy to cloud | Works with Streamlit Community Cloud out of the box |

---

