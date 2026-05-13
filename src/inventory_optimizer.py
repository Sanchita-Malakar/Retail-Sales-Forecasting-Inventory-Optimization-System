"""
STEP 3B — Inventory Intelligence
Computes: reorder points, safety stock, overstock flags,
dead stock alerts, days-to-stockout, and restock schedule.
"""
import pandas as pd
import numpy as np
from scipy.stats import norm


def compute_inventory_intelligence(
    historical_sales: pd.DataFrame,
    inventory_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    service_level: float = 0.95,
) -> pd.DataFrame:
    """
    Returns one row per product with full inventory intelligence metrics.
    """
    historical_sales = historical_sales.copy()
    inventory_df     = inventory_df.copy()
    historical_sales["date"] = pd.to_datetime(historical_sales["date"])
    inventory_df["date"]     = pd.to_datetime(inventory_df["date"])

    # ── 1. Demand stats from last 90 days ─────────────────────────────────────
    cutoff = historical_sales["date"].max() - pd.Timedelta(days=90)
    recent = (
        historical_sales[historical_sales["date"] > cutoff]
        .groupby("product_id")["quantity"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "daily_mean_demand", "std": "daily_std_demand"})
    )
    recent["daily_std_demand"] = recent["daily_std_demand"].fillna(1)

    # ── 2. Current stock (latest date per product, summed across stores) ──────
    latest_inv = (
        inventory_df.sort_values("date")
        .groupby("product_id")
        .last()
        .reset_index()[["product_id", "closing_stock"]]
        .rename(columns={"closing_stock": "current_stock"})
    )
    # Sum across stores for multi-store
    current_stock_by_store = (
        inventory_df[inventory_df["date"] == inventory_df["date"].max()]
        .groupby("product_id")["closing_stock"]
        .sum()
        .reset_index()
        .rename(columns={"closing_stock": "current_stock"})
    )

    # ── 3. Last restock date ──────────────────────────────────────────────────
    last_restock = (
        inventory_df[inventory_df["was_restocked"] == 1]
        .groupby("product_id")["date"]
        .max()
        .reset_index()
        .rename(columns={"date": "last_restock_date"})
    )

    # ── 4. Last sold date ─────────────────────────────────────────────────────
    last_sold = (
        historical_sales.groupby("product_id")["date"]
        .max()
        .reset_index()
        .rename(columns={"date": "last_sold_date"})
    )

    # ── 5. 15-day forecast demand per product ─────────────────────────────────
    fc_demand = (
        forecast_df.groupby("product_id")["forecast_sales"]
        .sum()
        .reset_index()
        .rename(columns={"forecast_sales": "forecast_demand_15d"})
    )

    # ── 6. Merge everything ───────────────────────────────────────────────────
    rec = recent.copy()
    rec = rec.merge(current_stock_by_store, on="product_id", how="left")
    rec = rec.merge(last_restock,           on="product_id", how="left")
    rec = rec.merge(last_sold,              on="product_id", how="left")
    rec = rec.merge(fc_demand,              on="product_id", how="left")
    rec = rec.merge(
        catalog_df[["product_id","product_name","category","base_price",
                    "reorder_lead_days","min_order_qty"]],
        on="product_id", how="left"
    )

    rec["current_stock"]       = rec["current_stock"].fillna(0)
    rec["forecast_demand_15d"] = rec["forecast_demand_15d"].fillna(
        rec["daily_mean_demand"] * 15
    )

    # ── 7. Safety stock & reorder point ──────────────────────────────────────
    z = norm.ppf(service_level)
    rec["safety_stock"] = (
        z * np.sqrt(rec["reorder_lead_days"]) * rec["daily_std_demand"]
    ).round(1)
    rec["reorder_point"] = (
        rec["daily_mean_demand"] * rec["reorder_lead_days"] + rec["safety_stock"]
    ).round(1)

    # ── 8. Reorder quantity ───────────────────────────────────────────────────
    rec["suggested_reorder_qty"] = np.maximum(
        rec["min_order_qty"],
        (rec["reorder_point"] - rec["current_stock"] + rec["forecast_demand_15d"])
    ).clip(lower=0).round(0).astype(int)

    # ── 9. Days to stockout ───────────────────────────────────────────────────
    rec["days_to_stockout"] = np.where(
        rec["daily_mean_demand"] > 0,
        (rec["current_stock"] / rec["daily_mean_demand"]).round(1),
        999
    )

    # ── 10. Status flags ──────────────────────────────────────────────────────
    max_date = historical_sales["date"].max()
    rec["last_sold_date"]   = pd.to_datetime(rec["last_sold_date"])
    rec["last_restock_date"] = pd.to_datetime(rec["last_restock_date"])
    rec["days_since_sold"]  = (max_date - rec["last_sold_date"]).dt.days.fillna(999)
    rec["days_since_restock"] = (max_date - rec["last_restock_date"]).dt.days.fillna(999)

    rec["needs_reorder"]    = (rec["current_stock"] <= rec["reorder_point"]).astype(int)
    rec["is_out_of_stock"]  = (rec["current_stock"] == 0).astype(int)
    rec["is_overstock"]     = (rec["days_to_stockout"] > 90).astype(int)
    rec["is_dead_stock"]    = (rec["days_since_sold"] > 30).astype(int)
    rec["service_level"]    = service_level

    # ── 11. Alert priority ────────────────────────────────────────────────────
    def priority(row):
        if row["is_out_of_stock"]:  return "🔴 OUT OF STOCK"
        if row["needs_reorder"]:    return "🟠 REORDER NOW"
        if row["is_dead_stock"]:    return "⚫ DEAD STOCK"
        if row["is_overstock"]:     return "🔵 OVERSTOCK"
        return "🟢 OK"
    rec["alert_status"] = rec.apply(priority, axis=1)

    cols = [
        "product_id", "product_name", "category", "base_price",
        "current_stock", "daily_mean_demand", "daily_std_demand",
        "safety_stock", "reorder_point", "suggested_reorder_qty",
        "forecast_demand_15d", "days_to_stockout",
        "last_sold_date", "days_since_sold",
        "last_restock_date", "days_since_restock",
        "needs_reorder", "is_out_of_stock", "is_overstock", "is_dead_stock",
        "alert_status", "service_level", "reorder_lead_days", "min_order_qty",
    ]
    return rec[cols].reset_index(drop=True)


if __name__ == "__main__":
    sales   = pd.read_csv("data/processed/cleaned_sales.csv", parse_dates=["date"])
    inv     = pd.read_csv("data/processed/cleaned_inventory.csv", parse_dates=["date"])
    fc      = pd.read_csv("data/forecasts/forecast_15days.csv", parse_dates=["forecast_date"])
    catalog = pd.read_csv("data/raw/product_catalog.csv")
    rec = compute_inventory_intelligence(sales, inv, fc, catalog)
    os.makedirs("outputs", exist_ok=True)
    rec.to_csv("outputs/inventory_intelligence.csv", index=False)
    print(rec[["product_id","alert_status","current_stock","days_to_stockout"]].to_string())