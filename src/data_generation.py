"""
STEP 1 — Data Generation
Generates 4 CSV files that power all ML modules and dashboard panels.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
import os

np.random.seed(42)

START_DATE = date(2022, 1, 1)
END_DATE   = date(2024, 3, 31)
OUTPUT_DIR = "data/raw"

CATEGORIES = {
    "Electronics": {"count": 6, "base_price": (300, 1200), "base_demand": (3, 15),  "seasonality": "christmas"},
    "Clothing":    {"count": 6, "base_price": (20,  150),  "base_demand": (10, 50), "seasonality": "summer"},
    "Food":        {"count": 6, "base_price": (2,   30),   "base_demand": (30, 120),"seasonality": "flat"},
    "Beverages":   {"count": 5, "base_price": (1,   15),   "base_demand": (40, 150),"seasonality": "summer"},
    "Furniture":   {"count": 4, "base_price": (80,  600),  "base_demand": (1, 8),   "seasonality": "flat"},
    "Sports":      {"count": 5, "base_price": (25,  300),  "base_demand": (5, 25),  "seasonality": "summer"},
    "Beauty":      {"count": 4, "base_price": (10,  80),   "base_demand": (8, 40),  "seasonality": "christmas"},
    "Books":       {"count": 4, "base_price": (8,   45),   "base_demand": (5, 20),  "seasonality": "flat"},
}

HOLIDAY_BOOSTS = {
    (1, 1): 1.5, (2, 14): 1.4, (3, 8): 1.3, (10, 31): 1.4,
    (11, 25): 2.2, (11, 26): 2.0, (12, 24): 2.5, (12, 25): 1.8, (12, 31): 1.6,
}

WEATHER_EFFECT = {
    "Electronics": (1.1, 0.9, 1.15), "Clothing": (1.2, 0.8, 0.9),
    "Food":        (1.1, 1.0, 1.05), "Beverages": (0.8, 1.4, 0.9),
    "Furniture":   (1.0, 1.0, 1.0),  "Sports": (0.7, 1.3, 0.6),
    "Beauty":      (1.1, 1.0, 1.0),  "Books": (1.2, 0.9, 1.3),
}

PAYMENT_METHODS = ["Cash", "Credit Card", "Debit Card", "UPI", "Wallet"]
LOCATIONS       = ["In-store", "Online", "Mobile App"]
N_STORES        = 3


def build_catalog():
    rows, pid = [], 1
    for cat, cfg in CATEGORIES.items():
        for i in range(1, cfg["count"] + 1):
            price = round(np.random.uniform(*cfg["base_price"]), 2)
            rows.append({
                "product_id": f"SKU_{pid:03d}", "product_name": f"{cat}_Item_{i}",
                "category": cat, "base_price": price,
                "supplier": f"Supplier_{np.random.randint(1,8)}",
                "reorder_lead_days": np.random.choice([3,5,7,10,14]),
                "min_order_qty": np.random.choice([10,20,50,100]),
            })
            pid += 1
    return pd.DataFrame(rows)


def build_weather(date_range):
    rows = []
    for d in date_range:
        doy  = d.timetuple().tm_yday
        temp = 15 + 15 * np.sin((doy - 80) / 365 * 2 * np.pi) + np.random.normal(0, 3)
        rain = max(0, np.random.normal(3, 4) if np.random.rand() < 0.3 else 0)
        rows.append({
            "date": d, "temp_c": round(temp, 1), "rainfall_mm": round(rain, 1),
            "is_rainy": int(rain > 2), "is_cold": int(temp < 10), "is_hot": int(temp > 28),
        })
    return pd.DataFrame(rows)


def seasonal_mult(d, pattern):
    m = d.month
    if pattern == "christmas":
        return 1.0 + 0.8 * max(0, (m - 9) / 3) if m >= 10 else 1.0
    elif pattern == "summer":
        return 1.0 + 0.5 * np.sin((m - 3) / 12 * 2 * np.pi)
    return 1.0


def build_sales(catalog, weather_df):
    date_range = [START_DATE + timedelta(days=i) for i in range((END_DATE - START_DATE).days + 1)]
    weather_map = weather_df.set_index("date").to_dict("index")
    dead_skus   = set(catalog.sample(4)["product_id"])
    rows, txn_id = [], 1

    for d in date_range:
        w = weather_map.get(d, {"temp_c": 15, "rainfall_mm": 0, "is_cold": 0, "is_hot": 0, "is_rainy": 0})
        holiday_boost = HOLIDAY_BOOSTS.get((d.month, d.day), 1.0)
        weekday       = d.weekday()
        weekend_mult  = 1.3 if weekday >= 5 else 1.0

        for _, prod in catalog.iterrows():
            cat, pid = prod["category"], prod["product_id"]
            if pid in dead_skus and d >= date(2023, 10, 1):
                continue

            cfg   = CATEGORIES[cat]
            base  = np.random.uniform(*cfg["base_demand"])
            smult = seasonal_mult(d, cfg["seasonality"])
            wc, wh, wr = WEATHER_EFFECT[cat]
            wmult = 1.0
            if w["is_cold"]:  wmult *= wc
            if w["is_hot"]:   wmult *= wh
            if w["is_rainy"]: wmult *= wr

            trend_mult = 1.0 + 0.0003 * (d - START_DATE).days
            demand = max(0, int(round(
                base * smult * wmult * weekend_mult * holiday_boost * trend_mult
                * (1 + np.random.normal(0, 0.12))
            )))
            if demand == 0:
                continue

            n_txns = max(1, demand // 3)
            for _ in range(n_txns):
                qty      = max(1, demand // n_txns + np.random.randint(-1, 2))
                price    = prod["base_price"] * np.random.uniform(0.92, 1.08)
                discount = np.random.choice([0, 0.05, 0.10, 0.15, 0.20], p=[0.6, 0.15, 0.1, 0.1, 0.05])
                rows.append({
                    "transaction_id": f"TXN_{txn_id:07d}", "date": d,
                    "product_id": pid, "product_name": prod["product_name"],
                    "category": cat,
                    "store_id": f"STORE_{np.random.randint(1, N_STORES+1):02d}",
                    "customer_id": f"CUST_{np.random.randint(1,501):04d}",
                    "quantity": qty, "unit_price": round(price, 2),
                    "discount_pct": discount,
                    "total_spent": round(qty * price * (1 - discount), 2),
                    "payment_method": np.random.choice(PAYMENT_METHODS),
                    "location": np.random.choice(LOCATIONS),
                    "is_weekend": int(weekday >= 5),
                    "is_holiday": int((d.month, d.day) in HOLIDAY_BOOSTS),
                    "temp_c": w["temp_c"], "rainfall_mm": w["rainfall_mm"],
                    "is_rainy": w["is_rainy"], "is_cold": w["is_cold"], "is_hot": w["is_hot"],
                })
                txn_id += 1

    return pd.DataFrame(rows)


def build_inventory(catalog, sales_df):
    date_range  = [START_DATE + timedelta(days=i) for i in range((END_DATE - START_DATE).days + 1)]
    daily_sales = (sales_df.groupby(["date","product_id","store_id"])["quantity"]
                   .sum().reset_index())
    rows = []
    for _, prod in catalog.iterrows():
        pid = prod["product_id"]
        for store in [f"STORE_{i:02d}" for i in range(1, N_STORES+1)]:
            stock        = np.random.randint(80, 300)
            restock_date = START_DATE
            for d in date_range:
                sold = int(daily_sales[
                    (daily_sales["date"] == d) &
                    (daily_sales["product_id"] == pid) &
                    (daily_sales["store_id"] == store)
                ]["quantity"].sum())
                if stock < 30 and d >= restock_date:
                    restock_qty  = int(np.random.randint(100, 300))
                    stock       += restock_qty
                    restock_date = d + timedelta(days=int(prod["reorder_lead_days"]))
                    restock_flag = 1
                else:
                    restock_qty  = 0
                    restock_flag = 0
                stock = max(0, stock - sold)
                rows.append({
                    "date": d, "product_id": pid, "store_id": store,
                    "opening_stock": stock + sold - restock_qty,
                    "units_sold": sold, "units_restocked": restock_qty,
                    "closing_stock": stock, "was_restocked": restock_flag,
                })
    return pd.DataFrame(rows)


def generate_all(start=START_DATE, end=END_DATE, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    global START_DATE, END_DATE
    START_DATE, END_DATE = start, end
    date_range = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    print("  [1/4] Building product catalog...")
    catalog = build_catalog()
    catalog.to_csv(f"{out_dir}/product_catalog.csv", index=False)

    print("  [2/4] Building weather data...")
    weather = build_weather(date_range)
    weather.to_csv(f"{out_dir}/weather_data.csv", index=False)

    print("  [3/4] Building sales transactions...")
    sales = build_sales(catalog, weather.assign(date=pd.to_datetime(weather["date"]).dt.date))
    sales.to_csv(f"{out_dir}/sales_transactions.csv", index=False)

    print("  [4/4] Building inventory log...")
    inventory = build_inventory(catalog, sales)
    inventory.to_csv(f"{out_dir}/inventory_log.csv", index=False)

    print(f"\n  Done. {len(sales):,} transactions | {len(inventory):,} inventory rows | {len(catalog)} SKUs")
    return catalog, weather, sales, inventory


if __name__ == "__main__":
    generate_all()