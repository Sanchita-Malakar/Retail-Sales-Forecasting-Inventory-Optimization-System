"""
STEP 2B — Feature Engineering
Builds lag features, rolling stats, weather embeddings, seasonal flags,
velocity metrics, and dead-stock indicators — all in one place.
"""
import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"]       = pd.to_datetime(df["date"])
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["month"]      = df["date"].dt.month
    df["quarter"]    = df["date"].dt.quarter
    df["dayofyear"]  = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "daily_sales") -> pd.DataFrame:
    """Must be called on daily-aggregated data (one row per product per day)."""
    df = df.sort_values(["product_id", "date"])
    grp = df.groupby("product_id")[target_col]
    for lag in [1, 3, 7, 14, 21, 28]:
        df[f"lag_{lag}"]  = grp.shift(lag)
    for win in [7, 14, 30]:
        df[f"roll_mean_{win}"] = grp.transform(lambda x: x.rolling(win, min_periods=1).mean())
        df[f"roll_std_{win}"]  = grp.transform(lambda x: x.rolling(win, min_periods=1).std().fillna(0))
        df[f"roll_max_{win}"]  = grp.transform(lambda x: x.rolling(win, min_periods=1).max())
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selling velocity = units sold per day (rolling 7d).
    Days-to-stockout = current_stock / velocity.
    """
    df = df.copy()
    if "roll_mean_7" in df.columns:
        df["velocity_7d"] = df["roll_mean_7"]
    else:
        df["velocity_7d"] = df["daily_sales"].rolling(7, min_periods=1).mean()
    return df


def build_daily_product_features(sales_df: pd.DataFrame,
                                  inventory_df: pd.DataFrame,
                                  weather_df: pd.DataFrame,
                                  catalog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature table: one row per (date, product_id).
    Joins sales aggregates + inventory + weather + catalog metadata.
    """
    # Aggregate sales to daily per product
    daily = (
        sales_df.groupby(["date", "product_id", "category"]).agg(
            daily_sales    = ("quantity", "sum"),
            daily_revenue  = ("total_spent", "sum"),
            n_transactions = ("transaction_id", "count"),
            avg_price      = ("unit_price", "mean"),
            avg_discount   = ("discount_pct", "mean"),
        ).reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])

    # Aggregate inventory to daily per product (sum across stores)
    inv_daily = (
        inventory_df.groupby(["date", "product_id"]).agg(
            total_stock    = ("closing_stock", "sum"),
            units_restocked= ("units_restocked", "sum"),
            was_restocked  = ("was_restocked", "max"),
        ).reset_index()
    )
    inv_daily["date"] = pd.to_datetime(inv_daily["date"])

    # Merge
    df = daily.merge(inv_daily, on=["date", "product_id"], how="left")
    df = df.merge(weather_df[["date","temp_c","rainfall_mm","is_rainy","is_cold","is_hot"]],
                  on="date", how="left")
    df = df.merge(catalog_df[["product_id","base_price","reorder_lead_days","min_order_qty"]],
                  on="product_id", how="left")

    # Time features
    df = add_time_features(df)

    # Lag + rolling features
    df = add_lag_features(df, target_col="daily_sales")

    # Velocity
    df = add_velocity_features(df)

    # Days-to-stockout estimate
    df["days_to_stockout"] = np.where(
        df["velocity_7d"] > 0,
        (df["total_stock"] / df["velocity_7d"]).round(1),
        999
    )

    # Dead stock flag: no sales in last 30 days
    last_sold = (
        sales_df.groupby("product_id")["date"]
        .max().reset_index()
        .rename(columns={"date": "last_sold_date"})
    )
    last_sold["last_sold_date"] = pd.to_datetime(last_sold["last_sold_date"])
    df = df.merge(last_sold, on="product_id", how="left")
    max_date = df["date"].max()
    df["days_since_sold"]  = (max_date - df["last_sold_date"]).dt.days
    df["is_dead_stock"]    = (df["days_since_sold"] > 30).astype(int)

    # Overstock flag: stock > 90 days of supply
    df["is_overstock"]     = (df["days_to_stockout"] > 90).astype(int)

    # Trend score: slope of last-14-day sales (simple linear regression coefficient)
    def trend_slope(series):
        n = len(series)
        if n < 3:
            return 0
        x = np.arange(n)
        return np.polyfit(x, series, 1)[0]

    df["trend_score"] = (
        df.sort_values("date")
          .groupby("product_id")["daily_sales"]
          .transform(lambda x: x.rolling(14, min_periods=3).apply(trend_slope, raw=True))
    )
    df["is_trending"] = (df["trend_score"] > df["trend_score"].quantile(0.75)).astype(int)

    return df.dropna(subset=["lag_7"]).reset_index(drop=True)


if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)
    sales   = pd.read_csv("data/processed/cleaned_sales.csv", parse_dates=["date"])
    inv     = pd.read_csv("data/processed/cleaned_inventory.csv", parse_dates=["date"])
    weather = pd.read_csv("data/raw/weather_data.csv", parse_dates=["date"])
    catalog = pd.read_csv("data/raw/product_catalog.csv")
    features = build_daily_product_features(sales, inv, weather, catalog)
    features.to_csv("data/processed/featured_data.csv", index=False)
    print(f"Feature table: {features.shape[0]:,} rows x {features.shape[1]} columns")