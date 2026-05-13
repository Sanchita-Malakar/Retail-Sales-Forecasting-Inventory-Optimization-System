"""
STEP 3C — Intelligence Engine
Trend detection, velocity scoring, seasonal decomposition, category analytics.
"""
import pandas as pd
import numpy as np


def compute_trend_scores(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-product trend metrics using the most recent 30 days.
    trend_score  > 0  → growing
    trend_score  < 0  → declining
    momentum_pct: % change vs 30 days ago
    """
    df = feature_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()
    recent   = df[df["date"] >= max_date - pd.Timedelta(days=30)]
    prev     = df[(df["date"] >= max_date - pd.Timedelta(days=60)) &
                  (df["date"] <  max_date - pd.Timedelta(days=30))]

    def slope(series):
        x = np.arange(len(series))
        if len(x) < 3:
            return 0
        return float(np.polyfit(x, series, 1)[0])

    recent_stats = (
        recent.groupby(["product_id","category"])["daily_sales"]
        .agg(recent_mean="mean", recent_sum="sum", recent_slope=slope)
        .reset_index()
    )
    prev_stats = (
        prev.groupby("product_id")["daily_sales"]
        .mean().reset_index().rename(columns={"daily_sales": "prev_mean"})
    )
    trends = recent_stats.merge(prev_stats, on="product_id", how="left")
    trends["prev_mean"] = trends["prev_mean"].fillna(trends["recent_mean"])
    trends["momentum_pct"] = (
        (trends["recent_mean"] - trends["prev_mean"]) / (trends["prev_mean"] + 1e-6) * 100
    ).round(1)
    trends["trend_label"] = pd.cut(
        trends["momentum_pct"],
        bins=[-np.inf, -20, -5, 5, 20, np.inf],
        labels=["Declining fast", "Declining", "Stable", "Growing", "Hot 🔥"]
    )
    # Rank within category
    trends["category_rank"] = (
        trends.groupby("category")["recent_mean"].rank(ascending=False).astype(int)
    )
    return trends.sort_values("momentum_pct", ascending=False).reset_index(drop=True)


def compute_velocity_table(feature_df: pd.DataFrame,
                            inventory_rec: pd.DataFrame) -> pd.DataFrame:
    """
    Selling rate table: units/day, revenue/day, days-to-stockout.
    """
    df = feature_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()
    last7    = df[df["date"] >= max_date - pd.Timedelta(days=7)]
    last30   = df[df["date"] >= max_date - pd.Timedelta(days=30)]

    v7  = last7.groupby("product_id")["daily_sales"].mean().rename("velocity_7d")
    v30 = last30.groupby("product_id")["daily_sales"].mean().rename("velocity_30d")
    rev = last30.groupby("product_id")["daily_revenue"].mean().rename("revenue_per_day")

    vel = pd.concat([v7, v30, rev], axis=1).reset_index()
    vel = vel.merge(
        inventory_rec[["product_id","product_name","category",
                        "current_stock","days_to_stockout","alert_status"]],
        on="product_id", how="left"
    )
    vel["velocity_7d"]  = vel["velocity_7d"].round(1)
    vel["velocity_30d"] = vel["velocity_30d"].round(1)
    vel["revenue_per_day"] = vel["revenue_per_day"].round(2)
    return vel.sort_values("velocity_7d", ascending=False).reset_index(drop=True)


def compute_seasonal_summary(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Month × category aggregation — useful for heatmap and bar charts.
    """
    df = sales_df.copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year
    summary = (
        df.groupby(["year","month","category"]).agg(
            total_units   = ("quantity", "sum"),
            total_revenue = ("total_spent", "sum"),
            n_transactions= ("transaction_id", "count"),
        ).reset_index()
    )
    summary["month_name"] = pd.to_datetime(
        summary["month"].astype(str), format="%m"
    ).dt.strftime("%b")
    return summary


def compute_weather_impact(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation between weather features and sales per category.
    """
    df = feature_df.copy()
    results = []
    for cat, grp in df.groupby("category"):
        for wcol in ["temp_c", "rainfall_mm", "is_cold", "is_hot", "is_rainy"]:
            if wcol in grp.columns:
                corr = grp["daily_sales"].corr(grp[wcol])
                results.append({
                    "category":        cat,
                    "weather_feature": wcol,
                    "correlation":     round(corr, 3),
                })
    return pd.DataFrame(results).sort_values(["category","correlation"], ascending=[True, False])


if __name__ == "__main__":
    import os
    feat    = pd.read_csv("data/processed/featured_data.csv",  parse_dates=["date"])
    sales   = pd.read_csv("data/processed/cleaned_sales.csv",  parse_dates=["date"])
    inv_rec = pd.read_csv("outputs/inventory_intelligence.csv")

    trends  = compute_trend_scores(feat)
    vel     = compute_velocity_table(feat, inv_rec)
    seasonal= compute_seasonal_summary(sales)
    weather = compute_weather_impact(feat)

    os.makedirs("outputs", exist_ok=True)
    trends.to_csv("outputs/trend_scores.csv",    index=False)
    vel.to_csv("outputs/velocity_table.csv",     index=False)
    seasonal.to_csv("outputs/seasonal_summary.csv", index=False)
    weather.to_csv("outputs/weather_impact.csv", index=False)
    print("Intelligence engine done.")
    print(trends[["product_id","category","momentum_pct","trend_label"]].head(10).to_string())