"""
STEP 2A — Data Cleaning
Handles nulls, outliers, type fixes, and consistency checks across all 4 CSVs.
"""
import pandas as pd
import numpy as np


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Drop rows with no product or date
    df = df.dropna(subset=["product_id", "date"])
    # Remove negative quantities / prices
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    # Cap outlier quantities at 99th percentile per category
    cap = df.groupby("category")["quantity"].transform(lambda x: x.quantile(0.99))
    df["quantity"] = np.where(df["quantity"] > cap, cap, df["quantity"]).astype(int)
    # Recalculate total_spent for consistency
    df["total_spent"] = (df["quantity"] * df["unit_price"] * (1 - df["discount_pct"])).round(2)
    # Fill missing payment/location with mode
    for col in ["payment_method", "location"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df.sort_values(["product_id", "date"]).reset_index(drop=True)


def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["closing_stock"] = df["closing_stock"].clip(lower=0)
    return df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["temp_c"] = df["temp_c"].clip(-20, 50)
    df["rainfall_mm"] = df["rainfall_mm"].clip(0, 200)
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)
    sales = clean_sales(pd.read_csv("data/raw/sales_transactions.csv"))
    inv   = clean_inventory(pd.read_csv("data/raw/inventory_log.csv"))
    sales.to_csv("data/processed/cleaned_sales.csv", index=False)
    inv.to_csv("data/processed/cleaned_inventory.csv", index=False)
    print("Cleaning done.")