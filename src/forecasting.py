"""
STEP 3A — Sales Forecasting (XGBoost)
Trains per-category XGBoost models and forecasts next 15 days per product.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

FEATURES = [
    "temp_c", "rainfall_mm", "is_rainy", "is_cold", "is_hot",
    "dayofweek", "month", "quarter", "dayofyear", "is_weekend",
    "is_month_end", "is_holiday",
    "lag_1", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_30",
    "roll_std_7", "roll_std_14", "roll_max_7",
    "avg_price", "avg_discount", "base_price",
]
TARGET = "daily_sales"


def train_forecast_model(feature_df: pd.DataFrame,
                         models_dir: str = "models",
                         forecast_days: int = 15):
    os.makedirs(models_dir, exist_ok=True)
    df = feature_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Fill is_holiday if missing
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    # Train/test split: last forecast_days per product = test
    split_date = df["date"].max() - pd.Timedelta(days=forecast_days)
    train = df[df["date"] <= split_date].copy()
    test  = df[df["date"] >  split_date].copy()

    avail_features = [f for f in FEATURES if f in df.columns]

    # One global model (enough data per product for XGBoost)
    X_train = train[avail_features].fillna(0)
    y_train = train[TARGET]
    X_test  = test[avail_features].fillna(0)
    y_test  = test[TARGET]

    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  Forecast model — MAE: {mae:.2f}  RMSE: {rmse:.2f}")

    model_path = os.path.join(models_dir, "xgboost_forecaster.pkl")
    joblib.dump({"model": model, "features": avail_features, "mae": mae, "rmse": rmse}, model_path)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=avail_features).sort_values(ascending=False)

    return model, avail_features, {"mae": mae, "rmse": rmse}, imp


def generate_15day_forecast(feature_df: pd.DataFrame,
                             model,
                             avail_features: list,
                             weather_future: pd.DataFrame = None,
                             forecast_days: int = 15) -> pd.DataFrame:
    """
    Rolls forward 15 days from the last known date per product.
    weather_future: optional DataFrame with future weather predictions.
    """
    df = feature_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    last_date  = df["date"].max()

    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    HOLIDAY_DAYS = {(1,1),(2,14),(3,8),(10,31),(11,25),(11,26),(12,24),(12,25),(12,31)}

    all_forecasts = []
    for prod_id in df["product_id"].unique():
        prod_df  = df[df["product_id"] == prod_id].sort_values("date")
        last_row = prod_df.iloc[-1].copy()
        sales_history = list(prod_df["daily_sales"].values)

        for day_offset in range(1, forecast_days + 1):
            fdate = last_date + pd.Timedelta(days=day_offset)
            row   = last_row.copy()
            row["date"]          = fdate
            row["dayofweek"]     = fdate.dayofweek
            row["month"]         = fdate.month
            row["quarter"]       = (fdate.month - 1) // 3 + 1
            row["dayofyear"]     = fdate.timetuple().tm_yday
            row["is_weekend"]    = int(fdate.dayofweek >= 5)
            row["is_month_end"]  = int(fdate.day == pd.Timestamp(fdate.year, fdate.month, 1).days_in_month)
            row["is_holiday"]    = int((fdate.month, fdate.day) in HOLIDAY_DAYS)

            # Use future weather if available
            if weather_future is not None:
                wrow = weather_future[weather_future["date"] == fdate]
                if not wrow.empty:
                    for wcol in ["temp_c","rainfall_mm","is_rainy","is_cold","is_hot"]:
                        if wcol in wrow.columns:
                            row[wcol] = wrow[wcol].values[0]

            # Update lags from rolling history
            for lag in [1, 3, 7, 14, 21, 28]:
                row[f"lag_{lag}"] = sales_history[-lag] if len(sales_history) >= lag else sales_history[0]
            for win in [7, 14, 30]:
                window = sales_history[-win:] if len(sales_history) >= win else sales_history
                row[f"roll_mean_{win}"] = np.mean(window)
                row[f"roll_std_{win}"]  = np.std(window)
                row[f"roll_max_{win}"]  = np.max(window)

            X = pd.DataFrame([row[avail_features].fillna(0)])
            pred = max(0, float(model.predict(X)[0]))
            sales_history.append(pred)

            all_forecasts.append({
                "product_id":      prod_id,
                "product_name":    last_row.get("product_name", prod_id),
                "category":        last_row.get("category", "Unknown"),
                "forecast_date":   fdate,
                "forecast_sales":  round(pred, 1),
                "day_offset":      day_offset,
            })

    return pd.DataFrame(all_forecasts)


if __name__ == "__main__":
    feat = pd.read_csv("data/processed/featured_data.csv", parse_dates=["date"])
    model, features, metrics, imp = train_forecast_model(feat)
    forecast = generate_15day_forecast(feat, model, features)
    os.makedirs("data/forecasts", exist_ok=True)
    forecast.to_csv("data/forecasts/forecast_15days.csv", index=False)
    print(f"  Forecast saved: {len(forecast)} rows")
    print(f"  Top features: {list(imp.head(5).index)}")