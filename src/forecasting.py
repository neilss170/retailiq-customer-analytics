"""Weekly net-revenue forecasting with rolling backtests and uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

TARGET = "net_revenue"
LAGS = (1, 2, 4, 8, 13)
ROLLING_WINDOWS = (4, 8, 13)
FORECAST_HORIZON = 4


def complete_weekly_series(weekly: pd.DataFrame) -> pd.DataFrame:
    """Remove boundary weeks that do not contain seven observable days."""
    data = weekly.sort_values("invoice_week").reset_index(drop=True).copy()
    if len(data) < 20:
        raise ValueError("At least 20 weekly observations are required")
    return data.iloc[1:-1].reset_index(drop=True)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    return float(
        100
        * np.mean(
            np.divide(
                2 * np.abs(predicted - actual),
                denominator,
                out=np.zeros_like(actual, dtype=float),
                where=denominator != 0,
            )
        )
    )


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "smape": smape(actual, predicted),
    }


def make_supervised(history: pd.DataFrame) -> pd.DataFrame:
    frame = history[["invoice_week", TARGET]].copy()
    for lag in LAGS:
        frame[f"lag_{lag}"] = frame[TARGET].shift(lag)
    shifted = frame[TARGET].shift(1)
    for window in ROLLING_WINDOWS:
        frame[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        frame[f"rolling_std_{window}"] = shifted.rolling(window).std()
    week = frame["invoice_week"].dt.isocalendar().week.astype(int)
    frame["week_sin"] = np.sin(2 * np.pi * week / 52)
    frame["week_cos"] = np.cos(2 * np.pi * week / 52)
    frame["month"] = frame["invoice_week"].dt.month
    frame["quarter"] = frame["invoice_week"].dt.quarter
    return frame.dropna().reset_index(drop=True)


def _future_feature(history: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    values = history[TARGET].to_numpy()
    row: dict[str, float] = {}
    for lag in LAGS:
        row[f"lag_{lag}"] = float(values[-lag])
    for window in ROLLING_WINDOWS:
        row[f"rolling_mean_{window}"] = float(np.mean(values[-window:]))
        row[f"rolling_std_{window}"] = float(np.std(values[-window:], ddof=1))
    week = int(date.isocalendar().week)
    row["week_sin"] = np.sin(2 * np.pi * week / 52)
    row["week_cos"] = np.cos(2 * np.pi * week / 52)
    row["month"] = date.month
    row["quarter"] = date.quarter
    return pd.DataFrame([row])


def recursive_ml_forecast(
    model: object, history: pd.DataFrame, horizon: int
) -> np.ndarray:
    supervised = make_supervised(history)
    feature_names = [
        column for column in supervised.columns if column not in ["invoice_week", TARGET]
    ]
    model.fit(supervised[feature_names], supervised[TARGET])
    extended = history[["invoice_week", TARGET]].copy()
    predictions: list[float] = []
    for _ in range(horizon):
        next_date = extended["invoice_week"].max() + pd.Timedelta(days=7)
        feature = _future_feature(extended, next_date)[feature_names]
        prediction = max(0.0, float(model.predict(feature)[0]))
        predictions.append(prediction)
        extended.loc[len(extended)] = [next_date, prediction]
    return np.asarray(predictions)


def forecast_method(name: str, history: pd.DataFrame, horizon: int) -> np.ndarray:
    values = history[TARGET].to_numpy()
    if name == "last_value":
        return np.repeat(values[-1], horizon)
    if name == "moving_average_4":
        return np.repeat(np.mean(values[-4:]), horizon)
    if name == "seasonal_naive_52":
        return np.asarray([values[-52 + index] for index in range(horizon)])
    if name == "exponential_smoothing":
        model = ExponentialSmoothing(
            values,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=13,
            initialization_method="estimated",
        ).fit(optimized=True)
        return np.maximum(0, np.asarray(model.forecast(horizon)))
    if name == "sarima":
        model = SARIMAX(
            values,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 0, 13),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return np.maximum(0, np.asarray(model.forecast(horizon)))
    if name == "gradient_boosting":
        return recursive_ml_forecast(
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=4,
                loss="huber",
                random_state=42,
            ),
            history,
            horizon,
        )
    if name == "xgboost":
        return recursive_ml_forecast(
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=4,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=42,
            ),
            history,
            horizon,
        )
    raise KeyError(name)


def rolling_backtest(
    development: pd.DataFrame, methods: list[str], horizon: int = FORECAST_HORIZON
) -> tuple[pd.DataFrame, pd.DataFrame]:
    minimum_train = 56
    cutoffs = list(range(minimum_train, len(development) - horizon + 1, horizon))
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for cutoff in cutoffs:
        history = development.iloc[:cutoff]
        actual_frame = development.iloc[cutoff : cutoff + horizon]
        for name in methods:
            predicted = forecast_method(name, history, len(actual_frame))
            score = forecast_metrics(actual_frame[TARGET].to_numpy(), predicted)
            rows.append({"model": name, "cutoff": history["invoice_week"].max(), **score})
            predictions.extend(
                {
                    "model": name,
                    "cutoff": history["invoice_week"].max(),
                    "invoice_week": date,
                    "actual": float(actual),
                    "predicted": float(prediction),
                    "error": float(actual - prediction),
                }
                for date, actual, prediction in zip(
                    actual_frame["invoice_week"],
                    actual_frame[TARGET],
                    predicted,
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def run_forecasting(output_dir: Path = Path("reports/generated")) -> dict[str, object]:
    weekly = complete_weekly_series(
        pd.read_parquet("data/processed/weekly_revenue.parquet")
    )
    train_end = int(len(weekly) * 0.70)
    test_start = int(len(weekly) * 0.85)
    development = weekly.iloc[:test_start].copy()
    final_test = weekly.iloc[test_start:].copy()
    methods = [
        "last_value",
        "moving_average_4",
        "seasonal_naive_52",
        "exponential_smoothing",
        "sarima",
        "gradient_boosting",
        "xgboost",
    ]
    backtest, backtest_predictions = rolling_backtest(development, methods)
    summary = (
        backtest.groupby("model")
        .agg(
            folds=("cutoff", "nunique"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            smape_mean=("smape", "mean"),
        )
        .sort_values("mae_mean")
        .reset_index()
    )
    selected = str(summary.iloc[0]["model"])

    frozen_prediction = forecast_method(selected, development, len(final_test))
    test_metrics = forecast_metrics(
        final_test[TARGET].to_numpy(), frozen_prediction
    )
    final_test_predictions = final_test[["invoice_week", TARGET]].copy()
    final_test_predictions["predicted_net_revenue"] = frozen_prediction
    final_test_predictions["error"] = (
        final_test_predictions[TARGET]
        - final_test_predictions["predicted_net_revenue"]
    )

    residuals = backtest_predictions.loc[
        backtest_predictions["model"].eq(selected), "error"
    ]
    lower_error, upper_error = residuals.quantile([0.10, 0.90])
    future_point = forecast_method(selected, weekly, 8)
    future_dates = pd.date_range(
        weekly["invoice_week"].max() + pd.Timedelta(days=7), periods=8, freq="7D"
    )
    future = pd.DataFrame(
        {
            "invoice_week": future_dates,
            "forecast_net_revenue": future_point,
            "lower_80": np.maximum(0, future_point + lower_error),
            "upper_80": np.maximum(0, future_point + upper_error),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(output_dir / "forecast_backtest_metrics.csv", index=False)
    backtest_predictions.to_parquet(
        output_dir / "forecast_backtest_predictions.parquet", index=False
    )
    summary.to_csv(output_dir / "forecast_model_comparison.csv", index=False)
    final_test_predictions.to_parquet(
        output_dir / "forecast_final_test_predictions.parquet", index=False
    )
    future.to_parquet("data/processed/revenue_forecast.parquet", index=False)
    joblib.dump({"model_name": selected}, "models/revenue_forecast_pipeline.joblib")
    metadata = {
        "selected_model": selected,
        "selection_metric": "rolling_backtest_mae",
        "complete_weeks": int(len(weekly)),
        "development_weeks": int(len(development)),
        "final_test_weeks": int(len(final_test)),
        "train_reference_end_index": train_end,
        "final_test_period_start": final_test["invoice_week"].min().isoformat(),
        "final_test_period_end": final_test["invoice_week"].max().isoformat(),
        "final_test_metrics": test_metrics,
        "forecast_horizon_weeks": 8,
        "interval_method": "10th and 90th percentiles of rolling-backtest residuals",
        "limitations": [
            "Only 104 complete weekly observations are available.",
            "The source covers one historical retailer and excludes external drivers.",
            "The 80% range is empirical and not a guaranteed coverage interval.",
        ],
    }
    Path("models/revenue_forecast_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(run_forecasting(), indent=2))
