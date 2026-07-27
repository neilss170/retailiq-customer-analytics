import numpy as np
import pandas as pd

from src.forecasting import complete_weekly_series, make_supervised, smape


def test_boundary_weeks_are_removed() -> None:
    weekly = pd.DataFrame(
        {
            "invoice_week": pd.date_range("2010-01-04", periods=20, freq="7D"),
            "net_revenue": np.arange(20, dtype=float),
        }
    )
    complete = complete_weekly_series(weekly)
    assert len(complete) == 18
    assert complete["invoice_week"].min() == weekly["invoice_week"].iloc[1]


def test_lag_features_use_only_prior_values() -> None:
    weekly = pd.DataFrame(
        {
            "invoice_week": pd.date_range("2010-01-04", periods=30, freq="7D"),
            "net_revenue": np.arange(30, dtype=float),
        }
    )
    supervised = make_supervised(weekly)
    first = supervised.iloc[0]
    assert first["lag_1"] == first["net_revenue"] - 1
    assert first["rolling_mean_4"] < first["net_revenue"]


def test_smape_is_zero_for_perfect_forecast() -> None:
    actual = np.array([10.0, 20.0])
    assert smape(actual, actual.copy()) == 0.0
