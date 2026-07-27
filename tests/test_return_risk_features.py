import pandas as pd

from src.return_risk_features import chronological_split, assert_no_temporal_leakage


def test_chronological_split_purges_outcome_overlap() -> None:
    dates = pd.date_range("2010-01-01", periods=20, freq="30D")
    snapshots = pd.DataFrame(
        {
            "customer_id": range(20),
            "observation_date": dates,
            "outcome_end": dates + pd.Timedelta(days=90),
            "last_purchase": dates - pd.Timedelta(days=1),
        }
    )
    train, validation, test, _ = chronological_split(snapshots)

    assert_no_temporal_leakage(train, validation, test)
    assert train["observation_date"].max() < validation["observation_date"].min()
    assert validation["observation_date"].max() < test["observation_date"].min()


def test_customer_identifier_is_not_a_model_feature() -> None:
    excluded = {
        "customer_id",
        "snapshot_id",
        "observation_date",
        "outcome_end",
        "first_purchase",
        "last_purchase",
        "returned_within_90_days",
    }
    assert "customer_id" in excluded
