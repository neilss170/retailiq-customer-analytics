"""Build point-in-time customer snapshots for 90-day return modelling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOME_DAYS = 90
MIN_HISTORY_DAYS = 90
SNAPSHOT_FREQUENCY = "30D"


def _customer_features(
    sales: pd.DataFrame, cancellations: pd.DataFrame, observation_date: pd.Timestamp
) -> pd.DataFrame:
    history = sales.loc[sales["invoice_date"].lt(observation_date)].copy()
    if history.empty:
        return pd.DataFrame()

    base = (
        history.groupby("customer_id")
        .agg(
            first_purchase=("invoice_date", "min"),
            last_purchase=("invoice_date", "max"),
            orders_lifetime=("invoice", "nunique"),
            revenue_lifetime=("gross_line_revenue", "sum"),
            units_lifetime=("quantity", "sum"),
            products_lifetime=("stock_code", "nunique"),
            primary_country=("country", lambda values: values.mode().iat[0]),
        )
        .reset_index()
    )
    order_values = (
        history.groupby(["customer_id", "invoice"])
        .agg(order_value=("gross_line_revenue", "sum"), basket_units=("quantity", "sum"))
        .reset_index()
        .groupby("customer_id")
        .agg(
            average_order_value=("order_value", "mean"),
            average_basket_units=("basket_units", "mean"),
            order_value_std=("order_value", "std"),
        )
        .reset_index()
    )
    base = base.merge(order_values, on="customer_id", how="left")
    base["recency_days"] = (observation_date - base["last_purchase"]).dt.days
    base["tenure_days"] = (observation_date - base["first_purchase"]).dt.days
    base["orders_per_30_days"] = (
        base["orders_lifetime"] / base["tenure_days"].clip(lower=1) * 30
    )

    for days in (30, 60, 90, 180):
        recent = history.loc[
            history["invoice_date"].ge(observation_date - pd.Timedelta(days=days))
        ]
        aggregate = (
            recent.groupby("customer_id")
            .agg(
                **{
                    f"orders_{days}d": ("invoice", "nunique"),
                    f"revenue_{days}d": ("gross_line_revenue", "sum"),
                    f"units_{days}d": ("quantity", "sum"),
                }
            )
            .reset_index()
        )
        base = base.merge(aggregate, on="customer_id", how="left")

    cancellation_history = cancellations.loc[
        cancellations["invoice_date"].lt(observation_date)
    ]
    cancellation_features = (
        cancellation_history.groupby("customer_id")
        .agg(
            cancellation_lines_lifetime=("transaction_id", "size"),
            returned_value_lifetime=("returned_value", "sum"),
        )
        .reset_index()
    )
    base = base.merge(cancellation_features, on="customer_id", how="left")

    numeric_fill = [
        column
        for column in base.columns
        if column.startswith(("orders_", "revenue_", "units_", "cancellation_", "returned_"))
    ]
    base[numeric_fill] = base[numeric_fill].fillna(0)
    base["recent_revenue_change"] = (
        base["revenue_30d"] - (base["revenue_90d"] - base["revenue_30d"]) / 2
    )
    base["recent_order_change"] = (
        base["orders_30d"] - (base["orders_90d"] - base["orders_30d"]) / 2
    )
    base["historical_return_value_rate"] = (
        base["returned_value_lifetime"]
        / (base["revenue_lifetime"] + base["returned_value_lifetime"]).replace(0, np.nan)
    ).fillna(0)
    base["observation_date"] = observation_date
    return base


def build_snapshots(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create non-overlapping feature/outcome windows for each snapshot date."""
    known = transactions.loc[transactions["customer_id"].notna()].copy()
    sales = known.loc[known["transaction_status"].eq("valid_sale")].copy()
    cancellations = known.loc[
        known["transaction_status"].eq("cancellation_or_return")
    ].copy()

    first_date = sales["invoice_date"].min().normalize() + pd.Timedelta(
        days=MIN_HISTORY_DAYS
    )
    last_date = sales["invoice_date"].max().normalize() - pd.Timedelta(
        days=OUTCOME_DAYS
    )
    observation_dates = pd.date_range(first_date, last_date, freq=SNAPSHOT_FREQUENCY)
    snapshots: list[pd.DataFrame] = []

    for observation_date in observation_dates:
        features = _customer_features(sales, cancellations, observation_date)
        features = features.loc[features["tenure_days"].ge(MIN_HISTORY_DAYS)].copy()
        outcome_end = observation_date + pd.Timedelta(days=OUTCOME_DAYS)
        returners = sales.loc[
            sales["invoice_date"].ge(observation_date)
            & sales["invoice_date"].lt(outcome_end),
            "customer_id",
        ].unique()
        features["returned_within_90_days"] = features["customer_id"].isin(returners).astype(int)
        features["outcome_end"] = outcome_end
        snapshots.append(features)

    result = pd.concat(snapshots, ignore_index=True)
    result["snapshot_id"] = (
        result["customer_id"].astype("string")
        + "_"
        + result["observation_date"].dt.strftime("%Y%m%d")
    )
    if result["snapshot_id"].duplicated().any():
        raise ValueError("Duplicate customer snapshot IDs detected")
    return result.sort_values(["observation_date", "customer_id"]).reset_index(drop=True)


def chronological_split(
    snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Split by observation dates and purge outcome overlap at boundaries."""
    dates = np.array(sorted(snapshots["observation_date"].unique()))
    validation_start = pd.Timestamp(dates[max(1, int(len(dates) * 0.70))])
    test_start = pd.Timestamp(dates[max(2, int(len(dates) * 0.85))])

    train = snapshots.loc[
        snapshots["outcome_end"].le(validation_start)
        & snapshots["observation_date"].lt(validation_start)
    ].copy()
    validation = snapshots.loc[
        snapshots["observation_date"].ge(validation_start)
        & snapshots["outcome_end"].le(test_start)
    ].copy()
    test = snapshots.loc[snapshots["observation_date"].ge(test_start)].copy()

    metadata = {
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "train_max_outcome_end": train["outcome_end"].max().isoformat(),
        "validation_max_outcome_end": validation["outcome_end"].max().isoformat(),
    }
    return train, validation, test, metadata


def assert_no_temporal_leakage(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> None:
    if train["outcome_end"].max() > validation["observation_date"].min():
        raise AssertionError("Training outcomes overlap validation features")
    if validation["outcome_end"].max() > test["observation_date"].min():
        raise AssertionError("Validation outcomes overlap test features")
    for frame in (train, validation, test):
        if frame["last_purchase"].ge(frame["observation_date"]).any():
            raise AssertionError("A feature includes a purchase at/after observation time")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transactions",
        type=Path,
        default=Path("data/processed/transactions.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed")
    )
    args = parser.parse_args()

    transactions = pd.read_parquet(args.transactions)
    snapshots = build_snapshots(transactions)
    train, validation, test, metadata = chronological_split(snapshots)
    assert_no_temporal_leakage(train, validation, test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(args.output_dir / "return_risk_snapshots.parquet", index=False)
    train.to_parquet(args.output_dir / "return_risk_train.parquet", index=False)
    validation.to_parquet(args.output_dir / "return_risk_validation.parquet", index=False)
    test.to_parquet(args.output_dir / "return_risk_test.parquet", index=False)
    pd.Series(metadata).to_json(
        args.output_dir / "return_risk_split_metadata.json", indent=2
    )
    for name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        print(
            f"{name}: {len(frame):,} snapshots, "
            f"{frame['returned_within_90_days'].mean():.1%} returned"
        )


if __name__ == "__main__":
    main()
