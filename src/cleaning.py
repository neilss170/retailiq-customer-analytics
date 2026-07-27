"""Create analysis-ready RetailIQ transaction tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_quality import load_workbook

SPECIAL_STOCK_CODES = {
    "ADJUST",
    "ADJUST2",
    "AMAZONFEE",
    "BANK CHARGES",
    "C2",
    "CRUK",
    "D",
    "DCGSSBOY",
    "DCGSSGIRL",
    "DOT",
    "M",
    "PADS",
    "POST",
    "S",
    "SP1002",
    "TEST001",
    "TEST002",
}


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize fields and classify rows without hiding commercial events."""
    exact_duplicate_mask = raw.duplicated(keep="first")
    duplicates = raw.loc[exact_duplicate_mask].copy()
    data = raw.loc[~exact_duplicate_mask].copy()
    data.columns = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
        "source_sheet",
    ]

    data["invoice"] = data["invoice"].astype("string").str.strip().str.upper()
    data["stock_code"] = data["stock_code"].astype("string").str.strip().str.upper()
    data["description"] = (
        data["description"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    )
    data["country"] = (
        data["country"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    )
    data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="coerce")
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    data["unit_price"] = pd.to_numeric(data["unit_price"], errors="coerce")
    data["customer_id"] = pd.to_numeric(data["customer_id"], errors="coerce").astype(
        "Int64"
    )

    cancelled_invoice = data["invoice"].str.startswith("C", na=False)
    special_code = data["stock_code"].isin(SPECIAL_STOCK_CODES)
    missing_required = (
        data["invoice"].isna()
        | data["stock_code"].isna()
        | data["invoice_date"].isna()
        | data["quantity"].isna()
        | data["unit_price"].isna()
    )

    conditions = [
        missing_required,
        data["unit_price"].lt(0),
        special_code,
        cancelled_invoice | data["quantity"].lt(0),
        data["quantity"].gt(0) & data["unit_price"].gt(0),
        data["quantity"].gt(0) & data["unit_price"].eq(0),
    ]
    choices = [
        "invalid",
        "invalid",
        "adjustment",
        "cancellation_or_return",
        "valid_sale",
        "zero_price",
    ]
    data["transaction_status"] = np.select(
        conditions, choices, default="review_required"
    )

    data["gross_line_revenue"] = np.where(
        data["transaction_status"].eq("valid_sale"),
        data["quantity"] * data["unit_price"],
        0.0,
    )
    data["returned_value"] = np.where(
        data["transaction_status"].eq("cancellation_or_return")
        & data["unit_price"].gt(0),
        data["quantity"].abs() * data["unit_price"],
        0.0,
    )
    data["net_line_revenue"] = (
        data["gross_line_revenue"] - data["returned_value"]
    )
    data["known_customer"] = data["customer_id"].notna()
    data["eligible_customer_model"] = (
        data["known_customer"]
        & data["transaction_status"].isin(["valid_sale", "cancellation_or_return"])
    )
    data["invoice_month"] = data["invoice_date"].dt.to_period("M").astype("string")
    data["invoice_week"] = (
        data["invoice_date"].dt.to_period("W-SUN").dt.start_time
    )
    data = data.sort_values(
        ["invoice_date", "invoice", "stock_code"], kind="stable"
    ).reset_index(drop=True)
    data.insert(0, "transaction_id", np.arange(1, len(data) + 1))
    return data, duplicates


def build_quality_summary(
    cleaned: pd.DataFrame, duplicates: pd.DataFrame, raw_rows: int
) -> pd.DataFrame:
    """Summarize row impact and monetary impact for every cleaning class."""
    grouped = (
        cleaned.groupby("transaction_status", dropna=False)
        .agg(
            rows=("transaction_id", "size"),
            gross_revenue=("gross_line_revenue", "sum"),
            returned_value=("returned_value", "sum"),
            net_revenue=("net_line_revenue", "sum"),
        )
        .reset_index()
    )
    grouped["row_share"] = grouped["rows"] / raw_rows
    duplicate_row = pd.DataFrame(
        {
            "transaction_status": ["exact_duplicate_removed"],
            "rows": [len(duplicates)],
            "gross_revenue": [np.nan],
            "returned_value": [np.nan],
            "net_revenue": [np.nan],
            "row_share": [len(duplicates) / raw_rows],
        }
    )
    return pd.concat([grouped, duplicate_row], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/source/online_retail_II.xlsx"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed")
    )
    args = parser.parse_args()

    raw = load_workbook(args.input)
    cleaned, duplicates = clean_transactions(raw)
    summary = build_quality_summary(cleaned, duplicates, len(raw))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(args.output_dir / "transactions.parquet", index=False)
    duplicates.to_csv(args.output_dir / "exact_duplicates.csv", index=False)
    summary.to_csv(args.output_dir / "cleaning_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
