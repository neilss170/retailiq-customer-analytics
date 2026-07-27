"""Profile the raw Online Retail II workbook before transformation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = {
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
}


def load_workbook(path: Path) -> pd.DataFrame:
    """Load both source years and retain the originating sheet."""
    workbook = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet)
        missing = EXPECTED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{sheet} is missing columns: {sorted(missing)}")
        frame["source_sheet"] = sheet
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def profile_transactions(data: pd.DataFrame) -> dict[str, object]:
    """Return compact, JSON-serializable evidence for cleaning decisions."""
    invoice = data["Invoice"].astype("string").str.strip()
    stock_code = data["StockCode"].astype("string").str.strip()
    description = data["Description"].astype("string").str.strip()
    country = data["Country"].astype("string").str.strip()
    quantity = pd.to_numeric(data["Quantity"], errors="coerce")
    price = pd.to_numeric(data["Price"], errors="coerce")
    invoice_date = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    customer_id = pd.to_numeric(data["Customer ID"], errors="coerce")

    exact_duplicates = int(data.duplicated().sum())
    cancellation_mask = invoice.str.upper().str.startswith("C", na=False)
    negative_quantity = quantity.lt(0)
    nonpositive_price = price.le(0)
    positive_sale = quantity.gt(0) & price.gt(0) & ~cancellation_mask
    line_value = quantity * price

    return {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "source_sheets": {
            str(key): int(value)
            for key, value in data["source_sheet"].value_counts().items()
        },
        "date_min": invoice_date.min().isoformat(),
        "date_max": invoice_date.max().isoformat(),
        "distinct_invoices": int(invoice.nunique(dropna=True)),
        "distinct_customers": int(customer_id.nunique(dropna=True)),
        "distinct_products": int(stock_code.nunique(dropna=True)),
        "distinct_countries": int(country.nunique(dropna=True)),
        "exact_duplicate_rows": exact_duplicates,
        "exact_duplicate_rate": exact_duplicates / len(data),
        "missing": {
            str(column): {
                "count": int(data[column].isna().sum()),
                "rate": float(data[column].isna().mean()),
            }
            for column in data.columns
        },
        "cancelled_invoice_rows": int(cancellation_mask.sum()),
        "cancelled_invoice_rate": float(cancellation_mask.mean()),
        "negative_quantity_rows": int(negative_quantity.sum()),
        "nonpositive_price_rows": int(nonpositive_price.sum()),
        "missing_customer_rows": int(customer_id.isna().sum()),
        "blank_description_rows": int(description.isna().sum()),
        "future_dated_rows": int(invoice_date.gt(pd.Timestamp.now()).sum()),
        "valid_positive_sale_rows": int(positive_sale.sum()),
        "gross_positive_sales": float(line_value.where(positive_sale, 0).sum()),
        "negative_line_value": float(line_value.where(line_value.lt(0), 0).sum()),
        "quantity_summary": {
            str(key): float(value)
            for key, value in quantity.describe(
                percentiles=[0.01, 0.5, 0.95, 0.99]
            ).items()
        },
        "price_summary": {
            str(key): float(value)
            for key, value in price.describe(
                percentiles=[0.01, 0.5, 0.95, 0.99]
            ).items()
        },
        "top_countries": {
            str(key): int(value) for key, value in country.value_counts().head(10).items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/source/online_retail_II.xlsx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/data_quality_profile.json"),
    )
    args = parser.parse_args()

    data = load_workbook(args.input)
    profile = profile_transactions(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
