"""Build reusable analytical tables and a local SQLite database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def build_orders(transactions: pd.DataFrame) -> pd.DataFrame:
    commercial = transactions.loc[
        transactions["transaction_status"].isin(
            ["valid_sale", "cancellation_or_return"]
        )
    ].copy()
    return (
        commercial.groupby(["invoice", "invoice_date", "customer_id", "country"], dropna=False)
        .agg(
            distinct_products=("stock_code", "nunique"),
            units=("quantity", "sum"),
            gross_revenue=("gross_line_revenue", "sum"),
            returned_value=("returned_value", "sum"),
            net_revenue=("net_line_revenue", "sum"),
            line_count=("transaction_id", "size"),
        )
        .reset_index()
        .sort_values("invoice_date")
    )


def build_weekly_revenue(transactions: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        transactions.groupby("invoice_week", dropna=False)
        .agg(
            gross_revenue=("gross_line_revenue", "sum"),
            returned_value=("returned_value", "sum"),
            net_revenue=("net_line_revenue", "sum"),
            invoice_count=("invoice", "nunique"),
            active_customers=("customer_id", "nunique"),
        )
        .reset_index()
        .sort_values("invoice_week")
    )
    full_weeks = pd.DataFrame(
        {
            "invoice_week": pd.date_range(
                weekly["invoice_week"].min(),
                weekly["invoice_week"].max(),
                freq="7D",
            )
        }
    )
    return full_weeks.merge(weekly, on="invoice_week", how="left").fillna(
        {
            "gross_revenue": 0.0,
            "returned_value": 0.0,
            "net_revenue": 0.0,
            "invoice_count": 0,
            "active_customers": 0,
        }
    )


def build_customers(transactions: pd.DataFrame) -> pd.DataFrame:
    known_sales = transactions.loc[
        transactions["customer_id"].notna()
        & transactions["transaction_status"].eq("valid_sale")
    ].copy()
    return (
        known_sales.groupby("customer_id")
        .agg(
            first_purchase=("invoice_date", "min"),
            last_purchase=("invoice_date", "max"),
            orders=("invoice", "nunique"),
            products=("stock_code", "nunique"),
            units=("quantity", "sum"),
            gross_revenue=("gross_line_revenue", "sum"),
            primary_country=("country", lambda values: values.mode().iat[0]),
        )
        .reset_index()
    )


def build_products(transactions: pd.DataFrame) -> pd.DataFrame:
    product_rows = transactions.loc[
        transactions["description"].notna()
        & transactions["transaction_status"].isin(
            ["valid_sale", "cancellation_or_return"]
        )
    ].copy()
    return (
        product_rows.groupby("stock_code")
        .agg(
            description=("description", lambda values: values.mode().iat[0]),
            gross_revenue=("gross_line_revenue", "sum"),
            returned_value=("returned_value", "sum"),
            net_revenue=("net_line_revenue", "sum"),
            invoices=("invoice", "nunique"),
            customers=("customer_id", "nunique"),
        )
        .reset_index()
    )


def build_dates(transactions: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DataFrame(
        {
            "date": pd.date_range(
                transactions["invoice_date"].min().normalize(),
                transactions["invoice_date"].max().normalize(),
                freq="D",
            )
        }
    )
    dates["year"] = dates["date"].dt.year
    dates["quarter"] = dates["date"].dt.quarter
    dates["month"] = dates["date"].dt.month
    dates["month_name"] = dates["date"].dt.month_name()
    dates["week"] = dates["date"].dt.isocalendar().week.astype(int)
    dates["day_of_week"] = dates["date"].dt.day_name()
    dates["is_weekend"] = dates["date"].dt.dayofweek.ge(5)
    return dates


def write_sqlite(
    database_path: Path,
    transactions: pd.DataFrame,
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    dates: pd.DataFrame,
    weekly: pd.DataFrame,
) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        transactions.to_sql("fact_transactions", connection, if_exists="replace", index=False)
        orders.to_sql("fact_orders", connection, if_exists="replace", index=False)
        customers.to_sql("dim_customers", connection, if_exists="replace", index=False)
        products.to_sql("dim_products", connection, if_exists="replace", index=False)
        dates.to_sql("dim_dates", connection, if_exists="replace", index=False)
        weekly.to_sql("weekly_revenue", connection, if_exists="replace", index=False)
        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_transactions_invoice
                ON fact_transactions(invoice);
            CREATE INDEX IF NOT EXISTS idx_transactions_customer
                ON fact_transactions(customer_id);
            CREATE INDEX IF NOT EXISTS idx_transactions_date
                ON fact_transactions(invoice_date);
            CREATE INDEX IF NOT EXISTS idx_orders_date
                ON fact_orders(invoice_date);
            """
        )


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
    parser.add_argument(
        "--database", type=Path, default=Path("data/retailiq.db")
    )
    args = parser.parse_args()

    transactions = pd.read_parquet(args.transactions)
    orders = build_orders(transactions)
    customers = build_customers(transactions)
    products = build_products(transactions)
    dates = build_dates(transactions)
    weekly = build_weekly_revenue(transactions)

    tables = {
        "orders": orders,
        "customers": customers,
        "products": products,
        "dates": dates,
        "weekly_revenue": weekly,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(args.output_dir / f"{name}.parquet", index=False)

    write_sqlite(
        args.database,
        transactions,
        orders,
        customers,
        products,
        dates,
        weekly,
    )
    for name, table in tables.items():
        print(f"{name}: {len(table):,} rows")


if __name__ == "__main__":
    main()
