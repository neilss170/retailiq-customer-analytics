"""Business metric and narrative helpers for the RetailIQ dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


def filter_transactions(
    transactions: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    countries: list[str],
) -> pd.DataFrame:
    mask = transactions["invoice_date"].between(start, end + pd.Timedelta(days=1), inclusive="left")
    if countries:
        mask &= transactions["country"].isin(countries)
    return transactions.loc[mask].copy()


def executive_metrics(transactions: pd.DataFrame) -> dict[str, float]:
    sales = transactions.loc[transactions["transaction_status"].eq("valid_sale")]
    orders = (
        sales.groupby("invoice", dropna=False)["gross_line_revenue"].sum()
        if not sales.empty
        else pd.Series(dtype=float)
    )
    known = sales.loc[sales["customer_id"].notna()]
    customer_orders = known.groupby("customer_id")["invoice"].nunique()
    return {
        "gross_revenue": float(transactions["gross_line_revenue"].sum()),
        "returned_value": float(transactions["returned_value"].sum()),
        "net_revenue": float(transactions["net_line_revenue"].sum()),
        "orders": int(sales["invoice"].nunique()),
        "active_customers": int(known["customer_id"].nunique()),
        "average_order_value": float(orders.mean()) if len(orders) else 0.0,
        "repeat_purchase_rate": float(customer_orders.ge(2).mean()) if len(customer_orders) else 0.0,
        "return_value_rate": float(
            transactions["returned_value"].sum()
            / max(
                transactions["gross_line_revenue"].sum()
                + transactions["returned_value"].sum(),
                1,
            )
        ),
    }


def metric_deltas(current: dict[str, float], previous: dict[str, float]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key, value in current.items():
        denominator = previous.get(key, 0)
        result[key] = (value - denominator) / abs(denominator) if denominator else None
    return result


def executive_narrative(
    current: dict[str, float], previous: dict[str, float], deltas: dict[str, float | None]
) -> tuple[str, str]:
    revenue_delta = deltas["net_revenue"]
    if revenue_delta is None:
        return (
            "A comparable previous period is not available for this selection.",
            "Use the current period as a descriptive baseline rather than a growth comparison.",
        )
    direction = "increased" if revenue_delta >= 0 else "decreased"
    candidates = {
        "order volume": deltas.get("orders"),
        "active customers": deltas.get("active_customers"),
        "average order value": deltas.get("average_order_value"),
    }
    available = {key: value for key, value in candidates.items() if value is not None}
    driver = max(available, key=lambda key: abs(available[key])) if available else "available activity metrics"
    driver_change = available.get(driver)
    takeaway = (
        f"Net revenue {direction} {abs(revenue_delta):.1%} versus the previous equivalent "
        f"period. The largest accompanying movement was {driver}"
        + (f" at {driver_change:+.1%}." if driver_change is not None else ".")
    )
    action = (
        "Investigate the customer and product breakdowns before taking action; "
        "the dashboard shows association, not causation."
    )
    return takeaway, action


def monthly_revenue(transactions: pd.DataFrame) -> pd.DataFrame:
    data = transactions.copy()
    data["month"] = data["invoice_date"].dt.to_period("M").dt.to_timestamp()
    return (
        data.groupby("month")
        .agg(
            gross_revenue=("gross_line_revenue", "sum"),
            returned_value=("returned_value", "sum"),
            net_revenue=("net_line_revenue", "sum"),
        )
        .reset_index()
    )


def country_performance(transactions: pd.DataFrame) -> pd.DataFrame:
    return (
        transactions.groupby("country")
        .agg(
            net_revenue=("net_line_revenue", "sum"),
            gross_revenue=("gross_line_revenue", "sum"),
            returned_value=("returned_value", "sum"),
            customers=("customer_id", "nunique"),
        )
        .reset_index()
        .sort_values("net_revenue", ascending=False)
    )


def product_performance(transactions: pd.DataFrame) -> pd.DataFrame:
    products = (
        transactions.groupby(["stock_code", "description"], dropna=False)
        .agg(
            net_revenue=("net_line_revenue", "sum"),
            gross_revenue=("gross_line_revenue", "sum"),
            returned_value=("returned_value", "sum"),
            invoices=("invoice", "nunique"),
        )
        .reset_index()
    )
    products["return_value_rate"] = (
        products["returned_value"]
        / (products["gross_revenue"] + products["returned_value"]).replace(0, np.nan)
    ).fillna(0)
    products["product_label"] = products["description"].fillna(products["stock_code"]).str[:48]
    return products.sort_values("net_revenue", ascending=False)


def cohort_retention(transactions: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    sales = transactions.loc[
        transactions["customer_id"].notna()
        & transactions["transaction_status"].eq("valid_sale")
    ].copy()
    if sales.empty:
        return pd.DataFrame()
    sales["order_month"] = sales["invoice_date"].dt.to_period("M")
    first = sales.groupby("customer_id")["order_month"].min().rename("cohort_month")
    activity = sales[["customer_id", "order_month"]].drop_duplicates().join(first, on="customer_id")
    activity["period"] = activity["order_month"].astype(int) - activity["cohort_month"].astype(int)
    counts = activity.groupby(["cohort_month", "period"])["customer_id"].nunique().unstack(fill_value=0)
    retention = counts.div(counts[0], axis=0).iloc[:, : months + 1]
    retention.index = retention.index.astype(str)
    return retention


def campaign_simulation(
    scored: pd.DataFrame,
    threshold: float,
    contact_cost: float,
    missed_returner_cost: float,
) -> dict[str, float]:
    """Summarize capacity and illustrative cost at a frozen score cutoff."""
    predicted = scored["estimated_return_likelihood"].ge(threshold)
    actual = scored["returned_within_90_days"].astype(bool)
    true_positive = int((predicted & actual).sum())
    false_positive = int((predicted & ~actual).sum())
    false_negative = int((~predicted & actual).sum())
    contacts = true_positive + false_positive
    return {
        "contacts": contacts,
        "precision": true_positive / max(contacts, 1),
        "recall": true_positive / max(true_positive + false_negative, 1),
        "illustrative_cost": contacts * contact_cost
        + false_negative * missed_returner_cost,
    }
