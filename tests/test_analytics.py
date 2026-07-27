import pandas as pd

from src.analytics import build_orders, build_weekly_revenue


def test_order_aggregation_does_not_double_count_invoices() -> None:
    transactions = pd.DataFrame(
        {
            "invoice": ["1", "1", "2"],
            "invoice_date": pd.to_datetime(["2011-01-01", "2011-01-01", "2011-01-08"]),
            "invoice_week": pd.to_datetime(["2010-12-27", "2010-12-27", "2011-01-03"]),
            "customer_id": pd.Series([10, 10, 11], dtype="Int64"),
            "country": ["UK", "UK", "UK"],
            "stock_code": ["A", "B", "A"],
            "quantity": [1, 2, 1],
            "transaction_id": [1, 2, 3],
            "transaction_status": ["valid_sale", "valid_sale", "valid_sale"],
            "gross_line_revenue": [5.0, 10.0, 7.0],
            "returned_value": [0.0, 0.0, 0.0],
            "net_line_revenue": [5.0, 10.0, 7.0],
        }
    )
    orders = build_orders(transactions)

    assert len(orders) == 2
    assert orders.loc[orders["invoice"].eq("1"), "gross_revenue"].iat[0] == 15.0


def test_weekly_revenue_reconciles_to_transaction_totals() -> None:
    transactions = pd.DataFrame(
        {
            "invoice_week": pd.to_datetime(["2011-01-03", "2011-01-10"]),
            "invoice": ["1", "2"],
            "customer_id": pd.Series([10, 11], dtype="Int64"),
            "gross_line_revenue": [20.0, 8.0],
            "returned_value": [0.0, 3.0],
            "net_line_revenue": [20.0, 5.0],
        }
    )
    weekly = build_weekly_revenue(transactions)

    assert weekly["gross_revenue"].sum() == 28.0
    assert weekly["returned_value"].sum() == 3.0
    assert weekly["net_revenue"].sum() == 25.0
