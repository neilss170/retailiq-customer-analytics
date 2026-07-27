"""Independent SQLite reconciliation for headline EDA claims."""

import sqlite3

import pandas as pd


with sqlite3.connect("data/retailiq.db") as connection:
    total = pd.read_sql_query(
        "SELECT SUM(net_line_revenue) AS value FROM fact_transactions", connection
    )["value"].iat[0]
    uk = pd.read_sql_query(
        """
        SELECT SUM(net_line_revenue) AS value
        FROM fact_transactions
        WHERE country = 'United Kingdom'
        """,
        connection,
    )["value"].iat[0]
    customers = pd.read_sql_query(
        "SELECT gross_revenue FROM dim_customers ORDER BY gross_revenue DESC",
        connection,
    )
    monthly = pd.read_sql_query(
        """
        SELECT
            SUBSTR(invoice_date, 1, 7) AS month,
            SUM(gross_revenue) AS revenue,
            COUNT(DISTINCT invoice) AS orders
        FROM fact_orders
        WHERE gross_revenue > 0
        GROUP BY 1
        ORDER BY 1
        """,
        connection,
    )

top_count = max(1, int(len(customers) * 0.20 + 0.999))
top_share = customers.head(top_count)["gross_revenue"].sum() / customers[
    "gross_revenue"
].sum()

print(f"UK net-revenue share: {uk / total:.6f}")
print(f"Top-20%-customer gross-revenue share: {top_share:.6f}")
print(monthly.tail(3).to_string(index=False))

assert abs(uk / total - 0.856) < 0.002
assert abs(top_share - 0.772) < 0.002
assert monthly.iloc[-1]["month"] == "2011-12"
assert monthly.iloc[-2]["month"] == "2011-11"
