# Data dictionary

## `fact_transactions`

Grain: one deduplicated source invoice line.

| Field | Meaning |
|---|---|
| `transaction_id` | Deterministic row identifier assigned after cleaning |
| `invoice` | Source invoice identifier; cancellation invoices commonly start with `C` |
| `stock_code` | Source product or adjustment code |
| `invoice_date` | Transaction timestamp |
| `customer_id` | Source customer identifier; missing for anonymous transactions |
| `transaction_status` | Cleaning classification: valid sale, cancellation/return, adjustment, zero price, invalid, or review required |
| `gross_line_revenue` | Positive quantity multiplied by positive unit price for valid sales |
| `returned_value` | Absolute value associated with cancellation/return rows that have a positive unit price |
| `net_line_revenue` | Gross line revenue less returned value |
| `eligible_customer_model` | Whether the line can contribute to known-customer modelling |

## `fact_orders`

Grain: one invoice, timestamp, customer, and country combination.

## `dim_customers`

Grain: one known customer with at least one valid sale.

## `dim_products`

Grain: one product stock code appearing in a valid sale or cancellation/return.

## `dim_dates`

Grain: one calendar date across the observed transaction range.

## `weekly_revenue`

Grain: one calendar week, including zero-filled weeks inside the observed range.
