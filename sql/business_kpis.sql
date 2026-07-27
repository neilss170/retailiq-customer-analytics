-- Executive metrics at invoice-line grain.
SELECT
    ROUND(SUM(gross_line_revenue), 2) AS gross_revenue,
    ROUND(SUM(returned_value), 2) AS returned_value,
    ROUND(SUM(net_line_revenue), 2) AS net_revenue,
    COUNT(DISTINCT CASE WHEN transaction_status = 'valid_sale' THEN invoice END) AS sales_orders,
    COUNT(DISTINCT CASE WHEN transaction_status = 'valid_sale' THEN customer_id END) AS purchasing_customers
FROM fact_transactions;

-- Country performance with returns kept separate from sales.
SELECT
    country,
    ROUND(SUM(gross_line_revenue), 2) AS gross_revenue,
    ROUND(SUM(returned_value), 2) AS returned_value,
    ROUND(SUM(net_line_revenue), 2) AS net_revenue,
    COUNT(DISTINCT invoice) AS invoices
FROM fact_transactions
GROUP BY country
ORDER BY net_revenue DESC;

-- Customer concentration for Pareto analysis.
SELECT
    customer_id,
    primary_country,
    orders,
    ROUND(gross_revenue, 2) AS gross_revenue
FROM dim_customers
ORDER BY gross_revenue DESC;
