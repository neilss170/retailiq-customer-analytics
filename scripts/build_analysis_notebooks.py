"""Create the first three reproducible RetailIQ notebooks with nbformat."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def notebook(title: str, context: str, tldr: str, cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n## tl;dr\n\n{tldr}"),
        nbf.v4.new_markdown_cell(f"## Context & Methods\n\n{context}\n\n### Key Assumptions\n\nRevenue is reported in the source currency. Cancellations reduce net revenue only when a positive source price is available."),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(ROOT))\n"
            "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "sns.set_theme(style='whitegrid')\n"
            "pd.options.display.float_format = '{:,.2f}'.format"
        ),
        *cells,
    ]
    return nb


def build_quality_notebook() -> nbf.NotebookNode:
    return notebook(
        "RetailIQ Data Quality and Cleaning",
        "This diagnostic validates the transaction grain and documents the effect of every cleaning class before business analysis.",
        "The source contains 1,067,371 rows. Exactly 12,133 duplicate source rows are removed, while sales, cancellations, adjustments, zero-price records, and anonymous purchases remain separately auditable.",
        [
            nbf.v4.new_markdown_cell("## Data\n\n### 1. Load the cleaning evidence"),
            nbf.v4.new_code_cell(
                "transactions = pd.read_parquet(ROOT / 'data/processed/transactions.parquet')\n"
                "summary = pd.read_csv(ROOT / 'data/processed/cleaning_summary.csv')\n"
                "print(f'Clean rows: {len(transactions):,}')\n"
                "display(summary)"
            ),
            nbf.v4.new_markdown_cell("## Results\n\n### 2. Check completeness and transaction classes"),
            nbf.v4.new_code_cell(
                "quality = pd.DataFrame({\n"
                " 'missing_count': transactions.isna().sum(),\n"
                " 'missing_rate': transactions.isna().mean()\n"
                "}).sort_values('missing_rate', ascending=False)\n"
                "display(quality.head(10))\n"
                "display(transactions['transaction_status'].value_counts().rename_axis('status').to_frame('rows'))"
            ),
            nbf.v4.new_markdown_cell("### 3. Reconcile monetary components"),
            nbf.v4.new_code_cell(
                "money = transactions[['gross_line_revenue','returned_value','net_line_revenue']].sum()\n"
                "display(money.to_frame('amount'))\n"
                "assert np.isclose(money['gross_line_revenue'] - money['returned_value'], money['net_line_revenue'])"
            ),
            nbf.v4.new_markdown_cell("## Takeaways\n\nThe retained statuses make sales, cancellations, adjustments, and unusable records auditable. Anonymous transactions remain in revenue reporting but are not eligible for customer-level modelling."),
        ],
    )


def build_eda_notebook() -> nbf.NotebookNode:
    return notebook(
        "RetailIQ Business EDA and Cohort Retention",
        "This analysis explains revenue, customer concentration, and repeat purchasing from cleaned tables. It is descriptive and does not claim causal effects.",
        "Customer value is concentrated: the top 20% of known purchasing customers generated 77.2% of customer revenue. Cohorts are evaluated by months since first purchase so recent cohorts are not mistaken for weak mature cohorts.",
        [
            nbf.v4.new_markdown_cell("## Data\n\n### 1. Load analytical tables"),
            nbf.v4.new_code_cell(
                "transactions = pd.read_parquet(ROOT / 'data/processed/transactions.parquet')\n"
                "orders = pd.read_parquet(ROOT / 'data/processed/orders.parquet')\n"
                "customers = pd.read_parquet(ROOT / 'data/processed/customers.parquet')\n"
                "weekly = pd.read_parquet(ROOT / 'data/processed/weekly_revenue.parquet')"
            ),
            nbf.v4.new_markdown_cell("## Results\n\n### 2. Executive metrics"),
            nbf.v4.new_code_cell(
                "sales_orders = orders.loc[orders['gross_revenue'].gt(0)]\n"
                "metrics = pd.Series({\n"
                " 'Net revenue': transactions['net_line_revenue'].sum(),\n"
                " 'Gross revenue': transactions['gross_line_revenue'].sum(),\n"
                " 'Returned value': transactions['returned_value'].sum(),\n"
                " 'Sales orders': sales_orders['invoice'].nunique(),\n"
                " 'Known purchasing customers': customers['customer_id'].nunique(),\n"
                " 'Average sales order value': sales_orders['gross_revenue'].mean(),\n"
                "})\n"
                "display(metrics.to_frame('value'))"
            ),
            nbf.v4.new_markdown_cell("### 3. Weekly revenue story"),
            nbf.v4.new_code_cell(
                "fig, ax = plt.subplots(figsize=(12,4))\n"
                "ax.plot(weekly['invoice_week'], weekly['net_revenue'], color='#2563EB')\n"
                "ax.set(title='Weekly net revenue', xlabel='', ylabel='Net revenue')\n"
                "plt.show()"
            ),
            nbf.v4.new_markdown_cell("### 4. Customer concentration"),
            nbf.v4.new_code_cell(
                "ranked = customers.sort_values('gross_revenue', ascending=False).copy()\n"
                "ranked['cumulative_share'] = ranked['gross_revenue'].cumsum() / ranked['gross_revenue'].sum()\n"
                "top_20_count = max(1, int(np.ceil(len(ranked) * .20)))\n"
                "print(f\"Top 20% customer revenue share: {ranked.head(top_20_count)['gross_revenue'].sum()/ranked['gross_revenue'].sum():.1%}\")"
            ),
            nbf.v4.new_markdown_cell("### 5. Which countries contribute most to net revenue?"),
            nbf.v4.new_code_cell(
                "country = transactions.groupby('country').agg(\n"
                " gross_revenue=('gross_line_revenue','sum'), returned_value=('returned_value','sum'),\n"
                " net_revenue=('net_line_revenue','sum'), customers=('customer_id','nunique')\n"
                ").sort_values('net_revenue', ascending=False)\n"
                "country['net_revenue_share'] = country['net_revenue'] / country['net_revenue'].sum()\n"
                "display(country.head(10))\n"
                "fig, ax = plt.subplots(figsize=(10,5))\n"
                "country.head(10).sort_values('net_revenue')['net_revenue'].plot.barh(ax=ax, color='#2563EB')\n"
                "ax.set(title='Net revenue by country — top 10', xlabel='Net revenue', ylabel='')\n"
                "plt.show()\n"
                "print(f\"United Kingdom share of net revenue: {country.loc['United Kingdom','net_revenue_share']:.1%}\")"
            ),
            nbf.v4.new_markdown_cell("### 6. Which products generate revenue, and which lose value to cancellations?"),
            nbf.v4.new_code_cell(
                "product = transactions.groupby(['stock_code','description'], dropna=False).agg(\n"
                " gross_revenue=('gross_line_revenue','sum'), returned_value=('returned_value','sum'),\n"
                " net_revenue=('net_line_revenue','sum'), invoices=('invoice','nunique')\n"
                ").reset_index()\n"
                "product['return_value_rate'] = product['returned_value'] / (product['gross_revenue'] + product['returned_value']).replace(0,np.nan)\n"
                "top_products = product.nlargest(12, 'net_revenue')\n"
                "display(top_products[['stock_code','description','net_revenue','invoices']])\n"
                "fig, ax = plt.subplots(figsize=(10,6))\n"
                "plot_data = top_products.sort_values('net_revenue')\n"
                "ax.barh(plot_data['description'].fillna(plot_data['stock_code']).str[:38], plot_data['net_revenue'], color='#2563EB')\n"
                "ax.set(title='Net revenue by product — top 12', xlabel='Net revenue', ylabel='')\n"
                "plt.tight_layout(); plt.show()\n"
                "return_watch = product.loc[product['gross_revenue'].ge(1000)].nlargest(10,'returned_value')\n"
                "display(return_watch[['stock_code','description','gross_revenue','returned_value','return_value_rate']])"
            ),
            nbf.v4.new_markdown_cell("### 7. Is revenue coming from new or returning customers?"),
            nbf.v4.new_code_cell(
                "known_orders = sales_orders.loc[sales_orders['customer_id'].notna()].copy()\n"
                "known_orders['order_month'] = known_orders['invoice_date'].dt.to_period('M').dt.to_timestamp()\n"
                "first_month = known_orders.groupby('customer_id')['order_month'].min().rename('first_month')\n"
                "known_orders = known_orders.join(first_month, on='customer_id')\n"
                "known_orders['customer_type'] = np.where(known_orders['order_month'].eq(known_orders['first_month']), 'New customer', 'Returning customer')\n"
                "customer_mix = known_orders.groupby(['order_month','customer_type'])['gross_revenue'].sum().unstack(fill_value=0)\n"
                "customer_mix.plot(figsize=(12,5), color=['#D4A72C','#2563EB'])\n"
                "plt.title('Monthly revenue from new and returning customers'); plt.xlabel(''); plt.ylabel('Gross revenue'); plt.show()\n"
                "mix_share = customer_mix.div(customer_mix.sum(axis=1), axis=0)\n"
                "display(mix_share.tail(6))"
            ),
            nbf.v4.new_markdown_cell("### 8. What does a typical order look like?"),
            nbf.v4.new_code_cell(
                "order_profile = sales_orders[['gross_revenue','units','distinct_products']].describe(percentiles=[.5,.75,.9,.95,.99])\n"
                "display(order_profile)\n"
                "fig, axes = plt.subplots(1,2,figsize=(12,4))\n"
                "sales_orders['gross_revenue'].clip(upper=sales_orders['gross_revenue'].quantile(.99)).hist(bins=40, ax=axes[0], color='#2563EB')\n"
                "axes[0].set(title='Sales order value distribution — capped at 99th percentile', xlabel='Order value', ylabel='Orders')\n"
                "sales_orders['distinct_products'].clip(upper=sales_orders['distinct_products'].quantile(.99)).hist(bins=35, ax=axes[1], color='#D4A72C')\n"
                "axes[1].set(title='Distinct products per order — capped at 99th percentile', xlabel='Distinct products', ylabel='Orders')\n"
                "plt.tight_layout(); plt.show()"
            ),
            nbf.v4.new_markdown_cell("### 9. How long do customers wait between purchases?"),
            nbf.v4.new_code_cell(
                "purchase_dates = known_orders[['customer_id','invoice','invoice_date']].drop_duplicates().sort_values(['customer_id','invoice_date'])\n"
                "purchase_dates['days_since_previous_order'] = purchase_dates.groupby('customer_id')['invoice_date'].diff().dt.total_seconds().div(86400)\n"
                "gaps = purchase_dates['days_since_previous_order'].dropna()\n"
                "display(gaps.describe(percentiles=[.25,.5,.75,.9,.95]).to_frame('days'))\n"
                "print(f'Median repeat-purchase gap: {gaps.median():.0f} days')"
            ),
            nbf.v4.new_markdown_cell("### 10. When does purchasing activity occur?"),
            nbf.v4.new_code_cell(
                "sales_lines = transactions.loc[transactions['transaction_status'].eq('valid_sale')].copy()\n"
                "sales_lines['weekday'] = sales_lines['invoice_date'].dt.day_name()\n"
                "sales_lines['hour'] = sales_lines['invoice_date'].dt.hour\n"
                "weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']\n"
                "weekday = sales_lines.groupby('weekday')['gross_line_revenue'].sum().reindex(weekday_order).fillna(0)\n"
                "hourly = sales_lines.groupby('hour')['gross_line_revenue'].sum()\n"
                "fig, axes = plt.subplots(1,2,figsize=(13,4))\n"
                "weekday.plot.bar(ax=axes[0], color='#2563EB'); axes[0].set(title='Gross revenue by weekday', xlabel='', ylabel='Gross revenue')\n"
                "hourly.plot(ax=axes[1], color='#D4A72C', marker='o'); axes[1].set(title='Gross revenue by invoice hour', xlabel='Hour', ylabel='Gross revenue')\n"
                "plt.tight_layout(); plt.show()"
            ),
            nbf.v4.new_markdown_cell("### 11. What changed in the latest complete month?"),
            nbf.v4.new_code_cell(
                "monthly_orders = sales_orders.copy()\n"
                "monthly_orders['month'] = monthly_orders['invoice_date'].dt.to_period('M').dt.to_timestamp()\n"
                "monthly_kpis = monthly_orders.groupby('month').agg(\n"
                " revenue=('gross_revenue','sum'), orders=('invoice','nunique'), active_customers=('customer_id','nunique')\n"
                ")\n"
                "monthly_kpis['average_order_value'] = monthly_kpis['revenue'] / monthly_kpis['orders']\n"
                "complete = monthly_kpis.iloc[1:-1].tail(2)\n"
                "change = complete.pct_change().iloc[-1].rename('change_vs_previous_month')\n"
                "display(complete)\n"
                "display(change.to_frame())\n"
                "driver = change.abs().sort_values(ascending=False).index[0]\n"
                "direction = 'increased' if change['revenue'] >= 0 else 'decreased'\n"
                "print(f\"Gross revenue {direction} {abs(change['revenue']):.1%} in {complete.index[-1]:%B %Y}. The largest accompanying KPI movement was {driver.replace('_',' ')} at {change[driver]:+.1%}.\")"
            ),
            nbf.v4.new_markdown_cell("### 12. Monthly acquisition cohorts"),
            nbf.v4.new_code_cell(
                "sales = transactions.loc[transactions['eligible_customer_model'] & transactions['transaction_status'].eq('valid_sale')].copy()\n"
                "sales['order_month'] = sales['invoice_date'].dt.to_period('M')\n"
                "first_month = sales.groupby('customer_id')['order_month'].min().rename('cohort_month')\n"
                "cohort = sales[['customer_id','order_month']].drop_duplicates().join(first_month, on='customer_id')\n"
                "cohort['period'] = (cohort['order_month'].astype(int) - cohort['cohort_month'].astype(int))\n"
                "counts = cohort.groupby(['cohort_month','period'])['customer_id'].nunique().unstack(fill_value=0)\n"
                "retention = counts.div(counts[0], axis=0)\n"
                "fig, ax = plt.subplots(figsize=(12,6))\n"
                "sns.heatmap(retention.iloc[:, :13], cmap='Blues', vmin=0, vmax=1, ax=ax)\n"
                "ax.set_title('Monthly cohort retention')\nplt.show()"
            ),
            nbf.v4.new_markdown_cell("## Takeaways\n\nThe analysis separates revenue level, customer mix, concentration, basket behaviour, cancellation exposure, and retention. Country and product rankings are descriptive; campaign or assortment decisions require margin and experimental evidence that this dataset does not contain."),
        ],
    )


def build_segmentation_notebook() -> nbf.NotebookNode:
    return notebook(
        "RetailIQ Customer Segmentation",
        "RFM provides an interpretable baseline. K-Means is evaluated only after log transformation and scaling, then translated into business-readable labels.",
        "A two-segment K-Means solution produced the strongest tested silhouette score (0.44). It separates 2,335 recently active, higher-value customers from 3,517 lower-activity customers with substantially longer recency.",
        [
            nbf.v4.new_markdown_cell("## Data\n\n### 1. Build point-in-time RFM features"),
            nbf.v4.new_code_cell(
                "from sklearn.cluster import KMeans\n"
                "from sklearn.metrics import silhouette_score\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "transactions = pd.read_parquet(ROOT / 'data/processed/transactions.parquet')\n"
                "sales = transactions.loc[transactions['customer_id'].notna() & transactions['transaction_status'].eq('valid_sale')].copy()\n"
                "as_of = sales['invoice_date'].max() + pd.Timedelta(days=1)\n"
                "rfm = sales.groupby('customer_id').agg(\n"
                " last_purchase=('invoice_date','max'), frequency=('invoice','nunique'), monetary=('gross_line_revenue','sum')\n"
                ").reset_index()\n"
                "rfm['recency_days'] = (as_of - rfm['last_purchase']).dt.days\n"
                "display(rfm[['recency_days','frequency','monetary']].describe())"
            ),
            nbf.v4.new_markdown_cell("## Results\n\n### 2. Compare cluster counts"),
            nbf.v4.new_code_cell(
                "features = np.log1p(rfm[['recency_days','frequency','monetary']])\n"
                "scaled = StandardScaler().fit_transform(features)\n"
                "scores = {}\n"
                "for k in range(2,8):\n"
                "    labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(scaled)\n"
                "    scores[k] = silhouette_score(scaled, labels)\n"
                "display(pd.Series(scores, name='silhouette').to_frame())\n"
                "best_k = max(scores, key=scores.get)\n"
                "model = KMeans(n_clusters=best_k, random_state=42, n_init=20)\n"
                "rfm['cluster'] = model.fit_predict(scaled)"
            ),
            nbf.v4.new_markdown_cell("### 3. Profile and name clusters"),
            nbf.v4.new_code_cell(
                "profile = rfm.groupby('cluster').agg(\n"
                " customers=('customer_id','size'), recency_days=('recency_days','median'),\n"
                " frequency=('frequency','median'), monetary=('monetary','median')\n"
                ").sort_values('monetary', ascending=False)\n"
                "profile['segment'] = 'Developing customers'\n"
                "profile.loc[profile['monetary'].idxmax(), 'segment'] = 'High-value active customers'\n"
                "profile.loc[profile['recency_days'].idxmax(), 'segment'] = 'Lower-activity at-risk customers'\n"
                "mapping = profile['segment'].to_dict()\n"
                "rfm['segment'] = rfm['cluster'].map(mapping)\n"
                "display(profile)\n"
                "rfm.to_parquet(ROOT / 'data/processed/customer_segments.parquet', index=False)"
            ),
            nbf.v4.new_markdown_cell("## Takeaways\n\nCluster labels are descriptive summaries, not causal truths. Before operational use, segment stability should be monitored as new transactions arrive."),
        ],
    )


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "01_data_quality_and_cleaning.ipynb": build_quality_notebook(),
        "02_business_eda_and_cohorts.ipynb": build_eda_notebook(),
        "03_customer_segmentation.ipynb": build_segmentation_notebook(),
    }
    for name, nb in outputs.items():
        nbf.write(nb, NOTEBOOKS / name)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
