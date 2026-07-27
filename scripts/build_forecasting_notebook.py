"""Build the weekly net-revenue forecasting notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "07_revenue_forecasting.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    nb.cells = [
        md(
            "# Weekly Net-Revenue Forecasting\n\n## tl;dr\n\n"
            "A 52-week seasonal naïve forecast beat Exponential Smoothing, SARIMA, "
            "Gradient Boosting, and XGBoost during rolling backtests. The frozen 16-week "
            "holdout achieved MAE of approximately 47,055 and sMAPE of 18.3%. Complexity "
            "was not rewarded merely for being complex."
        ),
        md(
            "## Context & Methods\n\n"
            "The target is weekly net revenue: valid sales value less cancellation/return "
            "value. The first and last partial calendar weeks are excluded. Models are "
            "selected using repeated expanding-window forecasts and evaluated on the latest "
            "16 complete weeks only after selection.\n\n"
            "### Key assumptions\n\n"
            "- Revenue uses the source currency; no currency conversion is available.\n"
            "- Future orders and active customers are unknown, so they are not forecast features.\n"
            "- Machine-learning lags and rolling features use prior weeks only.\n"
            "- The empirical 80% range uses historical backtest errors and is not guaranteed."
        ),
        code(
            "from pathlib import Path\nimport json, os, sys\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0,str(ROOT)); os.chdir(ROOT)\n"
            "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "from src.forecasting import complete_weekly_series, run_forecasting\n"
            "sns.set_theme(style='whitegrid')"
        ),
        md("## Data\n\n### 1. Prepare complete weekly observations"),
        code(
            "raw_weekly = pd.read_parquet(ROOT / 'data/processed/weekly_revenue.parquet')\n"
            "weekly = complete_weekly_series(raw_weekly)\n"
            "print(f'Raw weekly rows: {len(raw_weekly)}')\n"
            "print(f'Complete weekly rows: {len(weekly)}')\n"
            "display(weekly[['invoice_week','gross_revenue','returned_value','net_revenue']].head())\n"
            "fig, ax = plt.subplots(figsize=(13,5))\n"
            "ax.plot(weekly['invoice_week'],weekly['net_revenue'],color='#2563EB')\n"
            "ax.set(title='Weekly net revenue — complete weeks only',xlabel='',ylabel='Net revenue')\n"
            "plt.show()"
        ),
        md("### 2. Confirm lag features stop before the forecast week"),
        code(
            "from src.forecasting import make_supervised\n"
            "supervised = make_supervised(weekly)\n"
            "display(supervised.head())\n"
            "assert (supervised['lag_1'] != supervised['net_revenue']).any()\n"
            "print('Lag and rolling features are shifted before model fitting.')"
        ),
        md("## Rolling Backtest\n\n### 3. Execute all forecasting methods"),
        code(
            "metadata = run_forecasting(ROOT / 'reports/generated')\n"
            "print(json.dumps(metadata,indent=2))"
        ),
        md("### 4. Compare average rolling-backtest errors"),
        code(
            "comparison = pd.read_csv(ROOT / 'reports/generated/forecast_model_comparison.csv')\n"
            "display(comparison.style.format({c:'{:,.2f}' for c in comparison.columns if c not in ['model','folds']}))\n"
            "plot = comparison.sort_values('mae_mean',ascending=False)\n"
            "fig, ax = plt.subplots(figsize=(10,5))\n"
            "ax.barh(plot['model'],plot['mae_mean'],color='#2563EB')\n"
            "ax.set(title='Mean absolute error across rolling backtests',xlabel='MAE — lower is better',ylabel='')\n"
            "plt.show()"
        ),
        md(
            "A sophisticated model must beat the baselines consistently. The selected "
            "seasonal naïve model predicts each week from the corresponding week 52 weeks earlier."
        ),
        md("### 5. Inspect stability across forecast origins"),
        code(
            "folds = pd.read_csv(ROOT / 'reports/generated/forecast_backtest_metrics.csv',parse_dates=['cutoff'])\n"
            "selected = metadata['selected_model']\n"
            "focus = folds.loc[folds['model'].isin([selected,'exponential_smoothing','sarima','gradient_boosting','xgboost'])]\n"
            "fig, ax = plt.subplots(figsize=(12,5))\n"
            "for name, frame in focus.groupby('model'):\n"
            "    ax.plot(frame['cutoff'],frame['mae'],marker='o',label=name)\n"
            "ax.set(title='Backtest MAE by forecast origin',xlabel='Training data available through',ylabel='MAE')\n"
            "ax.legend(ncol=2); plt.show()"
        ),
        md("## Frozen Holdout\n\n### 6. Evaluate the selected model on the latest 16 complete weeks"),
        code(
            "holdout = pd.read_parquet(ROOT / 'reports/generated/forecast_final_test_predictions.parquet')\n"
            "display(pd.Series(metadata['final_test_metrics'],name='Final holdout').to_frame())\n"
            "fig, ax = plt.subplots(figsize=(12,5))\n"
            "ax.plot(holdout['invoice_week'],holdout['net_revenue'],marker='o',label='Actual',color='#2563EB')\n"
            "ax.plot(holdout['invoice_week'],holdout['predicted_net_revenue'],marker='o',label='Forecast',color='#D4A72C')\n"
            "ax.set(title='Frozen 16-week holdout — actual and forecast net revenue',xlabel='',ylabel='Net revenue')\n"
            "ax.legend(); plt.show()"
        ),
        md(
            "MAE is expressed in source-currency units. RMSE is higher because it penalizes "
            "large weekly misses. sMAPE gives a percentage-like error but should not be read "
            "as a guaranteed percentage accuracy."
        ),
        md("## Revenue Outlook\n\n### 7. Display the eight-week point forecast and empirical range"),
        code(
            "future = pd.read_parquet(ROOT / 'data/processed/revenue_forecast.parquet')\n"
            "display(future)\n"
            "fig, ax = plt.subplots(figsize=(12,5))\n"
            "ax.plot(future['invoice_week'],future['forecast_net_revenue'],marker='o',color='#2563EB',label='Expected net revenue')\n"
            "ax.fill_between(future['invoice_week'],future['lower_80'],future['upper_80'],color='#93C5FD',alpha=.35,label='Empirical 80% range')\n"
            "ax.set(title='Eight-week net-revenue outlook',xlabel='',ylabel='Net revenue')\n"
            "ax.legend(); plt.show()"
        ),
        md(
            "## Limitations\n\n"
            "- Only 104 complete weeks are available.\n"
            "- The annual seasonal baseline is supported by roughly two cycles only.\n"
            "- Promotions, prices, inventory, holidays, marketing, and macroeconomic drivers are unavailable.\n"
            "- The dataset is historical and represents one retailer.\n"
            "- The empirical interval may not achieve 80% coverage in a changed environment."
        ),
        md(
            "## Takeaways\n\n"
            "- The simple annual seasonal baseline won the rolling comparison.\n"
            "- XGBoost and Gradient Boosting did not justify their added complexity.\n"
            "- Final holdout error is reported separately from model-selection backtests.\n"
            "- Forecast ranges communicate uncertainty and should not be treated as guarantees.\n"
            "- A production forecast would require more history and external business drivers."
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
