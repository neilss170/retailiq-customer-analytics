"""Build the reader-facing point-in-time feature-engineering notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "04_return_risk_feature_engineering.ipynb"


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}
    notebook.cells = [
        markdown(
            "# Customer Return-Risk Feature Engineering\n\n"
            "## tl;dr\n\n"
            "This notebook converts historical transactions into point-in-time customer "
            "snapshots. Every feature stops before its observation date, while the label "
            "uses the following 90 days. A 90-day purge between evaluation blocks prevents "
            "training labels from using purchases that occur after validation features."
        ),
        markdown(
            "## Context & Methods\n\n"
            "The business question is: **which known customers are likely to purchase "
            "again during the next 90 days?** The prediction is an estimated likelihood, "
            "not proof that a customer has permanently churned.\n\n"
            "```text\n"
            "historical transactions → observation date → 90-day outcome window\n"
            "       features stop here ↑                 ↑ target only\n"
            "```\n\n"
            "### Key assumptions\n\n"
            "- Only customers with at least 90 days of observed history are eligible.\n"
            "- Anonymous purchases cannot support customer-level prediction.\n"
            "- A valid future sale counts as a return; cancellations do not.\n"
            "- Customer ID identifies rows but is never a model feature.\n"
            "- The test file remains sealed until the final-test notebook."
        ),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(ROOT))\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from src.return_risk_features import assert_no_temporal_leakage\n"
            "sns.set_theme(style='whitegrid')\n"
            "pd.options.display.max_columns = 60\n"
            "pd.options.display.float_format = '{:,.3f}'.format"
        ),
        markdown("## Data\n\n### 1. Load development partitions only"),
        code(
            "train = pd.read_parquet(ROOT / 'data/processed/return_risk_train.parquet')\n"
            "validation = pd.read_parquet(ROOT / 'data/processed/return_risk_validation.parquet')\n"
            "metadata = json.loads((ROOT / 'data/processed/return_risk_split_metadata.json').read_text())\n"
            "print(f'Training snapshots: {len(train):,}')\n"
            "print(f'Validation snapshots: {len(validation):,}')\n"
            "print('The final test partition is intentionally not loaded.')"
        ),
        markdown("### 2. Confirm target and snapshot grain"),
        code(
            "target = 'returned_within_90_days'\n"
            "assert train['snapshot_id'].is_unique\n"
            "assert validation['snapshot_id'].is_unique\n"
            "assert set(train[target].unique()).issubset({0, 1})\n"
            "assert set(validation[target].unique()).issubset({0, 1})\n"
            "grain = pd.Series({\n"
            " 'training snapshots': len(train),\n"
            " 'training customers': train['customer_id'].nunique(),\n"
            " 'training observation dates': train['observation_date'].nunique(),\n"
            " 'validation snapshots': len(validation),\n"
            " 'validation customers': validation['customer_id'].nunique(),\n"
            " 'validation observation dates': validation['observation_date'].nunique(),\n"
            "})\n"
            "display(grain.to_frame('count'))"
        ),
        markdown(
            "## Feature Engineering\n\n"
            "### 3. Review the model feature catalogue\n\n"
            "Lifetime features summarize all valid history before the observation date. "
            "Window features describe only the most recent 30, 60, 90, or 180 days."
        ),
        code(
            "excluded = {'customer_id','snapshot_id','observation_date','outcome_end',"
            "'first_purchase','last_purchase',target}\n"
            "model_features = [column for column in train.columns if column not in excluded]\n"
            "feature_groups = {\n"
            " 'Customer context': ['primary_country'],\n"
            " 'Recency and tenure': ['recency_days','tenure_days'],\n"
            " 'Lifetime behaviour': [c for c in model_features if c.endswith('_lifetime')],\n"
            " 'Order behaviour': ['average_order_value','average_basket_units','order_value_std','orders_per_30_days'],\n"
            " 'Recent windows': [c for c in model_features if c.endswith(('30d','60d','90d','180d'))],\n"
            " 'Behaviour change': ['recent_revenue_change','recent_order_change'],\n"
            " 'Return history': ['historical_return_value_rate'],\n"
            "}\n"
            "catalogue = pd.DataFrame(\n"
            " [(group, feature) for group, features in feature_groups.items() for feature in features],\n"
            " columns=['feature_group','feature']\n"
            ")\n"
            "display(catalogue)\n"
            "catalogue.to_csv(ROOT / 'reports/generated/return_risk_feature_catalogue.csv', index=False)\n"
            "print(f'Model features: {len(model_features)}')"
        ),
        markdown("### 4. Inspect feature completeness and distributions"),
        code(
            "feature_quality = pd.DataFrame({\n"
            " 'dtype': train[model_features].dtypes.astype(str),\n"
            " 'missing_rate': train[model_features].isna().mean(),\n"
            " 'distinct_values': train[model_features].nunique(dropna=True),\n"
            "}).sort_values(['missing_rate','distinct_values'], ascending=[False,True])\n"
            "display(feature_quality)\n"
            "numeric = train[model_features].select_dtypes(include='number').columns.tolist()\n"
            "display(train[numeric].describe(percentiles=[.01,.25,.5,.75,.95,.99]).T)"
        ),
        markdown(
            "### 5. Show why transformations belong inside the model pipeline\n\n"
            "Revenue, frequency, and basket variables have long right tails. They are not "
            "deleted as outliers because high-value customers are commercially meaningful. "
            "Imputation, scaling, and any transformation must be fitted on training data only."
        ),
        code(
            "plot_features = ['recency_days','orders_lifetime','revenue_lifetime','average_order_value']\n"
            "fig, axes = plt.subplots(2,2,figsize=(12,8))\n"
            "for feature, ax in zip(plot_features, axes.flat):\n"
            "    train[feature].clip(upper=train[feature].quantile(.99)).hist(bins=35, ax=ax, color='#2563EB')\n"
            "    ax.set_title(f'{feature.replace(\"_\",\" \").title()} — capped at training p99')\n"
            "    ax.set_xlabel(feature.replace('_',' ')); ax.set_ylabel('Snapshots')\n"
            "plt.tight_layout(); plt.show()"
        ),
        markdown("## Chronological Splitting\n\n### 6. Inspect periods and class balance"),
        code(
            "split_summary = pd.DataFrame([\n"
            " {'partition':'Training','rows':len(train),'start':train['observation_date'].min(),"
            "'end':train['observation_date'].max(),'return_rate':train[target].mean()},\n"
            " {'partition':'Validation','rows':len(validation),'start':validation['observation_date'].min(),"
            "'end':validation['observation_date'].max(),'return_rate':validation[target].mean()},\n"
            "])\n"
            "display(split_summary)\n"
            "balance = pd.concat([\n"
            " train.assign(partition='Training'), validation.assign(partition='Validation')\n"
            "]).groupby(['observation_date','partition'])[target].mean().reset_index()\n"
            "fig, ax = plt.subplots(figsize=(12,4))\n"
            "for name, frame in balance.groupby('partition'):\n"
            "    ax.plot(frame['observation_date'], frame[target], marker='o', label=name)\n"
            "ax.set(title='Observed 90-day return rate by snapshot date', xlabel='', ylabel='Return rate')\n"
            "ax.legend(); plt.show()"
        ),
        markdown(
            "### 7. Explain why the final row shares are not exactly 70/15/15\n\n"
            "Cut points begin from a chronological 70/15/15 design. Rows near each boundary "
            "are then purged when their 90-day outcomes overlap the next partition. Customer "
            "volume also grows over time. Preserving exact row percentages would be less "
            "important than preventing look-ahead."
        ),
        code(
            "display(pd.Series(metadata, name='value').to_frame())\n"
            "print('Training maximum outcome end:', train['outcome_end'].max())\n"
            "print('Validation minimum observation:', validation['observation_date'].min())\n"
            "print('Validation maximum outcome end:', validation['outcome_end'].max())"
        ),
        markdown("## Leakage Audit\n\n### 8. Execute temporal leakage assertions"),
        code(
            "# A minimal sealed placeholder supplies only the test boundary required by the assertion.\n"
            "sealed_test_boundary = pd.DataFrame({\n"
            " 'observation_date': [pd.Timestamp(metadata['test_start'])],\n"
            " 'outcome_end': [pd.Timestamp(metadata['test_start']) + pd.Timedelta(days=90)],\n"
            " 'last_purchase': [pd.Timestamp(metadata['test_start']) - pd.Timedelta(days=1)],\n"
            "})\n"
            "assert_no_temporal_leakage(train, validation, sealed_test_boundary)\n"
            "assert train['last_purchase'].lt(train['observation_date']).all()\n"
            "assert validation['last_purchase'].lt(validation['observation_date']).all()\n"
            "assert train['outcome_end'].le(validation['observation_date'].min()).all()\n"
            "assert validation['outcome_end'].le(pd.Timestamp(metadata['test_start'])).all()\n"
            "print('PASS — features stop before observation dates.')\n"
            "print('PASS — training outcomes end before validation begins.')\n"
            "print('PASS — validation outcomes end before the sealed test boundary.')"
        ),
        markdown("### 9. Confirm forbidden fields are excluded from modelling"),
        code(
            "forbidden = {'customer_id','snapshot_id','observation_date','outcome_end',"
            "'first_purchase','last_purchase',target}\n"
            "assert forbidden.isdisjoint(model_features)\n"
            "display(pd.DataFrame({'excluded_field': sorted(forbidden), 'reason': [\n"
            " 'Entity identifier, not behaviour','Row identifier','Split/control timestamp',"
            " 'Outcome boundary','Raw timestamp summarized safely','Raw timestamp summarized safely',"
            " 'Prediction target']\n"
            "}))"
        ),
        markdown(
            "## Takeaways\n\n"
            "- The modelling unit is one customer at one observation date.\n"
            "- Twenty-eight behavioural features are available before prediction time.\n"
            "- Customer identifiers and raw control dates are excluded from the model.\n"
            "- Boundary purging is more important than exact partition percentages.\n"
            "- The test dataset exists as a sealed file and is not loaded here.\n"
            "- Differences in return rate over time mean shuffled cross-validation would be misleading."
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
