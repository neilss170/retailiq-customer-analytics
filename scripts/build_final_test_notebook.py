"""Build the one-time final classification test notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "06_final_classification_test.ipynb"


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
            "# Final Customer Return-Risk Test\n\n## tl;dr\n\n"
            "This notebook applies the frozen Gradient Boosting pipeline and the previously "
            "selected 0.37 campaign cutoff to the sealed test period. No feature, model, "
            "hyperparameter, or cutoff is changed after seeing these results."
        ),
        md(
            "## Finalization contract\n\n"
            "- Model: frozen from validation-stage selection\n"
            "- Campaign cutoff: frozen at 0.37\n"
            "- Test partition: opened by the evaluation function once\n"
            "- Later reruns: display saved predictions and metrics without reselection\n"
            "- Purpose: estimate performance in the latest historical period"
        ),
        code(
            "from pathlib import Path\nimport json, os, sys\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(ROOT)); os.chdir(ROOT)\n"
            "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "from src.final_test_evaluation import evaluate_once\n"
            "sns.set_theme(style='whitegrid')"
        ),
        md("## Results\n\n### 1. Run or recover the immutable final evaluation"),
        code(
            "result = evaluate_once()\n"
            "print(json.dumps(result, indent=2))\n"
            "assert result['evaluated_once'] is True"
        ),
        md("### 2. Compare validation and final-test metrics"),
        code(
            "metadata = json.loads((ROOT / 'models/return_risk_metadata.json').read_text())\n"
            "validation = pd.Series(metadata['validation_metrics'], name='Validation')\n"
            "test = pd.Series(result['metrics'], name='Final test')\n"
            "comparison = pd.concat([validation, test], axis=1)\n"
            "display(comparison.style.format('{:.3f}'))\n"
            "comparison.plot.bar(figsize=(10,5), color=['#94A3B8','#2563EB'])\n"
            "plt.title('Validation and final-test performance'); plt.ylabel('Score'); plt.xticks(rotation=0); plt.show()"
        ),
        md("### 3. Review final confusion matrix"),
        code(
            "cm = result['confusion_matrix']\n"
            "matrix = pd.DataFrame(\n"
            " [[cm['true_negative'],cm['false_positive']],[cm['false_negative'],cm['true_positive']]],\n"
            " index=['Actually did not return','Actually returned'],\n"
            " columns=['Not selected','Selected for campaign'])\n"
            "display(matrix)\n"
            "sns.heatmap(matrix, annot=True, fmt=',', cmap='Blues')\n"
            "plt.title('Final-test campaign selection outcomes'); plt.xlabel('Model decision'); plt.ylabel('Observed outcome'); plt.show()"
        ),
        md("### 4. Check final-period calibration"),
        code(
            "calibration = pd.read_csv(ROOT / 'reports/generated/classification_final_test_calibration.csv')\n"
            "plt.figure(figsize=(7,6))\n"
            "plt.plot(calibration['predicted_probability'], calibration['observed_return_rate'], marker='o', color='#2563EB', label='Final test')\n"
            "plt.plot([0,1],[0,1], '--', color='#334155', label='Ideal calibration')\n"
            "plt.title('Final-test calibration'); plt.xlabel('Estimated likelihood'); plt.ylabel('Observed return rate'); plt.legend(); plt.show()"
        ),
        md("### 5. Inspect performance by recency without changing the model"),
        code(
            "scored = pd.read_parquet(ROOT / 'reports/generated/classification_final_test_predictions.parquet')\n"
            "scored['recency_band'] = pd.cut(scored['recency_days'], [-float('inf'),30,90,180,float('inf')], labels=['0–30 days','31–90 days','91–180 days','181+ days'])\n"
            "segment = scored.groupby('recency_band', observed=True).agg(\n"
            " snapshots=('snapshot_id','size'), actual_return_rate=('returned_within_90_days','mean'),\n"
            " average_estimated_likelihood=('estimated_return_likelihood','mean'), campaign_selection_rate=('selected_for_campaign','mean'))\n"
            "display(segment)"
        ),
        md(
            "## Interpretation rules\n\n"
            "Test results are final evidence, not a new tuning signal. Any weakness found "
            "here becomes a limitation or future experiment; it does not authorize changing "
            "the current model and reporting a second test score."
        ),
        md(
            "## Takeaways\n\n"
            "- The latest historical period is evaluated using a frozen model and cutoff.\n"
            "- Validation-to-test changes reveal temporal generalization, not tuning opportunities.\n"
            "- Confusion-matrix counts translate model performance into campaign-selection consequences.\n"
            "- Calibration shows whether estimated likelihoods match observed rates.\n"
            "- The source covers an old, single-retailer period, so external validity remains limited."
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
