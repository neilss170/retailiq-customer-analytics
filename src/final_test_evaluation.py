"""Evaluate the frozen return-risk pipeline on the sealed test partition once."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from src.return_risk_modeling import TARGET, metrics


MODEL_PATH = Path("models/return_risk_pipeline.joblib")
METADATA_PATH = Path("models/return_risk_metadata.json")
TEST_PATH = Path("data/processed/return_risk_test.parquet")
OUTPUT_DIR = Path("reports/generated")


def evaluate_once() -> dict[str, object]:
    """Score the sealed partition once; later calls return saved evidence."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "classification_final_test_metrics.json"
    prediction_path = OUTPUT_DIR / "classification_final_test_predictions.parquet"

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("test_evaluated", False):
        if not result_path.exists() or not prediction_path.exists():
            raise RuntimeError(
                "Metadata says the test was evaluated, but saved evidence is missing."
            )
        return json.loads(result_path.read_text(encoding="utf-8"))

    pipeline = joblib.load(MODEL_PATH)
    test = pd.read_parquet(TEST_PATH).sort_values(
        ["observation_date", "customer_id"]
    )
    feature_names = list(pipeline.feature_names_in_)
    forbidden = {
        TARGET,
        "customer_id",
        "snapshot_id",
        "observation_date",
        "outcome_end",
        "first_purchase",
        "last_purchase",
    }
    if forbidden.intersection(feature_names):
        raise AssertionError("Frozen pipeline includes forbidden leakage fields")

    threshold = float(metadata["campaign_threshold"])
    probability = pipeline.predict_proba(test[feature_names])[:, 1]
    prediction = (probability >= threshold).astype(int)
    score = metrics(test[TARGET], probability, threshold)
    matrix = confusion_matrix(test[TARGET], prediction, labels=[0, 1])

    observed, predicted = calibration_curve(
        test[TARGET], probability, n_bins=10, strategy="quantile"
    )
    calibration = pd.DataFrame(
        {
            "predicted_probability": predicted,
            "observed_return_rate": observed,
        }
    )
    calibration.to_csv(
        OUTPUT_DIR / "classification_final_test_calibration.csv", index=False
    )

    scored = test[
        [
            "snapshot_id",
            "customer_id",
            "observation_date",
            "recency_days",
            "revenue_lifetime",
            TARGET,
        ]
    ].copy()
    scored["estimated_return_likelihood"] = probability
    scored["selected_for_campaign"] = prediction
    scored.to_parquet(prediction_path, index=False)

    validation_metrics = metadata["validation_metrics"]
    result: dict[str, object] = {
        "model": metadata["selected_model"],
        "campaign_threshold": threshold,
        "test_rows": int(len(test)),
        "test_period_start": test["observation_date"].min().isoformat(),
        "test_period_end": test["observation_date"].max().isoformat(),
        "actual_return_rate": float(test[TARGET].mean()),
        "metrics": score,
        "confusion_matrix": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "validation_to_test_change": {
            key: float(score[key] - validation_metrics[key])
            for key in score
            if key in validation_metrics
        },
        "evaluated_once": True,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    metadata["test_evaluated"] = True
    metadata["test_metrics_path"] = str(result_path.as_posix())
    metadata["test_predictions_path"] = str(prediction_path.as_posix())
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate_once(), indent=2))
