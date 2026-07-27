"""Validation-only model selection, calibration, error analysis, and explainability."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier

from src.return_risk_modeling import (
    TARGET,
    build_preprocessor,
    choose_threshold,
    feature_columns,
    metrics,
)


def evaluate_validation_candidates(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    numerical, categorical = feature_columns(train)
    columns = numerical + categorical
    X_train, y_train = train[columns], train[TARGET]
    X_validation, y_validation = validation[columns], validation[TARGET]
    preprocess = lambda: build_preprocessor(numerical, categorical)

    candidates: dict[str, Pipeline] = {
        "logistic_class_weight": Pipeline(
            [
                ("preprocessor", preprocess()),
                ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42)),
            ]
        ),
        "logistic_smote": ImbPipeline(
            [
                ("preprocessor", preprocess()),
                ("smote", SMOTE(random_state=42)),
                ("model", LogisticRegression(max_iter=2_000, random_state=42)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", preprocess()),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        "tuned_random_forest": Pipeline(
            [
                ("preprocessor", preprocess()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=506,
                        max_depth=16,
                        max_features=0.8,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "tuned_xgboost": Pipeline(
            [
                ("preprocessor", preprocess()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=198,
                        max_depth=7,
                        learning_rate=0.13203930308544914,
                        subsample=0.8574323980775167,
                        colsample_bytree=0.8852444528883149,
                        min_child_weight=9,
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    probabilities: dict[str, np.ndarray] = {
        "majority_baseline": np.repeat(float(y_train.mean()), len(validation)),
        "recency_heuristic": 1 / (1 + np.exp((validation["recency_days"].to_numpy() - 60) / 30)),
    }
    fitted: dict[str, Pipeline] = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        probabilities[name] = pipeline.predict_proba(X_validation)[:, 1]

    rows = []
    for name, probability in probabilities.items():
        row = {"model": name}
        row.update(metrics(y_validation, probability))
        row["brier_score"] = brier_score_loss(y_validation, probability)
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(
        ["pr_auc", "roc_auc"], ascending=False
    )
    return comparison, fitted, probabilities


def calibration_table(
    y_true: pd.Series, probabilities: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for name, probability in probabilities.items():
        observed, predicted = calibration_curve(
            y_true, probability, n_bins=10, strategy="quantile"
        )
        rows.extend(
            {
                "model": name,
                "predicted_probability": float(pred),
                "observed_return_rate": float(obs),
            }
            for pred, obs in zip(predicted, observed)
        )
    return pd.DataFrame(rows)


def segment_error_table(
    validation: pd.DataFrame, probability: np.ndarray, threshold: float
) -> pd.DataFrame:
    scored = validation[[TARGET, "recency_days", "revenue_lifetime"]].copy()
    scored["probability"] = probability
    scored["recency_band"] = pd.cut(
        scored["recency_days"],
        bins=[-np.inf, 30, 90, 180, np.inf],
        labels=["0–30 days", "31–90 days", "91–180 days", "181+ days"],
    )
    rows = []
    for band, frame in scored.groupby("recency_band", observed=True):
        row = {"recency_band": str(band), "snapshots": len(frame)}
        row.update(metrics(frame[TARGET], frame["probability"], threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def save_tree_visuals(
    train: pd.DataFrame, numerical: list[str], categorical: list[str], output: Path
) -> None:
    X = train[numerical + categorical]
    y = train[TARGET]
    preprocessor = build_preprocessor(numerical, categorical)
    transformed = preprocessor.fit_transform(X)
    names = preprocessor.get_feature_names_out()

    decision_tree = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=50, class_weight="balanced", random_state=42
    ).fit(transformed, y)
    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    ).fit(transformed, y)

    for filename, estimator, title in (
        ("decision_tree_structure.png", decision_tree, "Decision Tree — first three levels"),
        ("random_forest_example_tree.png", forest.estimators_[0], "Random Forest — one representative tree"),
    ):
        fig, ax = plt.subplots(figsize=(24, 10))
        plot_tree(
            estimator,
            feature_names=names,
            class_names=["No return", "Returned"],
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=7,
            ax=ax,
        )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)


def run(output_dir: Path = Path("reports/generated")) -> dict[str, object]:
    train = pd.read_parquet("data/processed/return_risk_train.parquet").sort_values(
        ["observation_date", "customer_id"]
    )
    validation = pd.read_parquet(
        "data/processed/return_risk_validation.parquet"
    ).sort_values(["observation_date", "customer_id"])
    numerical, categorical = feature_columns(train)
    comparison, fitted, probabilities = evaluate_validation_candidates(train, validation)
    selected_name = comparison.loc[
        ~comparison["model"].isin(["majority_baseline", "recency_heuristic"])
    ].iloc[0]["model"]
    selected_pipeline = fitted[selected_name]
    selected_probability = probabilities[selected_name]
    threshold, threshold_table = choose_threshold(
        validation[TARGET], selected_probability
    )
    selected_metrics = metrics(validation[TARGET], selected_probability, threshold)
    cv_results = cross_validate(
        selected_pipeline,
        train[numerical + categorical],
        train[TARGET],
        cv=TimeSeriesSplit(n_splits=5),
        scoring={
            "pr_auc": "average_precision",
            "roc_auc": "roc_auc",
            "f1": "f1",
        },
        n_jobs=-1,
        return_train_score=True,
    )
    cv_summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "validation_mean": float(np.mean(cv_results[f"test_{metric}"])),
                "validation_std": float(np.std(cv_results[f"test_{metric}"])),
                "training_mean": float(np.mean(cv_results[f"train_{metric}"])),
            }
            for metric in ("pr_auc", "roc_auc", "f1")
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path("reports/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "classification_candidate_comparison.csv", index=False)
    cv_summary.to_csv(output_dir / "classification_final_cv_summary.csv", index=False)
    threshold_table.to_csv(output_dir / "classification_thresholds_final.csv", index=False)
    calibration_table(validation[TARGET], probabilities).to_csv(
        output_dir / "classification_calibration.csv", index=False
    )
    segment_error_table(validation, selected_probability, threshold).to_csv(
        output_dir / "classification_segment_errors.csv", index=False
    )

    preprocessor = selected_pipeline.named_steps["preprocessor"]
    transformed_sample = preprocessor.transform(
        validation[numerical + categorical].sample(
            min(600, len(validation)), random_state=42
        )
    )
    explainer = shap.TreeExplainer(selected_pipeline.named_steps["model"])
    shap_values = explainer.shap_values(transformed_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    importance = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(output_dir / "classification_shap_importance.csv", index=False)
    save_tree_visuals(train, numerical, categorical, figure_dir)

    joblib.dump(selected_pipeline, "models/return_risk_pipeline.joblib")
    metadata = {
        "selected_model": selected_name,
        "selection_metric": "validation_pr_auc",
        "campaign_threshold": threshold,
        "validation_metrics": selected_metrics,
        "training_return_rate": float(train[TARGET].mean()),
        "smote_note": "Evaluated inside the training pipeline; training classes were already nearly balanced.",
        "test_evaluated": False,
    }
    Path("models/return_risk_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
