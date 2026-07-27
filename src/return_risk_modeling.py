"""Compare and tune leakage-safe customer return-risk classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

TARGET = "returned_within_90_days"
EXCLUDED = {
    TARGET,
    "customer_id",
    "snapshot_id",
    "observation_date",
    "outcome_end",
    "first_purchase",
    "last_purchase",
}


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = ["primary_country"]
    numerical = [
        column
        for column in frame.columns
        if column not in EXCLUDED.union(categorical)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return numerical, categorical


def build_preprocessor(numerical: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    category_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numerical), ("category", category_pipe, categorical)],
        verbose_feature_names_out=False,
    )


def candidate_models(random_state: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2_000, class_weight="balanced", random_state=random_state
        ),
        "gaussian_naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=25, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def metrics(y_true: pd.Series, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
    }


def compare_models(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    numerical, categorical = feature_columns(train)
    X_train = train[numerical + categorical]
    y_train = train[TARGET]
    X_validation = validation[numerical + categorical]
    y_validation = validation[TARGET]
    rows: list[dict[str, object]] = []
    fitted: dict[str, Pipeline] = {}

    for name, estimator in candidate_models().items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(numerical, categorical)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        train_probability = pipeline.predict_proba(X_train)[:, 1]
        validation_probability = pipeline.predict_proba(X_validation)[:, 1]
        row = {"model": name}
        row.update({f"train_{key}": value for key, value in metrics(y_train, train_probability).items()})
        row.update(
            {
                f"validation_{key}": value
                for key, value in metrics(y_validation, validation_probability).items()
            }
        )
        rows.append(row)
        fitted[name] = pipeline
    return pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False), fitted


def tune_tree_models(
    train: pd.DataFrame, numerical: list[str], categorical: list[str]
) -> dict[str, RandomizedSearchCV]:
    X = train[numerical + categorical]
    y = train[TARGET]
    splitter = TimeSeriesSplit(n_splits=5)
    searches: dict[str, RandomizedSearchCV] = {}
    specifications = {
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced_subsample", n_jobs=-1, random_state=42
            ),
            {
                "model__n_estimators": randint(200, 601),
                "model__max_depth": randint(5, 19),
                "model__min_samples_leaf": randint(2, 21),
                "model__max_features": ["sqrt", 0.5, 0.8],
            },
        ),
        "xgboost": (
            XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=42),
            {
                "model__n_estimators": randint(150, 501),
                "model__max_depth": randint(3, 9),
                "model__learning_rate": uniform(0.025, 0.175),
                "model__subsample": uniform(0.70, 0.30),
                "model__colsample_bytree": uniform(0.70, 0.30),
                "model__min_child_weight": randint(1, 10),
            },
        ),
    }
    for name, (estimator, distribution) in specifications.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(numerical, categorical)),
                ("model", estimator),
            ]
        )
        search = RandomizedSearchCV(
            pipeline,
            distribution,
            n_iter=10,
            scoring="average_precision",
            cv=splitter,
            n_jobs=-1,
            random_state=42,
            refit=True,
            return_train_score=True,
        )
        search.fit(X, y)
        searches[name] = search
    return searches


def focused_grid(
    name: str,
    randomized: RandomizedSearchCV,
    train: pd.DataFrame,
    numerical: list[str],
    categorical: list[str],
) -> GridSearchCV:
    best = randomized.best_params_
    if name == "random_forest":
        grid = {
            "model__n_estimators": sorted({max(100, best["model__n_estimators"] - 100), best["model__n_estimators"]}),
            "model__max_depth": sorted({max(3, best["model__max_depth"] - 2), best["model__max_depth"]}),
            "model__min_samples_leaf": sorted({best["model__min_samples_leaf"], best["model__min_samples_leaf"] + 3}),
            "model__max_features": [best["model__max_features"]],
        }
    else:
        grid = {
            "model__n_estimators": sorted({max(100, best["model__n_estimators"] - 75), best["model__n_estimators"]}),
            "model__max_depth": sorted({max(2, best["model__max_depth"] - 1), best["model__max_depth"]}),
            "model__learning_rate": [best["model__learning_rate"], max(0.01, best["model__learning_rate"] * 0.75)],
            "model__subsample": [best["model__subsample"]],
            "model__colsample_bytree": [best["model__colsample_bytree"]],
            "model__min_child_weight": [best["model__min_child_weight"]],
        }
    search = GridSearchCV(
        clone(randomized.estimator),
        grid,
        scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=5),
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(train[numerical + categorical], train[TARGET])
    return search


def choose_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.arange(0.20, 0.81, 0.01):
        prediction = probability >= threshold
        false_negative = int(((y_true == 1) & ~prediction).sum())
        false_positive = int(((y_true == 0) & prediction).sum())
        row = {"threshold": threshold, "business_cost": 2 * false_negative + false_positive}
        row.update(metrics(y_true, probability, threshold))
        rows.append(row)
    table = pd.DataFrame(rows)
    best = table.sort_values(["business_cost", "pr_auc"], ascending=[True, False]).iloc[0]
    return float(best["threshold"]), table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/return_risk_train.parquet"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/return_risk_validation.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    train = pd.read_parquet(args.train).sort_values(["observation_date", "customer_id"])
    validation = pd.read_parquet(args.validation).sort_values(["observation_date", "customer_id"])
    numerical, categorical = feature_columns(train)
    comparison, fitted = compare_models(train, validation)

    searches = tune_tree_models(train, numerical, categorical)
    randomized_summary = pd.DataFrame(
        [
            {"model": name, "cv_pr_auc": search.best_score_, **search.best_params_}
            for name, search in searches.items()
        ]
    )
    random_winner = randomized_summary.sort_values("cv_pr_auc", ascending=False).iloc[0]["model"]
    final_search = focused_grid(random_winner, searches[random_winner], train, numerical, categorical)
    final_pipeline = final_search.best_estimator_

    X_validation = validation[numerical + categorical]
    probability = final_pipeline.predict_proba(X_validation)[:, 1]
    threshold, threshold_table = choose_threshold(validation[TARGET], probability)
    final_metrics = metrics(validation[TARGET], probability, threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "classification_default_comparison.csv", index=False)
    randomized_summary.to_csv(args.output_dir / "classification_randomized_search.csv", index=False)
    threshold_table.to_csv(args.output_dir / "classification_thresholds.csv", index=False)
    pd.DataFrame([{"model": random_winner, "threshold": threshold, **final_metrics}]).to_csv(
        args.output_dir / "classification_validation_metrics.csv", index=False
    )
    joblib.dump(final_pipeline, args.model_dir / "return_risk_pipeline.joblib")
    metadata = {
        "selected_model": random_winner,
        "threshold": threshold,
        "best_params": final_search.best_params_,
        "features": numerical + categorical,
        "validation_metrics": final_metrics,
        "test_evaluated": False,
    }
    (args.model_dir / "return_risk_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(comparison[["model", "validation_pr_auc", "validation_roc_auc", "validation_f1"]].to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
