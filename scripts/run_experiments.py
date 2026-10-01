"""Run reproducible credit-risk experiments and track them with MLflow."""

import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pipeline.config import RAW_FEATURES, TARGET
from pipeline.preprocessing import build_preprocessor
from pipeline.training import build_model, build_pipeline

ROOT = Path(__file__).resolve().parents[1]


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series,
             threshold: float) -> dict[str, float]:
    """Evaluate probability quality and decisions at a fixed threshold."""
    probability = model.predict_proba(X)[:, 1]
    prediction = probability >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y, probability)),
        "positive_rate": float(prediction.mean()),
        "class_prevalence": float(y.mean()),
    }


def main() -> None:
    """Train ten candidates, select by validation PR AUC, then test once."""
    config_path = ROOT / "config" / "experiment.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    seed = int(config["seed"])
    threshold = float(config["threshold"])
    random.seed(seed)
    np.random.seed(seed)

    output = ROOT / "artifacts" / "assignment2"
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = ROOT / config["dataset"]
    frame = pd.read_csv(dataset_path)

    required = set(RAW_FEATURES + [TARGET])
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if len(frame) < 5000 or frame[RAW_FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Dataset failed row-count or missing-value gate.")
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain both binary classes.")

    X, y = frame[RAW_FEATURES], frame[TARGET]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=seed
    )
    split_path = output / "split_indices.json"
    split_path.write_text(json.dumps({
        "train": X_train.index.tolist(),
        "validation": X_val.index.tolist(),
        "test": X_test.index.tolist(),
    }), encoding="utf-8")

    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=False
    )
    commit = git_result.stdout.strip() or "unknown"
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    experiment = mlflow.set_experiment(config["experiment_name"])
    rows = []
    best_model = None
    best_row = None

    print(f"Train={len(X_train)} Validation={len(X_val)} Test={len(X_test)}")
    for item in config["experiments"]:
        print(f"Running {item['id']} / {item['model']}", flush=True)
        params = {**item["params"], "random_state": seed}
        if item["engineered"]:
            model = build_pipeline(item["model"], list(X.columns), **params)
        else:
            model = Pipeline([
                ("features", build_preprocessor(list(X.columns))),
                ("classifier", build_model(item["model"], **params)),
            ])

        with mlflow.start_run(run_name=item["id"]) as run:
            mlflow.set_tags({
                "git_commit": commit,
                "dataset_sha256": dataset_hash,
                "purpose": "validation_selection",
            })
            mlflow.log_params({
                **params, "model_type": item["model"],
                "engineered": item["engineered"], "threshold": threshold,
                "train_rows": len(X_train), "validation_rows": len(X_val),
            })
            mlflow.log_artifact(str(config_path))
            mlflow.log_artifact(str(split_path))

            started = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - started
            metrics = {
                f"val_{name}": value
                for name, value in evaluate(model, X_val, y_val, threshold).items()
            }

            sample = X_val.iloc[:1]
            for _ in range(5):
                model.predict_proba(sample)
            timings = []
            for _ in range(50):
                started = time.perf_counter()
                model.predict_proba(sample)
                timings.append((time.perf_counter() - started) * 1000)
            metrics.update({
                "fit_seconds": fit_seconds,
                "predict_one_p95_ms": float(np.percentile(timings, 95)),
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model, artifact_path="model",
                signature=infer_signature(
                    X_train.head(5),
                    model.predict_proba(X_train.head(5))[:, 1],
                ),
                input_example=X_train.head(5),
                pyfunc_predict_fn="predict_proba",
                code_paths=[str(ROOT / "pipeline")],
            )
            row = {
                "experiment_id": item["id"], "model": item["model"],
                "engineered": item["engineered"],
                "params": json.dumps(params),
                "run_id": run.info.run_id, **metrics,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "results.csv", index=False)
            if best_row is None or row[config["primary_metric"]] > best_row[config["primary_metric"]]:
                best_model, best_row = model, row

    assert best_model is not None and best_row is not None
    test_metrics = evaluate(best_model, X_test, y_test, threshold)
    summary = {
        "selected_experiment": best_row["experiment_id"],
        "selection_metric": config["primary_metric"],
        "validation_score": best_row[config["primary_metric"]],
        "selected_run_id": best_row["run_id"],
        "test_metrics": test_metrics,
        "dataset_sha256": dataset_hash,
        "git_commit": commit,
    }
    summary_path = output / "selected_model.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with mlflow.start_run(run_id=best_row["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifact(str(output / "results.csv"))
        mlflow.log_artifact(str(summary_path))
        mlflow.set_tag("selection", "highest_validation_pr_auc")

    # Registration creates a candidate version, not a production deployment.
    version = mlflow.register_model(
        f"runs:/{best_row['run_id']}/model", "assignment2-credit-risk"
    )
    client = mlflow.MlflowClient()
    client.set_model_version_tag(
        version.name, version.version, "status", "candidate"
    )
    print(pd.DataFrame(rows).sort_values(
        config["primary_metric"], ascending=False
    ).to_string(index=False))
    print(f"\nSelected: {best_row['experiment_id']}")
    print(json.dumps(test_metrics, indent=2))
    print(f"MLflow experiment ID: {experiment.experiment_id}")


if __name__ == "__main__":
    main()
