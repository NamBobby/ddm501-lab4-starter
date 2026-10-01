"""Run isolated stages of the Assignment 2 training pipeline.

Artifacts are separated by run ID. Validation gates precede registration.
The exported model is a staging artifact, not an automatic API deployment.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from pipeline.config import RAW_FEATURES, TARGET
from pipeline.evaluation import evaluate_model
from pipeline.training import build_pipeline
from pipeline.validation import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(
    os.environ.get("PIPELINE_RUNTIME_DIR", str(ROOT / "artifacts/orchestration"))
).resolve()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write portable JSON without a UTF-8 BOM."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def run_directory(run_id: str) -> Path:
    """Map an arbitrary Airflow run ID to a filesystem-safe directory."""
    key = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:20]
    folder = RUNTIME / "runs" / key
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def dataset(folder: Path) -> pd.DataFrame:
    """Verify the snapshot hash before reading it."""
    path = folder / "dataset.csv"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = read_json(folder / "manifest.json")["dataset_sha256"]
    if actual != expected:
        raise ValueError("Dataset snapshot changed after ingestion.")
    return pd.read_csv(path)


def ingest(folder: Path, run_id: str) -> None:
    """Snapshot data, configuration and source-code provenance."""
    # Reuse the same snapshot when an Airflow task is retried.
    if (folder / "manifest.json").exists():
        dataset(folder)
        return

    config = yaml.safe_load(
        (ROOT / "config/experiment.yaml").read_text(encoding="utf-8-sig")
    )
    orchestration = yaml.safe_load(
        (ROOT / "config/orchestration.yaml").read_text(encoding="utf-8-sig")
    )
    selected = next(
        item for item in config["experiments"]
        if item["id"] == orchestration["selected_experiment"]
    )
    if not selected["engineered"]:
        raise ValueError("This pipeline expects engineered features.")

    source = ROOT / config["dataset"]
    shutil.copyfile(source, folder / "dataset.csv")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = os.environ.get("GIT_COMMIT", "unknown")

    write_json(folder / "manifest.json", {
        "run_id": run_id,
        "git_commit": commit,
        "dataset_sha256": hashlib.sha256(
            (folder / "dataset.csv").read_bytes()
        ).hexdigest(),
        "seed": int(config["seed"]),
        "threshold": float(config["threshold"]),
        "selected": selected,
        "orchestration": orchestration,
    })


def validate(folder: Path) -> None:
    """Reject schema, statistical and semantic data-quality failures."""
    report = validate_dataset(dataset(folder), raise_on_error=False)
    write_json(folder / "data_validation.json", report)
    if not report["passed"]:
        raise ValueError(f"Data validation failed: {report}")


def prepare_features(folder: Path) -> None:
    """Split 60/20/20 and fit feature processing only on training rows."""
    if not read_json(folder / "data_validation.json")["passed"]:
        raise ValueError("A passed data-validation report is required.")

    frame = dataset(folder)
    manifest = read_json(folder / "manifest.json")
    seed = manifest["seed"]
    indices = np.arange(len(frame))
    development, test = train_test_split(
        indices, test_size=0.2,
        stratify=frame[TARGET], random_state=seed,
    )
    train, validation = train_test_split(
        development, test_size=0.25,
        stratify=frame.iloc[development][TARGET], random_state=seed,
    )
    write_json(folder / "split_indices.json", {
        "train": train.tolist(),
        "validation": validation.tolist(),
        "test": test.tolist(),
    })

    selected = manifest["selected"]
    params = {**selected["params"], "random_state": seed}
    model = build_pipeline(selected["model"], RAW_FEATURES, **params)
    features = model.named_steps["features"].fit_transform(
        frame.iloc[train][RAW_FEATURES],
        frame.iloc[train][TARGET],
    )
    joblib.dump(model, folder / "prepared_pipeline.joblib")
    joblib.dump(features, folder / "training_features.joblib")


def train(folder: Path) -> None:
    """Fit the classifier using the prepared training features."""
    frame = dataset(folder)
    indices = read_json(folder / "split_indices.json")["train"]
    model = joblib.load(folder / "prepared_pipeline.joblib")
    features = joblib.load(folder / "training_features.joblib")
    model.named_steps["classifier"].fit(
        features, frame.iloc[indices][TARGET]
    )
    joblib.dump(model, folder / "candidate.joblib")


def evaluate(folder: Path) -> None:
    """Evaluate the candidate on validation; keep test rows untouched."""
    frame = dataset(folder)
    manifest = read_json(folder / "manifest.json")
    indices = read_json(folder / "split_indices.json")["validation"]
    model = joblib.load(folder / "candidate.joblib")
    report = evaluate_model(
        model,
        frame.iloc[indices][RAW_FEATURES],
        frame.iloc[indices][TARGET],
        threshold=manifest["threshold"],
    )
    write_json(folder / "validation_metrics.json", report)


def quality_gate(folder: Path) -> None:
    """Block registration when performance or group checks fail."""
    manifest = read_json(folder / "manifest.json")
    metrics = read_json(folder / "validation_metrics.json")
    limits = manifest["orchestration"]["quality_gates"]

    checks = {
        "roc_auc": metrics["roc_auc"] >= limits["min_roc_auc"],
        "pr_auc": metrics["pr_auc"] >= limits["min_pr_auc"],
        "fairness_gap": metrics["fairness_gap"] <= limits["max_fairness_gap"],
        "group_coverage": len(metrics["group_metrics"]) >= 2,
    }
    report = {"passed": all(checks.values()), "checks": checks}
    write_json(folder / "quality_gate.json", report)
    if not report["passed"]:
        raise ValueError(f"Quality gate rejected candidate: {report}")


def register(folder: Path) -> None:
    """Log and register a gated candidate in the orchestration store."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    if not read_json(folder / "quality_gate.json")["passed"]:
        raise ValueError("A passed quality gate is required.")
    if (folder / "registered_model.json").exists():
        return

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(RUNTIME / 'mlflow.db').as_posix()}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    manifest = read_json(folder / "manifest.json")
    config = manifest["orchestration"]

    experiment = client.get_experiment_by_name(config["experiment_name"])
    if experiment is None:
        artifact_root = RUNTIME / "mlruns"
        artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_id = client.create_experiment(
            config["experiment_name"],
            artifact_location=artifact_root.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    frame = dataset(folder)
    rows = read_json(folder / "split_indices.json")["train"][:5]
    example = frame.iloc[rows][RAW_FEATURES]
    model = joblib.load(folder / "candidate.joblib")

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=manifest["run_id"],
    ) as run:
        mlflow.log_params({
            **manifest["selected"]["params"],
            "model": manifest["selected"]["model"],
            "seed": manifest["seed"],
            "threshold": manifest["threshold"],
            "selected_experiment": manifest["selected"]["id"],
        })
        mlflow.set_tags({
            "git_commit": manifest["git_commit"],
            "dataset_sha256": manifest["dataset_sha256"],
            "purpose": "orchestrated_candidate",
        })
        metrics = read_json(folder / "validation_metrics.json")
        mlflow.log_metrics({
            f"val_{key}": float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        })
        for name in (
            "manifest.json", "data_validation.json", "split_indices.json",
            "validation_metrics.json", "quality_gate.json",
        ):
            mlflow.log_artifact(str(folder / name))

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=example,
            signature=infer_signature(example, model.predict_proba(example)),
            pyfunc_predict_fn="predict_proba",
            code_paths=[str(ROOT / "pipeline")],
        )
        version = mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            config["registered_model_name"],
        )
        client.set_model_version_tag(
            version.name, version.version, "status", "candidate"
        )
        write_json(folder / "registered_model.json", {
            "tracking_uri": tracking_uri,
            "run_id": run.info.run_id,
            "model_uri": f"models:/{version.name}/{version.version}",
        })


def staging_smoke_test(folder: Path) -> None:
    """Compare MLflow predictions and export a verified staging bundle."""
    import mlflow
    import mlflow.pyfunc

    registered = read_json(folder / "registered_model.json")
    mlflow.set_tracking_uri(registered["tracking_uri"])
    exported = mlflow.pyfunc.load_model(registered["model_uri"])
    local = joblib.load(folder / "candidate.joblib")

    frame = dataset(folder)
    rows = read_json(folder / "split_indices.json")["validation"][:5]
    example = frame.iloc[rows][RAW_FEATURES]
    expected = local.predict_proba(example)
    actual = np.asarray(exported.predict(example))

    if actual.shape != (len(example), 2):
        raise ValueError(f"Unexpected probability shape: {actual.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)

    staging = folder / "staging"
    staging.mkdir(exist_ok=True)
    shutil.copyfile(folder / "candidate.joblib", staging / "model.joblib")
    for name in ("manifest.json", "registered_model.json", "quality_gate.json"):
        shutil.copyfile(folder / name, staging / name)
    write_json(folder / "smoke_test.json", {
        "passed": True,
        "probability_shape": list(actual.shape),
        "predictions_match": True,
        "staging_directory": str(staging),
    })


STAGES = {
    "ingest": ingest,
    "validate": validate,
    "prepare_features": prepare_features,
    "train": train,
    "evaluate": evaluate,
    "quality_gate": quality_gate,
    "register": register,
    "staging_smoke_test": staging_smoke_test,
}


def main() -> None:
    """Execute one stage; a nonzero exit code signals failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    folder = run_directory(args.run_id)

    if args.stage == "ingest":
        ingest(folder, args.run_id)
    else:
        STAGES[args.stage](folder)
    print(f"PASSED: {args.stage}")
    print(f"Run artifacts: {folder}")


if __name__ == "__main__":
    main()
