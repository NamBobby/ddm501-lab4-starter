"""Validate data and publish a corrected model signature without retraining."""

import hashlib
import json
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from pipeline.config import RAW_FEATURES
from pipeline.data_ingestion import load_raw
from pipeline.validation import validate_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Preserve the original run and register a corrected candidate."""
    output = ROOT / "artifacts" / "assignment2"
    summary_path = output / "selected_model.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset_path = ROOT / "data" / "credit_default.csv"

    current_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if current_hash != summary["dataset_sha256"]:
        raise ValueError("Dataset changed since the experiment run.")

    frame = load_raw(dataset_path)
    report = validate_dataset(frame, raise_on_error=False)
    report_path = output / "data_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise ValueError(
            "Data validation failed. Inspect artifacts/assignment2/data_validation.json"
        )

    split = json.loads(
        (output / "split_indices.json").read_text(encoding="utf-8")
    )
    sample = frame.loc[split["train"][:5], RAW_FEATURES]

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    client = mlflow.MlflowClient()

    # Reuse the correction if this script has already completed.
    if "corrected_model_uri" in summary:
        loaded = mlflow.pyfunc.load_model(summary["corrected_model_uri"])
        prediction = np.asarray(loaded.predict(sample))
        assert prediction.shape == (len(sample), 2)
        np.testing.assert_allclose(prediction.sum(axis=1), 1.0)
        print("Existing corrected model verified:", summary["corrected_model_uri"])
        return

    source_uri = f"runs:/{summary['selected_run_id']}/model"
    model = mlflow.sklearn.load_model(source_uri)
    expected = model.predict_proba(sample)

    mlflow.set_experiment("assignment2-credit-risk")
    with mlflow.start_run(run_name="E01-signature-correction") as run:
        mlflow.set_tags({
            "purpose": "artifact_correction_no_retraining",
            "source_run_id": summary["selected_run_id"],
            "dataset_sha256": current_hash,
            "original_git_commit": summary["git_commit"],
        })
        mlflow.log_artifact(str(report_path))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(sample, expected),
            input_example=sample,
            pyfunc_predict_fn="predict_proba",
            code_paths=[str(ROOT / "pipeline")],
        )

        corrected_uri = f"runs:/{run.info.run_id}/model"
        loaded = mlflow.pyfunc.load_model(corrected_uri)
        actual = np.asarray(loaded.predict(sample))
        assert actual.shape == (len(sample), 2)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actual.sum(axis=1), 1.0)

        version = mlflow.register_model(
            corrected_uri, "assignment2-credit-risk"
        )
        client.set_model_version_tag(
            version.name, version.version, "status", "candidate"
        )
        client.set_model_version_tag(
            version.name, version.version,
            "source_experiment", summary["selected_experiment"]
        )
        summary.update({
            "corrected_run_id": run.info.run_id,
            "corrected_model_uri": (
                f"models:/{version.name}/{version.version}"
            ),
            "registered_model_version": version.version,
            "probability_output": ["no_default", "default"],
        })
        summary_path.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(summary_path))

    print("Data validation: PASSED")
    print("Probability output shape:", actual.shape)
    print("Sklearn and MLflow predictions: MATCH")
    print("Registered candidate:", summary["corrected_model_uri"])


if __name__ == "__main__":
    main()
