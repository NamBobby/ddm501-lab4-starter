"""Monthly credit-risk training with explicit quality gates."""
import logging
import os
import subprocess
from datetime import timedelta

import pendulum
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context


def log_failure(context: dict) -> None:
    """Record failure details; external notifications are not configured."""
    logging.error(
        "Pipeline failed: dag=%s task=%s run=%s log=%s",
        context["dag"].dag_id,
        context["task_instance"].task_id,
        context["run_id"],
        context["task_instance"].log_url,
    )


@dag(
    dag_id="assignment2_credit_risk",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule="@monthly",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=30),
    default_args={
        "owner": "LeThanhPhuongNam",
        "retries": 2,
        "retry_delay": timedelta(minutes=1),
        "execution_timeout": timedelta(minutes=10),
        "on_failure_callback": log_failure,
    },
    tags=["assignment2", "credit-risk", "mlops"],
)
def credit_risk_pipeline():
    """Pass artifact paths through shared storage rather than large XComs."""

    @task
    def execute_stage(stage: str) -> None:
        """Run a stage in the isolated ML Python environment."""
        context = get_current_context()
        project = os.environ.get("PROJECT_ROOT", "/opt/project")
        python = os.environ.get("ML_PYTHON", "/opt/ml-venv/bin/python")
        subprocess.run(
            [
                python, "-m", "scripts.pipeline_stages",
                "--stage", stage,
                "--run-id", context["run_id"],
            ],
            cwd=project,
            check=True,
        )

    previous = None
    for stage in (
        "ingest", "validate", "prepare_features", "train",
        "evaluate", "quality_gate", "register", "staging_smoke_test",
    ):
        # Deterministic failures require a fix, not repeated execution.
        retries = 0 if stage in {
            "validate", "quality_gate", "register", "staging_smoke_test"
        } else 2
        current = execute_stage.override(
            task_id=stage, retries=retries
        )(stage)
        if previous is not None:
            previous >> current
        previous = current


credit_risk_pipeline()
