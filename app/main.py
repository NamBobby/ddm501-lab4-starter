"""
FastAPI application — credit risk scoring, instrumented for production.

Endpoints beyond Lab 1:

    GET  /metrics       Prometheus exposition
    GET  /monitoring    the same numbers as JSON, for humans and for tests
    POST /explain       SHAP contributions for one decision
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    DECLINE_THRESHOLD,
    MODEL_VERSION,
    REVIEW_THRESHOLD,
)
from app.explain import Explainer
from app.metrics import (
    BATCH_SIZE,
    DECISION_COUNT,
    EXPLAIN_COUNT,
    EXPLAIN_LATENCY,
    MODEL_INFO,
    MODEL_LAST_RELOAD,
    MODEL_LOADED,
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTION_SCORE,
)
from app.middleware import MetricsMiddleware
from app.model import CreditRiskModel
from app.monitoring import MonitoringWindow, ReferenceDistribution
from pipeline.preprocessing import add_derived_features
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CreditApplication,
    ExplanationResponse,
    HealthResponse,
    MonitoringResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_PATH = BASE_DIR / "models" / "reference.json"

model: CreditRiskModel | None = None
explainer: Explainer | None = None
window: MonitoringWindow = MonitoringWindow()


# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and the drift reference once, at startup."""
    global model, explainer, window
    try:
        model = CreditRiskModel()
        explainer = Explainer(model.model)
        MODEL_LOADED.set(1)
        MODEL_LAST_RELOAD.set(time.time())
        MODEL_INFO.info({
            "version": MODEL_VERSION,
            "type": str(model.metadata.get("model_type", "unknown")),
            "trained_at": str(model.metadata.get("trained_at", "unknown")),
        })
        logger.info("Model loaded")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model: %s", exc)
        model, explainer = None, None
        MODEL_LOADED.set(0)

    reference = None
    if REFERENCE_PATH.exists():
        try:
            reference = ReferenceDistribution.load(REFERENCE_PATH)
            logger.info("Drift reference loaded (%s features)", len(reference.features))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load drift reference: %s", exc)
    else:
        # Not fatal. The service scores fine without it; only drift is
        # unavailable, and /monitoring says so rather than reporting zero —
        # a drift score of 0.0 because nothing is configured looks exactly
        # like a drift score of 0.0 because nothing has drifted.
        logger.warning(
            "No drift reference at %s. Run scripts/make_reference.py.", REFERENCE_PATH
        )
    window = MonitoringWindow(reference=reference)

    yield
    model, explainer = None, None
    MODEL_LOADED.set(0)


app = FastAPI(
    title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION, lifespan=lifespan
)
app.add_middleware(MetricsMiddleware)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# =============================================================================
def _require_model() -> CreditRiskModel:
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def _observe(frame, scores, results) -> None:
    """Record predictions and derived monitoring features."""
    observed = add_derived_features(frame)

    for (_, row), score, result in zip(
        observed.iterrows(), scores, results
    ):
        score = float(score)

        PREDICTION_SCORE.labels(
            model_version=MODEL_VERSION
        ).observe(score)

        DECISION_COUNT.labels(
            decision=result["decision"],
            model_version=MODEL_VERSION,
        ).inc()

        # SEX is the fairness group in this lab.
        window.record(
            row.to_dict(),
            score,
            int(row["SEX"]),
        )

    PREDICTION_COUNT.labels(
        model_version=MODEL_VERSION
    ).inc(len(results))


# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(response: Response):
    """Readiness endpoint: unhealthy when the model is unavailable."""
    ready = model is not None and model.is_loaded()

    response.status_code = 200 if ready else 503

    return HealthResponse(
        status="healthy" if ready else "unhealthy",
        model_loaded=ready,
        model_version=MODEL_VERSION,
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Return Prometheus text exposition."""
    window.publish()

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/monitoring", response_model=MonitoringResponse, tags=["Monitoring"])
async def monitoring():
    """Return the current monitoring state as JSON."""
    return MonitoringResponse(**window.publish())


# =============================================================================
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(application: CreditApplication):
    """Score one applicant."""
    active = _require_model()
    payload = application.model_dump()
    try:
        start = time.perf_counter()
        frame = active.to_frame([payload])
        scores = active.predict_proba([payload])
        PREDICTION_LATENCY.labels(model_version=MODEL_VERSION).observe(
            time.perf_counter() - start
        )
        result = active.score(payload)
        _observe(frame, scores, [result])
        return PredictionResponse(**result)
    except Exception as exc:  # noqa: BLE001
        PREDICTION_ERRORS.labels(
            error_type=type(exc).__name__, model_version=MODEL_VERSION
        ).inc()
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Scoring failed") from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Score up to 500 applicants."""
    active = _require_model()
    payloads = [a.model_dump() for a in request.applications]
    BATCH_SIZE.observe(len(payloads))
    try:
        start = time.perf_counter()
        frame = active.to_frame(payloads)
        scores = active.predict_proba(payloads)
        PREDICTION_LATENCY.labels(model_version=MODEL_VERSION).observe(
            time.perf_counter() - start
        )
        results = active.score_batch(payloads)
        _observe(frame, scores, results)
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results],
            total_count=len(results),
        )
    except Exception as exc:  # noqa: BLE001
        PREDICTION_ERRORS.labels(
            error_type=type(exc).__name__, model_version=MODEL_VERSION
        ).inc()
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Scoring failed") from exc


@app.post("/explain", response_model=ExplanationResponse, tags=["Prediction"])
async def explain(application: CreditApplication):
    """Return the model explanation for one application."""
    active = _require_model()

    if explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Explainer not loaded",
        )

    try:
        payload = application.model_dump()
        result = active.score(payload)
        frame = active.to_frame([payload])

        started = time.perf_counter()
        explanation = explainer.explain(frame)
        EXPLAIN_LATENCY.observe(
            time.perf_counter() - started
        )
        EXPLAIN_COUNT.inc()

        return ExplanationResponse(
            default_probability=result["default_probability"],
            decision=result["decision"],
            model_version=MODEL_VERSION,
            base_value=explanation["base_value"],
            contributions=explanation["contributions"],
            note=explanation["note"],
        )

    except Exception as exc:
        PREDICTION_ERRORS.labels(
            error_type=type(exc).__name__,
            model_version=MODEL_VERSION,
        ).inc()

        logger.exception("Explanation failed")

        raise HTTPException(
            status_code=500,
            detail="Explanation failed",
        ) from exc


# =============================================================================
@app.get("/", tags=["Info"])
async def root():
    """API metadata."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "monitoring": "/monitoring",
    }


@app.get("/model/info", tags=["Info"])
async def model_info():
    """Model version, training metrics and the thresholds in force."""
    return {
        "model_version": MODEL_VERSION,
        "model_type": (model.metadata.get("model_type") if model else None),
        "trained_at": (model.metadata.get("trained_at") if model else None),
        "metrics": (model.metadata.get("metrics") if model else None),
        "review_threshold": REVIEW_THRESHOLD,
        "decline_threshold": DECLINE_THRESHOLD,
        "is_loaded": model is not None and model.is_loaded(),
        "drift_reference_loaded": window.reference is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
