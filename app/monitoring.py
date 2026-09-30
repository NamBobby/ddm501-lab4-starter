"""Input drift and group selection rates over a bounded monitoring window."""

import json
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import REVIEW_THRESHOLD
from app.metrics import (
    DRIFT_SCORE,
    DRIFT_WINDOW_SIZE,
    FAIRNESS_GAP,
    FEATURE_DRIFT_PSI,
    SELECTION_RATE,
)

MONITORED_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "utilisation_ratio",
    "payment_ratio",
    "max_delay",
]

EPSILON = 1e-4


class ReferenceDistribution:
    """Frozen training distribution shipped with the model."""

    def __init__(self, bins, expected):
        self.bins = bins
        self.expected = expected

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(payload["bins"], payload["expected"])

    @property
    def features(self):
        return list(self.bins)


def population_stability_index(actual, bin_edges, expected) -> float:
    """PSI = sum((actual - expected) * log(actual / expected))."""
    if len(actual) == 0:
        return 0.0

    counts, _ = np.histogram(actual, bins=bin_edges)
    if counts.sum() == 0:
        return 0.0

    observed = np.maximum(counts / counts.sum(), EPSILON)
    baseline = np.maximum(np.asarray(expected, dtype=float), EPSILON)

    return float(
        np.sum((observed - baseline) * np.log(observed / baseline))
    )


class MonitoringWindow:
    def __init__(
        self,
        reference=None,
        max_size=2000,
        min_size=200,
        threshold=REVIEW_THRESHOLD,
    ):
        self.reference = reference
        self.max_size = max_size
        self.min_size = min_size
        self.threshold = threshold
        self._rows = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def record(self, features, score, group):
        row = {name: features.get(name) for name in MONITORED_FEATURES}
        row["_score"] = float(score)
        row["_group"] = str(group)

        with self._lock:
            self._rows.append(row)
            DRIFT_WINDOW_SIZE.set(len(self._rows))

    def record_frame(self, frame, scores, group_column):
        for (_, row), score in zip(frame.iterrows(), scores):
            self.record(
                row.to_dict(),
                float(score),
                row.get(group_column),
            )

    def snapshot(self):
        with self._lock:
            return pd.DataFrame(list(self._rows))

    def __len__(self):
        with self._lock:
            return len(self._rows)

    def compute_drift(self):
        frame = self.snapshot()

        if self.reference is None or len(frame) < self.min_size:
            return {}

        result = {}

        for feature in self.reference.features:
            if feature not in frame:
                continue

            values = (
                pd.to_numeric(frame[feature], errors="coerce")
                .dropna()
                .to_numpy()
            )

            # Missing values must not look like a stable distribution.
            if len(values) < self.min_size:
                continue

            result[feature] = population_stability_index(
                values,
                self.reference.bins[feature],
                self.reference.expected[feature],
            )

        return result

    def compute_fairness(self):
        frame = self.snapshot()

        if len(frame) < self.min_size:
            return {}

        return {
            str(group): float(
                (rows["_score"] >= self.threshold).mean()
            )
            for group, rows in frame.groupby("_group")
            if len(rows) >= 30
        }

    def publish(self):
        drift = self.compute_drift()
        rates = self.compute_fairness()

        score = max(drift.values(), default=0.0)
        gap = (
            max(rates.values()) - min(rates.values())
            if len(rates) >= 2
            else 0.0
        )

        # Remove stale feature/group labels from earlier windows.
        FEATURE_DRIFT_PSI.clear()
        SELECTION_RATE.clear()

        for feature, value in drift.items():
            FEATURE_DRIFT_PSI.labels(feature=feature).set(value)

        for group, value in rates.items():
            SELECTION_RATE.labels(group=group).set(value)

        size = len(self)
        DRIFT_SCORE.set(score)
        FAIRNESS_GAP.set(gap)
        DRIFT_WINDOW_SIZE.set(size)

        sufficient = (
            self.reference is not None
            and bool(self.reference.features)
            and len(drift) == len(self.reference.features)
            and size >= self.min_size
        )

        return {
            "window_size": size,
            "min_window_size": self.min_size,
            "sufficient_data": sufficient,
            "feature_psi": drift,
            "drift_score": score,
            "drift_status": drift_status(score),
            "selection_rate": rates,
            "fairness_gap": gap,
        }


def drift_status(psi):
    if psi >= 0.25:
        return "significant"
    if psi >= 0.10:
        return "moderate"
    return "stable"
