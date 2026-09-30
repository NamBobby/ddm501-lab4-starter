"""Build the frozen drift reference from the training split."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.monitoring import MONITORED_FEATURES
from pipeline.data_ingestion import load_raw, split_data
from pipeline.preprocessing import add_derived_features

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = BASE_DIR / "models" / "reference.json"


def build_reference(frame: pd.DataFrame, n_bins: int = 10) -> dict:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    bins = {}
    expected = {}

    for feature in MONITORED_FEATURES:
        if feature not in frame:
            continue

        values = (
            pd.to_numeric(frame[feature], errors="coerce")
            .dropna()
            .to_numpy()
        )

        if len(values) == 0:
            continue

        edges = np.unique(
            np.quantile(values, np.linspace(0, 1, n_bins + 1))
        )

        if len(edges) < 3:
            continue

        edges[0] = -np.inf
        edges[-1] = np.inf

        counts, _ = np.histogram(values, bins=edges)

        bins[feature] = edges.tolist()
        expected[feature] = (counts / counts.sum()).tolist()

    return {
        "bins": bins,
        "expected": expected,
        "n_bins": n_bins,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    raw = load_raw()
    X_train, _, _, _ = split_data(raw)
    frame = add_derived_features(X_train)

    reference = build_reference(frame, args.bins)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(reference, indent=2),
        encoding="utf-8",
    )

    print(f"Training rows: {len(frame):,}")
    print(f"Monitored features: {len(reference['bins'])}")
    print(f"Reference saved: {args.out}")


if __name__ == "__main__":
    main()
