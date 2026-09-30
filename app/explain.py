"""
SHAP explanations for individual scoring decisions.

"""

import logging
from typing import Any, Dict, List

import numpy as np

from pipeline.preprocessing import add_derived_features

logger = logging.getLogger(__name__)


class Explainer:
    """Wraps a SHAP TreeExplainer around the fitted pipeline.

    The pipeline is (features -> classifier). SHAP needs the raw estimator and
    the TRANSFORMED matrix, so this class holds both halves and does the
    transformation itself. Handing the whole pipeline to shap and hoping is
    the usual mistake.
    """

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.feature_step = pipeline.named_steps["features"]
        self.classifier = pipeline.named_steps["classifier"]
        self.feature_names = list(
            self.feature_step.named_steps["preprocess"].get_feature_names_out()
        )
        self._explainer = None

    def _ensure_explainer(self):
        """Build the explainer on first use.

        Lazily, because importing shap costs about a second and a container
        that is not asked for explanations should not pay it at startup.
        """
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.classifier)
        return self._explainer

    def explain(self, frame, top_n: int = 8) -> Dict[str, Any]:
        """Return the largest positive-class SHAP contributions."""
        transformed = self.feature_step.transform(frame)

        explainer = self._ensure_explainer()
        shap_values = explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]

        contributions = shap_values[0]

        expected_value = np.asarray(
            explainer.expected_value
        ).reshape(-1)

        base_value = float(expected_value[-1])

        applicant = add_derived_features(frame).iloc[0]

        indexes = np.argsort(
            -np.abs(contributions),
            kind="stable",
        )[:top_n]

        items = []

        for index in indexes:
            feature_name = self.feature_names[index]
            contribution = float(contributions[index])

            if feature_name in applicant.index:
                raw_value = applicant[feature_name]
            else:
                raw_value = transformed[0, index]

            try:
                value = float(raw_value)
                if not np.isfinite(value):
                    value = 0.0
            except (TypeError, ValueError):
                value = 0.0

            items.append(
                {
                    "feature": feature_name,
                    "value": value,
                    "contribution": contribution,
                    "direction": (
                        "increases risk"
                        if contribution > 0
                        else "reduces risk"
                    ),
                }
            )

        return {
            "base_value": base_value,
            "contributions": items,
            "note": (
                "SHAP values describe the fitted model, not causal effects. "
                "Only the largest contributions are shown."
            ),
        }
