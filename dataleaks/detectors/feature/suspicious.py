from __future__ import annotations

import re

import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class SuspiciousFeatureDetector(BaseDetector):
    """Detect features with unusually strong target relationships."""

    name = "feature_suspicious"
    category = "feature_leakage"

    _POST_OUTCOME_PATTERN = re.compile(
        r"("
        r"post[_-]?outcome|"
        r"after[_-]?outcome|"
        r"post[_-]?churn|"
        r"after[_-]?churn|"
        r"post[_-]?purchase|"
        r"after[_-]?purchase|"
        r"outcome[_-]?refund|"
        r"refund[_-]?after|"
        r"post[_-]?refund|"
        r"after[_-]?refund|"
        r"post[_-]?result|"
        r"after[_-]?result"
        r")",
        re.IGNORECASE,
    )

    def __init__(
        self,
        correlation_threshold: float = 0.999,
        min_samples: int = 10,
        post_outcome_correlation_threshold: float = 0.90,
    ) -> None:
        if not 0.0 <= correlation_threshold <= 1.0:
            raise ValueError(
                "correlation_threshold must be between 0.0 and 1.0"
            )

        if min_samples < 2:
            raise ValueError(
                "min_samples must be at least 2"
            )

        if not 0.0 <= post_outcome_correlation_threshold <= 1.0:
            raise ValueError(
                "post_outcome_correlation_threshold must be "
                "between 0.0 and 1.0"
            )

        self.correlation_threshold = correlation_threshold
        self.min_samples = min_samples
        self.post_outcome_correlation_threshold = (
            post_outcome_correlation_threshold
        )

    def detect(
        self,
        context: DatasetContext,
    ) -> list[Finding]:
        if context.target is None:
            return []

        target = context.target
        data = context.data

        if target not in data.columns:
            return []

        if not pd.api.types.is_numeric_dtype(
            data[target]
        ):
            return []

        findings: list[Finding] = []

        for column in data.columns:
            if column == target:
                continue

            feature = data[column]

            if not pd.api.types.is_numeric_dtype(feature):
                continue

            pair = pd.concat(
                [feature, data[target]],
                axis=1,
            ).dropna()

            if len(pair) < self.min_samples:
                continue

            feature_values = pair.iloc[:, 0]
            target_values = pair.iloc[:, 1]

            if feature_values.nunique(
                dropna=True
            ) <= 1:
                continue

            if target_values.nunique(
                dropna=True
            ) <= 1:
                continue

            correlation = feature_values.corr(
                target_values
            )

            if pd.isna(correlation):
                continue

            correlation = float(correlation)

            if abs(abs(correlation) - 1.0) <= 1e-12:
                correlation = (
                    1.0
                    if correlation > 0
                    else -1.0
                )
            else:
                correlation = max(
                    -1.0,
                    min(1.0, correlation),
                )

            absolute_correlation = abs(
                correlation
            )

            is_post_outcome = bool(
                self._POST_OUTCOME_PATTERN.search(
                    column
                )
            )

            standard_threshold_met = (
                absolute_correlation
                >= self.correlation_threshold
            )

            post_outcome_threshold_met = (
                is_post_outcome
                and absolute_correlation
                >= self.post_outcome_correlation_threshold
            )

            if not (
                standard_threshold_met
                or post_outcome_threshold_met
            ):
                continue

            if post_outcome_threshold_met:
                severity = "high"
                detection_reason = (
                    "post-outcome feature name combined "
                    "with strong target correlation"
                )
            else:
                severity = "high"
                detection_reason = (
                    "extremely strong target correlation"
                )

            findings.append(
                Finding(
                    detector=self.name,
                    category=self.category,
                    severity=severity,
                    confidence=absolute_correlation,
                    explanation=(
                        f"Feature '{column}' has a "
                        "suspiciously strong relationship "
                        f"with target '{target}' "
                        f"(|correlation|="
                        f"{absolute_correlation:.4f}). "
                        + (
                            "Its name also suggests that "
                            "the feature may contain "
                            "post-outcome information. "
                            if is_post_outcome
                            else ""
                        )
                        + "This may indicate target-derived "
                        "or post-outcome information."
                    ),
                    recommendation=(
                        f"Review how '{column}' is generated "
                        "and whether it is available at "
                        "prediction time. Verify that the "
                        "feature does not contain "
                        "target-derived or post-outcome "
                        "information."
                    ),
                    affected_columns=[column],
                    evidence={
                        "type": (
                            "post_outcome_feature"
                            if post_outcome_threshold_met
                            else "suspicious_target_relationship"
                        ),
                        "feature": column,
                        "target": target,
                        "absolute_correlation": (
                            absolute_correlation
                        ),
                        "correlation": correlation,
                        "threshold": (
                            self.correlation_threshold
                        ),
                        "post_outcome_threshold": (
                            self.post_outcome_correlation_threshold
                        ),
                        "post_outcome_name_signal": (
                            is_post_outcome
                        ),
                        "detection_reason": detection_reason,
                        "samples": len(pair),
                    },
                )
            )

        return findings