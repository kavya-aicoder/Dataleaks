from __future__ import annotations

from dataleaks.schemas.finding import Finding
from dataleaks.scoring.severity import Severity


class RiskScorer:
    """Calculate an overall leakage risk score from findings."""

    SEVERITY_WEIGHTS: dict[str, float] = {
        "low": 0.25,
        "medium": 0.50,
        "high": 0.75,
        "critical": 1.00,
    }

    @classmethod
    def score(cls, findings: list[Finding]) -> float:
        """Return an overall risk score between 0.0 and 1.0."""

        if not findings:
            return 0.0

        weighted_scores = []

        for finding in findings:
            severity = Severity.normalize(finding.severity)
            weight = cls.SEVERITY_WEIGHTS[severity]

            weighted_scores.append(
                weight * finding.confidence
            )

        return min(1.0, max(weighted_scores))

    @classmethod
    def level(cls, findings: list[Finding]) -> str:
        """Return a human-readable overall risk level."""

        score = cls.score(findings)

        if score >= 1.0:
            return "critical"
        if score >= 0.75:
            return "high"
        if score >= 0.50:
            return "medium"

        return "low"