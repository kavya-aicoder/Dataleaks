from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dataleaks.schemas.execution import DetectorExecution
from dataleaks.schemas.finding import Finding


@dataclass
class LeakageReport:
    """Structured report containing DataLeaks analysis results."""

    findings: list[Finding] = field(default_factory=list)

    risk_score: float = 0.0
    risk_level: str = "low"

    average_confidence: float = 0.0
    highest_confidence: float = 0.0

    recommendations: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    execution: list[DetectorExecution] = field(default_factory=list)

    schema_validation: dict[str, Any] = field(
        default_factory=lambda: {
            "schema_mismatch": False,
            "missing_from_train": [],
            "missing_from_test": [],
            "dtype_mismatches": [],
            "target": {
                "train_present": True,
                "test_present": True,
            },
        }
    )

    @property
    def has_leakage(self) -> bool:
        """Return whether the report contains any findings."""
        return bool(self.findings)

    @property
    def finding_count(self) -> int:
        """Return the number of findings."""
        return len(self.findings)

    @property
    def failed_detectors(self) -> list[DetectorExecution]:
        """Return detectors that failed during execution."""
        return [
            execution
            for execution in self.execution
            if execution.status == "failed"
        ]

    @property
    def completed_detectors(self) -> list[DetectorExecution]:
        """Return detectors that completed successfully."""
        return [
            execution
            for execution in self.execution
            if execution.status == "completed"
        ]

    @property
    def skipped_detectors(self) -> list[DetectorExecution]:
        """Return detectors intentionally skipped during execution."""
        return [
            execution
            for execution in self.execution
            if execution.status == "skipped"
        ]

    @property
    def has_execution_warnings(self) -> bool:
        """Return whether any detector failed during execution."""
        return bool(self.failed_detectors)

    def findings_by_severity(self, severity: str) -> list[Finding]:
        """Return findings matching a severity level."""
        normalized = severity.strip().lower()

        return [
            finding
            for finding in self.findings
            if finding.severity.strip().lower() == normalized
        ]

    @classmethod
    def from_findings(
        cls,
        findings: list[Finding],
        *,
        metadata: dict[str, Any] | None = None,
        execution: list[DetectorExecution] | None = None,
        schema_validation: dict[str, Any] | None = None,
    ) -> "LeakageReport":
        """Build a complete report from detector findings."""

        if not isinstance(findings, list):
            raise TypeError("findings must be a list")

        if execution is not None and not isinstance(execution, list):
            raise TypeError("execution must be a list")

        if (
            schema_validation is not None
            and not isinstance(schema_validation, dict)
        ):
            raise TypeError("schema_validation must be a dictionary")

        from dataleaks.recommendations.recommendations import (
            RecommendationEngine,
        )
        from dataleaks.scoring.confidence import ConfidenceScorer
        from dataleaks.scoring.risk import RiskScorer

        risk_score = RiskScorer.score(findings)
        risk_level = RiskScorer.level(findings)

        average_confidence = ConfidenceScorer.average(findings)
        highest_confidence = ConfidenceScorer.highest(findings)

        recommendations = RecommendationEngine().generate(findings)

        return cls(
            findings=list(findings),
            risk_score=risk_score,
            risk_level=risk_level,
            average_confidence=average_confidence,
            highest_confidence=highest_confidence,
            recommendations=recommendations,
            metadata=dict(metadata or {}),
            execution=list(execution or []),
            schema_validation=dict(schema_validation or {
                "schema_mismatch": False,
                "missing_from_train": [],
                "missing_from_test": [],
                "dtype_mismatches": [],
                "target": {
                    "train_present": True,
                    "test_present": True,
                },
            }),
        )