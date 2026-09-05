from __future__ import annotations

import pandas as pd

from dataleaks.detectors.temporal.parsing import (
    parse_timestamps,
)
from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class FutureFeatureDetector(BaseDetector):
    """Detect feature timestamps that occur after prediction time."""

    name = "temporal_future_features"
    category = "temporal_leakage"

    MAX_EVIDENCE_ROWS = 20

    def __init__(
        self,
        prediction_time_column: str,
        feature_time_columns: list[str],
    ) -> None:
        if not prediction_time_column.strip():
            raise ValueError(
                "prediction_time_column must not be empty"
            )

        if not feature_time_columns:
            raise ValueError(
                "feature_time_columns must not be empty"
            )

        if prediction_time_column in feature_time_columns:
            raise ValueError(
                "prediction_time_column must not also appear in "
                "feature_time_columns"
            )

        if len(feature_time_columns) != len(set(feature_time_columns)):
            raise ValueError(
                "feature_time_columns must not contain duplicates"
            )

        self.prediction_time_column = prediction_time_column
        self.feature_time_columns = feature_time_columns

    def _get_conditional_config(
        self,
        context: DatasetContext,
        column: str,
    ) -> dict | None:
        temporal = context.metadata.get(
            "temporal",
            {},
        )

        if not isinstance(temporal, dict):
            return None

        conditional_columns = temporal.get(
            "conditional_columns",
            {},
        )

        if not isinstance(conditional_columns, dict):
            return None

        config = conditional_columns.get(column)

        if config is None:
            return None

        if not isinstance(config, dict):
            raise TypeError(
                f"Conditional temporal configuration for '{column}' "
                "must be a dictionary"
            )

        return config

    def _get_conditional_target(
        self,
        context: DatasetContext,
        column: str,
        conditional_config: dict,
    ) -> tuple[pd.Series, object]:
        target_column = conditional_config.get(
            "target_column",
            conditional_config.get("target"),
        )

        if (
            not isinstance(target_column, str)
            or not target_column.strip()
        ):
            raise ValueError(
                f"Conditional temporal configuration for '{column}' "
                "must define a non-empty 'target_column'"
            )

        if target_column not in context.data.columns:
            raise ValueError(
                f"Conditional temporal target '{target_column}' "
                "does not exist in the dataset"
            )

        if "present_when" not in conditional_config:
            raise ValueError(
                f"Conditional temporal configuration for '{column}' "
                "must define 'present_when'"
            )

        present_when = conditional_config["present_when"]

        expected_present = (
            context.data[target_column] == present_when
        )

        return expected_present, present_when

    def _build_invalid_mask(
        self,
        context: DatasetContext,
        column: str,
        raw_series: pd.Series,
        parse_result,
    ) -> pd.Series:
        invalid_mask = (
            raw_series.notna()
            & parse_result.values.isna()
        )

        conditional_config = self._get_conditional_config(
            context,
            column,
        )

        if conditional_config is None:
            return invalid_mask | raw_series.isna()

        expected_present, _ = self._get_conditional_target(
            context,
            column,
            conditional_config,
        )

        missing_mask = raw_series.isna()

        invalid_missing_mask = (
            missing_mask
            & expected_present
        )

        return (
            invalid_mask
            | invalid_missing_mask
        )

    def _build_prediction_invalid_mask(
        self,
        context: DatasetContext,
        raw_series: pd.Series,
        parse_result,
    ) -> tuple[pd.Series, dict | None]:
        """
        Build the invalid prediction-time mask.

        Prediction timestamps may also have conditional presence.
        For example, an outcome timestamp may only be expected when
        the target indicates that the outcome occurred.
        """

        invalid_mask = (
            raw_series.notna()
            & parse_result.values.isna()
            & ~parse_result.ambiguous_mask
        )

        conditional_config = self._get_conditional_config(
            context,
            self.prediction_time_column,
        )

        if conditional_config is None:
            return (
                raw_series.isna() | invalid_mask,
                None,
            )

        expected_present, _ = self._get_conditional_target(
            context,
            self.prediction_time_column,
            conditional_config,
        )

        missing_mask = raw_series.isna()

        invalid_missing_mask = (
            missing_mask
            & expected_present
        )

        return (
            invalid_mask | invalid_missing_mask,
            conditional_config,
        )

    def detect(
        self,
        context: DatasetContext,
    ) -> list[Finding]:
        data = context.data

        required_columns = [
            self.prediction_time_column,
            *self.feature_time_columns,
        ]

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Temporal columns are missing: {missing}"
            )

        prediction_result = parse_timestamps(
            data[self.prediction_time_column]
        )

        prediction_time = prediction_result.values

        (
            prediction_invalid_mask,
            prediction_conditional_config,
        ) = self._build_prediction_invalid_mask(
            context,
            data[self.prediction_time_column],
            prediction_result,
        )

        findings: list[Finding] = []

        for column in self.feature_time_columns:
            raw_feature_time = data[column]

            feature_result = parse_timestamps(
                raw_feature_time
            )

            feature_time = feature_result.values

            invalid_prediction_count = int(
                prediction_invalid_mask.sum()
            )

            conditional_config = self._get_conditional_config(
                context,
                column,
            )

            invalid_feature_mask = self._build_invalid_mask(
                context,
                column,
                raw_feature_time,
                feature_result,
            )

            invalid_feature_count = int(
                invalid_feature_mask.sum()
            )

            conditional_configured = (
                prediction_conditional_config is not None
                or conditional_config is not None
            )

            # --------------------------------------------------
            # Ambiguous timestamp evidence
            # --------------------------------------------------
            ambiguous_mask = (
                prediction_result.ambiguous_mask
                | feature_result.ambiguous_mask
            )

            ambiguous_rows: list[dict[str, object]] = []

            ambiguous_indices = data.index[
                ambiguous_mask
            ]

            for row_index in ambiguous_indices[
                : self.MAX_EVIDENCE_ROWS
            ]:
                ambiguous_rows.append(
                    {
                        "row_index": row_index,
                        "prediction_time": str(
                            data.loc[
                                row_index,
                                self.prediction_time_column,
                            ]
                        ),
                        "feature_time": str(
                            data.loc[
                                row_index,
                                column,
                            ]
                        ),
                        "prediction_time_ambiguous": bool(
                            prediction_result.ambiguous_mask.loc[
                                row_index
                            ]
                        ),
                        "feature_time_ambiguous": bool(
                            feature_result.ambiguous_mask.loc[
                                row_index
                            ]
                        ),
                    }
                )

            if ambiguous_rows:
                total_rows = len(data)

                ambiguous_count = int(
                    ambiguous_mask.sum()
                )

                ambiguous_ratio = (
                    ambiguous_count / total_rows
                    if total_rows
                    else 0.0
                )

                if ambiguous_ratio >= 0.5:
                    severity = "high"
                elif ambiguous_ratio >= 0.1:
                    severity = "medium"
                else:
                    severity = "low"

                findings.append(
                    Finding(
                        detector=self.name,
                        category=self.category,
                        severity=severity,
                        confidence=ambiguous_ratio,
                        explanation=(
                            f"Temporal column '{column}' contains "
                            f"{ambiguous_count} row(s) with ambiguous "
                            "date formats that cannot be safely "
                            "interpreted."
                        ),
                        recommendation=(
                            f"Use an explicit date format for "
                            f"'{column}' so temporal ordering can "
                            "be evaluated without guessing."
                        ),
                        affected_columns=[
                            self.prediction_time_column,
                            column,
                        ],
                        evidence={
                            "type": "ambiguous_timestamp",
                            "prediction_time_column": (
                                self.prediction_time_column
                            ),
                            "feature_time_column": column,
                            "ambiguous_count": ambiguous_count,
                            "ambiguous_ratio": ambiguous_ratio,
                            "ambiguous_rows": ambiguous_rows,
                            "evidence_limit": (
                                self.MAX_EVIDENCE_ROWS
                            ),
                            "prediction_inferred_format": (
                                prediction_result.inferred_format
                            ),
                            "feature_inferred_format": (
                                feature_result.inferred_format
                            ),
                        },
                    )
                )

            # --------------------------------------------------
            # Invalid timestamp evidence
            # --------------------------------------------------
            invalid_mask = (
                prediction_invalid_mask
                | invalid_feature_mask
            )

            invalid_rows: list[dict[str, object]] = []

            invalid_indices = data.index[
                invalid_mask
            ]

            for row_index in invalid_indices[
                : self.MAX_EVIDENCE_ROWS
            ]:
                prediction_value = data.loc[
                    row_index,
                    self.prediction_time_column,
                ]

                feature_value = data.loc[
                    row_index,
                    column,
                ]

                invalid_rows.append(
                    {
                        "row_index": row_index,
                        "prediction_time": (
                            None
                            if pd.isna(prediction_value)
                            else str(prediction_value)
                        ),
                        "feature_time": (
                            None
                            if pd.isna(feature_value)
                            else str(feature_value)
                        ),
                        "prediction_time_invalid": bool(
                            prediction_invalid_mask.loc[
                                row_index
                            ]
                        ),
                        "feature_time_invalid": bool(
                            invalid_feature_mask.loc[
                                row_index
                            ]
                        ),
                    }
                )

            if invalid_mask.any():
                total_rows = len(data)

                invalid_count = int(
                    invalid_mask.sum()
                )

                invalid_ratio = (
                    invalid_count / total_rows
                    if total_rows
                    else 0.0
                )

                if invalid_ratio >= 0.5:
                    severity = "high"
                elif invalid_ratio >= 0.1:
                    severity = "medium"
                else:
                    severity = "low"

                findings.append(
                    Finding(
                        detector=self.name,
                        category=self.category,
                        severity=severity,
                        confidence=invalid_ratio,
                        explanation=(
                            f"Temporal column '{column}' contains "
                            f"{invalid_count} row(s) with invalid or "
                            "unparseable timestamps."
                        ),
                        recommendation=(
                            f"Validate and normalize timestamp values "
                            f"in '{self.prediction_time_column}' and "
                            f"'{column}' before temporal leakage analysis."
                        ),
                        affected_columns=[
                            self.prediction_time_column,
                            column,
                        ],
                        evidence={
                            "type": "invalid_timestamp",
                            "prediction_time_column": (
                                self.prediction_time_column
                            ),
                            "feature_time_column": column,
                            "invalid_count": invalid_count,
                            "invalid_prediction_count": (
                                invalid_prediction_count
                            ),
                            "invalid_feature_count": (
                                invalid_feature_count
                            ),
                            "total_rows": total_rows,
                            "invalid_ratio": invalid_ratio,
                            "invalid_rows": invalid_rows,
                            "evidence_limit": (
                                self.MAX_EVIDENCE_ROWS
                            ),
                            "conditional_configured": (
                                conditional_configured
                            ),
                        },
                    )
                )

            # --------------------------------------------------
            # Future timestamp comparison
            # --------------------------------------------------
            comparable = (
                prediction_time.notna()
                & feature_time.notna()
                & ~prediction_result.ambiguous_mask
                & ~feature_result.ambiguous_mask
            )

            comparable_count = int(
                comparable.sum()
            )

            if comparable_count == 0:
                continue

            future_mask = (
                feature_time > prediction_time
            ) & comparable

            future_count = int(
                future_mask.sum()
            )

            if future_count == 0:
                continue

            future_ratio = (
                future_count / comparable_count
            )

            if future_ratio >= 0.5:
                severity = "critical"
            elif future_ratio >= 0.1:
                severity = "high"
            else:
                severity = "medium"

            violating_rows: list[dict[str, object]] = []

            violating_indices = data.index[
                future_mask
            ]

            for row_index in violating_indices[
                : self.MAX_EVIDENCE_ROWS
            ]:
                violating_rows.append(
                    {
                        "row_index": row_index,
                        "prediction_time": str(
                            prediction_time.loc[row_index]
                        ),
                        "feature_time": str(
                            feature_time.loc[row_index]
                        ),
                    }
                )

            findings.append(
                Finding(
                    detector=self.name,
                    category=self.category,
                    severity=severity,
                    confidence=future_ratio,
                    explanation=(
                        f"Feature timestamp '{column}' occurs after "
                        f"the prediction timestamp in {future_count} "
                        f"of {comparable_count} comparable rows."
                    ),
                    recommendation=(
                        f"Ensure '{column}' is available at or before "
                        f"'{self.prediction_time_column}' when generating "
                        "the prediction."
                    ),
                    affected_columns=[
                        self.prediction_time_column,
                        column,
                    ],
                    evidence={
                        "type": "future_feature_timestamp",
                        "prediction_time_column": (
                            self.prediction_time_column
                        ),
                        "feature_time_column": column,
                        "future_count": future_count,
                        "comparable_count": comparable_count,
                        "future_ratio": future_ratio,
                        "violating_rows": violating_rows,
                        "evidence_limit": (
                            self.MAX_EVIDENCE_ROWS
                        ),
                    },
                )
            )

        return findings