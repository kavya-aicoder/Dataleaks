import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class DerivedTargetLeakageDetector(BaseDetector):
    """Detect numeric features that are deterministic affine transforms
    of the target.
    """

    name = "target_derived"
    category = "target_leakage"

    def __init__(self, tolerance: float = 1e-9) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        self.tolerance = tolerance

    def detect(self, context: DatasetContext) -> list[Finding]:
        if context.target is None:
            return []

        target_name = context.target
        data = context.data
        target = data[target_name]

        if not pd.api.types.is_numeric_dtype(target):
            return []

        findings: list[Finding] = []

        for column in data.columns:
            if column == target_name:
                continue

            feature = data[column]

            if not pd.api.types.is_numeric_dtype(feature):
                continue

            pair = pd.concat(
                [feature, target],
                axis=1,
            ).dropna()

            if len(pair) < 2:
                continue

            feature_values = pair.iloc[:, 0]
            target_values = pair.iloc[:, 1]

            if self._is_affine_transform(feature_values, target_values):
                scale, offset = self._fit_affine_transform(
                    feature_values,
                    target_values,
                )

                findings.append(
                    Finding(
                        detector=self.name,
                        category=self.category,
                        severity="critical",
                        confidence=1.0,
                        explanation=(
                            f"Feature '{column}' can be deterministically "
                            f"expressed as approximately "
                            f"{scale:.6g} * '{target_name}' + {offset:.6g}."
                        ),
                        recommendation=(
                            f"Remove '{column}' or verify that it is "
                            f"available before the prediction target is "
                            f"generated."
                        ),
                        affected_columns=[column],
                        evidence={
                            "type": "affine_target_transform",
                            "target": target_name,
                            "scale": scale,
                            "offset": offset,
                            "samples": len(pair),
                            "tolerance": self.tolerance,
                        },
                    )
                )

        return findings

    def _is_affine_transform(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> bool:
        scale, offset = self._fit_affine_transform(feature, target)

        predicted = target * scale + offset

        return bool(
            (feature - predicted).abs().le(self.tolerance).all()
        )

    @staticmethod
    def _fit_affine_transform(
        feature: pd.Series,
        target: pd.Series,
    ) -> tuple[float, float]:
        target_mean = float(target.mean())
        feature_mean = float(feature.mean())

        centered_target = target - target_mean
        centered_feature = feature - feature_mean

        denominator = float((centered_target**2).sum())

        if denominator == 0.0:
            return 0.0, feature_mean

        scale = float(
            (centered_target * centered_feature).sum()
            / denominator
        )

        offset = feature_mean - scale * target_mean

        return scale, float(offset)