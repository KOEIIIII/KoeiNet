


"""Target policy registry for Step-7 fusion modeling/evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

import pandas as pd

CONFIRMATORY_REGRESSION_TARGETS: Tuple[str, ...] = (
    "comfort_score",
    "vitality_score",
    "soundscape_eventfulness",
)

EXPLORATORY_REGRESSION_TARGETS: Tuple[str, ...] = (
    "safety_score",
    "overall_problem_severity",
    "soundscape_pleasantness",
)

EXPLORATORY_CLASSIFICATION_TARGETS: Tuple[str, ...] = ("primary_problem_label",)


@dataclass(frozen=True)
class ClassificationDecision:
    """Classification label support decision for one target."""

    enabled: bool
    decision: str
    reason: str
    original_counts: Dict[str, int]
    transformed_counts: Dict[str, int]
    rare_classes: List[str]


def _safe_numeric_count(series: pd.Series) -> int:
    vals = pd.to_numeric(series, errors="coerce")
    return int(vals.notna().sum())


def _build_classification_decision(
    labels: pd.Series,
    min_class_count: int = 5,
    merge_rare_to_mixed: bool = True,
    mixed_label: str = "mixed_or_unclear",
) -> Tuple[ClassificationDecision, pd.Series]:
    """Decide whether and how to run classification under class sparsity constraints."""
    y_raw = labels.astype("string").str.strip()
    y_raw = y_raw.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    y_raw = y_raw.dropna()

    if y_raw.empty:
        decision = ClassificationDecision(
            enabled=False,
            decision="skipped",
            reason="all_labels_missing",
            original_counts={},
            transformed_counts={},
            rare_classes=[],
        )
        return decision, labels

    original_counts = {str(k): int(v) for k, v in y_raw.value_counts(dropna=True).to_dict().items()}
    rare_classes = [k for k, v in original_counts.items() if int(v) < int(min_class_count)]

    y_out = labels.copy()
    y_clean = labels.astype("string").str.strip()
    y_clean = y_clean.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    if rare_classes and merge_rare_to_mixed:
        y_out = y_clean.where(~y_clean.isin(rare_classes), other=mixed_label)
        y_out = y_out.astype("string")
        transformed_counts = {
            str(k): int(v)
            for k, v in y_out.dropna().value_counts(dropna=True).to_dict().items()
        }
        still_rare = [k for k, v in transformed_counts.items() if int(v) < int(min_class_count)]
        if still_rare or len(transformed_counts) < 2:
            decision = ClassificationDecision(
                enabled=False,
                decision="skipped",
                reason="class_support_too_sparse_after_merge",
                original_counts=original_counts,
                transformed_counts=transformed_counts,
                rare_classes=rare_classes,
            )
            return decision, y_out

        decision = ClassificationDecision(
            enabled=True,
            decision="merged_rare_to_mixed_or_unclear",
            reason="",
            original_counts=original_counts,
            transformed_counts=transformed_counts,
            rare_classes=rare_classes,
        )
        return decision, y_out

    transformed_counts = original_counts
    if rare_classes:
        decision = ClassificationDecision(
            enabled=False,
            decision="skipped",
            reason="class_support_too_sparse",
            original_counts=original_counts,
            transformed_counts=transformed_counts,
            rare_classes=rare_classes,
        )
        return decision, y_out

    if len(transformed_counts) < 2:
        decision = ClassificationDecision(
            enabled=False,
            decision="skipped",
            reason="single_class_only",
            original_counts=original_counts,
            transformed_counts=transformed_counts,
            rare_classes=[],
        )
        return decision, y_out

    decision = ClassificationDecision(
        enabled=True,
        decision="as_is",
        reason="",
        original_counts=original_counts,
        transformed_counts=transformed_counts,
        rare_classes=[],
    )
    return decision, y_out


def build_target_registry(
    labels_df: pd.DataFrame,
    min_class_count: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, pd.Series]]:
    """
    Build Step-7 target registry and transformed target series.

    Returns:
    - registry payload (JSON-serializable)
    - target series map (possibly transformed for classification)
    """
    if "segment_id" not in labels_df.columns:
        raise ValueError("labels dataframe missing required column: segment_id")

    target_series: Dict[str, pd.Series] = {}
    target_items: List[Dict[str, Any]] = []

    for target in CONFIRMATORY_REGRESSION_TARGETS:
        if target not in labels_df.columns:
            target_items.append(
                {
                    "target_name": target,
                    "target_type": "regression",
                    "tier": "confirmatory",
                    "enabled": False,
                    "reason": "missing_column",
                    "available_samples": 0,
                }
            )
            continue
        ser = pd.to_numeric(labels_df[target], errors="coerce")
        available = int(ser.notna().sum())
        enabled = available >= max(5, int(min_class_count))
        reason = "" if enabled else "insufficient_non_missing_samples"
        target_items.append(
            {
                "target_name": target,
                "target_type": "regression",
                "tier": "confirmatory",
                "enabled": bool(enabled),
                "reason": reason,
                "available_samples": available,
            }
        )
        target_series[target] = ser

    for target in EXPLORATORY_REGRESSION_TARGETS:
        if target not in labels_df.columns:
            target_items.append(
                {
                    "target_name": target,
                    "target_type": "regression",
                    "tier": "exploratory",
                    "enabled": False,
                    "reason": "missing_column",
                    "available_samples": 0,
                }
            )
            continue
        ser = pd.to_numeric(labels_df[target], errors="coerce")
        available = _safe_numeric_count(ser)
        enabled = available >= max(5, int(min_class_count))
        reason = "" if enabled else "insufficient_non_missing_samples"
        target_items.append(
            {
                "target_name": target,
                "target_type": "regression",
                "tier": "exploratory",
                "enabled": bool(enabled),
                "reason": reason,
                "available_samples": available,
            }
        )
        target_series[target] = ser

    for target in EXPLORATORY_CLASSIFICATION_TARGETS:
        if target not in labels_df.columns:
            target_items.append(
                {
                    "target_name": target,
                    "target_type": "classification",
                    "tier": "exploratory",
                    "enabled": False,
                    "reason": "missing_column",
                    "classification_policy": {},
                }
            )
            continue

        decision, y_out = _build_classification_decision(
            labels=labels_df[target],
            min_class_count=int(min_class_count),
            merge_rare_to_mixed=True,
            mixed_label="mixed_or_unclear",
        )
        target_items.append(
            {
                "target_name": target,
                "target_type": "classification",
                "tier": "exploratory",
                "enabled": bool(decision.enabled),
                "reason": decision.reason,
                "classification_policy": {
                    "min_class_count": int(min_class_count),
                    "decision": decision.decision,
                    "rare_classes": decision.rare_classes,
                    "original_class_counts": decision.original_counts,
                    "transformed_class_counts": decision.transformed_counts,
                },
            }
        )
        target_series[target] = y_out

    registry: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_regression_targets": list(CONFIRMATORY_REGRESSION_TARGETS),
        "exploratory_regression_targets": list(EXPLORATORY_REGRESSION_TARGETS),
        "exploratory_classification_targets": list(EXPLORATORY_CLASSIFICATION_TARGETS),
        "targets": target_items,
        "policy_note": (
            "Confirmatory targets are primary evidence; exploratory targets are secondary. "
            "Classification may be merged/skipped under sparse class support."
        ),
    }
    return registry, target_series


def enabled_targets(registry: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return enabled targets from registry payload."""
    out: List[Dict[str, Any]] = []
    for item in registry.get("targets", []):
        if bool(item.get("enabled", False)):
            out.append(dict(item))
    return out
