


"""Deterministic segment priority ranking for Step-8 design mapping."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import pandas as pd

from .design_rules import (
    adjudication_confidence_flag,
    clamp01,
    combine_problem_labels,
    compute_comfort_risk,
    compute_diagnosis_severity,
    compute_eventfulness_gap,
    compute_multimodal_consistency,
    compute_vitality_deficit,
    consistency_flag,
    estimate_observed_score,
    infer_street_type,
)

logger = logging.getLogger("design.priority_ranker")


def _priority_level(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _evidence_strength(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _confirmatory_focus(
    comfort_risk_score: float,
    vitality_deficit_score: float,
    eventfulness_risk_or_gap: float,
) -> List[str]:
    ordered = [
        ("comfort_score", comfort_risk_score),
        ("vitality_score", vitality_deficit_score),
        ("soundscape_eventfulness", eventfulness_risk_or_gap),
    ]
    ordered.sort(key=lambda item: item[1], reverse=True)
    selected = [name for name, value in ordered if value >= 0.33]
    if not selected:
        selected = [ordered[0][0]]
    return selected[:2]


def _theme(
    comfort_risk_score: float,
    vitality_deficit_score: float,
    eventfulness_risk_or_gap: float,
) -> str:
    ordered = sorted(
        [
            ("comfort_buffering", comfort_risk_score),
            ("vitality_activation", vitality_deficit_score),
            ("eventfulness_calibration", eventfulness_risk_or_gap),
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    if len(ordered) >= 2 and abs(ordered[0][1] - ordered[1][1]) <= 0.08:
        return "mixed_rebalancing"
    return ordered[0][0]


def _actionable_evidence_score(
    record: Mapping[str, Any],
    consistency_score: float,
    label_available: bool,
    diagnosis_available: bool,
    profile_available: bool,
) -> float:
    availability = 0.0
    availability += 0.30 if diagnosis_available else 0.0
    availability += 0.20 if profile_available else 0.0
    availability += 0.15 if label_available else 0.0
    availability += 0.20 if bool(record.get("model_feature_row_available", True)) else 0.0
    availability += 0.15 if bool(record.get("step75_evidence_available", True)) else 0.0
    return clamp01(0.55 * availability + 0.45 * consistency_score)


def build_segment_priority_ranking(
    segment_records: Sequence[Mapping[str, Any]],
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    Build a transparent segment-level priority ranking.

    The returned context dictionary contains intermediate scores used later by
    the design plan and prompt generation steps.
    """
    rows: List[Dict[str, Any]] = []
    contexts: Dict[int, Dict[str, Any]] = {}

    for raw_record in segment_records:
        record: MutableMapping[str, Any] = dict(raw_record)
        segment_id = int(record["segment_id"])
        street_type = infer_street_type(record)
        diagnosis = record.get("diagnosis_json", {}) if isinstance(record.get("diagnosis_json"), Mapping) else {}
        profile = record.get("profile_json", {}) if isinstance(record.get("profile_json"), Mapping) else {}
        label_value = record.get("comfort_score")
        label_available = label_value is not None and not pd.isna(label_value) and str(label_value).strip() != ""
        main_problem_labels = combine_problem_labels(
            diagnosis_labels=diagnosis.get("problem_labels", []),
            primary_problem_label=record.get("primary_problem_label"),
        )

        diagnosis_severity = compute_diagnosis_severity(record)
        comfort_risk_score = compute_comfort_risk(record, labels=main_problem_labels)
        vitality_deficit_score = compute_vitality_deficit(record, street_type=street_type, labels=main_problem_labels)
        eventfulness_risk_or_gap, eventfulness_direction, observed_eventfulness = compute_eventfulness_gap(
            record,
            street_type=street_type,
        )
        consistency_score = compute_multimodal_consistency(record)
        consistency = consistency_flag(consistency_score)
        evidence_score = _actionable_evidence_score(
            record,
            consistency_score=consistency_score,
            label_available=bool(label_available),
            diagnosis_available=bool(diagnosis),
            profile_available=bool(profile),
        )
        overall_problem_proxy = estimate_observed_score(record, "overall_problem_severity")
        label_severity_score = clamp01((overall_problem_proxy - 1.0) / 6.0)

        confirmatory_need_score = clamp01(
            0.42 * comfort_risk_score
            + 0.34 * vitality_deficit_score
            + 0.24 * eventfulness_risk_or_gap
        )
        priority_score = clamp01(
            0.30 * diagnosis_severity
            + 0.18 * label_severity_score
            + 0.24 * confirmatory_need_score
            + 0.14 * consistency_score
            + 0.14 * evidence_score
        )

        focus_targets = _confirmatory_focus(
            comfort_risk_score=comfort_risk_score,
            vitality_deficit_score=vitality_deficit_score,
            eventfulness_risk_or_gap=eventfulness_risk_or_gap,
        )
        theme = _theme(
            comfort_risk_score=comfort_risk_score,
            vitality_deficit_score=vitality_deficit_score,
            eventfulness_risk_or_gap=eventfulness_risk_or_gap,
        )

        context: Dict[str, Any] = {
            "segment_id": segment_id,
            "street_type": street_type,
            "main_problem_labels": list(main_problem_labels),
            "diagnosis_severity": diagnosis_severity,
            "comfort_risk_score": comfort_risk_score,
            "vitality_deficit_score": vitality_deficit_score,
            "eventfulness_risk_or_gap": eventfulness_risk_or_gap,
            "eventfulness_direction": eventfulness_direction,
            "observed_eventfulness": observed_eventfulness,
            "multimodal_consistency_score": consistency_score,
            "multimodal_consistency_flag": consistency,
            "evidence_strength_score": evidence_score,
            "evidence_strength": _evidence_strength(evidence_score),
            "adjudication_confidence_flag": adjudication_confidence_flag(record),
            "confirmatory_target_focus": focus_targets,
            "recommended_intervention_theme": theme,
        }
        contexts[segment_id] = context

        rows.append(
            {
                "segment_id": segment_id,
                "start_time_sec": record.get("start_time_sec"),
                "end_time_sec": record.get("end_time_sec"),
                "priority_score": priority_score,
                "priority_level": _priority_level(priority_score),
                "main_problem_labels": ";".join(main_problem_labels),
                "evidence_strength": context["evidence_strength"],
                "comfort_risk_score": comfort_risk_score,
                "vitality_deficit_score": vitality_deficit_score,
                "eventfulness_risk_or_gap": eventfulness_risk_or_gap,
                "multimodal_consistency_flag": consistency,
                "adjudication_confidence_flag": context["adjudication_confidence_flag"],
                "recommended_intervention_theme": theme,
                "street_type": street_type,
                "confirmatory_target_focus": ";".join(focus_targets),
                "diagnosis_severity": diagnosis_severity,
            }
        )

    ranking_df = pd.DataFrame(rows)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(
            by=["priority_score", "diagnosis_severity", "segment_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        ranking_df["priority_rank"] = ranking_df.index + 1
    logger.info("built segment priority ranking | segments=%s", len(ranking_df))
    return ranking_df, contexts
