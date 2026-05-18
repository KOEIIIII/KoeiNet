


"""Segment-level design plan mapping for Step-8."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from .design_rules import (
    classify_soundscape_state,
    resolve_interventions,
)
from .priority_ranker import build_segment_priority_ranking
from .prompt_builder import (
    build_expected_effects,
    build_prompt_record,
)

logger = logging.getLogger("design.design_mapper")


def _human_label_snapshot(record: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "safety_score",
        "comfort_score",
        "vitality_score",
        "overall_problem_severity",
        "soundscape_pleasantness",
        "soundscape_eventfulness",
        "primary_problem_label",
        "confidence_score",
    ):
        value = record.get(key)
        if value is None or pd.isna(value) or str(value).strip() == "":
            continue
        out[key] = value
    return out


def _observed_cues(record: Mapping[str, Any], ctx: Mapping[str, Any]) -> List[str]:
    cues: List[str] = []
    people_mean = record.get("people__total_people__mean")
    if people_mean is not None:
        cues.append(f"people_mean={float(people_mean):.1f}")
    green_view = record.get("green_view__greenviewindex__mean")
    if green_view is not None:
        cues.append(f"green_view={float(green_view):.3f}")
    traffic_ratio = record.get("audio_events__group_ratio_traffic")
    if traffic_ratio is not None:
        cues.append(f"traffic_audio_ratio={float(traffic_ratio):.3f}")
    human_ratio = record.get("audio_events__group_ratio_human")
    if human_ratio is not None:
        cues.append(f"human_audio_ratio={float(human_ratio):.3f}")
    loudness = record.get("audio_signal__loudness_proxy_db")
    if loudness is not None:
        cues.append(f"loudness_proxy_db={float(loudness):.2f}")
    cues.append(f"comfort_risk={ctx['comfort_risk_score']:.3f}")
    cues.append(f"vitality_deficit={ctx['vitality_deficit_score']:.3f}")
    cues.append(f"eventfulness_gap={ctx['eventfulness_risk_or_gap']:.3f}")
    return cues


def _design_targets(
    *,
    ctx: Mapping[str, Any],
    evidence_registry: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for target_name in ctx.get("confirmatory_target_focus", []):
        target_policy = (
            evidence_registry.get("confirmatory_targets", {}).get(target_name, {})
            if isinstance(evidence_registry.get("confirmatory_targets"), Mapping)
            else {}
        )
        objective = "increase"
        if target_name == "soundscape_eventfulness":
            objective = str(ctx.get("eventfulness_direction", "maintain"))
        targets.append(
            {
                "target_name": target_name,
                "objective": objective,
                "evidence_role": "confirmatory",
                "policy_summary": target_policy.get("policy_summary", ""),
                "observed_best_model_group": target_policy.get("observed_best_model_group", ""),
            }
        )
    return targets


def _planning_rationale(
    *,
    record: Mapping[str, Any],
    ctx: Mapping[str, Any],
    evidence_registry: Mapping[str, Any],
    rule_result: Mapping[str, Any],
    soundscape_state: Mapping[str, Any],
) -> List[str]:
    lines: List[str] = []
    lines.append(
        f"Street type inferred as {ctx['street_type']}, so intervention logic follows {ctx['street_type']} modulation rather than a universal activation rule."
    )
    for target_name in ctx.get("confirmatory_target_focus", []):
        policy = (
            evidence_registry.get("confirmatory_targets", {}).get(target_name, {})
            if isinstance(evidence_registry.get("confirmatory_targets"), Mapping)
            else {}
        )
        summary = str(policy.get("policy_summary", "")).strip()
        if summary:
            lines.append(f"{target_name}: {summary}")
    if soundscape_state.get("eventfulness_state") == "overstimulated":
        lines.append("Soundscape indicates overstimulation, so eventfulness is calibrated downward instead of amplified.")
    elif soundscape_state.get("eventfulness_state") == "understimulated":
        lines.append("Soundscape indicates insufficient liveliness for this street type, so moderate activation is justified.")
    diagnosis = record.get("diagnosis_json", {}) if isinstance(record.get("diagnosis_json"), Mapping) else {}
    cross_modal_reason = str(diagnosis.get("cross_modal_reason", "")).strip()
    if cross_modal_reason:
        lines.append(cross_modal_reason)
    lines.extend(str(item) for item in rule_result.get("rule_reasons", []))
    return lines[:8]


def build_design_artifacts(
    *,
    segment_records: Sequence[Mapping[str, Any]],
    evidence_registry: Mapping[str, Any],
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate Step-8 ranking, design plans, intervention matrix, and edit prompts.

    When `top_n` is None or <= 0, all ranked segments receive full design outputs.
    """
    ranking_df, contexts = build_segment_priority_ranking(segment_records)
    record_map = {int(item["segment_id"]): dict(item) for item in segment_records}

    if ranking_df.empty:
        return {
            "ranking_df": ranking_df,
            "plan_records": [],
            "intervention_df": pd.DataFrame(),
            "prompt_records": [],
            "summary_payload": {
                "total_segments_ranked": 0,
                "selected_segments_for_design_plan": 0,
            },
        }

    selected_count = len(ranking_df) if not top_n or int(top_n) <= 0 else min(int(top_n), len(ranking_df))
    selected_ids = set(int(x) for x in ranking_df.head(selected_count)["segment_id"].tolist())
    ranking_df = ranking_df.copy()
    ranking_df["selected_for_design_focus"] = ranking_df["segment_id"].isin(selected_ids)

    plan_records: List[Dict[str, Any]] = []
    prompt_records: List[Dict[str, Any]] = []
    intervention_rows: List[Dict[str, Any]] = []

    focus_counter: Counter[str] = Counter()
    theme_counter: Counter[str] = Counter()

    for segment_id in ranking_df["segment_id"].astype(int).tolist():
        if segment_id not in selected_ids:
            continue
        record = record_map[segment_id]
        ctx = contexts[segment_id]
        soundscape_state = classify_soundscape_state(record, ctx["street_type"])
        rule_result = resolve_interventions(
            record=record,
            street_type=ctx["street_type"],
            main_problem_labels=ctx["main_problem_labels"],
            comfort_risk_score=float(ctx["comfort_risk_score"]),
            vitality_deficit_score=float(ctx["vitality_deficit_score"]),
            eventfulness_risk_or_gap=float(ctx["eventfulness_risk_or_gap"]),
            eventfulness_direction=str(ctx["eventfulness_direction"]),
            soundscape_state=soundscape_state,
        )
        prompt_record = build_prompt_record(
            segment_id=segment_id,
            street_type=ctx["street_type"],
            theme=ctx["recommended_intervention_theme"],
            confirmatory_target_focus=ctx["confirmatory_target_focus"],
            allowed_interventions=rule_result["allowed_interventions"],
            forbidden_changes=rule_result["forbidden_changes"],
            soundscape_state=soundscape_state,
            eventfulness_direction=str(ctx["eventfulness_direction"]),
        )
        expected_effects = build_expected_effects(
            theme=ctx["recommended_intervention_theme"],
            allowed_interventions=rule_result["allowed_interventions"],
            eventfulness_direction=str(ctx["eventfulness_direction"]),
        )
        evidence_summary = {
            "observed_cues": _observed_cues(record, ctx),
            "human_label_snapshot": _human_label_snapshot(record),
            "step75_target_evidence": _design_targets(ctx=ctx, evidence_registry=evidence_registry),
            "evidence_strength": ctx["evidence_strength"],
            "multimodal_consistency_flag": ctx["multimodal_consistency_flag"],
        }
        diagnosis = record.get("diagnosis_json", {}) if isinstance(record.get("diagnosis_json"), Mapping) else {}
        cross_modal_reason = str(diagnosis.get("cross_modal_reason", "")).strip()
        if not cross_modal_reason:
            cross_modal_reason = (
                f"Multimodal consistency is {ctx['multimodal_consistency_flag']}; design therefore combines visual and soundscape evidence rather than relying on one modality."
            )

        plan_record = {
            "segment_id": segment_id,
            "start_time_sec": record.get("start_time_sec"),
            "end_time_sec": record.get("end_time_sec"),
            "priority_rank": int(ranking_df.loc[ranking_df["segment_id"] == segment_id, "priority_rank"].iloc[0]),
            "priority_score": float(ranking_df.loc[ranking_df["segment_id"] == segment_id, "priority_score"].iloc[0]),
            "street_type": ctx["street_type"],
            "main_problem_labels": list(ctx["main_problem_labels"]),
            "evidence_summary": evidence_summary,
            "confirmatory_target_focus": list(ctx["confirmatory_target_focus"]),
            "soundscape_state": soundscape_state,
            "cross_modal_reason": cross_modal_reason,
            "design_targets": _design_targets(ctx=ctx, evidence_registry=evidence_registry),
            "allowed_interventions": list(rule_result["allowed_interventions"]),
            "forbidden_changes": list(rule_result["forbidden_changes"]),
            "planning_rationale": _planning_rationale(
                record=record,
                ctx=ctx,
                evidence_registry=evidence_registry,
                rule_result=rule_result,
                soundscape_state=soundscape_state,
            ),
            "expected_effects": expected_effects,
            "edit_prompt": prompt_record["prompt"],
            "review_checklist": prompt_record["review_checklist"],
        }
        plan_records.append(plan_record)
        prompt_records.append(prompt_record)

        matrix_row: Dict[str, Any] = {
            "segment_id": segment_id,
            "street_type": ctx["street_type"],
            "priority_rank": plan_record["priority_rank"],
            "priority_score": plan_record["priority_score"],
            "priority_level": ranking_df.loc[ranking_df["segment_id"] == segment_id, "priority_level"].iloc[0],
            "recommended_intervention_theme": ctx["recommended_intervention_theme"],
            "confirmatory_target_focus": ";".join(ctx["confirmatory_target_focus"]),
            "eventfulness_direction": ctx["eventfulness_direction"],
        }
        matrix_row.update(rule_result["flags"])
        intervention_rows.append(matrix_row)

        focus_counter.update(ctx["confirmatory_target_focus"])
        theme_counter.update([ctx["recommended_intervention_theme"]])

    intervention_df = pd.DataFrame(intervention_rows)
    summary_payload = {
        "total_segments_ranked": int(len(ranking_df)),
        "selected_segments_for_design_plan": int(len(plan_records)),
        "street_type_counts": {
            str(k): int(v) for k, v in ranking_df["street_type"].value_counts(dropna=False).to_dict().items()
        },
        "priority_level_counts": {
            str(k): int(v) for k, v in ranking_df["priority_level"].value_counts(dropna=False).to_dict().items()
        },
        "focus_target_counts": {str(k): int(v) for k, v in focus_counter.items()},
        "theme_counts": {str(k): int(v) for k, v in theme_counter.items()},
        "top_priority_segments": ranking_df.head(min(10, len(ranking_df))).to_dict(orient="records"),
    }
    logger.info(
        "design artifacts built | ranked=%s selected=%s",
        len(ranking_df),
        len(plan_records),
    )
    return {
        "ranking_df": ranking_df,
        "plan_records": plan_records,
        "intervention_df": intervention_df,
        "prompt_records": prompt_records,
        "summary_payload": summary_payload,
    }
