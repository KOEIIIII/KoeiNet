


"""Constrained edit prompt generation for Step-8 design mapping."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


def _goal_sentence(theme: str, street_type: str, eventfulness_direction: str) -> str:
    if theme == "comfort_buffering":
        return (
            "prioritize pedestrian comfort by reducing traffic or mechanical exposure and clarifying protective edge conditions"
        )
    if theme == "vitality_activation":
        return (
            "improve human-scale vitality through feasible stay nodes, frontage activation, and encounter-supporting micro interventions"
        )
    if theme == "eventfulness_calibration":
        if eventfulness_direction == "decrease":
            return (
                f"calibrate eventfulness downward for a {street_type} street by reducing chaotic or traffic-dominated stimulation"
            )
        return (
            f"calibrate eventfulness upward for a {street_type} street through moderate, human-scale liveliness rather than noise"
        )
    return (
        "rebalance comfort, vitality, and eventfulness with small-scale, feasible urban design edits"
    )


def build_expected_effects(
    *,
    theme: str,
    allowed_interventions: Sequence[str],
    eventfulness_direction: str,
) -> Dict[str, str]:
    comfort_change = "maintain"
    vitality_change = "maintain"
    eventfulness_change = "maintain"

    if theme in {"comfort_buffering", "mixed_rebalancing"}:
        comfort_change = "increase_moderately"
    if any(
        ("stay nodes" in item)
        or ("support active frontage" in item)
        or ("micro public-use" in item)
        for item in allowed_interventions
    ):
        vitality_change = "increase_moderately"
    if any("calmness" in item or "traffic" in item for item in allowed_interventions):
        comfort_change = "increase_moderately"
    if eventfulness_direction == "decrease":
        eventfulness_change = "decrease_slightly"
    elif eventfulness_direction == "increase":
        eventfulness_change = "increase_slightly"

    return {
        "expected_comfort_change": comfort_change,
        "expected_vitality_change": vitality_change,
        "expected_eventfulness_change": eventfulness_change,
    }


def build_review_checklist(
    *,
    street_type: str,
    confirmatory_target_focus: Sequence[str],
    eventfulness_direction: str,
) -> List[str]:
    checklist = [
        "Preserve viewpoint, perspective, and scene identity.",
        "Keep interventions feasible at local street-element scale rather than changing urban massing.",
        "Ensure soundscape logic is explicit, not implied by generic beautification.",
        "Do not introduce unrealistic visible sound sources or staged crowds.",
    ]
    if "comfort_score" in confirmatory_target_focus:
        checklist.append("Verify that pedestrian comfort improves through shielding, buffering, order, or reduced exposure.")
    if "vitality_score" in confirmatory_target_focus:
        checklist.append("Verify that vitality support comes from human-scale use opportunities rather than spectacle.")
    if "soundscape_eventfulness" in confirmatory_target_focus:
        if eventfulness_direction == "decrease":
            checklist.append(
                f"Because street type is {street_type}, verify that overstimulation is reduced without deadening legitimate activity."
            )
        else:
            checklist.append(
                f"Because street type is {street_type}, verify that added liveliness stays moderate and contextually appropriate."
            )
    return checklist


def build_edit_prompt(
    *,
    street_type: str,
    theme: str,
    confirmatory_target_focus: Sequence[str],
    allowed_interventions: Sequence[str],
    forbidden_changes: Sequence[str],
    soundscape_state: Mapping[str, Any],
    eventfulness_direction: str,
) -> str:
    goal_sentence = _goal_sentence(theme, street_type, eventfulness_direction)
    interventions = "; ".join(allowed_interventions[:8])
    soundscape_sentence = (
        "Use soundscape-aware logic explicitly: read dominant sources as "
        f"{', '.join(str(x) for x in soundscape_state.get('dominant_sources', []) or ['mixed'])}, "
        f"treat pleasantness support as {soundscape_state.get('pleasantness_support_level', 'unknown')}, "
        f"and treat eventfulness as {soundscape_state.get('eventfulness_state', 'balanced')}."
    )
    focus_sentence = "Primary confirmatory target focus: " + ", ".join(confirmatory_target_focus) + "."
    constraints = "; ".join(item[0].upper() + item[1:] if item else "" for item in forbidden_changes if item) + "."

    return (
        "Edit this streetscape conservatively for planning visualization. "
        f"Street type context: {street_type}. "
        f"Primary goal: {goal_sentence}. "
        f"{focus_sentence} "
        f"Apply only feasible local interventions such as: {interventions}. "
        f"{soundscape_sentence} "
        "Preserve original geometry, camera viewpoint, road alignment, building massing, and scene identity. "
        f"{constraints}"
    )


def build_prompt_record(
    *,
    segment_id: int,
    street_type: str,
    theme: str,
    confirmatory_target_focus: Sequence[str],
    allowed_interventions: Sequence[str],
    forbidden_changes: Sequence[str],
    soundscape_state: Mapping[str, Any],
    eventfulness_direction: str,
) -> Dict[str, Any]:
    prompt = build_edit_prompt(
        street_type=street_type,
        theme=theme,
        confirmatory_target_focus=confirmatory_target_focus,
        allowed_interventions=allowed_interventions,
        forbidden_changes=forbidden_changes,
        soundscape_state=soundscape_state,
        eventfulness_direction=eventfulness_direction,
    )
    return {
        "segment_id": int(segment_id),
        "street_type": street_type,
        "goal_theme": theme,
        "confirmatory_target_focus": list(confirmatory_target_focus),
        "prompt": prompt,
        "forbidden_changes": list(forbidden_changes),
        "review_checklist": build_review_checklist(
            street_type=street_type,
            confirmatory_target_focus=confirmatory_target_focus,
            eventfulness_direction=eventfulness_direction,
        ),
    }
