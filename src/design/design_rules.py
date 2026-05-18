


"""Transparent street-type-aware design rules for Step-8 planning outputs."""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

logger = logging.getLogger("design.design_rules")

DEFAULT_FORBIDDEN_CHANGES: List[str] = [
    "do not change camera viewpoint",
    "do not alter road direction",
    "do not invent new buildings",
    "do not change overall massing",
    "do not create unrealistic sound sources visually",
]

INTERVENTION_LABELS: Dict[str, str] = {
    "add_green_buffer": "add or strengthen feasible vegetation buffers at pedestrian edge",
    "strengthen_edge": "strengthen curb or edge definition for pedestrian protection",
    "reduce_traffic_exposure": "reduce direct pedestrian exposure to traffic and mechanical noise",
    "soften_hardscape_interfaces": "soften harsh hardscape or frontage interfaces with human-scale materials/elements",
    "reorganize_curbside": "reorganize curbside occupation and remove chaotic edge clutter",
    "improve_order_and_management": "improve maintenance, order, and visual management quality",
    "enhance_stay_node": "add or strengthen stay nodes, pause points, and short-duration seating opportunities",
    "activate_frontage": "support active frontage and visible small-scale edge activity",
    "support_micro_public_use": "introduce micro public-use anchors that support encounter and optional staying",
    "preserve_positive_human_activity": "preserve existing positive human-scale activity cues",
    "support_natural_sound_buffer": "use greenery or shielding elements that can also improve soundscape pleasantness",
    "support_cultural_liveliness": "support moderate cultural or social liveliness where street type allows",
    "avoid_overactivation": "avoid adding excessive activity, clutter, or noisy attraction points",
    "prioritize_calmness": "prioritize calmness and legibility over extra activation",
    "protect_existing_greenery": "protect and reinforce existing greenery rather than replacing it",
    "maintain_accessibility": "maintain clear pedestrian accessibility and movement continuity",
}

INTERVENTION_COLUMNS: List[str] = list(INTERVENTION_LABELS.keys())

STREET_TYPE_POLICIES: Dict[str, Dict[str, Any]] = {
    "commercial_social": {
        "desired_vitality_range": [5.0, 6.5],
        "desired_eventfulness_range": [4.5, 6.0],
        "acceptable_human_sound_density": "moderate_high",
        "design_priority": "balanced_activation",
        "modulation_note": "Some liveliness is desirable, but traffic-dominated chaos is not.",
    },
    "transport_movement": {
        "desired_vitality_range": [3.0, 4.5],
        "desired_eventfulness_range": [2.5, 4.5],
        "acceptable_human_sound_density": "low_moderate",
        "design_priority": "calmness_and_legibility",
        "modulation_note": "Calmness, separation, and legibility outweigh extra activation.",
    },
    "leisure_cultural": {
        "desired_vitality_range": [4.5, 6.0],
        "desired_eventfulness_range": [3.5, 5.5],
        "acceptable_human_sound_density": "moderate",
        "design_priority": "comfortable_activation",
        "modulation_note": "Human-scale and culturally positive liveliness can be beneficial if comfort is protected.",
    },
    "mixed_uncertain": {
        "desired_vitality_range": [4.0, 5.5],
        "desired_eventfulness_range": [3.5, 5.0],
        "acceptable_human_sound_density": "moderate",
        "design_priority": "balanced_recalibration",
        "modulation_note": "Use moderate activation and avoid hard assumptions about desired liveliness.",
    },
}

LABEL_PATTERN_MAP: Tuple[Tuple[str, str], ...] = (
    ("traffic_noise", "traffic_noise"),
    ("noise", "traffic_noise"),
    ("traffic", "traffic_noise"),
    ("机械", "traffic_noise"),
    ("噪声", "traffic_noise"),
    ("acoustic", "traffic_noise"),
    ("crowd", "pedestrian_discomfort"),
    ("crowding", "pedestrian_discomfort"),
    ("pedestrian", "pedestrian_discomfort"),
    ("人群", "pedestrian_discomfort"),
    ("拥挤", "pedestrian_discomfort"),
    ("green", "low_greenery"),
    ("绿", "low_greenery"),
    ("vegetation", "low_greenery"),
    ("clutter", "visual_clutter"),
    ("visual fatigue", "visual_clutter"),
    ("杂乱", "visual_clutter"),
    ("装饰", "visual_clutter"),
    ("stay", "weak_stay_quality"),
    ("停留", "weak_stay_quality"),
    ("seat", "weak_stay_quality"),
    ("boring", "low_vitality"),
    ("vitality", "low_vitality"),
    ("活力", "low_vitality"),
    ("无聊", "low_vitality"),
    ("safety", "safety_risk"),
    ("risk", "safety_risk"),
    ("危险", "safety_risk"),
    ("人车", "safety_risk"),
    ("mixed", "mixed_or_unclear"),
    ("unclear", "mixed_or_unclear"),
    ("不明确", "mixed_or_unclear"),
    ("no_major_problem", "no_major_problem"),
    ("无明显", "no_major_problem"),
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return float(default)
        out = float(value)
        if math.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def clamp01(value: Any) -> float:
    return max(0.0, min(1.0, safe_float(value, 0.0)))


def scale_between(value: Any, low: float, high: float) -> float:
    numeric = safe_float(value, low)
    if high <= low:
        return 0.0
    return clamp01((numeric - low) / (high - low))


def inverse_scale(value: Any, low: float, high: float) -> float:
    return 1.0 - scale_between(value, low, high)


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def canonicalize_problem_label(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return "mixed_or_unclear"
    for pattern, label in LABEL_PATTERN_MAP:
        if pattern in text:
            return label
    return "mixed_or_unclear"


def combine_problem_labels(
    diagnosis_labels: Sequence[Any],
    primary_problem_label: Any = None,
) -> List[str]:
    ordered: List[str] = []
    for raw in list(diagnosis_labels) + ([primary_problem_label] if primary_problem_label else []):
        label = canonicalize_problem_label(raw)
        if label not in ordered:
            ordered.append(label)
    if not ordered:
        ordered.append("mixed_or_unclear")
    return ordered


def _feature(record: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    return safe_float(record.get(key), default)


def infer_street_type(record: Mapping[str, Any]) -> str:
    """Infer a lightweight street type from multimodal evidence."""
    people_mean = _feature(record, "people__total_people__mean")
    human_ratio = _feature(record, "audio_events__group_ratio_human")
    traffic_ratio = _feature(record, "audio_events__group_ratio_traffic")
    mechanical_ratio = _feature(record, "audio_events__group_ratio_mechanical")
    nature_ratio = _feature(record, "audio_events__group_ratio_nature")
    green_view = _feature(record, "green_view__greenviewindex__mean")
    vehicle_visual = _feature(record, "visual_major__vehicle__mean") + _feature(record, "visual_semantic__car__mean")
    road_visual = _feature(record, "visual_semantic__road__mean")
    shopping_score = _feature(record, "ai_activity__summary_shopping_mean_score") or _feature(record, "ai_activity__买菜购物_score__mean")
    walking_score = _feature(record, "ai_activity__summary_walking_mean_score") or _feature(record, "ai_activity__散步_score__mean")
    sitting_score = _feature(record, "ai_activity__summary_sitting_mean_score") or _feature(record, "ai_activity__坐下休息_score__mean")
    standing_score = _feature(record, "ai_activity__summary_standing_mean_score") or _feature(record, "ai_activity__站着停留_score__mean")
    music_signal = _feature(record, "audio_events_dist__music") + _feature(record, "audio_events_topk__music")

    diagnosis = record.get("diagnosis_json", {}) if isinstance(record.get("diagnosis_json"), Mapping) else {}
    profile = record.get("profile_json", {}) if isinstance(record.get("profile_json"), Mapping) else {}
    text_blob = " ".join(
        [
            _clean_text(x)
            for x in (
                diagnosis.get("cross_modal_reason"),
                profile.get("concise_summary"),
                " ".join(str(i) for i in as_list(diagnosis.get("problem_labels"))),
            )
            if str(x or "").strip()
        ]
    )

    scores = {
        "commercial_social": 0.0,
        "transport_movement": 0.0,
        "leisure_cultural": 0.0,
        "mixed_uncertain": 0.15,
    }

    scores["commercial_social"] += 0.30 * scale_between(people_mean, 8.0, 30.0)
    scores["commercial_social"] += 0.20 * scale_between(human_ratio, 0.25, 0.85)
    scores["commercial_social"] += 0.20 * scale_between(max(shopping_score, walking_score, standing_score), 2.0, 5.0)
    scores["commercial_social"] += 0.10 if "shopping" in text_blob or "commercial" in text_blob else 0.0
    scores["commercial_social"] += 0.10 if "public space" in text_blob or "pedestrian" in text_blob else 0.0

    scores["transport_movement"] += 0.35 * scale_between(traffic_ratio + mechanical_ratio, 0.03, 0.35)
    scores["transport_movement"] += 0.20 * scale_between(vehicle_visual, 0.02, 0.18)
    scores["transport_movement"] += 0.15 * scale_between(road_visual, 0.08, 0.30)
    scores["transport_movement"] += 0.10 if "truck" in text_blob or "traffic" in text_blob else 0.0
    scores["transport_movement"] += 0.10 if "人车" in text_blob or "movement" in text_blob else 0.0

    scores["leisure_cultural"] += 0.25 * scale_between(green_view, 0.18, 0.40)
    scores["leisure_cultural"] += 0.15 * scale_between(nature_ratio, 0.005, 0.10)
    scores["leisure_cultural"] += 0.15 * scale_between(music_signal, 0.0, 0.35)
    scores["leisure_cultural"] += 0.15 * scale_between(max(sitting_score, standing_score), 1.5, 4.5)
    scores["leisure_cultural"] += 0.10 if "cultural" in text_blob or "music" in text_blob else 0.0
    scores["leisure_cultural"] += 0.10 if "leisure" in text_blob or "park" in text_blob else 0.0

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    if (best_score - second_score) < 0.08:
        return "mixed_uncertain"
    return best_name


def desired_target_range(street_type: str, target_name: str) -> Tuple[float, float]:
    policy = STREET_TYPE_POLICIES.get(street_type, STREET_TYPE_POLICIES["mixed_uncertain"])
    if target_name == "vitality_score":
        low, high = policy["desired_vitality_range"]
        return float(low), float(high)
    if target_name == "soundscape_eventfulness":
        low, high = policy["desired_eventfulness_range"]
        return float(low), float(high)
    return 4.0, 5.5


def estimate_observed_score(record: Mapping[str, Any], target_name: str) -> float:
    raw = record.get(target_name)
    if raw is not None and str(raw).strip() != "":
        numeric = safe_float(raw, float("nan"))
        if not math.isnan(numeric):
            return numeric

    green_view = _feature(record, "green_view__greenviewindex__mean")
    safety_emotion = _feature(record, "emotion__safety__mean")
    beautiful = _feature(record, "emotion__beautiful__mean")
    lively = _feature(record, "emotion__lively__mean")
    boring = _feature(record, "emotion__boring__mean")
    people_mean = _feature(record, "people__total_people__mean")
    human_ratio = _feature(record, "audio_events__group_ratio_human")
    traffic_ratio = _feature(record, "audio_events__group_ratio_traffic")
    mechanical_ratio = _feature(record, "audio_events__group_ratio_mechanical")
    nature_ratio = _feature(record, "audio_events__group_ratio_nature")
    loudness = _feature(record, "audio_signal__loudness_proxy_db", -24.0)
    roughness = _feature(record, "audio_signal__roughness_proxy")
    walking = _feature(record, "ai_activity__summary_walking_mean_score") or _feature(record, "ai_activity__散步_score__mean")
    shopping = _feature(record, "ai_activity__summary_shopping_mean_score") or _feature(record, "ai_activity__买菜购物_score__mean")
    standing = _feature(record, "ai_activity__summary_standing_mean_score") or _feature(record, "ai_activity__站着停留_score__mean")
    music_signal = _feature(record, "audio_events_dist__music") + _feature(record, "audio_events_topk__music")

    if target_name == "comfort_score":
        comfort_goodness = (
            0.28 * inverse_scale(traffic_ratio + mechanical_ratio, 0.04, 0.30)
            + 0.18 * inverse_scale(roughness, 0.05, 0.18)
            + 0.18 * scale_between(green_view, 0.15, 0.35)
            + 0.14 * scale_between(safety_emotion, 0.30, 0.55)
            + 0.12 * scale_between(beautiful, 0.55, 0.85)
            + 0.10 * inverse_scale(loudness, -28.0, -15.0)
        )
        return 1.0 + 6.0 * clamp01(comfort_goodness)

    if target_name == "vitality_score":
        vitality_goodness = (
            0.28 * scale_between(people_mean, 4.0, 30.0)
            + 0.22 * scale_between(lively, 0.55, 0.85)
            + 0.20 * scale_between(max(walking, shopping, standing), 2.0, 5.0)
            + 0.20 * scale_between(human_ratio + music_signal, 0.15, 0.95)
            + 0.10 * inverse_scale(boring, 0.45, 0.78)
        )
        return 1.0 + 6.0 * clamp01(vitality_goodness)

    if target_name == "soundscape_eventfulness":
        eventfulness_goodness = (
            0.25 * scale_between(people_mean, 4.0, 32.0)
            + 0.20 * scale_between(lively, 0.55, 0.85)
            + 0.20 * scale_between(human_ratio + traffic_ratio + music_signal, 0.15, 1.0)
            + 0.20 * scale_between(loudness, -28.0, -14.0)
            + 0.15 * scale_between(max(walking, shopping, standing), 2.0, 5.0)
        )
        return 1.0 + 6.0 * clamp01(eventfulness_goodness)

    if target_name == "soundscape_pleasantness":
        pleasantness_goodness = (
            0.25 * inverse_scale(traffic_ratio + mechanical_ratio, 0.04, 0.30)
            + 0.25 * scale_between(nature_ratio + green_view * 0.3, 0.03, 0.20)
            + 0.20 * inverse_scale(loudness, -28.0, -15.0)
            + 0.15 * inverse_scale(roughness, 0.05, 0.18)
            + 0.15 * scale_between(beautiful, 0.55, 0.85)
        )
        return 1.0 + 6.0 * clamp01(pleasantness_goodness)

    return 4.0


def compute_multimodal_consistency(record: Mapping[str, Any]) -> float:
    people_signal = scale_between(_feature(record, "people__total_people__mean"), 4.0, 30.0)
    human_audio = scale_between(_feature(record, "audio_events__group_ratio_human"), 0.15, 0.85)
    traffic_visual = scale_between(
        _feature(record, "visual_major__vehicle__mean") + _feature(record, "visual_semantic__road__mean"),
        0.05,
        0.40,
    )
    traffic_audio = scale_between(
        _feature(record, "audio_events__group_ratio_traffic") + _feature(record, "audio_events__group_ratio_mechanical"),
        0.03,
        0.35,
    )
    green_visual = scale_between(_feature(record, "green_view__greenviewindex__mean"), 0.10, 0.35)
    nature_audio = scale_between(_feature(record, "audio_events__group_ratio_nature"), 0.0, 0.08)
    lively_visual = scale_between(
        (_feature(record, "emotion__lively__mean") + scale_between(_feature(record, "ai_activity__summary_walking_mean_score"), 1.0, 5.0)) / 2.0,
        0.35,
        0.90,
    )
    lively_audio = scale_between(
        _feature(record, "audio_events__group_ratio_human") + _feature(record, "audio_events_dist__music"),
        0.15,
        0.95,
    )
    pair_scores = [
        1.0 - abs(people_signal - human_audio),
        1.0 - abs(traffic_visual - traffic_audio),
        1.0 - abs(green_visual - nature_audio),
        1.0 - abs(lively_visual - lively_audio),
    ]
    score = sum(clamp01(x) for x in pair_scores) / float(len(pair_scores))
    diagnosis = record.get("diagnosis_json", {}) if isinstance(record.get("diagnosis_json"), Mapping) else {}
    if str(diagnosis.get("cross_modal_reason", "")).strip():
        score = clamp01(score + 0.05)
    return score


def consistency_flag(score: float) -> str:
    if score >= 0.67:
        return "consistent"
    if score >= 0.45:
        return "mixed"
    return "weak"


def compute_diagnosis_severity(record: Mapping[str, Any]) -> float:
    diagnosis = record.get("diagnosis_json", {}) if isinstance(record.get("diagnosis_json"), Mapping) else {}
    severity_scores = diagnosis.get("severity_scores", {})
    if isinstance(severity_scores, Mapping) and severity_scores:
        values = [clamp01(v) for v in severity_scores.values()]
        if values:
            return clamp01(0.60 * max(values) + 0.40 * (sum(values) / float(len(values))))
    return clamp01((7.0 - estimate_observed_score(record, "comfort_score")) / 6.0)


def compute_comfort_risk(record: Mapping[str, Any], labels: Sequence[str]) -> float:
    observed = estimate_observed_score(record, "comfort_score")
    traffic_noise = scale_between(
        _feature(record, "audio_events__group_ratio_traffic") + 1.2 * _feature(record, "audio_events__group_ratio_mechanical"),
        0.04,
        0.35,
    )
    loudness_risk = scale_between(_feature(record, "audio_signal__loudness_proxy_db"), -28.0, -14.0)
    roughness_risk = scale_between(_feature(record, "audio_signal__roughness_proxy"), 0.05, 0.18)
    green_deficit = inverse_scale(_feature(record, "green_view__greenviewindex__mean"), 0.16, 0.35)
    edge_harshness = clamp01(
        0.60 * scale_between(_feature(record, "visual_major__vehicle__mean"), 0.02, 0.18)
        + 0.40 * inverse_scale(_feature(record, "visual_semantic__sidewalk__mean"), 0.01, 0.06)
    )
    label_deficit = clamp01((5.5 - observed) / 4.5)
    label_bonus = 0.12 if any(x in labels for x in ("traffic_noise", "pedestrian_discomfort", "visual_clutter")) else 0.0
    score = (
        0.32 * label_deficit
        + 0.26 * traffic_noise
        + 0.10 * loudness_risk
        + 0.10 * roughness_risk
        + 0.12 * green_deficit
        + 0.10 * edge_harshness
        + label_bonus
    )
    return clamp01(score)


def compute_vitality_deficit(record: Mapping[str, Any], street_type: str, labels: Sequence[str]) -> float:
    observed = estimate_observed_score(record, "vitality_score")
    desired_low, _ = desired_target_range(street_type, "vitality_score")
    low_gap = clamp01((desired_low - observed) / 3.0) if observed < desired_low else 0.0
    people_support = scale_between(_feature(record, "people__total_people__mean"), 4.0, 28.0)
    activity_support = scale_between(
        max(
            _feature(record, "ai_activity__summary_walking_mean_score"),
            _feature(record, "ai_activity__summary_shopping_mean_score"),
            _feature(record, "ai_activity__summary_standing_mean_score"),
        ),
        2.0,
        5.0,
    )
    stay_support = scale_between(
        max(
            _feature(record, "ai_activity__summary_sitting_mean_score"),
            _feature(record, "ai_activity__summary_standing_mean_score"),
        ),
        1.5,
        4.5,
    )
    human_sound_support = scale_between(_feature(record, "audio_events__group_ratio_human"), 0.15, 0.85)
    lively_support = scale_between(_feature(record, "emotion__lively__mean"), 0.55, 0.85)
    boring_penalty = scale_between(_feature(record, "emotion__boring__mean"), 0.45, 0.80)
    label_bonus = 0.12 if any(x in labels for x in ("low_vitality", "weak_stay_quality")) else 0.0
    score = (
        0.34 * low_gap
        + 0.16 * (1.0 - people_support)
        + 0.16 * (1.0 - activity_support)
        + 0.14 * (1.0 - stay_support)
        + 0.10 * (1.0 - human_sound_support)
        + 0.05 * (1.0 - lively_support)
        + 0.05 * boring_penalty
        + label_bonus
    )
    if street_type == "transport_movement":
        score *= 0.80
    return clamp01(score)


def compute_eventfulness_gap(record: Mapping[str, Any], street_type: str) -> Tuple[float, str, float]:
    observed = estimate_observed_score(record, "soundscape_eventfulness")
    desired_low, desired_high = desired_target_range(street_type, "soundscape_eventfulness")
    low_gap = max(0.0, desired_low - observed)
    high_gap = max(0.0, observed - desired_high)

    traffic_chaos = scale_between(
        _feature(record, "audio_events__group_ratio_traffic") + _feature(record, "audio_events__group_ratio_mechanical"),
        0.04,
        0.35,
    )
    crowd_pressure = scale_between(_feature(record, "people__total_people__mean"), 12.0, 35.0)
    visual_clutter = scale_between(_feature(record, "emotion__boring__mean") + _feature(record, "visual_major__vehicle__mean"), 0.45, 1.0)

    if high_gap > low_gap and high_gap > 0.0:
        score = clamp01(high_gap / 2.5 + 0.35 * traffic_chaos + 0.15 * crowd_pressure + 0.10 * visual_clutter)
        return score, "decrease", observed
    if low_gap > 0.0:
        liveliness_absence = 1.0 - scale_between(
            _feature(record, "audio_events__group_ratio_human") + _feature(record, "audio_events_dist__music"),
            0.15,
            0.95,
        )
        score = clamp01(low_gap / 2.5 + 0.25 * liveliness_absence)
        return score, "increase", observed
    residual = clamp01(max(traffic_chaos - 0.60, 0.0))
    return residual, "maintain", observed


def classify_soundscape_state(record: Mapping[str, Any], street_type: str) -> Dict[str, Any]:
    profile = record.get("soundscape_json", {}) if isinstance(record.get("soundscape_json"), Mapping) else {}
    dominant_sources = [str(x) for x in as_list(profile.get("dominant_sources")) if str(x).strip()]
    if not dominant_sources:
        ratios = {
            "traffic": _feature(record, "audio_events__group_ratio_traffic"),
            "human": _feature(record, "audio_events__group_ratio_human"),
            "nature": _feature(record, "audio_events__group_ratio_nature"),
            "mechanical": _feature(record, "audio_events__group_ratio_mechanical"),
        }
        dominant_sources = [k for k, v in sorted(ratios.items(), key=lambda item: item[1], reverse=True) if v > 0.05][:2]

    pleasantness = estimate_observed_score(record, "soundscape_pleasantness")
    eventfulness_gap, direction, observed_eventfulness = compute_eventfulness_gap(record, street_type)
    traffic_dominance = clamp01(
        _feature(record, "audio_events__group_ratio_traffic") + _feature(record, "audio_events__group_ratio_mechanical")
    )
    state = "balanced"
    if direction == "decrease" and eventfulness_gap >= 0.25:
        state = "overstimulated"
    elif direction == "increase" and eventfulness_gap >= 0.25:
        state = "understimulated"

    pleasantness_support = "medium"
    if pleasantness <= 3.5 or traffic_dominance >= 0.18:
        pleasantness_support = "low"
    elif pleasantness >= 5.0 and traffic_dominance < 0.08:
        pleasantness_support = "high"

    return {
        "dominant_sources": dominant_sources,
        "traffic_dominance": traffic_dominance,
        "human_presence": clamp01(_feature(record, "audio_events__group_ratio_human")),
        "nature_presence": clamp01(_feature(record, "audio_events__group_ratio_nature")),
        "mechanical_presence": clamp01(_feature(record, "audio_events__group_ratio_mechanical")),
        "pleasantness_score_observed_or_proxy": pleasantness,
        "eventfulness_score_observed_or_proxy": observed_eventfulness,
        "eventfulness_state": state,
        "eventfulness_direction": direction,
        "pleasantness_support_level": pleasantness_support,
        "street_type_expected_eventfulness_range": list(desired_target_range(street_type, "soundscape_eventfulness")),
    }


def adjudication_confidence_flag(record: Mapping[str, Any]) -> str:
    confidence = safe_float(record.get("confidence_score"), -1.0)
    disagreement = bool(record.get("label_disagreement", False))
    if confidence < 0:
        return "unavailable"
    if confidence >= 5.0 and not disagreement:
        return "high"
    if confidence >= 4.0:
        return "medium"
    return "low"


def resolve_interventions(
    *,
    record: Mapping[str, Any],
    street_type: str,
    main_problem_labels: Sequence[str],
    comfort_risk_score: float,
    vitality_deficit_score: float,
    eventfulness_risk_or_gap: float,
    eventfulness_direction: str,
    soundscape_state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply transparent rule families and return machine-readable interventions."""
    flags: Dict[str, int] = {name: 0 for name in INTERVENTION_COLUMNS}
    reasons: List[str] = []

    traffic_dominant = (
        _feature(record, "audio_events__group_ratio_traffic") >= 0.10
        or _feature(record, "audio_events__group_ratio_mechanical") >= 0.03
        or soundscape_state.get("pleasantness_support_level") == "low"
    )
    green_low = _feature(record, "green_view__greenviewindex__mean") < 0.20
    curb_chaos = (
        _feature(record, "visual_major__vehicle__mean") >= 0.07
        or "visual_clutter" in main_problem_labels
        or _feature(record, "emotion__boring__mean") >= 0.62
    )
    activity_low = (
        _feature(record, "people__total_people__mean") < 10.0
        or max(
            _feature(record, "ai_activity__summary_walking_mean_score"),
            _feature(record, "ai_activity__summary_shopping_mean_score"),
            _feature(record, "ai_activity__summary_standing_mean_score"),
        ) < 3.0
    )

    if comfort_risk_score >= 0.45 or any(x in main_problem_labels for x in ("traffic_noise", "pedestrian_discomfort", "visual_clutter")):
        flags["reduce_traffic_exposure"] = 1 if traffic_dominant else flags["reduce_traffic_exposure"]
        flags["strengthen_edge"] = 1
        flags["soften_hardscape_interfaces"] = 1
        flags["reorganize_curbside"] = 1 if curb_chaos else flags["reorganize_curbside"]
        flags["improve_order_and_management"] = 1 if curb_chaos else flags["improve_order_and_management"]
        if green_low:
            flags["add_green_buffer"] = 1
            flags["support_natural_sound_buffer"] = 1
        if street_type == "transport_movement":
            flags["prioritize_calmness"] = 1
        reasons.append("Comfort-oriented rules triggered by weak comfort or traffic/mechanical exposure.")

    if vitality_deficit_score >= 0.45 or any(x in main_problem_labels for x in ("low_vitality", "weak_stay_quality")):
        flags["enhance_stay_node"] = 1
        flags["activate_frontage"] = 1 if street_type in {"commercial_social", "mixed_uncertain"} else flags["activate_frontage"]
        flags["support_micro_public_use"] = 1 if street_type != "transport_movement" else flags["support_micro_public_use"]
        flags["preserve_positive_human_activity"] = 1 if _feature(record, "audio_events__group_ratio_human") >= 0.30 else flags["preserve_positive_human_activity"]
        if street_type == "leisure_cultural":
            flags["support_cultural_liveliness"] = 1
        reasons.append("Vitality-oriented rules triggered by low vitality or weak human-scale activation.")

    if eventfulness_risk_or_gap >= 0.25:
        if eventfulness_direction == "increase":
            flags["preserve_positive_human_activity"] = 1
            flags["enhance_stay_node"] = 1 if street_type != "transport_movement" else flags["enhance_stay_node"]
            flags["activate_frontage"] = 1 if street_type in {"commercial_social", "mixed_uncertain"} else flags["activate_frontage"]
            flags["support_cultural_liveliness"] = 1 if street_type == "leisure_cultural" else flags["support_cultural_liveliness"]
            reasons.append("Eventfulness calibration indicates insufficient liveliness for this street type.")
        elif eventfulness_direction == "decrease":
            flags["avoid_overactivation"] = 1
            flags["prioritize_calmness"] = 1
            flags["reduce_traffic_exposure"] = 1
            flags["reorganize_curbside"] = 1
            reasons.append("Eventfulness calibration indicates overstimulation or traffic-dominated chaos.")

    if soundscape_state.get("pleasantness_support_level") == "low":
        flags["reduce_traffic_exposure"] = 1
        flags["support_natural_sound_buffer"] = 1 if green_low else flags["support_natural_sound_buffer"]
        flags["prioritize_calmness"] = 1 if street_type == "transport_movement" else flags["prioritize_calmness"]
        reasons.append("Soundscape pleasantness support rules triggered by low pleasantness support.")

    if _feature(record, "green_view__greenviewindex__mean") >= 0.24:
        flags["protect_existing_greenery"] = 1
    if street_type == "transport_movement":
        flags["maintain_accessibility"] = 1
    if street_type == "commercial_social" and activity_low:
        flags["activate_frontage"] = 1
    if street_type == "mixed_uncertain" and not any(flags.values()):
        flags["improve_order_and_management"] = 1
        flags["maintain_accessibility"] = 1
        reasons.append("Fallback mixed-context rule applied to maintain order and legibility.")

    allowed_interventions = [INTERVENTION_LABELS[key] for key, value in flags.items() if value]
    if not allowed_interventions:
        allowed_interventions = [
            INTERVENTION_LABELS["improve_order_and_management"],
            INTERVENTION_LABELS["maintain_accessibility"],
        ]
        flags["improve_order_and_management"] = 1
        flags["maintain_accessibility"] = 1
        reasons.append("Fallback intervention set applied because no higher-priority rule fired.")

    return {
        "flags": flags,
        "allowed_interventions": allowed_interventions,
        "rule_reasons": reasons,
        "forbidden_changes": list(DEFAULT_FORBIDDEN_CHANGES),
    }
