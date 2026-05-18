


"""Template-based episode summaries built from structured evidence."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("deliverable.issue_summarizer")


TITLE_MAP = {
    "crowded_and_noise_dominant": "人群密集且听觉压力偏高的街段",
    "low_greenery_with_high_mechanical_noise": "低绿量且缺少声景缓冲的街段",
    "active_but_acoustically_harsh": "活跃但声景偏粗糙的街段",
    "visually_tolerable_but_acoustically_stressful": "视觉尚可但声景偏紧张的街段",
}

THEME_MAP = {
    "comfort_buffering": "舒适性缓冲",
    "vitality_activation": "活力激活",
    "eventfulness_calibration": "事件性校准",
    "mixed_rebalancing": "综合再平衡",
}


def _severity_levels(priority_scores: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([x for x in priority_scores if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return 0.55, 0.62
    return float(np.quantile(arr, 0.50)), float(np.quantile(arr, 0.80))


def _severity_label(score: float, q50: float, q80: float) -> str:
    if not np.isfinite(score):
        return "medium"
    if score >= q80:
        return "high"
    if score >= q50:
        return "medium"
    return "moderate"


def _confidence_label(band: str) -> str:
    mapping = {"high": "high", "medium": "medium", "low": "low"}
    return mapping.get(str(band), "medium")


def _episode_title(row: pd.Series) -> str:
    fusion_tags = row.get("fusion_problem_tags", []) or []
    if fusion_tags:
        tag = str(fusion_tags[0])
        if tag in TITLE_MAP:
            return TITLE_MAP[tag]
    visual_tags = row.get("visual_problem_tags", []) or []
    sound_tags = row.get("soundscape_problem_tags", []) or []
    if "low_green_view" in visual_tags:
        return "绿量缓冲不足的街段"
    if "crowding" in visual_tags:
        return "使用压力偏高的街段"
    if "traffic_mechanical_dominant" in sound_tags:
        return "交通/机械声暴露偏高的街段"
    return f"{THEME_MAP.get(str(row.get('intervention_theme', 'mixed_rebalancing')), '综合问题街段')}"


def _why_problematic(row: pd.Series) -> str:
    fused = str(row.get("fusion_problem_summary", "")).strip()
    validation = str(row.get("validation_label_summary", "")).strip()
    if validation:
        return f"{fused} 同时，人工裁决标签显示 {validation}。"
    return fused


def _evidence_bullets(row: pd.Series) -> List[str]:
    bullets: List[str] = []
    for item in row.get("soundscape_evidence_features", []) or []:
        bullets.append(f"声景证据: {item}")
    for item in row.get("visual_evidence_features", []) or []:
        bullets.append(f"视觉证据: {item}")
    if str(row.get("validation_label_summary", "")).strip():
        bullets.append(f"人工裁决: {row['validation_label_summary']}")
    if str(row.get("proof_claim_snapshot", "")).strip():
        bullets.append(f"Step 7.5 / proof: {row['proof_claim_snapshot']}")
    if str(row.get("agent_diagnosis_summary", "")).strip():
        bullets.append(f"诊断摘要: {row['agent_diagnosis_summary']}")
    dedup: List[str] = []
    for bullet in bullets:
        if bullet not in dedup:
            dedup.append(bullet)
        if len(dedup) >= 6:
            break
    return dedup[:6]


def build_episode_summaries(evidence_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if evidence_df.empty:
        empty = pd.DataFrame()
        return empty, []

    q50, q80 = _severity_levels(pd.to_numeric(evidence_df["priority_score"], errors="coerce").tolist())
    rows: List[Dict[str, Any]] = []
    jsonl_rows: List[Dict[str, Any]] = []

    for _, row in evidence_df.iterrows():
        episode_title = _episode_title(row)
        one_sentence = (
            f"{episode_title}：{row['visual_problem_summary']} {row['soundscape_problem_summary']} "
            f"建议以“{THEME_MAP.get(str(row.get('intervention_theme', 'mixed_rebalancing')), str(row.get('intervention_theme', 'mixed_rebalancing')))}”为主线进行保守改造。"
        )
        summary_row = {
            "episode_id": str(row["episode_id"]),
            "episode_title": episode_title,
            "one_sentence_summary": one_sentence,
            "soundscape_problem": str(row.get("soundscape_problem_summary", "")),
            "visual_problem": str(row.get("visual_problem_summary", "")),
            "fused_problem": str(row.get("fusion_problem_summary", "")),
            "why_it_is_problematic": _why_problematic(row),
            "suggested_intervention_theme": str(row.get("intervention_theme", "")),
            "evidence_bullets": _evidence_bullets(row),
            "severity_level": _severity_label(float(row.get("priority_score", np.nan)), q50, q80),
            "confidence_level": _confidence_label(str(row.get("diagnosis_confidence_band", "medium"))),
        }
        rows.append(summary_row)
        jsonl_rows.append(dict(summary_row))

    summary_df = pd.DataFrame(rows)
    logger.info("deliverable summaries built | episodes=%s", len(summary_df))
    return summary_df, jsonl_rows
