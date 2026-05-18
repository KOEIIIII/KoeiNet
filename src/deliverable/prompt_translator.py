


"""Controlled prompt translation for episode-level design prompts."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .export_utils import sanitize_value

logger = logging.getLogger("deliverable.prompt_translator")

try:
    from src.config import ZHIPU_AGENT_MODEL
except Exception:
    ZHIPU_AGENT_MODEL = "glm-5"


def _theme_label(theme: str) -> str:
    mapping = {
        "comfort_buffering": "pedestrian comfort buffering",
        "vitality_activation": "human-scale vitality activation",
        "eventfulness_calibration": "eventfulness calibration",
        "mixed_rebalancing": "mixed rebalancing",
    }
    return mapping.get(str(theme), str(theme).replace("_", " "))


def _structured_input(evidence_row: pd.Series, summary_row: pd.Series) -> Dict[str, Any]:
    return {
        "scene_context": str(evidence_row.get("agent_profile_summary", "")),
        "location_type": str(evidence_row.get("street_type", "")),
        "episode_time_range": f"{float(evidence_row.get('start_time_sec', 0.0)):.1f}s–{float(evidence_row.get('end_time_sec', 0.0)):.1f}s",
        "soundscape_problem": str(summary_row.get("soundscape_problem", "")),
        "visual_problem": str(summary_row.get("visual_problem", "")),
        "fused_problem": str(summary_row.get("fused_problem", "")),
        "key_evidence": list(summary_row.get("evidence_bullets", []) or []),
        "intervention_theme": str(summary_row.get("suggested_intervention_theme", "")),
        "intervention_actions": list(evidence_row.get("intervention_actions", []) or []),
        "must_keep_elements": list(evidence_row.get("must_keep_elements", []) or []),
        "must_avoid_elements": list(evidence_row.get("must_avoid_elements", []) or []),
        "desired_rendering_style": (
            "realistic street-design planning visualization, same viewpoint, same road geometry, "
            "same building massing, feasible local interventions only"
        ),
        "proof_note": str(evidence_row.get("proof_claim_snapshot", "")),
    }


def _deterministic_prompt(payload: Mapping[str, Any]) -> Tuple[str, str, str]:
    actions = "; ".join([str(x) for x in payload.get("intervention_actions", [])[:8]]) or "small-scale edge, planting, frontage, seating, and management improvements"
    keep_items = "; ".join([str(x) for x in payload.get("must_keep_elements", [])[:5]])
    avoid_items = "; ".join([str(x) for x in payload.get("must_avoid_elements", [])[:8]])
    evidence = "; ".join([str(x) for x in payload.get("key_evidence", [])[:4]])
    edit_prompt = (
        "Conservatively edit this real streetscape for planning visualization. "
        f"Location type: {payload.get('location_type', 'urban street')}. "
        f"Episode time range: {payload.get('episode_time_range', '')}. "
        f"Soundscape problem: {payload.get('soundscape_problem', '')}. "
        f"Visual problem: {payload.get('visual_problem', '')}. "
        f"Fused problem: {payload.get('fused_problem', '')}. "
        f"Primary intervention theme: {_theme_label(str(payload.get('intervention_theme', 'mixed_rebalancing')))}. "
        f"Apply only feasible local interventions such as: {actions}. "
        f"Preserve the original camera viewpoint, road alignment, lane structure, and building massing. "
        f"Retain these existing elements when possible: {keep_items or 'keep valuable existing pedestrian activity, greenery, and accessibility cues'}. "
        f"Key evidence to respect: {evidence}. "
        f"Rendering style: {payload.get('desired_rendering_style', '')}."
    )
    negative_prompt = (
        f"Avoid: {avoid_items or 'new buildings, new road alignment, impossible structures, fantasy lighting, staged crowds, unreal sound sources'}. "
        "Do not change the viewpoint, do not invent landmarks, do not widen the street unrealistically, "
        "do not create theme-park style beautification, and do not remove all existing street life."
    )
    short_caption = (
        f"{payload.get('episode_time_range', '')} | "
        f"{payload.get('fused_problem', payload.get('visual_problem', '问题街段'))}"
    )
    return edit_prompt.strip(), negative_prompt.strip(), short_caption.strip()


def _try_glm_refine(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        from src.agents.zhipu_client import ZhipuAgentClient
    except Exception as exc:
        logger.warning("deliverable glm import failed: %s", exc)
        return None

    client = ZhipuAgentClient.from_apikey_env()
    if not client.available:
        logger.info("deliverable glm unavailable: %s", client.unavailable_reason)
        return None

    model_name = os.getenv("DELIVERABLE_GLM_MODEL") or os.getenv("ZHIPU_AGENT_MODEL") or ZHIPU_AGENT_MODEL or "glm-5"
    schema = {
        "edit_prompt": "string",
        "negative_prompt": "string",
        "short_caption": "string",
    }
    system_prompt = (
        "你是城市设计可视化提示词转写器。"
        "你的任务不是重新判断问题，而是将给定的结构化字段翻译成更清晰、更受约束的设计编辑提示词。"
        "必须保守、具体、可实现，不得改变场景几何、镜头、道路走向与建筑体量。"
    )
    result = client.request_json(
        model=str(model_name),
        system_prompt=system_prompt,
        user_payload=payload,
        output_schema=schema,
        max_retries=1,
        temperature=0.1,
    )
    if not bool(result.get("ok", False)):
        logger.info("deliverable glm request failed: %s", result.get("error"))
        return None
    obj = result.get("json")
    return obj if isinstance(obj, dict) else None


def build_episode_prompts(
    evidence_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    use_glm: bool,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    merged = evidence_df.merge(summary_df, on="episode_id", how="left", suffixes=("", "_summary"))
    rows: List[Dict[str, Any]] = []
    jsonl_rows: List[Dict[str, Any]] = []
    prompt_mode = "template"
    for _, row in merged.iterrows():
        payload = _structured_input(row, row)
        edit_prompt, negative_prompt, short_caption = _deterministic_prompt(payload)
        mode = "template"
        if use_glm:
            refined = _try_glm_refine(payload)
            if refined:
                edit_prompt = str(refined.get("edit_prompt", edit_prompt)).strip() or edit_prompt
                negative_prompt = str(refined.get("negative_prompt", negative_prompt)).strip() or negative_prompt
                short_caption = str(refined.get("short_caption", short_caption)).strip() or short_caption
                mode = "glm_refined"
                prompt_mode = "glm_refined"
        prompt_row = {
            "episode_id": str(row["episode_id"]),
            "structured_prompt_input_json": sanitize_value(payload),
            "edit_prompt": edit_prompt,
            "negative_prompt": negative_prompt,
            "short_caption": short_caption,
            "prompt_mode": mode,
        }
        rows.append(prompt_row)
        jsonl_rows.append(dict(prompt_row))
    prompt_df = pd.DataFrame(rows)
    logger.info("deliverable prompts built | episodes=%s mode=%s", len(prompt_df), prompt_mode)
    return prompt_df, jsonl_rows, prompt_mode
