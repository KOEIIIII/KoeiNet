


"""Soundscape interpreter agent for segment-level acoustic reasoning."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .base_agent import BaseAgent


SOUNDSCAPE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "segment_id",
        "dominant_sources",
        "pleasantness_reasoning",
        "eventfulness_reasoning",
        "acoustic_risk_tags",
    ],
    "properties": {
        "segment_id": {"type": "integer"},
        "dominant_sources": {"type": "array", "items": {"type": "string"}},
        "pleasantness_reasoning": {"type": "string"},
        "eventfulness_reasoning": {"type": "string"},
        "acoustic_risk_tags": {"type": "array", "items": {"type": "string"}},
    },
}


class SoundscapeInterpreterAgent(BaseAgent):
    agent_name = "soundscape_interpreter"
    output_schema = SOUNDSCAPE_SCHEMA
    system_prompt = (
        "你是声景解释代理。"
        "请基于输入的音频事件与信号特征，输出严格JSON。"
        "禁止编造不存在的声源或证据。"
    )

    def fallback_output(self, context_payload: Mapping[str, Any], reason: str) -> Dict[str, Any]:
        sid = int(context_payload.get("segment_id", -1))
        evidence = dict(context_payload.get("evidence", {}))
        top_events = evidence.get("audio_top_events", [])
        group_ratios = dict(evidence.get("audio_group_ratios", {}))

        dom = [str(x) for x in top_events[:3] if str(x).strip()]
        if not dom:
            dom = [k for k, v in sorted(group_ratios.items(), key=lambda kv: float(kv[1]), reverse=True)[:2]]
        if not dom:
            dom = ["unknown"]

        risk_tags = []
        if float(group_ratios.get("traffic", 0.0)) > 0.35:
            risk_tags.append("traffic_noise")
        if float(group_ratios.get("mechanical", 0.0)) > 0.25:
            risk_tags.append("mechanical_noise")
        if not risk_tags:
            risk_tags.append("uncertain")

        return {
            "segment_id": sid,
            "dominant_sources": dom,
            "pleasantness_reasoning": f"fallback_due_to_{reason}",
            "eventfulness_reasoning": f"event_count_hint={len(top_events)}",
            "acoustic_risk_tags": risk_tags,
        }

