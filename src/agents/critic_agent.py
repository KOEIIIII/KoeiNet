


"""Critic agent for consistency and missing-evidence QA checks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .base_agent import BaseAgent


CRITIC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "segment_id",
        "consistency_check",
        "missing_evidence_check",
        "confidence_score",
    ],
    "properties": {
        "segment_id": {"type": "integer"},
        "consistency_check": {"type": "object"},
        "missing_evidence_check": {"type": "object"},
        "confidence_score": {"type": "number"},
    },
}


class CriticAgent(BaseAgent):
    agent_name = "critic_agent"
    output_schema = CRITIC_SCHEMA
    system_prompt = (
        "你是质控批评代理。"
        "请检查 profile/soundscape/diagnosis 的一致性与证据完备性，输出JSON。"
    )

    def fallback_output(self, context_payload: Mapping[str, Any], reason: str) -> Dict[str, Any]:
        sid = int(context_payload.get("segment_id", -1))
        evidence = dict(context_payload.get("evidence", {}))
        visual_count = len(evidence.get("visual_major_top", []) or [])
        audio_count = len(evidence.get("audio_top_events", []) or [])
        missing_items = []
        if visual_count == 0:
            missing_items.append("visual_major_top")
        if audio_count == 0:
            missing_items.append("audio_top_events")
        return {
            "segment_id": sid,
            "consistency_check": {
                "is_consistent": True,
                "issues": [f"fallback_due_to_{reason}"],
            },
            "missing_evidence_check": {
                "has_missing_evidence": bool(missing_items),
                "missing_items": missing_items,
            },
            "confidence_score": float(4.0 if missing_items else 5.0),
        }

