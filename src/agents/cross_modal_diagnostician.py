


"""Cross-modal diagnostician agent for segment-level problem diagnosis."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .base_agent import BaseAgent


DIAGNOSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "segment_id",
        "problem_labels",
        "severity_scores",
        "evidence_visual",
        "evidence_audio",
        "cross_modal_reason",
        "priority_actions",
    ],
    "properties": {
        "segment_id": {"type": "integer"},
        "problem_labels": {"type": "array", "items": {"type": "string"}},
        "severity_scores": {"type": "object"},
        "evidence_visual": {"type": "array", "items": {"type": "string"}},
        "evidence_audio": {"type": "array", "items": {"type": "string"}},
        "cross_modal_reason": {"type": "string"},
        "priority_actions": {"type": "array", "items": {"type": "string"}},
    },
}


class CrossModalDiagnosticianAgent(BaseAgent):
    agent_name = "cross_modal_diagnostician"
    output_schema = DIAGNOSIS_SCHEMA
    system_prompt = (
        "你是跨模态诊断代理。"
        "输入包含视觉与音频证据，请输出结构化诊断JSON。"
        "请给出可追溯证据字段，避免空泛描述。"
    )

    def fallback_output(self, context_payload: Mapping[str, Any], reason: str) -> Dict[str, Any]:
        sid = int(context_payload.get("segment_id", -1))
        evidence = dict(context_payload.get("evidence", {}))
        green = float(evidence.get("green_view_mean", 0.0) or 0.0)
        traffic_ratio = float(dict(evidence.get("audio_group_ratios", {})).get("traffic", 0.0) or 0.0)
        labels = []
        if traffic_ratio > 0.35:
            labels.append("traffic_noise")
        if green < 0.15:
            labels.append("low_greenery")
        if not labels:
            labels = ["mixed_or_unclear"]
        severity = {
            "overall_problem_severity": float(min(7.0, max(1.0, 1.0 + 6.0 * max(traffic_ratio, 1.0 - green)))),
            "soundscape_pressure": float(min(7.0, max(1.0, 1.0 + 6.0 * traffic_ratio))),
        }
        visual_e = [f"{x.get('feature')}={x.get('value')}" for x in evidence.get("visual_major_top", [])[:3] if isinstance(x, dict)]
        audio_e = [str(x) for x in evidence.get("audio_top_events", [])[:3]]
        return {
            "segment_id": sid,
            "problem_labels": labels,
            "severity_scores": severity,
            "evidence_visual": visual_e or ["visual_evidence_limited"],
            "evidence_audio": audio_e or ["audio_evidence_limited"],
            "cross_modal_reason": f"fallback_due_to_{reason}",
            "priority_actions": ["collect_more_evidence"],
        }

