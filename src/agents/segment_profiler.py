


"""Segment profiler agent: concise visual+audio factual profiling."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .base_agent import BaseAgent


SEGMENT_PROFILE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["segment_id", "visual_facts", "audio_facts", "concise_summary"],
    "properties": {
        "segment_id": {"type": "integer"},
        "visual_facts": {"type": "array", "items": {"type": "string"}},
        "audio_facts": {"type": "array", "items": {"type": "string"}},
        "concise_summary": {"type": "string"},
    },
}


class SegmentProfilerAgent(BaseAgent):
    agent_name = "segment_profiler"
    output_schema = SEGMENT_PROFILE_SCHEMA
    system_prompt = (
        "你是城市空间多模态剖面助手。"
        "请根据输入证据输出严格JSON，不要输出解释文本。"
        "内容必须是可核验事实，不要引入输入中不存在的信息。"
    )

    def fallback_output(self, context_payload: Mapping[str, Any], reason: str) -> Dict[str, Any]:
        sid = int(context_payload.get("segment_id", -1))
        evidence = dict(context_payload.get("evidence", {}))
        visual_top = evidence.get("visual_major_top", []) or evidence.get("visual_semantic_top", [])
        audio_top = evidence.get("audio_top_events", [])
        visual_facts = []
        audio_facts = []
        for item in visual_top[:3]:
            if isinstance(item, dict):
                visual_facts.append(f"{item.get('feature')}={item.get('value')}")
        for item in audio_top[:3]:
            audio_facts.append(str(item))
        if not visual_facts:
            visual_facts = ["visual_evidence_limited"]
        if not audio_facts:
            audio_facts = ["audio_evidence_limited"]
        return {
            "segment_id": sid,
            "visual_facts": visual_facts,
            "audio_facts": audio_facts,
            "concise_summary": f"fallback_profile_due_to_{reason}",
        }

