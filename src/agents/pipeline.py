


"""Agents stage entrypoint for Step-6 multi-agent reasoning orchestration."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from .agent_orchestrator import run_agents_stage

logger = logging.getLogger("agents.pipeline")

def run_stage(context: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Run Step-6 agents pipeline on segment/soundscape/fusion artifacts.

    Label-agnostic contract:
    - Must not use any human validation or adjudication files.
    - Uses only segment manifest + soundscape + fusion tables/dictionaries.
    """
    options = dict(context.get("options", {}))
    video_dir = str(context["video_dir"])
    stage_progress = context.get("stage_progress_task")
    result = run_agents_stage(
        video_dir=video_dir,
        options=options,
        progress_callback=(
            lambda completed, total=None, description=None: stage_progress.update(
                completed=completed,
                total=total,
                description=description or "agents",
            )
        )
        if stage_progress
        else None,
    )
    logger.info(
        "agents stage done | segments=%s api_available=%s diagnosis=%s",
        result.get("segment_count"),
        result.get("api_available"),
        result.get("segment_diagnosis_jsonl"),
    )
    return result
