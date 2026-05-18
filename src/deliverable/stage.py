


"""Deliverable stage entrypoint for problem-episode packaging."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from .deliverable_runner import run_deliverable_layer

logger = logging.getLogger("deliverable.stage")


def run_stage(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Build deliverable-layer episode artifacts from existing Step-8 outputs."""
    video_dir = str(context["video_dir"])
    stage_progress = context.get("stage_progress_task")
    result = run_deliverable_layer(
        video_dir=video_dir,
        progress_callback=(
            lambda completed, total=None, description=None: stage_progress.update(
                completed=completed,
                total=total,
                description=description or "deliverable",
            )
        )
        if stage_progress
        else None,
    )
    logger.info(
        "deliverable stage done | episodes=%s summary=%s final_table=%s",
        result.get("problem_episodes_csv"),
        result.get("problem_episode_summary_csv"),
        result.get("final_problem_segments_table_csv"),
    )
    return result
