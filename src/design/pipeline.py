


"""Design stage entrypoint for Step-8 design mapping."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from src.config import STEP8_TOP_N

from .step8_runner import run_step8_design_mapping

logger = logging.getLogger("design.pipeline")

def run_stage(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Run Step-8 design mapping using existing diagnostics and Step-7.5 evidence."""
    options = dict(context.get("options", {}))
    video_dir = str(context["video_dir"])
    top_n = options.get("STEP8_TOP_N", STEP8_TOP_N)
    stage_progress = context.get("stage_progress_task")
    result = run_step8_design_mapping(
        video_dir=video_dir,
        step8_outdir=None,
        top_n=int(top_n) if top_n is not None else None,
        smoke_test=False,
        progress_callback=(
            lambda completed, total=None, description=None: stage_progress.update(
                completed=completed,
                total=total,
                description=description or "design",
            )
        )
        if stage_progress
        else None,
    )
    logger.info(
        "design stage done | ranked=%s selected=%s plan=%s",
        result.get("total_segments_ranked"),
        result.get("selected_segments_for_design_plan"),
        result.get("design_plan_jsonl"),
    )
    return result
