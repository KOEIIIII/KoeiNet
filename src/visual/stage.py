


"""Visual segment summary stage entrypoint."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from .segment_features import build_segment_visual_features

logger = logging.getLogger("visual.stage")


def run_stage(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Build manifest-aligned visual segment features from existing outputs."""
    video_dir = str(context["video_dir"])
    stage_progress = context.get("stage_progress_task")
    result = build_segment_visual_features(
        video_dir=video_dir,
        progress_callback=(
            lambda completed, total=None, description=None: stage_progress.update(
                completed=completed,
                total=total,
                description=description or "visual",
            )
        )
        if stage_progress
        else None,
    )
    logger.info(
        "visual stage done | segment_rows=%s frame_feature_rows=%s output=%s",
        result.get("segment_rows"),
        result.get("frame_feature_rows"),
        result.get("output_csv"),
    )
    return result
