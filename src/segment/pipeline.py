


"""Segment stage entrypoint for future multimodal segmentation logic."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from .segment_builder import build_segment_manifest

logger = logging.getLogger("segment.pipeline")

def run_stage(context: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Build segment-level synchronized manifest from existing artifacts.

    This stage is intentionally post-analysis-only: it reuses `frames/` and
    optional `audio_events/` alignment outputs under the existing video folder.
    """
    options = dict(context.get("options", {}))
    video_dir = str(context["video_dir"])
    video_path_hint = str(context.get("video_path") or "")
    frame_skip = int(context.get("frame_skip", 30))
    segment_seconds = float(options.get("SEGMENT_SECONDS", 5.0))
    segment_overlap = float(options.get("SEGMENT_OVERLAP", 2.5))
    stage_progress = context.get("stage_progress_task")

    result = build_segment_manifest(
        video_dir=video_dir,
        frame_skip=frame_skip,
        segment_seconds=segment_seconds,
        overlap_seconds=segment_overlap,
        video_path_hint=video_path_hint if video_path_hint else None,
        progress_callback=(
            lambda completed, total=None, description=None: stage_progress.update(
                completed=completed,
                total=total,
                description=description or "segment",
            )
        )
        if stage_progress
        else None,
    )

    qa = result.get("qa_summary", {})
    logger.info(
        "segment stage done | total=%s mean_frames=%.2f missing_audio=%s",
        qa.get("total_segments"),
        float(qa.get("mean_frames_per_segment", 0.0)),
        qa.get("missing_audio_count"),
    )
    return result
