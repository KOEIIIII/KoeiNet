


"""Web sync export stage entrypoint."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from src.config import WEB_SYNC_PREFER_WGS84

from .export import build_sync_map_data

logger = logging.getLogger("web_sync.stage")


def run_stage(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Export video-chart-map sync JSON for the dashboard."""
    options = dict(context.get("options", {}))
    video_dir = str(context["video_dir"])
    prefer_wgs84 = bool(options.get("WEB_SYNC_PREFER_WGS84", WEB_SYNC_PREFER_WGS84))
    stage_progress = context.get("stage_progress_task")
    result = build_sync_map_data(
        video_dir=video_dir,
        prefer_wgs84=prefer_wgs84,
        progress_callback=(
            lambda completed, total=None, description=None: stage_progress.update(
                completed=completed,
                total=total,
                description=description or "web_sync",
            )
        )
        if stage_progress
        else None,
    )
    logger.info(
        "web_sync stage done | frames=%s segments=%s problems=%s episodes=%s output=%s",
        result.get("frame_count"),
        result.get("segment_count"),
        result.get("problem_segment_count"),
        result.get("problem_episode_count"),
        result.get("output_json"),
    )
    return result
