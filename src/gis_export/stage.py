


"""GIS export stage entrypoint."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from src.config import GIS_EXPORT_PREFER_WGS84

from .export import build_gis_exports

logger = logging.getLogger("gis_export.stage")


def run_stage(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Export GIS-ready frame/segment/episode tables from existing outputs."""
    options = dict(context.get("options", {}))
    video_dir = str(context["video_dir"])
    prefer_wgs84 = bool(options.get("GIS_EXPORT_PREFER_WGS84", GIS_EXPORT_PREFER_WGS84))
    stage_progress = context.get("stage_progress_task")
    result = build_gis_exports(
        video_dir=video_dir,
        prefer_wgs84=prefer_wgs84,
        progress_callback=(
            lambda completed, total=None, description=None: stage_progress.update(
                completed=completed,
                total=total,
                description=description or "gis_export",
            )
        )
        if stage_progress
        else None,
    )
    logger.info(
        "gis_export stage done | frames=%s segments=%s episodes=%s output=%s",
        result.get("frame_rows"),
        result.get("segment_rows"),
        result.get("problem_episode_rows"),
        result.get("outputs", {}).get("frame_gis_export_csv"),
    )
    return result
