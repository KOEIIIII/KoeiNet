


"""Geo sync stage entrypoint for optional multimodal post-analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.config import (
    GEO_SYNC_EXPORT_WGS84,
    GEO_SYNC_FILENAME_TZ_OFFSET_HOURS,
    GEO_SYNC_ALIGN_TO_ANALYSIS_FRAMES,
    GEO_SYNC_FRAME_STEP,
    GEO_SYNC_GPS_CSV,
    GEO_SYNC_MAX_GAP_WARNING_SEC,
    GEO_SYNC_SIDECAR_PATH,
    GEO_SYNC_TIME_OFFSET_SECONDS,
    GEO_SYNC_USE_EXISTING_SEGMENTS,
    SEGMENT_OVERLAP,
    SEGMENT_SECONDS,
)

from .pipeline import run_geo_sync

logger = logging.getLogger("geo_sync.stage")


def _normalize_optional_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def run_stage(context: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Align one video against the GPS track and export frame/segment geo metadata.

    The stage is optional and intentionally does not require the segment stage:
    - if `segments/segment_manifest.csv` exists and reuse is enabled, it is reused
    - otherwise virtual segments are generated from SEGMENT_SECONDS / SEGMENT_OVERLAP
    """
    options = dict(context.get("options", {}))
    video_path = str(context["video_path"])
    video_dir = Path(str(context["video_dir"]))
    output_root = str(context["output_dir"])

    use_existing_segments = bool(
        options.get("GEO_SYNC_USE_EXISTING_SEGMENTS", GEO_SYNC_USE_EXISTING_SEGMENTS)
    )
    segment_manifest_path = None
    if use_existing_segments:
        candidate = video_dir / "segments" / "segment_manifest.csv"
        if candidate.is_file():
            segment_manifest_path = candidate.as_posix()
    stage_progress = context.get("stage_progress_task")

    result = run_geo_sync(
        video_path=video_path,
        gps_csv_path=str(options.get("GEO_SYNC_GPS_CSV", GEO_SYNC_GPS_CSV)),
        output_root=output_root,
        frame_step=int(options.get("GEO_SYNC_FRAME_STEP", GEO_SYNC_FRAME_STEP)),
        frames_dir=(video_dir / "frames").as_posix(),
        analysis_frame_skip=int(context.get("frame_skip", 0) or 0),
        sidecar_path=_normalize_optional_path(options.get("GEO_SYNC_SIDECAR_PATH", GEO_SYNC_SIDECAR_PATH)),
        time_offset_seconds=float(
            options.get("GEO_SYNC_TIME_OFFSET_SECONDS", GEO_SYNC_TIME_OFFSET_SECONDS)
        ),
        filename_tz_offset_hours=float(
            options.get("GEO_SYNC_FILENAME_TZ_OFFSET_HOURS", GEO_SYNC_FILENAME_TZ_OFFSET_HOURS)
        ),
        segment_manifest_path=segment_manifest_path,
        segment_seconds=float(options.get("SEGMENT_SECONDS", SEGMENT_SECONDS)),
        segment_overlap=float(options.get("SEGMENT_OVERLAP", SEGMENT_OVERLAP)),
        display=False,
        save_preview_count=0,
        export_wgs84=bool(options.get("GEO_SYNC_EXPORT_WGS84", GEO_SYNC_EXPORT_WGS84)),
        align_to_existing_frames=bool(
            options.get("GEO_SYNC_ALIGN_TO_ANALYSIS_FRAMES", GEO_SYNC_ALIGN_TO_ANALYSIS_FRAMES)
        ),
        max_gap_warning_sec=float(
            options.get("GEO_SYNC_MAX_GAP_WARNING_SEC", GEO_SYNC_MAX_GAP_WARNING_SEC)
        ),
        use_existing_segments=use_existing_segments,
        progress_callback=(
            lambda completed, total=None, description=None: stage_progress.update(
                completed=completed,
                total=total,
                description=description or "geo_sync",
            )
        )
        if stage_progress
        else None,
    )

    logger.info(
        "geo_sync stage done | segment_source=%s segment_rows=%s frame_rows=%s summary=%s",
        result.get("result_summary", {}).get("segment_source"),
        result.get("result_summary", {}).get("segment_rows"),
        result.get("result_summary", {}).get("sampled_rows"),
        result.get("outputs", {}).get("summary_json"),
    )
    return result
