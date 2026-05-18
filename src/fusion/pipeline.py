


"""Fusion stage entrypoint for segment-level multimodal feature warehouse."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping

from src.config import (
    BUILD_MODEL_FEATURE_TABLE,
    MODEL_DROP_HIGH_MISSING,
    MODEL_EVENT_VOCAB_TOP_N,
    MODEL_HIGH_MISSING_THRESHOLD,
    MODEL_TOPK_EVENT_VOCAB_TOP_N,
    SOUNDSCAPE_PANNS_EXPORT_DIMS,
)

from .feature_warehouse import build_segment_feature_warehouse
from .model_feature_builder import build_model_feature_table

logger = logging.getLogger("fusion.pipeline")


def run_stage(context: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Build one-row-per-segment multimodal feature table.

    This stage reuses existing outputs and does not retrigger legacy frame
    extraction / segmentation / reprojection pipeline.
    """
    options = dict(context.get("options", {}))
    video_dir = str(context["video_dir"])
    stage_progress = context.get("stage_progress_task")
    if stage_progress:
        stage_progress.update(completed=0, total=4, description="fusion | prepare")

    vdir = Path(video_dir)
    raw_table_path = vdir / "fusion" / "segment_feature_table.csv"
    raw_dict_path = vdir / "fusion" / "feature_dictionary.json"
    raw_report_path = vdir / "fusion" / "feature_quality_report.json"

    reused_raw_table = bool(
        raw_table_path.is_file() and raw_dict_path.is_file() and raw_report_path.is_file()
    )
    if reused_raw_table:
        if stage_progress:
            stage_progress.update(completed=1, total=4, description="fusion | reuse raw warehouse")
        raw_result: Dict[str, Any] = {
            "segment_feature_table_csv": raw_table_path.as_posix(),
            "feature_dictionary_json": raw_dict_path.as_posix(),
            "feature_quality_report_json": raw_report_path.as_posix(),
            "reused_existing_raw_table": True,
        }
    else:
        raw_result = build_segment_feature_warehouse(
            video_dir=video_dir,
            panns_export_dims=int(
                options.get("SOUNDSCAPE_PANNS_EXPORT_DIMS", SOUNDSCAPE_PANNS_EXPORT_DIMS)
            ),
        )
        raw_result["reused_existing_raw_table"] = False
        if stage_progress:
            stage_progress.update(completed=2, total=4, description="fusion | raw warehouse")

    model_enabled = bool(options.get("BUILD_MODEL_FEATURE_TABLE", BUILD_MODEL_FEATURE_TABLE))
    model_result: Dict[str, Any] = {"model_feature_table_enabled": model_enabled}
    if model_enabled:
        if stage_progress and reused_raw_table:
            stage_progress.update(completed=2, total=4, description="fusion | model table")
        model_result = build_model_feature_table(
            video_dir=video_dir,
            event_vocab_top_n=int(options.get("MODEL_EVENT_VOCAB_TOP_N", MODEL_EVENT_VOCAB_TOP_N)),
            topk_vocab_top_n=int(
                options.get("MODEL_TOPK_EVENT_VOCAB_TOP_N", MODEL_TOPK_EVENT_VOCAB_TOP_N)
            ),
            drop_high_missing=bool(options.get("MODEL_DROP_HIGH_MISSING", MODEL_DROP_HIGH_MISSING)),
            high_missing_threshold=float(
                options.get("MODEL_HIGH_MISSING_THRESHOLD", MODEL_HIGH_MISSING_THRESHOLD)
            ),
        )
        model_result["model_feature_table_enabled"] = True
        if stage_progress:
            stage_progress.update(completed=4, total=4, description="fusion | model table")
    else:
        logger.info("fusion model feature table disabled by config")
        if stage_progress:
            stage_progress.update(completed=4, total=4, description="fusion | model table disabled")

    logger.info(
        "fusion stage done | raw_reused=%s raw_csv=%s model_csv=%s model_enabled=%s",
        reused_raw_table,
        raw_result.get("segment_feature_table_csv"),
        model_result.get("model_feature_table_csv"),
        model_enabled,
    )
    return {
        **raw_result,
        **model_result,
        "raw_reused": reused_raw_table,
    }
