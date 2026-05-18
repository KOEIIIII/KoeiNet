


"""Lightweight registry for optional multimodal post-analysis stages."""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .integrity import stage_checks_map, stage_validity_map, validate_stage_outputs
from .progress import UnifiedProgressManager, create_progress_manager
from .readiness import (
    discover_existing_artifacts,
    find_missing_dependencies_for_stage,
    readiness_report_lines,
)
from .run_manifest import build_config_snapshot, write_run_manifest
from .stage_contracts import stage_contract_as_posix

logger = logging.getLogger("common.pipeline_registry")
NORMALIZED_STATES = ("disabled", "skipped_existing", "ran", "failed")

PIPELINE_OPTION_KEYS = (
    "ENABLE_SEGMENT_PIPELINE",
    "SEGMENT_SECONDS",
    "SEGMENT_OVERLAP",
    "ENABLE_VISUAL_SEGMENT_SUMMARY",
    "ENABLE_GEO_SYNC",
    "GEO_SYNC_GPS_CSV",
    "GEO_SYNC_TIME_OFFSET_SECONDS",
    "GEO_SYNC_EXPORT_WGS84",
    "GEO_SYNC_MAX_GAP_WARNING_SEC",
    "GEO_SYNC_USE_EXISTING_SEGMENTS",
    "GEO_SYNC_SIDECAR_PATH",
    "GEO_SYNC_FILENAME_TZ_OFFSET_HOURS",
    "GEO_SYNC_ALIGN_TO_ANALYSIS_FRAMES",
    "GEO_SYNC_FRAME_STEP",
    "ENABLE_WEB_SYNC_EXPORT",
    "WEB_SYNC_PREFER_WGS84",
    "ENABLE_GIS_EXPORT",
    "GIS_EXPORT_PREFER_WGS84",
    "ENABLE_SOUNDSCAPE",
    "ENABLE_FUSION",
    "ENABLE_AGENTS",
    "ENABLE_DESIGN",
    "ENABLE_DELIVERABLE",
    "EXPORT_DEBUG_JSON",
    "PANNS_DIR",
    "PANNS_CHECKPOINT_PATH",
    "PANNS_LABELS_PATH",
    "PANNS_FORCE_LOCAL_RESOURCES",
    "BUILD_MODEL_FEATURE_TABLE",
    "MODEL_EVENT_VOCAB_TOP_N",
    "MODEL_TOPK_EVENT_VOCAB_TOP_N",
    "MODEL_DROP_HIGH_MISSING",
    "MODEL_HIGH_MISSING_THRESHOLD",
    "STEP8_TOP_N",
    "ZHIPU_AGENT_MODEL",
    "ZHIPU_VISION_QA_MODEL",
    "AGENT_MAX_RETRIES",
    "AGENT_CACHE_ENABLED",
    "AGENT_DISABLE_LLM",

    "POST_ONLY",
    "RESUME_MISSING_ONLY",
    "FROM_EXISTING_OUTPUT",
)


@dataclass(frozen=True)
class StageSpec:
    """Description of a pluggable multimodal stage."""

    name: str
    enable_flag: str
    module_path: str
    entrypoint: str = "run_stage"


STAGE_SPECS: List[StageSpec] = [
    StageSpec("segment", "ENABLE_SEGMENT_PIPELINE", "src.segment.pipeline"),
    StageSpec("visual", "ENABLE_VISUAL_SEGMENT_SUMMARY", "src.visual.stage"),
    StageSpec("geo_sync", "ENABLE_GEO_SYNC", "src.geo_sync.stage"),
    StageSpec("soundscape", "ENABLE_SOUNDSCAPE", "src.soundscape.pipeline"),
    StageSpec("fusion", "ENABLE_FUSION", "src.fusion.pipeline"),
    StageSpec("agents", "ENABLE_AGENTS", "src.agents.pipeline"),
    StageSpec("design", "ENABLE_DESIGN", "src.design.pipeline"),
    StageSpec("deliverable", "ENABLE_DELIVERABLE", "src.deliverable.stage"),
    StageSpec("gis_export", "ENABLE_GIS_EXPORT", "src.gis_export.stage"),
    StageSpec("web_sync", "ENABLE_WEB_SYNC_EXPORT", "src.web_sync.stage"),
]


def resolve_runtime_options(
    runtime_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve pipeline options from config, optionally overridden at runtime."""
    import src.config as cfg

    options: Dict[str, Any] = {}
    for key in PIPELINE_OPTION_KEYS:
        options[key] = getattr(cfg, key, None)

    if runtime_overrides:
        for key, value in runtime_overrides.items():
            if key in options and value is not None:
                options[key] = value


    options["SEGMENT_SECONDS"] = float(options.get("SEGMENT_SECONDS", 5.0))
    options["SEGMENT_OVERLAP"] = float(options.get("SEGMENT_OVERLAP", 2.5))
    options["GEO_SYNC_TIME_OFFSET_SECONDS"] = float(options.get("GEO_SYNC_TIME_OFFSET_SECONDS", 25.0))
    options["GEO_SYNC_MAX_GAP_WARNING_SEC"] = float(options.get("GEO_SYNC_MAX_GAP_WARNING_SEC", 60.0))
    options["GEO_SYNC_FILENAME_TZ_OFFSET_HOURS"] = float(
        options.get("GEO_SYNC_FILENAME_TZ_OFFSET_HOURS", 8.0)
    )
    options["GEO_SYNC_FRAME_STEP"] = int(options.get("GEO_SYNC_FRAME_STEP", 60))
    for key in (
        "ENABLE_SEGMENT_PIPELINE",
        "ENABLE_VISUAL_SEGMENT_SUMMARY",
        "ENABLE_GEO_SYNC",
        "GEO_SYNC_ALIGN_TO_ANALYSIS_FRAMES",
        "ENABLE_SOUNDSCAPE",
        "ENABLE_WEB_SYNC_EXPORT",
        "ENABLE_GIS_EXPORT",
        "ENABLE_FUSION",
        "ENABLE_AGENTS",
        "ENABLE_DESIGN",
        "ENABLE_DELIVERABLE",
        "EXPORT_DEBUG_JSON",
        "GEO_SYNC_EXPORT_WGS84",
        "GEO_SYNC_USE_EXISTING_SEGMENTS",
        "WEB_SYNC_PREFER_WGS84",
        "GIS_EXPORT_PREFER_WGS84",
        "PANNS_FORCE_LOCAL_RESOURCES",
        "BUILD_MODEL_FEATURE_TABLE",
        "MODEL_DROP_HIGH_MISSING",
        "AGENT_CACHE_ENABLED",
        "AGENT_DISABLE_LLM",
        "POST_ONLY",
        "RESUME_MISSING_ONLY",
    ):
        options[key] = bool(options.get(key, False))

    options["MODEL_EVENT_VOCAB_TOP_N"] = int(options.get("MODEL_EVENT_VOCAB_TOP_N", 30))
    options["MODEL_TOPK_EVENT_VOCAB_TOP_N"] = int(options.get("MODEL_TOPK_EVENT_VOCAB_TOP_N", 20))
    options["MODEL_HIGH_MISSING_THRESHOLD"] = float(options.get("MODEL_HIGH_MISSING_THRESHOLD", 0.95))
    options["STEP8_TOP_N"] = int(options.get("STEP8_TOP_N", 0) or 0)
    options["AGENT_MAX_RETRIES"] = int(options.get("AGENT_MAX_RETRIES", 2))
    if options.get("ZHIPU_AGENT_MODEL") is not None:
        options["ZHIPU_AGENT_MODEL"] = str(options["ZHIPU_AGENT_MODEL"])
    if options.get("ZHIPU_VISION_QA_MODEL") is not None:
        options["ZHIPU_VISION_QA_MODEL"] = str(options["ZHIPU_VISION_QA_MODEL"])

    if options.get("FROM_EXISTING_OUTPUT") is not None:
        options["FROM_EXISTING_OUTPUT"] = str(options["FROM_EXISTING_OUTPUT"])

    return options


def has_enabled_stage(options: Mapping[str, Any]) -> bool:
    """Return True when at least one registry stage is enabled."""
    return any(bool(options.get(stage.enable_flag, False)) for stage in STAGE_SPECS)


def _summary_path(video_dir: str) -> Path:
    return Path(video_dir) / "multimodal" / "pipeline_summary.json"


def _stage_status_path(video_dir: str, stage_name: str) -> Path:
    return Path(video_dir) / "multimodal" / stage_name / "stage_status.json"


def _write_stage_status(
    video_dir: str,
    stage_name: str,
    state: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Persist normalized stage state for this run."""
    if state not in NORMALIZED_STATES:
        raise ValueError(f"invalid normalized stage state: {state}")

    path = _stage_status_path(video_dir, stage_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "stage": stage_name,
        "state": state,
        "status": state,
    }
    if extra:
        payload.update(dict(extra))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.as_posix()


def _record_stage(
    stage: str,
    state: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {"stage": stage, "state": state, "status": state}
    item.update(kwargs)
    return item


def run_post_analysis_pipeline(
    video_path: str,
    video_dir: str,
    output_dir: str,
    frame_skip: int,
    runtime_overrides: Optional[Mapping[str, Any]] = None,
    progress_manager: Optional[UnifiedProgressManager] = None,
    overall_progress_task: Any = None,
) -> Dict[str, Any]:
    """
    Run enabled multimodal stages after current frame-level analysis completes.

    Backward compatibility:
    - When all new stage flags are False, function exits without side effects.
    """
    options = resolve_runtime_options(runtime_overrides)
    discovered = discover_existing_artifacts(video_dir)
    readiness_lines = readiness_report_lines(video_dir)
    for line in readiness_lines:
        logger.info(line)

    if not has_enabled_stage(options):
        logger.info("[multimodal] all stages disabled; skip post-analysis pipeline")
        return {
            "pipeline_enabled": False,
            "reason": "all_stages_disabled",
            "stages": [],
            "readiness": discovered,
            "readiness_lines": readiness_lines,
        }

    config_snapshot = build_config_snapshot(options)
    manifest_path = write_run_manifest(
        video_path=video_path,
        video_dir=video_dir,
        output_dir=output_dir,
        config_snapshot=config_snapshot,
    )

    context: Dict[str, Any] = {
        "video_path": video_path,
        "video_dir": video_dir,
        "output_dir": output_dir,
        "frame_skip": int(frame_skip),
        "manifest_path": manifest_path,
        "options": options,
        "discovery": discovered,
    }

    stage_valid = stage_validity_map(video_dir)
    resume_missing_only = bool(options.get("RESUME_MISSING_ONLY", False))

    owns_progress_manager = progress_manager is None
    progress_manager = progress_manager or create_progress_manager()
    stage_results: List[Dict[str, Any]] = []

    def _advance_overall(stage_name: str, state: str) -> None:
        if overall_progress_task is not None:
            overall_progress_task.advance(1, description=f"pipeline | {stage_name} [{state}]")

    def _run_all_stages() -> None:
        for stage in STAGE_SPECS:
            stage_task = progress_manager.add_task(f"{stage.name} | pending", total=100.0)
            stage_context = dict(context)
            stage_context["progress_manager"] = progress_manager
            stage_context["stage_progress_task"] = stage_task
            if not bool(options.get(stage.enable_flag, False)):
                state = "disabled"
                extra = {"enable_flag": stage.enable_flag}
                status_file = _write_stage_status(video_dir, stage.name, state, extra=extra)
                stage_results.append(_record_stage(stage.name, state, status_file=status_file, **extra))
                stage_task.finish(status="skipped", description=f"{stage.name} | disabled")
                _advance_overall(stage.name, state)
                logger.info("[multimodal] stage=%s state=%s", stage.name, state)
                continue

            if resume_missing_only and bool(stage_valid.get(stage.name, False)):
                checks = validate_stage_outputs(video_dir, stage.name).get("checks", [])
                state = "skipped_existing"
                extra = {"checks": checks}
                status_file = _write_stage_status(video_dir, stage.name, state, extra=extra)
                stage_results.append(_record_stage(stage.name, state, status_file=status_file, **extra))
                stage_task.finish(status="skipped", description=f"{stage.name} | skipped_existing")
                _advance_overall(stage.name, state)
                logger.info("[multimodal] stage=%s state=%s", stage.name, state)
                continue

            missing_deps = find_missing_dependencies_for_stage(
                stage.name,
                discovered=discovered,
                stage_valid=stage_valid,
            )
            if missing_deps:
                state = "failed"
                extra = {
                    "error_type": "missing_dependency",
                    "missing_dependencies": missing_deps,
                }
                status_file = _write_stage_status(video_dir, stage.name, state, extra=extra)
                stage_results.append(_record_stage(stage.name, state, status_file=status_file, **extra))
                stage_task.finish(status="failed", description=f"{stage.name} | missing dependency")
                _advance_overall(stage.name, state)
                logger.warning(
                    "[multimodal] stage=%s state=%s missing=%s",
                    stage.name,
                    state,
                    ",".join(missing_deps),
                )
                continue

            try:
                module = importlib.import_module(stage.module_path)
                runner = getattr(module, stage.entrypoint)
            except Exception as exc:
                state = "failed"
                extra = {"error_type": "import_error", "error": str(exc)}
                status_file = _write_stage_status(video_dir, stage.name, state, extra=extra)
                stage_results.append(_record_stage(stage.name, state, status_file=status_file, **extra))
                stage_task.finish(status="failed", description=f"{stage.name} | import error")
                _advance_overall(stage.name, state)
                logger.exception("[multimodal] stage=%s state=%s error=%s", stage.name, state, exc)
                continue

            try:
                logger.info("[multimodal] stage=%s state=start", stage.name)
                stage_task.update(completed=0, total=100.0, description=f"{stage.name} | running")
                result = runner(stage_context)
                if not isinstance(result, dict):
                    result = {"runner_result": str(result)}
                check_after = validate_stage_outputs(video_dir, stage.name)
                if bool(check_after.get("valid", False)):
                    state = "ran"
                    stage_valid[stage.name] = True
                    extra = {"checks": check_after.get("checks", []), "runner_result": result}
                    stage_task.finish(status="done", description=f"{stage.name} | complete")
                else:
                    state = "failed"
                    extra = {
                        "error_type": "invalid_output",
                        "checks": check_after.get("checks", []),
                        "runner_result": result,
                    }
                    stage_valid[stage.name] = False
                    stage_task.finish(status="failed", description=f"{stage.name} | invalid output")
                status_file = _write_stage_status(video_dir, stage.name, state, extra=extra)
                stage_results.append(_record_stage(stage.name, state, status_file=status_file, **extra))
                _advance_overall(stage.name, state)
                logger.info("[multimodal] stage=%s state=%s", stage.name, state)
            except Exception as exc:
                state = "failed"
                extra = {"error_type": "runtime_error", "error": str(exc)}
                status_file = _write_stage_status(video_dir, stage.name, state, extra=extra)
                stage_results.append(_record_stage(stage.name, state, status_file=status_file, **extra))
                stage_task.finish(status="failed", description=f"{stage.name} | runtime error")
                _advance_overall(stage.name, state)
                logger.exception("[multimodal] stage=%s state=%s error=%s", stage.name, state, exc)

    if owns_progress_manager:
        with progress_manager:
            if overall_progress_task is None:
                overall_progress_task = progress_manager.add_task("pipeline | post stages", total=len(STAGE_SPECS))
            _run_all_stages()
    else:
        if overall_progress_task is None:
            overall_progress_task = progress_manager.add_task("pipeline | post stages", total=len(STAGE_SPECS))
        _run_all_stages()

    execution_summary = {
        "disabled": [s["stage"] for s in stage_results if s.get("state") == "disabled"],
        "skipped_existing": [s["stage"] for s in stage_results if s.get("state") == "skipped_existing"],
        "ran": [s["stage"] for s in stage_results if s.get("state") == "ran"],
        "failed": [s["stage"] for s in stage_results if s.get("state") == "failed"],
    }

    summary = {
        "pipeline_enabled": True,
        "manifest_path": manifest_path,
        "stages": stage_results,
        "execution_summary": execution_summary,
        "readiness": discovered,
        "readiness_lines": readiness_lines,
        "expected_output_contracts": stage_contract_as_posix(video_dir),
        "stage_output_checks": stage_checks_map(video_dir),
        "resume_missing_only": resume_missing_only,
        "from_existing_output": options.get("FROM_EXISTING_OUTPUT"),
    }

    summary_path = _summary_path(video_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("[multimodal] summary_written path=%s", summary_path.as_posix())

    return summary
