


"""Discovery and readiness helpers for post-analysis-from-existing mode."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .integrity import stage_validity_map
from .stage_contracts import DISCOVERY_KEYS, STAGE_REQUIRED_INPUTS


def discover_existing_artifacts(video_dir: str) -> Dict[str, object]:
    """Discover key legacy output folders used by multimodal post stages."""
    root = Path(video_dir)
    discovered: Dict[str, object] = {
        "video_dir": root.as_posix(),
        "exists": root.exists() and root.is_dir(),
    }
    for key, rel in DISCOVERY_KEYS.items():
        p = root / rel
        discovered[key] = p.is_dir()
        discovered[f"{key}_path"] = p.as_posix()

    discovered["stats"] = all(
        bool(discovered.get(k, False))
        for k in (
            "stats_visual_elements",
            "stats_green_view",
            "stats_emotion",
            "stats_people_count",
            "stats_color_analysis",
        )
    )
    return discovered


def readiness_report_lines(video_dir: str) -> List[str]:
    """
    Produce concise readiness report lines:
    - frames found?
    - stats found?
    - ai_evaluation found?
    - audio_events found?
    - which new stage outputs already exist?
    """
    discovered = discover_existing_artifacts(video_dir)
    stage_valid = stage_validity_map(video_dir)

    lines = [
        f"[readiness] video_dir={video_dir}",
        f"[readiness] frames_found={bool(discovered.get('frames', False))}",
        f"[readiness] stats_found={bool(discovered.get('stats', False))}",
        f"[readiness] ai_evaluation_found={bool(discovered.get('ai_evaluation', False))}",
        f"[readiness] audio_events_found={bool(discovered.get('audio_events', False))}",
        "[readiness] multimodal_outputs_valid="
        + ", ".join([f"{k}:{'yes' if v else 'no'}" for k, v in stage_valid.items()]),
    ]
    return lines


def find_missing_dependencies_for_stage(
    stage_name: str,
    discovered: Mapping[str, object],
    stage_valid: Mapping[str, bool],
) -> List[str]:
    """Return missing dependency tokens for the requested stage."""
    required = STAGE_REQUIRED_INPUTS.get(stage_name, [])
    missing: List[str] = []
    for dep in required:
        if dep.startswith("stage:"):
            dep_stage = dep.split(":", 1)[1]
            if not bool(stage_valid.get(dep_stage, False)):
                missing.append(dep)
        else:
            if not bool(discovered.get(dep, False)):
                missing.append(dep)
    return missing


def validate_requested_stage_dependencies(
    requested_stages: Sequence[str],
    discovered: Mapping[str, object],
    stage_valid: Mapping[str, bool],
) -> Dict[str, List[str]]:
    """Return missing dependencies per requested stage."""
    out: Dict[str, List[str]] = {}
    for stage in requested_stages:
        miss = find_missing_dependencies_for_stage(stage, discovered, stage_valid)
        if miss:
            out[stage] = miss
    return out
