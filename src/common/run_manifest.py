


"""Run manifest utilities for optional multimodal post-analysis stages."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger("common.run_manifest")

PIPELINE_CONFIG_KEYS = (
    "ENABLE_SEGMENT_PIPELINE",
    "SEGMENT_SECONDS",
    "SEGMENT_OVERLAP",
    "ENABLE_SOUNDSCAPE",
    "ENABLE_FUSION",
    "ENABLE_AGENTS",
    "ENABLE_DESIGN",
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

    "POST_ONLY",
    "RESUME_MISSING_ONLY",
    "FROM_EXISTING_OUTPUT",
)


def _project_root() -> Path:
    """Return project root based on current file location."""
    return Path(__file__).resolve().parents[2]


def _safe_git_version(root: Path) -> Optional[str]:
    """
    Return short git hash when repository metadata is available.

    Falls back to None on any error so the pipeline remains robust in
    zip/copy deployments or environments without git.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        commit = (result.stdout or "").strip()
        if result.returncode == 0 and commit:
            return f"git:{commit}"
    except Exception:
        return None
    return None


def _fallback_version() -> str:
    """Fallback project version string when git is unavailable."""
    try:
        import src

        version = getattr(src, "__version__", None)
        if version:
            return f"src:{version}"
    except Exception:
        pass
    return "unknown"


def resolve_version_string() -> str:
    """Return a git-like version string if available."""
    root = _project_root()
    return _safe_git_version(root) or _fallback_version()


def build_config_snapshot(
    runtime_options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a manifest-safe config snapshot for multimodal stages."""
    import src.config as cfg

    snapshot: Dict[str, Any] = {}
    for key in PIPELINE_CONFIG_KEYS:
        snapshot[key] = getattr(cfg, key, None)

    if runtime_options:
        for key in PIPELINE_CONFIG_KEYS:
            if key in runtime_options and runtime_options[key] is not None:
                snapshot[key] = runtime_options[key]

    return snapshot


def write_run_manifest(
    video_path: str,
    video_dir: str,
    output_dir: str,
    config_snapshot: Mapping[str, Any],
) -> str:
    """
    Persist run manifest under `<video_dir>/multimodal/run_manifest.json`.

    The file records minimal reproducibility metadata and extension config.
    """
    manifest_dir = Path(video_dir) / "multimodal"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_video_path": str(Path(video_path).as_posix()),
        "video_output_dir": str(Path(video_dir).as_posix()),
        "output_root_dir": str(Path(output_dir).as_posix()),
        "version": resolve_version_string(),
        "config_snapshot": dict(config_snapshot),
    }

    manifest_path = manifest_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("[multimodal] manifest_written path=%s", manifest_path.as_posix())
    return str(manifest_path)
