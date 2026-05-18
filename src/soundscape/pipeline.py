


"""Soundscape stage entrypoint for segment-level audio feature extraction."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from src.config import (
    PANNS_FORCE_LOCAL_RESOURCES,
    PANNS_CHECKPOINT_PATH,
    PANNS_LABELS_PATH,
    SOUNDSCAPE_ENABLE_PANNS,
    SOUNDSCAPE_PANNS_EXPORT_DIMS,
    SOUNDSCAPE_ROLLOFF_RATIO,
    SOUNDSCAPE_STFT_HOP,
    SOUNDSCAPE_STFT_N_FFT,
    SOUNDSCAPE_TOP_K_EVENTS,
)

from .soundscape_features import extract_soundscape_features_for_video

logger = logging.getLogger("soundscape.pipeline")


def run_stage(context: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Build segment-level soundscape feature table from existing artifacts.

    This stage is post-analysis-only and reuses:
    - `segments/segment_manifest.csv`
    - `audio_events/*.wav`
    - `audio_events/audio_events_time_sync*.csv`
    """
    options = dict(context.get("options", {}))
    video_dir = str(context["video_dir"])
    stage_progress = context.get("stage_progress_task")

    result = extract_soundscape_features_for_video(
        video_dir=video_dir,
        top_k_events=int(options.get("SOUNDSCAPE_TOP_K_EVENTS", SOUNDSCAPE_TOP_K_EVENTS)),
        panns_enabled=bool(options.get("SOUNDSCAPE_ENABLE_PANNS", SOUNDSCAPE_ENABLE_PANNS)),
        panns_checkpoint_path=str(options.get("PANNS_CHECKPOINT_PATH", PANNS_CHECKPOINT_PATH)),
        panns_labels_path=str(options.get("PANNS_LABELS_PATH", PANNS_LABELS_PATH)),
        panns_force_local_resources=bool(
            options.get("PANNS_FORCE_LOCAL_RESOURCES", PANNS_FORCE_LOCAL_RESOURCES)
        ),
        panns_export_dims=int(
            options.get("SOUNDSCAPE_PANNS_EXPORT_DIMS", SOUNDSCAPE_PANNS_EXPORT_DIMS)
        ),
        n_fft=int(options.get("SOUNDSCAPE_STFT_N_FFT", SOUNDSCAPE_STFT_N_FFT)),
        hop=int(options.get("SOUNDSCAPE_STFT_HOP", SOUNDSCAPE_STFT_HOP)),
        rolloff_ratio=float(options.get("SOUNDSCAPE_ROLLOFF_RATIO", SOUNDSCAPE_ROLLOFF_RATIO)),
        progress_callback=(
            lambda completed, total=None, description=None: stage_progress.update(
                completed=completed,
                total=total,
                description=description or "soundscape",
            )
        )
        if stage_progress
        else None,
    )

    logger.info(
        "soundscape stage done | segments=%s panns_available=%s csv=%s",
        result.get("total_segments"),
        result.get("panns_available"),
        result.get("csv_path"),
    )
    return result
