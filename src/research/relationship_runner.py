


"""Runner for relationship analysis using existing output artifacts only."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .relationship_analysis import build_relationship_package
from .utils import normalize_segment_id, read_json

logger = logging.getLogger("research.relationship_runner")


def _resolve_paths(video_dir: str, relationship_outdir: Optional[str]) -> Dict[str, Path]:
    vdir = Path(video_dir)
    out_dir = Path(relationship_outdir) if relationship_outdir else (vdir / "relationship")
    return {
        "video_dir": vdir,
        "out_dir": out_dir,
        "model_feature_csv": vdir / "fusion" / "model_feature_table.csv",
        "model_feature_dict_json": vdir / "fusion" / "model_feature_dictionary.json",
        "audio_segment_features_csv": vdir / "soundscape" / "audio_segment_features.csv",
        "segment_manifest_csv": vdir / "segments" / "segment_manifest.csv",
    }


def _validate_required_inputs(paths: Dict[str, Path]) -> None:
    missing = [path.as_posix() for key, path in paths.items() if key not in {"video_dir", "out_dir"} and not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Relationship analysis requires existing fusion/soundscape outputs; missing: " + ", ".join(missing)
        )


def run_relationship_analysis(
    *,
    video_dir: str,
    relationship_outdir: Optional[str] = None,
) -> Dict[str, str]:
    paths = _resolve_paths(video_dir=video_dir, relationship_outdir=relationship_outdir)
    _validate_required_inputs(paths)

    model_df = normalize_segment_id(pd.read_csv(paths["model_feature_csv"]), "model_feature_table")
    audio_df = normalize_segment_id(pd.read_csv(paths["audio_segment_features_csv"]), "audio_segment_features")
    manifest_df = normalize_segment_id(pd.read_csv(paths["segment_manifest_csv"]), "segment_manifest")
    feature_dict = read_json(paths["model_feature_dict_json"])

    common_ids = sorted(
        set(model_df["segment_id"].tolist())
        & set(audio_df["segment_id"].tolist())
        & set(manifest_df["segment_id"].tolist())
    )
    if not common_ids:
        raise RuntimeError("No shared segment_id values across relationship-analysis input files.")

    model_df = model_df[model_df["segment_id"].isin(common_ids)].copy().sort_values("segment_id").reset_index(drop=True)
    audio_df = audio_df[audio_df["segment_id"].isin(common_ids)].copy().sort_values("segment_id").reset_index(drop=True)
    manifest_df = manifest_df[manifest_df["segment_id"].isin(common_ids)].copy().reset_index(drop=True)

    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    result = build_relationship_package(
        video_dir=video_dir,
        out_dir=paths["out_dir"],
        model_df=model_df,
        audio_df=audio_df,
        manifest_df=manifest_df,
        feature_dict=feature_dict,
    )
    logger.info("relationship runner done | out=%s", paths["out_dir"].as_posix())
    return result
