


"""Runner for proof package using existing Step 7.5 refined outputs only."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .proof_package import build_proof_package
from .utils import normalize_segment_id

logger = logging.getLogger("research.proof_runner")


def _resolve_paths(video_dir: str, proof_outdir: Optional[str]) -> Dict[str, Path]:
    vdir = Path(video_dir)
    out_dir = Path(proof_outdir) if proof_outdir else (vdir / "proof")
    refined_dir = vdir / "fusion_eval_refined"
    return {
        "video_dir": vdir,
        "out_dir": out_dir,
        "segment_manifest_csv": vdir / "segments" / "segment_manifest.csv",
        "per_target_metrics_refined_csv": refined_dir / "per_target_metrics_refined.csv",
        "model_comparison_refined_csv": refined_dir / "model_comparison_refined.csv",
        "paired_deltas_refined_csv": refined_dir / "paired_deltas_refined.csv",
        "bootstrap_ci_refined_json": refined_dir / "bootstrap_ci_refined.json",
        "oof_predictions_refined_csv": refined_dir / "oof_predictions_refined.csv",
        "target_registry_refined_json": refined_dir / "target_registry_refined.json",
        "feature_group_registry_refined_json": refined_dir / "feature_group_registry_refined.json",
        "step75_summary_md": refined_dir / "step75_summary.md",
    }


def _validate_required_inputs(paths: Dict[str, Path]) -> None:
    missing = [path.as_posix() for key, path in paths.items() if key not in {"video_dir", "out_dir"} and not path.exists()]
    if missing:
        raise FileNotFoundError("Proof package requires existing Step 7.5 refined outputs; missing: " + ", ".join(missing))


def run_proof_package(
    *,
    video_dir: str,
    proof_outdir: Optional[str] = None,
) -> Dict[str, str]:
    paths = _resolve_paths(video_dir=video_dir, proof_outdir=proof_outdir)
    _validate_required_inputs(paths)

    manifest_df = normalize_segment_id(pd.read_csv(paths["segment_manifest_csv"]), "segment_manifest")
    oof_df = normalize_segment_id(pd.read_csv(paths["oof_predictions_refined_csv"]), "oof_predictions_refined")

    labeled_ids = sorted(oof_df["segment_id"].dropna().astype(int).unique().tolist())
    manifest_df = manifest_df[manifest_df["segment_id"].isin(labeled_ids)].copy().reset_index(drop=True)
    if manifest_df.empty:
        raise RuntimeError("Proof package could not align manifest rows with Step 7.5 OOF predictions.")

    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    result = build_proof_package(
        video_dir=video_dir,
        out_dir=paths["out_dir"],
        oof_df=oof_df,
        manifest_df=manifest_df,
    )
    logger.info("proof runner done | out=%s", paths["out_dir"].as_posix())
    return result
