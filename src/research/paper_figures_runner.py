


"""Runner for unified paper figures built from existing relationship/proof outputs."""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .paper_figures import build_paper_figures

logger = logging.getLogger("research.paper_figures_runner")


def _candidate_path(video_dir: Path, relative_path: str) -> Path:
    return video_dir / Path(relative_path)


def _closest_existing_path(video_dir: Path, relative_path: str) -> Optional[Path]:
    wanted = str(Path(relative_path).as_posix())
    all_files = [p for p in video_dir.rglob("*") if p.is_file()]
    if not all_files:
        return None

    exact_name = Path(relative_path).name.lower()
    name_matches = [p for p in all_files if p.name.lower() == exact_name]
    if name_matches:
        return sorted(name_matches)[0]

    relative_names = [str(p.relative_to(video_dir).as_posix()) for p in all_files]
    match = difflib.get_close_matches(wanted, relative_names, n=1, cutoff=0.45)
    if match:
        return video_dir / Path(match[0])
    return None


def _resolve_one(video_dir: Path, relative_path: str) -> Tuple[Path, Optional[str]]:
    candidate = _candidate_path(video_dir, relative_path)
    if candidate.exists():
        return candidate, None
    fallback = _closest_existing_path(video_dir, relative_path)
    if fallback and fallback.exists():
        note = f"`{relative_path}` was not found; used closest existing file `{fallback.relative_to(video_dir).as_posix()}`."
        return fallback, note
    raise FileNotFoundError(
        f"Paper figures require existing output artifact: {candidate.as_posix()}"
    )


def _resolve_paths(video_dir: str, paper_figures_outdir: Optional[str]) -> Tuple[Dict[str, Path], List[str]]:
    vdir = Path(video_dir)
    out_dir = Path(paper_figures_outdir) if paper_figures_outdir else (vdir / "paper_figures")
    expected = {
        "pls_latent_summary": "relationship/pls_latent_summary.csv",
        "pls_x_loadings": "relationship/pls_x_loadings.csv",
        "pls_y_loadings": "relationship/pls_y_loadings.csv",
        "pls_bootstrap_stability": "relationship/pls_bootstrap_stability.csv",
        "significant_pairwise_links": "relationship/significant_pairwise_links.csv",
        "group_pair_tests_combined": "relationship/group_confirmatory/group_pair_tests_combined.csv",
        "confirmatory_claim_registry": "relationship/group_confirmatory/confirmatory_claim_registry.csv",
        "hypothesis_registry": "relationship/group_confirmatory/hypothesis_registry.csv",
        "relationship_summary": "relationship/relationship_summary.md",
        "group_confirmatory_summary": "relationship/group_confirmatory/group_confirmatory_summary.md",
        "proof_model_comparison": "proof/proof_model_comparison.csv",
        "proof_paired_deltas": "proof/proof_paired_deltas.csv",
        "proof_bootstrap_ci": "proof/proof_bootstrap_ci.csv",
        "proof_permutation_tests": "proof/proof_permutation_tests.csv",
        "proof_claim_registry": "proof/proof_claim_registry.csv",
        "proof_summary": "proof/proof_summary.md",
        "per_target_metrics_refined": "fusion_eval_refined/per_target_metrics_refined.csv",
        "model_feature_table": "fusion/model_feature_table.csv",
        "segment_manifest": "segments/segment_manifest.csv",
    }
    resolved: Dict[str, Path] = {"video_dir": vdir, "out_dir": out_dir}
    notes: List[str] = []
    for logical_name, relative_path in expected.items():
        actual_path, note = _resolve_one(vdir, relative_path)
        resolved[logical_name] = actual_path
        if note:
            notes.append(note)
    if "relationship/pls_scores.csv" not in [str(p.relative_to(vdir).as_posix()) for p in vdir.rglob("*.csv")]:
        notes.append(
            "No persisted PLS score table was available under `relationship/`; "
            "Fig A1 therefore reconstructed LV1 scores from `fusion/model_feature_table.csv` "
            "using the saved PLS feature lists and loadings."
        )
    return resolved, notes


def run_paper_figures(
    *,
    video_dir: str,
    paper_figures_outdir: Optional[str] = None,
) -> Dict[str, str]:
    resolved_paths, notes = _resolve_paths(video_dir=video_dir, paper_figures_outdir=paper_figures_outdir)
    out_dir = resolved_paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_paper_figures(
        video_dir=video_dir,
        out_dir=out_dir,
        resolved_paths={k: v for k, v in resolved_paths.items() if k not in {"video_dir", "out_dir"}},
        resolved_notes=notes,
    )
    logger.info("paper figures runner done | out=%s", out_dir.as_posix())
    return result
