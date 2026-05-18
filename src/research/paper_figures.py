


"""Generate a unified four-figure paper bundle from existing relationship/proof outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSCanonical

from .paper_figures_plotting import (
    CLAIM_ORDER,
    plot_fig_a1_pls_lv1_coupling,
    plot_fig_a2_group_association_matrix,
    plot_fig_b1_targetwise_model_dumbbell,
    plot_fig_b2_fusion_incremental_forest,
)
from .utils import normalize_segment_id

logger = logging.getLogger("research.paper_figures")

FUSION_MODEL = "early_fusion_screened"
VISUAL_MODEL = "visual_only_screened"
AUDIO_MODEL = "audio_only_screened"

GROUP_VISUAL_ORDER = [
    "people_presence",
    "green_nature",
    "traffic_road_hardscape",
    "visual_emotion_aesthetic",
    "ai_activity",
    "visual_color",
    "visual_semantic_general",
]

GROUP_AUDIO_ORDER = [
    "audio_signal_level",
    "audio_event_human",
    "audio_event_traffic_mechanical",
    "audio_event_natural",
    "audio_event_general",
    "audio_embedding_general",
]


def _standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=float)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, ddof=1, keepdims=True)
    std[~np.isfinite(std) | (std <= 0)] = 1.0
    scaled = (arr - mean) / std
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


def _normalize_target_label(name: str) -> str:
    text = str(name).replace("_", " ").strip()
    if not text:
        return str(name)
    return text[0].upper() + text[1:]


def _claim_rank(label: str) -> int:
    try:
        return CLAIM_ORDER.index(str(label))
    except ValueError:
        return len(CLAIM_ORDER)


def _coerce_feature_block(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out: Dict[str, pd.Series] = {}
    for column in columns:
        if column not in df.columns:
            raise KeyError(f"PLS reconstruction requires feature column: {column}")
        series = pd.to_numeric(df[column], errors="coerce")
        fill_value = series.median()
        if not np.isfinite(fill_value):
            fill_value = 0.0
        out[str(column)] = series.fillna(fill_value).astype(float)
    return pd.DataFrame(out, index=df.index)


def _ordered_segments(model_df: pd.DataFrame, manifest_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    manifest = normalize_segment_id(manifest_df, "segment_manifest").copy()
    model = normalize_segment_id(model_df, "model_feature_table").copy()
    common_ids = sorted(set(model["segment_id"].tolist()) & set(manifest["segment_id"].tolist()))
    if not common_ids:
        raise RuntimeError("Paper figures could not align model features with segment manifest.")
    manifest = manifest[manifest["segment_id"].isin(common_ids)].copy()
    sort_cols = [c for c in ["center_time_sec", "start_time_sec", "segment_id"] if c in manifest.columns]
    if not sort_cols:
        sort_cols = ["segment_id"]
    manifest = manifest.sort_values(sort_cols).reset_index(drop=True)
    ordered_ids = manifest["segment_id"].astype(int).tolist()
    model = model[model["segment_id"].isin(ordered_ids)].copy().set_index("segment_id").loc[ordered_ids].reset_index()
    return model, manifest


def _reconstruct_pls_scores(
    latent_df: pd.DataFrame,
    x_loadings_df: pd.DataFrame,
    model_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ordered_model, ordered_manifest = _ordered_segments(model_df, manifest_df)
    lv1_row = latent_df.sort_values("component").iloc[0]
    visual_features = [x for x in str(lv1_row["visual_features"]).split("|") if x]
    audio_features = [x for x in str(lv1_row["audio_features"]).split("|") if x]
    n_components = int(max(1, latent_df["component"].max()))

    x_df = _standardize_frame(_coerce_feature_block(ordered_model, visual_features))
    y_df = _standardize_frame(_coerce_feature_block(ordered_model, audio_features))

    model = PLSCanonical(n_components=n_components, scale=False, max_iter=500)
    x_scores, y_scores = model.fit_transform(x_df, y_df)
    saved_x = x_loadings_df.pivot(index="feature_name", columns="component", values="loading")
    for component in range(n_components):
        saved_component = saved_x.reindex(visual_features).iloc[:, component].to_numpy(dtype=float)
        current_component = np.asarray(model.x_loadings_[:, component], dtype=float)
        mask = np.isfinite(saved_component) & np.isfinite(current_component)
        if mask.sum() >= 2:
            corr = np.corrcoef(saved_component[mask], current_component[mask])[0, 1]
            if np.isfinite(corr) and corr < 0:
                x_scores[:, component] *= -1.0
                y_scores[:, component] *= -1.0

    score_rows: List[Dict[str, Any]] = []
    ordered_ids = ordered_manifest["segment_id"].astype(int).tolist()
    for component in range(n_components):
        for row_idx, segment_id in enumerate(ordered_ids):
            score_rows.append(
                {
                    "segment_id": int(segment_id),
                    "component": int(component + 1),
                    "x_score": float(x_scores[row_idx, component]),
                    "y_score": float(y_scores[row_idx, component]),
                }
            )
    thin_n = int(len(ordered_manifest.iloc[::2]))
    return pd.DataFrame(score_rows), {
        "n_full": int(len(ordered_manifest)),
        "n_thin": thin_n,
        "visual_features": visual_features,
        "audio_features": audio_features,
    }


def _prepare_group_orders(combined_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    present_visual = combined_df.loc[combined_df["testable_flag"], "visual_group"].dropna().astype(str).unique().tolist()
    present_audio = combined_df.loc[combined_df["testable_flag"], "audio_group"].dropna().astype(str).unique().tolist()
    visual_order = [item for item in GROUP_VISUAL_ORDER if item in present_visual] + [
        item for item in present_visual if item not in GROUP_VISUAL_ORDER
    ]
    audio_order = [item for item in GROUP_AUDIO_ORDER if item in present_audio] + [
        item for item in present_audio if item not in GROUP_AUDIO_ORDER
    ]
    return visual_order, audio_order


def _prepare_target_order(claim_df: pd.DataFrame) -> List[str]:
    rank_df = claim_df.copy()
    rank_df["claim_rank"] = rank_df["claim_label"].map(_claim_rank)
    rank_df["fusion_advantage"] = -rank_df[
        ["fusion_vs_visual_delta", "fusion_vs_audio_delta"]
    ].mean(axis=1, skipna=True)
    rank_df = rank_df.sort_values(
        ["claim_rank", "fusion_advantage", "target_name"],
        ascending=[True, False, True],
    )
    return rank_df["target_name"].astype(str).tolist()


def _prepare_dumbbell_table(
    model_df: pd.DataFrame,
    claim_df: pd.DataFrame,
) -> pd.DataFrame:
    full_df = model_df[
        (model_df["subset"] == "full")
        & (model_df["model_group"].isin([VISUAL_MODEL, AUDIO_MODEL, FUSION_MODEL]))
    ].copy()
    if full_df.empty:
        raise RuntimeError("Paper figure B1 requires full-sample proof_model_comparison rows.")

    target_order = _prepare_target_order(claim_df)
    claim_lookup = claim_df.set_index("target_name")["claim_label"].to_dict()

    rows: List[Dict[str, Any]] = []
    for target_name in target_order:
        target_df = full_df[full_df["target_name"] == target_name].copy()
        if target_df.empty:
            continue
        perf = target_df.set_index("model_group")["primary_value"].to_dict()
        if not all(key in perf for key in [VISUAL_MODEL, AUDIO_MODEL, FUSION_MODEL]):
            continue
        rows.append(
            {
                "target_name": str(target_name),
                "target_label": _normalize_target_label(str(target_name)),
                "claim_label": str(claim_lookup.get(target_name, "")),
                "visual_value": float(perf[VISUAL_MODEL]),
                "audio_value": float(perf[AUDIO_MODEL]),
                "fusion_value": float(perf[FUSION_MODEL]),
                "primary_metric_label": str(target_df["primary_metric_label"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _prepare_forest_table(
    claim_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    perm_df: pd.DataFrame,
) -> pd.DataFrame:
    full_ci = ci_df[ci_df["subset"] == "full"].copy()
    full_perm = perm_df[perm_df["subset"] == "full"].copy()
    merged = full_ci.merge(
        full_perm[
            [
                "target_name",
                "comparison",
                "primary_metric",
                "p_value",
                "q_value",
                "better_direction",
            ]
        ],
        on=["target_name", "comparison", "primary_metric"],
        how="left",
    )
    merged = merged.merge(
        claim_df[["target_name", "claim_label"]],
        on="target_name",
        how="left",
    )
    target_order = _prepare_target_order(claim_df)
    order_lookup = {name: idx for idx, name in enumerate(target_order)}
    comparison_rank = {"fusion_vs_visual_only": 0, "fusion_vs_audio_only": 1}
    merged["target_rank"] = merged["target_name"].map(order_lookup)
    merged["comparison_rank"] = merged["comparison"].map(comparison_rank).fillna(9)
    merged = merged.sort_values(["target_rank", "comparison_rank"]).reset_index(drop=True)
    merged["significant"] = merged["q_value"].fillna(1.0) < 0.05
    comparison_label = {
        "fusion_vs_visual_only": "  vs visual-only",
        "fusion_vs_audio_only": "  vs audio-only",
    }
    row_labels: List[str] = []
    previous_target: Optional[str] = None
    for _, row in merged.iterrows():
        target_name = str(row["target_name"])
        label_prefix = _normalize_target_label(target_name) if target_name != previous_target else ""
        row_labels.append(label_prefix + comparison_label.get(str(row["comparison"]), ""))
        previous_target = target_name
    merged["row_label"] = row_labels
    return merged


def _resolve_figure_inputs_summary(path_map: Mapping[str, str]) -> str:
    items = []
    for logical_name, actual_path in sorted(path_map.items()):
        items.append(f"- `{logical_name}` -> `{actual_path}`")
    return "\n".join(items)


def _build_summary_markdown(
    *,
    out_dir: Path,
    resolved_paths: Mapping[str, str],
    fig_inputs: Mapping[str, Sequence[str]],
    mapped_notes: Sequence[str],
) -> str:
    lines: List[str] = [
        "# Paper Figures Summary",
        "",
        "## 1) Figure inventory",
        "- `figA1_pls_lv1_coupling`: relationship main-text figure for dominant latent coupling.",
        "- `figA2_group_association_matrix`: relationship main-text figure for the group-level association pattern and confirmatory-pair status.",
        "- `figB1_targetwise_model_dumbbell`: proof main-text figure for target-specific positions of visual-only, audio-only, and fusion models.",
        "- `figB2_fusion_incremental_forest`: proof main-text figure for paired fusion deltas and 95% confidence intervals.",
        "",
        "## 2) Input files by figure",
    ]
    for figure_name, inputs in fig_inputs.items():
        lines.append(f"### {figure_name}")
        lines.extend([f"- `{item}`" for item in inputs])
    lines.extend(
        [
            "",
            "## 3) Intended claim",
            "- `figA1`: the audio-visual relationship is visible at the dominant latent level.",
            "- `figA2`: the group-level pattern is structured, but confirmatory group pairs differ in strength and support status.",
            "- `figB1`: fusion is target-specific and should be interpreted alongside visual-only and audio-only baselines, not as uniformly best.",
            "- `figB2`: fusion increments are heterogeneous and uncertainty frequently crosses 0, which supports a complementary-value interpretation rather than blanket superiority.",
            "",
            "## 4) Placement suggestion",
            "- `figA1`: main text.",
            "- `figA2`: main text or near-main supplementary depending on space.",
            "- `figB1`: main text.",
            "- `figB2`: main text if model-comparison inference is central; otherwise first supplementary figure after `figB1`.",
            "",
            "## 5) Visual encoding",
            "- Common style: white background, restrained grid, thin lines, sans-serif defaults, 600 dpi PNG/PDF.",
            "- Relationship figures: diverging blue-white-amber matrix for signed association; confirmatory cells use border styles rather than color alone.",
            "- Proof figures: visual-only = open blue circle, audio-only = open amber square, fusion = filled dark diamond; forest comparisons use shape+fill to show comparison type and q-significance.",
            "- Direction conventions: lower is better in proof figures; negative delta favors fusion in the forest plot.",
            "",
            "## 6) File resolution and automatic adaptation",
            _resolve_figure_inputs_summary(resolved_paths),
        ]
    )
    if mapped_notes:
        lines.append("")
        lines.append("Additional adaptation notes:")
        lines.extend([f"- {item}" for item in mapped_notes])
    else:
        lines.append("")
        lines.append("- No filename remapping was needed.")
    lines.extend(
        [
            "",
            "## 7) Recommended display order",
            "- Relationship: `figA1_pls_lv1_coupling` then `figA2_group_association_matrix`.",
            "- Proof: `figB1_targetwise_model_dumbbell` then `figB2_fusion_incremental_forest`.",
            "",
            f"Output directory: `{out_dir.as_posix()}`",
        ]
    )
    return "\n".join(lines)


def build_paper_figures(
    *,
    video_dir: str,
    out_dir: Path,
    resolved_paths: Mapping[str, Path],
    resolved_notes: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_df = pd.read_csv(resolved_paths["pls_latent_summary"])
    x_loadings_df = pd.read_csv(resolved_paths["pls_x_loadings"])
    combined_group_df = pd.read_csv(resolved_paths["group_pair_tests_combined"])
    proof_model_df = pd.read_csv(resolved_paths["proof_model_comparison"])
    proof_ci_df = pd.read_csv(resolved_paths["proof_bootstrap_ci"])
    proof_perm_df = pd.read_csv(resolved_paths["proof_permutation_tests"])
    proof_claim_df = pd.read_csv(resolved_paths["proof_claim_registry"])
    model_feature_df = pd.read_csv(resolved_paths["model_feature_table"])
    manifest_df = pd.read_csv(resolved_paths["segment_manifest"])

    score_df, pls_meta = _reconstruct_pls_scores(
        latent_df=latent_df,
        x_loadings_df=x_loadings_df,
        model_df=model_feature_df,
        manifest_df=manifest_df,
    )
    lv1_score_df = score_df[score_df["component"] == 1].copy()
    lv1_row = latent_df.sort_values("component").iloc[0]
    fig_a1_base = out_dir / "figA1_pls_lv1_coupling"
    plot_fig_a1_pls_lv1_coupling(
        lv1_score_df,
        corr_value=float(lv1_row["score_correlation"]),
        permutation_p=float(lv1_row["permutation_p"]),
        n_full=int(pls_meta["n_full"]),
        n_thin=int(pls_meta["n_thin"]),
        out_base=fig_a1_base,
    )

    visual_groups, audio_groups = _prepare_group_orders(combined_group_df)
    fig_a2_base = out_dir / "figA2_group_association_matrix"
    plot_fig_a2_group_association_matrix(
        combined_group_df,
        visual_groups=visual_groups,
        audio_groups=audio_groups,
        out_base=fig_a2_base,
    )

    dumbbell_df = _prepare_dumbbell_table(proof_model_df, proof_claim_df)
    fig_b1_base = out_dir / "figB1_targetwise_model_dumbbell"
    plot_fig_b1_targetwise_model_dumbbell(dumbbell_df, out_base=fig_b1_base)

    forest_df = _prepare_forest_table(proof_claim_df, proof_ci_df, proof_perm_df)
    fig_b2_base = out_dir / "figB2_fusion_incremental_forest"
    plot_fig_b2_fusion_incremental_forest(forest_df, out_base=fig_b2_base)

    fig_inputs = {
        "figA1_pls_lv1_coupling": [
            resolved_paths["pls_latent_summary"].as_posix(),
            resolved_paths["pls_x_loadings"].as_posix(),
            resolved_paths["model_feature_table"].as_posix(),
            resolved_paths["segment_manifest"].as_posix(),
        ],
        "figA2_group_association_matrix": [
            resolved_paths["group_pair_tests_combined"].as_posix(),
            resolved_paths["confirmatory_claim_registry"].as_posix(),
            resolved_paths["hypothesis_registry"].as_posix(),
        ],
        "figB1_targetwise_model_dumbbell": [
            resolved_paths["proof_model_comparison"].as_posix(),
            resolved_paths["proof_claim_registry"].as_posix(),
        ],
        "figB2_fusion_incremental_forest": [
            resolved_paths["proof_bootstrap_ci"].as_posix(),
            resolved_paths["proof_permutation_tests"].as_posix(),
            resolved_paths["proof_claim_registry"].as_posix(),
        ],
    }
    summary_text = _build_summary_markdown(
        out_dir=out_dir,
        resolved_paths={key: path.as_posix() for key, path in resolved_paths.items()},
        fig_inputs=fig_inputs,
        mapped_notes=list(resolved_notes or []),
    )
    summary_path = out_dir / "paper_figures_summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")

    logger.info("paper figures done | out=%s video=%s", out_dir.as_posix(), video_dir)
    return {
        "paper_figures_outdir": out_dir.as_posix(),
        "figA1_pls_lv1_coupling_png": fig_a1_base.with_suffix(".png").as_posix(),
        "figA1_pls_lv1_coupling_pdf": fig_a1_base.with_suffix(".pdf").as_posix(),
        "figA2_group_association_matrix_png": fig_a2_base.with_suffix(".png").as_posix(),
        "figA2_group_association_matrix_pdf": fig_a2_base.with_suffix(".pdf").as_posix(),
        "figB1_targetwise_model_dumbbell_png": fig_b1_base.with_suffix(".png").as_posix(),
        "figB1_targetwise_model_dumbbell_pdf": fig_b1_base.with_suffix(".pdf").as_posix(),
        "figB2_fusion_incremental_forest_png": fig_b2_base.with_suffix(".png").as_posix(),
        "figB2_fusion_incremental_forest_pdf": fig_b2_base.with_suffix(".pdf").as_posix(),
        "paper_figures_summary_md": summary_path.as_posix(),
    }
