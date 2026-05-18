


"""Cross-modal relationship analysis built on existing fusion and soundscape outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSCanonical

from .plotting import (
    plot_cross_modal_heatmap,
    plot_pls_loadings,
    plot_pls_scores,
    plot_top_links_lollipop,
)
from .utils import (
    benjamini_hochberg,
    centered_distance_flatten,
    circular_block_permutations,
    contiguous_blocks,
    order_segments,
    prepare_cross_modal_feature_table,
    pretty_group_name,
    short_feature_label,
    standardize_rank_frame,
    write_json,
)

logger = logging.getLogger("research.relationship")


def _pairwise_association(
    visual_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    *,
    block_size: int,
) -> Dict[str, Any]:
    n_rows = int(len(visual_df))
    if n_rows != int(len(audio_df)):
        raise ValueError("visual_df and audio_df must have the same number of rows.")

    visual_features = visual_df.columns.tolist()
    audio_features = audio_df.columns.tolist()

    visual_rank = standardize_rank_frame(visual_df)
    audio_rank = standardize_rank_frame(audio_df)
    spearman_obs = (visual_rank.T @ audio_rank) / max(1, n_rows - 1)

    d_x, dvar_x = centered_distance_flatten(visual_df)
    d_y, dvar_y = centered_distance_flatten(audio_df)
    denom = np.sqrt(np.outer(dvar_x, dvar_y))
    denom[~np.isfinite(denom) | (denom <= 0)] = np.nan
    dcov2_obs = (d_x @ d_y.T) / float(max(1, n_rows * n_rows))
    dcor2_obs = np.clip(dcov2_obs / denom, 0.0, None)
    dcor_obs = np.sqrt(np.nan_to_num(dcor2_obs, nan=0.0))

    permutations = circular_block_permutations(n_rows=n_rows, block_size=block_size, include_identity=False)
    logger.info(
        "relationship pairwise | n=%s visual=%s audio=%s block_size=%s permutations=%s",
        n_rows,
        len(visual_features),
        len(audio_features),
        block_size,
        len(permutations),
    )

    if not permutations:
        spearman_p = np.ones_like(spearman_obs, dtype=float)
        dcor_p = np.ones_like(dcor_obs, dtype=float)
    else:
        spearman_counts = np.zeros_like(spearman_obs, dtype=float)
        dcor_counts = np.zeros_like(dcor_obs, dtype=float)
        abs_spearman_obs = np.abs(spearman_obs)

        for perm in permutations:
            perm = np.asarray(perm, dtype=int)
            spearman_perm = (visual_rank.T @ audio_rank[perm, :]) / max(1, n_rows - 1)
            spearman_counts += (np.abs(spearman_perm) >= abs_spearman_obs - 1e-12).astype(float)

            perm_ix = (perm[:, None] * n_rows + perm[None, :]).reshape(-1)
            d_y_perm = d_y[:, perm_ix]
            dcov2_perm = (d_x @ d_y_perm.T) / float(max(1, n_rows * n_rows))
            dcor2_perm = np.clip(dcov2_perm / denom, 0.0, None)
            dcor_perm = np.sqrt(np.nan_to_num(dcor2_perm, nan=0.0))
            dcor_counts += (dcor_perm >= dcor_obs - 1e-12).astype(float)

        spearman_p = (spearman_counts + 1.0) / (len(permutations) + 1.0)
        dcor_p = (dcor_counts + 1.0) / (len(permutations) + 1.0)

    spearman_q = benjamini_hochberg(spearman_p.reshape(-1)).reshape(spearman_p.shape)
    dcor_q = benjamini_hochberg(dcor_p.reshape(-1)).reshape(dcor_p.shape)

    rows: List[Dict[str, Any]] = []
    for i, v_feature in enumerate(visual_features):
        for j, a_feature in enumerate(audio_features):
            rows.append(
                {
                    "visual_feature": v_feature,
                    "audio_feature": a_feature,
                    "spearman_rho": float(spearman_obs[i, j]),
                    "spearman_p": float(spearman_p[i, j]),
                    "spearman_q": float(spearman_q[i, j]),
                    "dcor": float(dcor_obs[i, j]),
                    "dcor_p": float(dcor_p[i, j]),
                    "dcor_q": float(dcor_q[i, j]),
                    "combined_effect": float((abs(spearman_obs[i, j]) + dcor_obs[i, j]) / 2.0),
                    "n_segments": int(n_rows),
                    "block_size": int(block_size),
                    "permutation_scheme": "circular_block_shift",
                }
            )

    result_df = pd.DataFrame(rows).sort_values(["visual_feature", "audio_feature"]).reset_index(drop=True)
    spearman_matrix = pd.DataFrame(spearman_obs, index=visual_features, columns=audio_features)
    dcor_matrix = pd.DataFrame(dcor_obs, index=visual_features, columns=audio_features)
    return {
        "long_df": result_df,
        "spearman_matrix": spearman_matrix,
        "dcor_matrix": dcor_matrix,
    }


def _direction_label(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _component_count(n_rows: int, n_visual: int, n_audio: int) -> int:
    return int(max(1, min(2, n_rows - 1, n_visual, n_audio)))


def _standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=float)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, ddof=1, keepdims=True)
    std[~np.isfinite(std) | (std <= 0)] = 1.0
    scaled = (arr - mean) / std
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


def _select_pls_features(
    merged_pairs: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_registry: pd.DataFrame,
    *,
    top_k_per_modality: int = 12,
    max_per_group: int = 4,
) -> Tuple[List[str], List[str]]:
    score_df = merged_pairs.copy()
    score_df["stable_score"] = score_df["combined_effect"] * (
        1.0
        + 0.15 * score_df["direction_consistent"].astype(float)
        + 0.20 * score_df["thin_still_significant"].astype(float)
    )

    def _aggregate(feature_col: str, group_col: str) -> pd.DataFrame:
        agg = (
            score_df.groupby([feature_col, group_col], dropna=False)["stable_score"]
            .max()
            .reset_index()
            .sort_values("stable_score", ascending=False)
            .reset_index(drop=True)
        )
        return agg

    def _cap_group(df: pd.DataFrame, feature_col: str, group_col: str) -> List[str]:
        chosen: List[str] = []
        group_counts: Dict[str, int] = {}
        for _, row in df.iterrows():
            feature_name = str(row[feature_col])
            group_name = str(row[group_col])
            if group_counts.get(group_name, 0) >= int(max_per_group):
                continue
            chosen.append(feature_name)
            group_counts[group_name] = group_counts.get(group_name, 0) + 1
            if len(chosen) >= int(top_k_per_modality):
                break
        return chosen

    visual_ranked = _aggregate("visual_feature", "visual_group")
    audio_ranked = _aggregate("audio_feature", "audio_group")
    visual_features = _cap_group(visual_ranked, "visual_feature", "visual_group")
    audio_features = _cap_group(audio_ranked, "audio_feature", "audio_group")

    if len(visual_features) < min(6, top_k_per_modality):
        fallback = feature_registry[
            (feature_registry["kept_or_dropped"] == "kept") & (feature_registry["modality"] == "visual")
        ]["feature_name"].tolist()
        fallback = sorted(fallback, key=lambda c: float(feature_df[c].var(ddof=0)), reverse=True)
        for feature_name in fallback:
            if feature_name not in visual_features:
                visual_features.append(feature_name)
            if len(visual_features) >= int(top_k_per_modality):
                break

    if len(audio_features) < min(6, top_k_per_modality):
        fallback = feature_registry[
            (feature_registry["kept_or_dropped"] == "kept") & (feature_registry["modality"] == "audio")
        ]["feature_name"].tolist()
        fallback = sorted(fallback, key=lambda c: float(feature_df[c].var(ddof=0)), reverse=True)
        for feature_name in fallback:
            if feature_name not in audio_features:
                audio_features.append(feature_name)
            if len(audio_features) >= int(top_k_per_modality):
                break

    return visual_features[:top_k_per_modality], audio_features[:top_k_per_modality]


def _fit_pls(
    visual_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    *,
    n_components: int,
) -> Tuple[PLSCanonical, pd.DataFrame]:
    x_df = _standardize_frame(visual_df)
    y_df = _standardize_frame(audio_df)
    model = PLSCanonical(n_components=n_components, scale=False, max_iter=500)
    x_scores, y_scores = model.fit_transform(x_df, y_df)
    score_rows: List[Dict[str, Any]] = []
    for component in range(n_components):
        corr = float(np.corrcoef(x_scores[:, component], y_scores[:, component])[0, 1])
        score_rows.append(
            {
                "component": int(component + 1),
                "score_correlation": corr,
            }
        )
    score_df = pd.DataFrame(score_rows)
    return model, score_df


def _pls_permutation_test(
    visual_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    fitted_model: PLSCanonical,
    observed_scores: pd.DataFrame,
    *,
    block_size: int,
) -> Dict[str, Any]:
    n_rows = len(visual_df)
    permutations = circular_block_permutations(n_rows=n_rows, block_size=block_size, include_identity=False)
    if not permutations:
        return {
            "n_permutations": 0,
            "block_size": int(block_size),
            "components": [],
        }

    x_df = _standardize_frame(visual_df)
    y_df = _standardize_frame(audio_df)
    n_components = fitted_model.n_components
    observed = observed_scores["score_correlation"].to_numpy(dtype=float)
    counts = np.zeros(n_components, dtype=float)

    for perm in permutations:
        perm = np.asarray(perm, dtype=int)
        perm_model = PLSCanonical(n_components=n_components, scale=False, max_iter=500)
        x_scores, y_scores = perm_model.fit_transform(x_df, y_df.iloc[perm].reset_index(drop=True))
        perm_corr = np.asarray(
            [
                float(np.corrcoef(x_scores[:, idx], y_scores[:, idx])[0, 1])
                for idx in range(n_components)
            ],
            dtype=float,
        )
        counts += (np.abs(perm_corr) >= np.abs(observed) - 1e-12).astype(float)

    p_values = (counts + 1.0) / (len(permutations) + 1.0)
    components = [
        {
            "component": int(i + 1),
            "observed_score_correlation": float(observed[i]),
            "permutation_p": float(p_values[i]),
        }
        for i in range(n_components)
    ]
    return {
        "n_permutations": int(len(permutations)),
        "block_size": int(block_size),
        "components": components,
    }


def _bootstrap_pls_stability(
    visual_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    fitted_model: PLSCanonical,
    *,
    block_size: int,
    n_bootstrap: int,
    seed: int,
    group_lookup: Mapping[str, str],
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    n_rows = len(visual_df)
    blocks = contiguous_blocks(n_rows, block_size=block_size)
    x_df = _standardize_frame(visual_df)
    y_df = _standardize_frame(audio_df)
    observed_x = np.asarray(fitted_model.x_loadings_, dtype=float)
    observed_y = np.asarray(fitted_model.y_loadings_, dtype=float)
    n_components = observed_x.shape[1]

    storage: Dict[Tuple[str, str, int], List[float]] = {}
    for modality, columns in [("visual", visual_df.columns.tolist()), ("audio", audio_df.columns.tolist())]:
        for feature_name in columns:
            for component in range(n_components):
                storage[(modality, feature_name, component + 1)] = []

    for _ in range(int(n_bootstrap)):
        sample_idx: List[int] = []
        while len(sample_idx) < n_rows:
            block = blocks[int(rng.integers(0, len(blocks)))]
            sample_idx.extend(block.tolist())
        sample_idx = sample_idx[:n_rows]

        boot_model = PLSCanonical(n_components=n_components, scale=False, max_iter=500)
        try:
            boot_model.fit_transform(
                x_df.iloc[sample_idx].reset_index(drop=True),
                y_df.iloc[sample_idx].reset_index(drop=True),
            )
        except Exception:
            continue

        boot_x = np.asarray(boot_model.x_loadings_, dtype=float)
        boot_y = np.asarray(boot_model.y_loadings_, dtype=float)
        for component in range(n_components):
            corr = np.corrcoef(observed_x[:, component], boot_x[:, component])[0, 1]
            if np.isfinite(corr) and corr < 0:
                boot_x[:, component] *= -1.0
                boot_y[:, component] *= -1.0

        for col_idx, feature_name in enumerate(visual_df.columns):
            for component in range(n_components):
                storage[("visual", feature_name, component + 1)].append(float(boot_x[col_idx, component]))
        for col_idx, feature_name in enumerate(audio_df.columns):
            for component in range(n_components):
                storage[("audio", feature_name, component + 1)].append(float(boot_y[col_idx, component]))

    rows: List[Dict[str, Any]] = []
    for component in range(n_components):
        for col_idx, feature_name in enumerate(visual_df.columns):
            values = np.asarray(storage[("visual", feature_name, component + 1)], dtype=float)
            observed = float(observed_x[col_idx, component])
            rows.append(
                {
                    "modality": "visual",
                    "feature_name": feature_name,
                    "source_group": group_lookup.get(feature_name, ""),
                    "component": int(component + 1),
                    "loading_observed": observed,
                    "loading_mean": float(np.mean(values)) if values.size else np.nan,
                    "loading_std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                    "ci_lower": float(np.quantile(values, 0.025)) if values.size else np.nan,
                    "ci_upper": float(np.quantile(values, 0.975)) if values.size else np.nan,
                    "sign_stability": float(np.mean(np.sign(values) == np.sign(observed))) if values.size else np.nan,
                    "bootstrap_samples_retained": int(values.size),
                }
            )
        for col_idx, feature_name in enumerate(audio_df.columns):
            values = np.asarray(storage[("audio", feature_name, component + 1)], dtype=float)
            observed = float(observed_y[col_idx, component])
            rows.append(
                {
                    "modality": "audio",
                    "feature_name": feature_name,
                    "source_group": group_lookup.get(feature_name, ""),
                    "component": int(component + 1),
                    "loading_observed": observed,
                    "loading_mean": float(np.mean(values)) if values.size else np.nan,
                    "loading_std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                    "ci_lower": float(np.quantile(values, 0.025)) if values.size else np.nan,
                    "ci_upper": float(np.quantile(values, 0.975)) if values.size else np.nan,
                    "sign_stability": float(np.mean(np.sign(values) == np.sign(observed))) if values.size else np.nan,
                    "bootstrap_samples_retained": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def _build_summary_markdown(
    *,
    video_dir: str,
    n_full: int,
    n_thin: int,
    top_links_df: pd.DataFrame,
    stable_links_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    group_df: pd.DataFrame,
) -> str:
    top_links = top_links_df.copy().head(8)
    top_groups = group_df.sort_values("group_effect", ascending=False).head(6)
    significant_latent = latent_df["permutation_p"].lt(0.05).any() if not latent_df.empty else False

    lines: List[str] = [
        "# Relationship Analysis Summary",
        "",
        "## 1) Data sources and sample definition",
        f"- Video output directory: `{video_dir}`",
        "- Inputs: `fusion/model_feature_table.csv`, `fusion/model_feature_dictionary.json`, `soundscape/audio_segment_features.csv`, `segments/segment_manifest.csv`.",
        f"- Full sample: {int(n_full)} aligned segments after conservative feature screening.",
        f"- Thin sensitivity subset: {int(n_thin)} segments, keeping every other segment in chronological order to reduce overlap dependence.",
        "",
        "## 2) Full-sample and thin-sample strategy",
        "- Full-sample pairwise significance used circular block-shift tests on the ordered segment sequence.",
        "- Thin-sample sensitivity repeated the same association analysis on the near-non-overlapping subset.",
        "- Claims below emphasize links that kept the same direction under thinning.",
        "",
        "## 3) Strongest audio-visual links",
    ]
    if top_links.empty:
        lines.append("- No candidate pair remained after conservative screening.")
    else:
        for _, row in top_links.iterrows():
            lines.append(
                "- "
                f"`{row['visual_feature']}` vs `{row['audio_feature']}` | "
                f"rho={float(row['spearman_rho']):.3f}, q={float(row.get('spearman_q', np.nan)):.3f}; "
                f"dCor={float(row['dcor']):.3f}, q={float(row.get('dcor_q', np.nan)):.3f}; "
                f"thin_consistent={bool(row['direction_consistent'])}"
            )

    lines.extend(
        [
            "",
            "## 4) Links stable under thinning",
        ]
    )
    stable_only = stable_links_df[stable_links_df["direction_consistent"]].copy()
    if stable_only.empty:
        lines.append("- No full-sample significant link retained a clearly consistent direction in the thin subset.")
    else:
        for _, row in stable_only.head(8).iterrows():
            lines.append(
                "- "
                f"`{row['visual_feature']}` and `{row['audio_feature']}` kept a "
                f"{row['full_sample_direction']} association in both samples."
            )

    lines.extend(
        [
            "",
            "## 5) Multivariate PLS result",
            f"- Latent cross-modal association significant at q < 0.05: {'yes' if significant_latent else 'no'}",
        ]
    )
    if not latent_df.empty:
        for _, row in latent_df.iterrows():
            lines.append(
                "- "
                f"LV{int(row['component'])}: score correlation={float(row['score_correlation']):.3f}, "
                f"permutation p={float(row['permutation_p']):.3f}"
            )

    lines.extend(
        [
            "",
            "## 6) Group-to-group coupling",
        ]
    )
    if top_groups.empty:
        lines.append("- No group pair had enough stable pairwise evidence for a strong coupling summary.")
    else:
        for _, row in top_groups.iterrows():
            lines.append(
                "- "
                f"{pretty_group_name(str(row['visual_group']))} with {pretty_group_name(str(row['audio_group']))}: "
                f"mean combined effect={float(row['group_effect']):.3f}"
            )

    lines.extend(
        [
            "",
            "## 7) Conservative wording for manuscript use",
            "- The observed audio-visual links should be described as temporally local associations within overlapping 5 s windows, not as fully independent segment-level effects.",
            "- Links that remained directionally stable in the thin subset can be described as robust to a non-overlap sensitivity check.",
            "- PLS latent factors can be described as low-dimensional coupling patterns, with significance judged by block-shift permutation rather than by asymptotic assumptions.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_relationship_package(
    *,
    video_dir: str,
    out_dir: Path,
    model_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    feature_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    prepared = prepare_cross_modal_feature_table(model_df=model_df, feature_dict=feature_dict)
    feature_df = prepared["feature_df"]
    feature_registry = prepared["feature_registry"]
    visual_features = prepared["visual_features"]
    audio_features = prepared["audio_features"]

    ordered_manifest = order_segments(manifest_df=manifest_df, segment_ids=model_df["segment_id"].tolist())
    ordered_ids = ordered_manifest["segment_id"].tolist()

    ordered_feature_df = model_df[["segment_id"]].merge(
        feature_df.assign(segment_id=model_df["segment_id"].to_numpy()),
        on="segment_id",
        how="left",
    )
    ordered_feature_df = ordered_feature_df.set_index("segment_id").loc[ordered_ids].reset_index()

    visual_full = ordered_feature_df.set_index("segment_id")[visual_features]
    audio_full = ordered_feature_df.set_index("segment_id")[audio_features]
    thin_ids = ordered_manifest[ordered_manifest["analysis_order"] % 2 == 0]["segment_id"].tolist()
    visual_thin = visual_full.loc[thin_ids]
    audio_thin = audio_full.loc[thin_ids]

    full_pairwise = _pairwise_association(visual_full, audio_full, block_size=2)
    thin_pairwise = _pairwise_association(visual_thin, audio_thin, block_size=1)

    full_long = full_pairwise["long_df"].copy()
    thin_long = thin_pairwise["long_df"].copy()
    thin_long = thin_long.rename(
        columns={
            "spearman_rho": "thin_spearman_rho",
            "spearman_p": "thin_spearman_p",
            "spearman_q": "thin_spearman_q",
            "dcor": "thin_dcor",
            "dcor_p": "thin_dcor_p",
            "dcor_q": "thin_dcor_q",
            "combined_effect": "thin_combined_effect",
            "n_segments": "thin_n_segments",
            "block_size": "thin_block_size",
            "permutation_scheme": "thin_permutation_scheme",
        }
    )

    group_lookup = feature_registry.set_index("feature_name")["source_group"].to_dict()
    merged_pairs = full_long.merge(
        thin_long,
        on=["visual_feature", "audio_feature"],
        how="left",
    )
    merged_pairs["visual_group"] = merged_pairs["visual_feature"].map(group_lookup).fillna("")
    merged_pairs["audio_group"] = merged_pairs["audio_feature"].map(group_lookup).fillna("")
    merged_pairs["full_sample_direction"] = merged_pairs["spearman_rho"].map(_direction_label)
    merged_pairs["thin_sample_direction"] = merged_pairs["thin_spearman_rho"].map(_direction_label)
    merged_pairs["direction_consistent"] = (
        merged_pairs["full_sample_direction"] == merged_pairs["thin_sample_direction"]
    ) & merged_pairs["full_sample_direction"].isin(["positive", "negative"])
    merged_pairs["thin_still_significant"] = (
        merged_pairs["thin_spearman_q"].lt(0.05) | merged_pairs["thin_dcor_q"].lt(0.05)
    ).fillna(False)
    merged_pairs["significant_full"] = merged_pairs["spearman_q"].lt(0.05) | merged_pairs["dcor_q"].lt(0.05)

    significant_pairs = merged_pairs[merged_pairs["significant_full"]].copy()
    significant_pairs = significant_pairs.sort_values(
        ["combined_effect", "dcor", "spearman_rho"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    significant_pairs["effect_rank"] = np.arange(1, len(significant_pairs) + 1, dtype=int)
    top_links_df = (
        significant_pairs.copy()
        if not significant_pairs.empty
        else merged_pairs.sort_values(["combined_effect", "dcor"], ascending=[False, False]).head(15).copy()
    )
    if "effect_rank" not in top_links_df.columns:
        top_links_df["effect_rank"] = np.arange(1, len(top_links_df) + 1, dtype=int)

    group_source = significant_pairs if not significant_pairs.empty else merged_pairs.nlargest(100, "combined_effect")
    group_summary = (
        group_source.groupby(["visual_group", "audio_group"], dropna=False)["combined_effect"]
        .mean()
        .reset_index(name="group_effect")
        .sort_values("group_effect", ascending=False)
        .reset_index(drop=True)
    )

    pls_visual_features, pls_audio_features = _select_pls_features(
        merged_pairs=merged_pairs,
        feature_df=feature_df,
        feature_registry=feature_registry,
    )
    pls_visual_df = visual_full[pls_visual_features]
    pls_audio_df = audio_full[pls_audio_features]
    n_components = _component_count(len(pls_visual_df), len(pls_visual_features), len(pls_audio_features))
    pls_model, pls_scores = _fit_pls(pls_visual_df, pls_audio_df, n_components=n_components)
    perm_payload = _pls_permutation_test(
        visual_df=pls_visual_df,
        audio_df=pls_audio_df,
        fitted_model=pls_model,
        observed_scores=pls_scores,
        block_size=2,
    )

    latent_df = pls_scores.copy()
    latent_df["visual_feature_count"] = int(len(pls_visual_features))
    latent_df["audio_feature_count"] = int(len(pls_audio_features))
    latent_df["visual_features"] = "|".join(pls_visual_features)
    latent_df["audio_features"] = "|".join(pls_audio_features)
    perm_map = {int(item["component"]): item for item in perm_payload.get("components", [])}
    latent_df["permutation_p"] = latent_df["component"].map(
        lambda x: float(perm_map.get(int(x), {}).get("permutation_p", np.nan))
    )
    latent_df["significant_at_0_05"] = latent_df["permutation_p"].lt(0.05)

    x_loadings = np.asarray(pls_model.x_loadings_, dtype=float)
    y_loadings = np.asarray(pls_model.y_loadings_, dtype=float)
    x_rows: List[Dict[str, Any]] = []
    y_rows: List[Dict[str, Any]] = []
    score_plot_rows: List[Dict[str, Any]] = []
    x_scores, y_scores = pls_model.transform(_standardize_frame(pls_visual_df), _standardize_frame(pls_audio_df))
    for component in range(n_components):
        comp_corr = float(latent_df.loc[latent_df["component"] == component + 1, "score_correlation"].iloc[0])
        comp_p = float(latent_df.loc[latent_df["component"] == component + 1, "permutation_p"].iloc[0])
        for idx, seg_id in enumerate(ordered_ids):
            score_plot_rows.append(
                {
                    "segment_id": int(seg_id),
                    "component": int(component + 1),
                    "x_score": float(x_scores[idx, component]),
                    "y_score": float(y_scores[idx, component]),
                    "score_correlation": comp_corr,
                    "permutation_p": comp_p,
                }
            )
        for row_idx, feature_name in enumerate(pls_visual_features):
            x_rows.append(
                {
                    "modality": "visual",
                    "feature_name": feature_name,
                    "source_group": group_lookup.get(feature_name, ""),
                    "component": int(component + 1),
                    "loading": float(x_loadings[row_idx, component]),
                }
            )
        for row_idx, feature_name in enumerate(pls_audio_features):
            y_rows.append(
                {
                    "modality": "audio",
                    "feature_name": feature_name,
                    "source_group": group_lookup.get(feature_name, ""),
                    "component": int(component + 1),
                    "loading": float(y_loadings[row_idx, component]),
                }
            )
    x_loadings_df = pd.DataFrame(x_rows)
    y_loadings_df = pd.DataFrame(y_rows)
    loadings_plot_df = pd.concat([x_loadings_df, y_loadings_df], axis=0, ignore_index=True)

    bootstrap_df = _bootstrap_pls_stability(
        visual_df=pls_visual_df,
        audio_df=pls_audio_df,
        fitted_model=pls_model,
        block_size=2,
        n_bootstrap=300,
        seed=20260311,
        group_lookup=group_lookup,
    )

    figures_dir = out_dir / "figures"
    plot_cross_modal_heatmap(group_summary, figures_dir / "fig1_cross_modal_heatmap")
    plot_top_links_lollipop(top_links_df, figures_dir / "fig2_top_links_lollipop", top_n=15)
    plot_pls_scores(pd.DataFrame(score_plot_rows), figures_dir / "fig3_pls_scores")
    plot_pls_loadings(loadings_plot_df, figures_dir / "fig4_pls_loadings", top_n=8)

    feature_registry.to_csv(out_dir / "feature_registry.csv", index=False, encoding="utf-8")
    full_pairwise["spearman_matrix"].to_csv(out_dir / "feature_correlation_matrix_spearman.csv", encoding="utf-8")
    full_pairwise["dcor_matrix"].to_csv(out_dir / "feature_correlation_matrix_dcor.csv", encoding="utf-8")
    merged_pairs[
        [
            "visual_feature",
            "audio_feature",
            "spearman_p",
            "dcor_p",
            "thin_spearman_p",
            "thin_dcor_p",
            "n_segments",
            "thin_n_segments",
        ]
    ].to_csv(out_dir / "feature_correlation_pvalues.csv", index=False, encoding="utf-8")
    merged_pairs[
        [
            "visual_feature",
            "audio_feature",
            "spearman_q",
            "dcor_q",
            "thin_spearman_q",
            "thin_dcor_q",
        ]
    ].to_csv(out_dir / "feature_correlation_qvalues.csv", index=False, encoding="utf-8")
    significant_pairs.to_csv(out_dir / "significant_pairwise_links.csv", index=False, encoding="utf-8")
    latent_df.to_csv(out_dir / "pls_latent_summary.csv", index=False, encoding="utf-8")
    x_loadings_df.to_csv(out_dir / "pls_x_loadings.csv", index=False, encoding="utf-8")
    y_loadings_df.to_csv(out_dir / "pls_y_loadings.csv", index=False, encoding="utf-8")
    bootstrap_df.to_csv(out_dir / "pls_bootstrap_stability.csv", index=False, encoding="utf-8")
    write_json(out_dir / "pls_permutation_test.json", perm_payload)

    summary_text = _build_summary_markdown(
        video_dir=video_dir,
        n_full=len(ordered_ids),
        n_thin=len(thin_ids),
        top_links_df=top_links_df,
        stable_links_df=significant_pairs,
        latent_df=latent_df,
        group_df=group_summary,
    )
    (out_dir / "relationship_summary.md").write_text(summary_text, encoding="utf-8")

    logger.info(
        "relationship done | out=%s full_n=%s thin_n=%s visual=%s audio=%s",
        out_dir.as_posix(),
        len(ordered_ids),
        len(thin_ids),
        len(visual_features),
        len(audio_features),
    )
    return {
        "video_dir": str(video_dir),
        "relationship_outdir": out_dir.as_posix(),
        "feature_registry_csv": (out_dir / "feature_registry.csv").as_posix(),
        "feature_correlation_matrix_spearman_csv": (out_dir / "feature_correlation_matrix_spearman.csv").as_posix(),
        "feature_correlation_matrix_dcor_csv": (out_dir / "feature_correlation_matrix_dcor.csv").as_posix(),
        "feature_correlation_pvalues_csv": (out_dir / "feature_correlation_pvalues.csv").as_posix(),
        "feature_correlation_qvalues_csv": (out_dir / "feature_correlation_qvalues.csv").as_posix(),
        "significant_pairwise_links_csv": (out_dir / "significant_pairwise_links.csv").as_posix(),
        "pls_latent_summary_csv": (out_dir / "pls_latent_summary.csv").as_posix(),
        "pls_x_loadings_csv": (out_dir / "pls_x_loadings.csv").as_posix(),
        "pls_y_loadings_csv": (out_dir / "pls_y_loadings.csv").as_posix(),
        "pls_bootstrap_stability_csv": (out_dir / "pls_bootstrap_stability.csv").as_posix(),
        "pls_permutation_test_json": (out_dir / "pls_permutation_test.json").as_posix(),
        "relationship_summary_md": (out_dir / "relationship_summary.md").as_posix(),
        "figures_dir": figures_dir.as_posix(),
        "full_sample_n": int(len(ordered_ids)),
        "thin_sample_n": int(len(thin_ids)),
        "visual_feature_count": int(len(visual_features)),
        "audio_feature_count": int(len(audio_features)),
        "audio_source_rows": int(len(audio_df)),
        "manifest_rows": int(len(manifest_df)),
        "strongest_link_count": int(len(significant_pairs)),
        "example_pls_visual_features": [short_feature_label(x) for x in pls_visual_features[:5]],
        "example_pls_audio_features": [short_feature_label(x) for x in pls_audio_features[:5]],
    }
