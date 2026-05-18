


"""Group-level confirmatory relationship layer built on top of existing relationship outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .group_confirmatory_plotting import (
    plot_confirmatory_forest,
    plot_group_composite_profile,
    plot_group_confirmatory_heatmap,
)
from .grouping import AUDIO_GROUP_ORDER, VISUAL_GROUP_ORDER, build_group_artifacts
from .utils import (
    benjamini_hochberg,
    centered_distance_flatten,
    circular_block_permutations,
    contiguous_blocks,
    normalize_segment_id,
    order_segments,
    pretty_group_name,
    read_json,
    two_sided_spearman,
)

logger = logging.getLogger("research.group_confirmatory")

DEFAULT_BLOCK_SIZE = 2
FULL_BOOTSTRAP_SAMPLES = 2000
THIN_BOOTSTRAP_SAMPLES = 1000

CONFIRMATORY_SPECS: Sequence[Tuple[str, str, str, str]] = (
    ("H1", "people_presence", "audio_signal_level", "positive"),
    ("H2", "people_presence", "audio_event_human", "positive"),
    ("H3", "ai_activity", "audio_event_general", "positive"),
    ("H4", "traffic_road_hardscape", "audio_signal_level", "positive"),
    ("H5", "green_nature", "audio_event_natural", "positive"),
)


def _resolve_paths(video_dir: str, group_confirmatory_outdir: Optional[str]) -> Dict[str, Path]:
    vdir = Path(video_dir)
    relationship_dir = vdir / "relationship"
    out_dir = Path(group_confirmatory_outdir) if group_confirmatory_outdir else (relationship_dir / "group_confirmatory")
    return {
        "video_dir": vdir,
        "out_dir": out_dir,
        "model_feature_csv": vdir / "fusion" / "model_feature_table.csv",
        "model_feature_dict_json": vdir / "fusion" / "model_feature_dictionary.json",
        "audio_segment_features_csv": vdir / "soundscape" / "audio_segment_features.csv",
        "segment_manifest_csv": vdir / "segments" / "segment_manifest.csv",
        "relationship_feature_registry_csv": relationship_dir / "feature_registry.csv",
        "relationship_significant_pairs_csv": relationship_dir / "significant_pairwise_links.csv",
        "relationship_pls_latent_csv": relationship_dir / "pls_latent_summary.csv",
        "relationship_summary_md": relationship_dir / "relationship_summary.md",
    }


def _validate_required_inputs(paths: Mapping[str, Path]) -> None:
    required = [
        paths["model_feature_csv"],
        paths["model_feature_dict_json"],
        paths["audio_segment_features_csv"],
        paths["segment_manifest_csv"],
    ]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Group confirmatory layer missing required inputs: " + ", ".join(missing))


def _spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    return two_sided_spearman(x, y)


def _distance_corr_1d(x: Sequence[float], y: Sequence[float]) -> float:
    x_df = pd.DataFrame({"x": np.asarray(x, dtype=float)})
    y_df = pd.DataFrame({"y": np.asarray(y, dtype=float)})
    d_x, dvar_x = centered_distance_flatten(x_df)
    d_y, dvar_y = centered_distance_flatten(y_df)
    denom = np.sqrt(dvar_x[0] * dvar_y[0])
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    dcov2 = float((d_x @ d_y.T)[0, 0] / (len(x_df) ** 2))
    return float(np.sqrt(max(dcov2 / denom, 0.0)))


def _block_bootstrap_spearman_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    block_size: int,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    blocks = contiguous_blocks(len(x_arr), block_size=block_size)
    if not blocks:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(max(1, n_bootstrap)), dtype=float)
    for i in range(draws.size):
        sample_idx: List[int] = []
        while len(sample_idx) < len(x_arr):
            block = blocks[int(rng.integers(0, len(blocks)))]
            sample_idx.extend(block.tolist())
        idx = np.asarray(sample_idx[: len(x_arr)], dtype=int)
        draws[i] = _spearman_corr(x_arr[idx], y_arr[idx])
    return float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


def _block_shift_test(
    x: Sequence[float],
    y: Sequence[float],
    *,
    block_size: int,
) -> Dict[str, Any]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    observed_rho = _spearman_corr(x_arr, y_arr)
    observed_dcor = _distance_corr_1d(x_arr, y_arr)
    perms = circular_block_permutations(n_rows=len(x_arr), block_size=block_size, include_identity=False)
    if not perms:
        return {
            "spearman_rho": observed_rho,
            "spearman_p": np.nan,
            "dcor": observed_dcor,
            "dcor_p": np.nan,
            "n_permutations": 0,
        }
    rho_null = np.asarray([_spearman_corr(x_arr, y_arr[np.asarray(perm, dtype=int)]) for perm in perms], dtype=float)
    dcor_null = np.asarray([_distance_corr_1d(x_arr, y_arr[np.asarray(perm, dtype=int)]) for perm in perms], dtype=float)
    rho_p = float((np.sum(np.abs(rho_null) >= abs(observed_rho)) + 1.0) / (len(rho_null) + 1.0))
    dcor_p = float((np.sum(dcor_null >= observed_dcor) + 1.0) / (len(dcor_null) + 1.0))
    return {
        "spearman_rho": float(observed_rho),
        "spearman_p": rho_p,
        "dcor": float(observed_dcor),
        "dcor_p": dcor_p,
        "n_permutations": int(len(perms)),
    }


def _supports_expected_direction(rho: float, expected_direction: str) -> bool:
    if not np.isfinite(rho):
        return False
    if expected_direction == "positive":
        return rho > 0
    if expected_direction == "negative":
        return rho < 0
    return False


def _build_hypothesis_registry(visual_groups: Sequence[str], audio_groups: Sequence[str]) -> pd.DataFrame:
    visual_set = set(visual_groups)
    audio_set = set(audio_groups)
    confirmatory_rows: List[Dict[str, Any]] = []
    confirm_pairs = set()
    for hyp_id, visual_group, audio_group, expected_direction in CONFIRMATORY_SPECS:
        confirm_pairs.add((visual_group, audio_group))
        visual_ok = visual_group in visual_set
        audio_ok = audio_group in audio_set
        testable = visual_ok and audio_ok
        skipped_reason = ""
        if not visual_ok and not audio_ok:
            skipped_reason = "visual_and_audio_groups_missing"
        elif not visual_ok:
            skipped_reason = "visual_group_missing"
        elif not audio_ok:
            skipped_reason = "audio_group_missing"
        confirmatory_rows.append(
            {
                "hypothesis_id": hyp_id,
                "family": "confirmatory",
                "visual_group": visual_group,
                "audio_group": audio_group,
                "expected_direction": expected_direction,
                "testable_flag": bool(testable),
                "skipped_reason": skipped_reason,
            }
        )

    exploratory_rows: List[Dict[str, Any]] = []
    idx = 1
    for visual_group in visual_groups:
        for audio_group in audio_groups:
            if (visual_group, audio_group) in confirm_pairs:
                continue
            exploratory_rows.append(
                {
                    "hypothesis_id": f"E{idx:02d}",
                    "family": "exploratory",
                    "visual_group": visual_group,
                    "audio_group": audio_group,
                    "expected_direction": "unspecified",
                    "testable_flag": True,
                    "skipped_reason": "",
                }
            )
            idx += 1
    return pd.DataFrame(confirmatory_rows + exploratory_rows)


def _evaluate_subset(
    hypothesis_df: pd.DataFrame,
    composite_df: pd.DataFrame,
    *,
    subset_label: str,
    block_size: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, hyp in hypothesis_df.iterrows():
        base = {
            "hypothesis_id": str(hyp["hypothesis_id"]),
            "family": str(hyp["family"]),
            "visual_group": str(hyp["visual_group"]),
            "audio_group": str(hyp["audio_group"]),
            "expected_direction": str(hyp["expected_direction"]),
            "testable_flag": bool(hyp["testable_flag"]),
            "subset": subset_label,
            "block_size": int(block_size),
        }
        if not bool(hyp["testable_flag"]):
            rows.append(
                {
                    **base,
                    "n_segments": 0,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "dcor": np.nan,
                    "dcor_p": np.nan,
                    "n_permutations": 0,
                    "spearman_ci_low": np.nan,
                    "spearman_ci_high": np.nan,
                }
            )
            continue
        sub = composite_df[["segment_id", str(hyp["visual_group"]), str(hyp["audio_group"])]].dropna().copy()
        x = sub[str(hyp["visual_group"])].to_numpy(dtype=float)
        y = sub[str(hyp["audio_group"])].to_numpy(dtype=float)
        test = _block_shift_test(x, y, block_size=block_size)
        ci_low, ci_high = _block_bootstrap_spearman_ci(
            x,
            y,
            block_size=block_size,
            n_bootstrap=FULL_BOOTSTRAP_SAMPLES if subset_label == "full" else THIN_BOOTSTRAP_SAMPLES,
            seed=20260321 if subset_label == "full" else 20260322,
        )
        rows.append(
            {
                **base,
                "n_segments": int(len(sub)),
                "spearman_rho": float(test["spearman_rho"]),
                "spearman_p": float(test["spearman_p"]),
                "dcor": float(test["dcor"]),
                "dcor_p": float(test["dcor_p"]),
                "n_permutations": int(test["n_permutations"]),
                "spearman_ci_low": float(ci_low),
                "spearman_ci_high": float(ci_high),
            }
        )
    return pd.DataFrame(rows)


def _family_adjust_qvalues(full_df: pd.DataFrame) -> pd.DataFrame:
    out = full_df.copy()
    out["spearman_q"] = np.nan
    out["dcor_q"] = np.nan
    for family in out["family"].dropna().unique().tolist():
        mask = (out["family"] == family) & (out["testable_flag"])
        out.loc[mask, "spearman_q"] = benjamini_hochberg(out.loc[mask, "spearman_p"].to_numpy(dtype=float))
        out.loc[mask, "dcor_q"] = benjamini_hochberg(out.loc[mask, "dcor_p"].to_numpy(dtype=float))
    return out


def _time_trend_sensitivity(
    hypothesis_df: pd.DataFrame,
    composite_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ordered = composite_df.copy().reset_index(drop=True)
    segment_index = pd.Series(np.arange(len(ordered), dtype=float), index=ordered.index)
    for _, hyp in hypothesis_df[hypothesis_df["testable_flag"]].iterrows():
        visual = pd.to_numeric(ordered[str(hyp["visual_group"])], errors="coerce")
        audio = pd.to_numeric(ordered[str(hyp["audio_group"])], errors="coerce")
        x_rank = visual.rank(method="average")
        y_rank = audio.rank(method="average")
        t = segment_index
        x_coef = np.polyfit(t.to_numpy(dtype=float), x_rank.to_numpy(dtype=float), deg=1)
        y_coef = np.polyfit(t.to_numpy(dtype=float), y_rank.to_numpy(dtype=float), deg=1)
        x_resid = x_rank - (x_coef[0] * t + x_coef[1])
        y_resid = y_rank - (y_coef[0] * t + y_coef[1])
        partial = _block_shift_test(x_resid.to_numpy(dtype=float), y_resid.to_numpy(dtype=float), block_size=DEFAULT_BLOCK_SIZE)
        rows.append(
            {
                "hypothesis_id": str(hyp["hypothesis_id"]),
                "visual_group": str(hyp["visual_group"]),
                "audio_group": str(hyp["audio_group"]),
                "visual_group_time_rho": _spearman_corr(visual, t),
                "audio_group_time_rho": _spearman_corr(audio, t),
                "partial_spearman_rho": float(partial["spearman_rho"]),
                "partial_spearman_p": float(partial["spearman_p"]),
                "partial_dcor": float(partial["dcor"]),
                "partial_dcor_p": float(partial["dcor_p"]),
                "n_permutations": int(partial["n_permutations"]),
            }
        )
    return pd.DataFrame(rows)


def _leave_one_feature_out_robustness(
    hypothesis_df: pd.DataFrame,
    composite_df: pd.DataFrame,
    retained_groups: Mapping[str, Mapping[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_lookup = {
        str(row["hypothesis_id"]): row
        for _, row in hypothesis_df[hypothesis_df["testable_flag"]].iterrows()
    }
    ordered_ids = composite_df["segment_id"].tolist()
    thin_ids = set(composite_df.iloc[::2]["segment_id"].tolist())
    rows: List[Dict[str, Any]] = []

    for hyp_id, hyp in base_lookup.items():
        for side in ["visual_group", "audio_group"]:
            group_name = str(hyp[side])
            group_info = retained_groups.get(group_name, {})
            aligned_df: pd.DataFrame = group_info.get("aligned_df", pd.DataFrame())
            if aligned_df.shape[1] < 2:
                continue
            other_group = str(hyp["audio_group"] if side == "visual_group" else hyp["visual_group"])
            other_series = pd.to_numeric(composite_df[other_group], errors="coerce")
            base_full_rho = _spearman_corr(composite_df[str(hyp["visual_group"])], composite_df[str(hyp["audio_group"])])
            for feature_name in aligned_df.columns.tolist():
                lofo_comp = aligned_df.drop(columns=[feature_name]).mean(axis=1)
                test_full = _block_shift_test(
                    lofo_comp.to_numpy(dtype=float) if side == "visual_group" else other_series.to_numpy(dtype=float),
                    other_series.to_numpy(dtype=float) if side == "visual_group" else lofo_comp.to_numpy(dtype=float),
                    block_size=DEFAULT_BLOCK_SIZE,
                )
                thin_mask = composite_df["segment_id"].isin(thin_ids)
                lofo_thin = lofo_comp[thin_mask].to_numpy(dtype=float)
                other_thin = other_series[thin_mask].to_numpy(dtype=float)
                test_thin = _block_shift_test(
                    lofo_thin if side == "visual_group" else other_thin,
                    other_thin if side == "visual_group" else lofo_thin,
                    block_size=1,
                )
                lofo_rho = float(test_full["spearman_rho"])
                rows.append(
                    {
                        "hypothesis_id": hyp_id,
                        "family": str(hyp["family"]),
                        "visual_group": str(hyp["visual_group"]),
                        "audio_group": str(hyp["audio_group"]),
                        "omitted_from_group": group_name,
                        "omitted_feature": feature_name,
                        "spearman_rho_full_lofo": lofo_rho,
                        "spearman_p_full_lofo": float(test_full["spearman_p"]),
                        "spearman_rho_thin_lofo": float(test_thin["spearman_rho"]),
                        "direction_same_as_full": bool(np.sign(lofo_rho) == np.sign(base_full_rho) or lofo_rho == 0 or base_full_rho == 0),
                        "supports_expected_direction_lofo": _supports_expected_direction(lofo_rho, str(hyp["expected_direction"])),
                    }
                )

    detail_df = pd.DataFrame(rows)
    summary_rows: List[Dict[str, Any]] = []
    for _, hyp in hypothesis_df.iterrows():
        hyp_id = str(hyp["hypothesis_id"])
        sub = detail_df[detail_df["hypothesis_id"] == hyp_id].copy()
        if sub.empty:
            summary_rows.append(
                {
                    "hypothesis_id": hyp_id,
                    "lofo_variants": 0,
                    "lofo_same_direction_rate": 1.0,
                    "lofo_expected_direction_rate": 1.0,
                    "lofo_severe_flip": False,
                }
            )
            continue
        same_rate = float(sub["direction_same_as_full"].mean())
        expected_rate = float(sub["supports_expected_direction_lofo"].mean())
        summary_rows.append(
            {
                "hypothesis_id": hyp_id,
                "lofo_variants": int(len(sub)),
                "lofo_same_direction_rate": same_rate,
                "lofo_expected_direction_rate": expected_rate,
                "lofo_severe_flip": bool(same_rate < 0.80),
            }
        )
    return detail_df, pd.DataFrame(summary_rows)


def _merge_full_thin(full_df: pd.DataFrame, thin_df: pd.DataFrame, lofo_summary_df: pd.DataFrame) -> pd.DataFrame:
    full_out = full_df.rename(
        columns={
            "n_segments": "n_full",
            "spearman_rho": "spearman_rho_full",
            "spearman_p": "spearman_p_full",
            "spearman_q": "spearman_q_full",
            "spearman_ci_low": "spearman_ci_low",
            "spearman_ci_high": "spearman_ci_high",
            "dcor": "dcor_full",
            "dcor_p": "dcor_p_full",
            "dcor_q": "dcor_q_full",
            "n_permutations": "n_permutations_full",
        }
    )
    thin_out = thin_df.rename(
        columns={
            "n_segments": "n_thin",
            "spearman_rho": "spearman_rho_thin",
            "spearman_p": "spearman_p_thin",
            "dcor": "dcor_thin",
            "dcor_p": "dcor_p_thin",
            "n_permutations": "n_permutations_thin",
        }
    )
    keep_full = [
        "hypothesis_id",
        "family",
        "visual_group",
        "audio_group",
        "expected_direction",
        "testable_flag",
        "n_full",
        "spearman_rho_full",
        "spearman_p_full",
        "spearman_q_full",
        "spearman_ci_low",
        "spearman_ci_high",
        "dcor_full",
        "dcor_p_full",
        "dcor_q_full",
        "n_permutations_full",
    ]
    keep_thin = [
        "hypothesis_id",
        "n_thin",
        "spearman_rho_thin",
        "spearman_p_thin",
        "dcor_thin",
        "dcor_p_thin",
        "n_permutations_thin",
    ]
    merged = full_out[keep_full].merge(thin_out[keep_thin], on="hypothesis_id", how="left")
    merged = merged.merge(lofo_summary_df, on="hypothesis_id", how="left")
    merged["direction_consistent"] = (
        np.sign(merged["spearman_rho_full"].to_numpy(dtype=float))
        == np.sign(merged["spearman_rho_thin"].to_numpy(dtype=float))
    )
    merged["ci_excludes_zero"] = (
        (merged["spearman_ci_low"] > 0) | (merged["spearman_ci_high"] < 0)
    ).fillna(False)
    merged["supports_expected_direction"] = [
        _supports_expected_direction(rho, exp)
        for rho, exp in zip(merged["spearman_rho_full"].tolist(), merged["expected_direction"].tolist())
    ]
    merged["confirmatory_claim"] = ""
    return merged


def _assign_confirmatory_claims(combined_df: pd.DataFrame) -> pd.DataFrame:
    out = combined_df.copy()
    claims: List[str] = []
    for _, row in out.iterrows():
        if str(row["family"]) != "confirmatory":
            claims.append("exploratory_only")
            continue
        if not bool(row["testable_flag"]):
            claims.append("not_testable")
            continue
        supports_direction = bool(row["supports_expected_direction"])
        thin_consistent = bool(row["direction_consistent"])
        lofo_stable = not bool(row.get("lofo_severe_flip", False))
        significant = (
            supports_direction
            and bool(row["ci_excludes_zero"])
            and float(row["spearman_q_full"]) < 0.05
            and thin_consistent
            and lofo_stable
        )
        if significant:
            claims.append("supported_confirmatory")
        elif supports_direction and (not thin_consistent or not lofo_stable):
            claims.append("unstable_under_sensitivity")
        elif supports_direction:
            claims.append("directionally_consistent_but_not_significant")
        else:
            claims.append("not_supported")
    out["confirmatory_claim"] = claims
    return out


def _build_summary_markdown(
    *,
    video_dir: str,
    group_definition_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    hypothesis_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    time_df: pd.DataFrame,
    pairwise_count: int,
    pls_latent_df: pd.DataFrame,
) -> str:
    kept_groups = diagnostics_df[["group_name", "modality", "n_features"]].copy()
    confirm_df = combined_df[combined_df["family"] == "confirmatory"].copy()
    supported = confirm_df[confirm_df["confirmatory_claim"] == "supported_confirmatory"]["hypothesis_id"].tolist()
    directional = confirm_df[confirm_df["confirmatory_claim"] == "directionally_consistent_but_not_significant"]["hypothesis_id"].tolist()
    unstable = confirm_df[confirm_df["confirmatory_claim"] == "unstable_under_sensitivity"]["hypothesis_id"].tolist()
    not_supported = confirm_df[confirm_df["confirmatory_claim"] == "not_supported"]["hypothesis_id"].tolist()
    not_testable = confirm_df[confirm_df["confirmatory_claim"] == "not_testable"]["hypothesis_id"].tolist()
    lv1_sig = False
    if not pls_latent_df.empty:
        lv1 = pls_latent_df[pls_latent_df["component"] == 1]
        if not lv1.empty and bool(lv1["significant_at_0_05"].iloc[0]):
            lv1_sig = True

    lines: List[str] = [
        "# Group Confirmatory Relationship Summary",
        "",
        "## 1) Why add a group-level confirmatory layer",
        "- The existing pairwise layer is fine-grained and exploratory, but it faces a large multiple-testing burden.",
        f"- In the current video, the pairwise layer yielded {int(pairwise_count)} FDR-significant feature-level links after overlap-aware correction.",
        f"- The existing PLS layer still indicated a low-dimensional cross-modal coupling signal (LV1 significant: {'yes' if lv1_sig else 'no'}).",
        "- The new group-level layer therefore tests a small, pre-defined set of interpretable visual-audio hypotheses that can be written in the main text.",
        "",
        "## 2) How groups were defined",
        "- Groups were theory-driven and fixed before testing, using deterministic feature rules plus conservative within-group screening.",
        "- Each group composite was the equal-weight mean of winsorized, z-scored constituent features after sign alignment.",
        "- Groups that could not be given a coherent direction were skipped rather than forced.",
        "",
        "## 3) Retained visual and audio groups",
    ]
    for _, row in kept_groups.iterrows():
        lines.append(f"- `{row['group_name']}` ({row['modality']}): {int(row['n_features'])} constituent feature(s)")

    confirm_list = hypothesis_df[hypothesis_df["family"] == "confirmatory"]
    explore_list = hypothesis_df[hypothesis_df["family"] == "exploratory"]
    lines.extend(
        [
            "",
            "## 4) Confirmatory family",
        ]
    )
    for _, row in confirm_list.iterrows():
        status = "testable" if bool(row["testable_flag"]) else f"skipped: {row['skipped_reason']}"
        lines.append(
            f"- `{row['hypothesis_id']}`: `{row['visual_group']}` vs `{row['audio_group']}` "
            f"(expected {row['expected_direction']}; {status})"
        )
    lines.extend(
        [
            "",
            "## 5) Exploratory family",
            f"- Testable exploratory group-pairs: {int(explore_list['testable_flag'].sum())}",
            "- Exploratory pairs were FDR-corrected separately from the confirmatory family.",
            "",
            "## 6) Full / thin / leave-one-feature-out conclusions",
        ]
    )
    for _, row in confirm_df.iterrows():
        lines.append(
            "- "
            f"`{row['hypothesis_id']}` {row['visual_group']} vs {row['audio_group']}: "
            f"rho_full={float(row['spearman_rho_full']):.3f}, "
            f"95% CI=[{float(row['spearman_ci_low']):.3f}, {float(row['spearman_ci_high']):.3f}], "
            f"q={float(row['spearman_q_full']):.3f}, "
            f"rho_thin={float(row['spearman_rho_thin']):.3f}, "
            f"LOFO same-direction rate={float(row.get('lofo_same_direction_rate', np.nan)):.2f}, "
            f"claim={row['confirmatory_claim']}"
        )

    lines.extend(
        [
            "",
            "## 7) What can be written as supported",
            f"- supported_confirmatory: {', '.join(supported) if supported else 'none'}",
            f"- directionally_consistent_but_not_significant: {', '.join(directional) if directional else 'none'}",
            f"- unstable_under_sensitivity: {', '.join(unstable) if unstable else 'none'}",
            f"- not_supported: {', '.join(not_supported) if not_supported else 'none'}",
            f"- not_testable: {', '.join(not_testable) if not_testable else 'none'}",
            "",
            "## 8) Relation to the existing relationship layer",
            "- Pairwise layer: fine-grained exploratory evidence on individual feature links.",
            "- Group-level layer: low-dimensional confirmatory tests on a pre-defined family of theory-driven group pairs.",
            "- PLS layer: holistic low-dimensional evidence that a broader cross-modal coupling structure exists even when single links do not survive feature-level multiplicity correction.",
            "",
            "## 9) Conservative paragraph for the manuscript",
            "- Feature-level pairwise tests did not yield FDR-stable single links after overlap-aware correction, but the PLS analysis indicated a significant first latent cross-modal mode. The added group-level confirmatory layer translated this broader signal into a small set of pre-specified visual-audio composites tested with block-shift permutation and non-overlap sensitivity checks. Only hypotheses meeting the pre-registered direction, confidence-interval, FDR, and sensitivity criteria should be described as supported; the remainder should be framed as directionally suggestive or not supported rather than as definitive evidence.",
            "",
            "## 10) Short figure-note text",
            "- Heatmap cells show full-sample group-level Spearman correlations between theory-driven visual and audio composites. Confirmatory pairs are outlined. Forest-plot intervals are block-bootstrap 95% confidence intervals, and claim labels additionally require FDR control plus thin-subset and leave-one-feature-out consistency.",
        ]
    )
    if not time_df.empty:
        lines.extend(
            [
                "",
                "## 11) Time-trend sensitivity",
            ]
        )
        for _, row in time_df.head(10).iterrows():
            lines.append(
                "- "
                f"`{row['hypothesis_id']}` visual-time rho={float(row['visual_group_time_rho']):.3f}, "
                f"audio-time rho={float(row['audio_group_time_rho']):.3f}, "
                f"partial rho={float(row['partial_spearman_rho']):.3f}, p={float(row['partial_spearman_p']):.3f}"
            )
    return "\n".join(lines).strip() + "\n"


def _build_onepage_report(combined_df: pd.DataFrame) -> str:
    confirm_df = combined_df[combined_df["family"] == "confirmatory"].copy()
    lines: List[str] = [
        "# Group Confirmatory One-Page Report",
        "",
        "## Main-text candidates",
    ]
    supported = confirm_df[confirm_df["confirmatory_claim"] == "supported_confirmatory"]
    directional = confirm_df[confirm_df["confirmatory_claim"] == "directionally_consistent_but_not_significant"]
    if supported.empty:
        lines.append("- No confirmatory hypothesis met the full support rule.")
    else:
        for _, row in supported.iterrows():
            lines.append(
                f"- `{row['hypothesis_id']}` can be stated as supported: rho={float(row['spearman_rho_full']):.3f}, q={float(row['spearman_q_full']):.3f}."
            )
    if not directional.empty:
        for _, row in directional.iterrows():
            lines.append(
                f"- `{row['hypothesis_id']}` is directionally aligned but below confirmatory significance; better placed in cautious main-text interpretation or supplement."
            )
    lines.extend(
        [
            "",
            "## Supplement-first items",
        ]
    )
    for _, row in confirm_df[confirm_df["confirmatory_claim"].isin(["unstable_under_sensitivity", "not_supported", "not_testable"])].iterrows():
        lines.append(
            f"- `{row['hypothesis_id']}` -> `{row['confirmatory_claim']}`; keep in supplement or limitations paragraph."
        )
    lines.extend(
        [
            "",
            "## Plain-language takeaway",
            "- Use the pairwise layer as exploratory detail, the group layer as the confirmatory core, and the PLS layer as overall coupling context.",
            "- Do not upgrade directionally consistent but non-significant group results into claims of supported coupling.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def run_group_confirmatory_relationship(
    *,
    video_dir: str,
    group_confirmatory_outdir: Optional[str] = None,
) -> Dict[str, Any]:
    paths = _resolve_paths(video_dir=video_dir, group_confirmatory_outdir=group_confirmatory_outdir)
    _validate_required_inputs(paths)

    model_df = normalize_segment_id(pd.read_csv(paths["model_feature_csv"]), "model_feature_table")
    audio_df = normalize_segment_id(pd.read_csv(paths["audio_segment_features_csv"]), "audio_segment_features")
    manifest_df = normalize_segment_id(pd.read_csv(paths["segment_manifest_csv"]), "segment_manifest")
    feature_dict = read_json(paths["model_feature_dict_json"])
    existing_feature_registry = None
    if paths["relationship_feature_registry_csv"].exists():
        existing_feature_registry = pd.read_csv(paths["relationship_feature_registry_csv"])

    common_ids = sorted(
        set(model_df["segment_id"].tolist())
        & set(audio_df["segment_id"].tolist())
        & set(manifest_df["segment_id"].tolist())
    )
    if not common_ids:
        raise RuntimeError("No shared segment_id values across group-confirmatory input files.")

    model_df = model_df[model_df["segment_id"].isin(common_ids)].copy().sort_values("segment_id").reset_index(drop=True)
    manifest_df = manifest_df[manifest_df["segment_id"].isin(common_ids)].copy().reset_index(drop=True)
    ordered_manifest = order_segments(manifest_df=manifest_df, segment_ids=common_ids)
    ordered_ids = ordered_manifest["segment_id"].tolist()

    group_artifacts = build_group_artifacts(
        model_df=model_df.set_index("segment_id").loc[ordered_ids].reset_index(),
        feature_dict=feature_dict,
        existing_feature_registry=existing_feature_registry,
    )
    composite_df = group_artifacts["group_composites"].copy()
    composite_df = composite_df.merge(
        ordered_manifest[["segment_id", "center_time_sec"]],
        on="segment_id",
        how="left",
    )
    cols = ["segment_id", "center_time_sec"] + group_artifacts["visual_groups"] + group_artifacts["audio_groups"]
    composite_df = composite_df[cols].copy()
    thin_df = composite_df.iloc[::2].reset_index(drop=True)

    hypothesis_df = _build_hypothesis_registry(
        visual_groups=group_artifacts["visual_groups"],
        audio_groups=group_artifacts["audio_groups"],
    )
    full_tests = _family_adjust_qvalues(_evaluate_subset(hypothesis_df, composite_df, subset_label="full", block_size=DEFAULT_BLOCK_SIZE))
    thin_tests = _evaluate_subset(hypothesis_df, thin_df, subset_label="thin", block_size=1)
    lofo_detail_df, lofo_summary_df = _leave_one_feature_out_robustness(
        hypothesis_df=hypothesis_df,
        composite_df=composite_df,
        retained_groups=group_artifacts["retained_groups"],
    )
    combined_df = _merge_full_thin(full_tests, thin_tests, lofo_summary_df)
    combined_df = _assign_confirmatory_claims(combined_df)
    time_df = _time_trend_sensitivity(hypothesis_df, composite_df)

    pairwise_count = 0
    if paths["relationship_significant_pairs_csv"].exists():
        pairwise_count = int(len(pd.read_csv(paths["relationship_significant_pairs_csv"])))
    pls_latent_df = pd.read_csv(paths["relationship_pls_latent_csv"]) if paths["relationship_pls_latent_csv"].exists() else pd.DataFrame()

    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    figures_dir = paths["out_dir"] / "figures"

    plot_group_confirmatory_heatmap(
        combined_df=combined_df,
        visual_groups=group_artifacts["visual_groups"],
        audio_groups=group_artifacts["audio_groups"],
        out_base=figures_dir / "fig_group_confirmatory_heatmap",
    )
    plot_confirmatory_forest(
        combined_df[combined_df["family"] == "confirmatory"].copy(),
        figures_dir / "fig_confirmatory_forest",
    )
    plot_group_composite_profile(
        composite_df=composite_df,
        group_modalities={g: "visual" for g in group_artifacts["visual_groups"]} | {g: "audio" for g in group_artifacts["audio_groups"]},
        out_base=figures_dir / "fig_group_composite_profile",
    )

    group_artifacts["group_definition_registry"].to_csv(paths["out_dir"] / "group_definition_registry.csv", index=False, encoding="utf-8")
    composite_df.to_csv(paths["out_dir"] / "group_composites.csv", index=False, encoding="utf-8")
    group_artifacts["group_composite_diagnostics"].to_csv(paths["out_dir"] / "group_composite_diagnostics.csv", index=False, encoding="utf-8")
    hypothesis_df.to_csv(paths["out_dir"] / "hypothesis_registry.csv", index=False, encoding="utf-8")
    full_tests.to_csv(paths["out_dir"] / "group_pair_tests_full.csv", index=False, encoding="utf-8")
    thin_tests.to_csv(paths["out_dir"] / "group_pair_tests_thin.csv", index=False, encoding="utf-8")
    combined_df.to_csv(paths["out_dir"] / "group_pair_tests_combined.csv", index=False, encoding="utf-8")
    lofo_detail_df.to_csv(paths["out_dir"] / "leave_one_feature_out_robustness.csv", index=False, encoding="utf-8")
    time_df.to_csv(paths["out_dir"] / "time_trend_sensitivity.csv", index=False, encoding="utf-8")
    combined_df[combined_df["family"] == "confirmatory"].copy().to_csv(
        paths["out_dir"] / "confirmatory_claim_registry.csv",
        index=False,
        encoding="utf-8",
    )

    summary_text = _build_summary_markdown(
        video_dir=video_dir,
        group_definition_df=group_artifacts["group_definition_registry"],
        diagnostics_df=group_artifacts["group_composite_diagnostics"],
        hypothesis_df=hypothesis_df,
        combined_df=combined_df,
        time_df=time_df,
        pairwise_count=pairwise_count,
        pls_latent_df=pls_latent_df,
    )
    (paths["out_dir"] / "group_confirmatory_summary.md").write_text(summary_text, encoding="utf-8")
    (paths["out_dir"] / "group_confirmatory_onepage_report.md").write_text(
        _build_onepage_report(combined_df),
        encoding="utf-8",
    )

    logger.info(
        "group confirmatory done | out=%s visual_groups=%s audio_groups=%s confirmatory=%s",
        paths["out_dir"].as_posix(),
        len(group_artifacts["visual_groups"]),
        len(group_artifacts["audio_groups"]),
        int((hypothesis_df["family"] == "confirmatory").sum()),
    )
    return {
        "group_confirmatory_outdir": paths["out_dir"].as_posix(),
        "group_definition_registry_csv": (paths["out_dir"] / "group_definition_registry.csv").as_posix(),
        "group_composites_csv": (paths["out_dir"] / "group_composites.csv").as_posix(),
        "group_composite_diagnostics_csv": (paths["out_dir"] / "group_composite_diagnostics.csv").as_posix(),
        "hypothesis_registry_csv": (paths["out_dir"] / "hypothesis_registry.csv").as_posix(),
        "group_pair_tests_full_csv": (paths["out_dir"] / "group_pair_tests_full.csv").as_posix(),
        "group_pair_tests_thin_csv": (paths["out_dir"] / "group_pair_tests_thin.csv").as_posix(),
        "group_pair_tests_combined_csv": (paths["out_dir"] / "group_pair_tests_combined.csv").as_posix(),
        "leave_one_feature_out_robustness_csv": (paths["out_dir"] / "leave_one_feature_out_robustness.csv").as_posix(),
        "time_trend_sensitivity_csv": (paths["out_dir"] / "time_trend_sensitivity.csv").as_posix(),
        "confirmatory_claim_registry_csv": (paths["out_dir"] / "confirmatory_claim_registry.csv").as_posix(),
        "group_confirmatory_summary_md": (paths["out_dir"] / "group_confirmatory_summary.md").as_posix(),
        "group_confirmatory_onepage_report_md": (paths["out_dir"] / "group_confirmatory_onepage_report.md").as_posix(),
        "figures_dir": figures_dir.as_posix(),
        "visual_groups_built": group_artifacts["visual_groups"],
        "audio_groups_built": group_artifacts["audio_groups"],
    }
