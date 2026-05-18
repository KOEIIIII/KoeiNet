


"""Proof package built directly on Step 7.5 refined outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .plotting import plot_fusion_delta_forest, plot_oof_performance_by_target
from .utils import (
    benjamini_hochberg,
    block_bootstrap_ci,
    block_sign_flip_test,
    pretty_model_name,
    write_json,
)

logger = logging.getLogger("research.proof")

FUSION_MODEL = "early_fusion_screened"
BASELINE_COMPARISONS = (
    ("fusion_vs_visual_only", "visual_only_screened"),
    ("fusion_vs_audio_only", "audio_only_screened"),
)


def _multiclass_brier(y_true: pd.Series, proba_df: pd.DataFrame) -> float:
    class_names = [c.replace("proba__", "") for c in proba_df.columns]
    target = np.zeros((len(y_true), len(class_names)), dtype=float)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    for row_idx, label in enumerate(y_true.astype(str).tolist()):
        if label in class_to_idx:
            target[row_idx, class_to_idx[label]] = 1.0
    diff = proba_df.to_numpy(dtype=float) - target
    return float(np.mean(np.sum(diff * diff, axis=1)))


def _aggregate_oof_predictions(oof_df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [c for c in oof_df.columns if c.startswith("proba__")]
    id_cols = ["segment_id", "target_name", "target_type", "target_tier", "model_group"]
    rows: List[Dict[str, Any]] = []

    for keys, sub in oof_df.groupby(id_cols, dropna=False):
        segment_id, target_name, target_type, target_tier, model_group = keys
        row: Dict[str, Any] = {
            "segment_id": int(segment_id),
            "target_name": str(target_name),
            "target_type": str(target_type),
            "target_tier": str(target_tier),
            "model_group": str(model_group),
            "n_oof_rows": int(len(sub)),
        }
        if str(target_type) == "classification":
            probs = sub[prob_cols].mean(axis=0)
            best_col = probs.idxmax() if not probs.empty else ""
            row["y_true"] = str(sub["y_true"].iloc[0])
            row["y_pred"] = str(best_col.replace("proba__", "")) if best_col else ""
            for col in prob_cols:
                row[col] = float(probs.get(col, np.nan))
        else:
            row["y_true"] = float(pd.to_numeric(sub["y_true"], errors="coerce").mean())
            row["y_pred"] = float(pd.to_numeric(sub["y_pred"], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_spearman(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.nunique(dropna=True) <= 1 or y_pred.nunique(dropna=True) <= 1:
        return np.nan
    return float(y_true.corr(y_pred, method="spearman"))


def _performance_rows(agg_df: pd.DataFrame, thin_ids: List[int]) -> pd.DataFrame:
    prob_cols = [c for c in agg_df.columns if c.startswith("proba__")]
    rows: List[Dict[str, Any]] = []

    for (target_name, target_type, target_tier, model_group), sub_all in agg_df.groupby(
        ["target_name", "target_type", "target_tier", "model_group"],
        dropna=False,
    ):
        for subset_name, subset_ids in [("full", None), ("thin", set(thin_ids))]:
            sub = sub_all if subset_ids is None else sub_all[sub_all["segment_id"].isin(subset_ids)]
            if sub.empty:
                continue
            row: Dict[str, Any] = {
                "target_name": str(target_name),
                "target_type": str(target_type),
                "target_tier": str(target_tier),
                "model_group": str(model_group),
                "model_label": pretty_model_name(str(model_group)),
                "subset": subset_name,
                "n_segments": int(len(sub)),
            }
            if str(target_type) == "classification":
                proba = sub[prob_cols].copy()
                row["primary_metric"] = "brier"
                row["primary_metric_label"] = "Brier (lower is better)"
                row["primary_value"] = _multiclass_brier(sub["y_true"], proba)
                row["brier"] = row["primary_value"]
                row["balanced_accuracy"] = float(balanced_accuracy_score(sub["y_true"], sub["y_pred"]))
                try:
                    row["roc_auc_ovr"] = float(
                        roc_auc_score(
                            sub["y_true"],
                            proba.to_numpy(dtype=float),
                            labels=[c.replace("proba__", "") for c in prob_cols],
                            multi_class="ovr",
                            average="macro",
                        )
                    )
                except Exception:
                    row["roc_auc_ovr"] = np.nan
            else:
                y_true = pd.to_numeric(sub["y_true"], errors="coerce")
                y_pred = pd.to_numeric(sub["y_pred"], errors="coerce")
                row["primary_metric"] = "mae"
                row["primary_metric_label"] = "MAE (lower is better)"
                row["primary_value"] = float(np.mean(np.abs(y_true - y_pred)))
                row["mae"] = row["primary_value"]
                row["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                row["spearman"] = _safe_spearman(y_true, y_pred)
            rows.append(row)
    return pd.DataFrame(rows)


def _segment_level_losses(agg_df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [c for c in agg_df.columns if c.startswith("proba__")]
    rows: List[Dict[str, Any]] = []
    for _, row in agg_df.iterrows():
        out = {
            "segment_id": int(row["segment_id"]),
            "target_name": str(row["target_name"]),
            "target_type": str(row["target_type"]),
            "target_tier": str(row["target_tier"]),
            "model_group": str(row["model_group"]),
            "y_true": row["y_true"],
            "y_pred": row["y_pred"],
        }
        if str(row["target_type"]) == "classification":
            proba = pd.DataFrame([row[prob_cols].to_dict()])
            out["primary_metric"] = "brier"
            out["primary_loss"] = _multiclass_brier(pd.Series([row["y_true"]]), proba)
        else:
            y_true = float(row["y_true"])
            y_pred = float(row["y_pred"])
            out["primary_metric"] = "mae"
            out["primary_loss"] = abs(y_true - y_pred)
            out["squared_error"] = (y_true - y_pred) ** 2
        rows.append(out)
    return pd.DataFrame(rows)


def _build_paired_delta_rows(loss_df: pd.DataFrame, manifest_df: pd.DataFrame, thin_ids: List[int]) -> pd.DataFrame:
    order_lookup = manifest_df.set_index("segment_id")["center_time_sec"].to_dict() if "center_time_sec" in manifest_df.columns else {}
    rows: List[Dict[str, Any]] = []
    for comparison, baseline_model in BASELINE_COMPARISONS:
        fusion = loss_df[loss_df["model_group"] == FUSION_MODEL].copy()
        baseline = loss_df[loss_df["model_group"] == baseline_model].copy()
        merged = fusion.merge(
            baseline,
            on=["segment_id", "target_name", "target_type", "target_tier", "primary_metric"],
            how="inner",
            suffixes=("_fusion", "_baseline"),
        )
        if merged.empty:
            continue
        for _, row in merged.iterrows():
            subset_names = ["full"]
            if int(row["segment_id"]) in set(thin_ids):
                subset_names.append("thin")
            for subset_name in subset_names:
                rows.append(
                    {
                        "segment_id": int(row["segment_id"]),
                        "target_name": str(row["target_name"]),
                        "target_type": str(row["target_type"]),
                        "target_tier": str(row["target_tier"]),
                        "comparison": comparison,
                        "comparison_label": comparison.replace("_", " "),
                        "fusion_model": FUSION_MODEL,
                        "baseline_model": baseline_model,
                        "primary_metric": str(row["primary_metric"]),
                        "fusion_loss": float(row["primary_loss_fusion"]),
                        "baseline_loss": float(row["primary_loss_baseline"]),
                        "delta_loss": float(row["primary_loss_fusion"] - row["primary_loss_baseline"]),
                        "center_time_sec": float(order_lookup.get(int(row["segment_id"]), np.nan)),
                        "subset": subset_name,
                    }
                )
    out = pd.DataFrame(rows)
    return out.sort_values(["target_name", "comparison", "subset", "center_time_sec", "segment_id"]).reset_index(drop=True)


def _summarize_inference(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, sub in delta_df.groupby(["target_name", "target_type", "target_tier", "comparison", "subset"], dropna=False):
        target_name, target_type, target_tier, comparison, subset = keys
        ordered = sub.sort_values(["center_time_sec", "segment_id"]).reset_index(drop=True)
        block_size = 2 if str(subset) == "full" else 1
        ci = block_bootstrap_ci(
            ordered["delta_loss"].to_numpy(dtype=float),
            block_size=block_size,
            n_bootstrap=2000,
            seed=20260311,
            alpha=0.95,
        )
        test = block_sign_flip_test(
            ordered["delta_loss"].to_numpy(dtype=float),
            block_size=block_size,
            n_permutations=4000,
            seed=20260312,
        )
        delta_mean = float(np.mean(ordered["delta_loss"].to_numpy(dtype=float)))
        rows.append(
            {
                "target_name": str(target_name),
                "target_type": str(target_type),
                "target_tier": str(target_tier),
                "comparison": str(comparison),
                "comparison_label": str(comparison).replace("_", " "),
                "subset": str(subset),
                "primary_metric": str(ordered["primary_metric"].iloc[0]),
                "n_segments": int(len(ordered)),
                "block_size": int(block_size),
                "delta_mean": delta_mean,
                "ci_lower": float(ci["ci_lower"]),
                "ci_upper": float(ci["ci_upper"]),
                "p_value": float(test["p_value"]),
                "better_direction": "fusion_better" if delta_mean < 0 else ("baseline_better" if delta_mean > 0 else "tie"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = np.nan
    for subset_name in out["subset"].dropna().unique().tolist():
        for tier_name in out["target_tier"].dropna().unique().tolist():
            mask = (out["subset"] == subset_name) & (out["target_tier"] == tier_name)
            out.loc[mask, "q_value"] = benjamini_hochberg(out.loc[mask, "p_value"].to_numpy(dtype=float))
    return out.sort_values(["target_tier", "target_name", "comparison", "subset"]).reset_index(drop=True)


def _claim_label_for_target(full_df: pd.DataFrame, thin_df: pd.DataFrame) -> Tuple[str, str]:
    support_both = []
    any_fusion_better_point = False
    any_strong_baseline = False
    details: List[str] = []

    for comparison in ["fusion_vs_visual_only", "fusion_vs_audio_only"]:
        full_row = full_df[full_df["comparison"] == comparison]
        thin_row = thin_df[thin_df["comparison"] == comparison]
        if full_row.empty:
            continue
        full_item = full_row.iloc[0]
        thin_item = thin_row.iloc[0] if not thin_row.empty else None
        point_better = float(full_item["delta_mean"]) < 0
        strong_support = (
            point_better
            and float(full_item["ci_upper"]) < 0
            and float(full_item["p_value"]) < 0.05
            and float(full_item["q_value"]) < 0.05
            and thin_item is not None
            and float(thin_item["delta_mean"]) < 0
        )
        baseline_stronger = (
            float(full_item["delta_mean"]) > 0
            and float(full_item["ci_lower"]) > 0
            and float(full_item["p_value"]) < 0.05
            and float(full_item["q_value"]) < 0.05
            and thin_item is not None
            and float(thin_item["delta_mean"]) > 0
        )
        any_fusion_better_point = any_fusion_better_point or point_better
        any_strong_baseline = any_strong_baseline or baseline_stronger
        support_both.append(bool(strong_support))
        thin_text = f"thin_delta={float(thin_item['delta_mean']):.4f}" if thin_item is not None else "thin_delta=NA"
        details.append(
            f"{comparison}: delta={float(full_item['delta_mean']):.4f}, "
            f"CI=[{float(full_item['ci_lower']):.4f}, {float(full_item['ci_upper']):.4f}], "
            f"p={float(full_item['p_value']):.4f}, q={float(full_item['q_value']):.4f}, "
            f"{thin_text}"
        )

    if support_both and all(support_both):
        return "fusion_superior_to_both", " | ".join(details)
    if any_strong_baseline:
        return "single_modality_stronger", " | ".join(details)
    if any_fusion_better_point:
        return "fusion_complementary_but_not_superior", " | ".join(details)
    return "no_evidence_fusion_better", " | ".join(details)


def _build_claim_registry(stats_df: pd.DataFrame) -> pd.DataFrame:
    full_df = stats_df[stats_df["subset"] == "full"].copy()
    thin_df = stats_df[stats_df["subset"] == "thin"].copy()
    rows: List[Dict[str, Any]] = []
    for target_name in full_df["target_name"].dropna().unique().tolist():
        full_target = full_df[full_df["target_name"] == target_name].copy()
        thin_target = thin_df[thin_df["target_name"] == target_name].copy()
        if full_target.empty:
            continue
        label, rationale = _claim_label_for_target(full_target, thin_target)
        visual_row = full_target[full_target["comparison"] == "fusion_vs_visual_only"]
        audio_row = full_target[full_target["comparison"] == "fusion_vs_audio_only"]
        thin_visual = thin_target[thin_target["comparison"] == "fusion_vs_visual_only"]
        thin_audio = thin_target[thin_target["comparison"] == "fusion_vs_audio_only"]
        first = full_target.iloc[0]
        visual_full_item = visual_row.iloc[0] if not visual_row.empty else None
        audio_full_item = audio_row.iloc[0] if not audio_row.empty else None
        visual_thin_item = thin_visual.iloc[0] if not thin_visual.empty else None
        audio_thin_item = thin_audio.iloc[0] if not thin_audio.empty else None
        rows.append(
            {
                "target_name": target_name,
                "target_type": str(first["target_type"]),
                "target_tier": str(first["target_tier"]),
                "primary_metric": str(first["primary_metric"]),
                "claim_label": label,
                "fusion_vs_visual_delta": float(visual_full_item["delta_mean"]) if visual_full_item is not None else np.nan,
                "fusion_vs_visual_ci_lower": float(visual_full_item["ci_lower"]) if visual_full_item is not None else np.nan,
                "fusion_vs_visual_ci_upper": float(visual_full_item["ci_upper"]) if visual_full_item is not None else np.nan,
                "fusion_vs_visual_p": float(visual_full_item["p_value"]) if visual_full_item is not None else np.nan,
                "fusion_vs_visual_q": float(visual_full_item["q_value"]) if visual_full_item is not None else np.nan,
                "fusion_vs_visual_thin_delta": float(visual_thin_item["delta_mean"]) if visual_thin_item is not None else np.nan,
                "fusion_vs_visual_thin_direction_consistent": bool(
                    visual_full_item is not None
                    and visual_thin_item is not None
                    and np.sign(float(visual_full_item["delta_mean"])) == np.sign(float(visual_thin_item["delta_mean"]))
                ),
                "fusion_vs_audio_delta": float(audio_full_item["delta_mean"]) if audio_full_item is not None else np.nan,
                "fusion_vs_audio_ci_lower": float(audio_full_item["ci_lower"]) if audio_full_item is not None else np.nan,
                "fusion_vs_audio_ci_upper": float(audio_full_item["ci_upper"]) if audio_full_item is not None else np.nan,
                "fusion_vs_audio_p": float(audio_full_item["p_value"]) if audio_full_item is not None else np.nan,
                "fusion_vs_audio_q": float(audio_full_item["q_value"]) if audio_full_item is not None else np.nan,
                "fusion_vs_audio_thin_delta": float(audio_thin_item["delta_mean"]) if audio_thin_item is not None else np.nan,
                "fusion_vs_audio_thin_direction_consistent": bool(
                    audio_full_item is not None
                    and audio_thin_item is not None
                    and np.sign(float(audio_full_item["delta_mean"])) == np.sign(float(audio_thin_item["delta_mean"]))
                ),
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows).sort_values(["target_tier", "target_name"]).reset_index(drop=True)


def _build_summary_markdown(
    *,
    video_dir: str,
    model_perf_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    claim_df: pd.DataFrame,
) -> str:
    lines: List[str] = [
        "# Proof Package Summary",
        "",
        "## 1) Evidence base",
        f"- Video output directory: `{video_dir}`",
        "- Primary evidence layer: `fusion_eval_refined/oof_predictions_refined.csv`, aggregated to one segment-level OOF prediction per model and target.",
        "- Supporting Step 7.5 refined artifacts: `per_target_metrics_refined.csv`, `model_comparison_refined.csv`, `paired_deltas_refined.csv`, `bootstrap_ci_refined.json`, `target_registry_refined.json`, `step75_summary.md`.",
        "- The proof package does not retrain models and does not replace Step 7.5; it only re-expresses Step 7.5 evidence in a stricter paired framework.",
        "",
    ]

    section_no = 2
    for tier_name in ["confirmatory", "exploratory"]:
        tier_claims = claim_df[claim_df["target_tier"] == tier_name].copy()
        tier_stats = stats_df[(stats_df["target_tier"] == tier_name) & (stats_df["subset"] == "full")].copy()
        lines.append(f"## {section_no}) {tier_name.title()} targets")
        if tier_claims.empty:
            lines.append("- none")
            lines.append("")
            section_no += 2
            continue
        for _, claim in tier_claims.iterrows():
            lines.append(
                "- "
                f"`{claim['target_name']}` -> `{claim['claim_label']}` "
                f"(fusion-vs-visual delta={float(claim['fusion_vs_visual_delta']):.4f}, "
                f"fusion-vs-audio delta={float(claim['fusion_vs_audio_delta']):.4f})"
            )
        lines.append("")
        lines.append(f"## {section_no + 1}) {tier_name.title()} paired full-vs-thin comparison")
        for target_name in tier_claims["target_name"].tolist():
            sub_full = tier_stats[tier_stats["target_name"] == target_name].copy()
            sub_thin = stats_df[
                (stats_df["target_name"] == target_name)
                & (stats_df["target_tier"] == tier_name)
                & (stats_df["subset"] == "thin")
            ].copy()
            for comparison in ["fusion_vs_visual_only", "fusion_vs_audio_only"]:
                frow = sub_full[sub_full["comparison"] == comparison]
                trow = sub_thin[sub_thin["comparison"] == comparison]
                if frow.empty:
                    continue
                fitem = frow.iloc[0]
                if not trow.empty:
                    titem = trow.iloc[0]
                    thin_text = f"thin delta={float(titem['delta_mean']):.4f}, p={float(titem['p_value']):.4f}"
                else:
                    thin_text = "thin delta=NA"
                lines.append(
                    "- "
                    f"`{target_name}` {comparison}: full delta={float(fitem['delta_mean']):.4f}, "
                    f"95% CI=[{float(fitem['ci_lower']):.4f}, {float(fitem['ci_upper']):.4f}], "
                    f"p={float(fitem['p_value']):.4f}, q={float(fitem['q_value']):.4f}; {thin_text}"
                )
        lines.append("")
        section_no += 2

    lines.extend(
        [
            f"## {section_no}) Writing guidance",
        ]
    )
    superior = claim_df[claim_df["claim_label"] == "fusion_superior_to_both"]["target_name"].tolist()
    complementary = claim_df[claim_df["claim_label"] == "fusion_complementary_but_not_superior"]["target_name"].tolist()
    no_evidence = claim_df[claim_df["claim_label"] == "no_evidence_fusion_better"]["target_name"].tolist()
    single_stronger = claim_df[claim_df["claim_label"] == "single_modality_stronger"]["target_name"].tolist()
    lines.append(f"- Can write `fusion superior to both single modalities`: {', '.join(superior) if superior else 'none'}.")
    lines.append(f"- Can write `fusion provides complementary value`: {', '.join(complementary) if complementary else 'none'}.")
    lines.append(f"- Should not write `fusion better`: {', '.join(no_evidence) if no_evidence else 'none'}.")
    lines.append(f"- Single-modality stronger cases: {', '.join(single_stronger) if single_stronger else 'none'}.")
    lines.extend(
        [
            "",
            f"## {section_no + 1}) Conservative manuscript paragraph",
            "- Across targets, fusion should only be described as superior when paired segment-level OOF loss differences favored fusion against both baselines, the 95% block-bootstrap interval excluded zero, the block sign-flip test remained significant after within-tier FDR control, and the thin non-overlap sensitivity analysis preserved the same direction. In all other cases, the safer wording is that fusion may offer complementary information but not uniformly better predictive performance.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_proof_package(
    *,
    video_dir: str,
    out_dir: Path,
    oof_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
) -> Dict[str, Any]:
    agg_df = _aggregate_oof_predictions(oof_df)
    ordered_manifest = manifest_df.copy()
    sort_cols = [c for c in ["center_time_sec", "start_time_sec", "segment_id"] if c in ordered_manifest.columns]
    ordered_manifest = ordered_manifest.sort_values(sort_cols).reset_index(drop=True)
    thin_ids = ordered_manifest.loc[ordered_manifest.index % 2 == 0, "segment_id"].astype(int).tolist()

    model_perf_df = _performance_rows(agg_df, thin_ids=thin_ids)
    loss_df = _segment_level_losses(agg_df)
    paired_deltas_df = _build_paired_delta_rows(loss_df=loss_df, manifest_df=ordered_manifest, thin_ids=thin_ids)
    stats_df = _summarize_inference(paired_deltas_df)
    claim_df = _build_claim_registry(stats_df)

    bootstrap_df = stats_df[
        [
            "target_name",
            "target_type",
            "target_tier",
            "comparison",
            "comparison_label",
            "subset",
            "primary_metric",
            "n_segments",
            "block_size",
            "delta_mean",
            "ci_lower",
            "ci_upper",
        ]
    ].copy()
    permutation_df = stats_df[
        [
            "target_name",
            "target_type",
            "target_tier",
            "comparison",
            "comparison_label",
            "subset",
            "primary_metric",
            "n_segments",
            "block_size",
            "delta_mean",
            "p_value",
            "q_value",
            "better_direction",
        ]
    ].copy()

    figures_dir = out_dir / "figures"
    plot_fusion_delta_forest(stats_df[stats_df["subset"] == "full"].copy(), figures_dir / "fig5_fusion_delta_forest")
    plot_oof_performance_by_target(model_perf_df[model_perf_df["subset"] == "full"].copy(), figures_dir / "fig6_oof_performance_by_target")

    model_perf_df.to_csv(out_dir / "proof_model_comparison.csv", index=False, encoding="utf-8")
    paired_deltas_df.to_csv(out_dir / "proof_paired_deltas.csv", index=False, encoding="utf-8")
    bootstrap_df.to_csv(out_dir / "proof_bootstrap_ci.csv", index=False, encoding="utf-8")
    permutation_df.to_csv(out_dir / "proof_permutation_tests.csv", index=False, encoding="utf-8")
    claim_df.to_csv(out_dir / "proof_claim_registry.csv", index=False, encoding="utf-8")

    summary_text = _build_summary_markdown(
        video_dir=video_dir,
        model_perf_df=model_perf_df,
        stats_df=stats_df,
        claim_df=claim_df,
    )
    (out_dir / "proof_summary.md").write_text(summary_text, encoding="utf-8")
    write_json(
        out_dir / "proof_package_manifest.json",
        {
            "video_dir": video_dir,
            "full_sample_segments": int(len(ordered_manifest)),
            "thin_sample_segments": int(len(thin_ids)),
            "targets": sorted(claim_df["target_name"].tolist()),
        },
    )

    logger.info(
        "proof package done | out=%s targets=%s",
        out_dir.as_posix(),
        len(claim_df),
    )
    return {
        "video_dir": str(video_dir),
        "proof_outdir": out_dir.as_posix(),
        "proof_model_comparison_csv": (out_dir / "proof_model_comparison.csv").as_posix(),
        "proof_paired_deltas_csv": (out_dir / "proof_paired_deltas.csv").as_posix(),
        "proof_bootstrap_ci_csv": (out_dir / "proof_bootstrap_ci.csv").as_posix(),
        "proof_permutation_tests_csv": (out_dir / "proof_permutation_tests.csv").as_posix(),
        "proof_claim_registry_csv": (out_dir / "proof_claim_registry.csv").as_posix(),
        "proof_summary_md": (out_dir / "proof_summary.md").as_posix(),
        "figures_dir": figures_dir.as_posix(),
        "full_sample_n": int(len(ordered_manifest)),
        "thin_sample_n": int(len(thin_ids)),
    }
