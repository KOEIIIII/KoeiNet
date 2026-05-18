


"""Step-8 evidence registry built from Step-7.5 refined evaluation outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

logger = logging.getLogger("design.design_evidence")


CONFIRMATORY_POLICY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "comfort_score": {
        "role": "primary_design_driver",
        "modality_relevance": {
            "audio": "high",
            "visual": "moderate",
            "early_fusion": "supporting",
        },
        "policy_summary": (
            "Treat soundscape and audio evidence as highly relevant for comfort-oriented "
            "interventions. Visual conditions still matter, but comfort must not be treated "
            "as visual-only."
        ),
        "design_guidance": [
            "Traffic-noise-heavy, rough, mechanical, or unpleasant sound conditions should strongly influence comfort-oriented interventions.",
            "Visual buffering, edge definition, vegetation, curbside reorganization, and maintenance/order should be paired with acoustic relief logic.",
            "Soundscape evidence is a hard input, not a decorative afterthought.",
        ],
    },
    "vitality_score": {
        "role": "primary_design_driver",
        "modality_relevance": {
            "audio": "high",
            "visual": "high",
            "early_fusion": "high",
        },
        "policy_summary": (
            "Treat vitality as the most clearly multimodal design target. Use both visible "
            "activity/supportive frontage and audible human-scale liveliness as design clues."
        ),
        "design_guidance": [
            "Do not rely on visual cues alone when vitality is the main design target.",
            "Use both stay/support infrastructure and positive social sound presence to infer activation potential.",
            "Prefer human-scale activity support over generic beautification.",
        ],
    },
    "soundscape_eventfulness": {
        "role": "primary_design_driver",
        "modality_relevance": {
            "audio": "contextual",
            "visual": "high",
            "early_fusion": "contextual",
        },
        "policy_summary": (
            "Acknowledge that visual-only may remain strongest for eventfulness in the current "
            "sample. Use eventfulness-related design cautiously and contextually."
        ),
        "design_guidance": [
            "Do not assume that adding more sound activity is always beneficial.",
            "Distinguish beneficial liveliness from chaotic or traffic-dominated noise.",
            "Use street-type-aware eventfulness calibration rather than generic activation.",
        ],
    },
}


EXPLORATORY_POLICY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "safety_score": {
        "role": "supporting_only",
        "policy_summary": "Use as supporting evidence only, not as the sole design driver.",
    },
    "overall_problem_severity": {
        "role": "supporting_only",
        "policy_summary": "Use as supporting evidence only, not as the sole design driver.",
    },
    "soundscape_pleasantness": {
        "role": "supporting_only",
        "policy_summary": "Use as supporting evidence only, not as the sole design driver.",
    },
    "primary_problem_label": {
        "role": "supporting_only",
        "policy_summary": "Use as supporting evidence only; do not let the label override confirmatory target logic.",
    },
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _metric_value(row: Mapping[str, Any]) -> Optional[float]:
    try:
        value = row.get("primary_value")
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _top_importance_rows(
    permutation_df: pd.DataFrame,
    feature_metadata: Mapping[str, Any],
    target_name: str,
    model_group: str,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    subset = permutation_df[
        (permutation_df["target_name"] == target_name)
        & (permutation_df["model_group"] == model_group)
    ].copy()
    if subset.empty:
        return []

    subset = subset.sort_values(
        by=["importance_mean", "importance_std"],
        ascending=[False, True],
    ).head(int(top_k))

    rows: List[Dict[str, Any]] = []
    for _, item in subset.iterrows():
        feature = str(item["feature"])
        meta = feature_metadata.get(feature, {}) if isinstance(feature_metadata, Mapping) else {}
        rows.append(
            {
                "feature": feature,
                "source_group": str(meta.get("source_group", "unknown")),
                "importance_mean": float(item.get("importance_mean", 0.0)),
                "importance_std": float(item.get("importance_std", 0.0)),
                "metric_name": str(item.get("metric_name", "")),
                "description": str(meta.get("description", "")),
            }
        )
    return rows


def _ranking_snapshot(model_comparison_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if model_comparison_df.empty:
        return out

    for target_name, group_df in model_comparison_df.groupby("target_name", dropna=False):
        ranked = group_df.sort_values("rank", ascending=True)
        out[str(target_name)] = []
        for _, row in ranked.iterrows():
            metric_value = _metric_value(row)
            out[str(target_name)].append(
                {
                    "model_group": str(row.get("model_group", "")),
                    "primary_metric": str(row.get("primary_metric", "")),
                    "primary_value": metric_value,
                    "rank": int(row.get("rank", 0)),
                    "direction": str(row.get("direction", "")),
                }
            )
    return out


def _infer_target_specific_note(target_name: str, ranking: List[Dict[str, Any]]) -> str:
    if not ranking:
        return ""
    best_group = str(ranking[0].get("model_group", ""))
    ranking_names = [str(item.get("model_group", "")) for item in ranking]

    if target_name == "comfort_score":
        return (
            "Observed Step-7.5 ranking indicates that audio evidence is highly consequential "
            f"for comfort (best model: {best_group}). Comfort should therefore remain explicitly "
            "soundscape-aware."
        )
    if target_name == "vitality_score":
        early_rank = ranking_names.index("early_fusion_screened") + 1 if "early_fusion_screened" in ranking_names else None
        visual_rank = ranking_names.index("visual_only_screened") + 1 if "visual_only_screened" in ranking_names else None
        complementarity = early_rank is not None and visual_rank is not None and early_rank < visual_rank
        if complementarity:
            return (
                "Observed Step-7.5 ranking suggests multimodal complementarity for vitality: "
                "early fusion outperforms visual-only, so design should consider both visible and audible human-scale activation."
            )
        return (
            "Vitality should still be treated as multimodal. Even when audio performs strongly, "
            "visual support and active-use cues remain relevant to intervention planning."
        )
    if target_name == "soundscape_eventfulness":
        return (
            "Observed Step-7.5 ranking indicates that visual-only remains strongest for eventfulness "
            f"(best model: {best_group}). Eventfulness design should therefore be context-led rather than 'more sound equals better'."
        )
    return ""


def build_step8_evidence_registry(
    *,
    video_dir: str,
    model_comparison_csv: str,
    per_target_metrics_csv: str,
    permutation_importance_csv: str,
    step75_summary_md: str,
    feature_group_registry_json: str,
    model_feature_dictionary_json: str,
) -> Dict[str, Any]:
    """
    Build the explicit Step-8 evidence policy registry from Step-7.5 artifacts.

    The registry intentionally codifies target-specific evidence policy rather than
    assuming that multimodal fusion is uniformly superior.
    """
    model_comparison_df = pd.read_csv(model_comparison_csv)
    per_target_metrics_df = pd.read_csv(per_target_metrics_csv)
    permutation_df = pd.read_csv(permutation_importance_csv)

    feature_group_registry = _read_json(Path(feature_group_registry_json))
    feature_dictionary = _read_json(Path(model_feature_dictionary_json))
    feature_metadata = feature_dictionary.get("feature_metadata", {})

    ranking_map = _ranking_snapshot(model_comparison_df)
    per_target_metric_rows = per_target_metrics_df.to_dict(orient="records")

    confirmatory_targets: Dict[str, Any] = {}
    for target_name, template in CONFIRMATORY_POLICY_TEMPLATES.items():
        ranking = ranking_map.get(target_name, [])
        best_model_group = str(ranking[0]["model_group"]) if ranking else ""
        confirmatory_targets[target_name] = {
            "target_name": target_name,
            "role": template["role"],
            "modality_relevance": dict(template["modality_relevance"]),
            "policy_summary": template["policy_summary"],
            "design_guidance": list(template["design_guidance"]),
            "observed_model_ranking": ranking,
            "observed_best_model_group": best_model_group,
            "observed_note": _infer_target_specific_note(target_name, ranking),
            "best_model_top_features": _top_importance_rows(
                permutation_df,
                feature_metadata=feature_metadata,
                target_name=target_name,
                model_group=best_model_group or "early_fusion_screened",
                top_k=8,
            ),
            "early_fusion_top_features": _top_importance_rows(
                permutation_df,
                feature_metadata=feature_metadata,
                target_name=target_name,
                model_group="early_fusion_screened",
                top_k=8,
            ),
        }

    exploratory_targets: Dict[str, Any] = {}
    for target_name, template in EXPLORATORY_POLICY_TEMPLATES.items():
        exploratory_targets[target_name] = {
            "target_name": target_name,
            "role": template["role"],
            "policy_summary": template["policy_summary"],
            "observed_model_ranking": ranking_map.get(target_name, []),
        }

    registry: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_dir": Path(video_dir).as_posix(),
        "final_model_evidence_layer": "step75_refined",
        "soundscape_is_hard_input": True,
        "fusion_universally_superior_assumption": False,
        "source_files": {
            "model_comparison_refined_csv": Path(model_comparison_csv).as_posix(),
            "per_target_metrics_refined_csv": Path(per_target_metrics_csv).as_posix(),
            "permutation_importance_csv": Path(permutation_importance_csv).as_posix(),
            "step75_summary_md": Path(step75_summary_md).as_posix(),
            "feature_group_registry_refined_json": Path(feature_group_registry_json).as_posix(),
            "model_feature_dictionary_json": Path(model_feature_dictionary_json).as_posix(),
        },
        "global_policy": {
            "statement": (
                "Step-8 uses Step-7.5 as the final model-evidence layer. Fusion is not assumed "
                "to be universally superior; design logic is target-specific and evidence-weighted."
            ),
            "confirmatory_targets": [
                "comfort_score",
                "vitality_score",
                "soundscape_eventfulness",
            ],
            "exploratory_targets_supporting_only": [
                "safety_score",
                "overall_problem_severity",
                "soundscape_pleasantness",
                "primary_problem_label",
            ],
        },
        "confirmatory_targets": confirmatory_targets,
        "exploratory_targets": exploratory_targets,
        "ranking_snapshot": ranking_map,
        "per_target_metrics_snapshot": per_target_metric_rows,
        "feature_group_registry_snapshot": {
            "feature_counts": feature_group_registry.get("feature_counts", {}),
            "groups": feature_group_registry.get("groups", {}),
        },
    }
    logger.info(
        "step8 evidence registry built | confirmatory_targets=%s exploratory_targets=%s",
        len(confirmatory_targets),
        len(exploratory_targets),
    )
    return registry
