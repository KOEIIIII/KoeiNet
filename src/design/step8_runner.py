


"""Step-8 design mapping runner built on diagnostics and Step-7.5 evidence."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from src.config import STEP8_TOP_N

from .design_evidence import build_step8_evidence_registry
from .design_mapper import build_design_artifacts

logger = logging.getLogger("design.step8_runner")


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _sanitize_json_value(dict(payload))
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = _sanitize_json_value(dict(row))
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_jsonl_map(path: Path, nested_key: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                continue
            segment_id = item.get("segment_id")
            nested = item.get(nested_key)
            payload: Dict[str, Any] = dict(item)
            if isinstance(nested, dict):
                payload[nested_key] = nested
            try:
                out[int(segment_id)] = payload
            except Exception:
                continue
    return out


def _resolve_paths(video_dir: str, step8_outdir: Optional[str]) -> Dict[str, Path]:
    vdir = Path(video_dir)
    out_dir = Path(step8_outdir) if step8_outdir else (vdir / "design")
    return {
        "video_dir": vdir,
        "out_dir": out_dir,
        "diagnosis_jsonl": vdir / "diagnostics" / "segment_diagnosis.jsonl",
        "profiles_jsonl": vdir / "diagnostics" / "segment_profiles.jsonl",
        "model_feature_csv": vdir / "fusion" / "model_feature_table.csv",
        "model_feature_dict_json": vdir / "fusion" / "model_feature_dictionary.json",
        "labels_csv": vdir / "validation" / "final_annotation_labels_adjudicated.csv",
        "model_comparison_refined_csv": vdir / "fusion_eval_refined" / "model_comparison_refined.csv",
        "per_target_metrics_refined_csv": vdir / "fusion_eval_refined" / "per_target_metrics_refined.csv",
        "permutation_importance_csv": vdir / "fusion_eval_refined" / "permutation_importance.csv",
        "step75_summary_md": vdir / "fusion_eval_refined" / "step75_summary.md",
        "feature_group_registry_refined_json": vdir / "fusion_eval_refined" / "feature_group_registry_refined.json",
        "audio_segment_features_csv": vdir / "soundscape" / "audio_segment_features.csv",
        "segment_manifest_csv": vdir / "segments" / "segment_manifest.csv",
        "feature_screening_registry_json": vdir / "fusion_eval_refined" / "feature_screening_registry.json",
    }


def _required_paths(paths: Mapping[str, Path]) -> List[Path]:
    return [
        paths["diagnosis_jsonl"],
        paths["profiles_jsonl"],
        paths["model_feature_csv"],
        paths["model_feature_dict_json"],
        paths["labels_csv"],
        paths["model_comparison_refined_csv"],
        paths["per_target_metrics_refined_csv"],
        paths["permutation_importance_csv"],
        paths["step75_summary_md"],
        paths["feature_group_registry_refined_json"],
    ]


def _validate_required_inputs(paths: Mapping[str, Path]) -> None:
    missing = [path.as_posix() for path in _required_paths(paths) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Step-8 requires existing diagnostics + Step-7.5 refined outputs; missing: "
            + ", ".join(missing)
        )


def _merge_segment_records(
    *,
    model_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    diagnosis_map: Mapping[int, Mapping[str, Any]],
    profile_map: Mapping[int, Mapping[str, Any]],
    manifest_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    labels_lookup = (
        labels_df.set_index("segment_id").to_dict(orient="index")
        if not labels_df.empty and "segment_id" in labels_df.columns
        else {}
    )
    manifest_lookup = (
        manifest_df.set_index("segment_id").to_dict(orient="index")
        if manifest_df is not None and not manifest_df.empty and "segment_id" in manifest_df.columns
        else {}
    )

    records: List[Dict[str, Any]] = []
    for row in model_df.to_dict(orient="records"):
        segment_id = int(row["segment_id"])
        merged: Dict[str, Any] = dict(row)
        merged.update(labels_lookup.get(segment_id, {}))
        merged.update(manifest_lookup.get(segment_id, {}))

        diagnosis_row = dict(diagnosis_map.get(segment_id, {}))
        profile_row = dict(profile_map.get(segment_id, {}))
        merged["diagnosis_json"] = diagnosis_row.get("diagnosis_json", {})
        merged["diagnosis_status"] = diagnosis_row.get("diagnosis_status", "")
        merged["profile_json"] = profile_row.get("profile_json", {})
        merged["soundscape_json"] = profile_row.get("soundscape_json", {})
        merged["profile_status"] = profile_row.get("profile_status", "")
        merged["soundscape_status"] = profile_row.get("soundscape_status", "")
        merged["model_feature_row_available"] = True
        merged["step75_evidence_available"] = True
        records.append(merged)
    return records


def _summary_markdown(
    *,
    video_dir: str,
    evidence_registry: Mapping[str, Any],
    summary_payload: Mapping[str, Any],
    top_n_used: int,
    output_paths: Mapping[str, Path],
) -> str:
    theme_counts = summary_payload.get("theme_counts", {})
    street_type_counts = summary_payload.get("street_type_counts", {})
    focus_counts = summary_payload.get("focus_target_counts", {})
    top_rows = summary_payload.get("top_priority_segments", [])

    def _dict_lines(data: Mapping[str, Any]) -> List[str]:
        return [f"- {key}: {value}" for key, value in data.items()]

    lines: List[str] = [
        "# Step 8 Design Mapping Summary",
        "",
        "## 1) Evidence Sources",
        f"- diagnostics: `{Path(video_dir, 'diagnostics').as_posix()}`",
        f"- model features: `{Path(video_dir, 'fusion', 'model_feature_table.csv').as_posix()}`",
        f"- adjudicated labels: `{Path(video_dir, 'validation', 'final_annotation_labels_adjudicated.csv').as_posix()}`",
        f"- refined model evidence: `{Path(video_dir, 'fusion_eval_refined').as_posix()}`",
        "",
        "## 2) Final Model-Evidence Layer",
        "- Step 7.5 refined outputs are the final model-evidence layer for Step 8.",
        "- Step 7.6 outputs are not required and are not used.",
        "",
        "## 3) Universal Fusion Assumption",
        "- Fusion was not assumed to be universally superior.",
        "- Design logic is target-specific and evidence-weighted.",
        "",
        "## 4) Confirmatory Target Logic",
        "- comfort_score: soundscape/audio evidence remains highly relevant to comfort-oriented interventions.",
        "- vitality_score: multimodal complementarity is treated as important; visual and audible human-scale cues are combined.",
        "- soundscape_eventfulness: interpreted contextually; visual-only may remain strongest in the current sample.",
        "",
        "## 5) Street-Type Modulation",
        "- Street type categories: commercial_social, transport_movement, leisure_cultural, mixed_uncertain.",
        "- Desired vitality and eventfulness are adjusted by street type rather than using one global target.",
    ]
    lines.extend(_dict_lines(street_type_counts if isinstance(street_type_counts, Mapping) else {}))
    lines.extend(
        [
            "",
            "## 6) Main Recurring Intervention Themes",
        ]
    )
    lines.extend(_dict_lines(theme_counts if isinstance(theme_counts, Mapping) else {}))
    lines.extend(
        [
            "",
            "## 7) Soundscape Logic",
            "- Soundscape is a hard input in design mapping, especially for comfort and pleasantness support rules.",
            "- Eventfulness calibration distinguishes beneficial liveliness from chaotic or traffic-dominated noise.",
            "- No generic 'add sound' rule is used.",
            "",
            "## 8) Focus Targets Across Selected Segments",
        ]
    )
    lines.extend(_dict_lines(focus_counts if isinstance(focus_counts, Mapping) else {}))
    lines.extend(
        [
            "",
            "## 9) Output Files",
            f"- evidence_registry: `{output_paths['step8_evidence_registry_json'].as_posix()}`",
            f"- priority_ranking: `{output_paths['segment_priority_ranking_csv'].as_posix()}`",
            f"- design_plan: `{output_paths['design_plan_jsonl'].as_posix()}`",
            f"- intervention_matrix: `{output_paths['intervention_matrix_csv'].as_posix()}`",
            f"- edit_prompts: `{output_paths['edit_prompts_jsonl'].as_posix()}`",
            f"- summary: `{output_paths['step8_design_summary_md'].as_posix()}`",
            "",
            "## 10) Top Priority Segments",
        ]
    )
    if isinstance(top_rows, list) and top_rows:
        lines.append("| segment_id | priority_score | priority_level | street_type | theme |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in top_rows[:10]:
            lines.append(
                f"| {row.get('segment_id')} | {float(row.get('priority_score', 0.0)):.3f} | "
                f"{row.get('priority_level')} | {row.get('street_type')} | {row.get('recommended_intervention_theme')} |"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## 11) Caveats",
            "- Current evidence is target-specific, not a blanket fusion-wins claim.",
            "- Eventfulness should always be interpreted contextually and by street type.",
            "- Exploratory targets are supporting only and should not dominate design decisions.",
            f"- Selected design-plan scope in this run: {top_n_used} segment(s). Use `--step8_top_n 0` or omit it for all ranked segments.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def run_step8_design_mapping(
    *,
    video_dir: str,
    step8_outdir: Optional[str] = None,
    top_n: Optional[int] = None,
    smoke_test: bool = False,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the additive Step-8 design mapping stage using existing outputs only."""
    paths = _resolve_paths(video_dir, step8_outdir=step8_outdir)
    if progress_callback:
        progress_callback(0, 5, "design | validate inputs")
    _validate_required_inputs(paths)

    model_df = pd.read_csv(paths["model_feature_csv"])
    model_df["segment_id"] = pd.to_numeric(model_df["segment_id"], errors="coerce")
    model_df = model_df.dropna(subset=["segment_id"]).copy()
    model_df["segment_id"] = model_df["segment_id"].astype(int)

    labels_df = pd.read_csv(paths["labels_csv"])
    labels_df["segment_id"] = pd.to_numeric(labels_df["segment_id"], errors="coerce")
    labels_df = labels_df.dropna(subset=["segment_id"]).copy()
    labels_df["segment_id"] = labels_df["segment_id"].astype(int)

    diagnosis_map = _read_jsonl_map(paths["diagnosis_jsonl"], "diagnosis_json")
    profile_map = _read_jsonl_map(paths["profiles_jsonl"], "profile_json")
    manifest_df = pd.read_csv(paths["segment_manifest_csv"]) if paths["segment_manifest_csv"].exists() else None

    evidence_registry = build_step8_evidence_registry(
        video_dir=video_dir,
        model_comparison_csv=paths["model_comparison_refined_csv"].as_posix(),
        per_target_metrics_csv=paths["per_target_metrics_refined_csv"].as_posix(),
        permutation_importance_csv=paths["permutation_importance_csv"].as_posix(),
        step75_summary_md=paths["step75_summary_md"].as_posix(),
        feature_group_registry_json=paths["feature_group_registry_refined_json"].as_posix(),
        model_feature_dictionary_json=paths["model_feature_dict_json"].as_posix(),
    )
    if progress_callback:
        progress_callback(2, 5, "design | evidence registry")

    segment_records = _merge_segment_records(
        model_df=model_df,
        labels_df=labels_df,
        diagnosis_map=diagnosis_map,
        profile_map=profile_map,
        manifest_df=manifest_df,
    )
    effective_top_n = int(top_n if top_n is not None else STEP8_TOP_N)
    if smoke_test and effective_top_n <= 0:
        effective_top_n = min(8, len(segment_records))
    if progress_callback:
        progress_callback(3, 5, "design | merge records")

    artifacts = build_design_artifacts(
        segment_records=segment_records,
        evidence_registry=evidence_registry,
        top_n=effective_top_n,
    )
    if progress_callback:
        progress_callback(4, 5, "design | build artifacts")

    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "step8_evidence_registry_json": out_dir / "step8_evidence_registry.json",
        "segment_priority_ranking_csv": out_dir / "segment_priority_ranking.csv",
        "design_plan_jsonl": out_dir / "design_plan.jsonl",
        "intervention_matrix_csv": out_dir / "intervention_matrix.csv",
        "edit_prompts_jsonl": out_dir / "edit_prompts.jsonl",
        "step8_design_summary_md": out_dir / "step8_design_summary.md",
    }

    _write_json(output_paths["step8_evidence_registry_json"], evidence_registry)
    artifacts["ranking_df"].to_csv(output_paths["segment_priority_ranking_csv"], index=False, encoding="utf-8")
    _write_jsonl(output_paths["design_plan_jsonl"], artifacts["plan_records"])
    artifacts["intervention_df"].to_csv(output_paths["intervention_matrix_csv"], index=False, encoding="utf-8")
    _write_jsonl(output_paths["edit_prompts_jsonl"], artifacts["prompt_records"])
    output_paths["step8_design_summary_md"].write_text(
        _summary_markdown(
            video_dir=video_dir,
            evidence_registry=evidence_registry,
            summary_payload=artifacts["summary_payload"],
            top_n_used=len(artifacts["plan_records"]),
            output_paths=output_paths,
        ),
        encoding="utf-8",
    )
    if progress_callback:
        progress_callback(5, 5, "design | write outputs")

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_dir": Path(video_dir).as_posix(),
        "step8_outdir": out_dir.as_posix(),
        "step8_evidence_registry_json": output_paths["step8_evidence_registry_json"].as_posix(),
        "segment_priority_ranking_csv": output_paths["segment_priority_ranking_csv"].as_posix(),
        "design_plan_jsonl": output_paths["design_plan_jsonl"].as_posix(),
        "intervention_matrix_csv": output_paths["intervention_matrix_csv"].as_posix(),
        "edit_prompts_jsonl": output_paths["edit_prompts_jsonl"].as_posix(),
        "step8_design_summary_md": output_paths["step8_design_summary_md"].as_posix(),
        "total_segments_ranked": int(artifacts["summary_payload"].get("total_segments_ranked", 0)),
        "selected_segments_for_design_plan": int(
            artifacts["summary_payload"].get("selected_segments_for_design_plan", 0)
        ),
        "final_model_evidence_layer": "step75_refined",
    }
    logger.info(
        "step8 design mapping done | ranked=%s selected=%s out_dir=%s",
        result["total_segments_ranked"],
        result["selected_segments_for_design_plan"],
        out_dir.as_posix(),
    )
    return result
