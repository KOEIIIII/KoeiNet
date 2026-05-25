"""Configurable problem-segment detection built on existing segment artifacts."""

from __future__ import annotations

import ast
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from src.deliverable.episode_builder import build_problem_episodes, select_priority_segments


ANNOTATION_COLUMNS = [
    "segment_id",
    "street_type",
    "comfort_score",
    "vitality_score",
    "soundscape_pleasantness",
    "soundscape_eventfulness",
    "overall_problem_severity",
    "main_problem_labels",
    "primary_problem_label",
    "confidence_score",
    "annotator_notes",
]

DEFAULT_PROBLEM_LABELS = {"", "[]", "no_major_problem", "none", "nan"}


def _read_config(path: str | Path) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            payload = yaml.safe_load(text)
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            raise ValueError(
                f"Cannot parse configuration file {path}. Install PyYAML or keep the file JSON-compatible."
            ) from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _backup_if_exists(path: Path, backup_root: Path) -> Optional[str]:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = backup_root / stamp / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return target.as_posix()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except Exception:
        return default


def _norm_score(value: Any, *, low: float, high: float, default: float = 0.0) -> float:
    raw = _as_float(value, default=math.nan)
    if not math.isfinite(raw):
        return default
    if high <= low:
        return default
    return max(0.0, min(1.0, (raw - low) / (high - low)))


def _deficit(value: Any, *, low: float, high: float) -> float:
    raw = _as_float(value, default=math.nan)
    if not math.isfinite(raw):
        return 0.0
    return max(0.0, min(1.0, (high - raw) / max(1e-9, high - low)))


def _parse_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = []
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    return [item.strip() for item in text.replace("|", ";").split(";") if item.strip()]


def _has_problem_label(row: Mapping[str, Any]) -> bool:
    labels = [x.lower() for x in _parse_labels(row.get("main_problem_labels"))]
    primary = str(row.get("primary_problem_label", "")).strip().lower()
    return any(label not in DEFAULT_PROBLEM_LABELS for label in labels) or primary not in DEFAULT_PROBLEM_LABELS


def _priority_level(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _ensure_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "segment_id" not in out.columns:
        raise ValueError("segment_manifest.csv missing required column: segment_id")
    out["segment_id"] = pd.to_numeric(out["segment_id"], errors="coerce")
    out = out.dropna(subset=["segment_id"]).copy()
    out["segment_id"] = out["segment_id"].astype(int)
    if "start_time_sec" not in out.columns:
        out["start_time_sec"] = out.index.astype(float)
    if "end_time_sec" not in out.columns:
        out["end_time_sec"] = out["start_time_sec"].astype(float) + 1.0
    if "center_time_sec" not in out.columns:
        out["center_time_sec"] = (out["start_time_sec"].astype(float) + out["end_time_sec"].astype(float)) / 2.0
    for column in ("included_frame_indices", "included_frame_paths"):
        if column not in out.columns:
            out[column] = "[]"
    return out.sort_values(["start_time_sec", "segment_id"]).reset_index(drop=True)


def _resolve_video_dir(video_dir: str | Path) -> Path:
    vdir = Path(video_dir)
    if not vdir.is_dir():
        raise FileNotFoundError(f"Program 01 output folder not found: {vdir}")
    return vdir


def create_annotation_template(video_dir: str | Path, output_csv: str | Path | None = None) -> str:
    """Create or refresh a standard annotation CSV from Program 01 segment outputs."""
    vdir = _resolve_video_dir(video_dir)
    manifest_path = vdir / "segments" / "segment_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cannot create annotations without {manifest_path}")
    manifest = _ensure_manifest_columns(pd.read_csv(manifest_path))
    out = manifest[["segment_id", "start_time_sec", "end_time_sec"]].copy()
    defaults = {
        "street_type": "mixed_use",
        "comfort_score": "",
        "vitality_score": "",
        "soundscape_pleasantness": "",
        "soundscape_eventfulness": "",
        "overall_problem_severity": "",
        "main_problem_labels": "[]",
        "primary_problem_label": "no_major_problem",
        "confidence_score": "",
        "annotator_notes": "",
    }
    for column, value in defaults.items():
        out[column] = value
    target = Path(output_csv) if output_csv else vdir / "validation" / "final_annotation_labels_adjudicated.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False, encoding="utf-8-sig")
    return target.as_posix()


def _load_annotation(video_dir: Path, annotation_csv: str | Path | None) -> pd.DataFrame:
    candidates = [
        Path(annotation_csv) if annotation_csv else None,
        video_dir / "validation" / "final_annotation_labels_adjudicated.csv",
        video_dir / "validation" / "final_annotation_labels.csv",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            df = pd.read_csv(candidate)
            if "segment_id" not in df.columns:
                raise ValueError(f"Annotation file missing segment_id: {candidate}")
            df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce")
            df = df.dropna(subset=["segment_id"]).copy()
            df["segment_id"] = df["segment_id"].astype(int)
            for column in ANNOTATION_COLUMNS:
                if column not in df.columns:
                    df[column] = "" if column != "main_problem_labels" else "[]"
            return df
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def _feature_risk(row: Mapping[str, Any], feature_columns: list[str]) -> float:
    values = []
    for column in feature_columns:
        if column in row:
            values.append(_as_float(row.get(column), default=math.nan))
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return 0.0
    clipped = [max(0.0, min(1.0, v)) for v in values]
    return float(sum(clipped) / len(clipped))


def _score_row(row: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    street_type = str(row.get("street_type") or "mixed_use").strip() or "mixed_use"
    street_configs = config.get("street_types", {}) if isinstance(config.get("street_types"), Mapping) else {}
    default_cfg = street_configs.get("default", {})
    local_cfg = dict(default_cfg)
    if street_type in street_configs and isinstance(street_configs[street_type], Mapping):
        local_cfg.update(street_configs[street_type])
    coefficients = local_cfg.get("coefficients", {}) if isinstance(local_cfg.get("coefficients"), Mapping) else {}

    score_min = float(config.get("score_min", 1))
    score_max = float(config.get("score_max", 5))
    desired_eventfulness = local_cfg.get("desired_eventfulness", [2.5, 4.5])
    if not isinstance(desired_eventfulness, list) or len(desired_eventfulness) != 2:
        desired_eventfulness = [2.5, 4.5]
    event_mid = (float(desired_eventfulness[0]) + float(desired_eventfulness[1])) / 2.0
    event_gap = abs(_as_float(row.get("soundscape_eventfulness"), default=event_mid) - event_mid)
    event_gap_norm = max(0.0, min(1.0, event_gap / max(1e-9, score_max - score_min)))

    feature_columns = config.get("fusion_risk_feature_columns", [])
    if not isinstance(feature_columns, list):
        feature_columns = []

    parts = {
        "overall_problem_severity": _norm_score(row.get("overall_problem_severity"), low=score_min, high=score_max),
        "comfort_deficit": _deficit(row.get("comfort_score"), low=score_min, high=score_max),
        "vitality_deficit": _deficit(row.get("vitality_score"), low=score_min, high=score_max),
        "soundscape_pleasantness_deficit": _deficit(row.get("soundscape_pleasantness"), low=score_min, high=score_max),
        "soundscape_eventfulness_gap": event_gap_norm,
        "fusion_feature_risk": _feature_risk(row, [str(c) for c in feature_columns]),
        "label_presence": 1.0 if _has_problem_label(row) else 0.0,
    }
    weighted = 0.0
    total_weight = 0.0
    for key, value in parts.items():
        weight = _as_float(coefficients.get(key), default=0.0)
        if weight <= 0:
            continue
        weighted += weight * value
        total_weight += weight
    base_score = weighted / total_weight if total_weight > 0 else parts["overall_problem_severity"]
    confidence = _norm_score(row.get("confidence_score"), low=score_min, high=score_max, default=0.5)
    confidence_adjustment = _as_float(coefficients.get("confidence_adjustment"), default=0.0)
    priority_score = max(0.0, min(1.0, base_score * (1.0 + confidence_adjustment * (confidence - 0.5))))
    threshold = _as_float(local_cfg.get("severity_threshold"), default=float(config.get("default_threshold", 0.45)))
    return {
        "street_type": street_type,
        "priority_score": priority_score,
        "priority_level": _priority_level(priority_score),
        "is_problem_segment": bool(priority_score >= threshold),
        "severity_threshold": threshold,
        **{f"component_{k}": v for k, v in parts.items()},
    }


def run_problem_detection(
    *,
    video_dir: str | Path,
    annotation_csv: str | Path | None = None,
    coefficient_config: str | Path = "configs/street_type_coefficients.yaml",
    output_dir: str | Path | None = None,
    top_k: Optional[int] = None,
    top_percent: Optional[float] = None,
    priority_threshold: Optional[float] = None,
    max_gap_seconds: float = 5.0,
    update_visualization_artifacts: bool = True,
) -> Dict[str, Any]:
    """Run configurable problem detection and reuse the existing episode merger."""
    vdir = _resolve_video_dir(video_dir)
    config = _read_config(coefficient_config)
    out_dir = Path(output_dir) if output_dir else vdir / "problem_detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = vdir / "segments" / "segment_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Program 01 segment manifest: {manifest_path}")
    manifest = _ensure_manifest_columns(pd.read_csv(manifest_path))
    annotations = _load_annotation(vdir, annotation_csv)
    fusion = _read_csv(vdir / "fusion" / "segment_feature_table.csv")
    visual = _read_csv(vdir / "visual" / "segment_visual_features.csv")
    soundscape = _read_csv(vdir / "soundscape" / "audio_segment_features.csv")
    geo = _read_csv(vdir / "geo_sync" / "segment_geo_metadata.csv")

    merged = manifest.merge(annotations, on="segment_id", how="left", suffixes=("", "_annotation"))
    for name, table in (("fusion", fusion), ("visual", visual), ("soundscape", soundscape), ("geo", geo)):
        if not table.empty and "segment_id" in table.columns:
            table = table.copy()
            table["segment_id"] = pd.to_numeric(table["segment_id"], errors="coerce")
            table = table.dropna(subset=["segment_id"]).copy()
            table["segment_id"] = table["segment_id"].astype(int)
            merged = merged.merge(table, on="segment_id", how="left", suffixes=("", f"_{name}"))

    rows = []
    for _, row in merged.iterrows():
        payload = row.to_dict()
        scored = _score_row(payload, config)
        labels = _parse_labels(payload.get("main_problem_labels"))
        rows.append(
            {
                "segment_id": int(payload["segment_id"]),
                "start_time_sec": payload.get("start_time_sec"),
                "end_time_sec": payload.get("end_time_sec"),
                "priority_score": scored["priority_score"],
                "priority_level": scored["priority_level"],
                "is_problem_segment": scored["is_problem_segment"],
                "severity_threshold": scored["severity_threshold"],
                "street_type": scored["street_type"],
                "main_problem_labels": ";".join(labels),
                "primary_problem_label": payload.get("primary_problem_label", ""),
                "overall_problem_severity": payload.get("overall_problem_severity", ""),
                "comfort_score": payload.get("comfort_score", ""),
                "vitality_score": payload.get("vitality_score", ""),
                "soundscape_pleasantness": payload.get("soundscape_pleasantness", ""),
                "soundscape_eventfulness": payload.get("soundscape_eventfulness", ""),
                **{k: v for k, v in scored.items() if k.startswith("component_")},
            }
        )
    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking = ranking.sort_values(["priority_score", "segment_id"], ascending=[False, True]).reset_index(drop=True)
        ranking["priority_rank"] = ranking.index + 1
    ranking_csv = out_dir / "segment_problem_priority.csv"
    ranking.to_csv(ranking_csv, index=False, encoding="utf-8-sig")

    selected_source = ranking[ranking["is_problem_segment"]].copy()
    selected, selection_meta = select_priority_segments(
        selected_source,
        top_k=top_k,
        top_percent=top_percent,
        priority_threshold=priority_threshold,
    )
    episodes, _ = build_problem_episodes(
        selected,
        manifest,
        max_gap_seconds=max_gap_seconds,
    )
    episodes_csv = out_dir / "problem_episodes.csv"
    episodes.to_csv(episodes_csv, index=False, encoding="utf-8-sig")

    summary_md = out_dir / "problem_detection_summary.md"
    lines = [
        "# Problem Detection Summary",
        "",
        f"- Source Program 01 output: `{vdir.as_posix()}`",
        f"- Segments scored: {len(ranking)}",
        f"- Problem segments: {int(ranking['is_problem_segment'].sum()) if not ranking.empty else 0}",
        f"- Episodes: {len(episodes)}",
        f"- Selection mode: {selection_meta.get('selection_mode', '')}",
        f"- Coefficient configuration: `{Path(coefficient_config).as_posix()}`",
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    run_meta = {
        "video_dir": vdir.as_posix(),
        "annotation_csv": str(annotation_csv or ""),
        "coefficient_config": Path(coefficient_config).as_posix(),
        "ranking_csv": ranking_csv.as_posix(),
        "episodes_csv": episodes_csv.as_posix(),
        "summary_md": summary_md.as_posix(),
        "selection": selection_meta,
        "update_visualization_artifacts": bool(update_visualization_artifacts),
        "visualization_artifact_backups": [],
    }
    _write_json(out_dir / "problem_detection_run.json", run_meta)

    if update_visualization_artifacts:
        design_dir = vdir / "design"
        deliverable_dir = vdir / "deliverable"
        design_dir.mkdir(parents=True, exist_ok=True)
        deliverable_dir.mkdir(parents=True, exist_ok=True)
        backup_root = out_dir / "visualization_artifact_backups"
        for target in [
            design_dir / "segment_priority_ranking.csv",
            deliverable_dir / "problem_episodes.csv",
            deliverable_dir / "problem_episode_summary.csv",
        ]:
            backup = _backup_if_exists(target, backup_root)
            if backup:
                run_meta["visualization_artifact_backups"].append(backup)
        ranking.to_csv(design_dir / "segment_priority_ranking.csv", index=False, encoding="utf-8-sig")
        episodes.to_csv(deliverable_dir / "problem_episodes.csv", index=False, encoding="utf-8-sig")
        episodes.to_csv(deliverable_dir / "problem_episode_summary.csv", index=False, encoding="utf-8-sig")
        _write_json(out_dir / "problem_detection_run.json", run_meta)

    return {
        **run_meta,
        "segments_scored": int(len(ranking)),
        "problem_segments": int(ranking["is_problem_segment"].sum()) if not ranking.empty else 0,
        "episodes": int(len(episodes)),
    }
