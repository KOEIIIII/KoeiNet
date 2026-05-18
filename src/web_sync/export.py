


"""Build frontend-friendly sync data for video, charts, and route maps."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _parse_jsonish_list(value: Any) -> List[Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _coord_triplet(row: Mapping[str, Any], prefix: str = "matched_gps") -> Dict[str, Any]:
    return {
        "gcj02": {
            "lon": _jsonable(row.get(f"{prefix}_longitude_gcj02")),
            "lat": _jsonable(row.get(f"{prefix}_latitude_gcj02")),
        },
        "wgs84": {
            "lon": _jsonable(row.get(f"{prefix}_longitude_wgs84")),
            "lat": _jsonable(row.get(f"{prefix}_latitude_wgs84")),
        },
    }


def _display_point(coords: Mapping[str, Any], prefer_wgs84: bool) -> Dict[str, Any]:
    preferred = coords.get("wgs84") if prefer_wgs84 else coords.get("gcj02")
    fallback = coords.get("gcj02") if prefer_wgs84 else coords.get("wgs84")
    point = preferred if preferred and preferred.get("lat") is not None and preferred.get("lon") is not None else fallback
    lat = _safe_float(point.get("lat")) if point else None
    lon = _safe_float(point.get("lon")) if point else None
    return {
        "lat": lat,
        "lon": lon,
    }


def _point_for_mode(coords: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    key = "wgs84" if str(mode).strip().lower() == "wgs84" else "gcj02"
    point = coords.get(key) or {}
    return {
        "lat": _safe_float(point.get("lat")),
        "lon": _safe_float(point.get("lon")),
    }


def _valid_polyline_point(lat: Any, lon: Any) -> Optional[List[float]]:
    lat_f = _safe_float(lat)
    lon_f = _safe_float(lon)
    if lat_f is None or lon_f is None:
        return None
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        return None
    return [lat_f, lon_f]


def _polyline_from_rows(rows: List[Mapping[str, Any]], prefer_wgs84: bool) -> List[List[float]]:
    points: List[List[float]] = []
    for row in rows:
        coords = _coord_triplet(row)
        display = _display_point(coords, prefer_wgs84=prefer_wgs84)
        point = _valid_polyline_point(display["lat"], display["lon"])
        if point is not None:
            points.append(point)
    return points


def _polyline_from_rows_by_mode(
    rows: List[Mapping[str, Any]],
    *,
    mode: str,
    prefix: str = "matched_gps",
) -> List[List[float]]:
    points: List[List[float]] = []
    for row in rows:
        coords = _coord_triplet(row, prefix=prefix)
        point = _point_for_mode(coords, mode)
        poly = _valid_polyline_point(point["lat"], point["lon"])
        if poly is not None:
            points.append(poly)
    return points


def _infer_analysis_frame_skip(frame_geo_df: pd.DataFrame, geo_summary: Mapping[str, Any]) -> int:
    run_config = dict(geo_summary.get("run_config", {}) or {})
    summary_result = dict(geo_summary.get("result_summary", {}) or {})
    for candidate in (
        run_config.get("analysis_frame_skip"),
        run_config.get("frame_step"),
        summary_result.get("analysis_frame_skip"),
        summary_result.get("frame_step"),
    ):
        value = _safe_float(candidate)
        if value is not None and value > 0:
            return int(round(value))

    if "frame_step" in frame_geo_df.columns:
        steps = pd.to_numeric(frame_geo_df["frame_step"], errors="coerce").dropna()
        steps = steps[steps > 0]
        if not steps.empty:
            return int(round(float(steps.mode().iloc[0])))

    if {"frame_num", "video_frame_num"}.issubset(frame_geo_df.columns):
        ratios: List[int] = []
        subset = frame_geo_df[["frame_num", "video_frame_num"]].copy()
        subset["frame_num"] = pd.to_numeric(subset["frame_num"], errors="coerce")
        subset["video_frame_num"] = pd.to_numeric(subset["video_frame_num"], errors="coerce")
        subset = subset.dropna()
        subset = subset[subset["frame_num"] > 0]
        for _, row in subset.iterrows():
            ratio = float(row["video_frame_num"]) / float(row["frame_num"])
            if ratio > 0:
                ratios.append(int(round(ratio)))
        if ratios:
            return sorted(ratios)[len(ratios) // 2]

    return 1


def _infer_frame_alignment_mode(frame_geo_df: pd.DataFrame, geo_summary: Mapping[str, Any]) -> str:
    run_config = dict(geo_summary.get("run_config", {}) or {})
    summary_result = dict(geo_summary.get("result_summary", {}) or {})
    for candidate in (
        run_config.get("frame_alignment_mode"),
        summary_result.get("frame_alignment_mode"),
    ):
        text = str(candidate).strip() if candidate is not None else ""
        if text:
            return text

    if "frame_path" in frame_geo_df.columns and frame_geo_df["frame_path"].dropna().astype(str).str.contains("frames/|frames\\\\").any():
        return "existing_analysis_frames"
    return "video_sampling"


def _build_frame_payload(
    frame_geo_df: pd.DataFrame,
    *,
    prefer_wgs84: bool,
    progress_callback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    if frame_geo_df.empty:
        return []

    rows: List[Dict[str, Any]] = []
    ordered = frame_geo_df.sort_values("video_relative_time_sec")
    if progress_callback:
        progress_callback(0, len(ordered), "web_sync | frame payload")
    for point_index, (_, row) in enumerate(ordered.iterrows()):
        payload = row.to_dict()
        coords = _coord_triplet(payload)
        rows.append(
            {
                "point_index": int(point_index),
                "frame_index": int(payload.get("frame_index", 0)),
                "frame_num": _jsonable(payload.get("frame_num", payload.get("frame_index"))),
                "video_frame_num": _jsonable(payload.get("video_frame_num")),
                "frame_name": _jsonable(payload.get("frame_name")),
                "frame_path": _jsonable(payload.get("frame_path")),
                "segment_id": _jsonable(payload.get("segment_id")),
                "video_relative_time_sec": _jsonable(payload.get("video_relative_time_sec")),
                "video_absolute_time_utc": _jsonable(payload.get("video_absolute_time_utc")),
                "within_effective_overlap_window": bool(payload.get("within_effective_overlap_window", False)),
                "match_status": _jsonable(payload.get("match_status")),
                "confidence": _jsonable(payload.get("confidence")),
                "used_interpolation": bool(payload.get("used_interpolation", False)),
                "point_display": _display_point(coords, prefer_wgs84=prefer_wgs84),
                "point_gcj02": coords["gcj02"],
                "point_wgs84": coords["wgs84"],
            }
        )
        if progress_callback:
            progress_callback(point_index + 1, len(ordered), "web_sync | frame payload")
    return rows


def _normalize_segment_lookup(df: pd.DataFrame, key: str = "segment_id") -> pd.DataFrame:
    if df.empty or key not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[key] = pd.to_numeric(out[key], errors="coerce")
    out = out.dropna(subset=[key]).copy()
    out[key] = out[key].astype(int)
    return out


def _build_segment_payload(
    segment_geo_df: pd.DataFrame,
    *,
    manifest_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    frame_payload: List[Dict[str, Any]],
    prefer_wgs84: bool,
    progress_callback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    if segment_geo_df.empty:
        return []

    merged = _normalize_segment_lookup(segment_geo_df).merge(
        _normalize_segment_lookup(manifest_df),
        on="segment_id",
        how="left",
        suffixes=("", "_manifest"),
    )
    if not ranking_df.empty:
        keep_cols = [
            col
            for col in (
                "segment_id",
                "priority_rank",
                "priority_score",
                "priority_level",
                "main_problem_labels",
                "recommended_intervention_theme",
            )
            if col in ranking_df.columns
        ]
        merged = merged.merge(ranking_df[keep_cols], on="segment_id", how="left")

    frame_points_by_segment_display: Dict[int, List[List[float]]] = {}
    frame_points_by_segment_gcj02: Dict[int, List[List[float]]] = {}
    frame_points_by_segment_wgs84: Dict[int, List[List[float]]] = {}
    for frame in frame_payload:
        segment_id = frame.get("segment_id")
        if segment_id is None:
            continue
        sid = int(segment_id)
        for key, bucket in (
            ("point_display", frame_points_by_segment_display),
            ("point_gcj02", frame_points_by_segment_gcj02),
            ("point_wgs84", frame_points_by_segment_wgs84),
        ):
            point = frame.get(key) or {}
            poly = _valid_polyline_point(point.get("lat"), point.get("lon"))
            if poly is not None:
                bucket.setdefault(sid, []).append(poly)

    rows: List[Dict[str, Any]] = []
    ordered = merged.sort_values("segment_id")
    if progress_callback:
        progress_callback(0, len(ordered), "web_sync | segment payload")
    for idx, (_, row) in enumerate(ordered.iterrows()):
        payload = row.to_dict()
        coords = _coord_triplet(payload, prefix="matched_gps")
        segment_id = int(payload["segment_id"])
        rows.append(
            {
                "segment_id": segment_id,
                "start_time_sec": _jsonable(payload.get("segment_start_time_sec", payload.get("start_time_sec"))),
                "end_time_sec": _jsonable(payload.get("segment_end_time_sec", payload.get("end_time_sec"))),
                "center_time_sec": _jsonable(payload.get("segment_center_time_sec", payload.get("center_time_sec"))),
                "center_time_utc": _jsonable(payload.get("segment_center_time_utc")),
                "sampled_frame_count": _jsonable(payload.get("sampled_frame_count")),
                "included_frame_count": _jsonable(payload.get("included_frame_count")),
                "priority_rank": _jsonable(payload.get("priority_rank")),
                "priority_score": _jsonable(payload.get("priority_score")),
                "priority_level": _jsonable(payload.get("priority_level")),
                "main_problem_labels": _jsonable(payload.get("main_problem_labels")),
                "recommended_intervention_theme": _jsonable(payload.get("recommended_intervention_theme")),
                "center_point_display": _display_point(coords, prefer_wgs84=prefer_wgs84),
                "center_point_gcj02": coords["gcj02"],
                "center_point_wgs84": coords["wgs84"],
                "segment_polyline_display": frame_points_by_segment_display.get(segment_id, []),
                "segment_polyline_gcj02": frame_points_by_segment_gcj02.get(segment_id, []),
                "segment_polyline_wgs84": frame_points_by_segment_wgs84.get(segment_id, []),
                "within_effective_overlap_window": bool(payload.get("within_effective_overlap_window", False)),
                "match_status": _jsonable(payload.get("match_status")),
                "confidence": _jsonable(payload.get("confidence")),
            }
        )
        if progress_callback:
            progress_callback(idx + 1, len(ordered), "web_sync | segment payload")
    return rows


def _build_problem_segment_payload(ranking_df: pd.DataFrame, segment_payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if ranking_df.empty or not segment_payload:
        return []
    segment_by_id = {int(item["segment_id"]): item for item in segment_payload}
    ranked = ranking_df.copy()
    if "priority_level" in ranked.columns:
        candidates = ranked[ranked["priority_level"].astype(str).isin(["high", "medium"])].copy()
    else:
        candidates = ranked.copy()
    if candidates.empty and "priority_rank" in ranked.columns:
        candidates = ranked.sort_values("priority_rank").head(12).copy()
    elif "priority_rank" in candidates.columns:
        candidates = candidates.sort_values("priority_rank").copy()
    rows: List[Dict[str, Any]] = []
    for _, row in candidates.iterrows():
        sid = int(row["segment_id"])
        seg = segment_by_id.get(sid, {})
        rows.append(
            {
                "segment_id": sid,
                "priority_rank": _jsonable(row.get("priority_rank")),
                "priority_score": _jsonable(row.get("priority_score")),
                "priority_level": _jsonable(row.get("priority_level")),
                "main_problem_labels": _jsonable(row.get("main_problem_labels")),
                "recommended_intervention_theme": _jsonable(row.get("recommended_intervention_theme")),
                "center_point_display": seg.get("center_point_display"),
                "center_point_gcj02": seg.get("center_point_gcj02"),
                "center_point_wgs84": seg.get("center_point_wgs84"),
                "segment_polyline_display": seg.get("segment_polyline_display"),
                "segment_polyline_gcj02": seg.get("segment_polyline_gcj02"),
                "segment_polyline_wgs84": seg.get("segment_polyline_wgs84"),
                "start_time_sec": seg.get("start_time_sec"),
                "end_time_sec": seg.get("end_time_sec"),
            }
        )
    return rows


def _build_problem_episode_payload(
    problem_episodes_df: pd.DataFrame,
    problem_episode_summary_df: pd.DataFrame,
    segment_payload: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if problem_episodes_df.empty:
        return []
    segment_by_id = {int(item["segment_id"]): item for item in segment_payload}
    summary_lookup: Dict[str, Dict[str, Any]] = {}
    if not problem_episode_summary_df.empty and "episode_id" in problem_episode_summary_df.columns:
        summary_lookup = {
            str(row["episode_id"]): row.to_dict()
            for _, row in problem_episode_summary_df.iterrows()
        }

    rows: List[Dict[str, Any]] = []
    for _, row in problem_episodes_df.iterrows():
        payload = row.to_dict()
        segment_ids = [int(x) for x in _parse_jsonish_list(payload.get("segment_ids"))]
        representative_segment_id = payload.get("representative_segment_id")
        rep = segment_by_id.get(int(representative_segment_id), {}) if pd.notna(representative_segment_id) else {}
        summary = summary_lookup.get(str(payload.get("episode_id")), {})
        rows.append(
            {
                "episode_id": _jsonable(payload.get("episode_id")),
                "segment_ids": segment_ids,
                "representative_segment_id": _jsonable(representative_segment_id),
                "start_time_sec": _jsonable(payload.get("start_time_sec")),
                "end_time_sec": _jsonable(payload.get("end_time_sec")),
                "duration_sec": _jsonable(payload.get("duration_sec")),
                "n_segments": _jsonable(payload.get("n_segments")),
                "priority_rank": _jsonable(summary.get("priority_rank")),
                "priority_score": _jsonable(summary.get("priority_score")),
                "priority_level": _jsonable(summary.get("priority_level")),
                "episode_title": _jsonable(summary.get("episode_title")),
                "problem_summary": _jsonable(summary.get("one_sentence_summary")),
                "center_point_display": rep.get("center_point_display"),
                "center_point_gcj02": rep.get("center_point_gcj02"),
                "center_point_wgs84": rep.get("center_point_wgs84"),
            }
        )
    return rows


def build_sync_map_data(
    video_dir: str | Path,
    output_json: Optional[str | Path] = None,
    *,
    prefer_wgs84: bool = True,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a frontend-consumable sync payload under output/<video>/web/."""
    vdir = Path(video_dir)
    geo_dir = vdir / "geo_sync"
    segment_manifest_csv = vdir / "segments" / "segment_manifest.csv"
    frame_geo_csv = geo_dir / "frame_geo_metadata.csv"
    segment_geo_csv = geo_dir / "segment_geo_metadata.csv"
    geo_summary_json = geo_dir / "geo_sync_summary.json"
    ranking_csv = vdir / "design" / "segment_priority_ranking.csv"
    problem_episodes_csv = vdir / "deliverable" / "problem_episodes.csv"
    problem_episode_summary_csv = vdir / "deliverable" / "problem_episode_summary.csv"

    if not frame_geo_csv.is_file():
        raise FileNotFoundError(f"frame geo metadata not found: {frame_geo_csv.as_posix()}")
    if not segment_geo_csv.is_file():
        raise FileNotFoundError(f"segment geo metadata not found: {segment_geo_csv.as_posix()}")
    if not segment_manifest_csv.is_file():
        raise FileNotFoundError(f"segment manifest not found: {segment_manifest_csv.as_posix()}")

    out_dir = vdir / "web"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(output_json) if output_json else out_dir / "sync_map_data.json"

    frame_geo_df = _read_csv(frame_geo_csv)
    segment_geo_df = _read_csv(segment_geo_csv)
    manifest_df = _read_csv(segment_manifest_csv)
    ranking_df = _normalize_segment_lookup(_read_csv(ranking_csv))
    problem_episodes_df = _read_csv(problem_episodes_csv)
    problem_episode_summary_df = _read_csv(problem_episode_summary_csv)
    geo_summary = _read_json(geo_summary_json)

    if progress_callback:
        progress_callback(0, 4, "web_sync | load inputs")
    frame_payload = _build_frame_payload(
        frame_geo_df,
        prefer_wgs84=prefer_wgs84,
        progress_callback=(
            lambda completed, total=None, description=None: progress_callback(
                completed,
                total,
                description,
            )
        )
        if progress_callback
        else None,
    )
    segment_payload = _build_segment_payload(
        segment_geo_df=segment_geo_df,
        manifest_df=manifest_df,
        ranking_df=ranking_df,
        frame_payload=frame_payload,
        prefer_wgs84=prefer_wgs84,
        progress_callback=progress_callback,
    )
    if progress_callback:
        progress_callback(3, 4, "web_sync | overlays")
    problem_segments = _build_problem_segment_payload(ranking_df, segment_payload)
    problem_episodes = _build_problem_episode_payload(
        problem_episodes_df,
        problem_episode_summary_df,
        segment_payload,
    )

    frame_alignment_mode = _infer_frame_alignment_mode(frame_geo_df, geo_summary)
    analysis_frame_skip = _infer_analysis_frame_skip(frame_geo_df, geo_summary)

    display_mode = "wgs84" if prefer_wgs84 else "gcj02"
    fallback_mode = "gcj02" if prefer_wgs84 else "wgs84"
    frame_rows = frame_geo_df.to_dict(orient="records")
    route_polyline_display = _polyline_from_rows(frame_rows, prefer_wgs84=prefer_wgs84)
    route_polyline_gcj02 = _polyline_from_rows_by_mode(frame_rows, mode="gcj02")
    route_polyline_wgs84 = _polyline_from_rows_by_mode(frame_rows, mode="wgs84")

    payload: Dict[str, Any] = {
        "video": {
            "video_name": vdir.name,
            "video_dir": vdir.as_posix(),
            "duration_sec": _jsonable(geo_summary.get("video_probe", {}).get("duration_sec")),
            "fps": _jsonable(geo_summary.get("video_probe", {}).get("fps")),
            "frame_alignment_mode": frame_alignment_mode,
            "analysis_frame_skip": int(analysis_frame_skip),
        },
        "coordinate_policy": {
            "source_coordinate_system": "GCJ-02",
            "display_coordinate_system": "WGS84" if prefer_wgs84 else "GCJ-02",
            "preferred_map_mode": display_mode,
            "fallback_map_mode": fallback_mode,
            "available_coordinate_modes": ["gcj02", "wgs84"],
            "web_map_coordinate_note": (
                "Web map display uses derived WGS84 when available so the route aligns with standard web tiles; "
                "raw GCJ-02 remains preserved in geo_sync outputs."
                if prefer_wgs84
                else "Web map display uses raw GCJ-02 coordinates."
            ),
        },
        "route": {
            "full_polyline": route_polyline_display,
            "full_polyline_display": route_polyline_display,
            "full_polyline_gcj02": route_polyline_gcj02,
            "full_polyline_wgs84": route_polyline_wgs84,
            "total_points": int(
                max(len(route_polyline_display), len(route_polyline_gcj02), len(route_polyline_wgs84))
            ),
        },
        "timeline": {
            "frame_count": int(len(frame_payload)),
            "segment_count": int(len(segment_payload)),
        },
        "frames": frame_payload,
        "segments": segment_payload,
        "problem_overlays": {
            "problem_segments": problem_segments,
            "problem_episodes": problem_episodes,
        },
        "source_files": {
            "frame_geo_metadata_csv": frame_geo_csv.as_posix(),
            "segment_geo_metadata_csv": segment_geo_csv.as_posix(),
            "segment_manifest_csv": segment_manifest_csv.as_posix(),
            "segment_priority_ranking_csv": ranking_csv.as_posix() if ranking_csv.is_file() else None,
            "problem_episodes_csv": problem_episodes_csv.as_posix() if problem_episodes_csv.is_file() else None,
            "problem_episode_summary_csv": (
                problem_episode_summary_csv.as_posix() if problem_episode_summary_csv.is_file() else None
            ),
        },
    }
    output_json.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback(4, 4, "web_sync | write json")
    return {
        "video_dir": vdir.as_posix(),
        "output_json": output_json.as_posix(),
        "frame_count": int(len(frame_payload)),
        "segment_count": int(len(segment_payload)),
        "problem_segment_count": int(len(problem_segments)),
        "problem_episode_count": int(len(problem_episodes)),
        "display_coordinate_system": "WGS84" if prefer_wgs84 else "GCJ-02",
    }
