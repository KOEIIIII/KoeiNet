


"""Build GIS-friendly frame/segment/episode export tables from existing outputs."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


FRAME_AI_ACTIVITY_RENAME: Mapping[str, str] = {
    "坐下休息_score": "frame_ai_activity_sitting_score",
    "站着停留_score": "frame_ai_activity_standing_score",
    "散步_score": "frame_ai_activity_walking_score",
    "跑步_score": "frame_ai_activity_running_score",
    "健身锻炼_score": "frame_ai_activity_fitness_score",
    "买菜购物_score": "frame_ai_activity_shopping_score",
}

FRAME_STATS_SPECS: Sequence[Tuple[str, str]] = (
    ("stats/visual_elements/major_categories_proportion.csv", "frame_major_semantic_"),
    ("stats/visual_elements/detailed_categories_proportion.csv", "frame_detailed_semantic_"),
    ("stats/green_view/green_view_index.csv", "frame_green_view_"),
    ("stats/emotion/emotion_scores.csv", "frame_emotion_"),
    ("stats/people_count/people_count.csv", "frame_people_"),
    ("stats/color_analysis/color_categories_proportion.csv", "frame_color_"),
)

EMPTY_EPISODE_COLUMNS: Sequence[str] = (
    "episode_id",
    "segment_ids_json",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "n_segments",
    "problem_episode_rank",
    "problem_episode_label",
    "problem_severity_score",
    "display_coordinate_system",
    "center_longitude_display",
    "center_latitude_display",
    "center_longitude_gcj02",
    "center_latitude_gcj02",
    "center_longitude_wgs84",
    "center_latitude_wgs84",
    "route_point_count",
    "route_polyline_display_json",
    "route_polyline_gcj02_json",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


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


def _parse_frame_token(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


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


def _coord_columns(prefix: str) -> Tuple[str, str, str, str]:
    return (
        f"{prefix}_longitude_gcj02",
        f"{prefix}_latitude_gcj02",
        f"{prefix}_longitude_wgs84",
        f"{prefix}_latitude_wgs84",
    )


def _display_point(row: Mapping[str, Any], prefix: str, prefer_wgs84: bool) -> Dict[str, Optional[float]]:
    lon_gcj, lat_gcj, lon_wgs, lat_wgs = _coord_columns(prefix)
    if prefer_wgs84:
        lat = _safe_float(row.get(lat_wgs))
        lon = _safe_float(row.get(lon_wgs))
        if lat is None or lon is None:
            lat = _safe_float(row.get(lat_gcj))
            lon = _safe_float(row.get(lon_gcj))
    else:
        lat = _safe_float(row.get(lat_gcj))
        lon = _safe_float(row.get(lon_gcj))
        if lat is None or lon is None:
            lat = _safe_float(row.get(lat_wgs))
            lon = _safe_float(row.get(lon_wgs))
    return {"lat": lat, "lon": lon}


def _valid_point(lat: Any, lon: Any) -> Optional[List[float]]:
    lat_f = _safe_float(lat)
    lon_f = _safe_float(lon)
    if lat_f is None or lon_f is None:
        return None
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        return None
    return [lat_f, lon_f]


def _normalize_segment_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "segment_id" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["segment_id"] = pd.to_numeric(out["segment_id"], errors="coerce")
    out = out.dropna(subset=["segment_id"]).copy()
    out["segment_id"] = out["segment_id"].astype(int)
    return out.sort_values("segment_id").reset_index(drop=True)


def _prefix_segment_columns(df: pd.DataFrame, prefix: str, exclude: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rename_map = {
        col: f"{prefix}{col}"
        for col in df.columns
        if col not in exclude
    }
    return df.rename(columns=rename_map)


def _prepare_activity_frame_df(activity_csv: Path) -> pd.DataFrame:
    df = _read_csv(activity_csv)
    if df.empty or "frame_num" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["video_frame_num"] = pd.to_numeric(out["frame_num"], errors="coerce")
    out = out.dropna(subset=["video_frame_num"]).copy()
    out["video_frame_num"] = out["video_frame_num"].astype(int)

    keep = ["video_frame_num"]
    rename_map: Dict[str, str] = {}
    for src, dst in FRAME_AI_ACTIVITY_RENAME.items():
        if src in out.columns:
            keep.append(src)
            rename_map[src] = dst
    if not rename_map:
        return pd.DataFrame()
    out = out[keep].rename(columns=rename_map)
    for col in rename_map.values():
        out[col] = pd.to_numeric(out[col], errors="coerce")

    activity_cols = [col for col in rename_map.values() if col in out.columns]
    if activity_cols:
        out["frame_ai_activity_major_label"] = out[activity_cols].idxmax(axis=1).str.replace(
            "frame_ai_activity_", "", regex=False
        ).str.replace("_score", "", regex=False)
    return out.drop_duplicates("video_frame_num", keep="last")


def _prepare_optional_frame_stats(path: Path, prefix: str) -> Tuple[str, pd.DataFrame]:
    df = _read_csv(path)
    if df.empty:
        return "none", pd.DataFrame()

    out = df.copy()
    join_mode = "none"
    if "frame_num" in out.columns:
        out["video_frame_num"] = pd.to_numeric(out["frame_num"], errors="coerce")
        join_mode = "video_frame_num"
    elif "frame_index" in out.columns:
        out["frame_index"] = pd.to_numeric(out["frame_index"], errors="coerce")
        join_mode = "frame_index"
    elif "frame_name" in out.columns:
        out["frame_index"] = out["frame_name"].map(_parse_frame_token)
        join_mode = "frame_index"
    elif "frame_path" in out.columns:
        out["frame_index"] = out["frame_path"].map(lambda x: _parse_frame_token(Path(str(x)).name))
        join_mode = "frame_index"

    if join_mode == "none":
        return "none", pd.DataFrame()

    out = out.dropna(subset=[join_mode]).copy()
    out[join_mode] = out[join_mode].astype(int)
    exclude = {
        "frame_num",
        "frame_index",
        "frame_name",
        "frame_path",
        "video_frame_num",
    }
    rename_map = {col: f"{prefix}{col}" for col in out.columns if col not in exclude and col != join_mode}
    selected = [join_mode] + [col for col in out.columns if col in rename_map]
    out = out[selected].rename(columns=rename_map).drop_duplicates(join_mode, keep="last")
    return join_mode, out


def _optional_path(path: Path) -> Optional[str]:
    return path.as_posix() if path.is_file() else None


def _build_problem_tables(
    *,
    manifest_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    problem_episodes_df: pd.DataFrame,
    problem_episode_summary_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    base = manifest_df[["segment_id"]].copy()
    base["segment_id"] = pd.to_numeric(base["segment_id"], errors="coerce").astype(int)
    base["is_problem_segment"] = False
    base["problem_priority_rank"] = pd.NA
    base["problem_priority_level"] = pd.NA
    base["problem_main_label"] = pd.NA
    base["problem_episode_id"] = pd.NA
    base["problem_episode_rank"] = pd.NA
    base["problem_episode_label"] = pd.NA
    base["problem_severity_score"] = pd.NA
    base["problem_annotation_source"] = pd.NA

    mode = "none"
    ranking = _normalize_segment_df(ranking_df)
    if not ranking.empty:
        mode = "segment_ranking"
        keep_cols = [
            col
            for col in ("segment_id", "priority_rank", "priority_level", "main_problem_labels", "priority_score")
            if col in ranking.columns
        ]
        if keep_cols:
            merged = ranking[keep_cols].rename(
                columns={
                    "priority_rank": "problem_priority_rank",
                    "priority_level": "problem_priority_level",
                    "main_problem_labels": "problem_main_label",
                    "priority_score": "problem_severity_score",
                }
            )
            base = base.merge(merged, on="segment_id", how="left", suffixes=("", "_ranking"))
            for col in ("problem_priority_rank", "problem_priority_level", "problem_main_label", "problem_severity_score"):
                ranking_col = f"{col}_ranking"
                if ranking_col in base.columns:
                    base[col] = base[ranking_col].where(base[ranking_col].notna(), base[col])
                    base = base.drop(columns=[ranking_col])
            base["is_problem_segment"] = base["problem_priority_rank"].notna() | base["problem_main_label"].notna()
            base.loc[base["is_problem_segment"], "problem_annotation_source"] = "design_segment_ranking"

    episode_rows: List[Dict[str, Any]] = []
    episode_index: Dict[int, Dict[str, Any]] = {}
    if not problem_episodes_df.empty:
        summary_lookup: Dict[str, Dict[str, Any]] = {}
        if not problem_episode_summary_df.empty and "episode_id" in problem_episode_summary_df.columns:
            summary_lookup = {
                str(row["episode_id"]): row.to_dict()
                for _, row in problem_episode_summary_df.iterrows()
            }

        for _, row in problem_episodes_df.iterrows():
            episode = row.to_dict()
            episode_id = episode.get("episode_id")
            segment_ids = [int(x) for x in _parse_jsonish_list(episode.get("segment_ids")) if str(x).strip()]
            summary = summary_lookup.get(str(episode_id), {})
            record = {
                "episode_id": episode_id,
                "segment_ids": segment_ids,
                "problem_episode_rank": summary.get("priority_rank", episode.get("priority_rank")),
                "problem_episode_label": (
                    summary.get("episode_title")
                    or summary.get("one_sentence_summary")
                    or episode.get("problem_summary")
                    or episode.get("episode_title")
                ),
                "problem_severity_score": summary.get("priority_score", episode.get("priority_score")),
                "start_time_sec": episode.get("start_time_sec"),
                "end_time_sec": episode.get("end_time_sec"),
                "duration_sec": episode.get("duration_sec"),
                "n_segments": episode.get("n_segments"),
            }
            episode_rows.append(record)
            for segment_id in segment_ids:
                episode_index[segment_id] = record

        if episode_rows:
            mode = "deliverable_episode"
            ep_df = pd.DataFrame(
                [
                    {
                        "segment_id": sid,
                        "problem_episode_id": record.get("episode_id"),
                        "problem_episode_rank": record.get("problem_episode_rank"),
                        "problem_episode_label": record.get("problem_episode_label"),
                        "problem_severity_score": record.get("problem_severity_score"),
                    }
                    for sid, record in episode_index.items()
                ]
            )
            base = base.merge(ep_df, on="segment_id", how="left", suffixes=("", "_episode"))
            for col in ("problem_episode_id", "problem_episode_rank", "problem_episode_label", "problem_severity_score"):
                episode_col = f"{col}_episode"
                if episode_col in base.columns:
                    if col == "problem_severity_score":
                        base[col] = base[episode_col].where(base[episode_col].notna(), base[col])
                    else:
                        base[col] = base[episode_col].where(base[episode_col].notna(), base[col])
                    base = base.drop(columns=[episode_col])
            episode_mask = base["problem_episode_id"].notna()
            base.loc[episode_mask, "is_problem_segment"] = True
            base.loc[episode_mask, "problem_annotation_source"] = "deliverable_episode"

    return base, pd.DataFrame(episode_rows), mode


def _build_problem_episode_export(
    *,
    episode_df: pd.DataFrame,
    frame_geo_df: pd.DataFrame,
    segment_geo_df: pd.DataFrame,
    prefer_wgs84: bool,
) -> pd.DataFrame:
    if episode_df.empty:
        return pd.DataFrame(columns=list(EMPTY_EPISODE_COLUMNS))

    rows: List[Dict[str, Any]] = []
    for _, row in episode_df.iterrows():
        segment_ids = [int(x) for x in row.get("segment_ids", [])]
        frame_subset = frame_geo_df[frame_geo_df["segment_id"].isin(segment_ids)].copy() if "segment_id" in frame_geo_df.columns else pd.DataFrame()
        frame_subset = frame_subset.sort_values("video_relative_time_sec") if not frame_subset.empty else frame_subset

        polyline_display: List[List[float]] = []
        polyline_gcj02: List[List[float]] = []
        for _, frame_row in frame_subset.iterrows():
            display = _display_point(frame_row.to_dict(), "matched_gps", prefer_wgs84=prefer_wgs84)
            point = _valid_point(display["lat"], display["lon"])
            if point is not None:
                polyline_display.append(point)
            gcj02 = _valid_point(frame_row.get("matched_gps_latitude_gcj02"), frame_row.get("matched_gps_longitude_gcj02"))
            if gcj02 is not None:
                polyline_gcj02.append(gcj02)

        segment_subset = segment_geo_df[segment_geo_df["segment_id"].isin(segment_ids)].copy() if "segment_id" in segment_geo_df.columns else pd.DataFrame()
        center_lat = center_lon = center_gcj_lat = center_gcj_lon = center_wgs_lat = center_wgs_lon = None
        if not segment_subset.empty:
            seg_row = segment_subset.iloc[0].to_dict()
            display_center = _display_point(seg_row, "matched_gps", prefer_wgs84=prefer_wgs84)
            center_lat = display_center["lat"]
            center_lon = display_center["lon"]
            center_gcj_lat = _safe_float(seg_row.get("matched_gps_latitude_gcj02"))
            center_gcj_lon = _safe_float(seg_row.get("matched_gps_longitude_gcj02"))
            center_wgs_lat = _safe_float(seg_row.get("matched_gps_latitude_wgs84"))
            center_wgs_lon = _safe_float(seg_row.get("matched_gps_longitude_wgs84"))

        rows.append(
            {
                "episode_id": row.get("episode_id"),
                "segment_ids_json": json.dumps(segment_ids, ensure_ascii=False),
                "start_time_sec": row.get("start_time_sec"),
                "end_time_sec": row.get("end_time_sec"),
                "duration_sec": row.get("duration_sec"),
                "n_segments": row.get("n_segments"),
                "problem_episode_rank": row.get("problem_episode_rank"),
                "problem_episode_label": row.get("problem_episode_label"),
                "problem_severity_score": row.get("problem_severity_score"),
                "display_coordinate_system": "WGS84" if prefer_wgs84 else "GCJ-02",
                "center_longitude_display": center_lon,
                "center_latitude_display": center_lat,
                "center_longitude_gcj02": center_gcj_lon,
                "center_latitude_gcj02": center_gcj_lat,
                "center_longitude_wgs84": center_wgs_lon,
                "center_latitude_wgs84": center_wgs_lat,
                "route_point_count": int(len(polyline_display)),
                "route_polyline_display_json": json.dumps(polyline_display, ensure_ascii=False),
                "route_polyline_gcj02_json": json.dumps(polyline_gcj02, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def build_gis_exports(
    video_dir: str | Path,
    *,
    prefer_wgs84: bool = True,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    vdir = Path(video_dir)
    gis_dir = vdir / "gis"
    gis_dir.mkdir(parents=True, exist_ok=True)

    frame_geo_csv = vdir / "geo_sync" / "frame_geo_metadata.csv"
    segment_geo_csv = vdir / "geo_sync" / "segment_geo_metadata.csv"
    geo_summary_json = vdir / "geo_sync" / "geo_sync_summary.json"
    manifest_csv = vdir / "segments" / "segment_manifest.csv"
    visual_csv = vdir / "visual" / "segment_visual_features.csv"
    soundscape_csv = vdir / "soundscape" / "audio_segment_features.csv"
    fusion_csv = vdir / "fusion" / "segment_feature_table.csv"
    activity_csv = vdir / "ai_evaluation" / "activity_scores.csv"
    ranking_csv = vdir / "design" / "segment_priority_ranking.csv"
    problem_episodes_csv = vdir / "deliverable" / "problem_episodes.csv"
    problem_episode_summary_csv = vdir / "deliverable" / "problem_episode_summary.csv"

    if not frame_geo_csv.is_file():
        raise FileNotFoundError(f"frame geo metadata not found: {frame_geo_csv.as_posix()}")
    if not segment_geo_csv.is_file():
        raise FileNotFoundError(f"segment geo metadata not found: {segment_geo_csv.as_posix()}")
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"segment manifest not found: {manifest_csv.as_posix()}")

    if progress_callback:
        progress_callback(0, 5, "gis_export | load inputs")

    frame_geo_df = _read_csv(frame_geo_csv)
    segment_geo_df = _normalize_segment_df(_read_csv(segment_geo_csv))
    manifest_df = _normalize_segment_df(_read_csv(manifest_csv))
    geo_summary = _read_json(geo_summary_json)
    visual_df = _normalize_segment_df(_read_csv(visual_csv))
    soundscape_df = _normalize_segment_df(_read_csv(soundscape_csv))
    fusion_df = _normalize_segment_df(_read_csv(fusion_csv))
    ranking_df = _normalize_segment_df(_read_csv(ranking_csv))
    problem_episodes_df = _read_csv(problem_episodes_csv)
    problem_episode_summary_df = _read_csv(problem_episode_summary_csv)

    problem_segment_df, episode_df, problem_mode = _build_problem_tables(
        manifest_df=manifest_df,
        ranking_df=ranking_df,
        problem_episodes_df=problem_episodes_df,
        problem_episode_summary_df=problem_episode_summary_df,
    )
    if progress_callback:
        progress_callback(1, 5, "gis_export | problem mapping")

    display_coordinate_system = "WGS84" if prefer_wgs84 else "GCJ-02"

    frame_export_df = frame_geo_df.copy()
    frame_export_df["video_name"] = vdir.name
    frame_export_df["display_coordinate_system"] = display_coordinate_system
    frame_export_df["display_latitude"] = frame_export_df.apply(
        lambda row: _display_point(row.to_dict(), "matched_gps", prefer_wgs84=prefer_wgs84)["lat"],
        axis=1,
    )
    frame_export_df["display_longitude"] = frame_export_df.apply(
        lambda row: _display_point(row.to_dict(), "matched_gps", prefer_wgs84=prefer_wgs84)["lon"],
        axis=1,
    )

    manifest_frame_cols = [
        col for col in ("segment_id", "start_time_sec", "end_time_sec", "center_time_sec", "included_frame_count")
        if col in manifest_df.columns
    ]
    if manifest_frame_cols:
        frame_export_df = frame_export_df.merge(
            manifest_df[manifest_frame_cols].rename(
                columns={
                    "start_time_sec": "segment_start_time_sec",
                    "end_time_sec": "segment_end_time_sec",
                    "center_time_sec": "segment_center_time_sec",
                    "included_frame_count": "segment_included_frame_count",
                }
            ),
            on="segment_id",
            how="left",
        )

    frame_ai_df = _prepare_activity_frame_df(activity_csv)
    if not frame_ai_df.empty and "video_frame_num" in frame_export_df.columns:
        frame_export_df = frame_export_df.merge(frame_ai_df, on="video_frame_num", how="left")

    frame_stats_loaded: List[str] = []
    for rel_path, prefix in FRAME_STATS_SPECS:
        join_mode, stats_df = _prepare_optional_frame_stats(vdir / rel_path, prefix)
        if join_mode == "none" or stats_df.empty:
            continue
        frame_export_df = frame_export_df.merge(stats_df, on=join_mode, how="left")
        frame_stats_loaded.append(rel_path)

    if not visual_df.empty:
        visual_prefixed = _prefix_segment_columns(
            visual_df,
            prefix="segment_visual_",
            exclude=("segment_id", "start_time_sec", "end_time_sec", "center_time_sec", "included_frame_count"),
        )
        frame_export_df = frame_export_df.merge(visual_prefixed, on="segment_id", how="left")

    frame_export_df = frame_export_df.merge(problem_segment_df, on="segment_id", how="left")
    if progress_callback:
        progress_callback(2, 5, "gis_export | frame table")

    segment_export_df = manifest_df.merge(segment_geo_df, on="segment_id", how="left", suffixes=("_manifest", ""))
    segment_export_df["video_name"] = vdir.name
    segment_export_df["display_coordinate_system"] = display_coordinate_system
    segment_export_df["display_latitude"] = segment_export_df.apply(
        lambda row: _display_point(row.to_dict(), "matched_gps", prefer_wgs84=prefer_wgs84)["lat"],
        axis=1,
    )
    segment_export_df["display_longitude"] = segment_export_df.apply(
        lambda row: _display_point(row.to_dict(), "matched_gps", prefer_wgs84=prefer_wgs84)["lon"],
        axis=1,
    )

    if not visual_df.empty:
        segment_export_df = segment_export_df.merge(
            _prefix_segment_columns(
                visual_df,
                prefix="segment_visual_",
                exclude=("segment_id",),
            ),
            on="segment_id",
            how="left",
        )
    if not soundscape_df.empty:
        segment_export_df = segment_export_df.merge(
            _prefix_segment_columns(soundscape_df, prefix="soundscape_", exclude=("segment_id",)),
            on="segment_id",
            how="left",
        )
    if not fusion_df.empty:
        segment_export_df = segment_export_df.merge(
            _prefix_segment_columns(fusion_df, prefix="fusion_", exclude=("segment_id",)),
            on="segment_id",
            how="left",
        )
    segment_export_df = segment_export_df.merge(problem_segment_df, on="segment_id", how="left")
    if progress_callback:
        progress_callback(3, 5, "gis_export | segment table")

    problem_episode_export_df = _build_problem_episode_export(
        episode_df=episode_df,
        frame_geo_df=frame_geo_df,
        segment_geo_df=segment_geo_df,
        prefer_wgs84=prefer_wgs84,
    )
    if progress_callback:
        progress_callback(4, 5, "gis_export | episode table")

    frame_csv = gis_dir / "frame_gis_export.csv"
    segment_csv = gis_dir / "segment_gis_export.csv"
    episode_csv = gis_dir / "problem_episode_gis_export.csv"
    summary_json = gis_dir / "gis_export_summary.json"

    frame_export_df.to_csv(frame_csv, index=False, encoding="utf-8-sig")
    segment_export_df.to_csv(segment_csv, index=False, encoding="utf-8-sig")
    problem_episode_export_df.to_csv(episode_csv, index=False, encoding="utf-8-sig")

    missing_sources = [
        rel
        for rel, path in {
            "visual/segment_visual_features.csv": visual_csv,
            "soundscape/audio_segment_features.csv": soundscape_csv,
            "fusion/segment_feature_table.csv": fusion_csv,
            "design/segment_priority_ranking.csv": ranking_csv,
            "deliverable/problem_episodes.csv": problem_episodes_csv,
            "deliverable/problem_episode_summary.csv": problem_episode_summary_csv,
            "ai_evaluation/activity_scores.csv": activity_csv,
        }.items()
        if not path.is_file()
    ]
    frame_visual_mode_parts: List[str] = []
    if not frame_ai_df.empty:
        frame_visual_mode_parts.append("frame_ai_activity")
    if frame_stats_loaded:
        frame_visual_mode_parts.append("frame_stats")
    if not visual_df.empty:
        frame_visual_mode_parts.append("segment_visual_backfill")
    if not frame_visual_mode_parts:
        frame_visual_mode_parts.append("none")

    summary = {
        "video_name": vdir.name,
        "video_dir": vdir.as_posix(),
        "display_coordinate_system": display_coordinate_system,
        "frame_visual_mode": " + ".join(frame_visual_mode_parts),
        "problem_annotation_mode": problem_mode,
        "source_files": {
            "frame_geo_metadata_csv": frame_geo_csv.as_posix(),
            "segment_geo_metadata_csv": segment_geo_csv.as_posix(),
            "segment_manifest_csv": manifest_csv.as_posix(),
            "visual_segment_csv": _optional_path(visual_csv),
            "soundscape_segment_csv": _optional_path(soundscape_csv),
            "fusion_segment_csv": _optional_path(fusion_csv),
            "design_segment_priority_csv": _optional_path(ranking_csv),
            "deliverable_problem_episodes_csv": _optional_path(problem_episodes_csv),
            "deliverable_problem_episode_summary_csv": _optional_path(problem_episode_summary_csv),
            "frame_activity_csv": _optional_path(activity_csv),
            "geo_sync_summary_json": _optional_path(geo_summary_json),
        },
        "missing_sources": missing_sources,
        "frame_stats_loaded": frame_stats_loaded,
        "row_counts": {
            "frame_rows": int(len(frame_export_df)),
            "segment_rows": int(len(segment_export_df)),
            "problem_episode_rows": int(len(problem_episode_export_df)),
        },
        "empty_column_notes": {
            "problem_fields": (
                "deliverable/design outputs missing; problem-related columns are blank"
                if problem_mode == "none"
                else None
            ),
            "frame_visual_fields": (
                "frame-level stats directory missing; only frame_ai_activity and/or segment_visual backfill available"
                if not frame_stats_loaded
                else None
            ),
        },
        "geo_sync_run_config": dict(geo_summary.get("run_config", {}) or {}),
        "outputs": {
            "frame_gis_export_csv": frame_csv.as_posix(),
            "segment_gis_export_csv": segment_csv.as_posix(),
            "problem_episode_gis_export_csv": episode_csv.as_posix(),
            "gis_export_summary_json": summary_json.as_posix(),
        },
    }
    summary_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback(5, 5, "gis_export | write outputs")
    return {
        "video_dir": vdir.as_posix(),
        "frame_rows": int(len(frame_export_df)),
        "segment_rows": int(len(segment_export_df)),
        "problem_episode_rows": int(len(problem_episode_export_df)),
        "problem_annotation_mode": problem_mode,
        "frame_visual_mode": " + ".join(frame_visual_mode_parts),
        "outputs": summary["outputs"],
    }
