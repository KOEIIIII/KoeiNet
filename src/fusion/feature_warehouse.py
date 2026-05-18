


"""Build segment-level multimodal feature warehouse from existing outputs."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("fusion.feature_warehouse")

FRAME_INDEX_COL_CANDIDATES: Tuple[str, ...] = (
    "FrameNum",
    "frame_num",
    "frame_index",
    "frame_idx",
    "Frame",
    "frame_name",
    "frame",
)

AI_DEFAULT_SCORE_COLUMNS: Tuple[str, ...] = (
    "坐下休息_score",
    "站着停留_score",
    "散步_score",
    "跑步_score",
    "健身锻炼_score",
    "买菜购物_score",
)

REQUIRED_GROUPS: Tuple[str, ...] = (
    "visual_semantic",
    "visual_major",
    "green_view",
    "emotion",
    "people",
    "color",
    "ai_activity",
    "audio_events",
    "audio_signal",
    "audio_embedding",
)


def _safe_col_name(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("%", "pct")
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "col"


def _parse_frame_index(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(round(value))
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else None


def _read_csv_safe(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not path.is_file():
        return None, f"missing: {path.as_posix()}"
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return None, f"read_error: {path.as_posix()} ({exc})"


def _load_segments(segment_csv_path: Path) -> pd.DataFrame:
    if not segment_csv_path.is_file():
        raise FileNotFoundError(f"missing segment manifest: {segment_csv_path.as_posix()}")
    df = pd.read_csv(segment_csv_path)
    required = {"segment_id", "start_time_sec", "end_time_sec", "center_time_sec", "included_frame_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"segment manifest missing columns: {sorted(missing)}")
    return df.sort_values("segment_id").reset_index(drop=True)


def _build_frame_to_segments_map(segments_df: pd.DataFrame) -> Dict[int, List[int]]:
    frame_to_segments: Dict[int, List[int]] = {}
    for _, row in segments_df.iterrows():
        sid = int(row["segment_id"])
        raw = row.get("included_frame_indices", "[]")
        try:
            idxs = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            idxs = []
        if not isinstance(idxs, list):
            continue
        for idx in idxs:
            p = _parse_frame_index(idx)
            if p is None:
                continue
            frame_to_segments.setdefault(p, [])
            if sid not in frame_to_segments[p]:
                frame_to_segments[p].append(sid)
    return frame_to_segments


def _infer_frame_index_series(df: pd.DataFrame) -> pd.Series:
    for col in FRAME_INDEX_COL_CANDIDATES:
        if col in df.columns:
            parsed = df[col].map(_parse_frame_index)
            if parsed.notna().sum() > 0:
                return parsed
    return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")


def _expand_rows_by_segment(
    df: pd.DataFrame,
    frame_to_segments: Mapping[int, Sequence[int]],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["segment_id"])

    work = df.copy()
    work["_frame_index"] = _infer_frame_index_series(work)
    work["__segment_ids"] = work["_frame_index"].apply(
        lambda x: list(frame_to_segments.get(int(x), [])) if pd.notna(x) else []
    )
    work = work.explode("__segment_ids")
    work["segment_id"] = pd.to_numeric(work["__segment_ids"], errors="coerce")
    work = work.dropna(subset=["segment_id"]).copy()
    if work.empty:
        return work
    work["segment_id"] = work["segment_id"].astype(int)
    return work.drop(columns=["__segment_ids"])


def _detect_numeric_cols(df: pd.DataFrame, excluded: Iterable[str]) -> List[str]:
    excluded_set = set(excluded)
    cols: List[str] = []
    for col in df.columns:
        if col in excluded_set:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            cols.append(col)
    return cols


def _aggregate_numeric_group(
    mapped_df: pd.DataFrame,
    group_name: str,
    stats: Sequence[str],
    force_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if mapped_df.empty:
        return pd.DataFrame(columns=["segment_id"]), [], []

    numeric_cols = _detect_numeric_cols(
        mapped_df,
        excluded=(
            "segment_id",
            "_frame_index",
            "Frame",
            "frame",
            "frame_name",
            "FrameNum",
            "frame_num",
            "frame_idx",
            "frame_index",
            "full_response",
        ),
    )

    if force_cols:
        force_set = set(force_cols)
        numeric_cols = [c for c in numeric_cols if c in force_set]

    if not numeric_cols:
        return pd.DataFrame(columns=["segment_id"]), [], []

    work = mapped_df[["segment_id"] + numeric_cols].copy()
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = work.groupby("segment_id")[numeric_cols].agg(list(stats))
    out = grouped.reset_index()

    rename_map: Dict[str, str] = {}
    feature_columns: List[str] = []
    if isinstance(grouped.columns, pd.MultiIndex):
        for col, st in grouped.columns:
            old = (col, st)
            new = f"{group_name}__{_safe_col_name(col)}__{st}"
            rename_map[old] = new
            feature_columns.append(new)
    else:
        for col in grouped.columns:
            old = col
            new = f"{group_name}__{_safe_col_name(str(col))}__mean"
            rename_map[old] = new
            feature_columns.append(new)

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["segment_id"] + feature_columns
    else:
        out = out.rename(columns=rename_map)

    return out, feature_columns, numeric_cols


def _build_default_audio_schema(panns_export_dims: int = 16) -> Dict[str, List[str]]:
    events = [
        "top_k_events",
        "event_class_distribution_json",
        "audio_event_row_count",
        "group_ratio_traffic",
        "group_ratio_human",
        "group_ratio_nature",
        "group_ratio_mechanical",
        "group_ratio_other",
    ]
    signal = [
        "linked_audio_start_time_sec",
        "linked_audio_end_time_sec",
        "waveform_available",
        "waveform_loader",
        "sample_rate",
        "roughness_proxy_method",
        "rms_energy",
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flatness",
        "spectral_flux",
        "loudness_proxy_db",
        "sharpness_proxy",
        "roughness_proxy",
    ]
    embedding = [
        "panns_available",
        "panns_unavailable_reason",
        "panns_embedding_dim",
        "panns_emb_mean",
        "panns_emb_std",
        "panns_emb_l2",
    ] + [f"panns_emb_{i:03d}" for i in range(max(1, int(panns_export_dims)))]
    return {
        "audio_events": events,
        "audio_signal": signal,
        "audio_embedding": embedding,
    }


def _split_audio_groups(cols: Sequence[str], panns_export_dims: int = 16) -> Dict[str, List[str]]:
    default = _build_default_audio_schema(panns_export_dims=panns_export_dims)
    event_set = set(default["audio_events"])
    embedding_prefix = "panns_"
    core_skip = {"segment_id", "start_time_sec", "end_time_sec", "center_time_sec", "included_frame_count"}

    groups = {"audio_events": [], "audio_signal": [], "audio_embedding": []}
    for col in cols:
        if col in core_skip:
            continue
        if col.startswith(embedding_prefix):
            groups["audio_embedding"].append(col)
        elif col in event_set:
            groups["audio_events"].append(col)
        else:
            groups["audio_signal"].append(col)

    for g in groups:
        groups[g] = sorted(set(groups[g]))
    return groups


def _build_ai_global_features(ai_summary_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if ai_summary_df.empty:
        return out
    for _, row in ai_summary_df.iterrows():
        label = row.get("Activity_EN", row.get("Activity", "activity"))
        label_key = _safe_col_name(str(label))
        for metric in ("Mean_Score", "Max_Score", "Min_Score", "Is_Suitable"):
            if metric not in ai_summary_df.columns:
                continue
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            out[f"ai_activity__summary_{label_key}_{_safe_col_name(metric)}"] = float(value)
    return out


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def build_segment_feature_warehouse(
    video_dir: str,
    panns_export_dims: int = 16,
) -> Dict[str, Any]:
    """
    Build one-row-per-segment multimodal feature warehouse.

    Outputs:
    - `<video_dir>/fusion/segment_feature_table.csv`
    - `<video_dir>/fusion/feature_dictionary.json`
    - `<video_dir>/fusion/feature_quality_report.json`
    """
    vdir = Path(video_dir)
    fusion_dir = vdir / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    segments_path = vdir / "segments" / "segment_manifest.csv"
    segments_df = _load_segments(segments_path)
    frame_to_segments = _build_frame_to_segments_map(segments_df)

    table = segments_df[
        ["segment_id", "start_time_sec", "end_time_sec", "center_time_sec", "included_frame_count"]
    ].copy()

    group_columns: Dict[str, List[str]] = {k: [] for k in REQUIRED_GROUPS}
    source_status: Dict[str, Dict[str, Any]] = {}

    frame_sources = [
        (
            "visual_semantic",
            vdir / "stats" / "visual_elements" / "detailed_categories_proportion.csv",
            ("mean", "std"),
            None,
        ),
        (
            "visual_major",
            vdir / "stats" / "visual_elements" / "major_categories_proportion.csv",
            ("mean", "std"),
            None,
        ),
        (
            "green_view",
            vdir / "stats" / "green_view" / "green_view_index.csv",
            ("mean", "std", "var"),
            None,
        ),
        (
            "emotion",
            vdir / "stats" / "emotion" / "emotion_scores.csv",
            ("mean", "std", "var"),
            None,
        ),
        (
            "people",
            vdir / "stats" / "people_count" / "people_count.csv",
            ("mean", "max", "std"),
            None,
        ),
        (
            "color",
            vdir / "stats" / "color_analysis" / "color_categories_proportion.csv",
            ("mean", "std"),
            None,
        ),
    ]

    for group_name, path, stats, force_cols in frame_sources:
        df, err = _read_csv_safe(path)
        if err or df is None:
            source_status[group_name] = {
                "exists": False,
                "path": path.as_posix(),
                "warning": err,
            }
            logger.warning("[%s] source missing/unreadable: %s", group_name, err)
            continue

        mapped = _expand_rows_by_segment(df, frame_to_segments=frame_to_segments)
        if mapped.empty:
            source_status[group_name] = {
                "exists": True,
                "path": path.as_posix(),
                "rows": int(len(df)),
                "warning": "no rows mapped to segments",
            }
            logger.warning("[%s] no rows mapped to segments", group_name)
            continue

        agg_df, created_cols, used_cols = _aggregate_numeric_group(
            mapped_df=mapped,
            group_name=group_name,
            stats=stats,
            force_cols=force_cols,
        )
        if agg_df.empty or not created_cols:
            source_status[group_name] = {
                "exists": True,
                "path": path.as_posix(),
                "rows": int(len(df)),
                "mapped_rows": int(len(mapped)),
                "warning": "no numeric columns aggregated",
            }
            logger.warning("[%s] no numeric columns aggregated", group_name)
            continue

        table = table.merge(agg_df, on="segment_id", how="left")
        group_columns[group_name] = created_cols
        source_status[group_name] = {
            "exists": True,
            "path": path.as_posix(),
            "rows": int(len(df)),
            "mapped_rows": int(len(mapped)),
            "used_numeric_columns": used_cols,
            "aggregated_feature_count": int(len(created_cols)),
        }


    ai_scores_path = vdir / "ai_evaluation" / "activity_scores.csv"
    ai_scores_df, ai_scores_err = _read_csv_safe(ai_scores_path)
    if ai_scores_df is None:
        logger.warning("[ai_activity] activity_scores missing: %s", ai_scores_err)
        ai_placeholder_cols = []
        for score_col in AI_DEFAULT_SCORE_COLUMNS:
            for st in ("mean", "max", "std"):
                ai_placeholder_cols.append(f"ai_activity__{_safe_col_name(score_col)}__{st}")
        table = _ensure_columns(table, ai_placeholder_cols)
        group_columns["ai_activity"].extend(ai_placeholder_cols)
        source_status["ai_activity"] = {
            "exists": False,
            "path": ai_scores_path.as_posix(),
            "warning": ai_scores_err,
            "placeholder_columns": ai_placeholder_cols,
        }
    else:
        mapped_ai = _expand_rows_by_segment(ai_scores_df, frame_to_segments=frame_to_segments)
        ai_score_cols = [c for c in ai_scores_df.columns if str(c).endswith("_score")]
        agg_ai, ai_cols, used_ai = _aggregate_numeric_group(
            mapped_df=mapped_ai,
            group_name="ai_activity",
            stats=("mean", "max", "std"),
            force_cols=ai_score_cols if ai_score_cols else None,
        )
        if not agg_ai.empty and ai_cols:
            table = table.merge(agg_ai, on="segment_id", how="left")
            group_columns["ai_activity"].extend(ai_cols)
        source_status["ai_activity"] = {
            "exists": True,
            "path": ai_scores_path.as_posix(),
            "rows": int(len(ai_scores_df)),
            "mapped_rows": int(len(mapped_ai)),
            "used_numeric_columns": used_ai,
            "aggregated_feature_count": int(len(ai_cols)),
        }


    ai_summary_path = vdir / "ai_evaluation" / "activity_suitable_summary.csv"
    ai_summary_df, ai_summary_err = _read_csv_safe(ai_summary_path)
    if ai_summary_df is not None:
        global_ai = _build_ai_global_features(ai_summary_df)
        for col, value in global_ai.items():
            table[col] = value
        group_columns["ai_activity"].extend(list(global_ai.keys()))
        source_status["ai_activity_summary"] = {
            "exists": True,
            "path": ai_summary_path.as_posix(),
            "rows": int(len(ai_summary_df)),
            "broadcast_feature_count": int(len(global_ai)),
        }
    else:
        source_status["ai_activity_summary"] = {
            "exists": False,
            "path": ai_summary_path.as_posix(),
            "warning": ai_summary_err,
        }


    soundscape_path = vdir / "soundscape" / "audio_segment_features.csv"
    sound_df, sound_err = _read_csv_safe(soundscape_path)
    if sound_df is None:
        logger.warning("[soundscape] feature table missing: %s", sound_err)
        default_audio = _build_default_audio_schema(panns_export_dims=panns_export_dims)
        missing_audio_cols: List[str] = []
        for group in ("audio_events", "audio_signal", "audio_embedding"):
            for raw_col in default_audio[group]:
                col = f"{group}__{_safe_col_name(raw_col)}"
                missing_audio_cols.append(col)
                group_columns[group].append(col)
        table = _ensure_columns(table, missing_audio_cols)
        source_status["soundscape"] = {
            "exists": False,
            "path": soundscape_path.as_posix(),
            "warning": sound_err,
            "placeholder_feature_count": int(len(missing_audio_cols)),
        }
    else:
        if "segment_id" not in sound_df.columns:
            logger.warning("[soundscape] missing segment_id; fallback to placeholders")
            sound_df = None
            sound_err = "missing segment_id column"
            default_audio = _build_default_audio_schema(panns_export_dims=panns_export_dims)
            missing_audio_cols = []
            for group in ("audio_events", "audio_signal", "audio_embedding"):
                for raw_col in default_audio[group]:
                    col = f"{group}__{_safe_col_name(raw_col)}"
                    missing_audio_cols.append(col)
                    group_columns[group].append(col)
            table = _ensure_columns(table, missing_audio_cols)
            source_status["soundscape"] = {
                "exists": False,
                "path": soundscape_path.as_posix(),
                "warning": sound_err,
                "placeholder_feature_count": int(len(missing_audio_cols)),
            }
        else:
            sound_df = sound_df.copy()
            sound_df = sound_df.sort_values("segment_id").drop_duplicates(subset=["segment_id"], keep="last")
            split_groups = _split_audio_groups(sound_df.columns.tolist(), panns_export_dims=panns_export_dims)
            rename_map: Dict[str, str] = {}
            for group, cols in split_groups.items():
                for raw_col in cols:
                    prefixed = f"{group}__{_safe_col_name(raw_col)}"
                    rename_map[raw_col] = prefixed
                    group_columns[group].append(prefixed)
            keep_cols = ["segment_id"] + list(rename_map.keys())
            sound_merge = sound_df[keep_cols].rename(columns=rename_map)
            table = table.merge(sound_merge, on="segment_id", how="left")
            source_status["soundscape"] = {
                "exists": True,
                "path": soundscape_path.as_posix(),
                "rows": int(len(sound_df)),
                "aggregated_feature_count": int(len(rename_map)),
            }


    for group in REQUIRED_GROUPS:
        deduped = []
        seen = set()
        for col in group_columns.get(group, []):
            if col in seen:
                continue
            deduped.append(col)
            seen.add(col)
        group_columns[group] = deduped

    feature_cols = [c for c in table.columns if c not in {"segment_id"}]
    missing_ratio = table[feature_cols].isna().mean().to_dict() if feature_cols else {}
    columns_all_missing = [c for c, r in missing_ratio.items() if float(r) >= 1.0]

    csv_path = fusion_dir / "segment_feature_table.csv"
    dict_path = fusion_dir / "feature_dictionary.json"
    quality_path = fusion_dir / "feature_quality_report.json"

    table.to_csv(csv_path, index=False, encoding="utf-8")

    feature_counts_by_group = {k: int(len(v)) for k, v in group_columns.items()}
    dictionary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_dir": vdir.as_posix(),
        "source_files": source_status,
        "segment_columns": [
            "segment_id",
            "start_time_sec",
            "end_time_sec",
            "center_time_sec",
            "included_frame_count",
        ],
        "feature_groups": {
            group: {
                "feature_count": int(len(group_columns[group])),
                "columns": group_columns[group],
            }
            for group in REQUIRED_GROUPS
        },
        "feature_counts_by_group": feature_counts_by_group,
        "table_schema": list(table.columns),
    }
    dict_path.write_text(json.dumps(dictionary, ensure_ascii=False, indent=2), encoding="utf-8")

    quality = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_dir": vdir.as_posix(),
        "rows": int(len(table)),
        "columns": int(len(table.columns)),
        "feature_counts_by_group": feature_counts_by_group,
        "source_status": source_status,
        "missing_value_ratio_by_column": {k: float(v) for k, v in missing_ratio.items()},
        "columns_all_missing": columns_all_missing,
        "segments_with_any_missing": int(table.isna().any(axis=1).sum()) if len(table) else 0,
        "missing_data_handling": {
            "frame_level_sources": "rows are mapped to overlapping segments by included_frame_indices and aggregated.",
            "ai_missing": "pipeline keeps running; ai_activity placeholder columns are created when activity_scores.csv is absent.",
            "soundscape_missing": "pipeline keeps running; audio group columns are kept with NaN placeholders.",
        },
    }
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "fusion feature warehouse done | rows=%d cols=%d groups=%s",
        len(table),
        len(table.columns),
        feature_counts_by_group,
    )
    return {
        "segment_feature_table_csv": csv_path.as_posix(),
        "feature_dictionary_json": dict_path.as_posix(),
        "feature_quality_report_json": quality_path.as_posix(),
        "rows": int(len(table)),
        "columns": int(len(table.columns)),
        "feature_counts_by_group": feature_counts_by_group,
        "preview_rows": table.head(5).to_dict("records"),
        "table_schema": list(table.columns),
    }

