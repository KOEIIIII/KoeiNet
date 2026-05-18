


"""Build problem episodes by merging nearby high-priority segments."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .export_utils import parse_json_list

logger = logging.getLogger("deliverable.episode_builder")

DEFAULT_TOP_K = 12
DEFAULT_OVERLAP_RATIO = 0.20
REPRESENTATIVE_TOLERANCE = 0.02


def _normalize_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    required = {"segment_id", "start_time_sec", "end_time_sec"}
    missing = required - set(manifest_df.columns)
    if missing:
        raise ValueError(f"segment_manifest missing required columns: {sorted(missing)}")
    out = manifest_df.copy()
    out["segment_id"] = pd.to_numeric(out["segment_id"], errors="coerce")
    out = out.dropna(subset=["segment_id"]).copy()
    out["segment_id"] = out["segment_id"].astype(int)
    sort_cols = [c for c in ["start_time_sec", "center_time_sec", "segment_id"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def select_priority_segments(
    ranking_df: pd.DataFrame,
    *,
    top_k: Optional[int],
    top_percent: Optional[float],
    priority_threshold: Optional[float],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if ranking_df.empty:
        return ranking_df.copy(), {"selection_mode": "empty", "selected_segments": 0}

    rank = ranking_df.copy()
    rank["segment_id"] = pd.to_numeric(rank["segment_id"], errors="coerce")
    rank = rank.dropna(subset=["segment_id"]).copy()
    rank["segment_id"] = rank["segment_id"].astype(int)
    rank = rank.sort_values(["priority_rank", "priority_score"], ascending=[True, False]).reset_index(drop=True)

    selected_mask = pd.Series(False, index=rank.index)
    selection_notes: List[str] = []

    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), len(rank))
        selected_mask |= rank.index < k
        selection_notes.append(f"top_k={k}")
    if top_percent is not None and float(top_percent) > 0:
        frac = min(max(float(top_percent), 0.0), 1.0)
        k = max(1, int(np.ceil(len(rank) * frac)))
        selected_mask |= rank.index < k
        selection_notes.append(f"top_percent={frac:.3f}")
    if priority_threshold is not None:
        selected_mask |= pd.to_numeric(rank["priority_score"], errors="coerce").fillna(-np.inf) >= float(priority_threshold)
        selection_notes.append(f"priority_threshold={float(priority_threshold):.4f}")

    if not selected_mask.any():
        fallback_k = min(DEFAULT_TOP_K, len(rank))
        selected_mask = rank.index < fallback_k
        selection_notes.append(f"default_top_k={fallback_k}")

    selected = rank[selected_mask].copy()
    selected = selected.sort_values(["start_time_sec", "segment_id"]).reset_index(drop=True)
    meta = {
        "selection_mode": " | ".join(selection_notes),
        "selected_segments": int(len(selected)),
        "selected_segment_ids": selected["segment_id"].astype(int).tolist(),
    }
    logger.info("deliverable selection | %s", meta)
    return selected, meta


def _interval_overlap_ratio(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    overlap = max(0.0, min(float(end_a), float(end_b)) - max(float(start_a), float(start_b)))
    shorter = max(1e-6, min(float(end_a) - float(start_a), float(end_b) - float(start_b)))
    return float(overlap / shorter)


def _should_merge(
    previous_end: float,
    next_start: float,
    overlap_ratio: float,
    *,
    max_gap_seconds: float,
    overlap_threshold: float,
) -> bool:
    gap = float(next_start) - float(previous_end)
    return bool(gap <= float(max_gap_seconds) or overlap_ratio >= float(overlap_threshold))


def _representative_segment(group_df: pd.DataFrame) -> int:
    episode_center = float((group_df["start_time_sec"].min() + group_df["end_time_sec"].max()) / 2.0)
    best_score = float(group_df["priority_score"].max())
    candidates = group_df[group_df["priority_score"] >= best_score - REPRESENTATIVE_TOLERANCE].copy()
    candidates["distance_to_center"] = (pd.to_numeric(candidates["center_time_sec"], errors="coerce") - episode_center).abs()
    candidates = candidates.sort_values(
        ["distance_to_center", "priority_rank", "segment_id"],
        ascending=[True, True, True],
    )
    return int(candidates["segment_id"].iloc[0])


def _frame_triplet_for_episode(
    group_df: pd.DataFrame,
    manifest_map: Dict[int, Dict[str, Any]],
    representative_segment_id: int,
) -> Tuple[int, str, List[int], List[str]]:
    group_df = group_df.sort_values(["start_time_sec", "segment_id"]).reset_index(drop=True)
    start_seg = int(group_df["segment_id"].iloc[0])
    end_seg = int(group_df["segment_id"].iloc[-1])
    rep_row = manifest_map.get(int(representative_segment_id), {})
    start_row = manifest_map.get(start_seg, {})
    end_row = manifest_map.get(end_seg, {})

    start_indices = [int(x) for x in parse_json_list(start_row.get("included_frame_indices")) if str(x).strip()]
    rep_indices = [int(x) for x in parse_json_list(rep_row.get("included_frame_indices")) if str(x).strip()]
    end_indices = [int(x) for x in parse_json_list(end_row.get("included_frame_indices")) if str(x).strip()]
    start_paths = [str(x) for x in parse_json_list(start_row.get("included_frame_paths")) if str(x).strip()]
    rep_paths = [str(x) for x in parse_json_list(rep_row.get("included_frame_paths")) if str(x).strip()]
    end_paths = [str(x) for x in parse_json_list(end_row.get("included_frame_paths")) if str(x).strip()]

    triplet_idx: List[int] = []
    triplet_paths: List[str] = []
    candidates = [
        (
            start_indices[0] if start_indices else None,
            start_paths[0] if start_paths else None,
        ),
        (
            rep_indices[len(rep_indices) // 2] if rep_indices else None,
            rep_paths[len(rep_paths) // 2] if rep_paths else None,
        ),
        (
            end_indices[-1] if end_indices else None,
            end_paths[-1] if end_paths else None,
        ),
    ]
    for frame_idx, frame_path in candidates:
        if frame_idx is None or frame_path is None:
            continue
        if int(frame_idx) in triplet_idx or str(frame_path) in triplet_paths:
            continue
        triplet_idx.append(int(frame_idx))
        triplet_paths.append(str(frame_path))

    hero_idx = triplet_idx[1] if len(triplet_idx) >= 2 else (triplet_idx[0] if triplet_idx else -1)
    hero_path = triplet_paths[1] if len(triplet_paths) >= 2 else (triplet_paths[0] if triplet_paths else "")
    return hero_idx, hero_path, triplet_idx, triplet_paths


def build_problem_episodes(
    selected_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    *,
    max_gap_seconds: float,
    min_episode_segments: int = 1,
    merge_if_overlap_ratio_exceeds: float = DEFAULT_OVERLAP_RATIO,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    selected = selected_df.copy()
    if selected.empty:
        empty = pd.DataFrame(
            columns=[
                "episode_id",
                "start_time_sec",
                "end_time_sec",
                "duration_sec",
                "segment_ids",
                "n_segments",
                "representative_segment_id",
                "hero_frame_index",
                "hero_frame_path",
                "representative_frame_indices",
                "representative_frame_paths",
            ]
        )
        return empty, []

    manifest = _normalize_manifest(manifest_df)
    manifest_map = manifest.set_index("segment_id").to_dict(orient="index")
    merged = selected.merge(
        manifest[["segment_id", "start_time_sec", "end_time_sec", "center_time_sec", "included_frame_indices", "included_frame_paths"]],
        on="segment_id",
        how="left",
        suffixes=("", "_manifest"),
    )
    merged["start_time_sec"] = merged["start_time_sec_manifest"].fillna(merged["start_time_sec"])
    merged["end_time_sec"] = merged["end_time_sec_manifest"].fillna(merged["end_time_sec"])
    if "center_time_sec" not in merged.columns:
        merged["center_time_sec"] = (merged["start_time_sec"] + merged["end_time_sec"]) / 2.0
    merged = merged.sort_values(["start_time_sec", "segment_id"]).reset_index(drop=True)

    episode_rows: List[Dict[str, Any]] = []
    episode_jsonl: List[Dict[str, Any]] = []
    current_ids: List[int] = []
    current_start = None
    current_end = None

    def flush_episode(segment_ids: List[int]) -> None:
        if len(segment_ids) < int(min_episode_segments):
            return
        group_df = merged[merged["segment_id"].isin(segment_ids)].copy()
        group_df = group_df.sort_values(["start_time_sec", "segment_id"]).reset_index(drop=True)
        episode_index = len(episode_rows) + 1
        episode_id = f"episode_{episode_index:03d}"
        representative_segment_id = _representative_segment(group_df)
        hero_idx, hero_path, frame_indices, frame_paths = _frame_triplet_for_episode(
            group_df,
            manifest_map=manifest_map,
            representative_segment_id=representative_segment_id,
        )
        episode_record = {
            "episode_id": episode_id,
            "start_time_sec": float(group_df["start_time_sec"].min()),
            "end_time_sec": float(group_df["end_time_sec"].max()),
            "duration_sec": float(group_df["end_time_sec"].max() - group_df["start_time_sec"].min()),
            "segment_ids": [int(x) for x in group_df["segment_id"].astype(int).tolist()],
            "n_segments": int(len(group_df)),
            "representative_segment_id": int(representative_segment_id),
            "hero_frame_index": int(hero_idx) if hero_idx >= 0 else None,
            "hero_frame_path": str(hero_path),
            "representative_frame_indices": [int(x) for x in frame_indices],
            "representative_frame_paths": [str(x) for x in frame_paths],
        }
        episode_rows.append(episode_record)
        episode_jsonl.append(dict(episode_record))

    for _, row in merged.iterrows():
        segment_id = int(row["segment_id"])
        start = float(row["start_time_sec"])
        end = float(row["end_time_sec"])
        if not current_ids:
            current_ids = [segment_id]
            current_start = start
            current_end = end
            continue
        overlap_ratio = _interval_overlap_ratio(float(current_start), float(current_end), start, end)
        if _should_merge(
            float(current_end),
            start,
            overlap_ratio,
            max_gap_seconds=float(max_gap_seconds),
            overlap_threshold=float(merge_if_overlap_ratio_exceeds),
        ):
            current_ids.append(segment_id)
            current_end = max(float(current_end), end)
        else:
            flush_episode(current_ids)
            current_ids = [segment_id]
            current_start = start
            current_end = end
    flush_episode(current_ids)

    episode_df = pd.DataFrame(episode_rows)
    logger.info("deliverable episodes built | n=%s", len(episode_df))
    return episode_df, episode_jsonl
