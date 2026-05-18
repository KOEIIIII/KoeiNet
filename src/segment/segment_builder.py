


"""Build synchronized segment-level manifests from existing frame-level outputs."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import pandas as pd

logger = logging.getLogger("segment.segment_builder")


def parse_frame_index(value: Any) -> Optional[int]:
    """Parse frame index from int/float/string values like `frame_000123`."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(round(value))
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def frame_index_to_time_sec(frame_index: int, frame_skip: int, source_fps: float) -> float:
    """Convert analyzed-frame index to timeline seconds."""
    if source_fps <= 0:
        source_fps = 30.0
    return float(frame_index) * float(frame_skip) / float(source_fps)


def map_time_to_segment(time_sec: float, segments_df: pd.DataFrame) -> Optional[int]:
    """Map timestamp to segment_id."""
    rows = segments_df[
        (segments_df["start_time_sec"] <= time_sec) & (segments_df["end_time_sec"] >= time_sec)
    ]
    if rows.empty:
        return None
    return int(rows.iloc[0]["segment_id"])


def map_frame_index_to_segment(
    frame_index: int,
    segments_df: pd.DataFrame,
    frame_skip: int,
    source_fps: float,
) -> Optional[int]:
    """Map frame index to segment_id using timeline conversion."""
    t = frame_index_to_time_sec(frame_index, frame_skip=frame_skip, source_fps=source_fps)
    return map_time_to_segment(t, segments_df)


def map_frame_row_to_segment(
    row: Mapping[str, Any],
    segments_df: pd.DataFrame,
    frame_skip: int,
    source_fps: float,
    time_col_candidates: Sequence[str] = ("time_sec", "timestamp_sec", "center_time_sec", "start_time_sec"),
    frame_col_candidates: Sequence[str] = ("FrameNum", "frame_num", "frame_idx", "frame_index", "Frame", "frame_name"),
) -> Optional[int]:
    """
    Map any frame-level row into segment_id.

    Priority:
    1) explicit timestamp columns
    2) frame index-like columns
    """
    for col in time_col_candidates:
        if col in row:
            try:
                value = float(row[col])
                if not math.isnan(value):
                    return map_time_to_segment(value, segments_df)
            except Exception:
                pass

    for col in frame_col_candidates:
        if col in row:
            idx = parse_frame_index(row[col])
            if idx is not None:
                return map_frame_index_to_segment(
                    idx,
                    segments_df=segments_df,
                    frame_skip=frame_skip,
                    source_fps=source_fps,
                )
    return None


def attach_segment_id_to_frame_df(
    frame_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    frame_skip: int,
    source_fps: float,
) -> pd.DataFrame:
    """Attach `segment_id` to a frame-level dataframe without mutating input."""
    out = frame_df.copy()
    out["segment_id"] = out.apply(
        lambda r: map_frame_row_to_segment(
            r.to_dict(),
            segments_df=segments_df,
            frame_skip=frame_skip,
            source_fps=source_fps,
        ),
        axis=1,
    )
    return out


def _discover_frame_paths(frames_dir: Path) -> List[Path]:
    return sorted(
        [p for p in frames_dir.glob("frame_*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: parse_frame_index(p.name) if parse_frame_index(p.name) is not None else 10**12,
    )


def _estimate_source_fps(video_dir: Path, video_path_hint: Optional[str]) -> float:
    """
    Estimate source FPS using the best available local source.

    Priority:
    1) input video path (if exists)
    2) processed video in output folder
    3) fallback 30.0
    """
    candidates: List[Path] = []
    if video_path_hint:
        p = Path(video_path_hint)
        if p.exists():
            candidates.append(p)

    video_name = video_dir.name
    candidates.append(video_dir / f"{video_name}_processed.mp4")
    candidates.append(video_dir / f"{video_name}_processed_h264.mp4")

    for candidate in candidates:
        if not candidate.exists():
            continue
        cap = cv2.VideoCapture(str(candidate))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        finally:
            cap.release()
        if fps and fps > 0:
            return float(fps)
    return 30.0


def _bootstrap_frames_from_video(
    video_path_hint: Optional[str],
    frames_dir: Path,
    frame_skip: int,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Extract lightweight timeline frames when legacy `frames/` is missing.

    This keeps segment generation inside the project-native pipeline while
    avoiding a full legacy recompute just to obtain `segment_manifest.csv`.
    """
    if not video_path_hint:
        raise FileNotFoundError(
            f"missing frames directory and no video_path_hint available for bootstrap: {frames_dir.as_posix()}"
        )

    video_path = Path(video_path_hint)
    if not video_path.is_file():
        raise FileNotFoundError(f"video_path_hint not found for frame bootstrap: {video_path.as_posix()}")

    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"unable to open video for frame bootstrap: {video_path.as_posix()}")

    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_index = -1
    try:
        if progress_callback and total_frames > 0:
            progress_callback(0, total_frames, "segment | bootstrap frames")
        while True:
            grabbed = cap.grab()
            if not grabbed:
                break
            frame_index += 1
            if progress_callback and total_frames > 0 and (frame_index == 0 or frame_index % 50 == 0):
                progress_callback(min(frame_index + 1, total_frames), total_frames, "segment | bootstrap frames")
            if frame_skip > 0 and frame_index % int(frame_skip) != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            frame_path = frames_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
    finally:
        cap.release()
    if progress_callback and total_frames > 0:
        progress_callback(total_frames, total_frames, "segment | bootstrap frames")

    if saved_count <= 0:
        raise RuntimeError(f"frame bootstrap produced no frames: {video_path.as_posix()}")

    logger.info(
        "segment bootstrap frames complete | video=%s saved=%d frame_skip=%d fps=%.6f total_frames=%d",
        video_path.as_posix(),
        saved_count,
        int(frame_skip),
        float(fps),
        int(total_frames),
    )
    return {
        "video_path": video_path.as_posix(),
        "frames_dir": frames_dir.as_posix(),
        "saved_frame_count": int(saved_count),
        "frame_skip": int(frame_skip),
        "source_fps": float(fps),
        "total_video_frames": int(total_frames),
    }


def _load_audio_time_sync(audio_events_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load audio timing outputs when available.

    Accepts:
    - audio_events_time_sync_simple.csv
    - audio_events_time_sync.csv
    """
    simple = audio_events_dir / "audio_events_time_sync_simple.csv"
    detailed = audio_events_dir / "audio_events_time_sync.csv"
    path = simple if simple.exists() else detailed if detailed.exists() else None
    if path is None:
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("failed to read audio time-sync csv: %s | %s", path.as_posix(), exc)
        return None

    if "start_time_sec" not in df.columns or "end_time_sec" not in df.columns:
        logger.warning("audio time-sync csv missing start_time_sec/end_time_sec: %s", path.as_posix())
        return None

    df = df.copy()
    df["start_time_sec"] = pd.to_numeric(df["start_time_sec"], errors="coerce")
    df["end_time_sec"] = pd.to_numeric(df["end_time_sec"], errors="coerce")
    df = df.dropna(subset=["start_time_sec", "end_time_sec"])
    return df


def _iter_segment_starts(timeline_end: float, segment_seconds: float, overlap_seconds: float) -> Iterable[float]:
    step = segment_seconds - overlap_seconds
    if step <= 0:
        raise ValueError("segment overlap must be smaller than segment length")

    if timeline_end <= 0:
        yield 0.0
        return

    start = 0.0
    guard = 0
    while start < timeline_end + 1e-9:
        yield float(start)
        start += step
        guard += 1
        if guard > 2_000_000:
            break


def build_segment_manifest(
    video_dir: str,
    frame_skip: int,
    segment_seconds: float = 5.0,
    overlap_seconds: float = 2.5,
    video_path_hint: Optional[str] = None,
    min_frames_warn_ratio: float = 0.35,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Build segment manifest from existing output artifacts.

    Outputs:
    - `<video_dir>/segments/segment_manifest.csv`
    - `<video_dir>/segments/segment_manifest.json`
    """
    vdir = Path(video_dir)
    frames_dir = vdir / "frames"
    segments_dir = vdir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_info: Optional[Dict[str, Any]] = None
    if not frames_dir.is_dir():
        bootstrap_info = _bootstrap_frames_from_video(
            video_path_hint=video_path_hint,
            frames_dir=frames_dir,
            frame_skip=frame_skip,
            progress_callback=progress_callback,
        )

    frame_paths = _discover_frame_paths(frames_dir)
    if not frame_paths:
        bootstrap_info = _bootstrap_frames_from_video(
            video_path_hint=video_path_hint,
            frames_dir=frames_dir,
            frame_skip=frame_skip,
            progress_callback=progress_callback,
        )
        frame_paths = _discover_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"frames directory is empty; cannot build segment manifest: {frames_dir.as_posix()}")

    source_fps = _estimate_source_fps(vdir, video_path_hint=video_path_hint)
    frame_interval_sec = float(frame_skip) / float(source_fps if source_fps > 0 else 30.0)

    indexed_frames: List[Tuple[int, float, str]] = []
    for path in frame_paths:
        idx = parse_frame_index(path.name)
        if idx is None:
            continue
        t = frame_index_to_time_sec(idx, frame_skip=frame_skip, source_fps=source_fps)
        indexed_frames.append((idx, t, path.as_posix()))
    if not indexed_frames:
        raise RuntimeError("no valid frame indices were discovered; cannot build segment manifest")

    indexed_frames = sorted(indexed_frames, key=lambda x: x[0])
    timeline_end = indexed_frames[-1][1] + frame_interval_sec

    audio_df = _load_audio_time_sync(vdir / "audio_events")
    audio_available = audio_df is not None and not audio_df.empty
    if not audio_available:
        logger.warning(
            "audio time-sync artifacts are missing; segment manifest will keep time windows but audio rows remain empty"
        )

    expected_frames = segment_seconds / max(frame_interval_sec, 1e-6)
    warn_threshold = max(1, int(math.floor(expected_frames * min_frames_warn_ratio)))

    rows: List[Dict[str, Any]] = []
    segment_starts = list(
        _iter_segment_starts(
            timeline_end=timeline_end,
            segment_seconds=segment_seconds,
            overlap_seconds=overlap_seconds,
        )
    )
    if progress_callback:
        progress_callback(0, len(segment_starts), "segment | build manifest")
    missing_audio_count = 0
    for seg_id, start_sec in enumerate(segment_starts):
        end_sec = min(start_sec + segment_seconds, timeline_end)
        center_sec = (start_sec + end_sec) / 2.0

        in_seg = [
            (idx, t, path)
            for idx, t, path in indexed_frames
            if (t >= start_sec and (t < end_sec or math.isclose(t, end_sec)))
        ]
        frame_indices = [x[0] for x in in_seg]
        frame_path_list = [x[2] for x in in_seg]

        if len(frame_indices) < warn_threshold:
            logger.warning(
                "segment[%d] has sparse frame coverage: %d (threshold=%d)",
                seg_id,
                len(frame_indices),
                warn_threshold,
            )

        audio_row_count = 0
        audio_start = None
        audio_end = None
        if audio_available and audio_df is not None:
            overlap = audio_df[
                (audio_df["end_time_sec"] >= start_sec) & (audio_df["start_time_sec"] <= end_sec)
            ]
            audio_row_count = int(len(overlap))
            if audio_row_count > 0:
                audio_start = float(overlap["start_time_sec"].min())
                audio_end = float(overlap["end_time_sec"].max())
        if audio_row_count == 0:
            missing_audio_count += 1
            logger.warning("segment[%d] has no aligned audio rows: [%.3f, %.3f]", seg_id, start_sec, end_sec)

        rows.append(
            {
                "segment_id": int(seg_id),
                "start_time_sec": round(float(start_sec), 6),
                "end_time_sec": round(float(end_sec), 6),
                "center_time_sec": round(float(center_sec), 6),
                "included_frame_count": int(len(frame_indices)),
                "included_frame_indices": frame_indices,
                "included_frame_paths": frame_path_list,
                "audio_start_time_sec": audio_start,
                "audio_end_time_sec": audio_end,
                "audio_row_count": int(audio_row_count),
                "audio_alignment_missing": bool(audio_row_count == 0),
            }
        )
        if progress_callback:
            progress_callback(seg_id + 1, len(segment_starts), "segment | build manifest")

        if end_sec >= timeline_end:
            break

    if not rows:
        raise RuntimeError("no segments were generated")

    csv_path = segments_dir / "segment_manifest.csv"
    json_path = segments_dir / "segment_manifest.json"

    csv_df = pd.DataFrame(rows).copy()
    csv_df["included_frame_indices"] = csv_df["included_frame_indices"].apply(
        lambda v: json.dumps(v, ensure_ascii=False)
    )
    csv_df["included_frame_paths"] = csv_df["included_frame_paths"].apply(
        lambda v: json.dumps(v, ensure_ascii=False)
    )
    csv_df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    frame_counts = [int(r["included_frame_count"]) for r in rows]
    qa_summary = {
        "total_segments": int(len(rows)),
        "mean_frames_per_segment": float(sum(frame_counts) / len(frame_counts)),
        "min_frames_per_segment": int(min(frame_counts)),
        "max_frames_per_segment": int(max(frame_counts)),
        "missing_audio_count": int(missing_audio_count),
    }

    logger.info(
        "segment manifest complete | segments=%d mean_frames=%.2f min=%d max=%d missing_audio=%d",
        qa_summary["total_segments"],
        qa_summary["mean_frames_per_segment"],
        qa_summary["min_frames_per_segment"],
        qa_summary["max_frames_per_segment"],
        qa_summary["missing_audio_count"],
    )

    preview = pd.DataFrame(rows).head(5).to_dict("records")
    return {
        "csv_path": csv_path.as_posix(),
        "json_path": json_path.as_posix(),
        "source_fps": float(source_fps),
        "frame_skip": int(frame_skip),
        "segment_seconds": float(segment_seconds),
        "segment_overlap": float(overlap_seconds),
        "frames_dir": frames_dir.as_posix(),
        "frame_bootstrap_used": bool(bootstrap_info is not None),
        "frame_bootstrap": bootstrap_info,
        "qa_summary": qa_summary,
        "preview_rows": preview,
    }
