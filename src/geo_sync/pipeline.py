


"""Utilities for aligning sampled video frames with GPS tracks."""

from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import pandas as pd

from .coords import gcj02_to_wgs84_approx


SUPPORTED_VIDEO_EXTENSIONS: Sequence[str] = (".mp4", ".mov", ".avi", ".insv", ".mkv")
GPS_REQUIRED_COLUMNS: Sequence[str] = ("groupTime", "gps_longitude", "gps_latitude")
EFFECTIVE_OVERLAP_GAP_SEC = 60.0
OFFSET_SCAN_RANGE_SEC = 300
OFFSET_SCAN_STEP_SEC = 1
OFFSET_SCAN_SAMPLE_COUNT = 31
SOURCE_COORDINATE_SYSTEM = "GCJ-02"
DERIVED_COORDINATE_SYSTEM = "WGS84"
COORDINATE_SYSTEM_SOURCE = "user_assertion"
WGS84_CONVERSION_METHOD = "iterative_approx_inverse_from_gcj02"
WGS84_CONVERSION_NOTE = (
    "Approximate inverse transform from GCJ-02 to WGS84 for downstream GIS interoperability; "
    "original GCJ-02 values remain authoritative."
)
UNQUALIFIED_COORD_FIELDS_REMOVED = True


def _parse_frame_index_from_name(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def _discover_existing_frame_paths(frames_dir: Path) -> List[Path]:
    if not frames_dir.is_dir():
        return []
    frame_paths = [
        p
        for p in frames_dir.glob("frame_*.*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(
        frame_paths,
        key=lambda p: (
            _parse_frame_index_from_name(p.name) if _parse_frame_index_from_name(p.name) is not None else 10**12,
            p.name,
        ),
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_timezone(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def parse_iso_datetime(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("empty datetime text")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return _ensure_timezone(dt).astimezone(timezone.utc)


def _convert_gcj_to_wgs84_nullable(lon: Any, lat: Any) -> Tuple[Optional[float], Optional[float]]:
    if lon is None or lat is None:
        return None, None
    if pd.isna(lon) or pd.isna(lat):
        return None, None
    lon_wgs84, lat_wgs84 = gcj02_to_wgs84_approx(float(lon), float(lat))
    return float(lon_wgs84), float(lat_wgs84)


def _ffprobe_json(video_path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def probe_video(video_path: Path) -> Dict[str, Any]:
    data = _ffprobe_json(video_path)
    streams = data.get("streams", [])
    format_info = data.get("format", {})
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    primary = video_streams[0] if video_streams else {}

    creation_time = None
    creation_source = None
    for source_name, tags in (
        ("stream_tags", primary.get("tags", {}) or {}),
        ("format_tags", format_info.get("tags", {}) or {}),
    ):
        if tags.get("creation_time"):
            creation_time = str(tags["creation_time"])
            creation_source = source_name
            break

    avg_frame_rate = str(primary.get("avg_frame_rate") or "")
    fps = 0.0
    if avg_frame_rate and "/" in avg_frame_rate:
        num, den = avg_frame_rate.split("/", 1)
        try:
            fps = float(num) / float(den)
        except Exception:
            fps = 0.0
    else:
        try:
            fps = float(avg_frame_rate)
        except Exception:
            fps = 0.0

    duration_sec = 0.0
    for candidate in (primary.get("duration"), format_info.get("duration")):
        try:
            duration_sec = float(candidate)
            if duration_sec > 0:
                break
        except Exception:
            continue

    return {
        "video_path": str(video_path.as_posix()),
        "video_stream_count": int(len(video_streams)),
        "audio_stream_count": int(len(audio_streams)),
        "creation_time": creation_time,
        "creation_time_source": creation_source,
        "duration_sec": float(duration_sec),
        "fps": float(fps),
        "time_base": str(primary.get("time_base") or ""),
        "codec_name": str(primary.get("codec_name") or ""),
        "width": int(primary.get("width") or 0),
        "height": int(primary.get("height") or 0),
        "format_name": str(format_info.get("format_name") or ""),
        "format_long_name": str(format_info.get("format_long_name") or ""),
        "stream_tags": dict(primary.get("tags", {}) or {}),
        "format_tags": dict(format_info.get("tags", {}) or {}),
        "side_data_list": list(primary.get("side_data_list") or []),
    }


def _load_sidecar(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    sidecar_path = Path(path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"sidecar file not found: {sidecar_path.as_posix()}")
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def _parse_filename_start_time(video_path: Path, tz_offset_hours: float) -> Optional[datetime]:
    match = re.search(r"(\d{8})_(\d{6})", video_path.name)
    if not match:
        return None
    naive = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    hours = int(math.trunc(tz_offset_hours))
    minutes = int(round((float(tz_offset_hours) - float(hours)) * 60.0))
    tz = timezone(timedelta(hours=hours, minutes=minutes))
    return naive.replace(tzinfo=tz).astimezone(timezone.utc)


def resolve_video_start_time(
    video_path: Path,
    probe: Mapping[str, Any],
    explicit_start_time: Optional[str] = None,
    sidecar_path: Optional[str] = None,
    time_offset_seconds: float = 0.0,
    filename_tz_offset_hours: float = 8.0,
) -> Dict[str, Any]:
    sidecar = _load_sidecar(sidecar_path)

    if explicit_start_time:
        base_time = parse_iso_datetime(explicit_start_time)
        base_source = "explicit_arg"
    elif sidecar.get("start_time"):
        base_time = parse_iso_datetime(str(sidecar["start_time"]))
        base_source = "sidecar.start_time"
    elif probe.get("creation_time"):
        base_time = parse_iso_datetime(str(probe["creation_time"]))
        base_source = str(probe.get("creation_time_source") or "video_creation_time")
    else:
        tz_hours = float(sidecar.get("filename_tz_offset_hours", filename_tz_offset_hours))
        parsed = _parse_filename_start_time(video_path, tz_offset_hours=tz_hours)
        if parsed is None:
            raise ValueError("unable to resolve video start time from explicit args, sidecar, metadata, or filename")
        base_time = parsed
        base_source = "filename_fallback"

    applied_offset = float(sidecar.get("time_offset_seconds", time_offset_seconds))
    resolved_time = base_time + timedelta(seconds=applied_offset)
    return {
        "resolved_start_time_utc": resolved_time.astimezone(timezone.utc).isoformat(),
        "base_start_time_utc": base_time.astimezone(timezone.utc).isoformat(),
        "start_time_source": base_source,
        "time_offset_seconds": float(applied_offset),
        "sidecar_used": bool(sidecar_path),
    }


def discover_video(video_path: Optional[str], input_dir: str) -> Path:
    if video_path:
        candidate = Path(video_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"video not found: {candidate.as_posix()}")
        return candidate

    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_path.as_posix()}")

    candidates = [
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"no supported video found under: {input_path.as_posix()}")
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return candidates[0]


def load_gps_dataframe(csv_path: Path, export_wgs84: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    missing = [col for col in GPS_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"GPS CSV missing required columns: {missing}")

    raw_rows = int(len(df))
    work = df.copy()
    work["_input_row"] = range(1, len(work) + 1)
    for col in ("groupTime", "gps_longitude", "gps_latitude", "speed", "horizontalAccuracy"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    invalid_coord_mask = (
        work["gps_longitude"].isna()
        | work["gps_latitude"].isna()
        | (work["gps_longitude"] < -180)
        | (work["gps_longitude"] > 180)
        | (work["gps_latitude"] < -90)
        | (work["gps_latitude"] > 90)
    )
    invalid_time_mask = work["groupTime"].isna()
    cleaned = work.loc[~(invalid_coord_mask | invalid_time_mask)].copy()

    was_monotonic = bool(cleaned["groupTime"].is_monotonic_increasing)
    duplicate_times = int(cleaned["groupTime"].duplicated().sum())
    cleaned = cleaned.sort_values(["groupTime", "_input_row"]).drop_duplicates("groupTime", keep="last")
    cleaned["groupTime"] = cleaned["groupTime"].astype("int64")
    cleaned["groupTime_ms"] = cleaned["groupTime"] * 1000
    cleaned["groupTime_utc"] = pd.to_datetime(cleaned["groupTime"], unit="s", utc=True)
    cleaned["gps_dt_sec"] = cleaned["groupTime"].diff()
    cleaned["gps_longitude_gcj02"] = cleaned["gps_longitude"].astype(float)
    cleaned["gps_latitude_gcj02"] = cleaned["gps_latitude"].astype(float)
    if export_wgs84:
        converted = [
            _convert_gcj_to_wgs84_nullable(lon, lat)
            for lon, lat in zip(cleaned["gps_longitude_gcj02"], cleaned["gps_latitude_gcj02"])
        ]
        cleaned["gps_longitude_wgs84"] = [item[0] for item in converted]
        cleaned["gps_latitude_wgs84"] = [item[1] for item in converted]
    else:
        cleaned["gps_longitude_wgs84"] = None
        cleaned["gps_latitude_wgs84"] = None
    cleaned["coordinate_system"] = SOURCE_COORDINATE_SYSTEM
    cleaned["derived_coordinate_system"] = DERIVED_COORDINATE_SYSTEM if export_wgs84 else None

    summary = {
        "raw_rows": raw_rows,
        "clean_rows": int(len(cleaned)),
        "dropped_invalid_rows": int((invalid_coord_mask | invalid_time_mask).sum()),
        "was_monotonic_increasing": was_monotonic,
        "duplicate_groupTime_before_dedup": duplicate_times,
        "groupTime_min": int(cleaned["groupTime"].min()) if not cleaned.empty else None,
        "groupTime_max": int(cleaned["groupTime"].max()) if not cleaned.empty else None,
        "groupTime_interval_sec_stats": (
            cleaned["gps_dt_sec"].dropna().describe().to_dict() if "gps_dt_sec" in cleaned.columns else {}
        ),
        "negative_speed_rows": int((cleaned["speed"] < 0).sum()) if "speed" in cleaned.columns else None,
        "large_gap_gt_60s": int((cleaned["gps_dt_sec"] > 60).sum()) if "gps_dt_sec" in cleaned.columns else None,
        "column_names": list(df.columns),
        "coordinate_system": SOURCE_COORDINATE_SYSTEM,
        "coordinate_system_source": COORDINATE_SYSTEM_SOURCE,
        "derived_coordinate_system": DERIVED_COORDINATE_SYSTEM if export_wgs84 else None,
        "derived_coordinate_method": WGS84_CONVERSION_METHOD if export_wgs84 else None,
        "wgs84_fields_exported": bool(export_wgs84),
    }
    return cleaned.reset_index(drop=True), summary


def _find_surrounding_rows(gps_df: pd.DataFrame, timestamp_ms: int) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    values = gps_df["groupTime_ms"].to_numpy()
    if len(values) == 0:
        return None, None
    pos = int(values.searchsorted(timestamp_ms, side="left"))
    if pos < len(gps_df):
        candidate = gps_df.iloc[pos]
        if int(candidate["groupTime_ms"]) == timestamp_ms:
            return candidate, candidate
        after = candidate
    else:
        after = None
    before = gps_df.iloc[pos - 1] if pos > 0 else None
    return before, after


def _evaluate_match(
    gps_df: pd.DataFrame,
    target_ms: int,
    max_gap_warning_sec: float = 60.0,
    export_wgs84: bool = True,
) -> Dict[str, Any]:
    before, after = _find_surrounding_rows(gps_df, target_ms)

    def _row_payload(row: Optional[pd.Series], prefix: str) -> Dict[str, Any]:
        if row is None:
            return {
                f"{prefix}_groupTime": None,
                f"{prefix}_groupTime_ms": None,
                f"{prefix}_groupTime_utc": None,
                f"{prefix}_gps_longitude_gcj02": None,
                f"{prefix}_gps_latitude_gcj02": None,
                f"{prefix}_gps_longitude_wgs84": None,
                f"{prefix}_gps_latitude_wgs84": None,
                f"{prefix}_speed": None,
                f"{prefix}_horizontalAccuracy": None,
            }
        return {
            f"{prefix}_groupTime": int(row["groupTime"]),
            f"{prefix}_groupTime_ms": int(row["groupTime_ms"]),
            f"{prefix}_groupTime_utc": row["groupTime_utc"].isoformat(),
            f"{prefix}_gps_longitude_gcj02": float(row["gps_longitude_gcj02"]),
            f"{prefix}_gps_latitude_gcj02": float(row["gps_latitude_gcj02"]),
            f"{prefix}_gps_longitude_wgs84": (
                float(row["gps_longitude_wgs84"]) if pd.notna(row["gps_longitude_wgs84"]) else None
            ),
            f"{prefix}_gps_latitude_wgs84": (
                float(row["gps_latitude_wgs84"]) if pd.notna(row["gps_latitude_wgs84"]) else None
            ),
            f"{prefix}_speed": float(row["speed"]) if "speed" in row and pd.notna(row["speed"]) else None,
            f"{prefix}_horizontalAccuracy": (
                float(row["horizontalAccuracy"])
                if "horizontalAccuracy" in row and pd.notna(row["horizontalAccuracy"])
                else None
            ),
        }

    payload = {}
    payload.update(_row_payload(before, "before"))
    payload.update(_row_payload(after, "after"))

    if before is None and after is None:
        payload.update(
            {
                "match_status": "no_gps_available",
                "confidence": "none",
                "bracket_available": False,
                "gap_seconds": None,
                "interp_ratio": None,
                "matched_gps_longitude_gcj02": None,
                "matched_gps_latitude_gcj02": None,
                "matched_gps_longitude_wgs84": None,
                "matched_gps_latitude_wgs84": None,
                "within_effective_overlap_window": False,
                "used_interpolation": False,
            }
        )
        return payload

    if before is not None and after is not None:
        before_ms = int(before["groupTime_ms"])
        after_ms = int(after["groupTime_ms"])
        gap_ms = max(0, after_ms - before_ms)

        if before_ms == after_ms:
            status = "exact"
            confidence = "high"
            ratio = 0.0
            matched_lon_gcj02 = float(before["gps_longitude_gcj02"])
            matched_lat_gcj02 = float(before["gps_latitude_gcj02"])
            used_interpolation = False
        elif gap_ms > 0:
            ratio = min(max((target_ms - before_ms) / gap_ms, 0.0), 1.0)
            matched_lon_gcj02 = float(before["gps_longitude_gcj02"]) + (
                float(after["gps_longitude_gcj02"]) - float(before["gps_longitude_gcj02"])
            ) * ratio
            matched_lat_gcj02 = float(before["gps_latitude_gcj02"]) + (
                float(after["gps_latitude_gcj02"]) - float(before["gps_latitude_gcj02"])
            ) * ratio
            gap_sec = gap_ms / 1000.0
            used_interpolation = True
            if gap_sec <= 15:
                confidence = "high"
            elif gap_sec <= 30:
                confidence = "medium"
            elif gap_sec <= max_gap_warning_sec:
                confidence = "low"
            else:
                confidence = "very_low"
            status = "interpolated" if gap_sec <= max_gap_warning_sec else "interpolated_large_gap"
        else:
            ratio = None
            matched_lon_gcj02 = None
            matched_lat_gcj02 = None
            used_interpolation = False
            status = "invalid_time_window"
            confidence = "none"

        if export_wgs84:
            matched_lon_wgs84, matched_lat_wgs84 = _convert_gcj_to_wgs84_nullable(
                matched_lon_gcj02,
                matched_lat_gcj02,
            )
        else:
            matched_lon_wgs84, matched_lat_wgs84 = None, None
        payload.update(
            {
                "match_status": status,
                "confidence": confidence,
                "bracket_available": True,
                "gap_seconds": gap_ms / 1000.0,
                "interp_ratio": ratio,
                "matched_gps_longitude_gcj02": matched_lon_gcj02,
                "matched_gps_latitude_gcj02": matched_lat_gcj02,
                "matched_gps_longitude_wgs84": matched_lon_wgs84,
                "matched_gps_latitude_wgs84": matched_lat_wgs84,
                "within_effective_overlap_window": bool((gap_ms / 1000.0) <= max_gap_warning_sec),
                "used_interpolation": used_interpolation,
            }
        )
        return payload

    single = before if before is not None else after
    prefix = "before" if before is not None else "after"
    matched_lon_gcj02 = float(single["gps_longitude_gcj02"]) if single is not None else None
    matched_lat_gcj02 = float(single["gps_latitude_gcj02"]) if single is not None else None
    if export_wgs84:
        matched_lon_wgs84, matched_lat_wgs84 = _convert_gcj_to_wgs84_nullable(
            matched_lon_gcj02,
            matched_lat_gcj02,
        )
    else:
        matched_lon_wgs84, matched_lat_wgs84 = None, None
    payload.update(
        {
            "match_status": f"{prefix}_only",
            "confidence": "low",
            "bracket_available": False,
            "gap_seconds": None,
            "interp_ratio": None,
            "matched_gps_longitude_gcj02": matched_lon_gcj02,
            "matched_gps_latitude_gcj02": matched_lat_gcj02,
            "matched_gps_longitude_wgs84": matched_lon_wgs84,
            "matched_gps_latitude_wgs84": matched_lat_wgs84,
            "within_effective_overlap_window": False,
            "used_interpolation": False,
        }
    )
    return payload


def _classify_global_overlap(
    gps_start_ms: int,
    gps_end_ms: int,
    video_start_ms: int,
    video_end_ms: int,
) -> str:
    if video_end_ms < gps_start_ms or video_start_ms > gps_end_ms:
        return "no_overlap"
    if video_start_ms >= gps_start_ms and video_end_ms <= gps_end_ms:
        return "fully_within_gps_range"
    return "partial_overlap_with_gps_range"


def _extract_boundary_diagnostic(label: str, target_ms: int, match: Mapping[str, Any]) -> Dict[str, Any]:
    before_ms = match.get("before_groupTime_ms")
    after_ms = match.get("after_groupTime_ms")
    return {
        "label": label,
        "target_time_utc": datetime.fromtimestamp(target_ms / 1000.0, tz=timezone.utc).isoformat(),
        "match_status": match.get("match_status"),
        "confidence": match.get("confidence"),
        "within_effective_overlap_window": bool(match.get("within_effective_overlap_window", False)),
        "before_groupTime_utc": match.get("before_groupTime_utc"),
        "after_groupTime_utc": match.get("after_groupTime_utc"),
        "before_diff_seconds": ((target_ms - int(before_ms)) / 1000.0) if before_ms is not None else None,
        "after_diff_seconds": ((int(after_ms) - target_ms) / 1000.0) if after_ms is not None else None,
        "gap_seconds": match.get("gap_seconds"),
    }


def _boundary_offset_candidates(boundary_diag: Mapping[str, Any]) -> Dict[str, Any]:
    before_diff = boundary_diag.get("before_diff_seconds")
    after_diff = boundary_diag.get("after_diff_seconds")
    return {
        "align_to_before_seconds": (-float(before_diff)) if before_diff is not None else None,
        "align_to_after_seconds": float(after_diff) if after_diff is not None else None,
    }


def _sample_timepoints(duration_sec: float, sample_count: int) -> List[float]:
    if duration_sec <= 0:
        return [0.0]
    if sample_count <= 1:
        return [0.0, float(duration_sec)]
    step = float(duration_sec) / float(sample_count - 1)
    return [round(i * step, 6) for i in range(sample_count)]


def _diagnose_window_for_start(
    gps_df: pd.DataFrame,
    start_time_utc: datetime,
    duration_sec: float,
    gap_threshold_sec: float,
    sample_count: int,
) -> Dict[str, Any]:
    start_ms = int(round(start_time_utc.timestamp() * 1000.0))
    end_ms = start_ms + int(round(float(duration_sec) * 1000.0))
    gps_start_ms = int(gps_df["groupTime_ms"].min())
    gps_end_ms = int(gps_df["groupTime_ms"].max())

    start_match = _evaluate_match(gps_df, start_ms, max_gap_warning_sec=gap_threshold_sec)
    end_match = _evaluate_match(gps_df, end_ms, max_gap_warning_sec=gap_threshold_sec)

    sample_rows: List[Dict[str, Any]] = []
    for rel_sec in _sample_timepoints(duration_sec=float(duration_sec), sample_count=sample_count):
        sample_ms = start_ms + int(round(rel_sec * 1000.0))
        match = _evaluate_match(gps_df, sample_ms, max_gap_warning_sec=gap_threshold_sec)
        sample_rows.append(
            {
                "relative_time_sec": float(rel_sec),
                "absolute_time_utc": datetime.fromtimestamp(sample_ms / 1000.0, tz=timezone.utc).isoformat(),
                "match_status": match.get("match_status"),
                "confidence": match.get("confidence"),
                "bracket_available": bool(match.get("bracket_available", False)),
                "within_effective_overlap_window": bool(match.get("within_effective_overlap_window", False)),
                "gap_seconds": match.get("gap_seconds"),
            }
        )

    effective_rows = [row for row in sample_rows if bool(row["within_effective_overlap_window"])]
    bracket_rows = [row for row in sample_rows if bool(row["bracket_available"])]
    ineffective_indices = [idx for idx, row in enumerate(sample_rows) if not bool(row["within_effective_overlap_window"])]
    first_effective_idx = next(
        (idx for idx, row in enumerate(sample_rows) if bool(row["within_effective_overlap_window"])),
        None,
    )
    last_effective_idx = next(
        (idx for idx in range(len(sample_rows) - 1, -1, -1) if bool(sample_rows[idx]["within_effective_overlap_window"])),
        None,
    )
    mid_gap = False
    if first_effective_idx is not None and last_effective_idx is not None and last_effective_idx > first_effective_idx:
        for idx in range(first_effective_idx, last_effective_idx + 1):
            if not bool(sample_rows[idx]["within_effective_overlap_window"]):
                mid_gap = True
                break

    gap_values = [float(row["gap_seconds"]) for row in effective_rows if row.get("gap_seconds") is not None]
    mean_gap = (sum(gap_values) / len(gap_values)) if gap_values else None

    return {
        "video_start_utc": start_time_utc.astimezone(timezone.utc).isoformat(),
        "video_end_utc": datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc).isoformat(),
        "video_duration_sec": float(duration_sec),
        "global_overlap_relation": _classify_global_overlap(
            gps_start_ms=gps_start_ms,
            gps_end_ms=gps_end_ms,
            video_start_ms=start_ms,
            video_end_ms=end_ms,
        ),
        "start_boundary": _extract_boundary_diagnostic("video_start", start_ms, start_match),
        "end_boundary": _extract_boundary_diagnostic("video_end", end_ms, end_match),
        "sample_count": int(len(sample_rows)),
        "bracket_success_rate": (len(bracket_rows) / len(sample_rows)) if sample_rows else 0.0,
        "effective_overlap_rate": (len(effective_rows) / len(sample_rows)) if sample_rows else 0.0,
        "effective_sample_count": int(len(effective_rows)),
        "mean_effective_gap_seconds": mean_gap,
        "status_counts": pd.Series([row["match_status"] for row in sample_rows]).value_counts().to_dict()
        if sample_rows
        else {},
        "stable_local_overlap": bool(
            sample_rows
            and len(effective_rows) >= max(1, math.ceil(len(sample_rows) * 0.8))
            and not mid_gap
        ),
        "effective_overlap_window_start_utc": effective_rows[0]["absolute_time_utc"] if effective_rows else None,
        "effective_overlap_window_end_utc": effective_rows[-1]["absolute_time_utc"] if effective_rows else None,
        "head_gap": bool(first_effective_idx not in (None, 0)),
        "tail_gap": bool(last_effective_idx is not None and last_effective_idx != len(sample_rows) - 1),
        "mid_gap": bool(mid_gap),
        "sample_preview": sample_rows[:5] + sample_rows[-5:] if len(sample_rows) > 10 else sample_rows,
        "ineffective_sample_indexes": ineffective_indices,
    }


def _score_offset_candidate(
    gps_df: pd.DataFrame,
    base_start_time_utc: datetime,
    duration_sec: float,
    offset_seconds: int,
    gap_threshold_sec: float,
    sample_count: int,
) -> Dict[str, Any]:
    shifted = base_start_time_utc + timedelta(seconds=float(offset_seconds))
    diag = _diagnose_window_for_start(
        gps_df=gps_df,
        start_time_utc=shifted,
        duration_sec=duration_sec,
        gap_threshold_sec=gap_threshold_sec,
        sample_count=sample_count,
    )
    mean_gap = diag.get("mean_effective_gap_seconds")
    return {
        "offset_seconds": int(offset_seconds),
        "effective_sample_count": int(diag.get("effective_sample_count", 0)),
        "stable_local_overlap": bool(diag.get("stable_local_overlap", False)),
        "effective_overlap_rate": float(diag.get("effective_overlap_rate", 0.0)),
        "mean_effective_gap_seconds": float(mean_gap) if mean_gap is not None else 1e9,
        "start_boundary": diag.get("start_boundary", {}),
        "end_boundary": diag.get("end_boundary", {}),
    }


def _suggest_offset_range(
    gps_df: pd.DataFrame,
    base_start_time_utc: datetime,
    duration_sec: float,
    gap_threshold_sec: float = EFFECTIVE_OVERLAP_GAP_SEC,
    sample_count: int = OFFSET_SCAN_SAMPLE_COUNT,
) -> Dict[str, Any]:
    scores: List[Dict[str, Any]] = []
    for offset in range(-OFFSET_SCAN_RANGE_SEC, OFFSET_SCAN_RANGE_SEC + 1, OFFSET_SCAN_STEP_SEC):
        scores.append(
            _score_offset_candidate(
                gps_df=gps_df,
                base_start_time_utc=base_start_time_utc,
                duration_sec=duration_sec,
                offset_seconds=offset,
                gap_threshold_sec=gap_threshold_sec,
                sample_count=sample_count,
            )
        )

    best = sorted(
        scores,
        key=lambda item: (
            item["effective_sample_count"],
            item["stable_local_overlap"],
            item["effective_overlap_rate"],
            -item["mean_effective_gap_seconds"],
        ),
        reverse=True,
    )[0]

    top_offsets = [
        item
        for item in scores
        if item["effective_sample_count"] == best["effective_sample_count"]
        and item["stable_local_overlap"] == best["stable_local_overlap"]
        and abs(item["mean_effective_gap_seconds"] - best["mean_effective_gap_seconds"]) <= 2.0
    ]
    offset_values = sorted(int(item["offset_seconds"]) for item in top_offsets)

    return {
        "recommended_offset_seconds": int(best["offset_seconds"]),
        "recommended_offset_range_seconds": [int(offset_values[0]), int(offset_values[-1])] if offset_values else None,
        "search_range_seconds": [-OFFSET_SCAN_RANGE_SEC, OFFSET_SCAN_RANGE_SEC],
        "sample_count": int(sample_count),
        "gap_threshold_sec": float(gap_threshold_sec),
        "top_candidates": [
            {
                "offset_seconds": int(item["offset_seconds"]),
                "effective_sample_count": int(item["effective_sample_count"]),
                "stable_local_overlap": bool(item["stable_local_overlap"]),
                "effective_overlap_rate": float(item["effective_overlap_rate"]),
                "mean_effective_gap_seconds": float(item["mean_effective_gap_seconds"]),
            }
            for item in sorted(
                top_offsets,
                key=lambda item: (item["offset_seconds"]),
            )[:10]
        ],
    }


def _candidate_segment_manifest(output_root: Path, video_stem: str, explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.is_file() else None
    default_path = output_root / video_stem / "segments" / "segment_manifest.csv"
    return default_path if default_path.is_file() else None


def _build_virtual_segments(duration_sec: float, segment_seconds: float, overlap_seconds: float) -> pd.DataFrame:
    step = float(segment_seconds) - float(overlap_seconds)
    if step <= 0:
        raise ValueError("segment_overlap must be smaller than segment_seconds")
    rows: List[Dict[str, Any]] = []
    seg_id = 0
    start_sec = 0.0
    while start_sec <= duration_sec + 1e-9:
        end_sec = min(start_sec + float(segment_seconds), duration_sec)
        rows.append(
            {
                "segment_id": int(seg_id),
                "start_time_sec": round(float(start_sec), 6),
                "end_time_sec": round(float(end_sec), 6),
                "center_time_sec": round(float((start_sec + end_sec) / 2.0), 6),
            }
        )
        if end_sec >= duration_sec:
            break
        start_sec += step
        seg_id += 1
    return pd.DataFrame(rows)


def _load_or_build_segments(
    output_root: Path,
    video_stem: str,
    explicit_manifest_path: Optional[str],
    duration_sec: float,
    segment_seconds: float,
    segment_overlap: float,
    use_existing_segments: bool,
) -> Tuple[pd.DataFrame, str, Optional[str]]:
    manifest_path = None
    if use_existing_segments:
        manifest_path = _candidate_segment_manifest(output_root, video_stem, explicit_manifest_path)
    if manifest_path is not None:
        df = pd.read_csv(manifest_path)
        required = {"segment_id", "start_time_sec", "end_time_sec", "center_time_sec"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"segment manifest missing required columns: {sorted(required - set(df.columns))}")
        return df.copy(), "existing_manifest", manifest_path.as_posix()
    return (
        _build_virtual_segments(duration_sec=duration_sec, segment_seconds=segment_seconds, overlap_seconds=segment_overlap),
        "virtual_segments",
        None,
    )


def _map_time_to_segment(time_sec: float, segments_df: pd.DataFrame) -> Optional[int]:
    rows = segments_df[
        (segments_df["start_time_sec"] <= time_sec) & (segments_df["end_time_sec"] >= time_sec)
    ]
    if rows.empty:
        return None
    return int(rows.iloc[0]["segment_id"])


def _annotate_frame(
    frame: Any,
    relative_time_sec: float,
    absolute_time_iso: str,
    match: Mapping[str, Any],
) -> Any:
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    lines = [
        f"t={relative_time_sec:.3f}s",
        f"utc={absolute_time_iso}",
        f"status={match.get('match_status')} confidence={match.get('confidence')}",
        f"matched_gcj02=({match.get('matched_gps_longitude_gcj02')}, {match.get('matched_gps_latitude_gcj02')})",
        f"before={match.get('before_groupTime_utc')}",
        f"after={match.get('after_groupTime_utc')}",
    ]
    y = 30
    for line in lines:
        cv2.putText(out, line, (10, y), font, 0.6, color, 1, cv2.LINE_AA)
        y += 28
    return out


def _build_frame_rows_from_existing_frames(
    *,
    frame_paths: Sequence[Path],
    start_time_utc: datetime,
    fps: float,
    analysis_frame_skip: int,
    gps_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    max_gap_warning_sec: float,
    export_wgs84: bool,
    progress_callback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    start_epoch_ms = int(start_time_utc.timestamp() * 1000)
    if progress_callback:
        progress_callback(0, len(frame_paths), "geo_sync | frame alignment")
    for idx, frame_path in enumerate(frame_paths):
        analysis_frame_index = _parse_frame_index_from_name(frame_path.name)
        if analysis_frame_index is None:
            continue
        source_frame_num = int(analysis_frame_index) * int(max(1, analysis_frame_skip))
        relative_time_sec = float(source_frame_num) / float(fps)
        absolute_ms = start_epoch_ms + int(round(relative_time_sec * 1000.0))
        absolute_utc = datetime.fromtimestamp(absolute_ms / 1000.0, tz=timezone.utc)
        match = _evaluate_match(
            gps_df,
            absolute_ms,
            max_gap_warning_sec=max_gap_warning_sec,
            export_wgs84=export_wgs84,
        )
        segment_id = _map_time_to_segment(relative_time_sec, segments_df)
        rows.append(
            {
                "frame_index": int(analysis_frame_index),
                "frame_num": int(analysis_frame_index),
                "video_frame_num": int(source_frame_num),
                "frame_name": frame_path.name,
                "frame_path": frame_path.as_posix(),
                "frame_step": int(analysis_frame_skip),
                "video_relative_time_sec": round(relative_time_sec, 6),
                "video_absolute_time_ms": int(absolute_ms),
                "video_absolute_time_utc": absolute_utc.isoformat(),
                "segment_id": int(segment_id) if segment_id is not None else None,
                "source_coordinate_system": SOURCE_COORDINATE_SYSTEM,
                "derived_coordinate_system": DERIVED_COORDINATE_SYSTEM if export_wgs84 else None,
                "coordinate_system_source": COORDINATE_SYSTEM_SOURCE,
                "wgs84_conversion_method": WGS84_CONVERSION_METHOD if export_wgs84 else None,
                **match,
            }
        )
        if progress_callback:
            progress_callback(idx + 1, len(frame_paths), "geo_sync | frame alignment")
    return rows


def _build_segment_rows(
    segments_df: pd.DataFrame,
    gps_df: pd.DataFrame,
    start_time_utc: datetime,
    frame_matches_df: pd.DataFrame,
    max_gap_warning_sec: float,
    export_wgs84: bool,
    progress_callback: Optional[Any] = None,
) -> pd.DataFrame:
    counts = (
        frame_matches_df["segment_id"].value_counts(dropna=True).to_dict()
        if "segment_id" in frame_matches_df.columns and not frame_matches_df.empty
        else {}
    )
    rows: List[Dict[str, Any]] = []
    start_epoch_ms = int(start_time_utc.timestamp() * 1000)
    ordered_segments = segments_df.sort_values("segment_id")
    if progress_callback:
        progress_callback(0, len(ordered_segments), "geo_sync | segment alignment")
    for idx, (_, seg) in enumerate(ordered_segments.iterrows()):
        center_time_sec = float(seg["center_time_sec"])
        target_ms = start_epoch_ms + int(round(center_time_sec * 1000.0))
        match = _evaluate_match(
            gps_df,
            target_ms,
            max_gap_warning_sec=max_gap_warning_sec,
            export_wgs84=export_wgs84,
        )
        rows.append(
            {
                "segment_id": int(seg["segment_id"]),
                "segment_start_time_sec": float(seg["start_time_sec"]),
                "segment_end_time_sec": float(seg["end_time_sec"]),
                "segment_center_time_sec": center_time_sec,
                "segment_center_time_utc": (
                    start_time_utc + timedelta(seconds=center_time_sec)
                ).astimezone(timezone.utc).isoformat(),
                "sampled_frame_count": int(counts.get(int(seg["segment_id"]), 0)),
                "source_coordinate_system": SOURCE_COORDINATE_SYSTEM,
                "derived_coordinate_system": DERIVED_COORDINATE_SYSTEM if export_wgs84 else None,
                "coordinate_system_source": COORDINATE_SYSTEM_SOURCE,
                "wgs84_conversion_method": WGS84_CONVERSION_METHOD if export_wgs84 else None,
                **match,
            }
        )
        if progress_callback:
            progress_callback(idx + 1, len(ordered_segments), "geo_sync | segment alignment")
    return pd.DataFrame(rows)


def run_geo_sync(
    video_path: Optional[str] = None,
    input_dir: str = "input",
    gps_csv_path: str = "output_gps.csv",
    output_root: str = "output",
    frame_step: int = 60,
    frames_dir: Optional[str] = None,
    analysis_frame_skip: Optional[int] = None,
    start_time: Optional[str] = None,
    sidecar_path: Optional[str] = None,
    time_offset_seconds: float = 0.0,
    filename_tz_offset_hours: float = 8.0,
    segment_manifest_path: Optional[str] = None,
    segment_seconds: float = 5.0,
    segment_overlap: float = 2.5,
    max_samples: Optional[int] = None,
    display: bool = False,
    save_preview_count: int = 0,
    export_wgs84: bool = True,
    align_to_existing_frames: bool = True,
    max_gap_warning_sec: float = EFFECTIVE_OVERLAP_GAP_SEC,
    use_existing_segments: bool = True,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    selected_video = discover_video(video_path=video_path, input_dir=input_dir)
    sidecar_payload = _load_sidecar(sidecar_path)
    gps_csv = Path(gps_csv_path)
    if not gps_csv.is_file():
        raise FileNotFoundError(f"GPS CSV not found: {gps_csv.as_posix()}")

    probe = probe_video(selected_video)
    start_info = resolve_video_start_time(
        video_path=selected_video,
        probe=probe,
        explicit_start_time=start_time,
        sidecar_path=sidecar_path,
        time_offset_seconds=time_offset_seconds,
        filename_tz_offset_hours=filename_tz_offset_hours,
    )
    base_start_time_utc = parse_iso_datetime(start_info["base_start_time_utc"])
    start_time_utc = parse_iso_datetime(start_info["resolved_start_time_utc"])
    duration_sec = float(probe.get("duration_sec") or 0.0)

    gps_df, gps_summary = load_gps_dataframe(gps_csv, export_wgs84=export_wgs84)
    video_stem = selected_video.stem
    output_root_path = Path(output_root)
    geo_dir = output_root_path / video_stem / "geo_sync"
    preview_dir = geo_dir / "previews"
    geo_dir.mkdir(parents=True, exist_ok=True)
    if save_preview_count > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    segments_df, segment_source, resolved_segment_manifest = _load_or_build_segments(
        output_root=output_root_path,
        video_stem=video_stem,
        explicit_manifest_path=segment_manifest_path,
        duration_sec=duration_sec,
        segment_seconds=segment_seconds,
        segment_overlap=segment_overlap,
        use_existing_segments=use_existing_segments,
    )

    overlap_without_offset = _diagnose_window_for_start(
        gps_df=gps_df,
        start_time_utc=base_start_time_utc,
        duration_sec=duration_sec,
        gap_threshold_sec=max_gap_warning_sec,
        sample_count=OFFSET_SCAN_SAMPLE_COUNT,
    )
    overlap_with_offset = _diagnose_window_for_start(
        gps_df=gps_df,
        start_time_utc=start_time_utc,
        duration_sec=duration_sec,
        gap_threshold_sec=max_gap_warning_sec,
        sample_count=OFFSET_SCAN_SAMPLE_COUNT,
    )
    offset_scan = _suggest_offset_range(
        gps_df=gps_df,
        base_start_time_utc=base_start_time_utc,
        duration_sec=duration_sec,
        gap_threshold_sec=max_gap_warning_sec,
        sample_count=OFFSET_SCAN_SAMPLE_COUNT,
    )

    fps = float(probe.get("fps") or 0.0) or 30.0
    total_frames = 0
    sampled_rows: List[Dict[str, Any]] = []
    frame_alignment_mode = "video_sampling"
    resolved_analysis_frame_skip = int(
        analysis_frame_skip if analysis_frame_skip is not None and int(analysis_frame_skip) > 0 else frame_step
    )

    existing_frame_paths: List[Path] = []
    if align_to_existing_frames and frames_dir:
        existing_frame_paths = _discover_existing_frame_paths(Path(frames_dir))

    if existing_frame_paths:
        frame_alignment_mode = "existing_analysis_frames"
        sampled_rows = _build_frame_rows_from_existing_frames(
            frame_paths=existing_frame_paths[: int(max_samples)] if max_samples is not None else existing_frame_paths,
            start_time_utc=start_time_utc,
            fps=fps,
            analysis_frame_skip=resolved_analysis_frame_skip,
            gps_df=gps_df,
            segments_df=segments_df,
            max_gap_warning_sec=max_gap_warning_sec,
            export_wgs84=export_wgs84,
            progress_callback=progress_callback,
        )
        total_frames = int(len(existing_frame_paths))
    else:
        cap = cv2.VideoCapture(str(selected_video))
        if not cap.isOpened():
            raise ValueError(f"unable to open video with cv2: {selected_video.as_posix()}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or probe.get("fps") or 0.0) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        saved_preview = 0

        frame_index = -1
        sampled_count = 0
        if progress_callback:
            progress_callback(0, max(total_frames, 1), "geo_sync | frame alignment")
        while True:
            grabbed = cap.grab()
            if not grabbed:
                break
            frame_index += 1
            if progress_callback and total_frames > 0 and (frame_index == 0 or frame_index % 50 == 0):
                progress_callback(min(frame_index + 1, total_frames), total_frames, "geo_sync | frame alignment")
            if frame_step <= 0 or frame_index % frame_step != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue

            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if pos_msec > 0:
                relative_time_sec = pos_msec / 1000.0
            else:
                relative_time_sec = float(frame_index) / fps

            absolute_ms = int(start_time_utc.timestamp() * 1000) + int(round(relative_time_sec * 1000.0))
            absolute_utc = datetime.fromtimestamp(absolute_ms / 1000.0, tz=timezone.utc)
            match = _evaluate_match(
                gps_df,
                absolute_ms,
                max_gap_warning_sec=max_gap_warning_sec,
                export_wgs84=export_wgs84,
            )
            segment_id = _map_time_to_segment(relative_time_sec, segments_df)

            row = {
                "frame_index": int(frame_index),
                "frame_num": int(frame_index),
                "video_frame_num": int(frame_index),
                "frame_name": None,
                "frame_path": None,
                "frame_step": int(frame_step),
                "video_relative_time_sec": round(relative_time_sec, 6),
                "video_absolute_time_ms": int(absolute_ms),
                "video_absolute_time_utc": absolute_utc.isoformat(),
                "segment_id": int(segment_id) if segment_id is not None else None,
                "source_coordinate_system": SOURCE_COORDINATE_SYSTEM,
                "derived_coordinate_system": DERIVED_COORDINATE_SYSTEM if export_wgs84 else None,
                "coordinate_system_source": COORDINATE_SYSTEM_SOURCE,
                "wgs84_conversion_method": WGS84_CONVERSION_METHOD if export_wgs84 else None,
                **match,
            }
            sampled_rows.append(row)
            sampled_count += 1

            if display or saved_preview < int(save_preview_count):
                annotated = _annotate_frame(frame, relative_time_sec, absolute_utc.isoformat(), match)
                if saved_preview < int(save_preview_count):
                    preview_path = preview_dir / f"sample_{saved_preview + 1:04d}.jpg"
                    cv2.imwrite(str(preview_path), annotated)
                    saved_preview += 1
                if display:
                    cv2.imshow("geo_sync_preview", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if max_samples is not None and sampled_count >= int(max_samples):
                break

        cap.release()
        if display:
            cv2.destroyAllWindows()
        if progress_callback and total_frames > 0:
            progress_callback(total_frames, total_frames, "geo_sync | frame alignment")

    frame_matches_df = pd.DataFrame(sampled_rows)
    segment_matches_df = _build_segment_rows(
        segments_df=segments_df,
        gps_df=gps_df,
        start_time_utc=start_time_utc,
        frame_matches_df=frame_matches_df,
        max_gap_warning_sec=max_gap_warning_sec,
        export_wgs84=export_wgs84,
        progress_callback=progress_callback,
    )

    frame_csv = geo_dir / "frame_geo_metadata.csv"
    segment_csv = geo_dir / "segment_geo_metadata.csv"
    summary_json = geo_dir / "geo_sync_summary.json"

    frame_matches_df.to_csv(frame_csv, index=False, encoding="utf-8")
    segment_matches_df.to_csv(segment_csv, index=False, encoding="utf-8")

    status_counts = (
        frame_matches_df["match_status"].value_counts(dropna=False).to_dict()
        if not frame_matches_df.empty
        else {}
    )
    confidence_counts = (
        frame_matches_df["confidence"].value_counts(dropna=False).to_dict()
        if not frame_matches_df.empty
        else {}
    )
    summary = {
        "run_at_utc": _utc_now_iso(),
        "video_path": selected_video.as_posix(),
        "video_probe": probe,
        "start_time_resolution": start_info,
        "gps_csv_path": gps_csv.as_posix(),
        "gps_summary": gps_summary,
        "coordinate_system": {
            "source_coordinate_system": SOURCE_COORDINATE_SYSTEM,
            "source_coordinate_system_source": COORDINATE_SYSTEM_SOURCE,
            "derived_coordinate_system": DERIVED_COORDINATE_SYSTEM if export_wgs84 else None,
            "derived_coordinate_method": WGS84_CONVERSION_METHOD if export_wgs84 else None,
            "derived_coordinate_note": WGS84_CONVERSION_NOTE if export_wgs84 else None,
            "raw_gcj02_fields_preserved": True,
            "wgs84_fields_exported": bool(export_wgs84),
        },
        "coordinate_field_policy": {
            "unqualified_lon_lat_fields_removed_from_csv": bool(UNQUALIFIED_COORD_FIELDS_REMOVED),
            "primary_coordinate_fields": {
                "before_after_gcj02": [
                    "before_gps_longitude_gcj02",
                    "before_gps_latitude_gcj02",
                    "after_gps_longitude_gcj02",
                    "after_gps_latitude_gcj02",
                ],
                "matched_gcj02": [
                    "matched_gps_longitude_gcj02",
                    "matched_gps_latitude_gcj02",
                ],
                "before_after_wgs84": (
                    [
                        "before_gps_longitude_wgs84",
                        "before_gps_latitude_wgs84",
                        "after_gps_longitude_wgs84",
                        "after_gps_latitude_wgs84",
                    ]
                    if export_wgs84
                    else []
                ),
                "matched_wgs84": (
                    [
                        "matched_gps_longitude_wgs84",
                        "matched_gps_latitude_wgs84",
                    ]
                    if export_wgs84
                    else []
                ),
            },
        },
        "offset_strategy": {
            "default_offset_seconds": float(time_offset_seconds),
            "applied_offset_seconds": float(start_info.get("time_offset_seconds") or 0.0),
            "offset_source": (
                "sidecar.time_offset_seconds"
                if sidecar_payload.get("time_offset_seconds") is not None
                else "runtime_parameter_or_config"
            ),
            "recommended_management": "prefer per-video sidecar; use CLI override for one-off calibration",
            "why_not_global_constant": (
                "video export metadata and GPS clocks can drift or shift per clip, so a single project-wide "
                "offset constant is unsafe"
            ),
        },
        "overlap_diagnosis": {
            "gps_full_range_utc": {
                "start": gps_df.iloc[0]["groupTime_utc"].isoformat(),
                "end": gps_df.iloc[-1]["groupTime_utc"].isoformat(),
            },
            "video_covers_only_partial_gps_track": bool(
                duration_sec < max(
                    0.0,
                    (int(gps_df.iloc[-1]["groupTime_ms"]) - int(gps_df.iloc[0]["groupTime_ms"])) / 1000.0,
                )
            ),
            "without_manual_offset": overlap_without_offset,
            "with_current_offset": overlap_with_offset,
            "boundary_offset_candidates_seconds": {
                "from_base_start_boundary": _boundary_offset_candidates(overlap_without_offset["start_boundary"]),
                "from_base_end_boundary": _boundary_offset_candidates(overlap_without_offset["end_boundary"]),
            },
            "offset_scan": offset_scan,
        },
        "run_config": {
            "frame_step": int(frame_step),
            "frame_alignment_mode": frame_alignment_mode,
            "analysis_frame_skip": int(resolved_analysis_frame_skip),
            "align_to_existing_frames": bool(align_to_existing_frames),
            "frames_dir": Path(frames_dir).as_posix() if frames_dir else None,
            "max_samples": int(max_samples) if max_samples is not None else None,
            "display": bool(display),
            "save_preview_count": int(save_preview_count),
            "segment_seconds": float(segment_seconds),
            "segment_overlap": float(segment_overlap),
            "max_gap_warning_sec": float(max_gap_warning_sec),
            "export_wgs84": bool(export_wgs84),
            "use_existing_segments": bool(use_existing_segments),
        },
        "outputs": {
            "frame_geo_metadata_csv": frame_csv.as_posix(),
            "segment_geo_metadata_csv": segment_csv.as_posix(),
            "summary_json": summary_json.as_posix(),
            "preview_dir": preview_dir.as_posix() if save_preview_count > 0 else None,
        },
        "result_summary": {
            "sampled_rows": int(len(frame_matches_df)),
            "segment_rows": int(len(segment_matches_df)),
            "status_counts": status_counts,
            "confidence_counts": confidence_counts,
            "video_stream_count": int(probe.get("video_stream_count") or 0),
            "audio_stream_count": int(probe.get("audio_stream_count") or 0),
            "total_video_frames_reported_by_cv2": int(total_frames),
            "frame_alignment_mode": frame_alignment_mode,
            "segment_source": segment_source,
            "segment_manifest_path": resolved_segment_manifest,
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
