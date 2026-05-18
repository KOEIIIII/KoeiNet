


"""Build a manifest-aligned visual segment feature table from existing outputs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import cv2
import numpy as np
import pandas as pd


ACTIVITY_SCORE_COLUMNS: Mapping[str, str] = {
    "ai_activity_sitting_score_mean": "坐下休息_score",
    "ai_activity_standing_score_mean": "站着停留_score",
    "ai_activity_walking_score_mean": "散步_score",
    "ai_activity_running_score_mean": "跑步_score",
    "ai_activity_fitness_score_mean": "健身锻炼_score",
    "ai_activity_shopping_score_mean": "买菜购物_score",
}

ACTIVITY_LABEL_BY_OUTPUT: Mapping[str, str] = {
    "ai_activity_sitting_score_mean": "sitting",
    "ai_activity_standing_score_mean": "standing",
    "ai_activity_walking_score_mean": "walking",
    "ai_activity_running_score_mean": "running",
    "ai_activity_fitness_score_mean": "fitness",
    "ai_activity_shopping_score_mean": "shopping",
}


def _parse_jsonish_list(value: object) -> List[object]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    parsed = ast.literal_eval(text)
    if isinstance(parsed, list):
        return parsed
    raise ValueError(f"expected list-like value, got: {value}")


def _frame_num_from_path(path: str) -> int:
    return int(Path(path).stem.split("_")[-1])


def _compute_frame_metrics(image_path: Path) -> Optional[Dict[str, float]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    target_w = 384
    if w > target_w:
        scale = target_w / float(w)
        rgb = cv2.resize(rgb, (target_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    hue = hsv[:, :, 0].astype(np.uint8)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0

    green_mask = ((hue >= 35) & (hue <= 85) & (sat >= 0.18) & (val >= 0.18)).astype(np.float32)
    warm_mask = (((hue <= 25) | (hue >= 150)) & (sat >= 0.18) & (val >= 0.2)).astype(np.float32)
    cool_mask = ((hue >= 85) & (hue <= 145) & (sat >= 0.15) & (val >= 0.2)).astype(np.float32)
    grayish_mask = (sat <= 0.12).astype(np.float32)
    dark_mask = (val <= 0.22).astype(np.float32)

    top = hsv[: max(1, hsv.shape[0] // 3), :, :]
    top_hue = top[:, :, 0]
    top_sat = top[:, :, 1].astype(np.float32) / 255.0
    top_val = top[:, :, 2].astype(np.float32) / 255.0
    sky_mask = (
        (top_hue >= 90) & (top_hue <= 125) & (top_sat >= 0.1) & (top_val >= 0.35)
    ).astype(np.float32)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).astype(np.float64).ravel()
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum()) if hist.size else 0.0

    return {
        "vis_brightness_mean": float(val.mean()),
        "vis_brightness_std": float(val.std()),
        "vis_contrast_std": float(gray.astype(np.float32).std() / 255.0),
        "vis_saturation_mean": float(sat.mean()),
        "vis_green_pixel_ratio": float(green_mask.mean()),
        "vis_sky_blue_ratio_top": float(sky_mask.mean()),
        "vis_warm_color_ratio": float(warm_mask.mean()),
        "vis_cool_color_ratio": float(cool_mask.mean()),
        "vis_grayish_pixel_ratio": float(grayish_mask.mean()),
        "vis_dark_pixel_ratio": float(dark_mask.mean()),
        "vis_edge_density": float(edge_density),
        "vis_gray_entropy": float(entropy),
    }


def _load_ai_scores(activity_csv: Path, max_frame_num: int) -> pd.DataFrame:
    df = pd.read_csv(activity_csv, encoding="utf-8-sig")
    if "frame_num" not in df.columns:
        raise ValueError(f"activity score file missing frame_num: {activity_csv.as_posix()}")

    keep = ["frame_num"] + [src for src in ACTIVITY_SCORE_COLUMNS.values() if src in df.columns]
    base = df[keep].copy()
    base["frame_num"] = pd.to_numeric(base["frame_num"], errors="coerce").astype("Int64")
    base = base.dropna(subset=["frame_num"]).copy()
    base["frame_num"] = base["frame_num"].astype(int)
    base = base.sort_values("frame_num").drop_duplicates("frame_num", keep="last")

    full = pd.DataFrame({"frame_num": np.arange(max_frame_num + 1, dtype=int)})
    merged = full.merge(base, on="frame_num", how="left").sort_values("frame_num")
    score_cols = [src for src in ACTIVITY_SCORE_COLUMNS.values() if src in merged.columns]
    if score_cols:
        merged[score_cols] = (
            merged[score_cols]
            .apply(pd.to_numeric, errors="coerce")
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

    rename_map = {src: dst for dst, src in ACTIVITY_SCORE_COLUMNS.items() if src in merged.columns}
    return merged.rename(columns=rename_map)


def _aggregate_segment_row(
    segment_row: pd.Series,
    frame_feature_map: Mapping[int, Dict[str, float]],
    ai_full: pd.DataFrame,
    visual_feature_sources: str,
) -> Dict[str, object]:
    frame_paths = [str(item) for item in _parse_jsonish_list(segment_row["included_frame_paths"])]
    frame_indices = [int(item) for item in _parse_jsonish_list(segment_row["included_frame_indices"])]
    frame_nums = [_frame_num_from_path(path) for path in frame_paths] if frame_paths else frame_indices

    feature_rows = [frame_feature_map[num] for num in frame_nums if num in frame_feature_map]
    feature_df = pd.DataFrame(feature_rows)
    ai_rows = ai_full[ai_full["frame_num"].isin(frame_nums)].copy()

    out: Dict[str, object] = {
        "segment_id": int(segment_row["segment_id"]),
        "start_time_sec": float(segment_row["start_time_sec"]),
        "end_time_sec": float(segment_row["end_time_sec"]),
        "center_time_sec": float(segment_row["center_time_sec"]),
        "included_frame_count": int(segment_row["included_frame_count"]),
        "visual_frame_count": int(len(feature_rows)),
        "visual_frame_coverage_ratio": float(len(feature_rows) / max(1, len(frame_nums))),
        "ai_activity_frame_count": int(len(ai_rows)),
        "visual_feature_sources": visual_feature_sources,
    }

    if not feature_df.empty:
        for col in feature_df.columns:
            out[f"{col}_mean"] = float(feature_df[col].mean())
            out[f"{col}_std"] = float(feature_df[col].std(ddof=0))
    else:
        for name in next(iter(frame_feature_map.values())).keys():
            out[f"{name}_mean"] = np.nan
            out[f"{name}_std"] = np.nan

    ai_score_cols = [col for col in ACTIVITY_SCORE_COLUMNS if col in ai_rows.columns]
    if ai_score_cols:
        for col in ai_score_cols:
            out[col] = float(pd.to_numeric(ai_rows[col], errors="coerce").mean())

        score_series = {col: out[col] for col in ai_score_cols}
        major_col = max(score_series, key=score_series.get)
        out["ai_activity_major_label"] = ACTIVITY_LABEL_BY_OUTPUT.get(major_col, major_col)
        out["ai_activity_suitable_labels"] = json.dumps(
            [
                ACTIVITY_LABEL_BY_OUTPUT.get(col, col)
                for col, value in score_series.items()
                if value >= 3.0
            ],
            ensure_ascii=False,
        )
    else:
        for col in ACTIVITY_SCORE_COLUMNS:
            out[col] = np.nan
        out["ai_activity_major_label"] = None
        out["ai_activity_suitable_labels"] = "[]"

    return out


def build_segment_visual_features(
    video_dir: str | Path,
    output_csv: Optional[str | Path] = None,
    progress_callback: Optional[object] = None,
) -> Dict[str, object]:
    video_dir = Path(video_dir)
    manifest_path = video_dir / "segments" / "segment_manifest.csv"
    activity_csv = video_dir / "ai_evaluation" / "activity_scores.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"segment manifest not found: {manifest_path.as_posix()}")
    out_dir = video_dir / "visual"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_csv) if output_csv else out_dir / "segment_visual_features.csv"

    manifest_df = pd.read_csv(manifest_path)
    manifest_df["segment_id"] = pd.to_numeric(manifest_df["segment_id"], errors="raise").astype(int)
    manifest_df = manifest_df.sort_values("segment_id").reset_index(drop=True)

    all_frame_paths: List[str] = []
    for value in manifest_df["included_frame_paths"]:
        all_frame_paths.extend([str(item) for item in _parse_jsonish_list(value)])
    unique_paths = sorted(set(all_frame_paths))
    if not unique_paths:
        raise ValueError("segment manifest contains no included_frame_paths")

    frame_feature_map: Dict[int, Dict[str, float]] = {}
    if progress_callback:
        progress_callback(0, len(unique_paths), "visual | frame metrics")
    for idx, rel_path in enumerate(unique_paths):
        frame_path = Path(rel_path)
        if not frame_path.is_absolute():
            frame_path = Path.cwd() / rel_path
        metrics = _compute_frame_metrics(frame_path)
        if metrics is None:
            continue
        frame_num = _frame_num_from_path(frame_path.as_posix())
        frame_feature_map[frame_num] = metrics
        if progress_callback:
            progress_callback(idx + 1, len(unique_paths), "visual | frame metrics")

    if not frame_feature_map:
        raise ValueError("failed to compute any frame-level visual metrics")

    max_frame_num = max(frame_feature_map.keys())
    if activity_csv.is_file():
        ai_full = _load_ai_scores(activity_csv, max_frame_num=max_frame_num)
        visual_feature_sources = "frames + ai_evaluation/activity_scores.csv"
    else:
        ai_full = pd.DataFrame({"frame_num": np.arange(max_frame_num + 1, dtype=int)})
        visual_feature_sources = "frames"

    rows = []
    if progress_callback:
        progress_callback(0, len(manifest_df), "visual | segment aggregation")
    for idx, (_, row) in enumerate(manifest_df.iterrows()):
        rows.append(
            _aggregate_segment_row(
                row,
                frame_feature_map=frame_feature_map,
                ai_full=ai_full,
                visual_feature_sources=visual_feature_sources,
            )
        )
        if progress_callback:
            progress_callback(idx + 1, len(manifest_df), "visual | segment aggregation")
    result_df = pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return {
        "video_dir": video_dir.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "activity_csv": activity_csv.as_posix() if activity_csv.is_file() else None,
        "output_csv": output_csv.as_posix(),
        "segment_rows": int(len(result_df)),
        "frame_feature_rows": int(len(frame_feature_map)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build manifest-aligned visual segment features.")
    parser.add_argument("--video-dir", required=True, help="Existing output/<video_name> directory.")
    parser.add_argument("--output-csv", help="Optional custom output CSV path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_segment_visual_features(video_dir=args.video_dir, output_csv=args.output_csv)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
