

"""
视频分析展示Web应用 - 主网站
专为Bing浏览器优化，确保视频和图表完美加载
"""

import os
import ast
import logging
import webbrowser
import json
from threading import Timer
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import mimetypes
from flask import Flask, render_template, jsonify, send_from_directory, Response, request, send_file
from flask_cors import CORS

app = Flask(__name__,
           static_folder='web/static',
           template_folder='web/templates')
CORS(app)


OUTPUT_DIR = "output"
INPUT_DIR = "input"
PLAYABLE_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".insv", ".webm")
VALIDATION_FILENAME = "final_annotation_labels_adjudicated.csv"
VALIDATION_REQUIRED_COLUMNS = [
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
VALIDATION_STREET_TYPES = [
    "mixed_use",
    "commercial",
    "residential",
    "arterial",
    "park",
    "campus",
]
VALIDATION_PROBLEM_LABEL_OPTIONS = [
    "traffic_noise",
    "pedestrian_discomfort",
    "visual_clutter",
    "low_green_view",
    "high_hardscape",
    "vehicle_dominance",
    "poor_walkability_cues",
    "low_aesthetic_quality",
    "high_loudness",
    "low_natural_sound",
    "human_voice_dominant",
    "high_eventfulness",
    "low_eventfulness",
    "noisy_but_low_pleasantness",
]


mimetypes.add_type('video/mp4', '.mp4')
mimetypes.add_type('video/webm', '.webm')
mimetypes.add_type('video/ogg', '.ogg')
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')


def _video_dir(video_name: str) -> Path:
    return Path(OUTPUT_DIR) / video_name


def _resolve_playable_video_path(video_name: str) -> Path | None:
    video_dir = _video_dir(video_name)
    preferred = [
        video_dir / f"{video_name}_processed_h264.mp4",
        video_dir / f"{video_name}_processed.mp4",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate

    if video_dir.is_dir():
        for ext in PLAYABLE_VIDEO_EXTENSIONS:
            candidate = video_dir / f"{video_name}{ext}"
            if candidate.is_file():
                return candidate
        generic = sorted(
            [
                path
                for path in video_dir.iterdir()
                if path.is_file() and path.suffix.lower() in PLAYABLE_VIDEO_EXTENSIONS
            ]
        )
        if generic:
            return generic[0]

    input_dir = Path(INPUT_DIR)
    for ext in PLAYABLE_VIDEO_EXTENSIONS:
        candidate = input_dir / f"{video_name}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _resolve_source_video_path(video_name: str) -> Path | None:
    input_dir = Path(INPUT_DIR)
    for ext in PLAYABLE_VIDEO_EXTENSIONS:
        candidate = input_dir / f"{video_name}{ext}"
        if candidate.is_file():
            return candidate
    return _resolve_playable_video_path(video_name)


def _has_modern_analysis_assets(video_dir: Path) -> bool:
    if not video_dir.is_dir():
        return False
    required_any = [
        video_dir / "web" / "sync_map_data.json",
        video_dir / "segments" / "segment_manifest.csv",
        video_dir / "geo_sync" / "frame_geo_metadata.csv",
        video_dir / "geo_sync" / "segment_geo_metadata.csv",
        video_dir / "visual" / "segment_visual_features.csv",
        video_dir / "soundscape" / "audio_segment_features.csv",
        video_dir / "fusion" / "segment_feature_table.csv",
    ]
    if any(path.is_file() for path in required_any):
        return True
    frames_dir = video_dir / "frames"
    return frames_dir.is_dir() and any(frames_dir.glob("frame_*.*"))


def _is_analyzed_video_dir(video_name: str, video_dir: Path) -> bool:
    return _has_modern_analysis_assets(video_dir) and (_resolve_playable_video_path(video_name) is not None)


def _read_csv_records(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path).to_dict('records')


def _read_json_payload(path: Path):
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _read_csv_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _validation_dir(video_name: str) -> Path:
    return _video_dir(video_name) / "validation"


def _validation_csv_path(video_name: str) -> Path:
    return _validation_dir(video_name) / VALIDATION_FILENAME


def _safe_int(value: Any):
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any):
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _parse_json_array(value: Any) -> List[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            continue
    return [part.strip() for part in text.split(",") if part.strip()]


def _json_array_string(value: Any) -> str:
    return json.dumps(_parse_json_array(value), ensure_ascii=False)


def _clean_scalar(value: Any):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _pick_present(record: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        payload[key] = value
    return payload


def _relative_video_asset_path(video_name: str, raw_path: Any) -> str | None:
    if raw_path is None:
        return None
    try:
        if pd.isna(raw_path):
            return None
    except Exception:
        pass
    if raw_path is None:
        return None
    text = str(raw_path).replace("\\", "/").strip()
    if not text:
        return None
    prefix = f"output/{video_name}/"
    if text.startswith(prefix):
        return text[len(prefix):]
    marker = f"/{video_name}/"
    if marker in text:
        return text.split(marker, 1)[1]
    return text.lstrip("/")


def _asset_url(video_name: str, raw_path: Any) -> str | None:
    relative_path = _relative_video_asset_path(video_name, raw_path)
    if not relative_path:
        return None
    return f"/api/asset/{video_name}/{relative_path}"


def _parse_manifest_paths(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    try:
        if pd.isna(raw_value):
            return []
    except Exception:
        pass
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            continue
    return []


def _segment_frame_preview_urls(video_name: str, raw_paths: Any) -> List[str]:
    paths = _parse_manifest_paths(raw_paths)
    if not paths:
        return []
    indices = [0, len(paths) // 2, len(paths) - 1]
    unique_indices: List[int] = []
    for idx in indices:
        if idx not in unique_indices and 0 <= idx < len(paths):
            unique_indices.append(idx)
    urls: List[str] = []
    for idx in unique_indices:
        url = _asset_url(video_name, paths[idx])
        if url:
            urls.append(url)
    return urls


def _discover_first_existing_asset(base_dir: Path, patterns: List[str]) -> Path | None:
    if not base_dir.is_dir():
        return None
    for pattern in patterns:
        matches = sorted(base_dir.glob(pattern))
        for match in matches:
            if match.is_file():
                return match
    return None


def _segment_validation_asset_urls(video_name: str, segment_id: int) -> Dict[str, str]:
    sid = int(segment_id)
    base = _video_dir(video_name) / "validation"
    preview_path = _discover_first_existing_asset(
        base / "previews",
        [
            f"segment_{sid:04d}.*",
            f"segment_{sid}.*",
            f"segment_{sid:04d}_preview.*",
            f"segment_{sid}_preview.*",
        ],
    )
    frame_strip_path = _discover_first_existing_asset(
        base / "frame_strips",
        [
            f"segment_{sid:04d}_strip.*",
            f"segment_{sid}_strip.*",
            f"segment_{sid:04d}.*",
            f"segment_{sid}.*",
        ],
    )
    audio_clip_path = _discover_first_existing_asset(
        base / "audio_clips",
        [
            f"segment_{sid:04d}.*",
            f"segment_{sid}.*",
            f"clip_{sid:04d}.*",
            f"clip_{sid}.*",
        ],
    )
    candidates = {
        "primary_preview_url": preview_path,
        "frame_strip_url": frame_strip_path,
        "audio_clip_url": audio_clip_path,
    }
    out: Dict[str, str] = {}
    for key, path in candidates.items():
        if path is not None and path.is_file():
            url = _asset_url(video_name, path.as_posix())
            if url:
                out[key] = url
    return out


def _load_validation_relationship_note(video_name: str, segment_id: int) -> Dict[str, Any]:
    sid = int(segment_id)
    relation_dir = _video_dir(video_name) / "validation" / "relationship"
    note_path = _discover_first_existing_asset(
        relation_dir,
        [
            f"segment_{sid:04d}.json",
            f"segment_{sid}.json",
            f"segment_{sid:04d}.md",
            f"segment_{sid}.md",
            f"segment_{sid:04d}.txt",
            f"segment_{sid}.txt",
        ],
    )
    if note_path is None or not note_path.is_file():
        return {}
    try:
        if note_path.suffix.lower() == ".json":
            payload = json.loads(note_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
            return {"relationship_note": json.dumps(payload, ensure_ascii=False)}
        return {"relationship_note": note_path.read_text(encoding="utf-8").strip()}
    except Exception:
        return {"relationship_note": note_path.read_text(encoding="utf-8", errors="ignore").strip()}


def _build_relationship_summary(
    diagnostics_row: Dict[str, Any],
    visual_row: Dict[str, Any],
    soundscape_row: Dict[str, Any],
    fusion_row: Dict[str, Any],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if diagnostics_row.get("cross_modal_reason"):
        summary["cross_modal_reason"] = diagnostics_row.get("cross_modal_reason")
    if diagnostics_row.get("problem_labels"):
        summary["problem_labels"] = diagnostics_row.get("problem_labels")
    if soundscape_row.get("top_k_events") is not None:
        summary["sound_top_events"] = soundscape_row.get("top_k_events")
    if soundscape_row.get("group_ratio_traffic") is not None:
        summary["traffic_sound_ratio"] = soundscape_row.get("group_ratio_traffic")
    if soundscape_row.get("group_ratio_nature") is not None:
        summary["natural_sound_ratio"] = soundscape_row.get("group_ratio_nature")
    if fusion_row.get("visual_semantic__road__mean") is not None:
        summary["road_ratio"] = fusion_row.get("visual_semantic__road__mean")
    if fusion_row.get("visual_semantic__sidewalk__mean") is not None:
        summary["sidewalk_ratio"] = fusion_row.get("visual_semantic__sidewalk__mean")
    if fusion_row.get("people__total_people__mean") is not None:
        summary["people_mean"] = fusion_row.get("people__total_people__mean")
    if visual_row.get("vis_green_pixel_ratio_mean") is not None:
        summary["green_ratio"] = visual_row.get("vis_green_pixel_ratio_mean")
    if visual_row.get("vis_sky_blue_ratio_top_mean") is not None:
        summary["sky_ratio"] = visual_row.get("vis_sky_blue_ratio_top_mean")
    return summary


def _send_media_file(resolved_path: Path):
    if resolved_path is None or not resolved_path.exists():
        return "Video not found", 404

    full_path = resolved_path.as_posix()
    parent_path = resolved_path.parent.as_posix()
    file_size = os.path.getsize(full_path)
    content_type = mimetypes.guess_type(full_path)[0] or "application/octet-stream"

    range_header = request.headers.get('Range', None)
    if range_header:
        byte_start = 0
        byte_end = file_size - 1
        match = range_header.replace('bytes=', '').split('-')
        if match[0]:
            byte_start = int(match[0])
        if len(match) > 1 and match[1]:
            byte_end = int(match[1])

        with open(full_path, 'rb') as f:
            f.seek(byte_start)
            data = f.read(byte_end - byte_start + 1)

        return Response(
            data,
            206,
            headers={
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(len(data)),
                'Content-Type': content_type,
                'Cache-Control': 'no-cache',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Range',
            }
        )

    response = send_from_directory(parent_path, os.path.basename(full_path))
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Content-Type'] = content_type
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Range'
    return response


def _normalize_validation_df(manifest_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    base = manifest_df[["segment_id"]].copy()
    base["segment_id"] = pd.to_numeric(base["segment_id"], errors="coerce").astype("Int64")
    base = base.dropna(subset=["segment_id"]).drop_duplicates("segment_id").sort_values("segment_id")

    if existing_df.empty:
        merged = base.copy()
    else:
        current = existing_df.copy()
        if "segment_id" not in current.columns:
            current["segment_id"] = pd.NA
        current["segment_id"] = pd.to_numeric(current["segment_id"], errors="coerce").astype("Int64")
        merged = base.merge(current, on="segment_id", how="left")

    defaults = {
        "street_type": "mixed_use",
        "comfort_score": pd.NA,
        "vitality_score": pd.NA,
        "soundscape_pleasantness": pd.NA,
        "soundscape_eventfulness": pd.NA,
        "overall_problem_severity": pd.NA,
        "main_problem_labels": "[]",
        "primary_problem_label": "no_major_problem",
        "confidence_score": pd.NA,
        "annotator_notes": pd.NA,
    }
    for column, default in defaults.items():
        if column not in merged.columns:
            merged[column] = default
        elif column == "street_type":
            merged[column] = merged[column].fillna(default)
        elif column == "main_problem_labels":
            merged[column] = merged[column].apply(_json_array_string)

    return merged.sort_values("segment_id").reset_index(drop=True)


def _is_reviewed_validation_row(row: Dict[str, Any]) -> bool:
    required_score_fields = [
        "comfort_score",
        "vitality_score",
        "soundscape_pleasantness",
        "soundscape_eventfulness",
        "overall_problem_severity",
    ]
    for field in required_score_fields:
        value = _safe_int(row.get(field))
        if value is None or value < 1 or value > 5:
            return False
    return bool(_clean_scalar(row.get("street_type"))) and bool(_clean_scalar(row.get("primary_problem_label")))


def _build_segment_lookup(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    if df.empty or "segment_id" not in df.columns:
        return {}
    payload = df.copy()
    payload["segment_id"] = pd.to_numeric(payload["segment_id"], errors="coerce")
    payload = payload.dropna(subset=["segment_id"])
    lookup: Dict[int, Dict[str, Any]] = {}
    for _, row in payload.iterrows():
        lookup[int(row["segment_id"])] = row.to_dict()
    return lookup


def _build_diagnostics_lookup(path: Path) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for record in _read_jsonl_records(path):
        segment_id = _safe_int(record.get("segment_id"))
        if segment_id is None:
            continue
        diagnosis_json = record.get("diagnosis_json") or {}
        lookup[segment_id] = {
            "problem_labels": diagnosis_json.get("problem_labels") or [],
            "severity_scores": diagnosis_json.get("severity_scores") or {},
            "cross_modal_reason": diagnosis_json.get("cross_modal_reason"),
            "priority_actions": diagnosis_json.get("priority_actions") or [],
            "status": record.get("diagnosis_status"),
        }
    return lookup

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/validation/<video_name>')
def validation_page(video_name):
    """Segment-level validation annotation frontend."""
    video_dir = _video_dir(video_name)
    if not video_dir.exists():
        return "未找到对应视频目录", 404
    return render_template('validation.html', video_name=video_name)


@app.route('/simple-test')
def simple_test():
    """简单视频测试页面"""
    return render_template('simple_video_test.html')



@app.route('/debug')
def debug():
    """调试页面"""
    return send_file('debug.html')

@app.route('/api/videos')
def get_videos():
    """获取所有已分析的视频列表"""
    videos = []

    output_root = Path(OUTPUT_DIR)
    if not output_root.exists():
        return jsonify({"videos": []})

    for video_dir in sorted([p for p in output_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        video_name = video_dir.name
        if _is_analyzed_video_dir(video_name, video_dir):
            videos.append({
                "name": video_name,
                "path": video_dir.as_posix(),
                "processed_video": f"/api/video/{video_name}"
            })

    return jsonify({"videos": videos})

@app.route('/api/video/<video_name>')
def serve_video(video_name):
    """提供视频文件 - 支持范围请求以兼容Bing浏览器"""
    resolved_video = _resolve_playable_video_path(video_name)
    if resolved_video is None or not resolved_video.exists():
        return "Video not found", 404
    full_path = resolved_video.as_posix()
    video_path = resolved_video.parent.as_posix()


    file_size = os.path.getsize(full_path)


    range_header = request.headers.get('Range', None)
    if range_header:

        byte_start = 0
        byte_end = file_size - 1

        if range_header:
            match = range_header.replace('bytes=', '').split('-')
            if match[0]:
                byte_start = int(match[0])
            if match[1]:
                byte_end = int(match[1])


        with open(full_path, 'rb') as f:
            f.seek(byte_start)
            data = f.read(byte_end - byte_start + 1)


        response = Response(
            data,
            206,
            headers={
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(len(data)),
                'Content-Type': 'video/mp4',
                'Cache-Control': 'no-cache',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Range',
            }
        )
        return response
    else:


        actual_filename = os.path.basename(full_path)
        response = send_from_directory(video_path, actual_filename)
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Type'] = 'video/mp4'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Range'
        return response

@app.route('/api/source-video/<video_name>')
def serve_source_video(video_name):
    """Serve the original/source video when validation needs reliable audio."""
    resolved_video = _resolve_source_video_path(video_name)
    return _send_media_file(resolved_video)


@app.route('/api/data/<video_name>')
def get_video_data(video_name):
    """Get all analysis data for the video"""
    video_path = os.path.join(OUTPUT_DIR, video_name)

    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    data = {}

    try:

        emotion_csv = os.path.join(video_path, "stats", "emotion", "emotion_scores.csv")
        if os.path.exists(emotion_csv):
            df_emotion = pd.read_csv(emotion_csv)
            data['emotion'] = df_emotion.to_dict('records')


        visual_csv = os.path.join(video_path, "stats", "visual_elements", "major_categories_proportion.csv")
        if os.path.exists(visual_csv):
            df_visual = pd.read_csv(visual_csv)
            data['visual_elements'] = df_visual.to_dict('records')


        color_csv = os.path.join(video_path, "stats", "color_analysis", "color_categories_proportion.csv")
        if os.path.exists(color_csv):
            df_color = pd.read_csv(color_csv)
            data['color_analysis'] = df_color.to_dict('records')


        green_csv = os.path.join(video_path, "stats", "green_view", "green_view_index.csv")
        if os.path.exists(green_csv):
            df_green = pd.read_csv(green_csv)
            data['green_view'] = df_green.to_dict('records')


        people_csv = os.path.join(video_path, "stats", "people_count", "people_count.csv")
        if os.path.exists(people_csv):
            df_people = pd.read_csv(people_csv)
            data['people_count'] = df_people.to_dict('records')


        audio_csv = os.path.join(video_path, "audio_events", "audio_events_detail.csv")
        if os.path.exists(audio_csv):
            df_audio = pd.read_csv(audio_csv)
            data['audio_events'] = df_audio.to_dict('records')


        audio_prop_csv = os.path.join(video_path, "audio_events", "audio_events_proportion.csv")
        if os.path.exists(audio_prop_csv):
            df_audio_prop = pd.read_csv(audio_prop_csv)
            data['audio_events_proportion'] = df_audio_prop.to_dict('records')


        activity_csv = os.path.join(video_path, "ai_evaluation", "activity_scores.csv")
        if os.path.exists(activity_csv):
            df_activity = pd.read_csv(activity_csv)
            data['ai_activity'] = df_activity.to_dict('records')

        sync_map_json = Path(video_path) / "web" / "sync_map_data.json"
        if sync_map_json.exists():
            data['sync_map'] = _read_json_payload(sync_map_json)

        visual_segment_csv = Path(video_path) / "visual" / "segment_visual_features.csv"
        if visual_segment_csv.exists():
            data['visual_segment'] = pd.read_csv(visual_segment_csv).to_dict('records')

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/video-info/<video_name>')
def get_video_info(video_name):
    """Get basic video information"""
    video_path = os.path.join(OUTPUT_DIR, video_name)
    resolved_video = _resolve_playable_video_path(video_name)
    if resolved_video is None or not resolved_video.exists():
        return jsonify({"error": "Video not found"}), 404

    try:
        sync_map_json = Path(video_path) / "web" / "sync_map_data.json"
        geo_summary_json = Path(video_path) / "geo_sync" / "geo_sync_summary.json"

        sync_payload = _read_json_payload(sync_map_json) or {}
        geo_summary = _read_json_payload(geo_summary_json) or {}

        total_frames = int(
            sync_payload.get("timeline", {}).get("frame_count")
            or sync_payload.get("video", {}).get("frame_count")
            or 0
        )
        fps = float(
            sync_payload.get("video", {}).get("fps")
            or geo_summary.get("video_probe", {}).get("fps")
            or 30.0
        )
        duration = float(
            sync_payload.get("video", {}).get("duration_sec")
            or geo_summary.get("video_probe", {}).get("duration_sec")
            or 0.0
        )

        if total_frames <= 0:
            emotion_csv = os.path.join(video_path, "stats", "emotion", "emotion_scores.csv")
            if os.path.exists(emotion_csv):
                total_frames = len(pd.read_csv(emotion_csv))
        if duration <= 0 and total_frames > 0 and fps > 0:
            duration = total_frames / fps

        return jsonify({
            "name": video_name,
            "duration": duration,
            "fps": fps,
            "total_frames": total_frames
        })
    except Exception as e:
        return jsonify({
            "name": video_name,
            "duration": 30,
            "fps": 30,
            "total_frames": 32,
            "error": str(e)
        })


@app.route('/api/sync/<video_name>')
def get_sync_payload(video_name):
    """Get unified video-chart-map sync payload for the dashboard."""
    video_dir = _video_dir(video_name)
    if not video_dir.exists():
        return jsonify({"error": "Video not found"}), 404

    sync_path = video_dir / "web" / "sync_map_data.json"
    try:
        if not sync_path.exists():
            from src.web_sync import build_sync_map_data

            build_sync_map_data(video_dir=video_dir)
        payload = _read_json_payload(sync_path)
        if payload is None:
            return jsonify({"error": "sync payload missing"}), 404
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route('/api/validation/<video_name>')
def get_validation_payload(video_name):
    """Return segment-level evidence and current validation labels for manual review."""
    video_dir = _video_dir(video_name)
    if not video_dir.exists():
        return jsonify({"error": "未找到对应视频目录"}), 404

    manifest_path = video_dir / "segments" / "segment_manifest.csv"
    if not manifest_path.exists():
        return jsonify({"error": "缺少 segment_manifest.csv，无法打开评分页"}), 404

    manifest_df = _read_csv_df(manifest_path)
    validation_path = _validation_csv_path(video_name)
    validation_df = _normalize_validation_df(manifest_df, _read_csv_df(validation_path))

    visual_lookup = _build_segment_lookup(_read_csv_df(video_dir / "visual" / "segment_visual_features.csv"))
    soundscape_lookup = _build_segment_lookup(_read_csv_df(video_dir / "soundscape" / "audio_segment_features.csv"))
    fusion_lookup = _build_segment_lookup(_read_csv_df(video_dir / "fusion" / "segment_feature_table.csv"))
    geo_lookup = _build_segment_lookup(_read_csv_df(video_dir / "geo_sync" / "segment_geo_metadata.csv"))
    diagnostics_lookup = _build_diagnostics_lookup(video_dir / "diagnostics" / "segment_diagnosis.jsonl")
    validation_lookup = _build_segment_lookup(validation_df)

    video_url = f"/api/source-video/{video_name}"
    segments: List[Dict[str, Any]] = []
    reviewed_count = 0

    for _, manifest_row in manifest_df.iterrows():
        segment_id = _safe_int(manifest_row.get("segment_id"))
        if segment_id is None:
            continue

        validation_row = validation_lookup.get(segment_id, {})
        reviewed = _is_reviewed_validation_row(validation_row)
        if reviewed:
            reviewed_count += 1

        visual_row = visual_lookup.get(segment_id, {})
        soundscape_row = soundscape_lookup.get(segment_id, {})
        fusion_row = fusion_lookup.get(segment_id, {})
        geo_row = geo_lookup.get(segment_id, {})
        diagnostics_row = diagnostics_lookup.get(segment_id, {})
        validation_assets = _segment_validation_asset_urls(video_name, segment_id)
        relationship_summary = _build_relationship_summary(
            diagnostics_row=diagnostics_row,
            visual_row=visual_row,
            soundscape_row=soundscape_row,
            fusion_row=fusion_row,
        )
        relationship_summary.update(_load_validation_relationship_note(video_name, segment_id))

        segments.append(
            {
                "segment_id": segment_id,
                "start_time_sec": _safe_float(manifest_row.get("start_time_sec")),
                "end_time_sec": _safe_float(manifest_row.get("end_time_sec")),
                "center_time_sec": _safe_float(manifest_row.get("center_time_sec")),
                "included_frame_count": _safe_int(manifest_row.get("included_frame_count")),
                "frame_preview_urls": _segment_frame_preview_urls(video_name, manifest_row.get("included_frame_paths")),
                "primary_preview_url": validation_assets.get("primary_preview_url"),
                "frame_strip_url": validation_assets.get("frame_strip_url"),
                "video_url": video_url,
                "audio_url": validation_assets.get("audio_clip_url") or video_url,
                "audio_mode": "clip" if validation_assets.get("audio_clip_url") else "full_video",
                "reviewed": reviewed,
                "validation": {
                    "segment_id": segment_id,
                    "street_type": _clean_scalar(validation_row.get("street_type")) or "mixed_use",
                    "comfort_score": _safe_int(validation_row.get("comfort_score")),
                    "vitality_score": _safe_int(validation_row.get("vitality_score")),
                    "soundscape_pleasantness": _safe_int(validation_row.get("soundscape_pleasantness")),
                    "soundscape_eventfulness": _safe_int(validation_row.get("soundscape_eventfulness")),
                    "overall_problem_severity": _safe_int(validation_row.get("overall_problem_severity")),
                    "main_problem_labels": _parse_json_array(validation_row.get("main_problem_labels")),
                    "primary_problem_label": _clean_scalar(validation_row.get("primary_problem_label")) or "no_major_problem",
                    "confidence_score": _safe_int(validation_row.get("confidence_score")),
                    "annotator_notes": _clean_scalar(validation_row.get("annotator_notes")) or "",
                },
                "geo_summary": _pick_present(
                    geo_row,
                    "matched_gps_longitude_gcj02",
                    "matched_gps_latitude_gcj02",
                    "matched_gps_longitude_wgs84",
                    "matched_gps_latitude_wgs84",
                    "segment_center_time_utc",
                    "match_status",
                    "confidence",
                ),
                "visual_summary": _pick_present(
                    visual_row,
                    "vis_green_pixel_ratio_mean",
                    "vis_sky_blue_ratio_top_mean",
                    "vis_brightness_mean_mean",
                    "vis_edge_density_mean",
                    "ai_activity_major_label",
                    "ai_activity_suitable_labels",
                ),
                "soundscape_summary": _pick_present(
                    soundscape_row,
                    "top_k_events",
                    "event_class_distribution_json",
                    "group_ratio_traffic",
                    "group_ratio_human",
                    "group_ratio_nature",
                    "group_ratio_mechanical",
                    "audio_signal__loudness_proxy_db",
                ),
                "fusion_summary": _pick_present(
                    fusion_row,
                    "visual_semantic__road__mean",
                    "visual_semantic__sidewalk__mean",
                    "visual_semantic__building__mean",
                    "green_view__greenviewindex__mean",
                    "people__total_people__mean",
                    "emotion__beautiful__mean",
                    "emotion__depressing__mean",
                    "audio_signal__loudness_proxy_db",
                ),
                "diagnostics_summary": diagnostics_row,
                "relationship_summary": relationship_summary,
            }
        )

    return jsonify(
        {
            "video_name": video_name,
            "video_url": video_url,
            "validation_csv_path": validation_path.as_posix(),
            "summary": {
                "total_segments": int(len(segments)),
                "reviewed_segments": int(reviewed_count),
                "unreviewed_segments": int(len(segments) - reviewed_count),
            },
            "options": {
                "street_type": VALIDATION_STREET_TYPES,
                "problem_labels": VALIDATION_PROBLEM_LABEL_OPTIONS,
                "score_values": [1, 2, 3, 4, 5],
            },
            "segments": segments,
        }
    )


@app.route('/api/validation/<video_name>/save', methods=['POST'])
def save_validation_payload(video_name):
    """Persist one segment's validation annotations into final_annotation_labels_adjudicated.csv."""
    video_dir = _video_dir(video_name)
    if not video_dir.exists():
        return jsonify({"error": "未找到对应视频目录"}), 404

    manifest_path = video_dir / "segments" / "segment_manifest.csv"
    if not manifest_path.exists():
        return jsonify({"error": "缺少 segment_manifest.csv，无法保存评分"}), 404

    payload = request.get_json(silent=True) or {}
    segment_id = _safe_int(payload.get("segment_id"))
    if segment_id is None:
        return jsonify({"error": "缺少 segment_id，无法保存"}), 400

    manifest_df = _read_csv_df(manifest_path)
    validation_dir = _validation_dir(video_name)
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_path = _validation_csv_path(video_name)
    validation_df = _normalize_validation_df(manifest_df, _read_csv_df(validation_path))

    mask = validation_df["segment_id"].astype("Int64") == segment_id
    if not mask.any():
        return jsonify({"error": f"未找到 segment_id={segment_id} 的评分行"}), 404

    for column in VALIDATION_REQUIRED_COLUMNS:
        if column not in validation_df.columns:
            validation_df[column] = pd.NA

    updates = {
        "street_type": _clean_scalar(payload.get("street_type")) or "mixed_use",
        "comfort_score": _safe_int(payload.get("comfort_score")),
        "vitality_score": _safe_int(payload.get("vitality_score")),
        "soundscape_pleasantness": _safe_int(payload.get("soundscape_pleasantness")),
        "soundscape_eventfulness": _safe_int(payload.get("soundscape_eventfulness")),
        "overall_problem_severity": _safe_int(payload.get("overall_problem_severity")),
        "main_problem_labels": _json_array_string(payload.get("main_problem_labels")),
        "primary_problem_label": _clean_scalar(payload.get("primary_problem_label")) or "no_major_problem",
        "confidence_score": _safe_int(payload.get("confidence_score")),
        "annotator_notes": _clean_scalar(payload.get("annotator_notes")),
    }

    for column, value in updates.items():
        validation_df.loc[mask, column] = value

    validation_df = validation_df.sort_values("segment_id")
    validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")

    reviewed_count = sum(_is_reviewed_validation_row(record) for record in validation_df.to_dict("records"))
    return jsonify(
        {
            "status": "ok",
            "segment_id": segment_id,
            "saved_path": validation_path.as_posix(),
            "summary": {
                "total_segments": int(len(validation_df)),
                "reviewed_segments": int(reviewed_count),
                "unreviewed_segments": int(len(validation_df) - reviewed_count),
            },
        }
    )

@app.route('/api/image/<video_name>/<path:image_path>')
def serve_image(video_name, image_path):
    """Serve static images from video analysis results"""
    video_path = os.path.join(OUTPUT_DIR, video_name)
    full_image_path = os.path.join(video_path, image_path)

    if not os.path.exists(full_image_path):
        return "Image not found", 404

    return send_file(full_image_path)


@app.route('/api/asset/<video_name>/<path:asset_path>')
def serve_asset(video_name, asset_path):
    """Serve validation assets such as audio clips, previews, and frame strips."""
    video_path = os.path.join(OUTPUT_DIR, video_name)
    full_asset_path = os.path.join(video_path, asset_path)

    if not os.path.exists(full_asset_path):
        return "Asset not found", 404

    mime_type = mimetypes.guess_type(full_asset_path)[0] or ""
    if mime_type.startswith("audio/") or mime_type.startswith("video/"):
        return _send_media_file(Path(full_asset_path))
    return send_file(full_asset_path)


def run_web_server(host='0.0.0.0', port=5000, open_browser=False, quiet=True):
    """启动 Web 服务，可选自动打开浏览器。"""
    os.makedirs('web/templates', exist_ok=True)
    os.makedirs('web/static/css', exist_ok=True)
    os.makedirs('web/static/js', exist_ok=True)

    if quiet:
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

    if open_browser:
        browser_host = '127.0.0.1' if host in ('0.0.0.0', '::') else host
        url = f"http://{browser_host}:{port}"
        Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"Web 可视化服务已启动: http://127.0.0.1:{port}")
    print("按 Ctrl+C 停止服务")
    app.run(debug=False, host=host, port=port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    try:
        run_web_server(host='0.0.0.0', port=5000, open_browser=False, quiet=False)
    except KeyboardInterrupt:
        print("服务器已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
