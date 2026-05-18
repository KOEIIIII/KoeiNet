


"""
主程序（精简版）

能力范围：
1. 视频处理主流程
2. 全景投影与重建
3. 视觉分析
4. 统计与可视化输出
5. 音频分析
6. Web 服务配套数据产出
7. AI 六项活动评估（智谱 Prompt）
"""

import os
import sys
import glob
import time
import logging
import argparse
import warnings
import importlib.util
import subprocess
import io
import platform
import json
from typing import Any, Dict, Mapping, Optional
from contextlib import redirect_stdout


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2

from src.common.progress import create_progress_manager
from src.config import (
    INPUT_DIR,
    OUTPUT_DIR,
    VIDEO_EXTENSIONS,
    VIDEO_FRAME_SKIP,
    SEGMENT_SECONDS,
    SEGMENT_OVERLAP,
    PANNS_CHECKPOINT_PATH,
    PANNS_FORCE_LOCAL_RESOURCES,
    PANNS_LABELS_PATH,
    SOUNDSCAPE_PANNS_EXPORT_DIMS,
    VALIDATION_HIDDEN_DUPLICATES_PER_RATER,
    VALIDATION_RANDOM_SEED,
    VALIDATION_UNIQUE_SEGMENTS,
    STEP7_SEED,
    STEP75_SEED,
    STEP8_TOP_N,
    DEVICE,
    SEGMENTATION_MODEL_TYPE,
    PEOPLE_DETECTION_MODEL_TYPE,
    YOLO11_CONFIG,
    YOLOV8_CONFIG,
)

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
    )

    RICH_AVAILABLE = True
    console = Console()
    progress_console = Console(stderr=True)
except Exception:
    RICH_AVAILABLE = False
    console = None
    progress_console = None


def _print(msg: str, style: str = ""):
    if RICH_AVAILABLE and style:
        console.print(msg, style=style, markup=False)
    else:
        print(msg)


def _silence_cv_logs():
    try:
        cv2.setLogLevel(0)
    except Exception:
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except Exception:
            pass


def check_tensorflow():
    """仅检查 TensorFlow 是否已安装（不触发重导入日志）。"""
    return importlib.util.find_spec("tensorflow") is not None


_silence_cv_logs()
tensorflow_available = check_tensorflow()


def get_audio_analyzer_cls():
    """懒加载音频分析器，减少启动噪音。"""
    if not tensorflow_available:
        return None
    try:
        from src.audio.audio_analyzer import AudioAnalyzer

        return AudioAnalyzer
    except Exception:
        return None


warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("insta360_segmentation.log", encoding="utf-8")],
)
logger = logging.getLogger("main")


STEP_NAMES = {
    "环境准备": "环境准备",
    "提取帧": "视频帧提取",
    "投影处理": "全景投影",
    "情感分析": "情感评分",
    "人数统计": "人数检测",
    "色彩分析": "色彩分析",
    "语义分割": "语义分割",
    "边框标注": "边框标注",
    "反投影": "结果合成",
    "生成视频": "视频生成",
    "H.264转换": "H.264 转换",
    "统计分析": "统计输出",
    "图表生成": "图表生成",
    "关系分析": "关系分析",
    "音频分析": "音频分析",
}


def _build_progress_callback(video_label: str):
    """返回 (progress_obj, callback, finalize_fn)。"""
    if RICH_AVAILABLE:
        progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=32, complete_style="green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=False,
            console=progress_console,
        )
        task_id = progress.add_task(f"{video_label} | 准备中", total=100)

        def update_progress(step_name, percent):
            display = STEP_NAMES.get(step_name, step_name)
            pct = max(0.0, min(100.0, float(percent)))
            progress.update(task_id, completed=pct, description=f"{video_label} | {display}")

        def finalize():
            return None

        return progress, update_progress, finalize

    last_state = {"step": "", "bucket": -1}

    def update_progress(step_name, percent):
        display = STEP_NAMES.get(step_name, step_name)
        pct = max(0.0, min(100.0, float(percent)))
        bucket = int(pct // 10)
        if step_name != last_state["step"] or bucket != last_state["bucket"] or pct >= 100:
            _print(f"[{video_label}] {display}: {pct:5.1f}%")
            last_state["step"] = step_name
            last_state["bucket"] = bucket

    def finalize():
        return None

    return None, update_progress, finalize


def _run_quiet_stdout(fn, *args, **kwargs):
    """
    静默执行函数：屏蔽底层模块的大量 print 输出。
    返回 (result, captured_text)。
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def print_runtime_overview():
    """打印简洁的运行环境信息（CPU/GPU/设备）。"""
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "Unknown CPU")
    cpu_cores = os.cpu_count() or 0

    people_device = YOLO11_CONFIG.get("device", "cpu") if PEOPLE_DETECTION_MODEL_TYPE == "yolo11" else YOLOV8_CONFIG.get("device", "cpu")
    seg_device = DEVICE

    gpu_line = "GPU: 未检测到可用 CUDA 设备"
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            gpu_line = f"GPU: {', '.join(names)}"
    except Exception:
        pass

    _print("运行环境")
    _print(f"- CPU: {cpu_name} | 逻辑核: {cpu_cores}")
    _print(f"- {gpu_line}")
    _print(f"- 语义分割: {SEGMENTATION_MODEL_TYPE} @ {seg_device}")
    _print(f"- 人数检测: {PEOPLE_DETECTION_MODEL_TYPE} @ {people_device}")


def find_videos(input_dir):
    """查找目录中的视频文件。"""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    return sorted(videos)


def generate_all_charts(video_dir, progress_callback=None):
    """生成全部可视化图表。"""
    from src.utils.visualization import visualize_all_stats
    from src.utils.ai_evaluation import generate_ai_activity_outputs
    from src.utils.visual_metrics_analysis import analyze_visual_elements_vs_metrics

    chart_files = []

    if progress_callback:
        progress_callback("图表生成", 0)

    chart_files.extend(visualize_all_stats(video_dir) or [])
    if progress_callback:
        progress_callback("图表生成", 35)

    chart_files.extend(generate_ai_activity_outputs(video_dir) or [])
    if progress_callback:
        progress_callback("图表生成", 55)

    def rel_progress(step_name, percent):
        if progress_callback:
            progress_callback("关系分析", percent)

    chart_files.extend(analyze_visual_elements_vs_metrics(video_dir, progress_callback=rel_progress) or [])
    if progress_callback:
        progress_callback("图表生成", 100)

    return chart_files


def collect_csv_files(video_dir):
    """收集关键 CSV 结果文件。"""
    csv_files = {}

    stats_dir = os.path.join(video_dir, "stats")
    paths = {
        "detailed": os.path.join(stats_dir, "visual_elements", "detailed_categories_proportion.csv"),
        "major": os.path.join(stats_dir, "visual_elements", "major_categories_proportion.csv"),
        "green_view": os.path.join(stats_dir, "green_view", "green_view_index.csv"),
        "emotion": os.path.join(stats_dir, "emotion", "emotion_scores.csv"),
        "people_count": os.path.join(stats_dir, "people_count", "people_count.csv"),
        "color_analysis": os.path.join(stats_dir, "color_analysis", "color_categories_proportion.csv"),
        "ai_activity": os.path.join(video_dir, "ai_evaluation", "activity_scores.csv"),
        "audio_events_proportion": os.path.join(video_dir, "audio_events", "audio_events_proportion.csv"),
        "audio_events_detail": os.path.join(video_dir, "audio_events", "audio_events_detail.csv"),
        "audio_events_time_sync": os.path.join(video_dir, "audio_events", "audio_events_time_sync.csv"),
        "audio_events_time_sync_simple": os.path.join(video_dir, "audio_events", "audio_events_time_sync_simple.csv"),
    }

    for key, path in paths.items():
        if os.path.exists(path):
            csv_files[key] = path

    return csv_files


def _build_pipeline_overrides(args) -> Dict[str, Any]:
    """
    Build runtime overrides for optional multimodal post-analysis stages.

    Only non-None CLI values are used so `src/config.py` remains the default
    source of truth.
    """
    raw = {
        "ENABLE_SEGMENT_PIPELINE": getattr(args, "enable_segment_pipeline", None),
        "SEGMENT_SECONDS": getattr(args, "segment_seconds", None),
        "SEGMENT_OVERLAP": getattr(args, "segment_overlap", None),
        "ENABLE_VISUAL_SEGMENT_SUMMARY": getattr(args, "enable_visual_segment_summary", None),
        "ENABLE_GEO_SYNC": getattr(args, "enable_geo_sync", None),
        "GEO_SYNC_GPS_CSV": getattr(args, "geo_sync_gps_csv", None),
        "GEO_SYNC_TIME_OFFSET_SECONDS": getattr(args, "geo_sync_time_offset_seconds", None),
        "GEO_SYNC_EXPORT_WGS84": getattr(args, "geo_sync_export_wgs84", None),
        "GEO_SYNC_MAX_GAP_WARNING_SEC": getattr(args, "geo_sync_max_gap_warning_sec", None),
        "GEO_SYNC_USE_EXISTING_SEGMENTS": getattr(args, "geo_sync_use_existing_segments", None),
        "GEO_SYNC_SIDECAR_PATH": getattr(args, "geo_sync_sidecar_path", None),
        "GEO_SYNC_FILENAME_TZ_OFFSET_HOURS": getattr(args, "geo_sync_filename_tz_offset_hours", None),
        "GEO_SYNC_ALIGN_TO_ANALYSIS_FRAMES": getattr(args, "geo_sync_align_to_analysis_frames", None),
        "GEO_SYNC_FRAME_STEP": getattr(args, "geo_sync_frame_step", None),
        "ENABLE_WEB_SYNC_EXPORT": getattr(args, "enable_web_sync_export", None),
        "WEB_SYNC_PREFER_WGS84": getattr(args, "web_sync_prefer_wgs84", None),
        "ENABLE_GIS_EXPORT": getattr(args, "enable_gis_export", None),
        "GIS_EXPORT_PREFER_WGS84": getattr(args, "gis_export_prefer_wgs84", None),
        "ENABLE_SOUNDSCAPE": getattr(args, "enable_soundscape", None),
        "ENABLE_FUSION": getattr(args, "enable_fusion", None),
        "ENABLE_AGENTS": getattr(args, "enable_agents", None),
        "ENABLE_DESIGN": getattr(args, "enable_design", None),
        "ENABLE_DELIVERABLE": getattr(args, "enable_deliverable", None),
        "EXPORT_DEBUG_JSON": getattr(args, "export_debug_json", None),
        "STEP8_TOP_N": getattr(args, "step8_top_n", None),
        "POST_ONLY": bool(getattr(args, "post_only", False)),
        "RESUME_MISSING_ONLY": bool(getattr(args, "resume_missing_only", False)),
        "FROM_EXISTING_OUTPUT": getattr(args, "from_existing_output", None),
    }
    overrides = {k: v for k, v in raw.items() if v is not None}
    if "GEO_SYNC_FRAME_STEP" not in overrides and getattr(args, "frame_skip", None) is not None:
        overrides["GEO_SYNC_FRAME_STEP"] = int(getattr(args, "frame_skip"))
    return overrides


def _resolve_existing_video_dir(from_existing_output: str, video_name: Optional[str] = None) -> str:
    """
    Resolve existing output video directory.

    Supports:
    1) Direct video output folder: output/<video_name>
    2) Output root + helper name: output + --video_name <video_name>
    """
    path = os.path.abspath(from_existing_output)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"--from_existing_output 不存在或不是目录: {from_existing_output}")

    marker_dirs = ("frames", "stats", "ai_evaluation", "audio_events", "reproj", "split")
    if any(os.path.isdir(os.path.join(path, d)) for d in marker_dirs):
        return path

    if video_name:
        candidate = os.path.join(path, video_name)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
        raise FileNotFoundError(f"未找到指定视频输出目录: {candidate}")

    raise ValueError(
        "无法从 --from_existing_output 自动识别视频目录。"
        "请传入 output/<video_name> 或补充 --video_name。"
    )


def _print_readiness_report(video_dir: str):
    """Print concise readiness report for existing outputs."""
    try:
        from src.common.readiness import readiness_report_lines

        for line in readiness_report_lines(video_dir):
            _print(line)
    except Exception as exc:
        logger.error("读取 readiness 报告失败: %s", exc, exc_info=True)
        _print(f"[readiness] unavailable: {exc}", "yellow" if RICH_AVAILABLE else "")


def _resume_missing_legacy_outputs(
    video_dir: str,
    frame_skip: int,
    video_path_hint: Optional[str] = None,
):
    """
    Resume only missing lightweight legacy artifacts from existing outputs.

    Rules:
    - Never reruns expensive frame extraction/projection/segmentation pipeline.
    - Rebuilds old AI activity/audio outputs only when missing.
    """
    ai_csv = os.path.join(video_dir, "ai_evaluation", "activity_scores.csv")
    audio_dir = os.path.join(video_dir, "audio_events")
    audio_required = [
        os.path.join(audio_dir, "audio_events_proportion.csv"),
        os.path.join(audio_dir, "audio_events_detail.csv"),
        os.path.join(audio_dir, "audio_events_time_sync.csv"),
        os.path.join(audio_dir, "audio_events_time_sync_simple.csv"),
    ]

    if not os.path.exists(ai_csv):
        try:
            from src.utils.ai_evaluation import generate_ai_activity_outputs

            _print("[resume] 缺少 ai_evaluation/activity_scores.csv，正在补生成...")
            generate_ai_activity_outputs(video_dir)
            _print("[resume] AI 活动评估补生成完成。")
        except Exception as exc:
            logger.error("补生成 AI 活动评估失败: %s", exc, exc_info=True)
            _print(f"[resume] AI 活动评估补生成失败: {exc}", "yellow" if RICH_AVAILABLE else "")
    else:
        _print("[resume] AI 活动评估已存在，跳过重算。")

    missing_audio = [p for p in audio_required if not os.path.exists(p)]
    if not missing_audio:
        _print("[resume] 音频分析结果已存在，跳过重算。")
        return

    analyzer_cls = get_audio_analyzer_cls()
    if analyzer_cls is None:
        _print("[resume] 音频分析依赖不可用，无法补生成缺失音频结果。", "yellow" if RICH_AVAILABLE else "")
        return

    analyzer = analyzer_cls()
    wav_candidates = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))

    try:
        if wav_candidates:
            _print(f"[resume] 发现已有 WAV，补生成缺失音频结果: {os.path.basename(wav_candidates[0])}")
            analyzer.analyze_audio(wav_candidates[0], video_dir, video_fps=None, frame_skip=frame_skip)
            _print("[resume] 音频结果补生成完成。")
            return

        if video_path_hint and os.path.exists(video_path_hint):
            _print("[resume] 未发现 WAV，使用视频路径补生成音频结果。")
            analyzer.process_video(video_path_hint, video_dir, video_fps=None, frame_skip=frame_skip)
            _print("[resume] 音频结果补生成完成。")
            return

        _print(
            "[resume] 缺少可用 WAV 且未提供有效 --video_path，跳过音频补生成。",
            "yellow" if RICH_AVAILABLE else "",
        )
    except Exception as exc:
        logger.error("补生成音频结果失败: %s", exc, exc_info=True)
        _print(f"[resume] 音频结果补生成失败: {exc}", "yellow" if RICH_AVAILABLE else "")


def _run_optional_multimodal_pipeline(
    video_path: str,
    video_dir: str,
    output_dir: str,
    frame_skip: int,
    pipeline_overrides: Optional[Mapping[str, Any]] = None,
    progress_manager: Optional[Any] = None,
    overall_progress_task: Optional[Any] = None,
):
    """
    Run optional extension pipeline after current analysis artifacts exist.

    Backward compatibility note:
    - When all extension flags are disabled, this function exits without
      creating any new files.
    """
    try:
        from src.common.pipeline_registry import run_post_analysis_pipeline

        summary = run_post_analysis_pipeline(
            video_path=video_path,
            video_dir=video_dir,
            output_dir=output_dir,
            frame_skip=frame_skip,
            runtime_overrides=pipeline_overrides,
            progress_manager=progress_manager,
            overall_progress_task=overall_progress_task,
        )
        if summary.get("pipeline_enabled"):
            stages = summary.get("stages", [])
            disabled = [s["stage"] for s in stages if s.get("state") == "disabled"]
            skipped_existing = [s["stage"] for s in stages if s.get("state") == "skipped_existing"]
            ran = [s["stage"] for s in stages if s.get("state") == "ran"]
            failed = [s["stage"] for s in stages if s.get("state") == "failed"]
            _print(f"[post] disabled stages: {', '.join(disabled) if disabled else 'none'}")
            _print(
                f"[post] skipped_existing stages: {', '.join(skipped_existing) if skipped_existing else 'none'}"
            )
            _print(f"[post] ran stages: {', '.join(ran) if ran else 'none'}")
            _print(f"[post] failed stages: {', '.join(failed) if failed else 'none'}")
        return summary
    except Exception as exc:
        logger.error(f"扩展流水线执行失败: {exc}", exc_info=True)
        _print(f"扩展流水线执行失败: {exc}", "yellow" if RICH_AVAILABLE else "")
        return {"pipeline_enabled": False, "error": str(exc)}


def run_panns_self_check() -> int:
    """Validate local-only PANNs wiring without running full video pipeline."""
    try:
        from src.soundscape.panns_embedder import run_local_panns_self_check

        result = run_local_panns_self_check(
            checkpoint_path=PANNS_CHECKPOINT_PATH,
            labels_path=PANNS_LABELS_PATH,
            force_local_resources=bool(PANNS_FORCE_LOCAL_RESOURCES),
            export_dims=int(SOUNDSCAPE_PANNS_EXPORT_DIMS),
        )
        _print(str(result.get("summary", "[check_panns] status=fail reason=unknown")))
        return 0 if bool(result.get("ok", False)) else 1
    except Exception as exc:
        logger.error("PANNs 自检失败: %s", exc, exc_info=True)
        _print(f"[check_panns] status=fail reason={exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def _resolve_video_dir_for_validation(args, output_dir: str) -> str:
    """Resolve target video output dir for step-5 validation tasks."""
    if args.from_existing_output:
        return _resolve_existing_video_dir(args.from_existing_output, video_name=args.video_name)
    if args.video_name:
        candidate = os.path.join(output_dir, args.video_name)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    raise ValueError("Step-5 验证任务请提供 --from_existing_output output/<video_name>（或 --video_name）。")


def run_validation_pack(video_dir: str) -> int:
    """Generate two-rater blind annotation packs (step-5)."""
    try:
        from src.fusion.validation_pack import ValidationPackConfig, generate_two_rater_validation_pack

        result = generate_two_rater_validation_pack(
            video_dir=video_dir,
            config=ValidationPackConfig(
                unique_segments=int(VALIDATION_UNIQUE_SEGMENTS),
                hidden_duplicates_per_rater=int(VALIDATION_HIDDEN_DUPLICATES_PER_RATER),
                random_seed=int(VALIDATION_RANDOM_SEED),
            ),
        )
        _print(
            "[validation] pack generated "
            f"| unique={result.get('unique_segments')} "
            f"| dup_per_rater={result.get('hidden_duplicates_per_rater')} "
            f"| rows_per_rater={result.get('rows_per_rater')}"
        )
        _print(f"[validation] rater_A_pack={result.get('rater_A_annotation_pack_csv')}")
        _print(f"[validation] rater_B_pack={result.get('rater_B_annotation_pack_csv')}")
        _print(f"[validation] admin_manifest={result.get('sample_manifest_admin_csv')}")
        _print(f"[validation] randomization={result.get('session_randomization_json')}")
        _print(
            f"[validation] downstream_schema_compatible_step7={result.get('downstream_schema_compatible_step7')}"
        )
        return 0
    except Exception as exc:
        logger.error("生成验证包失败: %s", exc, exc_info=True)
        _print(f"[validation] pack generation failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_validation_web_app(
    video_dir: str,
    rater: str = "A",
    csv_path: Optional[str] = None,
) -> int:
    """Launch local Streamlit blind-annotation UI for step-5."""
    rid = str(rater or "A").strip().upper()
    if rid not in {"A", "B"}:
        rid = "A"

    target_csv = csv_path or os.path.join(video_dir, "validation", f"rater_{rid}_annotation_pack.csv")
    target_csv = os.path.abspath(target_csv)
    if not os.path.isfile(target_csv):
        _print(f"[validation_web] 标注 CSV 不存在: {target_csv}", "bold red" if RICH_AVAILABLE else "")
        return 1

    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "web", "validation_app.py"))
    if not os.path.isfile(app_path):
        _print(f"[validation_web] 未找到应用文件: {app_path}", "bold red" if RICH_AVAILABLE else "")
        return 1

    if importlib.util.find_spec("streamlit") is None:
        _print("[validation_web] 缺少 streamlit。请先安装: pip install streamlit", "bold red" if RICH_AVAILABLE else "")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.address",
        "127.0.0.1",
        "--",
        "--video_dir",
        os.path.abspath(video_dir),
        "--rater",
        rid,
        "--csv_path",
        target_csv,
    ]
    _print(f"[validation_web] launching rater={rid} csv={target_csv}")
    try:
        proc = subprocess.run(cmd, check=False)
        return int(proc.returncode)
    except Exception as exc:
        logger.error("启动 validation web 失败: %s", exc, exc_info=True)
        _print(f"[validation_web] launch failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_validation_reliability(
    video_dir: str,
    rater_a_csv: Optional[str] = None,
    rater_b_csv: Optional[str] = None,
    admin_manifest_csv: Optional[str] = None,
) -> int:
    """Compute inter/intra-rater reliability (step-5)."""
    try:
        from src.fusion.validation_reliability import compute_validation_reliability

        result = compute_validation_reliability(
            video_dir=video_dir,
            rater_a_csv=rater_a_csv,
            rater_b_csv=rater_b_csv,
            admin_manifest_csv=admin_manifest_csv,
        )
        _print(
            "[validation] reliability computed "
            f"| inter_unique={result.get('inter_rater_unique_segments')}"
        )
        _print(f"[validation] reliability_json={result.get('reliability_report_json')}")
        _print(f"[validation] reliability_md={result.get('reliability_summary_md')}")
        return 0
    except Exception as exc:
        logger.error("计算验证一致性失败: %s", exc, exc_info=True)
        _print(f"[validation] reliability failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_validation_finalize(
    video_dir: str,
    rater_a_csv: Optional[str] = None,
    rater_b_csv: Optional[str] = None,
    admin_manifest_csv: Optional[str] = None,
) -> int:
    """Finalize two-rater annotations into one segment-level label file (step-5)."""
    try:
        from src.fusion.validation_finalize import finalize_two_rater_labels

        result = finalize_two_rater_labels(
            video_dir=video_dir,
            rater_a_csv=rater_a_csv,
            rater_b_csv=rater_b_csv,
            admin_manifest_csv=admin_manifest_csv,
        )
        _print(f"[validation] final_labels={result.get('final_annotation_labels_csv')}")
        _print(f"[validation] finalization_report={result.get('finalization_report_json')}")
        _print(
            f"[validation] downstream_schema_compatible_step7={result.get('downstream_schema_compatible_step7')}"
        )
        return 0
    except Exception as exc:
        logger.error("生成最终验证标签失败: %s", exc, exc_info=True)
        _print(f"[validation] finalize failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_adjudication_pack(
    video_dir: str,
    rater_a_csv: Optional[str] = None,
    rater_b_csv: Optional[str] = None,
    admin_manifest_csv: Optional[str] = None,
    reliability_report_json: Optional[str] = None,
) -> int:
    """Build Step-5.5 targeted adjudication subset pack."""
    try:
        from src.fusion.adjudication_pack import build_adjudication_pack

        result = build_adjudication_pack(
            video_dir=video_dir,
            rater_a_csv=rater_a_csv,
            rater_b_csv=rater_b_csv,
            admin_manifest_csv=admin_manifest_csv,
            reliability_report_json=reliability_report_json,
        )
        _print(
            "[adjudication] pack built "
            f"| flagged={result.get('flagged_segments')} "
            f"| scalar={result.get('segments_with_scalar_disagreement')} "
            f"| label={result.get('segments_with_label_disagreement')} "
            f"| intra={result.get('segments_with_intra_rater_instability')}"
        )
        _print(f"[adjudication] pack_csv={result.get('adjudication_pack_csv')}")
        _print(f"[adjudication] admin_manifest={result.get('adjudication_manifest_admin_csv')}")
        _print(f"[adjudication] instructions={result.get('adjudication_instructions_md')}")
        return 0
    except Exception as exc:
        logger.error("生成 Step-5.5 裁决包失败: %s", exc, exc_info=True)
        _print(f"[adjudication] build pack failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_adjudication_web_app(
    video_dir: str,
    adjudication_pack_csv: Optional[str] = None,
) -> int:
    """Launch local Streamlit adjudication UI for Step-5.5."""
    target_csv = adjudication_pack_csv or os.path.join(video_dir, "validation", "adjudication_pack.csv")
    target_csv = os.path.abspath(target_csv)
    if not os.path.isfile(target_csv):
        _print(f"[adjudication_web] 裁决 CSV 不存在: {target_csv}", "bold red" if RICH_AVAILABLE else "")
        return 1

    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "web", "adjudication_app.py"))
    if not os.path.isfile(app_path):
        _print(f"[adjudication_web] 未找到应用文件: {app_path}", "bold red" if RICH_AVAILABLE else "")
        return 1

    if importlib.util.find_spec("streamlit") is None:
        _print("[adjudication_web] 缺少 streamlit。请先安装: pip install streamlit", "bold red" if RICH_AVAILABLE else "")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.address",
        "127.0.0.1",
        "--",
        "--video_dir",
        os.path.abspath(video_dir),
        "--csv_path",
        target_csv,
    ]
    _print(f"[adjudication_web] launching csv={target_csv}")
    try:
        proc = subprocess.run(cmd, check=False)
        return int(proc.returncode)
    except Exception as exc:
        logger.error("启动 adjudication web 失败: %s", exc, exc_info=True)
        _print(f"[adjudication_web] launch failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_adjudication_finalize(
    video_dir: str,
    adjudication_pack_csv: Optional[str] = None,
    baseline_final_csv: Optional[str] = None,
    rater_a_csv: Optional[str] = None,
    rater_b_csv: Optional[str] = None,
    admin_manifest_csv: Optional[str] = None,
) -> int:
    """Apply Step-5.5 adjudicated values and export final adjudicated labels."""
    try:
        from src.fusion.adjudication_finalize import finalize_adjudicated_labels

        result = finalize_adjudicated_labels(
            video_dir=video_dir,
            adjudication_pack_csv=adjudication_pack_csv,
            baseline_final_csv=baseline_final_csv,
            rater_a_csv=rater_a_csv,
            rater_b_csv=rater_b_csv,
            admin_manifest_csv=admin_manifest_csv,
        )
        _print(f"[adjudication] final_labels={result.get('final_annotation_labels_adjudicated_csv')}")
        _print(f"[adjudication] report={result.get('adjudication_report_json')}")
        _print(
            f"[adjudication] step7_core_schema_compatible={result.get('step7_core_schema_compatible')}"
        )
        return 0
    except Exception as exc:
        logger.error("生成 Step-5.5 裁决最终标签失败: %s", exc, exc_info=True)
        _print(f"[adjudication] finalize failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_step7_fusion_eval_task(
    video_dir: str,
    feature_csv: Optional[str] = None,
    labels_csv: Optional[str] = None,
    step7_outdir: Optional[str] = None,
    seed: Optional[int] = None,
    smoke_test: bool = False,
    clean_outdir: bool = False,
    show_progress: bool = False,
) -> int:
    """Run Step-7 fusion modeling/evaluation using existing artifacts only."""
    try:
        from src.fusion.step7_runner import run_step7_fusion_eval

        result = run_step7_fusion_eval(
            video_dir=video_dir,
            feature_csv=feature_csv,
            labels_csv=labels_csv,
            step7_outdir=step7_outdir,
            seed=seed,
            smoke_test=bool(smoke_test),
            clean_outdir=bool(clean_outdir),
            show_progress=bool(show_progress),
        )
        _print(f"[step7] labels_source={result.get('label_source_csv')}")
        _print(f"[step7] modeling_dataset={result.get('step7_modeling_dataset_csv')}")
        _print(f"[step7] target_registry={result.get('target_registry_json')}")
        _print(f"[step7] feature_group_registry={result.get('feature_group_registry_json')}")
        _print(f"[step7] per_target_metrics={result.get('per_target_metrics_csv')}")
        _print(f"[step7] model_comparison={result.get('model_comparison_csv')}")
        _print(f"[step7] paired_deltas={result.get('paired_deltas_csv')}")
        _print(f"[step7] bootstrap_ci={result.get('bootstrap_ci_json')}")
        _print(f"[step7] oof_predictions={result.get('oof_predictions_csv')}")
        _print(f"[step7] cv_split_registry={result.get('cv_split_registry_json')}")
        _print(f"[step7] feature_importance={result.get('feature_importance_csv')}")
        _print(f"[step7] shap_summary={result.get('shap_summary_csv')}")
        _print(f"[step7] plots_dir={result.get('plots_dir')}")
        _print(f"[step7] summary={result.get('step7_summary_md')}")
        _print(f"[step7] progress_file={result.get('step7_progress_json')}")
        _print(f"[step7] run_manifest={result.get('step7_run_manifest_json')}")
        return 0
    except Exception as exc:
        logger.error("Step-7 融合评估失败: %s", exc, exc_info=True)
        _print(f"[step7] fusion_eval failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_step75_refined_eval_task(
    video_dir: str,
    feature_csv: Optional[str] = None,
    labels_csv: Optional[str] = None,
    step75_outdir: Optional[str] = None,
    seed: Optional[int] = None,
    smoke_test: bool = False,
    reuse_step7_splits: Optional[bool] = None,
) -> int:
    """Run Step-7.5 refined fusion evaluation using existing artifacts only."""
    try:
        from src.fusion.step75_runner import run_step75_refined_eval
        last_stage = "初始化"
        with create_progress_manager() as progress_manager:
            task = progress_manager.add_task("step75 | 初始化", total=100.0)

            def _on_progress(completed: float, total: float, description: str) -> None:
                nonlocal last_stage
                last_stage = str(description)
                pct = 0.0 if float(total) <= 0 else max(0.0, min(100.0, 100.0 * float(completed) / float(total)))
                task.update(total=100.0, completed=pct, description=f"step75 | {description}")

            result = run_step75_refined_eval(
                video_dir=video_dir,
                feature_csv=feature_csv,
                labels_csv=labels_csv,
                step75_outdir=step75_outdir,
                seed=seed,
                smoke_test=bool(smoke_test),
                reuse_step7_splits=reuse_step7_splits,
                progress_callback=_on_progress,
            )
            task.finish("done", "step75 | refined evaluation")
        _print(f"[step75] labels_source={result.get('labels_source_csv')}")
        labels_schema = result.get("labels_schema_summary", {}) or {}
        _print(f"[step75] labels_schema_compatible={labels_schema.get('compatible_for_step75')}")
        _print(f"[step75] labels_target_columns_present={labels_schema.get('target_columns_present')}")
        _print(f"[step75] labels_target_columns_missing={labels_schema.get('target_columns_missing')}")
        _print(f"[step75] labels_usable_targets={labels_schema.get('usable_target_columns')}")
        _print(f"[step75] modeling_dataset={result.get('step75_modeling_dataset_csv')}")
        _print(f"[step75] target_registry={result.get('target_registry_refined_json')}")
        _print(f"[step75] feature_group_registry={result.get('feature_group_registry_refined_json')}")
        _print(f"[step75] cv_split_registry={result.get('cv_split_registry_refined_json')}")
        _print(f"[step75] feature_screening_registry={result.get('feature_screening_registry_json')}")
        _print(f"[step75] per_target_metrics={result.get('per_target_metrics_refined_csv')}")
        _print(f"[step75] model_comparison={result.get('model_comparison_refined_csv')}")
        _print(f"[step75] paired_deltas={result.get('paired_deltas_refined_csv')}")
        _print(f"[step75] bootstrap_ci={result.get('bootstrap_ci_refined_json')}")
        _print(f"[step75] oof_predictions={result.get('oof_predictions_refined_csv')}")
        _print(f"[step75] permutation_importance={result.get('permutation_importance_csv')}")
        _print(f"[step75] shap_summary={result.get('shap_summary_refined_csv')}")
        _print(f"[step75] explainability_report={result.get('explainability_report_json')}")
        _print(f"[step75] step7_vs_step75={result.get('step7_vs_step75_comparison_csv')}")
        _print(f"[step75] summary={result.get('step75_summary_md')}")
        _print(f"[step75] plots_dir={result.get('plots_dir')}")
        return 0
    except Exception as exc:
        logger.error("Step-7.5 精细化融合评估失败: %s", exc, exc_info=True)
        stage_info = locals().get("last_stage", "初始化")
        _print(f"[step75] refined_eval failed at stage={stage_info}: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_step8_design_mapping_task(
    video_dir: str,
    step8_outdir: Optional[str] = None,
    top_n: Optional[int] = None,
    smoke_test: bool = False,
) -> int:
    """Run Step-8 design mapping using existing diagnostics and Step-7.5 evidence only."""
    try:
        from src.design.step8_runner import run_step8_design_mapping

        result = run_step8_design_mapping(
            video_dir=video_dir,
            step8_outdir=step8_outdir,
            top_n=top_n,
            smoke_test=bool(smoke_test),
        )
        _print(f"[step8] evidence_registry={result.get('step8_evidence_registry_json')}")
        _print(f"[step8] priority_ranking={result.get('segment_priority_ranking_csv')}")
        _print(f"[step8] design_plan={result.get('design_plan_jsonl')}")
        _print(f"[step8] intervention_matrix={result.get('intervention_matrix_csv')}")
        _print(f"[step8] edit_prompts={result.get('edit_prompts_jsonl')}")
        _print(f"[step8] summary={result.get('step8_design_summary_md')}")
        _print(f"[step8] total_segments_ranked={result.get('total_segments_ranked')}")
        _print(f"[step8] selected_segments_for_design_plan={result.get('selected_segments_for_design_plan')}")
        return 0
    except Exception as exc:
        logger.error("Step-8 设计映射失败: %s", exc, exc_info=True)
        _print(f"[step8] design_mapping failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_relationship_analysis_task(
    video_dir: str,
    relationship_outdir: Optional[str] = None,
) -> int:
    """Run research relationship analysis using existing fusion/soundscape artifacts only."""
    try:
        from src.research.relationship_runner import run_relationship_analysis

        result = run_relationship_analysis(
            video_dir=video_dir,
            relationship_outdir=relationship_outdir,
        )
        _print(f"[relationship] feature_registry={result.get('feature_registry_csv')}")
        _print(f"[relationship] spearman_matrix={result.get('feature_correlation_matrix_spearman_csv')}")
        _print(f"[relationship] dcor_matrix={result.get('feature_correlation_matrix_dcor_csv')}")
        _print(f"[relationship] pvalues={result.get('feature_correlation_pvalues_csv')}")
        _print(f"[relationship] qvalues={result.get('feature_correlation_qvalues_csv')}")
        _print(f"[relationship] significant_links={result.get('significant_pairwise_links_csv')}")
        _print(f"[relationship] pls_latent_summary={result.get('pls_latent_summary_csv')}")
        _print(f"[relationship] pls_x_loadings={result.get('pls_x_loadings_csv')}")
        _print(f"[relationship] pls_y_loadings={result.get('pls_y_loadings_csv')}")
        _print(f"[relationship] pls_bootstrap_stability={result.get('pls_bootstrap_stability_csv')}")
        _print(f"[relationship] pls_permutation_test={result.get('pls_permutation_test_json')}")
        _print(f"[relationship] summary={result.get('relationship_summary_md')}")
        _print(f"[relationship] figures_dir={result.get('figures_dir')}")
        return 0
    except Exception as exc:
        logger.error("Relationship analysis 失败: %s", exc, exc_info=True)
        _print(f"[relationship] analysis failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_proof_package_task(
    video_dir: str,
    proof_outdir: Optional[str] = None,
) -> int:
    """Run research proof package using existing Step-7.5 refined artifacts only."""
    try:
        from src.research.proof_runner import run_proof_package

        result = run_proof_package(
            video_dir=video_dir,
            proof_outdir=proof_outdir,
        )
        _print(f"[proof] model_comparison={result.get('proof_model_comparison_csv')}")
        _print(f"[proof] paired_deltas={result.get('proof_paired_deltas_csv')}")
        _print(f"[proof] bootstrap_ci={result.get('proof_bootstrap_ci_csv')}")
        _print(f"[proof] permutation_tests={result.get('proof_permutation_tests_csv')}")
        _print(f"[proof] claim_registry={result.get('proof_claim_registry_csv')}")
        _print(f"[proof] summary={result.get('proof_summary_md')}")
        _print(f"[proof] figures_dir={result.get('figures_dir')}")
        return 0
    except Exception as exc:
        logger.error("Proof package 失败: %s", exc, exc_info=True)
        _print(f"[proof] package failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_group_confirmatory_relationship_task(
    video_dir: str,
    group_confirmatory_outdir: Optional[str] = None,
) -> int:
    """Run group-level confirmatory relationship analysis using existing relationship outputs only."""
    try:
        from src.research.group_confirmatory import run_group_confirmatory_relationship

        result = run_group_confirmatory_relationship(
            video_dir=video_dir,
            group_confirmatory_outdir=group_confirmatory_outdir,
        )
        _print(f"[group_confirmatory] group_definition_registry={result.get('group_definition_registry_csv')}")
        _print(f"[group_confirmatory] group_composites={result.get('group_composites_csv')}")
        _print(f"[group_confirmatory] diagnostics={result.get('group_composite_diagnostics_csv')}")
        _print(f"[group_confirmatory] hypothesis_registry={result.get('hypothesis_registry_csv')}")
        _print(f"[group_confirmatory] tests_full={result.get('group_pair_tests_full_csv')}")
        _print(f"[group_confirmatory] tests_thin={result.get('group_pair_tests_thin_csv')}")
        _print(f"[group_confirmatory] tests_combined={result.get('group_pair_tests_combined_csv')}")
        _print(f"[group_confirmatory] lofo_robustness={result.get('leave_one_feature_out_robustness_csv')}")
        _print(f"[group_confirmatory] time_trend={result.get('time_trend_sensitivity_csv')}")
        _print(f"[group_confirmatory] claim_registry={result.get('confirmatory_claim_registry_csv')}")
        _print(f"[group_confirmatory] summary={result.get('group_confirmatory_summary_md')}")
        _print(f"[group_confirmatory] onepage_report={result.get('group_confirmatory_onepage_report_md')}")
        _print(f"[group_confirmatory] figures_dir={result.get('figures_dir')}")
        return 0
    except Exception as exc:
        logger.error("Group confirmatory relationship 失败: %s", exc, exc_info=True)
        _print(f"[group_confirmatory] analysis failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_paper_figures_task(
    video_dir: str,
    paper_figures_outdir: Optional[str] = None,
) -> int:
    """Generate four unified paper figures from existing relationship/proof outputs."""
    try:
        from src.research.paper_figures_runner import run_paper_figures

        result = run_paper_figures(
            video_dir=video_dir,
            paper_figures_outdir=paper_figures_outdir,
        )
        _print(f"[paper_figures] figA1_png={result.get('figA1_pls_lv1_coupling_png')}")
        _print(f"[paper_figures] figA1_pdf={result.get('figA1_pls_lv1_coupling_pdf')}")
        _print(f"[paper_figures] figA2_png={result.get('figA2_group_association_matrix_png')}")
        _print(f"[paper_figures] figA2_pdf={result.get('figA2_group_association_matrix_pdf')}")
        _print(f"[paper_figures] figB1_png={result.get('figB1_targetwise_model_dumbbell_png')}")
        _print(f"[paper_figures] figB1_pdf={result.get('figB1_targetwise_model_dumbbell_pdf')}")
        _print(f"[paper_figures] figB2_png={result.get('figB2_fusion_incremental_forest_png')}")
        _print(f"[paper_figures] figB2_pdf={result.get('figB2_fusion_incremental_forest_pdf')}")
        _print(f"[paper_figures] summary={result.get('paper_figures_summary_md')}")
        return 0
    except Exception as exc:
        logger.error("Paper figures 失败: %s", exc, exc_info=True)
        _print(f"[paper_figures] generation failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def run_deliverable_layer_task(
    video_dir: str,
    *,
    deliverable_top_k: Optional[int] = None,
    deliverable_top_percent: Optional[float] = None,
    deliverable_priority_threshold: Optional[float] = None,
    deliverable_max_gap_seconds: float = 5.0,
    deliverable_export_cards: bool = False,
    deliverable_use_glm: bool = False,
    deliverable_render_html: bool = False,
    deliverable_render_pdf: bool = False,
) -> int:
    """Generate the final deliverable layer from existing Step-8 outputs."""
    try:
        from src.deliverable.deliverable_runner import run_deliverable_layer

        result = run_deliverable_layer(
            video_dir=video_dir,
            deliverable_top_k=deliverable_top_k,
            deliverable_top_percent=deliverable_top_percent,
            deliverable_priority_threshold=deliverable_priority_threshold,
            deliverable_max_gap_seconds=deliverable_max_gap_seconds,
            deliverable_export_cards=bool(deliverable_export_cards),
            deliverable_use_glm=bool(deliverable_use_glm),
            deliverable_render_html=bool(deliverable_render_html),
            deliverable_render_pdf=bool(deliverable_render_pdf),
        )
        _print(f"[deliverable] problem_episodes={result.get('problem_episodes_csv')}")
        _print(f"[deliverable] episode_evidence={result.get('problem_episode_evidence_csv')}")
        _print(f"[deliverable] episode_summary={result.get('problem_episode_summary_csv')}")
        _print(f"[deliverable] episode_prompts={result.get('problem_episode_prompts_csv')}")
        _print(f"[deliverable] final_table_csv={result.get('final_problem_segments_table_csv')}")
        _print(f"[deliverable] final_table_xlsx={result.get('final_problem_segments_table_xlsx')}")
        _print(f"[deliverable] cards_dir={result.get('problem_episode_cards_dir')}")
        _print(f"[deliverable] cards_html={result.get('problem_episode_cards_html')}")
        _print(f"[deliverable] contact_sheet_pdf={result.get('problem_episode_contact_sheet_pdf')}")
        _print(f"[deliverable] summary={result.get('deliverable_summary_md')}")
        _print(f"[deliverable] onepage={result.get('deliverable_onepage_report_md')}")
        _print(f"[deliverable] shadow_eval={result.get('shadow_eval_notes_md')}")
        _print(f"[deliverable] prompt_mode={result.get('prompt_mode')}")
        _print(f"[deliverable] card_font={result.get('card_font_name')}")
        return 0
    except Exception as exc:
        logger.error("Deliverable layer 失败: %s", exc, exc_info=True)
        _print(f"[deliverable] generation failed: {exc}", "yellow" if RICH_AVAILABLE else "")
        return 1


def process_from_existing_output(
    from_existing_output: str,
    output_dir: str,
    frame_skip: int,
    pipeline_overrides: Optional[Mapping[str, Any]] = None,
    post_only: bool = False,
    resume_missing_only: bool = False,
    video_name: Optional[str] = None,
    video_path_hint: Optional[str] = None,
):
    """
    Process from already existing output folder without recomputing legacy frames.

    Safety note:
    - Always target one concrete `output/<video_name>/` directory.
    - Do not delete the whole `output/` root just to rebuild one video's
      downstream stages.
    """
    video_dir = _resolve_existing_video_dir(from_existing_output, video_name=video_name)
    resolved_video_name = os.path.basename(video_dir.rstrip("\\/"))
    pseudo_video_path = video_path_hint or os.path.join("input", f"{resolved_video_name}.mp4")

    _print(f"[from_existing_output] 已启用，跳过 legacy 帧级重计算: {video_dir}", "cyan" if RICH_AVAILABLE else "")
    _print_readiness_report(video_dir)

    if post_only:
        _print("[post_only] 仅运行新多模态阶段（segment/visual/geo_sync/soundscape/fusion/agents/design/deliverable/gis_export/web_sync）。")
    elif resume_missing_only:
        _print("[resume_missing_only] 仅补生成缺失的旧 AI/音频产物，不重跑 legacy 帧级处理。")
        _resume_missing_legacy_outputs(video_dir, frame_skip=frame_skip, video_path_hint=video_path_hint)
    else:
        _print("[from_existing_output] 未启用 --resume_missing_only，旧 AI/音频产物不会重算。")

    try:
        from src.common.pipeline_registry import STAGE_SPECS

        total_stages = len(STAGE_SPECS)
    except Exception:
        total_stages = 9

    with create_progress_manager() as progress_manager:
        overall_task = progress_manager.add_task(
            f"{resolved_video_name} | post-analysis pipeline",
            total=total_stages,
        )
        summary = _run_optional_multimodal_pipeline(
            video_path=pseudo_video_path,
            video_dir=video_dir,
            output_dir=output_dir,
            frame_skip=frame_skip,
            pipeline_overrides=pipeline_overrides,
            progress_manager=progress_manager,
            overall_progress_task=overall_task,
        )

    csv_files = collect_csv_files(video_dir)
    return video_dir, csv_files, summary


def process_video(video_path, output_dir, frame_skip=VIDEO_FRAME_SKIP, pipeline_overrides=None):
    """处理单个视频。"""
    try:
        from src.model.processor import PanoSegmentationProcessor
    except ModuleNotFoundError as exc:
        _print(f"缺少依赖，无法处理视频: {exc}", "bold red")
        _print("请先执行: pip install -r requirements.txt", "yellow")
        return None, {}

    video_name = os.path.basename(video_path)
    label = os.path.splitext(video_name)[0]

    _print(f"\n开始处理: {video_name}", "bold cyan" if RICH_AVAILABLE else "")
    _print(f"参数: output={output_dir}, frame_skip={frame_skip}")

    progress, update_progress, finalize = _build_progress_callback(label)
    start_time = time.time()

    try:
        if progress:
            with progress:
                def _run_processor():
                    processor_ = PanoSegmentationProcessor(
                        input_video=video_path,
                        output_dir=output_dir,
                        progress_callback=update_progress,
                    )
                    return processor_.process_video(frame_skip=frame_skip)

                output_video, _ = _run_quiet_stdout(_run_processor)
        else:
            def _run_processor_plain():
                processor_ = PanoSegmentationProcessor(
                    input_video=video_path,
                    output_dir=output_dir,
                    progress_callback=update_progress,
                )
                return processor_.process_video(frame_skip=frame_skip)

            output_video, _ = _run_quiet_stdout(_run_processor_plain)
    finally:
        finalize()

    if not output_video:
        _print("处理失败：未生成输出视频。", "bold red" if RICH_AVAILABLE else "")
        return None, {}

    video_dir = os.path.join(output_dir, os.path.splitext(video_name)[0])

    chart_progress, chart_update_progress, chart_finalize = _build_progress_callback(f"{label}#charts")
    try:
        if chart_progress:
            with chart_progress:
                chart_files, _ = _run_quiet_stdout(generate_all_charts, video_dir, chart_update_progress)
        else:
            chart_files, _ = _run_quiet_stdout(generate_all_charts, video_dir, chart_update_progress)
    finally:
        chart_finalize()

    if tensorflow_available:
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
            cap.release()

            _print("执行音频分析...")
            analyzer_cls = get_audio_analyzer_cls()
            if analyzer_cls is None:
                raise RuntimeError("音频分析模块加载失败（TensorFlow 依赖不可用或导入异常）")
            analyzer = analyzer_cls()
            audio_progress, audio_update_progress, audio_finalize = _build_progress_callback(f"{label}#audio")
            try:
                if audio_progress:
                    with audio_progress:
                        audio_update_progress("音频分析", 5)
                        _, _ = _run_quiet_stdout(
                            analyzer.process_video,
                            video_path,
                            video_dir,
                            video_fps=fps,
                            frame_skip=frame_skip,
                        )
                        audio_update_progress("音频分析", 100)
                else:
                    audio_update_progress("音频分析", 5)
                    _, _ = _run_quiet_stdout(
                        analyzer.process_video,
                        video_path,
                        video_dir,
                        video_fps=fps,
                        frame_skip=frame_skip,
                    )
                    audio_update_progress("音频分析", 100)
            finally:
                audio_finalize()
        except Exception as exc:
            logger.error(f"音频分析失败: {exc}", exc_info=True)
            _print(f"音频分析失败: {exc}", "yellow" if RICH_AVAILABLE else "")
    else:
        _print("TensorFlow 不可用，已跳过音频分析。", "yellow" if RICH_AVAILABLE else "")

    _run_optional_multimodal_pipeline(
        video_path=video_path,
        video_dir=video_dir,
        output_dir=output_dir,
        frame_skip=frame_skip,
        pipeline_overrides=pipeline_overrides,
    )

    csv_files = collect_csv_files(video_dir)
    elapsed = time.time() - start_time

    _print(
        f"完成: {video_name} | 耗时 {elapsed:.1f}s | 图表 {len(chart_files)} | CSV {len(csv_files)}",
        "bold green" if RICH_AVAILABLE else "",
    )
    return output_video, csv_files


def process_video_with_progress(video_path, output_dir, frame_skip=VIDEO_FRAME_SKIP, pipeline_overrides=None):
    """Process one video with a unified overall progress bar plus per-stage progress."""
    try:
        from src.model.processor import PanoSegmentationProcessor
        from src.common.pipeline_registry import STAGE_SPECS
    except ModuleNotFoundError as exc:
        _print(f"缺少依赖，无法处理视频: {exc}", "bold red")
        _print("请先执行: pip install -r requirements.txt", "yellow")
        return None, {}

    video_name = os.path.basename(video_path)
    label = os.path.splitext(video_name)[0]

    _print(f"\n开始处理: {video_name}", "bold cyan" if RICH_AVAILABLE else "")
    _print(f"参数: output={output_dir}, frame_skip={frame_skip}")

    start_time = time.time()
    total_pipeline_stages = 1 + len(STAGE_SPECS)

    with create_progress_manager() as progress_manager:
        overall_task = progress_manager.add_task(
            f"{label} | total pipeline",
            total=total_pipeline_stages,
        )
        base_task = progress_manager.add_task(
            f"{label} | 基础视觉/音频处理",
            total=100.0,
        )

        def update_progress(step_name, percent):
            display = STEP_NAMES.get(step_name, step_name)
            pct = max(0.0, min(100.0, float(percent)))
            base_task.update_percent(pct * 0.7, description=f"{label} | {display}")

        def _run_processor():
            processor_ = PanoSegmentationProcessor(
                input_video=video_path,
                output_dir=output_dir,
                progress_callback=update_progress,
            )
            return processor_.process_video(frame_skip=frame_skip)

        try:
            output_video, _ = _run_quiet_stdout(_run_processor)
        except Exception:
            base_task.finish(status="failed", description=f"{label} | 基础视觉/音频处理")
            raise

        if not output_video:
            base_task.finish(status="failed", description=f"{label} | 基础视觉/音频处理")
            _print("处理失败：未生成输出视频。", "bold red" if RICH_AVAILABLE else "")
            return None, {}

        video_dir = os.path.join(output_dir, os.path.splitext(video_name)[0])

        def chart_update_progress(step_name, percent):
            display = STEP_NAMES.get(step_name, step_name)
            pct = max(0.0, min(100.0, float(percent)))
            base_task.update_percent(70.0 + pct * 0.15, description=f"{label} | {display}")

        chart_files, _ = _run_quiet_stdout(generate_all_charts, video_dir, chart_update_progress)

        if tensorflow_available:
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
                cap.release()

                _print("执行音频分析...")
                analyzer_cls = get_audio_analyzer_cls()
                if analyzer_cls is None:
                    raise RuntimeError("音频分析模块加载失败（TensorFlow 依赖不可用或导入异常）")
                analyzer = analyzer_cls()

                def audio_update_progress(step_name, percent):
                    display = STEP_NAMES.get(step_name, step_name)
                    pct = max(0.0, min(100.0, float(percent)))
                    base_task.update_percent(85.0 + pct * 0.15, description=f"{label} | {display}")

                audio_update_progress("音频分析", 5)
                _, _ = _run_quiet_stdout(
                    analyzer.process_video,
                    video_path,
                    video_dir,
                    video_fps=fps,
                    frame_skip=frame_skip,
                )
                audio_update_progress("音频分析", 100)
            except Exception as exc:
                logger.error("音频分析失败: %s", exc, exc_info=True)
                _print(f"音频分析失败: {exc}", "yellow" if RICH_AVAILABLE else "")
                base_task.update_percent(100.0, description=f"{label} | 基础视觉/音频处理（音频失败）")
        else:
            _print("TensorFlow 不可用，已跳过音频分析。", "yellow" if RICH_AVAILABLE else "")
            base_task.update_percent(100.0, description=f"{label} | 基础视觉/音频处理（无音频）")

        base_task.finish(status="done", description=f"{label} | 基础视觉/音频处理")
        overall_task.advance(1, description=f"{label} | 基础视觉/音频处理 [done]")

        _run_optional_multimodal_pipeline(
            video_path=video_path,
            video_dir=video_dir,
            output_dir=output_dir,
            frame_skip=frame_skip,
            pipeline_overrides=pipeline_overrides,
            progress_manager=progress_manager,
            overall_progress_task=overall_task,
        )

    csv_files = collect_csv_files(video_dir)
    elapsed = time.time() - start_time
    _print(
        f"完成: {video_name} | 耗时 {elapsed:.1f}s | 图表 {len(chart_files)} | CSV {len(csv_files)}",
        "bold green" if RICH_AVAILABLE else "",
    )
    return output_video, csv_files


def process_all_videos(input_dir, output_dir, frame_skip=VIDEO_FRAME_SKIP, pipeline_overrides=None):
    """批量处理目录中的视频。"""
    videos = find_videos(input_dir)
    if not videos:
        _print(f"未找到视频文件: {input_dir}", "yellow" if RICH_AVAILABLE else "")
        return []

    _print(f"发现 {len(videos)} 个视频，开始批处理。", "cyan" if RICH_AVAILABLE else "")
    outputs = []
    for idx, video_path in enumerate(videos, 1):
        _print(f"\n[{idx}/{len(videos)}]")
        output_video, _ = process_video_with_progress(
            video_path,
            output_dir,
            frame_skip=frame_skip,
            pipeline_overrides=pipeline_overrides,
        )
        if output_video:
            outputs.append(output_video)
    return outputs


def process_existing_videos(output_dir=None):
    """为已有 output 子目录重生成图表。"""
    output_dir = output_dir or OUTPUT_DIR
    if not os.path.exists(output_dir):
        _print(f"输出目录不存在: {output_dir}", "yellow" if RICH_AVAILABLE else "")
        return 0

    video_dirs = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ]
    if not video_dirs:
        _print("未找到已处理视频目录。", "yellow" if RICH_AVAILABLE else "")
        return 0

    total = 0
    for video_dir in video_dirs:
        chart_count = len(generate_all_charts(video_dir))
        total += chart_count
        _print(f"重建图表: {os.path.basename(video_dir)} -> {chart_count}")

    _print(f"图表重建完成，共生成 {total} 个文件。", "green" if RICH_AVAILABLE else "")
    return total


def launch_web_server(host: str = "127.0.0.1", port: int = 5000):
    """自动启动并打开 Web 可视化页面。"""
    try:
        from app import run_web_server
    except Exception as exc:
        _print(f"无法启动 Web 服务: {exc}", "bold red" if RICH_AVAILABLE else "")
        return 1

    _print(f"\n分析完成，正在启动 Web 可视化: http://{host}:{port}", "bold cyan" if RICH_AVAILABLE else "")
    try:
        run_web_server(host=host, port=port, open_browser=True, quiet=True)
        return 0
    except KeyboardInterrupt:
        _print("Web 服务已停止。")
        return 0
    except Exception as exc:
        _print(f"Web 服务启动失败: {exc}", "bold red" if RICH_AVAILABLE else "")
        return 1


def main(args):
    t_start = time.time()

    input_dir = args.input_dir or INPUT_DIR
    output_dir = args.output_dir or OUTPUT_DIR
    frame_skip = args.frame_skip or VIDEO_FRAME_SKIP
    pipeline_overrides = _build_pipeline_overrides(args)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print_runtime_overview()
    if bool(getattr(args, "check_panns", False)):
        return run_panns_self_check()
    if bool(getattr(args, "launch_validation_web", False)):
        try:
            video_dir = _resolve_video_dir_for_validation(args, output_dir=output_dir)
        except Exception as exc:
            _print(f"[validation_web] 无法定位视频输出目录: {exc}", "bold red" if RICH_AVAILABLE else "")
            return 1

        rater = str(getattr(args, "validation_rater", "A") or "A").strip().upper()
        if rater == "B":
            csv_override = args.validation_rater_b_csv
        else:
            csv_override = args.validation_rater_a_csv
            rater = "A"
        return run_validation_web_app(video_dir=video_dir, rater=rater, csv_path=csv_override)
    if bool(getattr(args, "launch_adjudication_web", False)):
        try:
            video_dir = _resolve_video_dir_for_validation(args, output_dir=output_dir)
        except Exception as exc:
            _print(f"[adjudication_web] 无法定位视频输出目录: {exc}", "bold red" if RICH_AVAILABLE else "")
            return 1
        return run_adjudication_web_app(
            video_dir=video_dir,
            adjudication_pack_csv=getattr(args, "adjudication_pack_csv", None),
        )
    if any(
        [
            bool(getattr(args, "enable_validation_pack", False)),
            bool(getattr(args, "compute_validation_reliability", False)),
            bool(getattr(args, "finalize_validation_labels", False)),
            bool(getattr(args, "build_adjudication_pack", False)),
            bool(getattr(args, "finalize_adjudicated_labels", False)),
        ]
    ):
        try:
            video_dir = _resolve_video_dir_for_validation(args, output_dir=output_dir)
        except Exception as exc:
            _print(f"[validation] 无法定位视频输出目录: {exc}", "bold red" if RICH_AVAILABLE else "")
            return 1

        code = 0
        if bool(getattr(args, "enable_validation_pack", False)):
            code = max(
                code,
                run_validation_pack(video_dir=video_dir),
            )
        if bool(getattr(args, "compute_validation_reliability", False)):
            code = max(
                code,
                run_validation_reliability(
                    video_dir=video_dir,
                    rater_a_csv=args.validation_rater_a_csv,
                    rater_b_csv=args.validation_rater_b_csv,
                    admin_manifest_csv=args.validation_admin_csv,
                ),
            )
        if bool(getattr(args, "finalize_validation_labels", False)):
            code = max(
                code,
                run_validation_finalize(
                    video_dir=video_dir,
                    rater_a_csv=args.validation_rater_a_csv,
                    rater_b_csv=args.validation_rater_b_csv,
                    admin_manifest_csv=args.validation_admin_csv,
                ),
            )
        if bool(getattr(args, "build_adjudication_pack", False)):
            code = max(
                code,
                run_adjudication_pack(
                    video_dir=video_dir,
                    rater_a_csv=args.validation_rater_a_csv,
                    rater_b_csv=args.validation_rater_b_csv,
                    admin_manifest_csv=args.validation_admin_csv,
                    reliability_report_json=args.reliability_report_json,
                ),
            )
        if bool(getattr(args, "finalize_adjudicated_labels", False)):
            code = max(
                code,
                run_adjudication_finalize(
                    video_dir=video_dir,
                    adjudication_pack_csv=args.adjudication_pack_csv,
                    baseline_final_csv=args.baseline_final_labels_csv,
                    rater_a_csv=args.validation_rater_a_csv,
                    rater_b_csv=args.validation_rater_b_csv,
                    admin_manifest_csv=args.validation_admin_csv,
                ),
            )
        return int(code)
    if bool(getattr(args, "run_step7_fusion_eval", False)):
        try:
            video_dir = _resolve_video_dir_for_validation(args, output_dir=output_dir)
        except Exception as exc:
            _print(f"[step7] 无法定位视频输出目录: {exc}", "bold red" if RICH_AVAILABLE else "")
            return 1
        return run_step7_fusion_eval_task(
            video_dir=video_dir,
            feature_csv=getattr(args, "feature_csv", None),
            labels_csv=getattr(args, "labels_csv", None),
            step7_outdir=getattr(args, "step7_outdir", None),
            seed=getattr(args, "step7_seed", None),
            smoke_test=bool(getattr(args, "step7_smoke_test", False)),
            clean_outdir=bool(getattr(args, "step7_clean_outdir", False)),
            show_progress=bool(getattr(args, "step7_show_progress", False)),
        )
    if bool(getattr(args, "run_step75_refined_eval", False)):
        try:
            video_dir = _resolve_video_dir_for_validation(args, output_dir=output_dir)
        except Exception as exc:
            _print(f"[step75] 无法定位视频输出目录: {exc}", "bold red" if RICH_AVAILABLE else "")
            return 1
        return run_step75_refined_eval_task(
            video_dir=video_dir,
            feature_csv=getattr(args, "feature_csv", None),
            labels_csv=getattr(args, "labels_csv", None),
            step75_outdir=getattr(args, "step75_outdir", None),
            seed=getattr(args, "step75_seed", None),
            smoke_test=bool(getattr(args, "step75_smoke_test", False)),
            reuse_step7_splits=getattr(args, "reuse_step7_splits", None),
        )
    if bool(getattr(args, "run_step8_design_mapping", False)):
        try:
            video_dir = _resolve_video_dir_for_validation(args, output_dir=output_dir)
        except Exception as exc:
            _print(f"[step8] 无法定位视频输出目录: {exc}", "bold red" if RICH_AVAILABLE else "")
            return 1
        return run_step8_design_mapping_task(
            video_dir=video_dir,
            step8_outdir=getattr(args, "step8_outdir", None),
            top_n=getattr(args, "step8_top_n", None),
            smoke_test=bool(getattr(args, "step8_smoke_test", False)),
        )
    if bool(getattr(args, "run_relationship_analysis", False)):
        try:
            video_dir = _resolve_video_dir_for_validation(args, output_dir=output_dir)
        except Exception as exc:
            _print(f"[relationship] 无法定位视频输出目录: {exc}", "bold red" if RICH_AVAILABLE else "")
            return 1
        return run_relationship_analysis_task(
            video_dir=video_dir,
            relationship_outdir=getattr(args, "relationship_outdir", None),
        )
    if bool(getattr(args, "run_proof_package", False)):
        try:
            video_dir = _resolve_video_dir_for_validation(args, output_dir=output_dir)
        except Exception as exc:
            _print(f"[proof] 无法定位视频输出目录: {exc}", "bold red" if RICH_AVAILABLE else "")
            return 1
        return run_proof_package_task(
            video_dir=video_dir,
            proof_outdir=getattr(args, "proof_outdir", None),
        )
    if bool(getattr(args, "run_group_confirmatory_relationship", False)):
        try:
            video_dir = _resolve_video_dir_for_validation(args, output_dir=output_dir)
        except Exception as exc:
            _print(f"[group_confirmatory] 无法定位视频输出目录: {exc}", "bold red" if RICH_AVAILABLE else "")
            return 1
        return run_group_confirmatory_relationship_task(
            video_dir=video_dir,
            group_confirmatory_outdir=getattr(args, "group_confirmatory_outdir", None),
        )
    if bool(getattr(args, "run_paper_figures", False)):
        try:
            video_dir = _resolve_video_dir_for_validation(args, output_dir=output_dir)
        except Exception as exc:
            _print(f"[paper_figures] 无法定位视频输出目录: {exc}", "bold red" if RICH_AVAILABLE else "")
            return 1
        return run_paper_figures_task(
            video_dir=video_dir,
            paper_figures_outdir=getattr(args, "paper_figures_outdir", None),
        )
    if bool(getattr(args, "run_deliverable_layer", False)):
        try:
            video_dir = _resolve_video_dir_for_validation(args, output_dir=output_dir)
        except Exception as exc:
            _print(f"[deliverable] 无法定位视频输出目录: {exc}", "bold red" if RICH_AVAILABLE else "")
            return 1
        return run_deliverable_layer_task(
            video_dir=video_dir,
            deliverable_top_k=getattr(args, "deliverable_top_k", None),
            deliverable_top_percent=getattr(args, "deliverable_top_percent", None),
            deliverable_priority_threshold=getattr(args, "deliverable_priority_threshold", None),
            deliverable_max_gap_seconds=getattr(args, "deliverable_max_gap_seconds", 5.0),
            deliverable_export_cards=bool(getattr(args, "deliverable_export_cards", False)),
            deliverable_use_glm=bool(getattr(args, "deliverable_use_glm", False)),
            deliverable_render_html=bool(getattr(args, "deliverable_render_html", False)),
            deliverable_render_pdf=bool(getattr(args, "deliverable_render_pdf", False)),
        )
    has_result = False
    if args.post_only and not args.from_existing_output:
        _print(
            "[post_only] 未提供 --from_existing_output；为兼容将继续 legacy 主流程。",
            "yellow" if RICH_AVAILABLE else "",
        )

    if args.from_existing_output:
        try:
            video_dir, _, _ = process_from_existing_output(
                from_existing_output=args.from_existing_output,
                output_dir=output_dir,
                frame_skip=frame_skip,
                pipeline_overrides=pipeline_overrides,
                post_only=bool(args.post_only),
                resume_missing_only=bool(args.resume_missing_only),
                video_name=args.video_name,
                video_path_hint=args.video_path,
            )
            has_result = os.path.isdir(video_dir)
        except Exception as exc:
            _print(f"从既有输出继续运行失败: {exc}", "bold red" if RICH_AVAILABLE else "")
            return 1
    elif args.regen_charts:
        count = process_existing_videos(output_dir)
        has_result = count >= 0
    elif args.video_path:
        if not os.path.exists(args.video_path):
            _print(f"视频不存在: {args.video_path}", "bold red" if RICH_AVAILABLE else "")
            return 1
        output_video, _ = process_video_with_progress(
            args.video_path,
            output_dir,
            frame_skip=frame_skip,
            pipeline_overrides=pipeline_overrides,
        )
        has_result = bool(output_video)
    else:
        outputs = process_all_videos(
            input_dir,
            output_dir,
            frame_skip=frame_skip,
            pipeline_overrides=pipeline_overrides,
        )
        has_result = len(outputs) > 0

    t_total = time.time() - t_start
    _print(f"\n任务结束，总耗时: {t_total:.2f} 秒", "bold green" if RICH_AVAILABLE else "")

    if args.no_web:
        return 0

    return launch_web_server(host=args.web_host, port=args.web_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insta360 全景视频分析（精简版）")
    parser.add_argument("-i", "--input_dir", help="输入视频目录")
    parser.add_argument("-o", "--output_dir", help="输出目录")
    parser.add_argument("-v", "--video_path", help="单个视频路径")
    parser.add_argument("-s", "--frame_skip", type=int, help="帧间隔")
    parser.add_argument("--regen_charts", action="store_true", help="仅重生成图表")
    parser.add_argument(
        "--from_existing_output",
        help="从已有 output/<video_name> 目录继续运行后处理（跳过 legacy 帧级重算）",
    )
    parser.add_argument(
        "--post_only",
        action="store_true",
        help="仅运行新多模态后处理阶段（segment/visual/geo_sync/soundscape/fusion/agents/design/deliverable/gis_export/web_sync）",
    )
    parser.add_argument(
        "--resume_missing_only",
        action="store_true",
        help="仅补生成缺失产物；已存在且有效的阶段输出将自动跳过",
    )
    parser.add_argument(
        "--video_name",
        help="当 --from_existing_output 指向 output 根目录时，用于指定子目录名",
    )
    parser.add_argument("--no_web", action="store_true", help="处理完成后不自动启动 Web")
    parser.add_argument("--web_host", default="127.0.0.1", help="自动启动 Web 的监听地址")
    parser.add_argument("--web_port", type=int, default=5000, help="自动启动 Web 的端口")
    parser.add_argument(
        "--check_panns",
        action="store_true",
        help="仅检查本地 PANNs 资源与初始化（不运行视频处理流程）",
    )
    parser.add_argument(
        "--launch_validation_web",
        action="store_true",
        help="Step-5: 启动本地盲标注 Web（Streamlit，直接写回 rater CSV）",
    )
    parser.add_argument(
        "--validation_rater",
        choices=["A", "B", "a", "b"],
        default="A",
        help="Step-5: 启动盲标注 Web 时选择评审者（A 或 B）",
    )
    parser.add_argument(
        "--launch_adjudication_web",
        action="store_true",
        help="Step-5.5: 启动争议裁决 Web（Streamlit，读写 adjudication_pack.csv）",
    )
    parser.add_argument(
        "--enable_validation_pack",
        action="store_true",
        help="Step-5: 生成双评审盲标注包（60 unique + 每位8隐藏重复）",
    )
    parser.add_argument(
        "--compute_validation_reliability",
        action="store_true",
        help="Step-5: 在两位评审完成后计算一致性（ICC/Spearman/MAE/Kappa）",
    )
    parser.add_argument(
        "--finalize_validation_labels",
        action="store_true",
        help="Step-5: 汇总两位评审标签为最终 segment 级标签文件",
    )
    parser.add_argument(
        "--build_adjudication_pack",
        action="store_true",
        help="Step-5.5: 构建争议聚焦裁决包（仅筛选分歧条目）",
    )
    parser.add_argument(
        "--finalize_adjudicated_labels",
        action="store_true",
        help="Step-5.5: 用裁决结果生成 final_annotation_labels_adjudicated.csv",
    )
    parser.add_argument(
        "--validation_rater_a_csv",
        help="Step-5: 指定评审A已填写CSV路径（默认 validation/rater_A_annotation_pack.csv）",
    )
    parser.add_argument(
        "--validation_rater_b_csv",
        help="Step-5: 指定评审B已填写CSV路径（默认 validation/rater_B_annotation_pack.csv）",
    )
    parser.add_argument(
        "--validation_admin_csv",
        help="Step-5: 指定管理员清单CSV路径（默认 validation/sample_manifest_admin.csv）",
    )
    parser.add_argument(
        "--reliability_report_json",
        help="Step-5.5: 指定 reliability_report.json 路径（默认 validation/reliability_report.json）",
    )
    parser.add_argument(
        "--adjudication_pack_csv",
        help="Step-5.5: 指定 adjudication_pack.csv 路径（默认 validation/adjudication_pack.csv）",
    )
    parser.add_argument(
        "--baseline_final_labels_csv",
        help="Step-5.5: 指定基线 final_annotation_labels.csv 路径（默认 validation/final_annotation_labels.csv）",
    )
    parser.add_argument(
        "--run_step7_fusion_eval",
        action="store_true",
        help="Step-7: 使用现有 model_feature_table + 裁决标签运行融合建模评估（输出到 fusion_eval）",
    )
    parser.add_argument(
        "--labels_csv",
        help="Step-7: 标签CSV路径（默认优先 validation/final_annotation_labels_adjudicated.csv）",
    )
    parser.add_argument(
        "--feature_csv",
        help="Step-7: 特征CSV路径（默认 fusion/model_feature_table.csv）",
    )
    parser.add_argument(
        "--step7_outdir",
        help="Step-7: 输出目录（默认 output/<video>/fusion_eval）",
    )
    parser.add_argument(
        "--step7_seed",
        type=int,
        default=STEP7_SEED,
        help=f"Step-7: 随机种子（默认 {STEP7_SEED}）",
    )
    parser.add_argument(
        "--step7_smoke_test",
        action="store_true",
        help="Step-7: 烟雾测试模式（减少重复次数以快速验证流程）",
    )
    parser.add_argument(
        "--step7_clean_outdir",
        action="store_true",
        help="Step-7: 开始前清理 output/<video>/fusion_eval（仅该目录）",
    )
    parser.add_argument(
        "--step7_show_progress",
        action="store_true",
        help="Step-7: 显示详细进度并持续写出 fusion_eval/step7_progress.json",
    )
    parser.add_argument(
        "--run_step75_refined_eval",
        action="store_true",
        help="Step-7.5: 在 fusion_eval_refined/ 运行泄漏安全筛选+精细化融合评估（不覆盖 Step-7）",
    )
    parser.add_argument(
        "--step75_outdir",
        help="Step-7.5: 输出目录（默认 output/<video>/fusion_eval_refined）",
    )
    parser.add_argument(
        "--step75_seed",
        type=int,
        default=STEP75_SEED,
        help=f"Step-7.5: 随机种子（默认 {STEP75_SEED}）",
    )
    parser.add_argument(
        "--step75_smoke_test",
        action="store_true",
        help="Step-7.5: 烟雾测试模式（减少重复次数并加速验证）",
    )
    parser.add_argument(
        "--reuse_step7_splits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Step-7.5: 是否优先复用 Step-7 外层 CV 划分（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--run_step8_design_mapping",
        action="store_true",
        help="Step-8: 基于诊断 + Step-7.5 refined 证据生成设计映射、干预矩阵与受限编辑提示",
    )
    parser.add_argument(
        "--step8_outdir",
        help="Step-8: 输出目录（默认 output/<video>/design）",
    )
    parser.add_argument(
        "--step8_top_n",
        type=int,
        default=None,
        help=f"Step-8: 仅对优先级前 N 个片段生成完整设计计划；0 表示全部（默认 {STEP8_TOP_N}）",
    )
    parser.add_argument(
        "--step8_smoke_test",
        action="store_true",
        help="Step-8: 烟雾测试模式（若未指定 --step8_top_n，则仅处理少量高优先级片段）",
    )
    parser.add_argument(
        "--run_relationship_analysis",
        action="store_true",
        help="Research: 运行声景-视觉关系分析（relationship/）",
    )
    parser.add_argument(
        "--relationship_outdir",
        help="Research: relationship analysis 输出目录（默认 output/<video>/relationship）",
    )
    parser.add_argument(
        "--run_proof_package",
        action="store_true",
        help="Research: 运行融合优于单模态的严格证明包（proof/）",
    )
    parser.add_argument(
        "--proof_outdir",
        help="Research: proof package 输出目录（默认 output/<video>/proof）",
    )
    parser.add_argument(
        "--run_group_confirmatory_relationship",
        action="store_true",
        help="Research: 运行 group-level confirmatory relationship 分析（relationship/group_confirmatory/）",
    )
    parser.add_argument(
        "--group_confirmatory_outdir",
        help="Research: group confirmatory 输出目录（默认 output/<video>/relationship/group_confirmatory）",
    )
    parser.add_argument(
        "--run_paper_figures",
        action="store_true",
        help="Research: 生成 relationship/proof 论文图包（paper_figures/）",
    )
    parser.add_argument(
        "--paper_figures_outdir",
        help="Research: paper figures 输出目录（默认 output/<video>/paper_figures）",
    )
    parser.add_argument(
        "--run_deliverable_layer",
        action="store_true",
        help="Deliverable: 在 Step-8 后生成最终问题路段交付包（deliverable/）",
    )
    parser.add_argument(
        "--deliverable_top_k",
        type=int,
        default=12,
        help="Deliverable: 按优先级取前 K 个 segment 参与 episode 合并（默认 12）",
    )
    parser.add_argument(
        "--deliverable_top_percent",
        type=float,
        default=None,
        help="Deliverable: 按优先级取前百分比 segment（0-1）参与 episode 合并",
    )
    parser.add_argument(
        "--deliverable_priority_threshold",
        type=float,
        default=None,
        help="Deliverable: 仅保留 priority_score 不低于该阈值的 segment",
    )
    parser.add_argument(
        "--deliverable_max_gap_seconds",
        type=float,
        default=5.0,
        help="Deliverable: 合并相邻高优先级 segment 为 episode 时允许的最大时间间隔（秒，默认 5.0）",
    )
    parser.add_argument(
        "--deliverable_export_cards",
        action="store_true",
        help="Deliverable: 导出每个 episode 的静态 PNG 卡片",
    )
    parser.add_argument(
        "--deliverable_use_glm",
        action="store_true",
        help="Deliverable: 若智谱 GLM 可用，则对模板 prompt 做受控润色；失败时自动回退",
    )
    parser.add_argument(
        "--deliverable_render_html",
        action="store_true",
        help="Deliverable: 渲染 HTML 总览页",
    )
    parser.add_argument(
        "--deliverable_render_pdf",
        action="store_true",
        help="Deliverable: 渲染 contact-sheet PDF 总览",
    )
    parser.add_argument(
        "--enable_segment_pipeline",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用分段扩展流水线（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--segment_seconds",
        type=float,
        default=None,
        help=f"分段长度（秒，默认 {SEGMENT_SECONDS}，沿用 src/config.py）",
    )
    parser.add_argument(
        "--segment_overlap",
        type=float,
        default=None,
        help=f"分段重叠（秒，默认 {SEGMENT_OVERLAP}，沿用 src/config.py）",
    )
    parser.add_argument(
        "--enable_visual_segment_summary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 manifest 对齐的 visual segment summary 阶段（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--enable_geo_sync",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 geo_sync 阶段并输出 frame/segment 坐标元数据（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--geo_sync_gps_csv",
        default=None,
        help="geo_sync 使用的 GPS CSV 路径（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--geo_sync_time_offset_seconds",
        type=float,
        default=None,
        help="geo_sync 的视频级时间偏移（秒），建议配合 sidecar 管理",
    )
    parser.add_argument(
        "--geo_sync_sidecar_path",
        default=None,
        help="geo_sync 的单视频 sidecar JSON 路径",
    )
    parser.add_argument(
        "--geo_sync_max_gap_warning_sec",
        type=float,
        default=None,
        help="geo_sync 有效重叠窗口的最大 GPS gap 阈值（秒）",
    )
    parser.add_argument(
        "--geo_sync_filename_tz_offset_hours",
        type=float,
        default=None,
        help="geo_sync 文件名解析起始时间时使用的时区偏移（小时）",
    )
    parser.add_argument(
        "--geo_sync_export_wgs84",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="geo_sync 是否额外导出近似 WGS84 字段（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--geo_sync_use_existing_segments",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="geo_sync 是否强制复用真实 segment_manifest.csv（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--geo_sync_align_to_analysis_frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="geo_sync 是否优先直接对齐 output/<video>/frames/ 中的真实分析帧（默认开启）",
    )
    parser.add_argument(
        "--geo_sync_frame_step",
        type=int,
        default=None,
        help="geo_sync 的 legacy 视频采样步长；仅在不对齐现有分析帧时作为回退参数使用",
    )
    parser.add_argument(
        "--enable_web_sync_export",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用视频-图表-地图同步 JSON 导出阶段（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--web_sync_prefer_wgs84",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="web 地图输出是否优先使用 derived WGS84 坐标（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--enable_gis_export",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 GIS 总表导出阶段（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--gis_export_prefer_wgs84",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="GIS 导出中的 display 坐标是否优先使用 derived WGS84（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--enable_soundscape",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 soundscape 扩展阶段（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--enable_fusion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 fusion 扩展阶段（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--enable_agents",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 agents 扩展阶段（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--enable_design",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 design 扩展阶段（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--enable_deliverable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 deliverable 问题 episode 封装阶段（默认沿用 src/config.py）",
    )
    parser.add_argument(
        "--export_debug_json",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用扩展阶段调试JSON输出（默认沿用 src/config.py）",
    )

    try:
        cli_args = parser.parse_args()
    except Exception:
        cli_args = parser.parse_args([])

    sys.exit(main(cli_args))
