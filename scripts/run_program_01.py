"""CLI wrapper for Program 01: analysis and spatial visualization."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]


def _read_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        return payload if isinstance(payload, dict) else {}


def _bool_arg(args: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    args.append(flag if bool(value) else f"--no-{flag[2:]}")


def build_main_args(cfg: Dict[str, Any], cli: argparse.Namespace) -> list[str]:
    data = dict(cfg)
    for key, value in vars(cli).items():
        if key in {"config", "dry_run", "launch_gui"}:
            continue
        if value not in (None, ""):
            data[key] = value

    out = ["--output_dir", str(data.get("output_dir") or "output")]
    if data.get("frame_skip"):
        out += ["--frame_skip", str(int(data["frame_skip"]))]
    if data.get("from_existing_output"):
        out += ["--from_existing_output", str(data["from_existing_output"])]
        if data.get("post_only"):
            out.append("--post_only")
        if data.get("resume_missing_only"):
            out.append("--resume_missing_only")
    elif data.get("input_video"):
        out += ["--video_path", str(data["input_video"])]
    elif data.get("input_dir"):
        out += ["--input_dir", str(data["input_dir"])]

    if bool(data.get("no_web", True)):
        out.append("--no_web")
    else:
        out += ["--web_port", str(int(data.get("web_port", 5000)))]

    flag_map = {
        "enable_segment_pipeline": "--enable_segment_pipeline",
        "enable_visual_segment_summary": "--enable_visual_segment_summary",
        "enable_soundscape": "--enable_soundscape",
        "enable_fusion": "--enable_fusion",
        "enable_geo_sync": "--enable_geo_sync",
        "enable_gis_export": "--enable_gis_export",
        "enable_web_sync_export": "--enable_web_sync_export",
    }
    for key, flag in flag_map.items():
        _bool_arg(out, flag, data.get(key))

    if data.get("segment_seconds") is not None:
        out += ["--segment_seconds", str(float(data["segment_seconds"]))]
    if data.get("segment_overlap") is not None:
        out += ["--segment_overlap", str(float(data["segment_overlap"]))]
    if data.get("gps_file"):
        out += ["--geo_sync_gps_csv", str(data["gps_file"])]
    if data.get("gps_time_offset_seconds") is not None:
        out += ["--geo_sync_time_offset_seconds", str(float(data["gps_time_offset_seconds"]))]
    _bool_arg(out, "--geo_sync_align_to_analysis_frames", data.get("geo_sync_align_to_analysis_frames"))
    _bool_arg(out, "--geo_sync_export_wgs84", data.get("geo_sync_export_wgs84"))
    _bool_arg(out, "--web_sync_prefer_wgs84", data.get("web_sync_prefer_wgs84"))
    _bool_arg(out, "--gis_export_prefer_wgs84", data.get("gis_export_prefer_wgs84"))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Program 01: base analysis and spatial visualization")
    parser.add_argument("--config", default="configs/default_program_01_config.yaml")
    parser.add_argument("--input_video", default=None)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--gps_file", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--frame_skip", type=int, default=None)
    parser.add_argument("--segment_seconds", type=float, default=None)
    parser.add_argument("--segment_overlap", type=float, default=None)
    parser.add_argument("--gps_time_offset_seconds", type=float, default=None)
    parser.add_argument("--from_existing_output", default=None)
    parser.add_argument("--post_only", action="store_true")
    parser.add_argument("--resume_missing_only", action="store_true")
    parser.add_argument("--enable_segment_pipeline", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_visual_segment_summary", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_soundscape", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_geo_sync", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_gis_export", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_web_sync_export", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--launch_gui", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    cli = parse_args(argv)
    if cli.launch_gui:
        from apps.program_01_analysis_visualization.gui import main as gui_main

        return int(gui_main([]) or 0)
    cfg = _read_config(cli.config)
    main_args = build_main_args(cfg, cli)
    command = [sys.executable, str(ROOT / "main.py"), *main_args]
    print(" ".join(subprocess.list2cmdline([part]) for part in command))
    if cli.dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
