


from __future__ import annotations

import argparse
import copy
import contextlib
import json
import os
import runpy
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass, fields
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Optional


APP_NAME = "StreetSmartEvaluator"
WINDOW_TITLE = "全景街景评估启动器 / Panoramic Street Evaluation Launcher"
INTERNAL_FLAG = "--run-pipeline-internal"
CONFIG_FILENAME = "launcher_config.json"
RUN_LOG_ENV = "STREETSMART_LAUNCHER_LOG_FILE"


def bi(cn: str, en: str) -> str:
    return f"{cn} / {en}"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_app_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_resource_path(relative_path: str | Path) -> Path:
    return get_app_dir() / Path(relative_path)


def get_output_dir(default: str | Path = "output") -> Path:
    path = Path(default)
    return path if path.is_absolute() else get_app_dir() / path


def find_ffmpeg(preferred: str = "") -> Optional[Path]:
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    candidates.extend([get_app_dir() / "ffmpeg.exe", get_app_dir() / "ffmpeg"])
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_model_file(models_folder: str = "", name: str = "yolo11m.pt") -> Optional[Path]:
    roots = [Path(models_folder)] if models_folder else []
    roots.extend([get_app_dir(), get_app_dir() / "models"])
    for root in roots:
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_html_outputs(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    html_files = [p for p in root.rglob("*.html") if p.is_file()]

    def score(path: Path) -> tuple[int, int, str]:
        text = path.as_posix().lower()
        preferred_names = ("index.html", "report.html", "problem_episode_cards.html")
        priority = 50
        if path.name.lower() in preferred_names:
            priority -= 30
        if "web_sync" in text or "deliverable" in text:
            priority -= 20
        if "interactive" in text:
            priority -= 5
        return (priority, len(path.parts), text)

    return sorted(html_files, key=score)


def quote_for_preview(args: Iterable[str]) -> str:
    return subprocess.list2cmdline([str(a) for a in args])


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class FileTextSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8", buffering=1)

    def write(self, text: str) -> int:
        self._handle.write(text)
        self._handle.flush()
        return len(text)

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


@dataclass
class AnalysisConfig:
    run_mode: str = "raw"
    preset: str = "standard"
    input_path: str = ""
    existing_output: str = ""
    gps_file: str = ""
    output_root: str = ""
    frame_skip: int = 20
    segment_seconds: float = 5.0
    segment_overlap: float = 2.5
    gps_time_offset: float = 25.0
    enable_segment_pipeline: bool = False
    enable_visual_segment_summary: bool = False
    enable_soundscape: bool = False
    enable_fusion: bool = False
    enable_agents: bool = False
    enable_design: bool = False
    enable_deliverable: bool = False
    enable_gis_export: bool = False
    enable_web_sync_export: bool = False
    enable_geo_sync: bool = False
    geo_sync_export_wgs84: bool = True
    geo_sync_align_to_analysis_frames: bool = True
    web_sync_prefer_wgs84: bool = False
    gis_export_prefer_wgs84: bool = True
    post_only: bool = False
    resume_missing_only: bool = False
    run_deliverable_layer: bool = False
    deliverable_use_glm: bool = False
    deliverable_export_cards: bool = False
    deliverable_render_html: bool = False
    deliverable_render_pdf: bool = False
    deliverable_top_k: int = 12
    deliverable_max_gap_seconds: float = 5.0
    no_web: bool = True
    web_port: int = 5000
    ffmpeg_path: str = ""
    models_folder: str = ""
    api_env_file: str = ""

    @classmethod
    def defaults(cls) -> "AnalysisConfig":
        cfg = cls()
        app_dir = get_app_dir()
        cfg.output_root = str(app_dir / "output")
        cfg.ffmpeg_path = str(find_ffmpeg() or app_dir / "ffmpeg.exe")
        cfg.models_folder = str(app_dir / "models")
        env = app_dir / "apikey.env"
        cfg.api_env_file = str(env) if env.exists() else ""
        return cfg


def config_from_dict(data: dict[str, Any]) -> AnalysisConfig:
    cfg = AnalysisConfig.defaults()
    valid = {f.name for f in fields(AnalysisConfig)}
    for key, value in data.items():
        if key in valid:
            setattr(cfg, key, value)
    return cfg


def default_config_path() -> Path:
    return get_app_dir() / CONFIG_FILENAME


def validate_config(cfg: AnalysisConfig) -> list[str]:
    errors: list[str] = []
    output_root = Path(cfg.output_root or get_output_dir())
    if cfg.run_mode == "existing":
        if not cfg.existing_output:
            errors.append("请选择 Existing Output Folder。")
        elif not Path(cfg.existing_output).is_dir():
            errors.append(f"Existing Output Folder 不存在: {cfg.existing_output}")
    else:
        if not cfg.input_path:
            errors.append("请选择输入视频文件或输入文件夹。")
        elif not Path(cfg.input_path).exists():
            errors.append(f"输入路径不存在: {cfg.input_path}")

    if cfg.gps_file:
        gps = Path(cfg.gps_file)
        if not gps.is_file():
            errors.append(f"GPS 文件不存在: {cfg.gps_file}")
        elif gps.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            errors.append("GPS 文件只支持 csv / xlsx / xls。")
    elif cfg.enable_geo_sync or cfg.enable_gis_export or cfg.enable_web_sync_export:
        errors.append("未选择 GPS 文件，不能启用 geo sync / GIS export / web sync。")

    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        errors.append(f"输出目录无法创建或无权限: {output_root} ({exc})")

    if not find_ffmpeg(cfg.ffmpeg_path):
        errors.append("未找到 ffmpeg.exe。请在程序目录放置 ffmpeg.exe，或在 Resource Settings 中指定。")

    if not find_model_file(cfg.models_folder):
        errors.append("未找到 yolo11m.pt。请放在程序目录或 models 文件夹中。")

    web_dir = get_resource_path("web")
    if not web_dir.is_dir():
        errors.append(f"web 资源目录不存在: {web_dir}")

    api_env = Path(cfg.api_env_file) if cfg.api_env_file else get_app_dir() / "apikey.env"
    if (cfg.deliverable_use_glm or cfg.enable_agents) and not api_env.is_file():
        errors.append("启用了 GLM/Agents，但没有找到 apikey.env。请选择 API Key Env File 或放到程序目录。")

    return errors


def _converted_gps_csv_path(output_root: Path, source: Path) -> Path:
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source.stem)
    return output_root / "_launcher_cache" / f"{safe_stem}_gps.csv"


def prepare_runtime_config(cfg: AnalysisConfig) -> AnalysisConfig:
    """Return a copy with runtime-only adaptations, such as xlsx -> csv GPS."""
    runtime = copy.deepcopy(cfg)
    if not runtime.gps_file:
        return runtime
    source = Path(runtime.gps_file)
    if source.suffix.lower() not in {".xlsx", ".xls"}:
        return runtime
    try:
        import pandas as pd

        output_root = Path(runtime.output_root or get_output_dir())
        target = _converted_gps_csv_path(output_root, source)
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.read_excel(source).to_csv(target, index=False, encoding="utf-8-sig")
        runtime.gps_file = str(target)
    except Exception as exc:
        raise RuntimeError(f"GPS Excel 转 CSV 失败: {exc}") from exc
    return runtime


def build_args(cfg: AnalysisConfig) -> list[str]:
    args: list[str] = []
    output_root = str(Path(cfg.output_root or get_output_dir()))
    args.extend(["--output_dir", output_root])
    if cfg.frame_skip:
        args.extend(["--frame_skip", str(int(cfg.frame_skip))])

    if cfg.run_mode == "existing":
        args.extend(["--from_existing_output", str(Path(cfg.existing_output))])
        if cfg.post_only:
            args.append("--post_only")
        if cfg.resume_missing_only:
            args.append("--resume_missing_only")
    else:
        input_path = Path(cfg.input_path)
        if input_path.is_dir():
            args.extend(["--input_dir", str(input_path)])
        else:
            args.extend(["--video_path", str(input_path)])

    if cfg.no_web:
        args.append("--no_web")
    else:
        args.extend(["--web_port", str(int(cfg.web_port))])

    optional_flags = {
        "--enable_segment_pipeline": cfg.enable_segment_pipeline,
        "--enable_visual_segment_summary": cfg.enable_visual_segment_summary,
        "--enable_soundscape": cfg.enable_soundscape,
        "--enable_fusion": cfg.enable_fusion,
        "--enable_agents": cfg.enable_agents,
        "--enable_design": cfg.enable_design,
        "--enable_deliverable": cfg.enable_deliverable,
        "--enable_geo_sync": cfg.enable_geo_sync,
        "--enable_gis_export": cfg.enable_gis_export,
        "--enable_web_sync_export": cfg.enable_web_sync_export,
    }
    for flag, enabled in optional_flags.items():
        if enabled:
            args.append(flag)

    if cfg.enable_segment_pipeline:
        args.extend(["--segment_seconds", str(float(cfg.segment_seconds))])
        args.extend(["--segment_overlap", str(float(cfg.segment_overlap))])

    if cfg.gps_file:
        args.extend(["--geo_sync_gps_csv", str(Path(cfg.gps_file))])
        args.extend(["--geo_sync_time_offset_seconds", str(float(cfg.gps_time_offset))])
        args.append("--geo_sync_export_wgs84" if cfg.geo_sync_export_wgs84 else "--no-geo_sync_export_wgs84")
        args.append(
            "--geo_sync_align_to_analysis_frames"
            if cfg.geo_sync_align_to_analysis_frames
            else "--no-geo_sync_align_to_analysis_frames"
        )
    if cfg.enable_web_sync_export and cfg.web_sync_prefer_wgs84:
        args.append("--web_sync_prefer_wgs84")
    if cfg.enable_gis_export:
        args.append("--gis_export_prefer_wgs84" if cfg.gis_export_prefer_wgs84 else "--no-gis_export_prefer_wgs84")

    if cfg.run_deliverable_layer:
        args.append("--run_deliverable_layer")
        args.extend(["--deliverable_top_k", str(int(cfg.deliverable_top_k))])
        args.extend(["--deliverable_max_gap_seconds", str(float(cfg.deliverable_max_gap_seconds))])
        if cfg.deliverable_use_glm:
            args.append("--deliverable_use_glm")
        if cfg.deliverable_export_cards:
            args.append("--deliverable_export_cards")
        if cfg.deliverable_render_html:
            args.append("--deliverable_render_html")
        if cfg.deliverable_render_pdf:
            args.append("--deliverable_render_pdf")

    return args


class LocalHttpServer:
    def __init__(self) -> None:
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.root: Optional[Path] = None
        self.port: Optional[int] = None

    def close(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        self._server = None
        self._thread = None
        self.root = None
        self.port = None

    def start(self, root: Path, preferred_port: int = 8765) -> int:
        self.close()
        root = root.resolve()
        for port in range(preferred_port, preferred_port + 50):
            if _port_available(port):
                handler = lambda *a, **kw: SimpleHTTPRequestHandler(*a, directory=str(root), **kw)
                self._server = ThreadingHTTPServer(("127.0.0.1", port), handler)
                self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
                self._thread.start()
                self.root = root
                self.port = port
                return port
        raise RuntimeError("无法找到可用的本地 HTTP 端口。")


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def run_pipeline_internal(argv: list[str]) -> int:
    app_dir = get_app_dir()
    os.chdir(app_dir)
    sys.path.insert(0, str(app_dir))

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        os.environ["PATH"] = str(ffmpeg.parent) + os.pathsep + os.environ.get("PATH", "")

    env_file = app_dir / "apikey.env"
    for key, value in load_env_file(env_file).items():
        os.environ.setdefault(key, value)

    sys.argv = ["main.py", *argv]
    log_file = os.environ.get(RUN_LOG_ENV, "").strip()
    try:
        if log_file:
            sink = FileTextSink(Path(log_file))
            try:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    runpy.run_module("main", run_name="__main__", alter_sys=True)
            finally:
                sink.close()
        else:
            runpy.run_module("main", run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def launch_gui(smoke_test: bool = False) -> int:
    try:
        from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
        from PySide6.QtGui import QTextCursor
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QInputDialog,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QPlainTextEdit,
            QScrollArea,
            QSpinBox,
            QDoubleSpinBox,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:
        print("PySide6 未安装。请先运行: pip install -r requirements_gui.txt")
        print(exc)
        return 1

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.process: Optional[QProcess] = None
            self.status = "Idle"
            self.log_lines: list[str] = []
            self.run_log_path: Optional[Path] = None
            self.run_log_pos = 0
            self.last_run_dir: Optional[Path] = None
            self.http_server = LocalHttpServer()
            self.setWindowTitle(WINDOW_TITLE)
            self.resize(1180, 820)
            self._apply_style()
            self._build_ui()
            self.log_tail_timer = QTimer(self)
            self.log_tail_timer.setInterval(500)
            self.log_tail_timer.timeout.connect(self.tail_run_log)
            self.load_last_config()
            self.update_preview()

        def _apply_style(self) -> None:
            self.setStyleSheet(
                """
                QMainWindow {
                    background: #eef1f5;
                    color: #17202c;
                    font-family: "Microsoft YaHei UI", "Segoe UI", Arial, sans-serif;
                    font-size: 10.5pt;
                }
                QScrollArea {
                    border: 0;
                    background: transparent;
                }
                QWidget#PageBody {
                    background: #eef1f5;
                }
                QWidget#HeroHeader {
                    background: #101820;
                    border: 1px solid #1f2f3c;
                    border-radius: 8px;
                }
                QLabel#HeroTitle {
                    color: #f6f8fb;
                    font-size: 22pt;
                    font-weight: 700;
                    letter-spacing: 0px;
                }
                QLabel#HeroSubtitle {
                    color: #b8c4d2;
                    font-size: 10.5pt;
                }
                QLabel#Badge {
                    color: #16301f;
                    background: #c9f2d5;
                    border: 1px solid #8bd8a2;
                    border-radius: 6px;
                    padding: 6px 10px;
                    font-weight: 600;
                }
                QGroupBox {
                    background: #ffffff;
                    border: 1px solid #d7dde6;
                    border-radius: 8px;
                    margin-top: 18px;
                    padding: 16px 14px 14px 14px;
                    font-weight: 700;
                    color: #1f2a37;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 14px;
                    padding: 0 8px;
                    color: #263241;
                    background: #ffffff;
                }
                QLabel {
                    color: #2b3645;
                }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                    background: #fbfcfe;
                    border: 1px solid #cfd7e3;
                    border-radius: 6px;
                    padding: 7px 9px;
                    selection-background-color: #2f6fed;
                }
                QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
                    border: 1px solid #2f6fed;
                    background: #ffffff;
                }
                QPlainTextEdit {
                    font-family: "Cascadia Mono", Consolas, monospace;
                    font-size: 9.5pt;
                }
                QCheckBox {
                    spacing: 8px;
                    color: #253040;
                    padding: 4px 2px;
                }
                QPushButton {
                    background: #f7f9fc;
                    border: 1px solid #cbd4df;
                    border-radius: 6px;
                    padding: 8px 12px;
                    color: #1f2a37;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #edf3ff;
                    border-color: #8fb3f5;
                }
                QPushButton:pressed {
                    background: #dfeafe;
                }
                QPushButton:disabled {
                    color: #8b96a5;
                    background: #eef1f5;
                    border-color: #dde3eb;
                }
                QPushButton#PrimaryButton {
                    background: #1f6feb;
                    color: #ffffff;
                    border-color: #1d5fca;
                }
                QPushButton#PrimaryButton:hover {
                    background: #2f7cf0;
                }
                QPushButton#DangerButton {
                    background: #fff4f3;
                    color: #a92822;
                    border-color: #efb4af;
                }
                QLabel#StatusBadge {
                    background: #edf3ff;
                    color: #1f4fa3;
                    border: 1px solid #b7cef8;
                    border-radius: 6px;
                    padding: 8px 12px;
                    font-weight: 700;
                }
                """
            )

        def _build_ui(self) -> None:
            root = QWidget()
            outer = QVBoxLayout(root)
            outer.setContentsMargins(18, 18, 18, 18)
            outer.setSpacing(14)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            body = QWidget()
            body.setObjectName("PageBody")
            layout = QVBoxLayout(body)
            layout.setContentsMargins(2, 2, 2, 18)
            layout.setSpacing(12)
            scroll.setWidget(body)
            outer.addWidget(scroll)
            self.setCentralWidget(root)

            header = QWidget()
            header.setObjectName("HeroHeader")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(22, 18, 22, 18)
            header_text = QVBoxLayout()
            title = QLabel("全景街景智能评估 / Panoramic Street Evaluation")
            title.setObjectName("HeroTitle")
            subtitle = QLabel("桌面启动器，封装现有 main.py 流程 / Desktop launcher wrapping the existing main.py pipeline")
            subtitle.setObjectName("HeroSubtitle")
            header_text.addWidget(title)
            header_text.addWidget(subtitle)
            badge = QLabel("中文 + English")
            badge.setObjectName("Badge")
            header_layout.addLayout(header_text, 1)
            header_layout.addWidget(badge, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(header)

            input_group = QGroupBox(bi("输入与输出", "Input & Output"))
            input_form = QGridLayout(input_group)
            input_form.setHorizontalSpacing(12)
            input_form.setVerticalSpacing(10)
            self.run_mode_combo = QComboBox()
            self.run_mode_combo.addItem(bi("原始视频 / 输入文件夹", "Raw Video / Input Folder"), "raw")
            self.run_mode_combo.addItem(bi("已有输出目录", "Existing Output Folder"), "existing")
            self.input_edit = QLineEdit()
            self.existing_edit = QLineEdit()
            self.gps_edit = QLineEdit()
            self.output_edit = QLineEdit()
            self._add_path_row(input_form, 0, bi("运行模式", "Run Mode"), self.run_mode_combo)
            self._add_path_row(input_form, 1, bi("输入视频 / 文件夹", "Input Video / Folder"), self.input_edit, self.browse_input)
            self._add_path_row(input_form, 2, bi("已有输出目录", "Existing Output Folder"), self.existing_edit, self.browse_existing)
            self._add_path_row(input_form, 3, bi("GPS 文件（可选）", "GPS File Optional"), self.gps_edit, self.browse_gps)
            self._add_path_row(input_form, 4, bi("输出根目录", "Output Root Folder"), self.output_edit, self.browse_output)
            layout.addWidget(input_group)

            options_group = QGroupBox(bi("分析选项", "Analysis Options"))
            options = QGridLayout(options_group)
            options.setHorizontalSpacing(26)
            options.setVerticalSpacing(10)
            self.preset_combo = QComboBox()
            presets = [
                (bi("快速测试模式", "Quick Test"), "quick"),
                (bi("标准分析模式", "Standard Analysis"), "standard"),
                (bi("从已有结果生成交付物", "Deliverables from Existing Output"), "deliverable_existing"),
                (bi("带 GPS 的 Web 展示", "GPS Web Presentation"), "gps_web"),
                (bi("高质量完整模式", "High Quality Full Run"), "full"),
            ]
            for label, value in presets:
                self.preset_combo.addItem(label, value)
            self.frame_skip_spin = QSpinBox()
            self.frame_skip_spin.setRange(1, 100000)
            self.segment_seconds_spin = QDoubleSpinBox()
            self.segment_seconds_spin.setRange(0.1, 3600.0)
            self.segment_seconds_spin.setDecimals(2)
            self.segment_overlap_spin = QDoubleSpinBox()
            self.segment_overlap_spin.setRange(0.0, 3600.0)
            self.segment_overlap_spin.setDecimals(2)
            self.gps_offset_spin = QDoubleSpinBox()
            self.gps_offset_spin.setRange(-86400.0, 86400.0)
            self.gps_offset_spin.setDecimals(2)
            self.web_port_spin = QSpinBox()
            self.web_port_spin.setRange(1, 65535)

            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(10)
            form.addRow(bi("预设", "Preset"), self.preset_combo)
            form.addRow(bi("跳帧间隔", "Frame Skip"), self.frame_skip_spin)
            form.addRow(bi("分段长度（秒）", "Segment Length Sec"), self.segment_seconds_spin)
            form.addRow(bi("分段重叠（秒）", "Segment Overlap Sec"), self.segment_overlap_spin)
            form.addRow(bi("GPS 时间偏移（秒）", "GPS Time Offset Sec"), self.gps_offset_spin)
            form.addRow(bi("Web 端口", "Web Port"), self.web_port_spin)
            options.addLayout(form, 0, 0)

            checks = QGridLayout()
            checks.setHorizontalSpacing(18)
            checks.setVerticalSpacing(6)
            self.cb_segment = QCheckBox(bi("启用分段流水线", "Enable Segment Pipeline"))
            self.cb_visual = QCheckBox(bi("启用视觉分段摘要", "Enable Visual Segment Summary"))
            self.cb_soundscape = QCheckBox(bi("启用声景分析", "Enable Soundscape"))
            self.cb_fusion = QCheckBox(bi("启用多模态融合", "Enable Fusion"))
            self.cb_agents = QCheckBox(bi("启用智能体", "Enable Agents"))
            self.cb_design = QCheckBox(bi("启用设计映射", "Enable Design"))
            self.cb_deliverable = QCheckBox(bi("启用交付物阶段", "Enable Deliverable"))
            self.cb_geo = QCheckBox(bi("启用地理同步", "Enable Geo Sync"))
            self.cb_gis = QCheckBox(bi("启用 GIS 导出", "Enable GIS Export"))
            self.cb_web_sync = QCheckBox(bi("启用 Web 同步导出", "Enable Web Sync Export"))
            self.cb_align_frames = QCheckBox(bi("对齐分析帧", "Align to Analysis Frames"))
            self.cb_wgs84 = QCheckBox(bi("导出 WGS84 坐标", "Geo Export WGS84"))
            self.cb_web_wgs84 = QCheckBox(bi("Web 优先 WGS84", "Web Sync Prefer WGS84"))
            self.cb_gis_wgs84 = QCheckBox(bi("GIS 优先 WGS84", "GIS Prefer WGS84"))
            self.cb_post_only = QCheckBox(bi("仅后处理", "Post Only"))
            self.cb_resume_missing = QCheckBox(bi("仅补缺失结果", "Resume Missing Only"))
            self.cb_run_deliverable = QCheckBox(bi("运行交付物层", "Run Deliverable Layer"))
            self.cb_glm = QCheckBox(bi("使用 GLM", "Use GLM"))
            self.cb_cards = QCheckBox(bi("导出卡片", "Export Cards"))
            self.cb_html = QCheckBox(bi("渲染 HTML", "Render HTML"))
            self.cb_pdf = QCheckBox(bi("渲染 PDF", "Render PDF"))
            self.cb_no_web = QCheckBox(bi("不自动启动 Web", "No Web"))
            self.deliverable_top_k_spin = QSpinBox()
            self.deliverable_top_k_spin.setRange(1, 1000)
            self.deliverable_gap_spin = QDoubleSpinBox()
            self.deliverable_gap_spin.setRange(0.0, 3600.0)
            self.deliverable_gap_spin.setDecimals(2)

            self.checkboxes = [
                self.cb_segment,
                self.cb_visual,
                self.cb_soundscape,
                self.cb_fusion,
                self.cb_agents,
                self.cb_design,
                self.cb_deliverable,
                self.cb_geo,
                self.cb_gis,
                self.cb_web_sync,
                self.cb_align_frames,
                self.cb_wgs84,
                self.cb_web_wgs84,
                self.cb_gis_wgs84,
                self.cb_post_only,
                self.cb_resume_missing,
                self.cb_run_deliverable,
                self.cb_glm,
                self.cb_cards,
                self.cb_html,
                self.cb_pdf,
                self.cb_no_web,
            ]
            for index, cb in enumerate(self.checkboxes):
                checks.addWidget(cb, index // 2, index % 2)
            checks.addWidget(QLabel(bi("交付物 Top K", "Deliverable Top K")), 11, 0)
            checks.addWidget(self.deliverable_top_k_spin, 11, 1)
            checks.addWidget(QLabel(bi("交付物最大间隔（秒）", "Deliverable Max Gap Sec")), 12, 0)
            checks.addWidget(self.deliverable_gap_spin, 12, 1)
            options.addLayout(checks, 0, 1)
            layout.addWidget(options_group)

            resource_group = QGroupBox(bi("资源设置", "Resource Settings"))
            resource_form = QGridLayout(resource_group)
            self.ffmpeg_edit = QLineEdit()
            self.models_edit = QLineEdit()
            self.api_env_edit = QLineEdit()
            self._add_path_row(resource_form, 0, bi("FFmpeg 路径", "FFmpeg Path"), self.ffmpeg_edit, self.browse_ffmpeg)
            self._add_path_row(resource_form, 1, bi("模型文件夹", "Models Folder"), self.models_edit, self.browse_models)
            self._add_path_row(resource_form, 2, bi("API Key 环境文件", "API Key Env File"), self.api_env_edit, self.browse_env)
            layout.addWidget(resource_group)

            command_group = QGroupBox(bi("命令预览", "Command Preview"))
            command_layout = QVBoxLayout(command_group)
            self.command_preview = QPlainTextEdit()
            self.command_preview.setReadOnly(True)
            self.command_preview.setMaximumHeight(90)
            command_buttons = QHBoxLayout()
            copy_btn = QPushButton(bi("复制命令", "Copy Command"))
            copy_btn.clicked.connect(self.copy_command)
            command_buttons.addStretch(1)
            command_buttons.addWidget(copy_btn)
            command_layout.addWidget(self.command_preview)
            command_layout.addLayout(command_buttons)
            layout.addWidget(command_group)

            run_group = QGroupBox(bi("运行面板", "Run Panel"))
            run_layout = QHBoxLayout(run_group)
            self.status_label = QLabel(bi("状态", "Status") + ": Idle")
            self.status_label.setObjectName("StatusBadge")
            self.run_btn = QPushButton(bi("运行分析", "Run Analysis"))
            self.run_btn.setObjectName("PrimaryButton")
            self.stop_btn = QPushButton(bi("停止", "Stop"))
            self.stop_btn.setObjectName("DangerButton")
            self.open_output_btn = QPushButton(bi("打开输出目录", "Open Output Folder"))
            self.open_web_btn = QPushButton(bi("打开网页", "Open Web Page"))
            self.save_cfg_btn = QPushButton(bi("保存配置", "Save Config"))
            self.load_cfg_btn = QPushButton(bi("加载配置", "Load Config"))
            self.reset_btn = QPushButton(bi("恢复默认", "Reset Defaults"))
            self.save_log_btn = QPushButton(bi("保存日志", "Save Log"))
            for btn in [
                self.run_btn,
                self.stop_btn,
                self.open_output_btn,
                self.open_web_btn,
                self.save_cfg_btn,
                self.load_cfg_btn,
                self.reset_btn,
                self.save_log_btn,
            ]:
                run_layout.addWidget(btn)
            run_layout.addStretch(1)
            run_layout.addWidget(self.status_label)
            layout.addWidget(run_group)

            log_group = QGroupBox(bi("日志面板", "Log Panel"))
            log_layout = QVBoxLayout(log_group)
            self.log_view = QPlainTextEdit()
            self.log_view.setReadOnly(True)
            log_layout.addWidget(self.log_view)
            layout.addWidget(log_group)

            self.run_btn.clicked.connect(self.run_analysis)
            self.stop_btn.clicked.connect(self.stop_analysis)
            self.open_output_btn.clicked.connect(self.open_output_folder)
            self.open_web_btn.clicked.connect(self.open_web_page)
            self.save_cfg_btn.clicked.connect(self.save_config_dialog)
            self.load_cfg_btn.clicked.connect(self.load_config_dialog)
            self.reset_btn.clicked.connect(self.reset_defaults)
            self.save_log_btn.clicked.connect(self.save_log)
            self.stop_btn.setEnabled(False)

            self.preset_combo.currentIndexChanged.connect(lambda: self.apply_preset(self.preset_combo.currentData()))
            for widget in [
                self.run_mode_combo,
                self.frame_skip_spin,
                self.segment_seconds_spin,
                self.segment_overlap_spin,
                self.gps_offset_spin,
                self.web_port_spin,
                self.deliverable_top_k_spin,
                self.deliverable_gap_spin,
            ]:
                if hasattr(widget, "currentIndexChanged"):
                    widget.currentIndexChanged.connect(self.update_preview)
                else:
                    widget.valueChanged.connect(self.update_preview)
            for edit in [
                self.input_edit,
                self.existing_edit,
                self.gps_edit,
                self.output_edit,
                self.ffmpeg_edit,
                self.models_edit,
                self.api_env_edit,
            ]:
                edit.textChanged.connect(self.update_preview)
            for cb in self.checkboxes:
                cb.stateChanged.connect(self.update_preview)
            self.gps_edit.textChanged.connect(self._auto_enable_gps)
            self.run_mode_combo.currentIndexChanged.connect(self._update_mode_enabled)

        def _add_path_row(self, layout: QGridLayout, row: int, label: str, widget: QWidget, browse=None) -> None:
            layout.addWidget(QLabel(label), row, 0)
            layout.addWidget(widget, row, 1)
            if browse:
                btn = QPushButton(bi("浏览", "Browse"))
                btn.clicked.connect(browse)
                layout.addWidget(btn, row, 2)

        def _update_mode_enabled(self) -> None:
            existing = self.run_mode_combo.currentData() == "existing"
            self.input_edit.setEnabled(not existing)
            self.existing_edit.setEnabled(existing)
            self.cb_post_only.setEnabled(existing)
            self.cb_resume_missing.setEnabled(existing)
            self.update_preview()

        def browse_input(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, bi("选择输入视频", "Select Input Video"), str(get_app_dir()), "Videos (*.mp4 *.mov *.avi *.insv *.mkv);;All files (*)")
            if not path:
                folder = QFileDialog.getExistingDirectory(self, bi("选择输入文件夹", "Select Input Folder"), str(get_app_dir()))
                path = folder
            if path:
                self.input_edit.setText(path)

        def browse_existing(self) -> None:
            path = QFileDialog.getExistingDirectory(self, bi("选择已有输出目录", "Select Existing Output Folder"), self.output_edit.text() or str(get_output_dir()))
            if path:
                self.existing_edit.setText(path)

        def browse_gps(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, bi("选择 GPS 文件", "Select GPS File"), str(get_app_dir()), "GPS files (*.csv *.xlsx *.xls);;All files (*)")
            if path:
                self.gps_edit.setText(path)

        def browse_output(self) -> None:
            path = QFileDialog.getExistingDirectory(self, bi("选择输出根目录", "Select Output Root Folder"), self.output_edit.text() or str(get_output_dir()))
            if path:
                self.output_edit.setText(path)

        def browse_ffmpeg(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, bi("选择 ffmpeg", "Select FFmpeg"), str(get_app_dir()), "ffmpeg (ffmpeg.exe ffmpeg);;All files (*)")
            if path:
                self.ffmpeg_edit.setText(path)

        def browse_models(self) -> None:
            path = QFileDialog.getExistingDirectory(self, bi("选择模型文件夹", "Select Models Folder"), self.models_edit.text() or str(get_app_dir()))
            if path:
                self.models_edit.setText(path)

        def browse_env(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, bi("选择 apikey.env", "Select apikey.env"), str(get_app_dir()), "Env files (*.env);;All files (*)")
            if path:
                self.api_env_edit.setText(path)

        def _auto_enable_gps(self) -> None:
            if self.gps_edit.text().strip():
                for cb in [self.cb_geo, self.cb_gis, self.cb_web_sync, self.cb_segment, self.cb_visual]:
                    cb.setChecked(True)
            self.update_preview()

        def apply_preset(self, preset: str, update_preview: bool = True) -> None:
            if not preset:
                return
            if preset == "quick":
                self.frame_skip_spin.setValue(200)
                self.cb_no_web.setChecked(True)
            elif preset == "standard":
                self.frame_skip_spin.setValue(20)
                self.cb_no_web.setChecked(True)
            elif preset == "deliverable_existing":
                self.run_mode_combo.setCurrentIndex(1)
                self.cb_post_only.setChecked(True)
                self.cb_run_deliverable.setChecked(True)
                self.cb_glm.setChecked(True)
                self.cb_cards.setChecked(True)
                self.cb_html.setChecked(True)
                self.cb_pdf.setChecked(True)
                self.cb_no_web.setChecked(True)
            elif preset == "gps_web":
                for cb in [self.cb_segment, self.cb_visual, self.cb_geo, self.cb_gis, self.cb_web_sync, self.cb_align_frames, self.cb_wgs84]:
                    cb.setChecked(True)
                self.cb_no_web.setChecked(True)
            elif preset == "full":
                for cb in [
                    self.cb_segment,
                    self.cb_visual,
                    self.cb_soundscape,
                    self.cb_fusion,
                    self.cb_agents,
                    self.cb_design,
                    self.cb_deliverable,
                    self.cb_run_deliverable,
                    self.cb_cards,
                    self.cb_html,
                    self.cb_pdf,
                ]:
                    cb.setChecked(True)
                if self.gps_edit.text().strip():
                    for cb in [self.cb_geo, self.cb_gis, self.cb_web_sync]:
                        cb.setChecked(True)
                self.frame_skip_spin.setValue(10)
                self.cb_no_web.setChecked(True)
            self._update_mode_enabled()
            if update_preview:
                self.update_preview()

        def collect_config(self) -> AnalysisConfig:
            return AnalysisConfig(
                run_mode=str(self.run_mode_combo.currentData()),
                preset=str(self.preset_combo.currentData()),
                input_path=self.input_edit.text().strip(),
                existing_output=self.existing_edit.text().strip(),
                gps_file=self.gps_edit.text().strip(),
                output_root=self.output_edit.text().strip(),
                frame_skip=int(self.frame_skip_spin.value()),
                segment_seconds=float(self.segment_seconds_spin.value()),
                segment_overlap=float(self.segment_overlap_spin.value()),
                gps_time_offset=float(self.gps_offset_spin.value()),
                enable_segment_pipeline=self.cb_segment.isChecked(),
                enable_visual_segment_summary=self.cb_visual.isChecked(),
                enable_soundscape=self.cb_soundscape.isChecked(),
                enable_fusion=self.cb_fusion.isChecked(),
                enable_agents=self.cb_agents.isChecked(),
                enable_design=self.cb_design.isChecked(),
                enable_deliverable=self.cb_deliverable.isChecked(),
                enable_gis_export=self.cb_gis.isChecked(),
                enable_web_sync_export=self.cb_web_sync.isChecked(),
                enable_geo_sync=self.cb_geo.isChecked(),
                geo_sync_export_wgs84=self.cb_wgs84.isChecked(),
                geo_sync_align_to_analysis_frames=self.cb_align_frames.isChecked(),
                web_sync_prefer_wgs84=self.cb_web_wgs84.isChecked(),
                gis_export_prefer_wgs84=self.cb_gis_wgs84.isChecked(),
                post_only=self.cb_post_only.isChecked(),
                resume_missing_only=self.cb_resume_missing.isChecked(),
                run_deliverable_layer=self.cb_run_deliverable.isChecked(),
                deliverable_use_glm=self.cb_glm.isChecked(),
                deliverable_export_cards=self.cb_cards.isChecked(),
                deliverable_render_html=self.cb_html.isChecked(),
                deliverable_render_pdf=self.cb_pdf.isChecked(),
                deliverable_top_k=int(self.deliverable_top_k_spin.value()),
                deliverable_max_gap_seconds=float(self.deliverable_gap_spin.value()),
                no_web=self.cb_no_web.isChecked(),
                web_port=int(self.web_port_spin.value()),
                ffmpeg_path=self.ffmpeg_edit.text().strip(),
                models_folder=self.models_edit.text().strip(),
                api_env_file=self.api_env_edit.text().strip(),
            )

        def set_config(self, cfg: AnalysisConfig) -> None:
            self.run_mode_combo.setCurrentIndex(1 if cfg.run_mode == "existing" else 0)
            index = self.preset_combo.findData(cfg.preset)
            self.preset_combo.setCurrentIndex(max(index, 0))
            self.input_edit.setText(cfg.input_path)
            self.existing_edit.setText(cfg.existing_output)
            self.gps_edit.setText(cfg.gps_file)
            self.output_edit.setText(cfg.output_root)
            self.frame_skip_spin.setValue(int(cfg.frame_skip))
            self.segment_seconds_spin.setValue(float(cfg.segment_seconds))
            self.segment_overlap_spin.setValue(float(cfg.segment_overlap))
            self.gps_offset_spin.setValue(float(cfg.gps_time_offset))
            self.cb_segment.setChecked(cfg.enable_segment_pipeline)
            self.cb_visual.setChecked(cfg.enable_visual_segment_summary)
            self.cb_soundscape.setChecked(cfg.enable_soundscape)
            self.cb_fusion.setChecked(cfg.enable_fusion)
            self.cb_agents.setChecked(cfg.enable_agents)
            self.cb_design.setChecked(cfg.enable_design)
            self.cb_deliverable.setChecked(cfg.enable_deliverable)
            self.cb_gis.setChecked(cfg.enable_gis_export)
            self.cb_web_sync.setChecked(cfg.enable_web_sync_export)
            self.cb_geo.setChecked(cfg.enable_geo_sync)
            self.cb_wgs84.setChecked(cfg.geo_sync_export_wgs84)
            self.cb_align_frames.setChecked(cfg.geo_sync_align_to_analysis_frames)
            self.cb_web_wgs84.setChecked(cfg.web_sync_prefer_wgs84)
            self.cb_gis_wgs84.setChecked(cfg.gis_export_prefer_wgs84)
            self.cb_post_only.setChecked(cfg.post_only)
            self.cb_resume_missing.setChecked(cfg.resume_missing_only)
            self.cb_run_deliverable.setChecked(cfg.run_deliverable_layer)
            self.cb_glm.setChecked(cfg.deliverable_use_glm)
            self.cb_cards.setChecked(cfg.deliverable_export_cards)
            self.cb_html.setChecked(cfg.deliverable_render_html)
            self.cb_pdf.setChecked(cfg.deliverable_render_pdf)
            self.deliverable_top_k_spin.setValue(int(cfg.deliverable_top_k))
            self.deliverable_gap_spin.setValue(float(cfg.deliverable_max_gap_seconds))
            self.cb_no_web.setChecked(cfg.no_web)
            self.web_port_spin.setValue(int(cfg.web_port))
            self.ffmpeg_edit.setText(cfg.ffmpeg_path)
            self.models_edit.setText(cfg.models_folder)
            self.api_env_edit.setText(cfg.api_env_file)
            self._update_mode_enabled()
            self.update_preview()

        def update_preview(self) -> None:
            cfg = self.collect_config()
            try:
                args = build_args(cfg)
                program, prefix = self._process_program_and_prefix()
                self.command_preview.setPlainText(quote_for_preview([program, *prefix, *args]))
            except Exception as exc:
                self.command_preview.setPlainText(f"Command build error: {exc}")

        def _process_program_and_prefix(self) -> tuple[str, list[str]]:
            if is_frozen():
                return sys.executable, [INTERNAL_FLAG]
            return sys.executable, [str(Path(__file__).resolve()), INTERNAL_FLAG]

        def copy_command(self) -> None:
            QApplication.clipboard().setText(self.command_preview.toPlainText())

        def set_status(self, value: str) -> None:
            self.status = value
            status_cn = {
                "Idle": "空闲",
                "Running": "运行中",
                "Finished": "已完成",
                "Failed": "失败",
                "Cancelled": "已取消",
            }.get(value, value)
            self.status_label.setText(f"{bi('状态', 'Status')}: {status_cn} / {value}")

        def append_log(self, text: str) -> None:
            if not text:
                return
            self.log_lines.extend(text.splitlines())
            self.log_view.moveCursor(QTextCursor.MoveOperation.End)
            self.log_view.insertPlainText(text)
            self.log_view.moveCursor(QTextCursor.MoveOperation.End)

        def run_analysis(self) -> None:
            cfg = self.collect_config()
            errors = validate_config(cfg)
            if errors:
                QMessageBox.warning(self, bi("无法运行", "Cannot Run"), "\n".join(errors))
                return
            try:
                runtime_cfg = prepare_runtime_config(cfg)
            except Exception as exc:
                QMessageBox.critical(self, bi("GPS 错误", "GPS Error"), str(exc))
                return

            args = build_args(runtime_cfg)
            program, prefix = self._process_program_and_prefix()
            self.log_lines.clear()
            self.log_view.clear()
            self.append_log(f"Command: {quote_for_preview([program, *prefix, *args])}\n\n")
            self.run_log_path = self._new_run_log_path(runtime_cfg)
            self.run_log_pos = 0
            self.run_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.run_log_path.write_text("", encoding="utf-8")
            self.append_log(f"Run log: {self.run_log_path}\n\n")

            self.process = QProcess(self)
            self.process.setProgram(program)
            self.process.setArguments([*prefix, *args])
            self.process.setWorkingDirectory(str(get_app_dir()))
            env = QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONIOENCODING", "utf-8")
            env.insert(RUN_LOG_ENV, str(self.run_log_path))
            ffmpeg = find_ffmpeg(runtime_cfg.ffmpeg_path)
            if ffmpeg:
                env.insert("PATH", str(ffmpeg.parent) + os.pathsep + env.value("PATH"))
            api_env = Path(runtime_cfg.api_env_file) if runtime_cfg.api_env_file else get_app_dir() / "apikey.env"
            for key, value in load_env_file(api_env).items():
                env.insert(key, value)
            self.process.setProcessEnvironment(env)
            self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
            self.process.readyReadStandardOutput.connect(lambda: self._read_process(False))
            self.process.readyReadStandardError.connect(lambda: self._read_process(True))
            self.process.finished.connect(self._process_finished)
            self.process.errorOccurred.connect(lambda error: self.append_log(f"\n[QProcess error] {error}\n"))
            self.set_status("Running")
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.log_tail_timer.start()
            self.process.start()
            if not self.process.waitForStarted(3000):
                self.set_status("Failed")
                self.run_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.log_tail_timer.stop()
                QMessageBox.critical(self, bi("启动失败", "Start Failed"), self.process.errorString())

        def _new_run_log_path(self, cfg: AnalysisConfig) -> Path:
            output_root = Path(cfg.output_root or get_output_dir())
            stamp = time.strftime("%Y%m%d_%H%M%S")
            return output_root / "_launcher_logs" / f"run_{stamp}.txt"

        def tail_run_log(self) -> None:
            if not self.run_log_path or not self.run_log_path.is_file():
                return
            try:
                with self.run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(self.run_log_pos)
                    text = handle.read()
                    self.run_log_pos = handle.tell()
                if text:
                    self.append_log(text)
            except Exception:
                return

        def _read_process(self, stderr: bool) -> None:
            if not self.process:
                return
            data = self.process.readAllStandardError() if stderr else self.process.readAllStandardOutput()
            prefix = "[stderr] " if stderr else ""
            text = bytes(data).decode("utf-8", errors="replace")
            self.append_log(prefix + text if prefix and text.strip() else text)

        def _process_finished(self, exit_code: int, exit_status) -> None:
            self.tail_run_log()
            self.log_tail_timer.stop()
            cancelled = self.status == "Cancelled"
            if cancelled:
                status = "Cancelled"
            elif exit_code == 0:
                status = "Finished"
            else:
                status = "Failed"
            self.set_status(status)
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            cfg = self.collect_config()
            self.last_run_dir = self._infer_last_run_dir(cfg)
            self.save_config(default_config_path())
            if exit_code != 0 and not cancelled:
                tail = "\n".join(self.log_lines[-50:])
                QMessageBox.critical(self, bi("运行失败", "Run Failed"), f"Return code: {exit_code}\n\nLast 50 log lines:\n{tail}")
            else:
                self.append_log(f"\n[{status}] return code: {exit_code}\n")

        def stop_analysis(self) -> None:
            if not self.process:
                return
            self.set_status("Cancelled")
            self.process.terminate()
            QTimer.singleShot(5000, lambda: self.process and self.process.kill())

        def _infer_last_run_dir(self, cfg: AnalysisConfig) -> Path:
            if cfg.run_mode == "existing" and cfg.existing_output:
                return Path(cfg.existing_output)
            output_root = Path(cfg.output_root or get_output_dir())
            if cfg.input_path and Path(cfg.input_path).is_file():
                return output_root / Path(cfg.input_path).stem
            subdirs = [p for p in output_root.iterdir() if p.is_dir()] if output_root.is_dir() else []
            return max(subdirs, key=lambda p: p.stat().st_mtime) if subdirs else output_root

        def open_output_folder(self) -> None:
            target = self.last_run_dir or Path(self.collect_config().output_root or get_output_dir())
            self._open_path(target)

        def open_web_page(self) -> None:
            root = self.last_run_dir or Path(self.collect_config().output_root or get_output_dir())
            html_files = find_html_outputs(root)
            if not html_files:
                QMessageBox.information(self, bi("没有 HTML", "No HTML"), f"未在目录中找到 HTML / No HTML found:\n{root}")
                return
            selected = html_files[0]
            if len(html_files) > 1:
                labels = [str(p.relative_to(root)) if _is_relative_to(p, root) else str(p) for p in html_files[:100]]
                choice, ok = QInputDialog.getItem(self, bi("打开网页", "Open Web Page"), bi("选择 HTML", "Select HTML") + ":", labels, 0, False)
                if not ok:
                    return
                selected = root / choice
            try:
                port = self.http_server.start(root, preferred_port=8765)
                rel = selected.resolve().relative_to(root.resolve()).as_posix()
                webbrowser.open(f"http://127.0.0.1:{port}/{rel}")
            except Exception as exc:
                QMessageBox.critical(self, bi("打开网页失败", "Open Web Failed"), str(exc))

        def _open_path(self, path: Path) -> None:
            path = path.resolve()
            if not path.exists():
                QMessageBox.warning(self, bi("未找到", "Not Found"), f"路径不存在 / Path not found:\n{path}")
                return
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])

        def save_config(self, path: Path) -> None:
            path.write_text(json.dumps(asdict(self.collect_config()), ensure_ascii=False, indent=2), encoding="utf-8")

        def save_config_dialog(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, bi("保存配置", "Save Config"), str(default_config_path()), "JSON (*.json)")
            if path:
                self.save_config(Path(path))

        def load_config_dialog(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, bi("加载配置", "Load Config"), str(get_app_dir()), "JSON (*.json)")
            if path:
                self.set_config(config_from_dict(json.loads(Path(path).read_text(encoding="utf-8"))))

        def load_last_config(self) -> None:
            path = default_config_path()
            if path.is_file():
                try:
                    self.set_config(config_from_dict(json.loads(path.read_text(encoding="utf-8"))))
                    return
                except Exception:
                    pass
            self.set_config(AnalysisConfig.defaults())

        def reset_defaults(self) -> None:
            self.set_config(AnalysisConfig.defaults())

        def save_log(self) -> None:
            default = get_app_dir() / f"launcher_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            path, _ = QFileDialog.getSaveFileName(self, bi("保存日志", "Save Log"), str(default), "Text (*.txt)")
            if path:
                Path(path).write_text("\n".join(self.log_lines), encoding="utf-8")

        def closeEvent(self, event) -> None:
            self.http_server.close()
            if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
                self.process.terminate()
            event.accept()

    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except Exception:
            return False

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    win = MainWindow()
    win.show()
    if smoke_test:
        QTimer.singleShot(600, app.quit)
    return app.exec()


def parse_launcher_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(INTERNAL_FLAG, action="store_true", help="Run wrapped main.py pipeline instead of GUI")
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == INTERNAL_FLAG:
        return run_pipeline_internal(argv[1:])
    if argv and argv[0] == "--launcher-smoke-test":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        return launch_gui(smoke_test=True)
    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
