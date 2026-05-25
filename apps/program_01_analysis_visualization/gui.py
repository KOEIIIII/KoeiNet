"""Program 01 desktop interface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
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
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


INTERNAL_FLAG = "--run-program01-pipeline"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    return Path(sys.executable).resolve().parent if is_frozen() else Path(__file__).resolve().parents[2]


APP_DIR = app_root()
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", APP_DIR))
ROOT = RESOURCE_ROOT

TEXT = {
    "en": {
        "title": "Program 01 - Base Analysis and Spatial Visualization",
        "subtitle": "Stage 1: panoramic video, GPS alignment, GIS export, and local web visualization.",
        "language": "Language",
        "video": "Video File",
        "gps": "GPS File",
        "output": "Output Folder",
        "existing": "Existing Output",
        "select_video": "Select Video",
        "select_gps": "Select GPS File",
        "select_output": "Select Output Folder",
        "select_existing": "Select Existing Output",
        "basic": "Basic Settings",
        "advanced": "Advanced Settings",
        "frame_skip": "Frame Skip",
        "segment_seconds": "Segment Length Sec",
        "segment_overlap": "Segment Overlap Sec",
        "gps_offset": "GPS Time Offset Sec",
        "use_existing": "Use existing output",
        "post_only": "Post only",
        "resume": "Resume missing only",
        "segment": "Enable segment pipeline",
        "visual": "Enable visual segment summary",
        "soundscape": "Enable soundscape",
        "geo": "Enable trajectory alignment",
        "gis": "Enable GIS export",
        "web_sync": "Enable web sync export",
        "start": "Start Analysis",
        "open_output": "Open Output Folder",
        "open_web": "Open Visualization",
        "command": "Command Preview",
        "log": "Run Log",
        "idle": "Idle",
    },
    "zh": {
        "title": "Program 01 - 基础数据分析与空间可视化",
        "subtitle": "第一阶段：全景视频、GPS 对齐、GIS 导出和本地网页可视化。",
        "language": "语言",
        "video": "视频文件",
        "gps": "GPS 文件",
        "output": "输出文件夹",
        "existing": "已有输出",
        "select_video": "选择视频",
        "select_gps": "选择 GPS 文件",
        "select_output": "选择输出文件夹",
        "select_existing": "选择已有输出",
        "basic": "基础设置",
        "advanced": "高级设置",
        "frame_skip": "视频帧抽样间隔",
        "segment_seconds": "时间片段长度（秒）",
        "segment_overlap": "时间片段重叠（秒）",
        "gps_offset": "GPS 时间偏移（秒）",
        "use_existing": "使用已有输出",
        "post_only": "仅运行后处理",
        "resume": "仅补缺失结果",
        "segment": "启用时间片段流程",
        "visual": "启用视觉片段摘要",
        "soundscape": "启用声景分析",
        "geo": "启用轨迹对齐",
        "gis": "启用 GIS 导出",
        "web_sync": "启用网页同步导出",
        "start": "开始分析",
        "open_output": "打开输出文件夹",
        "open_web": "打开可视化网页",
        "command": "命令预览",
        "log": "运行日志",
        "idle": "空闲",
    },
}


class Program01Window(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.lang = "en"
        self.process: QProcess | None = None
        self.setWindowTitle(TEXT[self.lang]["title"])
        self.resize(1080, 760)
        self._build_ui()
        self._apply_style()
        self._load_defaults()
        self._translate()
        self._update_command()

    def tr(self, key: str) -> str:
        return TEXT[self.lang][key]

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { font-family: Segoe UI, Microsoft YaHei, sans-serif; font-size: 13px; color: #18202a; }
            QGroupBox { border: 1px solid #d6dee8; border-radius: 8px; margin-top: 12px; padding: 14px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
            QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { border: 1px solid #c8d1dc; border-radius: 6px; padding: 7px; background: #ffffff; }
            QPushButton { border: 1px solid #b8c4d2; border-radius: 6px; padding: 8px 12px; background: #f7f9fc; }
            QPushButton:hover { background: #edf4ff; }
            QPushButton#primary { background: #1f6feb; color: white; border-color: #1f6feb; font-weight: 700; }
            QLabel#title { font-size: 24px; font-weight: 800; }
            QLabel#subtitle { color: #5d6b7a; }
            """
        )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        title_block = QVBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("title")
        self.subtitle = QLabel()
        self.subtitle.setObjectName("subtitle")
        title_block.addWidget(self.title)
        title_block.addWidget(self.subtitle)
        self.language = QComboBox()
        self.language.addItem("English", "en")
        self.language.addItem("中文", "zh")
        self.language.currentIndexChanged.connect(self._change_language)
        header.addLayout(title_block, 1)
        header.addWidget(QLabel("Language"))
        header.addWidget(self.language)
        root.addLayout(header)

        io_group = QGroupBox()
        io = QGridLayout(io_group)
        self.video = QLineEdit()
        self.gps = QLineEdit()
        self.output = QLineEdit()
        self.existing = QLineEdit()
        self.use_existing = QCheckBox()
        self._add_path_row(io, 0, self.video, self._browse_video)
        self._add_path_row(io, 1, self.gps, self._browse_gps)
        self._add_path_row(io, 2, self.output, self._browse_output)
        self._add_path_row(io, 3, self.existing, self._browse_existing)
        io.addWidget(self.use_existing, 4, 1)
        root.addWidget(io_group)
        self.io_group = io_group

        basic_group = QGroupBox()
        basic = QFormLayout(basic_group)
        self.frame_skip = QSpinBox()
        self.frame_skip.setRange(1, 100000)
        self.segment_seconds = QDoubleSpinBox()
        self.segment_seconds.setRange(0.1, 3600)
        self.segment_seconds.setDecimals(2)
        self.segment_overlap = QDoubleSpinBox()
        self.segment_overlap.setRange(0.0, 3600)
        self.segment_overlap.setDecimals(2)
        self.gps_offset = QDoubleSpinBox()
        self.gps_offset.setRange(-86400, 86400)
        self.gps_offset.setDecimals(2)
        basic.addRow(QLabel(), self.frame_skip)
        basic.addRow(QLabel(), self.segment_seconds)
        basic.addRow(QLabel(), self.segment_overlap)
        basic.addRow(QLabel(), self.gps_offset)
        root.addWidget(basic_group)
        self.basic_group = basic_group
        self.basic_labels = [basic.labelForField(w) for w in [self.frame_skip, self.segment_seconds, self.segment_overlap, self.gps_offset]]

        adv_group = QGroupBox()
        adv = QGridLayout(adv_group)
        self.post_only = QCheckBox()
        self.resume = QCheckBox()
        self.enable_segment = QCheckBox()
        self.enable_visual = QCheckBox()
        self.enable_soundscape = QCheckBox()
        self.enable_geo = QCheckBox()
        self.enable_gis = QCheckBox()
        self.enable_web_sync = QCheckBox()
        self.checks = [
            self.post_only,
            self.resume,
            self.enable_segment,
            self.enable_visual,
            self.enable_soundscape,
            self.enable_geo,
            self.enable_gis,
            self.enable_web_sync,
        ]
        for i, cb in enumerate(self.checks):
            adv.addWidget(cb, i // 2, i % 2)
        root.addWidget(adv_group)
        self.adv_group = adv_group

        command_group = QGroupBox()
        cmd_layout = QVBoxLayout(command_group)
        self.command = QPlainTextEdit()
        self.command.setReadOnly(True)
        self.command.setMaximumHeight(72)
        cmd_layout.addWidget(self.command)
        root.addWidget(command_group)
        self.command_group = command_group

        actions = QHBoxLayout()
        self.start = QPushButton()
        self.start.setObjectName("primary")
        self.open_output = QPushButton()
        self.open_web = QPushButton()
        actions.addWidget(self.start)
        actions.addWidget(self.open_output)
        actions.addWidget(self.open_web)
        actions.addStretch(1)
        self.status = QLabel()
        actions.addWidget(self.status)
        root.addLayout(actions)

        log_group = QGroupBox()
        log_layout = QVBoxLayout(log_group)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        log_layout.addWidget(self.log)
        root.addWidget(log_group, 1)
        self.log_group = log_group

        for widget in [self.video, self.gps, self.output, self.existing]:
            widget.textChanged.connect(self._update_command)
        for widget in [self.frame_skip, self.segment_seconds, self.segment_overlap, self.gps_offset]:
            widget.valueChanged.connect(self._update_command)
        for cb in [self.use_existing, *self.checks]:
            cb.stateChanged.connect(self._update_command)
        self.start.clicked.connect(self._start)
        self.open_output.clicked.connect(lambda: self._open_path(Path(self.output.text() or "output")))
        self.open_web.clicked.connect(self._open_visualization)

    def _add_path_row(self, layout: QGridLayout, row: int, edit: QLineEdit, slot) -> None:
        label = QLabel()
        button = QPushButton()
        layout.addWidget(label, row, 0)
        layout.addWidget(edit, row, 1)
        layout.addWidget(button, row, 2)
        button.clicked.connect(slot)
        setattr(self, f"path_label_{row}", label)
        setattr(self, f"path_button_{row}", button)

    def _load_defaults(self) -> None:
        self.output.setText(str(APP_DIR / "output"))
        self.frame_skip.setValue(20)
        self.segment_seconds.setValue(5.0)
        self.segment_overlap.setValue(2.5)
        self.gps_offset.setValue(25.0)
        for cb in [self.enable_segment, self.enable_visual, self.enable_geo, self.enable_gis, self.enable_web_sync]:
            cb.setChecked(True)

    def _translate(self) -> None:
        self.setWindowTitle(self.tr("title"))
        self.title.setText(self.tr("title"))
        self.subtitle.setText(self.tr("subtitle"))
        self.io_group.setTitle("Input & Output" if self.lang == "en" else "输入与输出")
        for row, key in enumerate(["video", "gps", "output", "existing"]):
            getattr(self, f"path_label_{row}").setText(self.tr(key))
        for row, key in enumerate(["select_video", "select_gps", "select_output", "select_existing"]):
            getattr(self, f"path_button_{row}").setText(self.tr(key))
        self.use_existing.setText(self.tr("use_existing"))
        self.basic_group.setTitle(self.tr("basic"))
        for label, key in zip(self.basic_labels, ["frame_skip", "segment_seconds", "segment_overlap", "gps_offset"]):
            label.setText(self.tr(key))
        self.adv_group.setTitle(self.tr("advanced"))
        for cb, key in zip(self.checks, ["post_only", "resume", "segment", "visual", "soundscape", "geo", "gis", "web_sync"]):
            cb.setText(self.tr(key))
        self.command_group.setTitle(self.tr("command"))
        self.log_group.setTitle(self.tr("log"))
        self.start.setText(self.tr("start"))
        self.open_output.setText(self.tr("open_output"))
        self.open_web.setText(self.tr("open_web"))
        self.status.setText(self.tr("idle"))

    def _change_language(self) -> None:
        self.lang = str(self.language.currentData())
        self._translate()

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.tr("select_video"), str(APP_DIR), "Videos (*.mp4 *.mov *.avi *.insv *.mkv);;All files (*)")
        if path:
            self.video.setText(path)

    def _browse_gps(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.tr("select_gps"), str(APP_DIR), "GPS (*.csv *.xlsx *.xls);;All files (*)")
        if path:
            self.gps.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, self.tr("select_output"), self.output.text() or str(APP_DIR))
        if path:
            self.output.setText(path)

    def _browse_existing(self) -> None:
        path = QFileDialog.getExistingDirectory(self, self.tr("select_existing"), self.output.text() or str(APP_DIR / "output"))
        if path:
            self.existing.setText(path)
            self.use_existing.setChecked(True)

    def _pipeline_args(self) -> list[str]:
        args = ["--output_dir", self.output.text() or "output"]
        args += ["--frame_skip", str(self.frame_skip.value())]
        args += ["--segment_seconds", str(self.segment_seconds.value())]
        args += ["--segment_overlap", str(self.segment_overlap.value())]
        args += ["--gps_time_offset_seconds", str(self.gps_offset.value())]
        args.append("--enable_segment_pipeline" if self.enable_segment.isChecked() else "--no-enable_segment_pipeline")
        args.append(
            "--enable_visual_segment_summary"
            if self.enable_visual.isChecked()
            else "--no-enable_visual_segment_summary"
        )
        args.append("--enable_soundscape" if self.enable_soundscape.isChecked() else "--no-enable_soundscape")
        args.append("--enable_geo_sync" if self.enable_geo.isChecked() else "--no-enable_geo_sync")
        args.append("--enable_gis_export" if self.enable_gis.isChecked() else "--no-enable_gis_export")
        args.append("--enable_web_sync_export" if self.enable_web_sync.isChecked() else "--no-enable_web_sync_export")
        if self.use_existing.isChecked():
            args += ["--from_existing_output", self.existing.text()]
            if self.post_only.isChecked():
                args.append("--post_only")
            if self.resume.isChecked():
                args.append("--resume_missing_only")
        else:
            args += ["--input_video", self.video.text()]
        if self.gps.text():
            args += ["--gps_file", self.gps.text()]
        return args

    def _args(self) -> list[str]:
        pipeline_args = self._pipeline_args()
        if is_frozen():
            return [sys.executable, INTERNAL_FLAG, *pipeline_args]
        return [sys.executable, str(ROOT / "scripts" / "run_program_01.py"), *pipeline_args]

    def _update_command(self) -> None:
        self.command.setPlainText(subprocess.list2cmdline(self._args()))

    def _start(self) -> None:
        if self.use_existing.isChecked() and not self.existing.text():
            QMessageBox.warning(self, "Program 01", "Existing output folder is required.")
            return
        if not self.use_existing.isChecked() and not self.video.text():
            QMessageBox.warning(self, "Program 01", "Video file is required.")
            return
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(lambda code, _status: self.status.setText(f"Finished ({code})"))
        args = self._args()
        self.log.appendPlainText(subprocess.list2cmdline(args))
        self.status.setText("Running")
        self.process.start(args[0], args[1:])

    def _read_output(self) -> None:
        if not self.process:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _open_path(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path.resolve()))

    def _open_visualization(self) -> None:
        root = Path(self.output.text() or APP_DIR / "output")
        html = sorted(root.rglob("*.html"), key=lambda p: (0 if p.name == "index.html" else 1, len(p.parts)))
        if html:
            os.startfile(str(html[0].resolve()))
        else:
            QMessageBox.information(self, "Program 01", "No visualization HTML was found yet.")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == INTERNAL_FLAG:
        from launcher_gui import run_pipeline_internal
        from scripts.run_program_01 import _read_config, build_main_args

        class _Cli:
            config = None
            dry_run = False
            launch_gui = False

        cli = _Cli()
        for name in [
            "input_video",
            "input_dir",
            "gps_file",
            "output_dir",
            "frame_skip",
            "segment_seconds",
            "segment_overlap",
            "gps_time_offset_seconds",
            "from_existing_output",
        ]:
            setattr(cli, name, None)
        cli.post_only = False
        cli.resume_missing_only = False
        import argparse

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--input_video")
        parser.add_argument("--input_dir")
        parser.add_argument("--gps_file")
        parser.add_argument("--output_dir")
        parser.add_argument("--frame_skip", type=int)
        parser.add_argument("--segment_seconds", type=float)
        parser.add_argument("--segment_overlap", type=float)
        parser.add_argument("--gps_time_offset_seconds", type=float)
        parser.add_argument("--from_existing_output")
        parser.add_argument("--post_only", action="store_true")
        parser.add_argument("--resume_missing_only", action="store_true")
        parser.add_argument("--enable_segment_pipeline", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--enable_visual_segment_summary", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--enable_soundscape", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--enable_geo_sync", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--enable_gis_export", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--enable_web_sync_export", action=argparse.BooleanOptionalAction, default=None)
        parsed, _ = parser.parse_known_args(argv[1:])
        cfg = _read_config(str(RESOURCE_ROOT / "configs" / "default_program_01_config.yaml"))
        cfg["no_web"] = True
        main_args = build_main_args(cfg, parsed)
        return run_pipeline_internal(main_args)
    if argv and argv[0] == "--smoke-test":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication([sys.argv[0]])
        win = Program01Window()
        win.language.setCurrentIndex(1)
        sample = RESOURCE_ROOT / "examples" / "sample_outputs" / "minimal_program01_output"
        win.existing.setText(str(sample))
        win.use_existing.setChecked(True)
        win.post_only.setChecked(True)
        smoke_output = APP_DIR / "release_smoke" / "program01_output"
        smoke_output.mkdir(parents=True, exist_ok=True)
        win.output.setText(str(smoke_output))
        win._update_command()
        html = sample / "web" / "index.html"
        ok = (
            sample.is_dir()
            and smoke_output.is_dir()
            and html.is_file()
            and "Program 01" in win.windowTitle()
            and win.command.toPlainText()
        )
        print(f"program01_smoke={'passed' if ok else 'failed'}")
        app.quit()
        return 0 if ok else 1
    app = QApplication(sys.argv[:1] + argv)
    win = Program01Window()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
